import math
import bisect
import copy
import os.path as osp
import json
from functools import partial

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import normalize as tf_norm, to_tensor

from .. import cfg as global_cfg
from .augmentations import get_composed_augmentations
from .samplers.grouped_batch_sampler import GroupedBatchSampler
from .gt_producer import (
    ignore_index, convert_pan_m_to_sem_m, GTGenerator
)
from panopticapi.utils import rgb2id

from fabric.utils.logging import setup_logging
logger = setup_logging(__file__)

'''
Expect a data root directory with the following layout (only coco sub-tree is
expanded for simplicity)
.
├── coco
│   ├── annotations
│   │   ├── train/
│   │   ├── train.json
│   │   ├── val/
│   │   └── val.json
│   └── images
│       ├── train/
│       └── val/
├── mapillary/
└── cityscapes/
'''


'''
A reminder of Panoptic meta json structure: (only the relevant fields are listed)

info/
licenses/
categories:
    - id: 1
      name: person
      supercategory: person
      isthing: 1
    - id: 191
      name: pavement-merged
      supercategory: ground
      isthing: 0
images:
    - id: 397133
      file_name: 000000397133.jpg
      height: 427
      width: 640
    - id: 397134
      file_name: 000000397134.jpg
      height: 422
      width: 650
annotations:
    - image_id: 139
      file_name: 000000000139.png
      segments_info:
        - id: 3226956
          category_id: 1
          iscrowd: 0,
          bbox: [413, 158, 53, 138] in xywh form,
          area: 2840
        - repeat omitted
'''

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]


def imagenet_normalize(im):
    """This only operates on single image"""
    if im.shape[0] == 1:
        return im  # deal with these gray channel images in coco later. Hell
    return tf_norm(im, imagenet_mean, imagenet_std)


def caffe_imagenet_normalize(im):
    im = im * 255.0
    im = tf_norm(im, (102.9801, 115.9465, 122.7717), (1.0, 1.0, 1.0))
    return im


def check_and_tuplize_splits(splits):
    if not isinstance(splits, (tuple, list)):
        splits = (splits, )
    for split in splits:
        assert split in ('train', 'val', 'test')
    return splits


def mapify_iterable(iter_of_dict, field_name):
    """Convert an iterable of dicts into a big dict indexed by chosen field
    I can't think of a better name. 'Tis catchy.
    """
    acc = dict()
    for item in iter_of_dict:
        acc[item[field_name]] = item
    return acc


def ttic_find_data_root(dset_name):
    '''Find the fastest data root on TTIC slurm cluster'''
    default = osp.join('/share/data/vision-greg/panoptic', dset_name)
    return default
    fast = osp.join('/vscratch/vision/panoptic', dset_name)
    return fast if osp.isdir(fast) else default


def test_meta_conform(rmeta):
    '''the metadata for test set does not conform to panoptic format;
    Test ann only has 'images' and 'categories; let's fill in annotations'
    '''
    images = rmeta['images']
    anns = []
    for img in images:
        _curr_ann = {
            'image_id': img['id'],
            'segments_info': []
            # do not fill in file_name
        }
        anns.append(_curr_ann)
    rmeta['annotations'] = anns
    return rmeta


class PanopticBase(Dataset):
    def __init__(self, name, split):
        available_dsets = ('coco', 'cityscapes', 'mapillary')
        assert name in available_dsets, '{} dset is not available'.format(name)
        root = ttic_find_data_root(name)
        logger.info('using data root {}'.format(root))
        self.root = root
        self.name = name
        self.split = split
        self.img_root = osp.join(root, 'images', split)
        self.ann_root = osp.join(root, 'annotations', split)
        meta_fname = osp.join(root, 'annotations', '{}.json'.format(split))
        with open(meta_fname) as f:
            rmeta = json.load(f)  # rmeta stands for raw metadata
            if self.split.startswith('test'):
                rmeta = test_meta_conform(rmeta)

        # store category metadata
        self.meta = dict()
        self.meta['cats'] = mapify_iterable(rmeta['categories'], 'id')
        self.meta['cat_IdToName'] = dict()
        self.meta['cat_NameToId'] = dict()
        for cat in rmeta['categories']:
            id, name = cat['id'], cat['name']
            self.meta['cat_IdToName'][id] = name
            self.meta['cat_NameToId'][name] = id

        # store image and annotations metadata
        self.imgs = mapify_iterable(rmeta['images'], 'id')
        self.imgToAnns = mapify_iterable(rmeta['annotations'], 'image_id')
        self.imgIds = list(sorted(self.imgs.keys()))

    def confine_to_subset(self, imgIds):
        '''confine data loading to a subset of images
        This is used for figure making.
        '''
        # confirm that the supplied imgIds are all valid
        for supplied_id in imgIds:
            assert supplied_id in self.imgIds
        self.imgIds = imgIds

    def test_seek_imgs(self, i, total_splits):
        assert isinstance(total_splits, int)
        length = len(self.imgIds)
        portion_size = int(math.ceil(length * 1.0 / total_splits))
        start = i * portion_size
        end = min(length, (i + 1) * portion_size)
        self.imgIds = self.imgIds[start:end]
        acc = { k: self.imgs[k] for k in self.imgIds }
        self.imgs = acc

    def __len__(self):
        return len(self.imgIds)

    def read_img(self, img_fname):
        # there are some gray scale images in coco; convert to RGB
        return Image.open(img_fname).convert('RGB')

    def get_meta(self, index):
        imgId = self.imgIds[index]
        imgMeta = self.imgs[imgId]
        anns = self.imgToAnns[imgId]
        return imgMeta, anns

    def __getitem__(self, index):
        imgMeta, anns = self.get_meta(index)
        img_fname = osp.join(self.img_root, imgMeta['file_name'])
        img = self.read_img(img_fname)
        if self.split.startswith('test'):
            mask = Image.fromarray(
                np.zeros(np.array(img).shape, dtype=np.uint8)
            )
        else:
            mask = Image.open(osp.join(self.ann_root, anns['file_name']))
        segments_info = mapify_iterable(anns['segments_info'], 'id')
        return imgMeta, segments_info, img, mask


class SemanticSeg(PanopticBase):
    def __init__(self, name, split, transforms):
        super().__init__(name=name, split=split)
        self.transforms = transforms
        # produce train cat index id starting at 0
        self.meta['catId_2_trainId'] = dict()
        self.meta['trainId_2_catId'] = dict()
        self.meta['trainId_2_catName'] = dict()  # all things grouped into "things"
        self.meta['trainId_2_catName'][ignore_index] = 'ignore'
        self.prep_trainId()
        self.meta['num_classes'] = len(self.meta['trainId_2_catName']) - 1

    def prep_trainId(self):
        curr_inx = 0
        for catId, cat in self.meta['cats'].items():
            self.meta['catId_2_trainId'][catId] = curr_inx
            self.meta['trainId_2_catId'][curr_inx] = catId
            self.meta['trainId_2_catName'][curr_inx] = cat['name']
            curr_inx += 1

    def __getitem__(self, index):
        raise ValueError('diabling data loading through this class for now')
        _, segments_info, im, mask = super().__getitem__(index)
        mask = np.array(mask)
        mask = rgb2id(np.array(mask))
        mask = convert_pan_m_to_sem_m(
            mask, segments_info, self.meta['catId_2_trainId'])
        mask = Image.fromarray(mask, mode='I')
        if self.transforms is not None:
            im, mask = self.transforms(im, mask)
            im, mask = to_tensor(im), to_tensor(mask).squeeze(dim=0).long()
            im = imagenet_normalize(im)
        return im, mask


class PanopticSeg(SemanticSeg):
    def __init__(
        self, name, split, transforms, pcv, gt_producers,
        caffe_mode=False, tensorize=True
    ):
        super().__init__(name=name, split=split, transforms=transforms)
        self.pcv = pcv
        self.meta['stuff_pred_thresh'] = -1
        self.gt_producer_cfgs = gt_producers
        self.tensorize = tensorize
        self.caffe_mode = caffe_mode
        self.gt_prod_handle = partial(GTGenerator, producer_cfgs=gt_producers)

    def read_img(self, img_fname):
        if self.caffe_mode:
            # cv2 reads imgs in BGR, which is what caffe trained models expect
            img = cv2.imread(img_fname)  # cv2 auto converts gray to BGR
            img = Image.fromarray(img)
        else:
            img = Image.open(img_fname).convert('RGB')
        return img

    def pan_getitem(self, index, apply_trans=True):
        # this is now exposed as public API
        imgMeta, segments_info, img, mask = PanopticBase.__getitem__(self, index)
        if apply_trans and self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return imgMeta, segments_info, img, mask

    def __getitem__(self, index):
        _, segments_info, im, pan_mask = self.pan_getitem(index)
        gts = []
        if not self.split.startswith('test'):
            lo_pan_mask = pan_mask.resize(
                np.array(im.size, dtype=np.int) // 4, resample=Image.NEAREST
            )
            gts = self.gt_prod_handle(
                self.meta, self.pcv, lo_pan_mask, segments_info
            ).generate_gt()

            if self.split == 'train':
                sem_gt = gts[0]
            else:
                hi_sem_gt = self.gt_prod_handle(
                    self.meta, self.pcv, pan_mask, segments_info,
                ).sem_gt
                sem_gt = hi_sem_gt
            gts[0] = sem_gt
        # else for test/test-dev, do not produce ground truth at all

        if self.tensorize:
            im = to_tensor(im)
            gts = [ torch.as_tensor(elem) for elem in gts ]
            if self.caffe_mode:
                im = caffe_imagenet_normalize(im)
            else:
                im = imagenet_normalize(im)
        else:
            im = np.array(im)

        gts.insert(0, im)
        return tuple(gts)

    @classmethod
    def make_loader(
        cls, data_cfg, pcv_module, is_train, mp_distributed, world_size,
        val_split='val'
    ):
        if is_train:
            split = 'train'
            batch_size = data_cfg.train_batch_size
            transforms_cfg = data_cfg.train_transforms
        else:
            split = val_split
            batch_size = data_cfg.test_batch_size
            transforms_cfg = data_cfg.test_transforms

        num_workers = data_cfg.num_loading_threads
        if mp_distributed:
            num_workers = int((num_workers + world_size - 1) / world_size)
            if is_train:
                batch_size = int(batch_size / world_size)
            # at test time a model does not need to reduce its batch size

        # 1. dataset
        trans = get_composed_augmentations(transforms_cfg)
        instance = cls(
            split=split, transforms=trans, pcv=pcv_module,
            gt_producers=data_cfg.dataset.gt_producers,
            **data_cfg.dataset.params,
        )
        # if split.startswith('test'):
        #     inx = global_cfg.testing.inx
        #     total_splits = global_cfg.testing.portions
        #     instance.test_seek_imgs(inx, total_splits)

        # 2. sampler
        sampler = cls.make_sampler(instance, is_train, mp_distributed)

        # 3. batch sampler
        batch_sampler = cls.make_batch_sampler(
            instance, sampler, batch_size=batch_size,
            aspect_grouping=cls.aspect_grouping,
        )
        del sampler

        # 4. collator
        if split.startswith('test'):
            collator = None
        else:
            collator = BatchCollator(data_cfg.dataset.gt_producers)

        # 5. loader
        loader = DataLoader(
            instance,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            # pin_memory=True maskrcnn-benchmark does not pin memory
        )
        return loader

    @staticmethod
    def make_sampler(dataset, is_train, distributed):
        if is_train:
            if distributed:
                # as of pytorch 1.1.0 the distributed sampler always shuffles
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        return sampler

    @staticmethod
    def make_batch_sampler(
        dataset, sampler, batch_size, aspect_grouping,
    ):
        if aspect_grouping:
            aspect_ratios = _compute_aspect_ratios(dataset)
            group_ids = _quantize(aspect_ratios, bins=[1, ])
            batch_sampler = GroupedBatchSampler(
                sampler, group_ids, batch_size, drop_uneven=False
            )
        else:
            batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler, batch_size, drop_last=False
            )
        '''
        I've decided after much deliberation not to use iteration based training.
        Our cluster has a 4-hour time limit before job interrupt.
        Under this constraint, I have to resume from where the sampling stopped
        at the exact epoch the interrupt occurs and this requires checkpointing
        the sampler state.
        However, under distributed settings, each process has to checkpoint a
        different state since each sees only a portion of the data by virtue of the
        distributed sampler. This is bug-prone and brittle.
        '''
        # if num_iters is not None:
        #     batch_sampler = samplers.IterationBasedBatchSampler(
        #         batch_sampler, num_iters, start_iter
        #     )
        return batch_sampler


class BatchCollator(object):
    def __init__(self, producer_cfg):
        fills = [0, ignore_index, ]  # always 0 for img, ignore for sem_gt
        fills.extend(
            [cfg['params']['fill'] for cfg in producer_cfg]
        )
        self.fills = fills

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        assert len(self.fills) == len(transposed_batch), 'must match in length'
        acc = []
        for tsr_list, fill in zip(transposed_batch, self.fills):
            tsr = self.collate_tensor_list(tsr_list, fill=fill)
            acc.append(tsr)
        return tuple(acc)

    @staticmethod
    def collate_tensor_list(tensors, fill):
        """
        Pad the Tensors with the specified constant
        so that they have the same shape
        """
        assert isinstance(tensors, (tuple, list))
        # largest size along each dimension
        max_size = tuple(max(s) for s in zip(*[tsr.shape for tsr in tensors]))
        assert len(max_size) == 2 or len(max_size) == 3
        batch_shape = (len(tensors),) + max_size
        batched_tsr = tensors[0].new(*batch_shape).fill_(fill)
        for tsr, pad_tsr in zip(tensors, batched_tsr):
            # WARNING only pad the last 2, that is the spatial dimensions
            pad_tsr[..., :tsr.shape[-2], :tsr.shape[-1]].copy_(tsr)
        return batched_tsr


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        imgMeta, _ = dataset.get_meta(i)
        aspect_ratio = float(imgMeta["height"]) / float(imgMeta["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios
