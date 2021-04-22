import os
import os.path as osp
import json
import numpy as np
from PIL import Image
import torch.nn.functional as F
from easydict import EasyDict as edict
from .metric import PanMetric as Metric, PQMetric
from panopticapi.utils import rgb2id, id2rgb

from fabric.io import save_object

_VALID_ORACLE_MODES = ('full', 'sem', 'vote')


class BaseReporter():
    def __init__(self, infer_cfg, model, dset, output_root):
        self.output_root = output_root

    def process(self, model_outputs, inputs, dset):
        pass

    def generate_report(self):
        # return some report here
        pass


class mIoU(BaseReporter):
    def __init__(self, infer_cfg, model, dset, output_root):
        del infer_cfg  # not needed
        self.metric = Metric(
            model.dset_meta['num_classes'],
            model.pcv.num_votes, model.dset_meta['trainId_2_catName']
        )
        self.output_root = output_root

    def process(self, inx, model_outputs, inputs, dset):
        sem_pred, vote_pred = model_outputs
        sem_pred = F.interpolate(
            sem_pred, scale_factor=4, mode='nearest'
        )
        sem_pred, vote_pred = sem_pred.argmax(1), vote_pred.argmax(1)
        self.metric.update(sem_pred, vote_pred, *inputs[1:3])

    def generate_report(self):
        # return some report here
        return str(self.metric)


class PQ_report(BaseReporter):
    def __init__(self, infer_cfg, model, dset, output_root):
        self.infer_cfg = edict(infer_cfg.copy())
        self.output_root = output_root
        self.metric = PQMetric(dset.meta)
        self.overall_pred_meta = {
            'images': list(dset.imgs.values()),
            'categories': list(dset.meta['cats'].values()),
            'annotations': []
        }
        self.model = model
        self.oracle_mode = self.infer_cfg['oracle_mode']
        if self.oracle_mode is not None:
            assert self.oracle_mode in _VALID_ORACLE_MODES

        os.makedirs(osp.dirname(output_root), exist_ok=True)
        PRED_OUT_NAME = 'pred'
        self.pan_json_fname = osp.join(output_root, '{}.json'.format(PRED_OUT_NAME))
        self.pan_mask_dir = osp.join(output_root, PRED_OUT_NAME)

    def process(self, inx, model_outputs, inputs, dset):
        from panoptic.entry import gt_tsr_res_reduction  # HELPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
        sem_pd, vote_pd = model_outputs
        sem_pd, vote_pd = F.softmax(sem_pd, dim=1), F.softmax(vote_pd, dim=1)

        model = self.model
        oracle_mode = self.oracle_mode
        oracle_res = 4
        imgMeta, segments_info, _, pan_gt_mask = dset.pan_getitem(
            inx, apply_trans=False
        )
        if dset.transforms is not None:
            _, trans_pan_gt = dset.transforms(_, pan_gt_mask)
        else:
            trans_pan_gt = pan_gt_mask.copy()

        pan_gt_ann = {
            'image_id': imgMeta['id'],
            # shameful mogai; can only access image f_name here. alas...
            'file_name': imgMeta['file_name'].split('.')[0] + '.png',
            'segments_info': list(segments_info.values())
        }

        if oracle_mode is not None:
            sem_ora, vote_ora = gt_tsr_res_reduction(
                oracle_res, dset.gt_prod_handle,
                dset.meta, model.pcv, trans_pan_gt, segments_info
            )
            if oracle_mode == 'vote':
                pass  # if using model sem pd, maintain stuff pred thresh
            else:  # sem or full oracle, using gt sem pd, do not filter
                self.infer_cfg['stuff_pred_thresh'] = -1

            if oracle_mode == 'sem':
                sem_pd = sem_ora
            elif oracle_mode == 'vote':
                vote_pd = vote_ora
            else:
                sem_pd, vote_pd = sem_ora, vote_ora

        pan_pd_mask, pan_pd_ann = model.stitch_pan_mask(
            self.infer_cfg, sem_pd, vote_pd, pan_gt_mask.size
        )
        pan_pd_ann['image_id'] = pan_gt_ann['image_id']
        pan_pd_ann['file_name'] = pan_gt_ann['file_name']
        self.overall_pred_meta['annotations'].append(pan_pd_ann)

        self.metric.update(
            pan_gt_ann, rgb2id(np.array(pan_gt_mask)), pan_pd_ann, pan_pd_mask
        )

        pan_pd_mask = Image.fromarray(id2rgb(pan_pd_mask))
        fname = osp.join(self.pan_mask_dir, pan_pd_ann['file_name'])
        os.makedirs(osp.dirname(fname), exist_ok=True)  # make region subdir
        pan_pd_mask.save(fname)

    def generate_report(self):
        with open(self.pan_json_fname, 'w') as f:
            json.dump(self.overall_pred_meta, f)
        save_object(
            self.metric.state_dict(), osp.join(self.output_root, 'score.pkl')
        )
        return self.metric


reporter_modules = {
    'mIoU': mIoU,
    'pq': PQ_report
}
