import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import apex
from panoptic.models.base_model import BaseModel
from panoptic import get_loss, get_optimizer, get_scheduler
from ..pcv.inference.mask_from_vote import MaskFromVote, MFV_CatSeparate

from fabric.utils.logging import setup_logging
logger = setup_logging(__file__)


class PanopticBase(BaseModel):
    def __init__(self, cfg, pcv, dset_meta):
        super().__init__()
        self.cfg = cfg
        self.pcv = pcv
        self.dset_meta = dset_meta

        self.instantiate_network(cfg)
        assert self.net is not None, "instantiate network properly!"
        self.criteria = get_loss(cfg.loss)
        self.net.loss_module = self.criteria  # fold crit into net

        # torch.cuda.synchronize()
        self.add_optimizer(cfg.optimizer)
        assert self.optimizer is not None  # confirm the child has done the job
        # torch.cuda.synchronize()

        self.curr_epoch = 0
        self.total_train_epoch = cfg.scheduler.total_epochs
        self.scheduler = get_scheduler(cfg.scheduler)(
            optimizer=self.optimizer
        )

        # used as temporary place holder for model training inputs
        self._latest_loss = None
        self.img, self.sem_mask, self.vote_mask, self.weight_mask = \
            None, None, None, None

    def instantiate_network(self, cfg):
        # child class must overwrite the method
        raise NotImplementedError()

    def get_params_lr(self, initial_lr):
        return self.net.parameters()

    def add_optimizer(self, optim_cfg):
        """New Method for this class. Only retrive the module
        Children class should implement the function s.t self.optimizer
        is filled. The child class could add different learning rate for
        different params, etc
        """
        optim_handle = get_optimizer(optim_cfg)
        # self.optimizer = optim_handle(params=self.net.parameters())
        params_lr = self.get_params_lr(optim_cfg.params.lr)
        self.optimizer = optim_handle(params=params_lr)

    def ingest_train_input(self, *inputs):
        self.inputs = inputs

    def optimize_params(self):
        self.set_train_eval_state(True)
        self.optimizer.zero_grad()

        loss, sem_loss, vote_loss = self.net(*self.inputs)
        loss = loss.mean()  # warning here.
        if getattr(self.cfg, 'apex', False):
            with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()
        self._latest_loss = {
            'loss': loss.item(),
            'sem_loss': sem_loss.mean().item(),
            'vote_loss': vote_loss.mean().item()
        }

    def latest_loss(self, get_numeric=False):
        if get_numeric:
            return self._latest_loss
        else:
            return {k: '{:.4f}'.format(v) for k, v in self._latest_loss.items()}

    def curr_lr(self):
        lr = [ group['lr'] for group in self.optimizer.param_groups ][0]
        return lr

    def advance_to_next_epoch(self):
        self.curr_epoch += 1

    def log_statistics(self, step, level=0):
        if self.log_writer is not None:
            for k, v in self.latest_loss(get_numeric=True).items():
                self.log_writer.add_scalar(k, v, step)
            if level >= 1:
                curr_lr = self.curr_lr()
                self.log_writer.add_scalar('lr', curr_lr, step)

    def infer(self, x, *args, **kwargs):
        return self._infer(x, *args, **kwargs)

    def _infer(
        self, x, upsize_sem=False, take_argmax=False, softmax_normalize=False
    ):
        self.set_train_eval_state(False)
        with torch.no_grad():
            sem_pred, vote_pred = self.net(x)

        if upsize_sem:
            sem_pred = F.interpolate(
                sem_pred, scale_factor=4, mode='nearest'
            )

        if take_argmax:
            sem_pred, vote_pred = sem_pred.max(1)[1], vote_pred.max(1)[1] #sem_pred.argmax(1), vote_pred.argmax(1)
            return sem_pred, vote_pred

        if softmax_normalize:
            sem_pred = F.softmax(sem_pred, dim=1)
            vote_pred = F.softmax(vote_pred, dim=1)

        return sem_pred, vote_pred

    def _flip_infer(self, x, *args, **kwargs):
        # x: 1x3xhxw
        new_x = torch.cat([x, x.flip(-1)], 0)
        _sem_pred, _vote_pred = self._infer(new_x, *args, **kwargs)

        if not hasattr(self, 'flip_infer_mapping'):
            original_mask = torch.from_numpy(self.pcv.vote_mask)
            new_mask = original_mask.flip(-1)
            mapping = {}
            for ii, jj in zip(new_mask.view(-1).tolist(), original_mask.view(-1).tolist()):
                if ii in mapping:
                    assert mapping[ii] == jj
                else:
                    mapping[ii] = jj
            mapping[len(mapping)] = len(mapping) #abstain
            self.flip_infer_mapping = torch.LongTensor([mapping[_] for _ in range(len(mapping))])

        # sem_pred = (_sem_pred[:1] + _sem_pred[1:].flip(-1)) / 2
        sem_pred = F.softmax((_sem_pred[:1].log() + _sem_pred[1:].flip(-1).log()) / 2, dim=1)
        # vote_pred = (_vote_pred[:1] + _vote_pred[1:, self.flip_infer_mapping].flip(-1)) / 2
        vote_pred = F.softmax((_vote_pred[:1].log() + _vote_pred[1:, self.flip_infer_mapping].flip(-1).log()) / 2, dim=1)
        return sem_pred, vote_pred

    def stitch_pan_mask(self, infer_cfg, sem_pred, vote_pred, target_size=None, return_hmap=False):
        """
        Args:
            sem_pred:  [1, num_class, H, W] torch gpu tsr
            vote_pred: [1, num_bins,  H, W] torch gpu tsr
            target_size: optionally postprocess the output, resize, filter
                        stuff predictions on threshold, etc
        """
        assert self.dset_meta is not None and self.pcv is not None
        # make the meta data actually required explicit!!
        infer_m = MaskFromVote if infer_cfg.num_groups == 1 else MFV_CatSeparate
        mfv = infer_m(infer_cfg, self.dset_meta, self.pcv, sem_pred, vote_pred)
        pan_mask, meta = mfv.infer_panoptic_mask()
        vote_hmap = mfv.vote_hmap
        # pan_mask, meta = self.pcv.mask_from_sem_vote_tsr(
        #     self.dset_meta, sem_pred, vote_pred
        # ) don't go the extra route. Be straightforward

        if target_size is not None:
            stuff_filt_thresh = infer_cfg.get(
                'stuff_pred_thresh', self.dset_meta['stuff_pred_thresh']
            )
            pan_mask, meta = self._post_process_pan_pred(
                pan_mask, meta, target_size, stuff_filt_thresh
            )
        if not return_hmap:
            return pan_mask, meta
        else:
            return pan_mask, meta, vote_hmap

    @staticmethod
    def _post_process_pan_pred(pan_mask, pan_meta, target_size, stuff_thresh=-1):
        """Assume that pan_mask is np array and target is a PIL Image
        adjust annotations as needed when size change erases instances
        """
        pan_mask = Image.fromarray(pan_mask)
        pan_mask_size = pan_mask.size
        pan_mask = np.array(pan_mask.resize(target_size, resample=Image.NEAREST))

        # account for lost segments due to downsizing
        if pan_mask_size[0] > target_size[0]:
            # downsizing is the only case where instances could be erased
            remains = np.unique(pan_mask)
            segs = pan_meta['segments_info']
            acc = []
            for seg in segs:
                if seg['id'] not in remains:
                    logger.warn('segment erased due to downsizing')
                else:
                    acc.append(seg)
            pan_meta['segments_info'] = acc

        # filter out stuff segments based on area threshold
        segs = pan_meta['segments_info']
        acc = []
        for seg in segs:
            if seg['isthing'] == 0:
                _mask = (pan_mask == seg['id'])
                area = _mask.sum()
                if area > stuff_thresh:
                    acc.append(seg)
                else:
                    pan_mask[_mask] = 0
            else:
                acc.append(seg)
        pan_meta['segments_info'] = acc

        return pan_mask, pan_meta

    def load_latest_checkpoint_if_available(self, manager, direct_ckpt=None):
        ckpt = manager.load_latest() if direct_ckpt is None else direct_ckpt
        if ckpt:
            self.curr_epoch = ckpt['curr_epoch']
            self.net.module.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optim_state'])
            # scheduler is incremented by global_step in the training loop
            self.scheduler = get_scheduler(self.cfg.scheduler)(
                optimizer=self.optimizer,
            )
            logger.info("loaded checkpoint that completes epoch {}".format(
                ckpt['curr_epoch']))
            self.curr_epoch += 1
        else:
            logger.info("No checkpoint found")

    def write_checkpoint(self, manager):
        ckpt_obj = dict(
            curr_epoch=self.curr_epoch,
            model_state=self.net.module.state_dict(),
            optim_state=self.optimizer.state_dict(),
        )
        manager.save(self.curr_epoch, ckpt_obj)


class AbstractNet(nn.Module):
    def forward(self, *inputs):
        """
        If gt is supplied, then compute loss
        """
        x = inputs[0]
        x = self.stage1(x)  # usually are resnet, hrnet, mobilenet etc.
        x = self.stage2(x)  # like fpn, aspp, ups, etc.
        sem_pred = self.sem_classifier(x[0])
        vote_pred = self.vote_classifier(x[1])

        if len(inputs) > 1:
            assert self.loss_module is not None
            loss = self.loss_module(sem_pred, vote_pred, *inputs[1:])
            return loss
        else:
            return sem_pred, vote_pred
