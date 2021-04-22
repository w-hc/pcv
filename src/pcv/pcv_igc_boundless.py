from bisect import bisect_left
import numpy as np

from .pcv import PCV_base
from .components.snake import Snake
from .components.ballot import Ballot
from .components.grid_specs import igc_specs
from .inference.mask_from_vote import MaskFromVote, MFV_CatSeparate

from .. import cfg


def _flesh_out_grid(spec):
    '''used only by flesh_out_spec'''
    field_diam, grid_spec = Snake.flesh_out_grid_spec(spec)
    full_mask = Snake.paint_trail_mask(field_diam, grid_spec)
    boundless_mask = Snake.paint_bound_ignore_trail_mask(field_diam, grid_spec)
    return {
        'full': full_mask,
        'boundless': boundless_mask
    }


def flesh_out_spec(spec_group):
    ret = {'base': None, 'pyramid': dict()}
    ret['base'] = _flesh_out_grid(spec_group['base'])
    for k, v in spec_group['pyramid'].items():
        ret['pyramid'][k] = _flesh_out_grid(v)
    return ret


class PCV_IGC_Boundless(PCV_base):
    def __init__(self):
        # grid inx for now is a dummy
        spec_group = igc_specs[cfg.pcv.grid_inx]
        self.num_groups = cfg.pcv.num_groups
        self.centroid_mode = cfg.pcv.centroid
        self.raw_spec = spec_group['base']
        _, self.grid_spec = Snake.flesh_out_grid_spec(self.raw_spec)
        self.mask_group = flesh_out_spec(spec_group)
        self._vote_mask = self.mask_group['base']['full']
        self._ballot_module = None
        self.coalesce_thresh = list(self.mask_group['pyramid'].keys())

    @property
    def ballot_module(self):
        # instantiate on demand to prevent train time data loading workers to
        # hold on to GPU memory
        if self._ballot_module is None:
            self._ballot_module = Ballot(self.raw_spec, self.num_groups).cuda()
        return self._ballot_module

    # 1 for bull's eye center, 1 for abstain vote
    @property
    def num_bins(self):
        return len(self.grid_spec)

    @property
    def num_votes(self):
        return 1 + self.num_bins

    @property
    def vote_mask(self):
        return self._vote_mask

    @property
    def query_mask(self):
        """
        Flipped from inside out
        """
        diam = len(self.vote_mask)
        radius = (diam - 1) // 2
        center = (radius, radius)
        mask_shape = self.vote_mask.shape
        offset_grid = np.indices(mask_shape).transpose(1, 2, 0)[..., ::-1]
        offsets = center - offset_grid
        # allegiance = self.discrete_vote_inx_from_offset(
        #     offsets.reshape(-1, 2)
        # ).reshape(mask_shape)
        allegiance = self._discretize_offset(
            self.vote_mask, offsets.reshape(-1, 2)
        ).reshape(mask_shape)
        return allegiance

    def centroid_from_ins_mask(self, ins_mask):
        return super().centroid_from_ins_mask(ins_mask)

    def discrete_vote_inx_from_offset(self, offset):
        base_boundless_mask = self.mask_group['base']['boundless']
        return self._discretize_offset(base_boundless_mask, offset)

    def tensorized_vote_from_offset(self, offset):
        """
        Args:
            offset: [N, 2] array of offset towards each pixel's own center,
                    Each row is filled with (x, y) pair, not (y, x)!
        Returns:
            vote_tsr: [N, num_votes] of bool tsr where 0/1 denotes gt entries.
        """
        # dispatch to the proper grid
        base_mask = self.mask_group['base']['boundless']
        radius = (len(base_mask) - 1) // 2
        max_offset = min(radius, np.abs(offset).max())
        inx = bisect_left(self.coalesce_thresh, max_offset)
        key = self.coalesce_thresh[inx]
        vote_mask = self.mask_group['pyramid'][key]['boundless']

        tsr = np.zeros((len(offset), self.num_votes), dtype=np.bool)  # [N, num_votes]
        gt_indices = self._discretize_offset(vote_mask, offset)  # [N, ]
        for inx in np.unique(gt_indices):
            if inx == -1:
                continue
            entries = np.unique(base_mask[vote_mask == inx])
            inds = np.where(gt_indices == inx)[0]
            tsr[inds.reshape(-1, 1), entries.reshape(1, -1)] = True
        return tsr

    def mask_from_sem_vote_tsr(self, dset_meta, sem_pred, vote_pred):
        # make the meta data actually required explicit!!
        if self.num_groups == 1:
            mfv = MaskFromVote(dset_meta, self, sem_pred, vote_pred)
        else:
            mfv = MFV_CatSeparate(dset_meta, self, sem_pred, vote_pred)
        pan_mask, meta = mfv.infer_panoptic_mask()
        return pan_mask, meta
