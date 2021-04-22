import numpy as np

from .pcv import PCV_base
from .components.snake import Snake
from .components.ballot import Ballot
from .components.grid_specs import specs
from .gaussian_smooth.prob_tsr import MakeProbTsr
from .inference.mask_from_vote import MaskFromVote, MFV_CatSeparate

from .. import cfg


def flesh_out_grid(spec):
    field_diam, grid_spec = Snake.flesh_out_grid_spec(spec)
    vote_mask = Snake.paint_trail_mask(field_diam, grid_spec)
    return vote_mask


def flesh_out_spec(spec_group):
    ret = {'base': None, 'pyramid': dict()}
    ret['base'] = flesh_out_grid(spec_group['base'])
    for k, v in spec_group['pyramid'].items():
        ret['pyramid'][k] = flesh_out_grid(v)
    return ret


class PCV_Smooth(PCV_base):
    def __init__(self, grid_inx):
        self.num_groups = cfg.pcv.num_groups
        self.spec = specs[cfg.pcv.grid_inx]
        self.centroid_mode = cfg.pcv.centroid
        field_diam, grid_spec = Snake.flesh_out_grid_spec(self.spec)
        self.grid_spec = grid_spec
        self._vote_mask = Snake.paint_trail_mask(field_diam, grid_spec)
        self._ballot_module = None

        maker = MakeProbTsr(
            self.spec, field_diam, grid_spec, self.vote_mask, var=0.05
        )
        self.spatial_prob_tsr = maker.compute_voting_prob_tsr(normalize=True)

    @property
    def ballot_module(self):
        # instantiate on demand to prevent train time data loading workers to
        # hold on to GPU memory
        if self._ballot_module is None:
            self._ballot_module = Ballot(self.spec, self.num_groups).cuda()
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
        return super().query_mask

    def centroid_from_ins_mask(self, ins_mask):
        return super().centroid_from_ins_mask(ins_mask)

    def discrete_vote_inx_from_offset(self, offset):
        return self._discretize_offset(self.vote_mask, offset)

    def smooth_prob_tsr_from_offset(self, offset):
        """
        Args:
            offset: [N, 2] array of offset towards each pixel's own center,
                    Each row is filled with (x, y) pair, not (y, x)!
        Returns:
            vote_tsr: [N, num_votes] of float tsr with each entry being a prob
        """
        shape = offset.shape
        assert len(shape) == 2 and shape[1] == 2
        offset = offset[:, ::-1]  # swap to (y, x) for indexing

        diam = len(self.spatial_prob_tsr)
        radius = (diam - 1) // 2
        center = (radius, radius)
        coord = offset + center

        # [N, num_votes]
        tsr = np.zeros((len(offset), self.num_votes), dtype=np.float32)
        valid_inds = np.where(
            (coord[:, 0] >= 0) & (coord[:, 0] < diam)
            & (coord[:, 1] >= 0) & (coord[:, 1] < diam)
        )[0]
        _y_inds, _x_inds = coord[valid_inds].T
        tsr[valid_inds] = self.spatial_prob_tsr[_y_inds, _x_inds]
        return tsr

    def mask_from_sem_vote_tsr(self, dset_meta, sem_pred, vote_pred):
        # make the meta data actually required explicit!!
        if self.num_groups == 1:
            mfv = MaskFromVote(dset_meta, self, sem_pred, vote_pred)
        else:
            mfv = MFV_CatSeparate(dset_meta, self, sem_pred, vote_pred)
        pan_mask, meta = mfv.infer_panoptic_mask()
        return pan_mask, meta
