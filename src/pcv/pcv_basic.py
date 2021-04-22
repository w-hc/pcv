from .pcv import PCV_base
from .components.snake import Snake
from .components.ballot import Ballot
from .components.grid_specs import specs
from .inference.mask_from_vote import MaskFromVote, MFV_CatSeparate

from .. import cfg


class PCV_Basic(PCV_base):
    def __init__(self):
        spec = specs[cfg.pcv.grid_inx]
        self.num_groups = cfg.pcv.num_groups
        self.centroid_mode = cfg.pcv.centroid
        self.raw_spec = spec
        field_diam, grid_spec = Snake.flesh_out_grid_spec(spec)
        self.grid_spec = grid_spec
        self._vote_mask = Snake.paint_trail_mask(field_diam, grid_spec)
        self._ballot_module = None

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
        return super().query_mask

    def centroid_from_ins_mask(self, ins_mask):
        return super().centroid_from_ins_mask(ins_mask)

    def discrete_vote_inx_from_offset(self, offset):
        return self._discretize_offset(self.vote_mask, offset)

    def mask_from_sem_vote_tsr(self, dset_meta, sem_pred, vote_pred):
        # make the meta data actually required explicit!!
        if self.num_groups == 1:
            mfv = MaskFromVote(dset_meta, self, sem_pred, vote_pred)
        else:
            mfv = MFV_CatSeparate(dset_meta, self, sem_pred, vote_pred)
        pan_mask, meta = mfv.infer_panoptic_mask()
        return pan_mask, meta
