"""Module for dilation based pixel consensus votin
For now hardcode 3x3 voting kernel and see
"""
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from ..box_and_mask import get_xywh_bbox_from_binary_mask
from .. import cfg


class PCV_base(metaclass=ABCMeta):
    def __init__(self):
        # store the necessary modules
        pass

    @abstractproperty
    def num_bins(self):
        pass

    @abstractproperty
    def num_votes(self):
        pass

    @abstractproperty
    def vote_mask(self):
        pass

    @abstractproperty
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
        allegiance = self.discrete_vote_inx_from_offset(
            offsets.reshape(-1, 2)
        ).reshape(mask_shape)
        return allegiance

    @abstractmethod
    def centroid_from_ins_mask(self, ins_mask):
        mode = self.centroid_mode
        assert mode in ('bbox', 'cmass')
        if mode == 'bbox':
            bbox = get_xywh_bbox_from_binary_mask(ins_mask)
            x, y, w, h = bbox
            return [x + w // 2, y + h // 2]
        else:
            y, x = center_of_mass(ins_mask)
            x, y = int(x), int(y)
            return [x, y]

    @abstractmethod
    def discrete_vote_inx_from_offset(self, offset):
        pass

    @staticmethod
    def _discretize_offset(vote_mask, offset):
        """
        Args:
            offset: [N, 2] array of offset towards each pixel's own center,
                    Each row is filled with (x, y) pair, not (y, x)!
        """
        shape = offset.shape
        assert len(shape) == 2 and shape[1] == 2
        offset = offset[:, ::-1]  # swap to (y, x) for indexing

        diam = len(vote_mask)
        radius = (diam - 1) // 2
        center = (radius, radius)
        coord = offset + center
        del offset

        ret = -1 * np.ones(len(coord), dtype=np.int32)
        valid_inds = np.where(
            (coord[:, 0] >= 0) & (coord[:, 0] < diam)
            & (coord[:, 1] >= 0) & (coord[:, 1] < diam)
        )[0]
        _y_inds, _x_inds = coord[valid_inds].T
        vals = vote_mask[_y_inds, _x_inds]
        ret[valid_inds] = vals
        return ret

    @abstractmethod
    def mask_from_sem_vote_tsr(self, dset_meta, sem_pred, vote_pred):
        pass
