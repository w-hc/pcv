import numpy as np
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal

from ..components.snake import Snake

_nine_offsets = [
    ( 0,  0),
    ( 1,  1),
    ( 0,  1),
    (-1,  1),
    (-1,  0),
    (-1, -1),
    ( 0, -1),
    ( 1, -1),
    ( 1,  0),
]


class GaussianField():
    def __init__(self, diam, cov=0.05):
        assert (diam % 2 == 1), 'diam must be an odd'
        self.diam = diam
        self.cov = cov  # .05 leaves about 95% prob mass within central block
        # only consider the 3x3 region
        self.increment = 1 / diam
        # compute 3 units
        self.l, self.r = -1.5, 1.5
        self.field_shape = (3 * diam, 3 * diam)
        self.unit_area = self.increment ** 2
        self.prob_field = self.compute_prob_field()

    def compute_prob_field(self):
        cov = self.cov
        increment = self.increment
        l, r = self.l, self.r
        cov_mat = np.array([
            [cov, 0],
            [0, cov]
        ])
        rv = multivariate_normal([0, 0], cov_mat)
        half_increment = increment / 2
        xs, ys = np.mgrid[
            l + half_increment: r: increment,
            l + half_increment: r: increment
        ]  # use half increment to make things properly centered
        pos = np.dstack((xs, ys))
        prob_field = rv.pdf(pos).astype(np.float32)
        assert prob_field.shape == self.field_shape
        return prob_field

    @torch.no_grad()
    def compute_local_mass(self):
        kernel_size = self.diam
        pad = (kernel_size - 1) // 2
        prob_field = self.prob_field

        conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=kernel_size,
            padding=pad, bias=False
        )  # do not use cuda for now; no point
        conv.weight.data.copy_(torch.tensor(1.0))
        prob_field = torch.as_tensor(
            prob_field, device=conv.weight.device
        )[(None,) * 2]  # [1, 1, h, w]
        local_sum = conv(prob_field).squeeze().cpu().numpy()
        local_sum = local_sum * self.unit_area
        return local_sum


class MakeProbTsr():
    '''
    make a prob tsr of shape [h, w, num_votes] filled with the corresponding
    spatial voting prob
    '''
    def __init__(self, spec, diam, grid_spec, vote_mask, var=0.05):
        # indices grid of shape [2, H, W], where first dim is y, x; swap them
        # obtain [H, W, 2] where last channel is (y, x)
        self.spec = spec
        self.diam = diam
        self.vote_mask = vote_mask
        self.var = var

        # process grid spec to 0 based indexing and change radius to diam
        radius = (diam - 1) // 2
        center = np.array((radius, radius))
        grid_spec = grid_spec.copy()
        grid_spec[:, :2] += center
        grid_spec[:, -1] = 1 + 2 * grid_spec[:, -1]  # change from r to diam
        self.grid_spec = grid_spec

    def compute_voting_prob_tsr(self, normalize=True):
        spec = self.spec
        diam = self.diam
        grid_spec = self.grid_spec
        vote_mask = self.vote_mask

        spatial_shape = (diam, diam)
        spatial_yx = np.indices(spatial_shape).transpose(1, 2, 0).astype(int)
        # [H, W, 2] where each arr[y, x] is the containing grid's center
        spatial_cen_yx = np.empty_like(spatial_yx)
        # [H, W, 1] where each arr[y, x] is the containing grid's diam
        spatial_diam = np.empty(spatial_shape, dtype=int)[..., None]
        for i, (y, x, d) in enumerate(grid_spec):
            _m = vote_mask == i
            spatial_cen_yx[_m] = (y, x)
            spatial_diam[_m] = d

        max_vote_bin_diam = spec[-1][0]
        spatial_9_inds = self.nine_neighbor_inds(
            spatial_diam, spatial_yx, vote_mask,
            vote_mask_padding=max_vote_bin_diam
        )

        # spatial_9_probs = np.ones_like(spatial_9_inds).astype(float)
        spatial_9_probs = self.nine_neighbor_probs(
            spatial_diam, spatial_yx, spatial_cen_yx, self.var
        )

        # [H, W, num_votes + 1] 1 extra to trash the -1s
        spatial_prob = np.zeros((diam, diam, len(grid_spec) + 1))
        inds0, inds1, _ = np.ix_(range(diam), range(diam), range(1))
        np.add.at(spatial_prob, (inds0, inds1, spatial_9_inds), spatial_9_probs)
        spatial_prob[..., -1] = 0  # erase but keep the trash bin -> abstrain bin
        spatial_prob = self.erase_inward_prob_dist(spec, vote_mask, spatial_prob)
        if normalize:
            spatial_prob = spatial_prob / spatial_prob.sum(-1, keepdims=True)
        return spatial_prob

    @staticmethod
    def erase_inward_prob_dist(spec, vote_mask, spatial_prob):
        '''This is a measure of expedience borne of time constraints
        I can't help but feel ashamed of the time I have wasted dwelling on the
        right move; but the clock is ticking and I have to move on.
        '''
        splits = Snake.vote_channel_splits(spec)
        # ret = np.zeros_like(spatial_prob)
        layer_inds = np.cumsum(splits)
        for i in range(1, len(layer_inds)):
            curr = layer_inds[i]
            prev = layer_inds[i-1]
            belt_mask = (vote_mask < curr) & (vote_mask >= prev)
            spatial_prob[belt_mask, :prev] = 0
        return spatial_prob

    @staticmethod
    def nine_neighbor_inds(
        spatial_diam, spatial_yx, vote_mask, vote_mask_padding,
    ):
        # [H, W, 1, 1] * [9, 2] -> [H, W, 9, 2]
        spatial_9_offsets = spatial_diam[..., None] * np.array(_nine_offsets)
        # [H, W, 2] reshapes [H, W, 1, 2] + [H, W, 9, 2] -> [H, W, 9, 2]
        spatial_9_loc_yx = np.expand_dims(spatial_yx, 2) + spatial_9_offsets

        padded_vote_mask = np.pad(
            vote_mask, vote_mask_padding, mode='constant', constant_values=-1
        )
        # shift the inds
        spatial_9_loc_yx += (vote_mask_padding, vote_mask_padding)
        # [H, W, 9] where arr[y, x] contains the 9 inds centered on y, x
        spatial_9_inds = padded_vote_mask[
            tuple(np.split(spatial_9_loc_yx, 2, axis=-1))
        ].squeeze(-1)
        return spatial_9_inds

    @staticmethod
    def nine_neighbor_probs(spatial_diam, spatial_yx, spatial_cen_yx, var):
        spatial_cen_yx_offset = spatial_cen_yx - spatial_yx
        del spatial_cen_yx, spatial_yx
        single_cell_diam = 81
        field_diam = single_cell_diam * 3
        gauss = GaussianField(diam=single_cell_diam, cov=var)
        prob_local_mass = gauss.compute_local_mass()

        # now read off prob from every pix's 9 neighboring locations
        '''
        single_cell_diam: scalar; 1/3 of the field size for prob field
        prob_local_mass: [3 * single_cell_diam, 3 * single_cell_diam]
        spatial_diam: [H, W, 1]; arr[y, x] gives its grid diam
        spatial_cen_yx_offset: [H, W, 2] arr[y, x] gives dy, dx to its grid center
        '''
        assert field_diam == prob_local_mass.shape[0]
        assert prob_local_mass.shape[0] == prob_local_mass.shape[1]
        norm_spatial_cen_yx_offset = (
            spatial_cen_yx_offset * single_cell_diam / spatial_diam
        ).astype(np.int)  # [H, W, 2]
        del spatial_cen_yx_offset, spatial_diam

        spatial_9_offsets = (
            single_cell_diam * np.array(_nine_offsets)
        ).reshape(1, 1, 9, 2)
        field_radius = (field_diam - 1) // 2
        center = (field_radius, field_radius)
        spatial_yx_loc = center + norm_spatial_cen_yx_offset
        # [H, W, 2] reshapes [H, W, 1, 2] + [1, 1, 9, 2] -> [H, W, 9, 2]
        spatial_9_loc_yx = np.expand_dims(spatial_yx_loc, axis=2) + spatial_9_offsets

        spatial_9_probs = prob_local_mass[
            tuple(np.split(spatial_9_loc_yx, 2, axis=-1))
        ].squeeze(-1)
        return spatial_9_probs
