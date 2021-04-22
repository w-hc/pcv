import numpy as np
import torch
import torch.nn as nn

from .snake import Snake


class Ballot(nn.Module):
    def __init__(self, spec, num_groups):
        super().__init__()
        self.num_groups = num_groups
        votes = []
        smears = []

        splits = []
        acc_size = spec[0][0]  # inner most size
        for i, (size, num_rounds) in enumerate(spec):
            inner_blocks = acc_size // size
            total_blocks = inner_blocks + 2 * num_rounds
            acc_size += 2 * num_rounds * size

            prepend = (i == 0)
            num_in_chnls = (total_blocks ** 2 - inner_blocks ** 2) + int(prepend)
            num_in_chnls *= num_groups
            num_out_chnls = 1 * num_groups
            splits.append(num_in_chnls)
            deconv_vote = nn.ConvTranspose2d(
                in_channels=num_in_chnls, out_channels=num_out_chnls,
                kernel_size=total_blocks, stride=1,
                padding=(total_blocks - 1) // 2 * size, dilation=size,
                groups=num_groups, bias=False
            )
            # each group [ num_chnl, 1, 3, 3 ] -> [num_groups * num_chnl, 1, 3, 3]
            throw_away = 0 if prepend else inner_blocks
            weight = torch.cat([
                get_voting_deonv_kernel_weight(
                    side=total_blocks, throw_away=throw_away
                )
                for _ in range(num_groups)
            ], dim=0)
            deconv_vote.weight.data.copy_(weight)

            votes.append(deconv_vote)
            smear_ker = nn.AvgPool2d(
                # in_channels=num_out_chnls, out_channels=num_out_chnls,
                kernel_size=size, stride=1, padding=int( (size - 1) / 2 )
            )
            # smear_ker = nn.ConvTranspose2d(
            #     in_channels=num_out_chnls, out_channels=num_out_chnls,
            #     kernel_size=size, stride=1, padding=int( (size - 1) / 2 ),
            #     groups=num_groups, bias=False
            # )
            # smear_ker.weight.data.fill_(1 / (size ** 2 ))
            smears.append(smear_ker)

        self.splits = splits
        self.votes = nn.ModuleList(votes)
        self.smears = nn.ModuleList(smears)

    @torch.no_grad()
    def forward(self, x):
        num_groups = self.num_groups
        splitted = x.split(self.splits, dim=1)
        assert len(splitted) == len(self.votes)
        output = []
        for i in range(len(splitted)):
            # if i == (len(splitted) - 1):
            #     continue
            x = splitted[i]
            if num_groups > 1:  # painful lesson
                _, C, H, W = x.shape
                x = x.reshape(-1, C // num_groups, num_groups, H, W)
                x = x.transpose(1, 2)
                x = x.reshape(-1, C, H, W)
            x = self.votes[i](x)
            x = self.smears[i](x)
            output.append(x)
        return sum(output)


def get_voting_deonv_kernel_weight(side, throw_away, return_tsr=True):
    """
    The logic is neat; it makes use of the fact that negatives will be indexed
    from the last channel, and one simply needs to throw those away
    """
    assert throw_away <= side
    throw_away = throw_away ** 2
    rounds = (side - 1) // 2
    spatial_inds = Snake.paint_trail_mask(
        *Snake.flesh_out_grid_spec([[1, rounds]])
    ) - throw_away
    weight = np.zeros(shape=(side, side, side ** 2))
    dim_0_inds, dim_1_inds = np.ix_(range(side), range(side))
    weight[dim_0_inds, dim_1_inds, spatial_inds] = 1
    if throw_away > 0:
        weight = weight[:, :, :-throw_away]
    if not return_tsr:
        return weight
    else:
        weight = np.expand_dims(weight.transpose(2, 0, 1), axis=1)
        kernel = torch.as_tensor(weight).float()
        return kernel
