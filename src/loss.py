import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cfg
from .datasets.base import ignore_index
from detectron2.layers.batch_norm import AllReduce
import torch.distributed as dist


class Loss(nn.Module):
    def __init__(self, w_vote, w_sem):
        super().__init__()
        self.w_vote = w_vote
        self.w_sem = w_sem

    def forward(self, *args, **kwargs):
        '''this 3-step interface is enforced for decomposable loss visulization
        see /vis.py
        '''
        sem_loss_mask, vote_loss_mask = self.per_pix_loss(*args, **kwargs)
        if len(args) == 5:
            sem_weight_mask = vote_weight_mask = args[-1]  # this is risky. replying on the fact that if
        else:
            sem_weight_mask, vote_weight_mask = args[-2:]
        # weight_mask is not needed, then just supplying an ignored parameter
        sem_loss_mask, vote_loss_mask = self.normalize(
            sem_loss_mask, vote_loss_mask, sem_weight_mask, vote_weight_mask
        )
        loss, s_loss, v_loss = self.aggregate(sem_loss_mask, vote_loss_mask)
        return loss, s_loss, v_loss

    def per_pix_loss(self, *args, **kwargs):
        '''compute raw per pixel loss that reflects how good a pixel decision is
        do not apply any segment based normalization yet
        '''
        raise NotImplementedError()

    def normalize(self, sem_loss_mask, vote_loss_mask, weight_mask):
        raise NotImplementedError()

    def aggregate(self, sem_loss_mask, vote_loss_mask):
        s_loss, v_loss = sem_loss_mask.sum(), vote_loss_mask.sum()
        loss = self.w_sem * s_loss + self.w_vote * v_loss
        return loss, s_loss, v_loss


def loss_normalize(loss, weight_mask, local_batch_size):
    '''An abomination of a method. Have to live with it for now but
    I will delete it soon enough. This 'total' option makes no sense except to
    serve the expedience of making the loss small
    '''
    norm = cfg.loss.normalize
    if norm == 'batch':
        loss = (loss * weight_mask) / local_batch_size
    elif norm == 'total':
        loss = (loss * weight_mask) / (weight_mask.sum() + 1e-10)
    elif norm == 'segs_across_batch':
        global_seg_cnt = weight_mask.sum()
        if torch.distributed.is_initialized():
            global_seg_cnt = AllReduce.apply(global_seg_cnt) / dist.get_world_size()
        # print('local sum {:.2f} vs global sum {:.2f}'.format(weight_mask.sum(), weight_mask_sum))
        loss = (loss * weight_mask) / (global_seg_cnt + 1e-10)
    else:
        raise ValueError('invalid value for normalization: {}'.format(norm))
    return loss


class PanLoss(Loss):
    def per_pix_loss(self, sem_pred, vote_pred, sem_mask, vote_mask, *args, **kwargs):
        sem_loss_mask = F.cross_entropy(
            sem_pred, sem_mask, ignore_index=ignore_index, reduction='none'
        )
        vote_loss_mask = F.cross_entropy(
            vote_pred, vote_mask, ignore_index=ignore_index, reduction='none'
        )
        vote_loss_mask = vote_loss_mask  # * 0.1
        return sem_loss_mask, vote_loss_mask

    def normalize(self, sem_loss_mask, vote_loss_mask, *args, **kwargs):
        sem_loss_mask = sem_loss_mask / sem_loss_mask.numel()
        vote_loss_mask = vote_loss_mask / vote_loss_mask.numel()
        return sem_loss_mask, vote_loss_mask


class MaskedPanLoss(PanLoss):
    def normalize(self, sem_loss_mask, vote_loss_mask, sem_weight_mask, vote_weight_mask):
        assert len(sem_loss_mask) == len(vote_loss_mask)
        assert sem_weight_mask is not None and vote_weight_mask is not None
        batch_size = len(sem_loss_mask)
        sem_loss_mask = loss_normalize(sem_loss_mask, sem_weight_mask, batch_size)
        vote_loss_mask = loss_normalize(vote_loss_mask, vote_weight_mask, batch_size)
        return sem_loss_mask, vote_loss_mask


class DeeperlabPanLoss(PanLoss):
    def normalize(self, sem_loss_mask, vote_loss_mask, sem_weight_mask, vote_weight_mask):
        assert len(sem_loss_mask) == len(vote_loss_mask)
        assert sem_weight_mask is not None and vote_weight_mask is not None
        batch_size = len(sem_loss_mask)
        sem_loss_mask = deeperlab_loss_normalize(sem_loss_mask, sem_weight_mask, batch_size)
        vote_loss_mask = deeperlab_loss_normalize(vote_loss_mask, vote_weight_mask, batch_size)
        return sem_loss_mask, vote_loss_mask


def deeperlab_loss_normalize(loss, weight_mask, local_batch_size):
    new_weight = (weight_mask > 1/16/16).float()*2+1  # == area < 16x16
    flat_loss = loss.reshape(loss.shape[0], -1)  # b x h x w -> b x hw
    new_weight = new_weight.reshape(new_weight.shape[0], -1)

    topk_loss, topk_inx = flat_loss.topk(int(flat_loss.shape[-1] * 0.15), sorted=False, dim=-1)
    topk_weight = new_weight.gather(1, topk_inx)

    loss = (topk_loss * topk_weight).mean()
    return loss

# class TsrCoalesceLoss(Loss):
#     def per_pix_loss(
#         self, sem_pred, vote_pred, sem_mask, vote_mask, vote_bool_tsr, weight_mask
#     ):
#         del vote_mask  # vote_mask is accepted merely for parameter compatibility
#         sem_loss_mask = self.regular_ce_loss(sem_pred, sem_mask, weight_mask)
#         vote_loss_mask = self.unsophisticated_loss(vote_pred, vote_bool_tsr, weight_mask)
#         return sem_loss_mask, vote_loss_mask

#     @staticmethod
#     def regular_ce_loss(pred, lbls, weight_mask):
#         return MaskedPanLoss._single_loss(pred, lbls, weight_mask)

#     @staticmethod
#     def booltsr_loss(pred, bool_tsr, weight_mask):
#         raise ValueError('cannot be back-propagated')
#         is_valid = bool_tsr.any(dim=1)  # [N, H, W]
#         weight_mask = weight_mask[is_valid]  # [num_valid, ]
#         # pred = pred.permute(0, 2, 3, 1)
#         # bool_tsr = bool_tsr.permute(0, 2, 3, 1)
#         # pred, bool_tsr = pred[is_valid], bool_tsr[is_valid]  # [num_valid, C]

#         bottom = torch.logsumexp(pred, dim=1)
#         pred = torch.where(bool_tsr, pred, torch.tensor(float('-inf')).cuda())
#         pred = torch.logsumexp(pred, dim=1)
#         loss = (bottom - pred)[is_valid]  # -1 is implicit here by reversing order
#         loss = (loss * weight_mask).sum() / weight_mask.sum()
#         return loss

#     @staticmethod
#     def unsophisticated_loss(pred, bool_tsr, weight_mask):
#         raise ValueError('validity mask changes spatial shape; embarassing')
#         is_valid = bool_tsr.any(dim=1)  # [N, H, W]
#         weight_mask = weight_mask[is_valid]

#         pred = F.softmax(pred, dim=1)
#         pred = torch.where(bool_tsr, pred, torch.tensor(0.).cuda())
#         loss = torch.log(pred.sum(dim=1)[is_valid])
#         loss = loss_normalize(loss, weight_mask, len(pred))
#         loss = -1 * loss
#         return loss


# class MaskedKLPanLoss(PanLoss):
#     '''
#     cross entropy loss for semantic segmentation and KL-divergence loss for
#     voting
#     '''
#     def per_pix_loss(
#         self, sem_pred, vote_pred,
#         sem_mask, vote_mask, vote_gt_prob, weight_mask
#     ):
#         sem_loss_mask = self.sem_loss(sem_pred, sem_mask, weight_mask)
#         vote_loss_mask = self.vote_loss(vote_pred, vote_gt_prob, weight_mask)
#         return sem_loss_mask, vote_loss_mask

#     @staticmethod
#     def sem_loss(sem_pred, sem_mask, weight_mask):
#         loss = F.cross_entropy(
#             sem_pred, sem_mask, ignore_index=ignore_index, reduction='none'
#         )
#         loss = loss_normalize(loss, weight_mask, len(sem_pred))
#         return loss

#     @staticmethod
#     def vote_loss(vote_pred, vote_gt_prob, weight_mask):
#         loss = F.kl_div(
#             F.log_softmax(vote_pred, dim=1), vote_gt_prob, reduction='none'
#         )
#         loss = loss.sum(dim=1)
#         loss = loss_normalize(loss, weight_mask, len(vote_pred))
#         return loss


# class _CELoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, pred, mask):
#         """Assume that ignore index is set at 0
#         pred: [N, C, H, W]
#         mask: [N, H, W]
#         """
#         loss = F.cross_entropy(
#             pred, mask, ignore_index=ignore_index, reduction='mean'
#         )
#         return loss


class NormalizedFocalPanLoss(nn.Module):
    # Adapt from adpatis
    def __init__(self, w_vote=0.5, w_sem=0.5, gamma=0, alpha=1):
        super().__init__()
        # assert w_vote + w_sem == 1
        self.w_vote = w_vote
        self.w_sem = w_sem
        self.gamma = gamma

    def focal_loss(self, input, target):
        logpt = - F.cross_entropy(input, target, ignore_index=ignore_index, reduction='none')
        pt = torch.exp(logpt)

        beta = (1 - pt) ** self.gamma

        t = target != ignore_index
        t_sum = t.float().sum(axis=[-2, -1], keepdims=True)
        beta_sum = beta.sum(axis=[-2, -1], keepdims=True)

        eps = 1e-10

        mult = t_sum / (beta_sum + eps)
        if True:
            mult = mult.detach()
        beta = beta * mult

        loss = - beta * logpt.clamp(min=-20) # B x H x W

        # size average
        loss = loss.sum(axis=[-2,-1]) / t_sum.squeeze()
        loss = loss.sum()

        return loss

    def forward(self, sem_pred, vote_pred, sem_mask, vote_mask, *args, **kwargs):
        sem_loss = self.focal_loss(sem_pred, sem_mask)
        vote_loss = self.focal_loss(vote_pred, vote_mask)
        loss = self.w_vote * vote_loss + self.w_sem * sem_loss
        return loss, sem_loss, vote_loss


# class MaskedPanLoss(Loss):
#     def forward(self, sem_pred, vote_pred, sem_mask, vote_mask, weight_mask, vote_weight_mask=None):
#         sem_loss = self._single_loss(sem_pred, sem_mask, weight_mask)
#         if vote_weight_mask is None:
#             vote_weight_mask = weight_mask
#         vote_loss = self._single_loss(vote_pred, vote_mask, vote_weight_mask)
#         loss = self.w_vote * vote_loss + self.w_sem * sem_loss
#         return loss, sem_loss, vote_loss

#     @staticmethod
#     def _single_loss(pred, lbls, weight_mask):
#         loss = F.cross_entropy(
#             pred, lbls, ignore_index=ignore_index, reduction='none'
#         )
#         loss = loss_normalize(loss, weight_mask, len(pred))
#         return loss


# class TsrCoalesceLoss(nn.Module):
#     def __init__(self, w_vote=0.5, w_sem=0.5):
#         super().__init__()
#         self.w_vote = w_vote
#         self.w_sem = w_sem

#     def forward(
#         self, sem_pred, vote_pred, sem_mask, vote_mask, vote_bool_tsr, weight_mask
#     ):
#         del vote_mask
#         sem_loss = self.regular_ce_loss(sem_pred, sem_mask, weight_mask)
#         vote_loss = self.unsophisticated_loss(vote_pred, vote_bool_tsr, weight_mask)
#         loss = self.w_vote * vote_loss + self.w_sem * sem_loss
#         return loss, sem_loss, vote_loss

#     @staticmethod
#     def regular_ce_loss(pred, lbls, weight_mask):
#         return MaskedPanLoss._single_loss(pred, lbls, weight_mask)

#     @staticmethod
#     def booltsr_loss(pred, bool_tsr, weight_mask):
#         raise ValueError('cannot be back-propagated')
#         is_valid = bool_tsr.any(dim=1)  # [N, H, W]
#         weight_mask = weight_mask[is_valid]  # [num_valid, ]
#         # pred = pred.permute(0, 2, 3, 1)
#         # bool_tsr = bool_tsr.permute(0, 2, 3, 1)
#         # pred, bool_tsr = pred[is_valid], bool_tsr[is_valid]  # [num_valid, C]

#         bottom = torch.logsumexp(pred, dim=1)
#         pred = torch.where(bool_tsr, pred, torch.tensor(float('-inf')).cuda())
#         pred = torch.logsumexp(pred, dim=1)
#         loss = (bottom - pred)[is_valid]  # -1 is implicit here by reversing order
#         loss = (loss * weight_mask).sum() / weight_mask.sum()
#         return loss

#     @staticmethod
#     def unsophisticated_loss(pred, bool_tsr, weight_mask):
#         is_valid = bool_tsr.any(dim=1)  # [N, H, W]
#         weight_mask = weight_mask[is_valid]

#         pred = F.softmax(pred, dim=1)
#         pred = torch.where(bool_tsr, pred, torch.tensor(0.).cuda())
#         loss = torch.log(pred.sum(dim=1)[is_valid])
#         loss = loss_normalize(loss, weight_mask, len(pred))
#         loss = -1 * loss
#         return loss
