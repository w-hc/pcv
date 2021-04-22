import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.morphology import watershed, local_minima, h_minima

# from ... import cfg
from ...box_and_mask import (
    get_xywh_bbox_from_binary_mask,
    get_xywh_bbox_from_binary_mask_center, shrink_bbox
)
from panopticapi.utils import IdGenerator

from fabric.utils.logging import setup_logging
logger = setup_logging(__file__)

from fabric.utils.timer import Timer
t = Timer()


def conn_comp_peak_search(vote_hmap, hmap_thresh):
    label_mask, num_instances = ndi.label(vote_hmap >= hmap_thresh)
    assert num_instances == len(np.unique(label_mask)) - 1  # 0 is background
    peak_bbox = []
    for i in range(1, num_instances + 1):
        bbox = get_xywh_bbox_from_binary_mask(label_mask == i)
        peak_bbox.append(bbox)
    peak_bbox = np.array(peak_bbox).reshape(-1, 4)
    assert peak_bbox.shape == (num_instances, 4)
    peaks = np.stack((
        peak_bbox[:, 0] + peak_bbox[:, 2] // 2,
        peak_bbox[:, 1] + peak_bbox[:, 3] // 2
    ), axis=1)
    assert peaks.shape == (num_instances, 2)
    return label_mask, peaks, peak_bbox


def watershed_peak_search(vote_hmap, hmap_thresh):
    label_mask = vote_hmap >= hmap_thresh
    vote_hmap = -1 * vote_hmap
    markers_bool = local_minima(vote_hmap, connectivity=1) * label_mask
    markers, num_instances = ndi.label(markers_bool)
    ws_mask = watershed(vote_hmap, markers=markers, mask=label_mask)
    assert num_instances == len(np.unique(ws_mask)) - 1  # 0 is background
    peak_bbox = []
    for i in range(1, num_instances + 1):
        bbox = get_xywh_bbox_from_binary_mask(ws_mask == i)
        peak_bbox.append(bbox)
    peak_bbox = np.array(peak_bbox).reshape(-1, 4)
    assert peak_bbox.shape == (num_instances, 4)
    peaks = np.stack((
        peak_bbox[:, 0] + peak_bbox[:, 2] // 2,
        peak_bbox[:, 1] + peak_bbox[:, 3] // 2
    ), axis=1)
    assert peaks.shape == (num_instances, 2)
    return ws_mask, peaks, peak_bbox


def h_watershed_peak_search(vote_hmap, hmap_thresh):
    label_mask = vote_hmap >= hmap_thresh
    vote_hmap = -1 * vote_hmap
    markers_bool = h_minima(vote_hmap, h=hmap_thresh) * label_mask
    markers, num_instances = ndi.label(markers_bool)
    ws_mask = watershed(vote_hmap, markers=markers, mask=label_mask)
    # 0 is background
    assert num_instances == len(np.unique(ws_mask)) - 1, \
        '{} vs {}'.format(num_instances, len(np.unique(ws_mask)) - 1)
    peak_bbox = []
    for i in range(1, num_instances + 1):
        bbox = get_xywh_bbox_from_binary_mask(ws_mask == i)
        peak_bbox.append(bbox)
    peak_bbox = np.array(peak_bbox).reshape(-1, 4)
    assert peak_bbox.shape == (num_instances, 4)
    peaks = np.stack((
        peak_bbox[:, 0] + peak_bbox[:, 2] // 2,
        peak_bbox[:, 1] + peak_bbox[:, 3] // 2
    ), axis=1)
    assert peaks.shape == (num_instances, 2)
    return ws_mask, peaks, peak_bbox


def maxima_only(vote_hmap, hmap_thresh):
    label_mask = vote_hmap >= hmap_thresh
    vote_hmap = -1 * vote_hmap
    markers_bool = local_minima(vote_hmap, connectivity=1) * label_mask
    markers, num_instances = ndi.label(markers_bool)
    # ws_mask = watershed(vote_hmap, markers=markers, mask=label_mask)
    ws_mask = markers
    assert num_instances == len(np.unique(ws_mask)) - 1  # 0 is background
    peak_bbox = []
    for i in range(1, num_instances + 1):
        bbox = get_xywh_bbox_from_binary_mask(ws_mask == i)
        peak_bbox.append(bbox)
    peak_bbox = np.array(peak_bbox).reshape(-1, 4)
    assert peak_bbox.shape == (num_instances, 4)
    peaks = np.stack((
        peak_bbox[:, 0] + peak_bbox[:, 2] // 2,
        peak_bbox[:, 1] + peak_bbox[:, 3] // 2
    ), axis=1)
    assert peaks.shape == (num_instances, 2)
    return ws_mask, peaks, peak_bbox


peak_search_methods = {
    'conn_comp': conn_comp_peak_search,
    'watershed': h_watershed_peak_search  # watershed_peak_search
}


def unroll_img_inds(base_hinds, base_winds, filter_h, filter_w=None):
    """A primitive to implement a variety of convolutional operators.
    It uses advanced indexing.
    """
    if filter_w is None:
        filter_w = filter_h
    outer_h, outer_w, inner_h, inner_w = np.ix_(
        base_hinds, base_winds, range(filter_h), range(filter_w)
    )
    return outer_h + inner_h, outer_w + inner_w


class ConvEqualOpe():
    """
    A convolutional equality operator; Neat
    """
    def __init__(self, src_mask, query_mask):
        '''
        src_mask: gpu tsr: [H, W, top_k]
        query_mask: gpu tsr [1, 1, w, h, 1]
        '''
        assert query_mask.shape[2] == query_mask.shape[3]
        size = query_mask.shape[2]
        assert size % 2 == 1
        pad_width = int((size - 1) / 2)
        self.pad_width = pad_width
        self.size = size
        self.src_mask = F.pad(
            src_mask, (0, 0, pad_width, pad_width, pad_width, pad_width),
            mode='constant', value=-1
        )
        self.query_mask = query_mask

        # for deterministic argmax
        self.decrementer = torch.arange(
            -1 * src_mask.shape[-1], 0
        ).type(torch.int16).cuda()

    def query(self, h_inds, w_inds):
        pad = self.pad_width
        inds1, inds2 = unroll_img_inds(h_inds, w_inds, self.size)
        inds1, inds2 = torch.as_tensor(inds1), torch.as_tensor(inds2)
        candidates = self.src_mask[inds1, inds2]
        c_dim = len(h_inds) * len(w_inds)
        h, w, topk = self.src_mask.shape
        mask = torch.zeros([c_dim, h, w, topk], dtype=bool, device=candidates.device)
        inds0 = np.arange(c_dim).reshape(len(h_inds), len(w_inds), 1, 1)
        mask[inds0, inds1, inds2] = (candidates == self.query_mask)
        mask = mask.any(axis=0)
        mask = mask[pad:-pad, pad:-pad]
        if not mask.is_cuda:
            mask = mask.cuda()
        hit_any = mask.any(axis=-1)
        mask = mask.type(torch.int16) * self.decrementer
        match_mask = mask.argmin(dim=-1).type(torch.int16)
        match_mask[~hit_any] = -1  # those that did not hit any are -1
        return match_mask  # 6271


class MaskFromVote():
    def __init__(
        self, infer_cfg, dset_meta, pcv, sem_pred, vote_pred
    ):
        """
        Args:
            sem_pred:  [1, num_class, H, W] torch gpu tsr
            vote_pred: [1, num_bins,  H, W] torch gpu tsr
        """
        # self.dset_meta = dset_meta  # rt change
        self.cfg = infer_cfg
        self.categories = dset_meta['cats']
        self.trainId_2_catId = dset_meta['trainId_2_catId']
        # _semIds = np.unique(self.sem_decision)
        thing_trainIds = torch.tensor([
            trainId for trainId, catId in self.trainId_2_catId.items()
            if bool(self.categories[catId]['isthing'])  # and trainId in _semIds
        ])
        self.thing_trainIds = thing_trainIds.cuda()  # this takes 0.13s!!

        self.hmap_thresh = self.cfg.hmap_thresh
        self.ballot_module = pcv.ballot_module

        self.sem_pred, self.sem_decision, self.vote_pred, self.vote_decision = \
            self.digest_sem_and_vote_pred(sem_pred, vote_pred, pcv.num_bins)
        del sem_pred, vote_pred

        q_mask = pcv.query_mask.astype(np.int16)
        self.query_mask = torch.as_tensor(q_mask[None, None, ..., None]).cuda()


        # from fabric.utils.timer import global_timer
        # global_timer.vote.tic()
        self.vote_hmap = self.pixel_consensus_voting(self.vote_pred)
        # global_timer.vote.toc()
        # convert vote_pred to a benign form

        self.vote_pred_original = self.vote_pred.clone()
        self.vote_pred = self.vote_pred.cpu().numpy().squeeze(axis=0).transpose(1, 2, 0)

    def digest_sem_and_vote_pred(self, sem_pred, vote_pred, num_bins):
        # sem pred
        sem_pred = sem_pred.squeeze(axis=0).permute(1, 2, 0)
        sem_decision = sem_pred.argmax(axis=-1)
        # for oracle inference, ignore pixels must be ignored.
        _zero_mask = (sem_pred.sum(axis=-1) == 0)
        sem_decision[_zero_mask] = -1

        # store voting results
        vote_pred = vote_pred.permute(0, 2, 3, 1)
        assert vote_pred.shape[-1] == num_bins + 1
        # erase all votes from abstain locations
        vote_decision = vote_pred.argmax(dim=-1)
        if getattr(self.cfg, 'no_abstain', False):
            abstain_mask = np.isin(sem_decision.cpu().numpy(), self.thing_trainIds.tolist())
            abstain_mask = ~torch.from_numpy(abstain_mask).to(device=sem_pred.device).unsqueeze(0)
        else:
            abstain_mask = (vote_decision == num_bins)
        if self.cfg.remove_bg_votes:
            vote_pred[abstain_mask] = 0
        # remove abstain channel;
        vote_pred = vote_pred[..., :num_bins]
        # compute accurate vote_decision map;
        # where there is zero vote across grids, mark -
        topk = self.cfg.topk_match
        _, vote_decision = vote_pred.topk(k=topk, dim=-1, sorted=True)  # [1, H, W, K]
        # vote_decision = vote_pred.argmax(dim=-1)
        vote_decision[abstain_mask] = -1  # those who abstained do not participate
        vote_decision = vote_decision.squeeze(axis=0).type(torch.int16)  # [H, W, K]
        # note still a GPU tsr, will put to CPU later after voting
        vote_pred = vote_pred.permute(0, 3, 1, 2)
        # vote_pred[:, :49, :, :] = 0
        # print('sum of first 49 channels: {}'.format(vote_pred[:, :49, :, :].sum()))
        return sem_pred, sem_decision, vote_pred, vote_decision

    # WARN THIS IS UNSAFE MUST MAKE IT STATIC
    def pixel_consensus_voting(self, vote_tsr):
        vote_hmap = self.ballot_module(vote_tsr).cpu().data.numpy().squeeze()
        return vote_hmap

    def infer_panoptic_mask(
        self, instance_mask_only=False
    ):
        segments = []
        meta = {
            'image_id': None,  # not needed for now
            'file_name': None,
            'segments_info': segments
        }
        id_gen = IdGenerator(self.categories)

        instance_tsr, _, sem_cats = self.get_instance_tsr(resolve_overlap=True)
        assert instance_tsr.sum(0).max() <= 1, 'contested pixs should not exist'
        sem_cats = sem_cats.tolist()
        sem_cats = [ self.trainId_2_catId[el] for el in sem_cats ]
        iids = []
        for _cat, _ins_mask in zip(sem_cats, instance_tsr):
            _id = id_gen.get_id(_cat)
            iids.append(_id)
            segments.append({
                'id': _id,
                'category_id': _cat,
                'isthing': 1,
                'bbox': [int(elem) for elem in get_xywh_bbox_from_binary_mask(_ins_mask)],
                'area': int(_ins_mask.sum())
            })  # note numpy.int64 is not json serializable
        mask = (instance_tsr * np.array(iids).reshape(-1, 1, 1)).sum(axis=0)
        mask = mask.astype(np.uint32)
        if instance_mask_only:
            return mask, meta
        sem_remain = self.sem_decision.copy()
        sem_remain[mask > 0] = -1
        for trainId in np.unique(sem_remain):
            if trainId < 0:
                continue
            _cat = self.trainId_2_catId[trainId]
            if self.categories[_cat]['isthing']:
                continue  # abstain on ungrouped instance pixels
            _id = id_gen.get_id(_cat)
            segments.append({
                'id': _id,
                'category_id': _cat,
                'isthing': 0
            })
            mask[sem_remain == trainId] = _id

        for seg in meta['segments_info']:
            assert seg['id'] in mask
        return mask, meta

    def get_instance_tsr(self, resolve_overlap=False):
        _, loc_maxima, peak_bbox = \
            self.locate_peak_regions(self.vote_hmap, self.hmap_thresh)
        # from fabric.utils.timer import global_timer
        # global_timer.backproj.tic()
        used_inds, instance_tsr, sem_cats = self.peak_conv_mask_match(
            self.thing_trainIds, self.query_mask,
            self.vote_decision, self.sem_decision, peak_bbox
        )
        # global_timer.backproj.toc()
        # both components are equally slow. Sad
        self.sem_decision = self.sem_decision.cpu().numpy()
        used_inds = used_inds.cpu().numpy()
        instance_tsr = instance_tsr.cpu().numpy()
        sem_cats = sem_cats.cpu().numpy()

        loc_maxima, peak_bbox = loc_maxima[used_inds], peak_bbox[used_inds]
        if resolve_overlap:
            # sem resolve is dangerous for now
            instance_tsr = self.resolve_overlap_by_semantic_category(
                instance_tsr, sem_cats, self.sem_decision
            )
            instance_tsr, loc_maxima, sem_cats = self.filter_orphaned_layers(
                instance_tsr, loc_maxima, sem_cats
            )
            instance_tsr = self.resolve_overlap_by_dist_to_peaks(
                loc_maxima, instance_tsr
            )
            instance_tsr, loc_maxima, sem_cats = self.filter_orphaned_layers(
                instance_tsr, loc_maxima, sem_cats
            )
        return instance_tsr, loc_maxima, sem_cats

    def locate_peak_regions(self, vote_hmap, hmap_thresh):
        if self.cfg.peak_search_ope == 'sequential':
            return self.sequential_peak_search_ope(vote_hmap, hmap_thresh)
        handle_f = peak_search_methods[self.cfg.peak_search_ope]
        return handle_f(vote_hmap, hmap_thresh)

    def sequential_peak_search_ope(self, vote_hmap, hmap_thresh):
        # import pudb;pu.db
        tmp_vote_hmap = vote_hmap.copy()  # will be update at each timestep
        tmp_vote_pred = self.vote_pred_original.clone()

        peak_bbox = []

        label_mask, _ = ndi.label(tmp_vote_hmap >= 10000)  # all zero
        while (tmp_vote_hmap > hmap_thresh).any() and len(peak_bbox) < 100:
            # find global maxima
            max_loc = np.unravel_index(tmp_vote_hmap.argmax(), tmp_vote_hmap.shape)
            # set adaptive threosld
            # thresh = max(tmp_vote_hmap[max_loc] / 4, hmap_thresh)
            thresh = max(tmp_vote_hmap[max_loc] / 2, hmap_thresh)  # does this make the basin too shallow???
            # find connected component
            tmp_label_mask, _ = ndi.label(tmp_vote_hmap >= thresh)
            # find basin
            # get bbox around the connected component around the global maxima
            bbox = get_xywh_bbox_from_binary_mask(tmp_label_mask == tmp_label_mask[max_loc])
            # shrink the bbox
            shrink_radius = self.cfg.peak_bbox_shrink_radius
            if shrink_radius > 0:
                bbox = tuple(shrink_bbox(np.array([list(bbox)]), radius=shrink_radius)[0])
            peak_bbox.append(bbox)
            # get mask from the bbox
            _, mask, __ = self.peak_conv_mask_match(
                self.thing_trainIds, self.query_mask,
                self.vote_decision[..., :1], self.sem_decision, np.array(peak_bbox[-1:]).reshape(-1, 4)
            )  # only argmax matching here
            # can't find mask OR current mask already labeled
            if len(mask) == 0 or (label_mask[mask[0].cpu().numpy()] > 0).all():
                peak_bbox = peak_bbox[:-1]
                break
            mask = mask[0]  # size: HxW
            # Set label_mask
            label_mask[np.bitwise_and(mask.cpu().numpy(), label_mask == 0)] = len(peak_bbox)
            # relase votes
            tmp_vote_pred[mask.unsqueeze(0).unsqueeze(0).expand_as(tmp_vote_pred)] = 0
            tmp_vote_hmap = self.pixel_consensus_voting(tmp_vote_pred)
        peak_bbox = np.array(peak_bbox).reshape(-1, 4)
        peaks = np.stack((
            peak_bbox[:, 0] + peak_bbox[:, 2] // 2,
            peak_bbox[:, 1] + peak_bbox[:, 3] // 2
        ), axis=1)
        return label_mask, peaks, peak_bbox

    @staticmethod
    def ___peak_conv_mask_match(
        thing_trainIds, query_mask, vote_decision, sem_decision, peak_bbox
    ):
        """
        Ret:
            [num_instances, H, W] array where each channel encodes a mask
            Note that there can be cross channel conflicts
        """
        shrink_radius = self.cfg.peak_bbox_shrink_radius
        if shrink_radius > 0:
            peak_bbox = shrink_bbox(peak_bbox, radius=shrink_radius)
        conv_equal_mask_querier = ConvEqualOpe(vote_decision, query_mask)

        temp = []
        for i, bbox in enumerate(peak_bbox):
            x, y, w, h = bbox  # note the order, y first then x
            _m = conv_equal_mask_querier.query(range(y, y + h), range(x, x + w))
            temp.append(_m)
        if len(temp) == 0:
            ins_tsr = []
        else:
            temp = torch.stack(temp)
            MAX = torch.iinfo(temp.dtype).max  # np.iinfo(temp.dtype).max @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  atrocious hack
            temp[temp == -1] = MAX
            first_hit = temp.min(axis=0)[0]
            first_hit[first_hit == MAX] = -1
            ins_tsr = (temp == first_hit[None, ...])

        masks = []
        sem_cats = []
        realized_inds = []

        for i, _m, in enumerate(ins_tsr):
            if not _m.any():
                logger.warn("peak bbox matches no pixels")
                continue
            semIds, counts = torch.unique(
                sem_decision[_m], return_counts=True)
            is_thing = \
                torch.tensor([id in thing_trainIds for id in semIds], dtype=bool)
            if not is_thing.any():
                logger.warn("instance pixels matched are all predicted as stuff")
                continue
            if getattr(self.cfg, 'drop_uncertain', False): # does not help so much it seems
                if torch.max(counts[is_thing]).float() / torch.sum(counts) < 0.5:
                    print(torch.max(counts[is_thing]).float() / torch.sum(counts))
                    logger.warn("Semantic consensus too low, drop this")
                    continue
            semIds, counts = semIds[is_thing], counts[is_thing]
            sem_cats.append(semIds[torch.argmax(counts)])
            masks.append(_m)
            realized_inds.append(i)

        if len(masks) > 0:
            masks = torch.stack(masks)
        else:
            masks = torch.empty(0, *sem_decision.shape).cuda()
        realized_inds, sem_cats = \
            torch.tensor(realized_inds, dtype=int), \
            torch.tensor(sem_cats, dtype=int, device=vote_decision.device)
        return realized_inds, masks, sem_cats

    def peak_conv_mask_match(
        self, thing_trainIds, query_mask, vote_decision, sem_decision, peak_bbox
    ):
        """
        Ret:
            [num_instances, H, W] array where each channel encodes a mask
            Note that there can be cross channel conflicts
        """
        shrink_radius = self.cfg.peak_bbox_shrink_radius
        if shrink_radius > 0:
            peak_bbox = shrink_bbox(peak_bbox, radius=shrink_radius)
        conv_equal_mask_querier = ConvEqualOpe(vote_decision, query_mask)

        temp = []
        for i, bbox in enumerate(peak_bbox):
            x, y, w, h = bbox  # note the order, y first then x
            _m = conv_equal_mask_querier.query(range(y, y + h), range(x, x + w))
            temp.append(_m)
        if len(temp) == 0:
            ins_tsr = []
        else:
            temp = torch.stack(temp)
            MAX = torch.iinfo(temp.dtype).max  # np.iinfo(temp.dtype).max @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  atrocious hack
            temp[temp == -1] = MAX
            first_hit = temp.min(axis=0)[0]
            first_hit[first_hit == MAX] = -1
            ins_tsr = (temp == first_hit[None, ...])

        masks = []
        sem_cats = []
        realized_inds = []

        sem_pred = self.sem_pred
        for i, _m, in enumerate(ins_tsr):
            if not _m.any():
                logger.warn("peak bbox matches no pixels")
                continue
            _sem_aggregate = sem_pred[_m].mean(0)
            is_thing = \
                torch.tensor([id in thing_trainIds for id in range(len(_sem_aggregate))], dtype=bool)
            if getattr(self.cfg, 'drop_uncertain', False):  # does not help so much it seems
                if torch.max(_sem_aggregate[is_thing]) < 0.5:
                    print(torch.max(_sem_aggregate[is_thing]))
                    logger.warn("Max sem pred too low, drop this")
                    continue
            _sem_aggregate[~is_thing] = 0
            sem_cats.append(torch.argmax(_sem_aggregate))
            masks.append(_m)
            realized_inds.append(i)

        if len(masks) > 0:
            masks = torch.stack(masks)
        else:
            masks = torch.empty(0, *sem_decision.shape).cuda()
        realized_inds, sem_cats = \
            torch.tensor(realized_inds, dtype=int), \
            torch.tensor(sem_cats, dtype=int, device=vote_decision.device)
        return realized_inds, masks, sem_cats

    @staticmethod
    def resolve_overlap_by_semantic_category(
        instance_tsr, ins_sem_array, sem_decision
    ):
        """
        this logic is subtle: for a pixel x contested by multiple ins,
        x abandons all ins whose sem cats are different from sem_pred(x)
        """
        dup_mask = instance_tsr.sum(axis=0) > 1
        if not dup_mask.any():
            return instance_tsr

        pixel_sem = sem_decision[dup_mask]
        pixel_sem_agree = pixel_sem.reshape(1, -1) == ins_sem_array.reshape(-1, 1)
        instance_tsr[:, dup_mask] *= pixel_sem_agree
        return instance_tsr

    @staticmethod
    def resolve_overlap_by_dist_to_peaks(loc_maxima, instance_tsr):
        """
        Args: assume that N instances are detected
            loc_maxima: [N, 2] with each row x, y coordinates
            instance_tsr: [N, H, W]
        Ret:
            instance_tsr: [N, H, W] where overlaps are resolved by assigning
            pixels to instances whose voting peaks are the closest
        """
        dup_mask = instance_tsr.sum(axis=0) > 1
        if not dup_mask.any():
            return instance_tsr
        coord_grid = np.indices(
            instance_tsr.shape[1:]).transpose(1, 2, 0)[..., ::-1]
        # [N, 1, 1, 2] - [1, H, W, 2] -> [N, H, W, 2]
        dist_tsr = loc_maxima[:, None, None, :] - coord_grid[None, ...]
        dist_tsr = np.linalg.norm(dist_tsr, ord=2, axis=-1)  # [N, H, W]
        dup_dist_2_cen = dist_tsr[:, dup_mask] * instance_tsr[:, dup_mask]
        dup_dist_2_cen[instance_tsr[:, dup_mask] == 0] = 'nan'
        dup_loyalty = np.nanargmin(dup_dist_2_cen, axis=0)
        instance_tsr[:, dup_mask] = 0
        inds1, inds2 = np.where(dup_mask > 0)
        instance_tsr[dup_loyalty, inds1, inds2] = 1
        return instance_tsr

    @staticmethod
    def filter_orphaned_layers(instance_tsr, local_maxima, ins_sem_array):
        """A mask layer is orphaned if it is empty, maybe as a result of
        overlap resolution that reassignes all itx pixels to some other
        instances. Throw away such layers
        Args: given N instances
            ins_sem_array: [N, ]
            instance_tsr: [N, H, W]
            local_maxima: [N, 1]
        """
        size_across_layers = instance_tsr.sum(1).sum(1)  # [N, ]
        zero_mask = (size_across_layers == 0)
        if zero_mask.sum() == 0:
            return instance_tsr, local_maxima, ins_sem_array

        logger.warn(
            "losing {}/{} detected instances".format(zero_mask.sum(), len(ins_sem_array))
        )
        instance_tsr = instance_tsr[~zero_mask, ...]
        local_maxima = local_maxima[~zero_mask, ...]
        ins_sem_array = ins_sem_array[~zero_mask]
        return instance_tsr, local_maxima, ins_sem_array


class MFV_CatSeparate(MaskFromVote):
    def pixel_consensus_voting(self, vote_pred):
        """
        Separate out vote_pred base on semantic decision into layers
        """
        # [1, H, W] - [num_things, 1, 1] -> [num_things, H, W]
        sem_tsr = self.sem_decision[None, ...] \
            == self.thing_trainIds.reshape(-1, 1, 1)
        vote_pred = self.vote_pred  # [1, num_bins, H, W]
        vote_pred_acc = []
        for sem_mask in sem_tsr:
            _slot = torch.zeros_like(vote_pred)
            _slot[..., sem_mask] = vote_pred[..., sem_mask]
            vote_pred_acc.append(_slot)
        # [1, num_bins, num_things, H, W]
        vote_pred_acc = torch.stack(vote_pred_acc, axis=2)
        # this assertion would fail in regular models since non-thing pixels
        # often still have votes. There will not be an exact match
        # assert np.allclose(vote_pred_acc.sum(axis=2), vote_pred)
        _, _, _, H, W = vote_pred_acc.shape
        vote_pred_acc = vote_pred_acc.reshape(1, -1, H, W)
        vote_pred = torch.as_tensor(vote_pred_acc).cuda()
        vote_hmap = self.ballot_module(vote_pred).cpu().data.numpy().squeeze()
        return vote_hmap

    def get_instance_tsr(self, resolve_overlap=False):
        ins_tsr_acc = []
        sem_cats_acc = []
        loc_maxima_acc = []
        for trainId, cur_cat_hmap in zip(self.thing_trainIds, self.vote_hmap):
            _, loc_maxima, peak_bbox = \
                self.locate_peak_regions(cur_cat_hmap, self.hmap_thresh)
            cur_cat_vote_dec = self.vote_decision.clone()
            cur_cat_vote_dec[self.sem_decision != trainId] = -1
            used_inds, instance_tsr, sem_cats = self.peak_conv_mask_match(
                self.thing_trainIds, self.query_mask,
                cur_cat_vote_dec, self.sem_decision,
                peak_bbox
            )
            assert (sem_cats == trainId).all()

            used_inds = used_inds.cpu().numpy()
            instance_tsr = instance_tsr.cpu().numpy()
            sem_cats = sem_cats.cpu().numpy()

            ins_tsr_acc.append(instance_tsr)
            sem_cats_acc.append(sem_cats)
            loc_maxima_acc.append(loc_maxima[used_inds])
        instance_tsr = np.concatenate(ins_tsr_acc, axis=0)
        sem_cats = np.concatenate(sem_cats_acc, axis=0)
        loc_maxima = np.concatenate(loc_maxima_acc, axis=0)
        self.sem_decision = self.sem_decision.cpu().numpy()
        if resolve_overlap:
            instance_tsr = self.resolve_overlap_by_dist_to_peaks(
                loc_maxima, instance_tsr
            )
            instance_tsr, loc_maxima, sem_cats = self.filter_orphaned_layers(
                instance_tsr, loc_maxima, sem_cats
            )
        return instance_tsr, loc_maxima, sem_cats


# def _locate_peak_regions(self, vote_hmap, hmap_thresh):
#     label_mask = vote_hmap >= hmap_thresh
#     vote_hmap = -1 * vote_hmap
#     # import pudb;pu.db
#     # tmp = torch.nn.functional.max_pool2d(torch.from_numpy(vote_hmap).unsqueeze(0), 3, 1, 1)
#     # ttt = ((- tmp[0]) == torch.from_numpy(vote_hmap)).numpy()
#     markers_bool = local_minima(vote_hmap, connectivity=1) * label_mask
#     markers, num_instances = ndi.label(markers_bool)
#     if False: # 'tmp_gt_ins_centroids' in self.dset_meta:
#         gt_ins_centroids = self.dset_meta['tmp_gt_ins_centroids'].round().astype(np.int64)
#         # import pudb;pu.db
#         markers *= 0
#         num_instances = 0
#         for i in range(len(gt_ins_centroids)):
#             if markers[gt_ins_centroids[i][1], gt_ins_centroids[i][0]] == 0:
#                 num_instances += 1
#                 markers[gt_ins_centroids[i][1], gt_ins_centroids[i][0]] = num_instances
#         if len(gt_ins_centroids) > num_instances:
#             logger.warn("Duplicate centroids:", len(gt_ins_centroids) - num_instances)
#         del self.dset_meta['tmp_gt_ins_centroids']
#         label_mask.fill(1)
#     ws_mask = watershed(vote_hmap, markers=markers, mask=label_mask)
#     assert num_instances == len(np.unique(ws_mask[ws_mask > 0]))  # 0 is background
#     peak_bbox = []
#     # torch.save([markers, -vote_hmap, ws_mask], '/home-nfs/whc/tmp.pth')
#     for i in range(1, num_instances + 1):
#         bbox = get_xywh_bbox_from_binary_mask(ws_mask == i)
#         # bbox = get_xywh_bbox_from_binary_mask_center(ws_mask == i, np.where(markers == i))
#         peak_bbox.append(bbox)
#     peak_bbox = np.array(peak_bbox).reshape(-1, 4)
#     assert peak_bbox.shape == (num_instances, 4)
#     peaks = np.stack((
#         peak_bbox[:, 0] + peak_bbox[:, 2] // 2,
#         peak_bbox[:, 1] + peak_bbox[:, 3] // 2
#     ), axis=1)
#     # # Use markers as peaks
#     # peaks_ = np.stack(np.nonzero(markers)).T[:,::-1]
#     # peaks = peaks_
#     assert peaks.shape == (num_instances, 2), '{} vs {}'.format(peaks.shape, num_instances)
#     return ws_mask, peaks, peak_bbox
