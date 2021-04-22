import numpy as np
import torch
from panopticapi.utils import rgb2id

from fabric.io import save_object

ignore_index = -1


def convert_pan_m_to_sem_m(pan_mask, segments_info, catId_2_trainId):
    """Convert a panoptic mask to semantic segmentation mask
    pan_mask:        [H, W, 3] panoptic mask
    segments_info:   dict<id: segment_meta_dict>
    catId_2_trainId: dict<catId: trainId>
    """
    sem = np.zeros_like(pan_mask, np.int64)  # torch requires long tensor
    iid_to_catid = {
        el['id']: el['category_id'] for el in segments_info.values()
    }
    for iid in np.unique(pan_mask):
        if iid not in iid_to_catid:
            assert iid == 0
            sem[pan_mask == 0] = ignore_index
            continue
        cat_id = iid_to_catid[iid]
        train_id = catId_2_trainId[cat_id]
        sem[pan_mask == iid] = train_id
    return sem


def tensorize_2d_spatial_assignement(spatial, num_channels):
    """
    Args:
        spatial: [H, W] where -1 encodes invalid region
                Here we produce an extra, last channel to be set by inx -1,
                only to be throw out later. Neat
    Ret:
        np array of shape [1, C, H, W]
    """
    H, W = spatial.shape
    num_channels += 1  # add 1 extra channel to be turned on by inx -1
    tsr = np.zeros(shape=(num_channels, H, W))
    dim_0_inds, dim_1_inds = np.ix_(range(H), range(W))
    tsr[spatial, dim_0_inds, dim_1_inds] = 1
    tsr = tsr[:-1, :, :]  # throw out the extra dimension for -1s
    tsr = tsr[np.newaxis, ...]
    return tsr


class GtProducer():
    def __init__(self, pcv, mask_shape, params):
        self.pcv = pcv
        self.mask_shape = mask_shape
        self.params = params
        self.interpretable_as_prob_tsr = False

    def process(self, isthing, iscrowd, segment_mask, segment_area, ins_offset):
        raise NotImplementedError()

    def produce():
        raise NotImplementedError()

    def convert_to_prob_tsr():
        raise NotImplementedError()


class WeightMask(GtProducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = np.zeros(self.mask_shape, dtype=np.float32)
        self.power = self.params.get('power', 0.5)

    def process(self, isthing, iscrowd, segment_mask, segment_area, ins_offset):
        w = 1 / (segment_area ** self.power)
        if isthing and not iscrowd:
            self.weight[segment_mask] = w
        elif not isthing:
            self.weight[segment_mask] = w
        elif iscrowd:
            pass
        else:
            raise ValueError('unreachable')

    def produce(self):
        return self.weight


class _WeightMask(GtProducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = np.zeros(self.mask_shape, dtype=np.float32)
        self.power = self.params.get('power', 0.5)

    def process(self, isthing, iscrowd, segment_mask, segment_area, ins_offset):
        w = 1 / (segment_area ** self.power)
        if isthing and not iscrowd:
            lbls = self.pcv.discrete_vote_inx_from_offset(ins_offset)
            # pixs within an instance may be ignored if they sit on grid boundaries
            is_valid = lbls > ignore_index
            valid_area = is_valid.sum()
            # assert valid_area > 0, 'segment area {} vs valid area {}, lbls are {}'.format(
            #     segment_area, valid_area, lbls
            # )
            if valid_area == 0:
                w = 0
            else:
                w = 1 / (valid_area ** self.power)
            self.weight[segment_mask] = w * is_valid
        elif not isthing:
            self.weight[segment_mask] = w
        elif iscrowd:
            pass
        else:
            raise ValueError('unreachable')

    def produce(self):
        return self.weight


class ThingWeightMask(GtProducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = np.zeros(self.mask_shape, dtype=np.float32)
        self.power = self.params.get('power', 0.5)

    def process(self, isthing, iscrowd, segment_mask, segment_area, ins_offset):
        w = 1 / (segment_area ** self.power)
        if isthing and not iscrowd:
            lbls = self.pcv.discrete_vote_inx_from_offset(ins_offset)
            # pixs within an instance may be ignored if they sit on grid boundaries
            is_valid = lbls > ignore_index
            valid_area = is_valid.sum()
            # assert valid_area > 0, 'segment area {} vs valid area {}, lbls are {}'.format(
            #     segment_area, valid_area, lbls
            # )
            if valid_area == 0:
                w = 0
            else:
                w = 1 / (valid_area ** self.power)
            self.weight[segment_mask] = w * is_valid
        elif not isthing:
            self.weight[segment_mask] = 0
        elif iscrowd:
            pass
        else:
            raise ValueError('unreachable')

    def produce(self):
        return self.weight


class NoAbstainWeightMask(GtProducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = np.zeros(self.mask_shape, dtype=np.float32)
        self.power = self.params.get('power', 0.5)

    def process(self, isthing, iscrowd, segment_mask, segment_area, ins_offset):
        w = 1 / (segment_area ** self.power)
        if isthing and not iscrowd:
            self.weight[segment_mask] = w
        elif not isthing:  # only difference compared to WeightMask
            pass
        elif iscrowd:
            pass
        else:
            raise ValueError('unreachable')

    def produce(self):
        return self.weight


class InsWeightMask(GtProducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = np.zeros(self.mask_shape, dtype=np.float32)
        self.stuff_mask = np.zeros(self.mask_shape, dtype=bool)

    def process(self, isthing, iscrowd, segment_mask, segment_area, ins_offset):
        if isthing and not iscrowd:
            self.weight[segment_mask] = 1 / segment_area
        elif not isthing:
            self.stuff_mask[segment_mask] = True
        elif iscrowd:
            pass
        else:
            raise ValueError('unreachable')

    def produce(self):
        stuff_area = self.stuff_mask.sum()
        stuff_w = 20 / stuff_area
        self.weight[self.stuff_mask] = stuff_w
        self.weight = self.weight / 40
        return self.weight


class PCV_vote_no_abstain_mask(GtProducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interpretable_as_prob_tsr = True
        self.gt_mask = ignore_index * np.ones(self.mask_shape, np.int32)
        self.votable_mask = np.zeros(self.mask_shape, dtype=bool)
        self.offset_mask = np.empty(
            list(self.mask_shape) + [2], dtype=np.int32
        )
        pcv = self.pcv
        assert (pcv.num_votes - pcv.num_bins) == 1
        self.ABSTAIN_INX = ignore_index

    def process(self, isthing, iscrowd, segment_mask, segment_area, ins_offset):
        if isthing and not iscrowd:
            self.votable_mask[segment_mask] = True
            self.offset_mask[segment_mask] = ins_offset
        elif not isthing:
            self.gt_mask[segment_mask] = self.ABSTAIN_INX
        elif iscrowd:
            pass
        else:
            raise ValueError('unreachable')

    def produce(self):
        votable_mask = self.votable_mask
        self.gt_mask[votable_mask] = self.pcv.discrete_vote_inx_from_offset(
            self.offset_mask[votable_mask]
        )
        self.gt_mask = self.gt_mask.astype(np.int64)
        return self.gt_mask

    def convert_to_prob_tsr(self):
        return tensorize_2d_spatial_assignement(self.gt_mask, self.pcv.num_votes)


class PCV_vote_mask(GtProducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interpretable_as_prob_tsr = True
        self.gt_mask = ignore_index * np.ones(self.mask_shape, np.int32)
        self.votable_mask = np.zeros(self.mask_shape, dtype=bool)
        self.offset_mask = np.empty(
            list(self.mask_shape) + [2], dtype=np.int32
        )
        pcv = self.pcv
        assert (pcv.num_votes - pcv.num_bins) == 1
        self.ABSTAIN_INX = pcv.num_bins

    def process(self, isthing, iscrowd, segment_mask, segment_area, ins_offset):
        if isthing and not iscrowd:
            self.votable_mask[segment_mask] = True
            self.offset_mask[segment_mask] = ins_offset
        elif not isthing:
            self.gt_mask[segment_mask] = self.ABSTAIN_INX
        elif iscrowd:
            pass
        else:
            raise ValueError('unreachable')

    def produce(self):
        votable_mask = self.votable_mask
        self.gt_mask[votable_mask] = self.pcv.discrete_vote_inx_from_offset(
            self.offset_mask[votable_mask]
        )
        self.gt_mask = self.gt_mask.astype(np.int64)
        return self.gt_mask

    def convert_to_prob_tsr(self):
        return tensorize_2d_spatial_assignement(self.gt_mask, self.pcv.num_votes)


class PCV_igc_tsr(GtProducer):
    '''this code is problematic, but at least it is well exposed now'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interpretable_as_prob_tsr = True
        self.tsr = np.zeros(
            list(self.mask_shape) + [self.pcv.num_votes], dtype=np.bool
        )

    def process(self, isthing, iscrowd, segment_mask, segment_area, ins_offset):
        if isthing and not iscrowd:
            self.tsr[segment_mask] = self.pcv.tensorized_vote_from_offset(
                ins_offset
            )
        elif not isthing:
            self.tsr[segment_mask, -1] = True  # abstaining
        elif iscrowd:
            pass
        else:
            raise ValueError('unreachable')

    def produce(self):
        tsr = torch.as_tensor(
            self.tsr.transpose(2, 0, 1)  # [C, H, W]
        ).contiguous()
        self.tsr = tsr
        return tsr

    def convert_to_prob_tsr(self):
        tsr = self.tsr
        vote_tsr = tsr.clone().float()
        base = vote_tsr.sum(dim=0, keepdim=True)
        base[base == 0] = 1  # any number
        vote_tsr = (vote_tsr / base).unsqueeze(dim=0).numpy()
        return vote_tsr


class PCV_smooth_tsr(GtProducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interpretable_as_prob_tsr = True
        self.votable_mask = np.zeros(self.mask_shape, dtype=bool)
        self.abstain_mask = np.zeros(self.mask_shape, dtype=bool)
        self.offset_mask = np.empty(
            list(self.mask_shape) + [2], dtype=np.int32
        )
        self.tsr = np.zeros(
            list(self.mask_shape) + [self.pcv.num_votes], dtype=np.float32
        )
        pcv = self.pcv
        assert (pcv.num_votes - pcv.num_bins) == 1

    def process(self, isthing, iscrowd, segment_mask, segment_area, ins_offset):
        if isthing and not iscrowd:
            self.votable_mask[segment_mask] = True
            self.offset_mask[segment_mask] = ins_offset
        elif not isthing:
            self.abstain_mask[segment_mask] = True
        elif iscrowd:
            pass
        else:
            raise ValueError('unreachable')

    def produce(self):
        self.tsr[self.abstain_mask, -1] = 1.0
        self.tsr[self.votable_mask] = self.pcv.smooth_prob_tsr_from_offset(
            self.offset_mask[self.votable_mask]
        )
        self.tsr = torch.as_tensor(
            self.tsr.transpose(2, 0, 1)  # [C, H, W]
        ).contiguous()
        return self.tsr

    def convert_to_prob_tsr(self):
        tsr = self.tsr
        vote_tsr = tsr.clone().unsqueeze(dim=0).numpy()
        return vote_tsr


_REGISTRY = {
    'weight_mask': WeightMask,
    'thing_weight_mask': ThingWeightMask,
    'vote_mask': PCV_vote_mask,
    'vote_no_abstain_mask': PCV_vote_no_abstain_mask,
    'igc_tsr': PCV_igc_tsr,
    'smth_tsr': PCV_smooth_tsr,
    'ins_weight_mask': InsWeightMask,
    'no_abstain_weight_mask': NoAbstainWeightMask
}


class GTGenerator():
    def __init__(self, meta, pcv, pan_mask, segments_info, producer_cfgs=()):
        assert len(meta['catId_2_trainId']) == len(meta['cats'])
        self.pcv = pcv
        self.meta = meta
        self.pan_mask = rgb2id(np.array(pan_mask))
        self.segments_info = segments_info

        am_I_gt_producer = [
            int(cfg['params'].get('is_vote_gt', False)) for cfg in producer_cfgs
        ]
        if sum(am_I_gt_producer) != 1:
            raise ValueError('exactly 1 producer should be in charge of vote gt')
        self.gt_producer_inx = np.argmax(am_I_gt_producer)
        self.producer_cfgs = producer_cfgs
        self.producers = None

        self.ins_centroids = None
        self._sem_gt = None

    @property
    def sem_gt(self):
        if self._sem_gt is None:
            self._sem_gt = convert_pan_m_to_sem_m(
                self.pan_mask, self.segments_info, self.meta['catId_2_trainId']
            )
        return self._sem_gt

    def generate_gt(self):
        """Return a new mask with each pixel containing a discrete numeric label
        indicating which spatial bin the pixel should vote for
        Args:
            mask: [H, W] array filled with segment id
            segments_info:
        """
        pcv = self.pcv
        mask = self.pan_mask
        segments_info = self.segments_info
        category_meta = self.meta['cats']

        self.producers = [
            _REGISTRY[cfg['name']](pcv, mask.shape, cfg['params'])
            for cfg in self.producer_cfgs
        ]  # initalize the producers with params

        # indices grid of shape [2, H, W], where first dim is y, x; swap them
        # [H, W, 2] where last channel is x, y
        spatial_inds = np.indices(
            mask.shape).transpose(1, 2, 0)[..., ::-1].astype(np.int32)
        ins_centroids = []

        for segment_id, info in segments_info.items():
            cat, iscrowd = info['category_id'], info['iscrowd']
            isthing = category_meta[cat]['isthing']
            segment_mask = (mask == segment_id)
            area = segment_mask.sum()
            if area == 0:
                # cropping or extreme resizing might cause segments to disappear
                continue

            ins_offset = None
            if isthing and not iscrowd:
                ins_center = pcv.centroid_from_ins_mask(segment_mask)
                # ins_center = [math.ceil(_x) for _x in ins_center]
                ins_offset = (ins_center - spatial_inds[segment_mask]).astype(np.int32)
                # ins_center = np.array(ins_center, dtype=np.int32)
                # ins_offset = ins_center - spatial_inds[segment_mask]
                # ERROR must investigate this!!!
                ins_centroids.append(ins_center)

            for actor in self.producers:
                actor.process(isthing, iscrowd, segment_mask, area, ins_offset)

            # try:
            #     for actor in self.producers:
            #         actor.process(isthing, iscrowd, segment_mask, area, ins_offset)
            # except:
            #     dump = {
            #         'pan_mask': self.pan_mask,
            #         'segment_id': segment_id,
            #         'segments_info': segments_info,
            #         'area': area,
            #         'ins_offset': ins_offset,
            #         'cat': cat,
            #         'crowd': iscrowd,
            #         'isthing': isthing,
            #     }
            #     save_object(dump, './troubleshoot.pkl')
            #     raise ValueError('terminate here')

        self.ins_centroids = np.array(ins_centroids).reshape(-1, 2)

        training_gts = [self.sem_gt, ]
        training_gts.extend(
            [ actor.produce() for actor in self.producers ]
        )
        return training_gts

    def collect_prob_tsr(self):
        assert self.producers is not None, 'must create gt first'
        gt_producer = self.producers[self.gt_producer_inx]
        assert gt_producer.interpretable_as_prob_tsr, \
            '{} should be interpretable as prob tsr'.format(gt_producer.__class__)

        sem_prob_tsr = tensorize_2d_spatial_assignement(
            self.sem_gt, len(self.meta['catId_2_trainId'])
        )
        vote_prob_tsr = gt_producer.convert_to_prob_tsr()
        return sem_prob_tsr, vote_prob_tsr
