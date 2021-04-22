import bisect
import os.path as osp
from collections import defaultdict
import json

import numpy as np
import scipy.linalg as LA
from scipy.ndimage import binary_dilation, generate_binary_structure
import pandas as pd
from PIL import Image

from tabulate import tabulate

from panopticapi.utils import rgb2id
from panoptic.pan_eval import PQStat, OFFSET, VOID
from panoptic.datasets.base import mapify_iterable

from fabric.io import load_object, save_object

_SEGMENT_UNMATCHED = 0
_SEGMENT_MATCHED = 1
_SEGMENT_FORGIVEN = 2


def generalized_aspect_ratio(binary_mask):
    xs, ys = np.where(binary_mask)
    coords = np.array([xs, ys]).T
    # mean center the coords
    coords = coords - coords.mean(axis=0)
    # cov matrix
    cov = coords.T @ coords
    first, second = LA.eigvalsh(cov)[::-1]
    ratio = (first ** 0.5) / (second + 1e-8) ** 0.5
    return ratio


class Annotation():
    '''
    Overall Schema for Each Side (Either Gt or Pred)
    e.g. pred:
        cats: {id: cat}
        imgs: {id: image}
        segs  {sid: seg}
        img2seg: {image_id: {sid: seg}}
        cat2seg: {cat_id: {sid: seg}}

    cat:
        id: 7,
        name: road,
        supercategory: 'flat',
        color: [128, 64, 128],
        isthing: 0

    image:
        id: 'frankfurt_000000_005898',
        file_name: frankfurt/frankfurt_000000_005898_leftImg8bit.png
        ann_fname: abcde
        width: 2048,
        height: 1024,
        ---
        mask: a cached mask that is loaded

    seg:
        sid (seg id): gt/frankfurt_000000_000294/8405120
        image_id:     frankfurt_000000_000294
        id:           8405120
        category_id:  7
        area:         624611
        bbox:         [6, 432, 1909, 547]
        iscrowd:      0
        match_state:  one of (UNMATCHED, MATCHED, IGNORED)
        matchings:     {sid: iou} sorted from high to low
        (breakdown_flag): this can be optinally introduced for breakdown analysis
    '''
    def __init__(self, json_meta_fname, is_gt, state_dict=None):
        '''
        if state_dict is provided, then just load it and avoid the computation
        '''
        dirname, fname = osp.split(json_meta_fname)
        self.root = dirname
        self.mask_root = osp.join(dirname, fname.split('.')[0])

        if state_dict is None:
            with open(json_meta_fname) as f:
                state_dict = self.process_raw_meta(json.load(f), is_gt)
        self.state_dict = state_dict
        self.register_state(state_dict)

    def register_state(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    @staticmethod
    def process_raw_meta(raw_meta, is_gt):
        state = dict()
        state['cats'] = mapify_iterable(raw_meta['categories'], 'id')
        state['imgs'] = mapify_iterable(raw_meta['images'], 'id')
        state['segs'] = dict()
        state['img2seg'] = defaultdict(dict)
        state['cat2seg'] = defaultdict(dict)

        sid_prefix = 'gt' if is_gt else 'pd'

        for ann in raw_meta['annotations']:
            image_id = ann['image_id']
            segments = ann['segments_info']
            state['imgs'][image_id]['ann_fname'] = ann['file_name']
            for seg in segments:
                cat_id = seg['category_id']

                unique_id = '{}/{}/{}'.format(sid_prefix, image_id, seg['id'])
                seg['sid'] = unique_id
                seg['image_id'] = image_id
                seg['match_state'] = _SEGMENT_FORGIVEN
                seg['matchings'] = dict()

                state['segs'][unique_id] = seg
                state['img2seg'][image_id][unique_id] = seg
                state['cat2seg'][cat_id][unique_id] = seg
        return state

    def seg_sort_matchings(self):
        """sort matchings from high to low IoU"""
        for _, seg in self.segs.items():
            matchings = seg['matchings']
            seg['matchings'] = dict(
                sorted(matchings.items(), key=lambda x: x[1], reverse=True)
            )

    def match_summarize(self, breakdown_flag=None):
        '''
        ret: [num_cats, 4] where each row contains
                (iou_sum, num_matched, num_unmatched, total_inst)
        '''
        ret = []
        for cat in sorted(self.cats.keys()):
            segs = self.cat2seg[cat].values()
            iou_sum, num_matched, num_unmatched, total_inst = 0.0, 0.0, 0.0, 0.0
            for seg in segs:
                if breakdown_flag is not None and seg['breakdown_flag'] != breakdown_flag:
                    continue  # if breakdown is activated, only summarize those required
                total_inst += 1
                if seg['match_state'] == _SEGMENT_MATCHED:
                    iou_sum += list(seg['matchings'].values())[0]
                    num_matched += 1
                elif seg['match_state'] == _SEGMENT_UNMATCHED:
                    num_unmatched += 1
            ret.append([iou_sum, num_matched, num_unmatched, total_inst])
        ret = np.array(ret)
        return ret

    def catId_given_catName(self, catName):
        for catId, cat in self.cats.items():
            if cat['name'] == catName:
                return catId
        raise ValueError('what kind of category is this? {}'.format(catName))

    def get_mask_given_seg(self, seg):
        return self.get_mask_given_imgid(self, seg['image_id'])

    def get_img_given_imgid(self, image_id, img_root):
        img = self.imgs[image_id]
        img_fname = img['file_name']
        img_fname = osp.join(img_root, img_fname)
        img = Image.open(img_fname)
        return img

    def get_mask_given_imgid(self, image_id, store_in_cache=True):
        img = self.imgs[image_id]
        _MASK_KEYNAME = 'mask'
        cache_entry = img.get(_MASK_KEYNAME, None)
        if cache_entry is not None:
            assert isinstance(cache_entry, np.ndarray)
            return cache_entry
        else:
            mask_fname = img['ann_fname']
            mask = np.array(
                Image.open(osp.join(self.mask_root, mask_fname)),
                dtype=np.uint32
            )
            mask = rgb2id(mask)
            if store_in_cache:
                img[_MASK_KEYNAME] = mask
            return mask

    def compute_seg_shape_oddity(self):
        print('start computing shape oddity')
        i = 0
        for imgId, segs in self.img2seg.items():
            i += 1
            if (i % 50) == 0:
                print(i)
            mask = self.get_mask_given_imgid(imgId, store_in_cache=False)
            for _, s in segs.items():
                seg_id = s['id']
                binary_mask = (mask == seg_id)
                s['gen_aspect_ratio'] = generalized_aspect_ratio(binary_mask)

    def compute_seg_boundary_stats(self):
        print('start computing boundary stats')
        i = 0
        for imgId, segs in self.img2seg.items():
            i += 1
            if (i % 50) == 0:
                print(i)
            mask = self.get_mask_given_imgid(imgId, store_in_cache=False)
            for _, s in segs.items():
                seg_id = s['id']
                binary_mask = (mask == seg_id)
                self._per_seg_neighbors_stats(s, binary_mask, mask)

    def _per_seg_neighbors_stats(self, seg_dict, binary_mask, mask):
        area = binary_mask.sum()
        # struct = generate_binary_structure(2, 2)
        dilated = binary_dilation(binary_mask, structure=None, iterations=1)
        boundary = dilated ^ binary_mask

        # stats
        length = boundary.sum()
        ratio = length ** 2 / area
        seg_dict['la_ratio'] = ratio

        # get the neighbors
        ids, cnts = np.unique(mask[boundary], return_counts=True)

        sid_prefix = '/'.join(
            seg_dict['sid'].split('/')[:2]  # throw away the last
        )
        sids = [ '{}/{}'.format(sid_prefix, id) for id in ids ]

        thing_neighbors = {
            sid: cnt for sid, id, cnt in zip(sids, ids, cnts)
            if id > 0 and self.cats[self.segs[sid]['category_id']]['isthing']
        }
        seg_dict['thing_neighbors'] = thing_neighbors


class PanopticEvalAnalyzer():
    def __init__(self, gt_json_meta_fname, pd_json_meta_fname, load_state=True):
        # use the pd folder as root directory since a single gt ann can correspond
        # to many pd anns.
        root = osp.split(pd_json_meta_fname)[0]
        self.state_dump_fname = osp.join(root, 'analyze_dump.pkl')

        is_evaluated = False
        if osp.isfile(self.state_dump_fname) and load_state:
            state = load_object(self.state_dump_fname)
            gt_state, pd_state = state['gt'], state['pd']
            is_evaluated = True
        else:
            gt_state, pd_state = None, None

        self.gt = Annotation(gt_json_meta_fname, is_gt=True, state_dict=gt_state)
        self.pd = Annotation(pd_json_meta_fname, is_gt=False, state_dict=pd_state)

        # validate that gt and pd json completely match
        assert self.gt.imgs.keys() == self.pd.imgs.keys()
        assert self.gt.cats == self.pd.cats
        self.imgIds = list(sorted(self.gt.imgs.keys()))

        if not is_evaluated:
            # evaluate and then save the state
            self._evaluate()
            self.gt.compute_seg_shape_oddity()
            self.pd.compute_seg_shape_oddity()
            self.dump_state()

    def _gt_boundary_stats(self):
        self.gt.compute_seg_boundary_stats()

    def dump_state(self):
        state = {
            'gt': self.gt.state_dict,
            'pd': self.pd.state_dict
        }
        save_object(state, self.state_dump_fname)

    def _evaluate(self):
        stats = PQStat()
        cats = self.gt.cats
        for i, imgId in enumerate(self.imgIds):
            if (i % 50) == 0:
                print("progress {} / {}".format(i, len(self.imgIds)))
            # if (i > 100):
            #     break

            gt_ann = {
                'image_id': imgId, 'segments_info': self.gt.img2seg[imgId].values()
            }
            gt_mask = np.array(
                Image.open(osp.join(
                    self.gt.mask_root, self.gt.imgs[imgId]['ann_fname']
                )),
                dtype=np.uint32
            )
            gt_mask = rgb2id(gt_mask)

            pd_ann = {
                'image_id': imgId, 'segments_info': self.pd.img2seg[imgId].values()
            }
            pd_mask = np.array(
                Image.open(osp.join(
                    self.pd.mask_root, self.pd.imgs[imgId]['ann_fname']
                )),
                dtype=np.uint32
            )
            pd_mask = rgb2id(pd_mask)

            _single_stat = self.pq_compute_single_img(
                cats, gt_ann, gt_mask, pd_ann, pd_mask
            )
            stats += _single_stat

        self.gt.seg_sort_matchings()
        self.pd.seg_sort_matchings()
        return stats

    def summarize(self, flag=None):
        per_cat_res, overall_table, cat_table = self._aggregate(
            gt_stats=self.gt.match_summarize(flag),
            pd_stats=self.pd.match_summarize(flag),
            cats=self.gt.cats
        )
        return per_cat_res, overall_table, cat_table

    @staticmethod
    def _aggregate(gt_stats, pd_stats, cats):
        '''
        Args:
            pd/gt_stats: [num_cats, 4] with each row contains
                (iou_sum, num_matched, num_unmatched, total_inst)
            cats: a dict of {catId: catMetaData}
        Returns:
            1. per cat pandas dataframe; easy to programmatically  manipulate
            2. str formatted overall result table
            3. str formatted per category result table
        '''
        # each is of shape [num_cats]
        gt_iou, gt_matched, gt_unmatched, gt_tot_inst = gt_stats.T
        pd_iou, pd_matched, pd_unmatched, pd_tot_inst = pd_stats.T
        assert np.allclose(gt_iou, pd_iou) and (gt_matched == pd_matched).all()

        catIds = list(sorted(cats.keys()))
        catNames = [cats[id]['name'] for id in catIds]
        isthing = np.array([cats[id]['isthing'] for id in catIds], dtype=bool)

        RQ = gt_matched / (gt_matched + 0.5 * gt_unmatched + 0.5 * pd_unmatched)
        SQ = gt_iou / gt_matched
        RQ, SQ = np.nan_to_num(RQ), np.nan_to_num(SQ)
        PQ = RQ * SQ
        results = np.array([PQ, SQ, RQ]) * 100  # [3, num_cats]

        overall_table = tabulate(
            headers=['', 'PQ', 'SQ', 'RQ', 'num_cats'],
            floatfmt=".2f", tablefmt='fancy_grid',
            tabular_data=[
                ['all'] + list(map(lambda x: x.mean(), results)) + [len(catIds)],
                ['things'] + list(map(lambda x: x[isthing].mean(), results)) + [sum(isthing)],
                ['stuff'] + list(map(lambda x: x[~isthing].mean(), results)) + [sum(1 - isthing)],
            ]
        )

        headers = (
            'PQ', 'SQ', 'RQ',
            'num_matched', 'gt_unmatched', 'pd_unmatched', 'tot_gt_inst',
            'isthing'
        )
        results = np.array(
            list(results) + [gt_matched, gt_unmatched, pd_unmatched, gt_tot_inst, isthing]
        )
        results = results.T
        data_frame = pd.DataFrame(results, columns=headers, index=catNames)
        cat_table = tabulate(
            data_frame, headers='keys', floatfmt=".2f", tablefmt='fancy_grid'
        )
        return data_frame, overall_table, cat_table

    @staticmethod
    def pq_compute_single_img(cats, gt_ann, gt_mask, pd_ann, pd_mask):
        """
        This is the original eval function refactored for readability
        """
        pq_stat = PQStat()
        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pd_segms = {el['id']: el for el in pd_ann['segments_info']}

        # predicted segments area calculation + prediction sanity checks
        pd_labels_set = set(el['id'] for el in pd_ann['segments_info'])
        labels, labels_cnt = np.unique(pd_mask, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pd_segms:
                if label == VOID:
                    continue
                raise KeyError(
                    ('In the image with ID {} '
                    'segment with ID {} is presented in PNG '
                    'and not presented in JSON.').format(gt_ann['image_id'], label)
                )
            pd_segms[label]['area'] = int(label_cnt)
            pd_labels_set.remove(label)
            if pd_segms[label]['category_id'] not in cats:
                raise KeyError(
                    ('In the image with ID {} '
                    'segment with ID {} has unknown '
                    'category_id {}.').format(
                        gt_ann['image_id'], label, pd_segms[label]['category_id'])
                )
        if len(pd_labels_set) != 0:
            raise KeyError(
                ('In the image with ID {} '
                'the following segment IDs {} are presented '
                'in JSON and not presented in PNG.').format(
                    gt_ann['image_id'], list(pd_labels_set))
            )

        # confusion matrix calculation
        gt_vs_pd = gt_mask.astype(np.uint64) * OFFSET + pd_mask.astype(np.uint64)
        gt_pd_itrsct = {}
        labels, labels_cnt = np.unique(gt_vs_pd, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id, pd_id = label // OFFSET, label % OFFSET
            gt_pd_itrsct[(gt_id, pd_id)] = intersection

        # count all matched pairs
        gt_matched, pd_matched = set(), set()
        for label_tuple, intersection in gt_pd_itrsct.items():
            gt_label, pd_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pd_label not in pd_segms:
                continue

            gt_seg, pd_seg = gt_segms[gt_label], pd_segms[pd_label]
            union = pd_seg['area'] + gt_seg['area'] \
                - intersection - gt_pd_itrsct.get((VOID, pd_label), 0)
            iou = intersection / union
            if iou > 0.1:
                gt_seg['matchings'][pd_seg['sid']] = iou
                pd_seg['matchings'][gt_seg['sid']] = iou

            if gt_seg['iscrowd'] == 1:
                continue
            if gt_seg['category_id'] != pd_seg['category_id']:
                continue

            if iou > 0.5:
                gt_cat_id = gt_seg['category_id']
                pq_stat[gt_cat_id].tp += 1
                pq_stat[gt_cat_id].iou += iou
                gt_matched.add(gt_label)
                pd_matched.add(pd_label)
                gt_seg['match_state'] = _SEGMENT_MATCHED
                pd_seg['match_state'] = _SEGMENT_MATCHED

        # count false negatives
        # HC: assumption each category in image can only have a single crowd segment!
        # each img each cat, all crowd segments are merged into 1 segment. well
        crowd_cat_segid = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored;
            if gt_info['iscrowd'] == 1:
                crowd_cat_segid[gt_info['category_id']] = gt_label
                continue
            pq_stat[gt_info['category_id']].fn += 1
            gt_info['match_state'] = _SEGMENT_UNMATCHED

        # count false positives
        for pd_label, pd_info in pd_segms.items():
            if pd_label in pd_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pd_itrsct.get((VOID, pd_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pd_info['category_id'] in crowd_cat_segid:
                intersection += gt_pd_itrsct.get(
                    (crowd_cat_segid[pd_info['category_id']], pd_label), 0
                )
            # predicted segment is ignored if more than half of the segment
            # correspond to VOID and CROWD regions
            if intersection / pd_info['area'] > 0.5:
                continue
            pq_stat[pd_info['category_id']].fp += 1
            pd_info['match_state'] = _SEGMENT_UNMATCHED
        return pq_stat


class BreakdownPolicy():
    def __init__(self):
        self.flags = []

    def breakdown(self, gt_segs, pd_segs):
        pass


class DummyBreakdown(BreakdownPolicy):
    def __init__(self):
        self.flags = ['sector1', 'sector2']

    def breakdown(self, gt_segs, pd_segs):
        import numpy.random as npr
        gt_flags = [ npr.choice(self.flags) for _ in range(len(gt_segs)) ]
        for i, seg in enumerate(gt_segs.values()):
            seg['breakdown_flag'] = gt_flags[i]

        pd_flags = [ npr.choice(self.flags) for _ in range(len(pd_segs)) ]
        for i, seg in enumerate(pd_segs.values()):
            seg['breakdown_flag'] = pd_flags[i]


class BoxScaleBreakdown(BreakdownPolicy):
    def __init__(self):
        # self.flags = ['tiny', 'small', 'medium', 'large', 'huge']
        # self.scale_thresholds = [ 16 ** 2, 32 ** 2, 64 ** 2, 128 ** 2]

        self.flags = ['small', 'medium', 'large']
        self.scale_thresholds = [32 ** 2, 128 ** 2]

    def breakdown(self, gt_segs, pd_segs):
        thresh, flags = self.scale_thresholds, self.flags
        gt_areas = [
            seg['bbox'][-1] * seg['bbox'][-2] for seg in gt_segs.values()
        ]
        gt_flags = [
            flags[bisect.bisect_right(thresh, s_area)] for s_area in gt_areas
        ]
        # give each gt the flag
        for i, g_seg in enumerate(gt_segs.values()):
            g_seg['breakdown_flag'] = gt_flags[i]

        for p_seg in pd_segs.values():
            matchings = p_seg['matchings']
            flag = None
            if len(matchings) == 0:
                area = p_seg['bbox'][-1] * p_seg['bbox'][-2]
                flag = flags[bisect.bisect_right(thresh, area)]
            else:
                gt_s_sid = list(matchings.keys())[0]
                flag = gt_segs[gt_s_sid]['breakdown_flag']
            p_seg['breakdown_flag'] = flag


class MaskScaleBreakdown(BreakdownPolicy):
    def __init__(self):
        # self.flags = ['tiny', 'small', 'medium', 'large', 'huge']
        # self.scale_thresholds = [ 16 ** 2, 32 ** 2, 64 ** 2, 128 ** 2]

        self.flags = ['small', 'medium', 'large']
        self.scale_thresholds = [32 ** 2, 128 ** 2]

    def breakdown(self, gt_segs, pd_segs):
        thresh, flags = self.scale_thresholds, self.flags
        gt_areas = [ seg['area'] for seg in gt_segs.values() ]
        gt_flags = [
            flags[bisect.bisect_right(thresh, s_area)] for s_area in gt_areas
        ]
        # give each gt the flag
        for i, g_seg in enumerate(gt_segs.values()):
            g_seg['breakdown_flag'] = gt_flags[i]

        for p_seg in pd_segs.values():
            matchings = p_seg['matchings']
            flag = None
            if len(matchings) == 0:
                area = p_seg['area']
                flag = flags[bisect.bisect_right(thresh, area)]
            else:
                gt_s_sid = list(matchings.keys())[0]
                flag = gt_segs[gt_s_sid]['breakdown_flag']
            p_seg['breakdown_flag'] = flag


class BoxAspectRatioBreakdown(BreakdownPolicy):
    def __init__(self):
        self.flags = []

    def breakdown(self, gt_segs, pd_segs):
        pass


policy_register = {
    'dummy': DummyBreakdown,
    'bbox_scale': BoxScaleBreakdown,
    'mask_scale': MaskScaleBreakdown
}


class StatsBreakdown():
    def __init__(self, gt_json_meta_fname, pd_json_meta_fname, breakdown_policy):
        analyzer = PanopticEvalAnalyzer(gt_json_meta_fname, pd_json_meta_fname)
        self.gt_segs = analyzer.gt.segs
        self.pd_segs = analyzer.pd.segs
        self.analyzer = analyzer
        self.policy = policy_register[breakdown_policy]()
        self.policy.breakdown(self.gt_segs, self.pd_segs)
        self.verify_policy_execution(self.policy, self.gt_segs, self.pd_segs)

    @staticmethod
    def verify_policy_execution(policy, gt_segs, pd_segs):
        '''make sure that each seg has been given a flag'''
        flags = policy.flags
        for seg in gt_segs.values():
            assert seg['breakdown_flag'] in flags
        for seg in pd_segs.values():
            assert seg['breakdown_flag'] in flags

    def aggregate(self):
        res = []
        for flag in self.policy.flags:
            dataframe, overall_table, cat_table = self.analyzer.summarize(flag)
            res.append(dataframe)
        res = pd.concat(res, axis=1)

        # print results in semicolon separated format so that I can transfer to google doc
        print(self.policy.flags)
        # upper left corner of the table is 'name'
        cols = ';'.join(['name'] + list(res.columns))
        print(cols)
        for catName, row in res.iterrows():
            row = [ '{:.2f}'.format(elem) for elem in row.values ]
            score_str = ';'.join([catName] + row)
            print(score_str)


def test():
    model = 'pcv'
    dset = 'cityscapes'
    split = 'val'
    gt_json_fname = '/share/data/vision-greg/panoptic/{}/annotations/{}.json'.format(dset, split)
    pd_json_fname = '/home-nfs/whc/panout/{}/{}/{}/pred.json'.format(model, dset, split)

    # analyzer = PanopticEvalAnalyzer(gt_json_fname, pd_json_fname)
    # _, overall_table, cat_table = analyzer.aggregate(
    #     gt_stats=analyzer.gt.match_summarize(),
    #     pd_stats=analyzer.pd.match_summarize(),
    #     cats=analyzer.gt.cats
    # )
    # print(cat_table)
    # print(overall_table)

    breakdown_stats = StatsBreakdown(gt_json_fname, pd_json_fname, 'mask_scale')
    breakdown_stats.aggregate()


def draw_failure_cases_spatially():
    model = 'pcv'
    dset = 'cityscapes'
    split = 'val'
    gt_json_fname = '/share/data/vision-greg/panoptic/{}/annotations/{}.json'.format(dset, split)
    pd_json_fname = '/home-nfs/whc/panout/{}/{}/{}/pred.json'.format(model, dset, split)

    analyzer = PanopticEvalAnalyzer(gt_json_fname, pd_json_fname)
    # plot unmatched gt
    gt_accumulator = np.zeros((1024, 2048))
    # plt.imshow(gt_accumulator)
    gt = analyzer.gt
    for k, seg in gt.segs.items():
        if seg['match_state'] == _SEGMENT_UNMATCHED:
            x, y, w, h = seg['bbox']
            c = [y + h // 2, x + w // 2]
            y, x = c
            gt_accumulator[y, x] = gt_accumulator[y, x] + 1
    return gt_accumulator


if __name__ == "__main__":
    draw_failure_cases_spatially()
