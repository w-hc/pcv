import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from IPython.display import display as ipy_display
from ipywidgets import interactive
import ipywidgets as widgets

from panopticapi.utils import rgb2id
from panoptic.pan_analyzer import (
    PanopticEvalAnalyzer,
    _SEGMENT_MATCHED, _SEGMENT_UNMATCHED, _SEGMENT_FORGIVEN
)


class PanVis():
    def __init__(self, img_root, gt_json_meta_fname, pd_json_meta_fname):
        """Expect that the pan mask dir to be right beside the meta json file
            val.json
            val/
        Args:
            img_root: root dir where images are stored
            gt_json_meta_fname: abs fname to gt json
            pd_json_meta_fname: ...
        """
        self.img_root = img_root

        analyzer = PanopticEvalAnalyzer(gt_json_meta_fname, pd_json_meta_fname)
        self.gt, self.pd = analyzer.gt, analyzer.pd
        self.imgIds = analyzer.imgIds
        self.res_dframe, self.overall_table, self.cat_table = analyzer.summarize()
        # cached widgets
        self.global_walk = None
        self.__init_global_state__()

    def evaluate(self):
        # this is now a dud for backwards compatibility
        pass

    def summarize(self):
        print(self.cat_table)
        print(self.overall_table)

    def __init_global_state__(self):
        self.global_state = {
            'imgId': None, 'segId': None,
            'catId': None, 'tranche': None
        }

    # the following are the modules for widgets, from root to leaf
    def root_wdgt(self):
        """
        root widget delegates to either global or image
        """
        self.summarize()
        modes = ['Global', 'Single-Image']

        def logic(mode):
            # cache the widget later
            if mode == modes[0]:
                if self.global_walk is None:
                    self.global_walk = self.global_walk_specifier()
                ipy_display(self.global_walk)
            elif mode == modes[1]:
                self.image_view = self.single_image_selector()
                # if self.image_view is None:
                #     self.image_view = self.single_image_selector()
                # ipy_display(self.image_view)

        UI = interactive(
            logic, mode=widgets.ToggleButtons(options=modes, value=modes[0])
        )
        UI.children[-1].layout.height = '1000px'
        ipy_display(UI)

    def global_walk_specifier(self):
        tranche_map = self._tranche_filter(self.gt.segs, self.pd.segs)

        def logic(catId, tranche):
            if self.global_state['catId'] != catId \
                    or self.global_state['tranche'] != tranche:
                self.__init_global_state__()
                self.global_state['catId'] = catId
                self.global_state['tranche'] = tranche
            seg_list = self._cat_filter_and_merge_tranche_map(
                tranche_map, [catId], [tranche]
            )
            # areas = [ seg['area'] for seg in seg_list ]
            # plt.hist(areas)
            self.walk_primary(seg_list, is_global=True)
        UI = interactive(
            logic,
            catId=self._category_roulette(self.gt.cats.keys(), multi_select=False),
            tranche=widgets.Select(
                options=tranche_map.keys(), value=list(tranche_map.keys())[0]
            )
        )
        return UI

    def single_image_selector(self):
        imgIds = self.imgIds
        inx, txt = self._inx_txt_scroller_pair(
            imgIds, default_txt=self.global_state['imgId'])

        def logic(inx):
            print("curr image {}/{}".format(inx, len(imgIds)))
            imgId = imgIds[inx]
            self.single_image_view_specifier(imgId)
        UI = interactive(logic, inx=inx)
        ipy_display(txt)
        ipy_display(UI)

    def single_image_view_specifier(self, imgId):
        gt_segs, pd_segs = self.gt.img2seg[imgId], self.pd.img2seg[imgId]
        _gt_cats = {seg['category_id'] for seg in gt_segs.values()}
        _pd_cats = {seg['category_id'] for seg in pd_segs.values()}
        relevant_catIds = _gt_cats | _pd_cats
        tranche_map = self._tranche_filter(gt_segs, pd_segs)
        modes = ['bulk', 'walk']

        def logic(catIds, tranches, mode):
            # only for walk, not for bulk display
            seg_list = self._cat_filter_and_merge_tranche_map(
                tranche_map, catIds, tranches)
            if mode == modes[0]:
                self.single_image_bulk_display(seg_list)
            elif mode == modes[1]:
                self.walk_primary(seg_list)
        UI = interactive(
            logic,
            mode=widgets.ToggleButtons(options=modes, value=modes[0]),
            catIds=self._category_roulette(
                relevant_catIds, multi_select=True,
                default_cid=[self.global_state['catId']]
            ),
            tranches=widgets.SelectMultiple(
                options=tranche_map.keys(),
                value=[self.global_state['tranche']]
            )
        )
        ipy_display(UI)

    def single_image_bulk_display(self, segs):
        if len(segs) == 0:
            return 'no segments in this tranche'
        imgId = segs[0]['image_id']
        for seg in segs:
            assert seg['image_id'] == imgId
        segIds = list(map(lambda x: x['sid'], segs))
        gt_seg_ids = list(filter(lambda x: x.startswith('gt/'), segIds))
        pd_seg_ids = list(filter(lambda x: x.startswith('pd/'), segIds))
        self.single_image_plot(imgId, gt_seg_ids, pd_seg_ids)

    def walk_primary(self, segs, is_global=False):
        """segs: a list of seg"""
        # the watching logic here is quite messy
        sids = [seg['sid'] for seg in segs]
        if len(sids) == 0:
            return 'no available segs'
        inx, txt = self._inx_txt_scroller_pair(
            sids, default_txt=self.global_state['segId'] if is_global else None
        )

        def logic(inx):
            seg = segs[inx]
            if is_global:
                self.global_state['segId'] = seg['sid']
                self.global_state['imgId'] = seg['image_id']
            print("Primary seg {}/{} matches with {} segments".format(
                inx, len(segs), len(seg['matchings'])))
            self.walk_matched(seg)
        UI = interactive(logic, inx=inx)
        print("Primary Segment:")
        ipy_display(txt)
        ipy_display(UI)

    def walk_matched(self, ref_seg):
        """child of walk_primary"""
        ref_sid = ref_seg['sid']
        # note that matchings is {sid: IoU}
        matched_sids = list(ref_seg['matchings'].keys())
        matched_ious = list(ref_seg['matchings'].values())
        if len(matched_sids) == 0:
            matched_sids = (None, )
            matched_ious = (0, )

        def segid_to_catname(partition, sid):
            if sid is None:
                return 'NA'
            return self.gt.cats[partition.segs[sid]['category_id']]['name']

        def logic(inx):
            match_sid = matched_sids[inx]
            if ref_sid.startswith('gt/'):
                gt_sid, pd_sid, highlight = ref_sid, match_sid, 1
                imgId = self.gt.segs[ref_sid]['image_id']
            else:
                gt_sid, pd_sid, highlight = match_sid, ref_sid, 2
                imgId = self.pd.segs[ref_sid]['image_id']
            print('IoU: {:.3f}'.format(matched_ious[inx]))
            print('gt: {} vs pd: {}'.format(
                segid_to_catname(self.gt, gt_sid),
                segid_to_catname(self.pd, pd_sid)
            ))
            self.single_image_plot(imgId, gt_sid, pd_sid, highlight)

        inx, txt = self._inx_txt_scroller_pair(matched_sids)
        UI = interactive(logic, inx=inx)
        print("Matched Segment:")
        ipy_display(txt)
        ipy_display(UI)

    @staticmethod
    def _tranche_filter(gt_segs, pd_segs):
        """
        Args:
            gt_segs: {segId: seg}
            pd_segs: {segId: seg}
        """
        def _filter(state, seg_map):
            seg_list = [
                seg for seg in seg_map.values() if seg['match_state'] == state
            ]
            seg_list = sorted(seg_list, key=lambda x: x['area'], reverse=True)
            return seg_list

        tranche_map = {
            'TP': _filter(_SEGMENT_MATCHED, pd_segs),
            'FN': _filter(_SEGMENT_UNMATCHED, gt_segs),
            'FP': _filter(_SEGMENT_UNMATCHED, pd_segs),
            'GT_FORGIVEN': _filter(_SEGMENT_FORGIVEN, gt_segs),
            'PD_FORGIVEN': _filter(_SEGMENT_FORGIVEN, pd_segs)
        }
        assert len(gt_segs) == sum(
            map(lambda x: len(tranche_map[x]), ['TP', 'FN', 'GT_FORGIVEN'])
        )
        assert len(pd_segs) == sum(
            map(lambda x: len(tranche_map[x]), ['TP', 'FP', 'PD_FORGIVEN'])
        )
        return tranche_map

    @staticmethod
    def _cat_filter_and_merge_tranche_map(tranche_map, catIds, chosen_tranches):
        local_tranche_map = {
            k: list(filter(lambda seg: seg['category_id'] in catIds, seg_list))
            for k, seg_list in tranche_map.items()
        }
        for k, v in local_tranche_map.items():
            print("{}: {}".format(k, len(v)), end='; ')
        print('')
        seg_list = sum([local_tranche_map[_tr] for _tr in chosen_tranches], [])
        return seg_list

    @staticmethod
    def _inx_txt_scroller_pair(sids, default_txt=None):
        """
        Args:
            sids: [str, ] segment ids
        Note that since a handler is only called if 'value' changes, this mutual
        watching would not lead to infinite back-and-forth bouncing.
        In addition, bouncing-back is prevented by internal_change flag.
        """
        assert len(sids) > 0
        if default_txt is not None:
            default_inx, default_txt = sids.index(default_txt), default_txt
        else:
            default_inx, default_txt = 0, sids[0]
        inx = widgets.BoundedIntText(value=default_inx, min=0, max=len(sids) - 1)
        txt = widgets.Text(value=str(default_txt), description='ID')
        internal_change = False

        def inx_update_reaction(*args):
            nonlocal internal_change
            if internal_change:
                internal_change = False
                return
            curr_inx = inx.value
            curr_sid = sids[curr_inx]
            internal_change = True
            txt.value = curr_sid
        inx.observe(inx_update_reaction, 'value')

        def txt_update_reaction(*args):
            nonlocal internal_change
            if internal_change:
                internal_change = False
                return
            curr_sid = txt.value
            if curr_sid in sids:
                curr_inx = sids.index(curr_sid)
                internal_change = True
                inx.value = curr_inx
        txt.observe(txt_update_reaction, 'value')

        return inx, txt

    def _category_roulette(
        self, selected_catIds, multi_select=False, default_cid=None,
    ):
        """
        Things first, Stuff next, each sorted from high to low by PQ
        Note that this roulette is multi-selective, and return a tuple of catIds
        e.g.
            T, 16.60, Person
            T, 15.12, Bicycle
            S, 32.10, Road
        """
        catIds = np.array(sorted(self.gt.cats.keys()))
        PQ = self.res_dframe.values[:, 0]  # (num_cats, )

        # first filter by selection, then sort by PQ from high to low
        chosen_mask = np.array(
            [ catId in selected_catIds for catId in catIds ], dtype=np.bool)
        catIds, PQ = catIds[chosen_mask], PQ[chosen_mask]
        order = np.argsort(-PQ)  # high to low
        catIds, PQ = catIds[order], PQ[order]

        # now do things first followed by stuff
        acc = []
        isthing = np.array(
            [self.gt.cats[id]['isthing'] for id in catIds], dtype=bool)
        acc += [
            ('T, {:>4.2f}, {}'.format(pq, self.gt.cats[cid]['name']), cid)
            for pq, cid in zip(PQ[isthing], catIds[isthing])
        ]
        acc += [
            ('S, {:>4.2f}, {}'.format(pq, self.gt.cats[cid]['name']), cid)
            for pq, cid in zip(PQ[~isthing], catIds[~isthing])
        ]

        if default_cid is None:
            default_cid = acc[0][1]
        if multi_select and not isinstance(default_cid, (tuple, list)):
            default_cid = [default_cid]
        _module = widgets.SelectMultiple if multi_select else widgets.Select
        roulette = _module(options=acc, rows=15, value=default_cid)
        return roulette

    def single_image_plot(
        self, imgId, gt_seg_sid_list, pd_seg_sid_list,
        highlight=None, seg_alpha=0.7, seg_cmap='Blues'
    ):
        # first load image and annotations masks
        img = np.array(Image.open(
            osp.join(self.img_root, self.gt.imgs[imgId]['file_name'])
        ))
        gt_rgb = np.array(
            Image.open(osp.join(
                self.gt.mask_root, self.gt.imgs[imgId]['ann_fname']
            )),
            dtype=np.uint32
        )
        gt_mask = rgb2id(gt_rgb)
        pd_rgb = np.array(
            Image.open(osp.join(
                self.pd.mask_root, self.pd.imgs[imgId]['ann_fname']
            )),
            dtype=np.uint32
        )
        pd_mask = rgb2id(pd_rgb)

        # now aggregate the segment masks for both pd and gt
        def aggregate_seg_mask(sid_list, ref_mask, segs_map):
            if sid_list is None:
                sid_list = []
            if not isinstance(sid_list, (list, tuple)):
                sid_list = (sid_list, )
            seg_mask = np.zeros(ref_mask.shape, dtype=np.bool)
            for sid in sid_list:
                seg = segs_map[sid]
                assert seg['image_id'] == imgId
                seg_mask |= (ref_mask == seg['id'])
            return seg_mask

        gt_seg_mask = aggregate_seg_mask(gt_seg_sid_list, gt_mask, self.gt.segs)
        pd_seg_mask = aggregate_seg_mask(pd_seg_sid_list, pd_mask, self.pd.segs)

        # plot them together
        WHITE = [255, 255, 255]
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(30, 12))
        axes[0].imshow(img)
        gt_rgb[gt_seg_mask] = WHITE
        axes[1].imshow(gt_rgb)
        # axes[1].imshow(gt_seg_mask, alpha=seg_alpha, cmap=seg_cmap)
        pd_rgb[pd_seg_mask] = WHITE
        axes[2].imshow(pd_rgb)
        # axes[2].imshow(pd_seg_mask, alpha=seg_alpha, cmap=seg_cmap)

        if highlight is not None:
            axes[highlight].set_title(
                'frame in focus', bbox=dict(facecolor='orange')
            )
        plt.show()
