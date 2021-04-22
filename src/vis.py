import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import cm

from ipywidgets import Output
from IPython.display import display as ipy_display

from panoptic.pcv.inference.mask_from_vote import MaskFromVote
from panoptic.box_and_mask import get_xywh_bbox_from_binary_mask

from panopticapi.utils import rgb2id, id2rgb

_BLUE = (0, 0, 255)
_YELLOW = (255, 255, 102)
_GREEN = (102, 255, 102)
_RED = (255, 51, 51)


def add_vote_grid(ax, grid_spec, x, y, inds, conf, color_spec=None):
    # if color_spec is None:
    #     color_spec = ['white'] + ['yellow'] * (len(inds) - 1)
    color_mapper = cm.get_cmap('Oranges')
    grid_spec = grid_spec.copy()
    grid_spec[:, :2] += (y, x)
    del x, y
    center_boxes = []
    smear_boxes = []
    label_artists = []
    for i, (cell_inx, prob) in enumerate(zip(inds, conf)):
        y, x, r = grid_spec[cell_inx]
        s = 2 * r + 1
        color = color_mapper(prob)
        # color = color_spec[i]
        smear_boxes.append(
            Rectangle((x - r, y - r), s, s, color=color, alpha=0.5),
        )
        # label_artists.append(
        #     ax.text(
        #         x + r, y - r,  # put the text on the top right of the box
        #         s='{:.2f}'.format(prob),
        #         fontsize=12, fontweight='bold', color='yellow',
        #         horizontalalignment='center'
        #     )
        # )
        # center_boxes.append(
        #     Rectangle((x, y), 1, 1, color=color, alpha=1.0),
        # )
    total = smear_boxes + center_boxes
    vote = PatchCollection(total, match_original=True)
    ax.add_collection(vote)
    return vote, label_artists


class Plot():
    def __init__(self, ax, data, visualizer):
        self.centroid_color = 'red'
        self.ax = ax
        self.ax.axis('off')
        # self.ax.get_xaxis().set_visible(False)
        # self.ax.get_yaxis().set_visible(False)
        self.trainId_2_catName = visualizer.trainId_2_catName
        self.pcv = visualizer.pcv
        self.mfv = visualizer.mfv

        self.data = data
        self.ephemeral_artists = []
        # persisting data that can store auxiliary info for artists
        self.artists_buffer = dict()
        self.init_artists()
        self.render_visual()

        self.tx_data = []
        self.txArtist = ax.text(
            0, -0.4, s='', transform=ax.transAxes
        )
        self.render_text('inited', overwrite=True)

        self.vote_pred = data['vote_pred']
        self.pressed_xy = None

    def init_artists(self, clean_buffer=True):
        for vagabond in self.ephemeral_artists:
            vagabond.remove()

        self.ephemeral_artists = []

        if clean_buffer:
            self.artists_buffer = {'mask': []}

    def press_coord(self, x, y, button):
        self.pressed_xy = x, y
        self.query_coord(x, y, button)

    def query_coord(self, x, y, button):
        assert button is not None
        if button == 1:
            self.init_artists()
            val = self.val_at_xy(x, y)
            self.render_text(val, overwrite=True)
            self.render_mouse_clicker(x, y)
            self.render_vote(x, y)
        else:
            self.init_artists(clean_buffer=False)
            self.render_allegiance_mask(x, y)

    def motion_coord(self, x, y):
        if self.pressed_xy is None:
            return
        ref_x, ref_y = self.pressed_xy
        vote_pred = self.vote_pred
        conf = vote_pred[ref_y, ref_x]
        curr_offset = np.array([x, y]) - np.array([ref_x, ref_y])
        curr_offset = curr_offset.reshape(1, -1)
        bin = self.pcv.discrete_vote_inx_from_offset(curr_offset)
        bin = bin.squeeze()
        text = ''
        if bin == -1:
            text += "not voting here"
        else:
            conf = conf[bin]
            r = self.pcv.grid_spec[bin][-1]
            d = 2 * r + 1
            area = float(d ** 2)
            average_impact = conf / area
            text += "mouse tip: voting {:.2f}\n".format(conf)
            text += "mous tip: smear: {} voting impact {:.2f}".format(
                area, average_impact
            )
        tip_val = self.val_at_xy(x, y)
        if tip_val:
            text += "\nmouse tip: {}".format(tip_val)
        self.render_text(text, ephemeral=True)

    def val_at_xy(self, x, y):
        val = self.data['im'][y, x]
        return repr(val)

    def render_visual(self):
        self._render_base()
        # self._render_centroids()

    def _render_base(self):
        base_data = self.data['im']
        self.ax.imshow(base_data)

    def _render_centroids(self):
        cen = self.data['ins_centroids']
        centroid_boxes = []
        for x, y in cen:
            centroid_boxes.append(
                Rectangle((x, y), 1, 1)
            )
        centroid_boxes = PatchCollection(centroid_boxes, color=self.centroid_color)
        self.ax.add_collection(centroid_boxes)
        # self.ax.scatter(
        #     cen[:, 0], cen[:, 1], s=2, marker='.', c=self.centroid_color
        # )

    def render_mouse_clicker(self, x, y):
        # cir = Circle((x, y), radius=1, color='white')
        # self.ax.add_patch(cir)
        # return
        marker = self.ax.scatter(x, y, s=200, c='cyan', marker='x')
        self.ephemeral_artists.append(marker)

    def render_allegiance_mask(self, x, y):
        _, masks, sem_cats = self.mfv.peak_conv_mask_match(
            self.mfv.thing_trainIds, self.mfv.query_mask,
            self.data['vote_decision'], self.data['sem_decision'],
            peak_bbox=np.array([[x, y, 1, 1]])
        )
        if len(masks) == 0:
            return
        masks, sem_cats = masks.cpu().numpy(), sem_cats.cpu().numpy()
        mask, cat = masks[0], sem_cats[0]
        self.artists_buffer['mask'].append(mask)
        merged_mask = sum(self.artists_buffer['mask'])
        merged_mask = merged_mask > 0
        maskArtist = self.ax.imshow(merged_mask, alpha=0.5, cmap='Blues')
        self.ephemeral_artists.append(maskArtist)

    def render_vote(self, x, y):
        vote_pred = self.vote_pred
        conf = vote_pred[y, x]
        thresh = 0.01
        inds = np.argsort(-1 * conf)
        conf = conf[inds]
        filt = np.where(conf >= thresh)
        inds, conf = inds[filt], conf[filt]
        # topk = 5
        # inds, conf = inds[:topk], conf[:topk]

        voteArtist, score_labels = add_vote_grid(
            self.ax, self.pcv.grid_spec, x, y, inds, conf, color_spec=None
        )
        self.ephemeral_artists.append(voteArtist)
        self.ephemeral_artists.extend(score_labels)
        if len(conf) > 5:
            conf = conf[:5]
        text = "top votes > {}: {}".format(thresh, conf)
        self.render_text(text, overwrite=False)

    def render_text(self, text, ephemeral=False, overwrite=True):
        if not text:
            if overwrite:
                self.tx_data = []
            return

        if overwrite:
            content = [text]
        else:
            content = self.tx_data + [text]

        if not ephemeral:
            self.tx_data = content

        content = '\n'.join(content)
        self.txArtist.set_text(content)


class Im(Plot):
    def val_at_xy(self, x, y):
        return None

    def _render_base(self):
        base_data = self.data['im']
        self.ax.imshow(base_data, extent=[0, base_data.shape[1]//4, base_data.shape[0]//4, 0])


class PanImg(Plot):
    def _render_centroids(self):
        self.centroid_color = 'orange'
        super()._render_centroids()

    def _render_base(self):
        base_data = self.data['pan_img']
        self.ax.imshow(base_data)

    def val_at_xy(self, x, y):
        gt = self.data['sem_gt']
        id = gt[y, x]
        cls_name = self.trainId_2_catName[id]
        return "sem gt: {}".format(cls_name)


class SemPred(Plot):
    def _render_base(self):
        self.pred = self.data['sem_pred'].argmax(axis=2)
        self.ax.imshow(self.pred)

    def val_at_xy(self, x, y):
        pred = self.data['sem_pred']
        top_k = 3
        confidences = pred[y, x]
        assert len(confidences) == len(self.trainId_2_catName) - 1
        top_cls = np.argsort(-1 * confidences)[:top_k]
        confidences = confidences[top_cls]
        acc = ''
        for cls_ind, conf in zip(top_cls, confidences):
            catname = self.trainId_2_catName[cls_ind]
            acc += '{}: {:.2f}\n '.format(catname, conf)
        return acc


class PredHmap(Plot):
    def _render_base(self):
        base_data = self.data['vote_pred_hmap']
        self.ax.imshow(base_data)

    def val_at_xy(self, x, y):
        hmap = self.data['vote_pred_hmap']
        return "total vote here: {:.2f}".format(hmap[y, x])


class GtHmap(Plot):
    def __init__(self, ax, data, visualizer):
        super().__init__(ax, data, visualizer)
        self.vote_pred = data['vote_gt_pred']

    def _render_base(self):
        base_data = self.data['vote_gt_hmap']
        self.ax.imshow(base_data)

    def _render_centroids(self):
        pass

    def val_at_xy(self, x, y):
        hmap = self.data['vote_gt_hmap']
        return "total vote here: {:.2f}".format(hmap[y, x])


class VoteErr(Plot):
    def _render_base(self):
        vote_decision = self.data['vote_decision'][..., 0].cpu().numpy()
        vote_gt = self.data['vote_gt'].copy()
        abstain_inx = vote_gt.max()
        vote_gt[vote_gt == abstain_inx] = -1
        self.vote_match = (vote_decision == vote_gt)

        h, w = vote_decision.shape
        to_display = np.zeros((h, w, 3), dtype=np.uint8)
        to_display[(vote_gt != -1) & (vote_decision == vote_gt)] = _YELLOW
        to_display[(vote_gt != -1) & (vote_decision != vote_gt)] = _BLUE
        to_display[(vote_gt == -1) & (vote_decision == vote_gt)] = _GREEN
        to_display[(vote_gt == -1) & (vote_decision != vote_gt)] = _RED
        self.ax.imshow(to_display)

    def val_at_xy(self, x, y):
        return "correct? {} gt {} vs pd {}".format(
            self.vote_match[y, x],
            self.data['vote_gt'][y, x], self.data['vote_decision'][y, x]
        )

    def _render_centroids(self):
        pass


class WShed_Basins(Plot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_text(
            '{} instances detected'.format(len(self.data['ws_peak_points'])),
            ephemeral=True
        )

    def _render_base(self):
        self.ax.imshow(self.data['ws_mask'])

    def _render_centroids(self):
        cen = self.data['ws_peak_points']
        centroid_boxes = []
        for x, y in cen:
            centroid_boxes.append(
                Rectangle((x, y), 1, 1)
            )
        centroid_boxes = PatchCollection(centroid_boxes, color='white')
        self.ax.add_collection(centroid_boxes)

    def val_at_xy(self, x, y):
        return 'instance index: {}/{}'.format(
            self.data['ws_mask'][y, x], len(self.data['ws_peak_points'])
        )

    def render_allegiance_mask(self, x, y):
        ws_mask = self.data['ws_mask']
        mask_buffer = self.artists_buffer['mask']
        if ws_mask[y, x] == 0:
            return  # only activated if click falls inside a basin
        if len(mask_buffer) > 0 and sum(mask_buffer)[y, x] > 0:
            pass  # if the mask is already done, don't redo
        else:
            peak_bbox = get_xywh_bbox_from_binary_mask(ws_mask == ws_mask[y, x])
            _, masks, sem_cats = self.mfv.peak_conv_mask_match(
                self.mfv.thing_trainIds, self.mfv.query_mask,
                self.data['vote_decision'], self.data['sem_decision'],
                peak_bbox=np.array([peak_bbox])
            )
            masks, sem_cats = masks.cpu().numpy(), sem_cats.cpu().numpy()
            mask, cat = masks[0], sem_cats[0]
            mask_buffer.append(mask)
        merged_mask = sum(mask_buffer)
        # merged_mask = merged_mask > 0  Let the overlap be shown
        maskArtist = self.ax.imshow(merged_mask, alpha=0.5, cmap='Reds')
        self.ephemeral_artists.append(maskArtist)


class _LossPlot(Plot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _render_base(self):
        self.ax.imshow(self.data['loss_info'][self.loss_key]['raw'], cmap='Wistia')

    def _render_centroids(self):
        return
        # cen = self.data['ins_centroids']
        # centroid_boxes = []
        # for (x, y), loss_dict in zip(cen, self.data['per_inst_loss']):
        #     centroid_boxes.append(
        #         Rectangle((x, y), 1, 1)
        #     )
        #     key = 'sem' if self.loss_key == 'sem_pix_loss' else 'vote'
        #     loss = loss_dict[key]
        #     self.ax.text(x, y, s='{:.2f}'.format(loss), fontsize='xx-small')
        # centroid_boxes = PatchCollection(centroid_boxes, color=self.centroid_color)
        # self.ax.add_collection(centroid_boxes)

    def val_at_xy(self, x, y):
        pan_mask = self.data['pan_mask']
        seg_id = pan_mask[y, x]
        seg_mask = (pan_mask == seg_id)

        # 1. highlight the selected segment
        maskArtist = self.ax.imshow(seg_mask, alpha=0.2, cmap='hot')
        self.ephemeral_artists.append(maskArtist)

        if seg_id == 0:
            return 'ignored segment'

        # 2. display geometric info of this segment
        txt = ''
        area = seg_mask.sum()
        w, h = get_xywh_bbox_from_binary_mask(seg_mask)[2:]
        txt += 'width: {}, height: {}, area: {}\n'.format(w, h, area)

        # 3. display loss info of this segment
        raw_loss_mask = self.data['loss_info'][self.loss_key]['raw']
        # normalized_loss_mask = self.data['loss_info'][self.loss_key]['norm']
        txt += 'point loss {:.3f}, segment ave loss {:.3f}\n'.format(
            raw_loss_mask[y, x], raw_loss_mask[seg_mask].mean()
        )
        seg_stats = self.data['seg_loss_stats'].stats[seg_id]
        isthing = seg_stats['isthing']
        seg_loss_contrib = seg_stats[self.loss_key]
        overall_loss = self.data['loss_info']['overall_{}'.format(self.loss_key)]  # ugly line
        txt += 'isthing {}, contrib {:.3f}/{:.2f}%'.format(
            isthing, seg_loss_contrib, 100 * seg_loss_contrib / overall_loss
        )
        return txt

    def motion_coord(self, x, y):
        pass

    def render_vote(self, x, y):
        pass


class SemLoss(_LossPlot):
    def __init__(self, *args, **kwargs):
        self.loss_key = 'sem'
        super().__init__(*args, **kwargs)


class VoteLoss(_LossPlot):
    def __init__(self, *args, **kwargs):
        self.loss_key = 'vote'
        super().__init__(*args, **kwargs)


plot_device_registry = {
    'im': Im,
    'pan_img': PanImg,
    'sem_pred': SemPred,
    'pred_hmap': PredHmap,
    'gt_hmap': GtHmap,
    'vote_err': VoteErr,
    'WShed_basins': WShed_Basins,
    'SemLoss': SemLoss,
    'VoteLoss': VoteLoss
}


class Visualizer():
    def __init__(self, cfg, dset_meta, pcv):
        self.cfg = cfg
        self.output_widget = Output()
        self.dset_meta = dset_meta
        self.pcv = pcv
        self.trainId_2_catName = dset_meta['trainId_2_catName']
        self.category_meta = dset_meta['cats']
        self.catId_2_trainId = dset_meta['catId_2_trainId']
        self.init_state()
        self.pressed = False

        np.set_printoptions(
            formatter={'float': lambda x: "{:.2f}".format(x)}
        )

    def init_state(self):
        self.fig, self.canvas, self.plots = None, None, None

    def __del__(self):
        self.clear_state()
        self.output_widget.close()

    def clear_state(self):
        if self.fig is not None:
            self.disconnect()
            plt.close(self.fig)
            self.init_state()

    def display_stdout_and_err_in_curr_cell(self):
        """
        in JLab, stdout and stderr from widget callbacks
        must be displayed through a specialized output widget
        """
        ipy_display(self.output_widget)

    def connect(self):
        decor = self.output_widget.capture()
        self.cidpress = self.canvas.mpl_connect(
            'button_press_event', decor(self.on_press))
        self.cidrelease = self.canvas.mpl_connect(
            'button_release_event', decor(self.on_release))
        self.cidmotion = self.canvas.mpl_connect(
            'motion_notify_event', decor(self.on_motion))

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.canvas.mpl_disconnect(self.cidpress)
        self.canvas.mpl_disconnect(self.cidrelease)
        self.canvas.mpl_disconnect(self.cidmotion)

    def on_press(self, event):
        self.pressed = True
        ax_in_focus = event.inaxes
        if ax_in_focus is None:
            return
        x, y, button = int(event.xdata), int(event.ydata), event.button
        for k, plot in self.plots.items():
            if ax_in_focus == plot.ax:
                plot.press_coord(x, y, button)
            else:
                plot.query_coord(x, y, button)

    def on_motion(self, event):
        if not self.pressed:
            return
        ax_in_focus = event.inaxes
        if ax_in_focus is None:
            return

        x, y = int(event.xdata), int(event.ydata)
        for k, plot in self.plots.items():
            if ax_in_focus == plot.ax:
                plot.motion_coord(x, y)

    def on_release(self, event):
        self.pressed = False

    @torch.no_grad()
    def vis(
        self, im, pan_mask, segments_info, sem_pred, vote_pred,
        gt_prod_handle, loss_module, h_thresh
    ):
        """Bulk of the logic
        Args: these are possible data to visualize
            im:        [H, W, 3] of PIL Image
            pan_mask:  [H, W, 3] of PIL Image
            segments_info: dict
            sem_pred:  [1, num_classes, H, W] torch gpu tsr
            vote_pred: [1, num_bins, H, W] torch gpu tsr
        """

        ins_mask = MaskFromVote(
            self.cfg.pcv, self.dset_meta, self.pcv, sem_pred.clone(), vote_pred.clone()
        ).infer_panoptic_mask(instance_mask_only=True)[0]

        full_mask, pred_ann = MaskFromVote(
            self.cfg.pcv, self.dset_meta, self.pcv, sem_pred.clone(), vote_pred.clone()
        ).infer_panoptic_mask(instance_mask_only=False)

        # get_each_instance separately
        pairs = []
        tmp_mfv= MaskFromVote(
            self.cfg.pcv, self.dset_meta, self.pcv, sem_pred.clone(), vote_pred.clone()
        )
        peak_regions, _, peak_bbox = \
            tmp_mfv.locate_peak_regions(tmp_mfv.vote_hmap, tmp_mfv.hmap_thresh)
        _, instance_tsr, _ = tmp_mfv.peak_conv_mask_match(
            tmp_mfv.thing_trainIds, tmp_mfv.query_mask,
            tmp_mfv.vote_decision, tmp_mfv.sem_decision, peak_bbox
        )
        if len(np.unique(peak_regions)) - 1 == len(instance_tsr):
            for _i, _ins_mask in enumerate(instance_tsr.cpu().numpy()):
                _reg = peak_regions == (_i+1)
                pairs.append((_reg, _ins_mask))

        self.mfv = MaskFromVote(
            self.cfg.pcv, self.dset_meta, self.pcv, sem_pred.clone(), vote_pred.clone()
        )
        data = self.process_data(
            im, pan_mask, segments_info, sem_pred, vote_pred,
            gt_prod_handle, loss_module, h_thresh
        )
        self.data = data  # store it so that it can be accessed externally
        data['ins_mask'] = ins_mask
        data['full_mask'] = full_mask
        data['pairs'] = pairs
        # plt.imshow(id2rgb(full_mask))
        # plt.show()

        # data['d2_vis'] = d2_vis(self.dset_meta, full_mask, pred_ann, data['im'])

        self.clear_state()
        num_plots = len(plot_device_registry)
        num_per_row = 3
        nrows = (num_plots + num_per_row - 1) // num_per_row
        fig = plt.figure(figsize=(20, 12), constrained_layout=True)
        self.fig = fig
        self.canvas = fig.canvas
        self.plots = dict()
        gs = GridSpec(nrows, num_per_row, figure=fig)
        for i, k in enumerate(plot_device_registry.keys()):
            ax = fig.add_subplot(gs[i // num_per_row, i % num_per_row])
            ax.set_title(k)
            device = plot_device_registry[k]
            self.plots[k] = device(ax, data, self)
        # self.plots['sem_pred'].data['sem_pred'] = id2rgb(full_mask)
        # self.plots['sem_pred'].render_visual()
        self.connect()

    def process_data(
        self, im, pan_img, segments_info, sem_pred, vote_pred,
        gt_prod_handle, loss_module, h_thresh
    ):
        data = {}
        mfv = self.mfv

        # 1. store data derived from gt; sem and vote pred are already softmaxed!
        generator = gt_prod_handle(
            self.dset_meta, self.pcv, pan_img, segments_info
        )
        gts = generator.generate_gt()
        sem_gt, vote_gt = gts[:2]  # the first 2 are always these
        centroids = generator.ins_centroids
        _, vote_tsr = generator.collect_prob_tsr()
        vote_tsr = vote_tsr[:, :-1, :, :]

        data['im'], data['pan_img'] = np.array(im), np.array(pan_img)
        data['pan_mask'] = rgb2id(data['pan_img'])
        data['sem_gt'] = sem_gt
        data['vote_gt_pred'] = vote_tsr.squeeze(axis=0).transpose(1, 2, 0)
        data['vote_gt'], data['ins_centroids'] = vote_gt, centroids
        data['vote_gt_hmap'] = mfv.pixel_consensus_voting(
            torch.as_tensor(vote_tsr).float().cuda()
        )

        # 2. compute and analyze loss
        loss_info = compute_loss(loss_module, gts, sem_pred, vote_pred)
        stats = SegmentLossStats(
            loss_info, data['pan_mask'], segments_info, self.dset_meta['cats']
        )
        stats.summarize()
        data['loss_info'] = loss_info
        data['seg_loss_stats'] = stats

        # 3. store data derived from pred
        data['sem_pred'], data['sem_decision'] = mfv.sem_pred.cpu().numpy(), mfv.sem_decision
        data['vote_pred'], data['vote_decision'] = mfv.vote_pred, mfv.vote_decision
        data['vote_pred_hmap'] = mfv.vote_hmap
        ws_mask, peaks, peak_bbox = mfv.locate_peak_regions(mfv.vote_hmap, h_thresh)
        data['ws_mask'], data['ws_peak_points'] = ws_mask, peaks

        return data


def d2_vis(dset_meta, pan_mask, pan_ann, im, scale=0.7):
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import ColorMode, Visualizer
    # print(self.dset_meta)
    # if len(self.dset_meta['cats']) > 20:
    if len(dset_meta['cats']) > 20:
        meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
    else:
        thing_ids = [_['id'] for _ in dset_meta['cats'].values() if _['isthing']]
        stuff_ids = [_['id'] for _ in dset_meta['cats'].values() if not _['isthing']]
        thing_colors = [dset_meta['cats'][_]['color'] for _ in thing_ids]
        stuff_colors = [dset_meta['cats'][_]['color'] for _ in stuff_ids]
        thing_classes = [dset_meta['cats'][_]['name'] for _ in thing_ids]
        stuff_classes = [dset_meta['cats'][_]['name'] for _ in stuff_ids]
        thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
        stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}

        from detectron2.data.catalog import Metadata
        meta = Metadata().set(
            thing_ids=thing_ids,
            stuff_ids=stuff_ids,
            thing_colors=thing_colors,
            stuff_colors=stuff_colors,
            thing_classes=thing_classes,
            stuff_classes=stuff_classes,
            thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id
        )
    for seg in pan_ann['segments_info']:
        if seg['isthing']:
            seg["category_id"] = meta.thing_dataset_id_to_contiguous_id[seg["category_id"]]
        else:
            seg["category_id"] = meta.stuff_dataset_id_to_contiguous_id[seg["category_id"]]

    def upsample_mask(mask, im_size):
        # return torch.nn.functional.upsample(torch.from_numpy(mask.astype(np.int32)).float().unsqueeze(0).unsqueeze(0), scale_factor=scale_factor).squeeze(0).squeeze(0).long()
        mask = torch.from_numpy(mask.astype(np.int64)).cuda()
        # import pdb;pdb.set_trace()
        inx = torch.unique(mask[mask > 0])
        inx_len = len(inx)
        tmp = mask.new_zeros((inx_len, )+mask.shape, dtype=torch.bool)
        for i in range(inx_len):
            tmp[i, :, :] = mask == inx[i]
        tmp = torch.nn.functional.interpolate(
            tmp.float().unsqueeze(0), im_size, mode='bicubic'
        ).squeeze(0)
        tmp = torch.nn.functional.avg_pool2d(
            tmp.float().unsqueeze(0), kernel_size=7, stride=1, padding=3
        ).squeeze(0)
        _out_mask = tmp.argmax(dim=0)
        _out_mask[tmp.max(0)[0] < 0.5] = -1
        out_mask = torch.zeros_like(_out_mask)
        for i in range(inx_len):
            out_mask[_out_mask == i] = inx[i]
        return out_mask.cpu()

    # from PIL import Image
    # im = np.array(Image.fromarray(im).resize((im.shape[1] // 3, im.shape[0]//3)))
    # print(im.max())

    vis_img = Visualizer(
        im, meta, instance_mode=ColorMode.IMAGE_BW, scale=scale
    ).draw_pan_seg(
        upsample_mask(pan_mask, (im.shape[0], im.shape[1])),
        pan_ann['segments_info'], alpha=0.5
    ).get_image()
    return vis_img


@torch.no_grad()
def compute_loss(loss_module, gts, sem_pred, vote_pred):
    '''primarily a wrapper around loss_module, taking care of miscellaneous
    actions such as moving tensor to and from devices
    '''
    sem_pred, vote_pred = torch.log(sem_pred), torch.log(vote_pred)
    device = sem_pred.device
    gts = [ torch.as_tensor(elem).unsqueeze(dim=0).to(device) for elem in gts ]
    raw_sem_pix_loss, raw_vote_pix_loss = loss_module.per_pix_loss(
        sem_pred, vote_pred, *gts)
    if len(gts) == 3:
        weight_mask = vote_weight_mask = gts[-1]
    elif len(gts) == 4:
        weight_mask, vote_weight_mask = gts[-2:]  # this is ugly. replying on the fact that if
    # weight_mask is not needed, then just supplying an ignored parameter
    norm_sem_pix_loss, norm_vote_pix_loss = loss_module.normalize(
        raw_sem_pix_loss, raw_vote_pix_loss, weight_mask, vote_weight_mask
    )
    l, s_l, v_l = loss_module.aggregate(norm_sem_pix_loss, norm_vote_pix_loss)

    # calling the same function 7 times is hardly decent
    # use a recursion later to make it look respectable later
    def tsr_to_cpu(tsr):
        tsr = tsr.cpu().numpy()
        if len(tsr.shape) > 1:
            tsr = tsr.squeeze(0)
        return tsr

    loss = {
        'sem': {
            'raw': tsr_to_cpu(raw_sem_pix_loss),
            'norm': tsr_to_cpu(norm_sem_pix_loss)
        },
        'vote': {
            'raw': tsr_to_cpu(raw_vote_pix_loss),
            'norm': tsr_to_cpu(norm_vote_pix_loss)
        },
        'overall_combined': tsr_to_cpu(l),
        'overall_sem': tsr_to_cpu(s_l),
        'overall_vote': tsr_to_cpu(v_l)
    }
    return loss


class SegmentLossStats():
    def __init__(self, loss_dict, pan_mask, segments_info, category_meta):
        stats = dict()
        sem_loss = loss_dict['sem']['norm']
        vote_loss = loss_dict['vote']['norm']

        for segment_id, info in segments_info.items():
            cat, iscrowd = info['category_id'], info['iscrowd']
            isthing = category_meta[cat]['isthing']
            segment_mask = (pan_mask == segment_id)
            area = segment_mask.sum()
            if area == 0:
                # cropping or extreme resizing might cause segments to disappear
                continue

            seg_info = {
                'isthing': None,
                'sem': sem_loss[segment_mask].sum(),
                'vote': vote_loss[segment_mask].sum(),
            }
            if isthing and not iscrowd:
                seg_info['isthing'] = True
                stats[segment_id] = seg_info
            elif not isthing:
                assert not iscrowd  # stuff cannot be labelled crowd
                seg_info['isthing'] = False
                stats[segment_id] = seg_info
            elif iscrowd:
                assert isthing
                assert seg_info['vote'] == 0
                seg_info['isthing'] = True
                stats[segment_id] = seg_info
            else:
                raise ValueError('unreachable')
        self.stats = stats
        self.overall = {
            'combined': loss_dict['overall_combined'],
            'sem': loss_dict['overall_sem'],
            'vote': loss_dict['overall_vote']
        }
        sem_total = sum([ seg['sem'] for seg in self.stats.values() ])
        vote_total = sum([ seg['vote'] for seg in self.stats.values() ])
        assert np.allclose(sem_total, self.overall['sem']),\
            '{} vs {}'.format(sem_total, self.overall['sem'])
        assert np.allclose(vote_total, self.overall['vote']),\
            '{} vs {}'.format(vote_total, self.overall['vote'])

    def summarize(self):
        txt = ''
        txt += 'combined loss {:.3f}, sem loss {:.3f}, vote loss {:.3f}\n'.format(
            self.overall['combined'], self.overall['sem'], self.overall['vote']
        )
        txt += self._loss_contribution_breakdown('sem', self.stats)
        txt += self._loss_contribution_breakdown('vote', self.stats)
        print(txt)

    @staticmethod
    def _loss_contribution_breakdown(key, stats):
        '''helper called by summarize only'''
        losses = np.array([ seg[key] for seg in stats.values() ])
        isthing = np.array([ seg['isthing'] for seg in stats.values() ]).astype('bool')
        total = losses.sum()
        thing_losses = losses[isthing]
        stuff_losses = losses[~isthing]
        thing_total = thing_losses.sum()
        stuff_total = stuff_losses.sum()
        txt = '{:<5} total {:>6.3f}'.format(key, total)
        txt += '{:>3} stuff subtot {:>6.3f}/{:>4.1f}%; '.format(
            len(stuff_losses), stuff_total, 100 * stuff_total / total,
        )
        txt += '{:>3} thing subtot {:>6.3f}/{:>4.1f}%; '.format(
            len(thing_losses), thing_total, 100 * thing_total / total,
        )
        txt += '\n'
        return txt


def test():
    from panoptic.entry import Entry
    exp = '/home-nfs/whc/glab/panoptic/new_world/cs_loss_modulate/runs/bless_mask/'
    engine = Entry(exp, debug=False, val_split='train')
    engine.vis(oracle=False)


if __name__ == "__main__":
    test()
