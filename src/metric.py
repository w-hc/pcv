import os.path as osp
import numpy as np
import torch
from tabulate import tabulate
from panoptic.pan_eval import pq_compute_single_img, PQStat


class Metric(object):
    def __init__(self, num_classes, trainId_2_catName=None):
        """
        Args:
            window_size: the number of batch of visuals stated here

        All metrics are computed from a confusion matrix that aggregates pixel
        count across all images. This evaluation scheme only has a global
        view of all the pixels, and disregards image as a unit.
        "mean" always refers to averaging across pixel semantic classes.
        """
        self.num_classes = num_classes
        self.trainId_2_catName = trainId_2_catName
        self.init_state()

    def init_state(self):
        """caller may use it to reset the metric"""
        self.scores = dict()
        self.confusion_matrix = np.zeros(
            shape=(self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(self, pred, gt):
        """
        Args:
            pred: [N, H, W] torch tsr or ndarray
            gt:   [N, H, W] torch tsr or ndarray
        """
        if len(pred.shape) == 4:
            pred = pred.argmax(dim=1)
        assert pred.shape == gt.shape
        if isinstance(pred, torch.Tensor):
            pred, gt = pred.cpu().numpy(), gt.cpu().numpy()
        hist = self.fast_hist(pred, gt, self.num_classes)
        self.confusion_matrix += hist
        self.scores = self.compute_scores(
            self.confusion_matrix, self.trainId_2_catName
        )

    @staticmethod
    def fast_hist(pred, gt, num_classes):
        assert pred.shape == gt.shape
        valid_mask = (gt >= 0) & (gt < num_classes)
        hist = np.bincount(
            num_classes * gt[valid_mask] + pred[valid_mask],
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)
        return hist

    @staticmethod
    def compute_scores(hist, trainId_2_catName):
        res = dict()
        num_classes = hist.shape[0]

        # per class statistics
        gt_freq = hist.sum(axis=1)
        pd_freq = hist.sum(axis=0)  # pred frequency
        intersect = np.diag(hist)
        union = gt_freq + pd_freq - intersect
        iou = intersect / union

        cls_names = [
            trainId_2_catName[inx] for inx in range(num_classes)
        ] if trainId_2_catName is not None else range(num_classes)

        details = dict()
        for inx, name in enumerate(cls_names):
            details[name] = {
                'gt_freq': gt_freq[inx],
                'pd_freq': pd_freq[inx],
                'intersect': intersect[inx],
                'union': union[inx],
                'iou': iou[inx]
            }
        res['details'] = details

        # aggregate statistics
        pix_acc = intersect.sum() / hist.sum()
        m_iou = np.nanmean(iou)
        freq = gt_freq / gt_freq.sum()
        # masking to avoid potential nan in per cls iou
        fwm_iou = (freq[freq > 0] * iou[freq > 0]).sum()
        del freq
        res['pix_acc'] = pix_acc
        res['m_iou'] = m_iou
        res['fwm_iou'] = fwm_iou

        return res

    def __repr__(self):
        return repr(self.scores)

    def __str__(self):
        return self.display(self.scores)

    @staticmethod
    def display(src_dict):
        """Only print out scalar metric like mIoU in a nice tabular form
        Detailed per cls info, etc, are withheld for clear presentation
        """
        to_display = dict()
        for k, v in src_dict.items():
            if isinstance(v, dict):
                continue  # ignore those which cannot be tabulated
            to_display[k] = [v]
        table = tabulate(
            to_display,
            headers='keys', tablefmt='fancy_grid',
            floatfmt=".3f", numalign='decimal'
        )
        return str(table)

    # these save and load functions are ugly cuz they are tied to the
    # infrastructure. Re-write them later.
    def save(self, epoch_or_fname, manager):
        assert manager is not None
        state = self.scores
        # now save the acc with manager
        if isinstance(epoch_or_fname, int):
            epoch = epoch_or_fname
            manager.save(epoch, state)
        else:
            fname = epoch_or_fname
            save_path = osp.join(manager.root, fname)
            manager.save_f(state, save_path)

    def load(self, state):
        """Assume that the state is already read by the caller
        Args:
            state: dict with fields 'scores' and 'visuals'
        """
        self.scores = state


class PanMetric(Metric):
    """
    This is a metric that evaluates predictions from both heads as pixel-wise
    classification
    """
    def __init__(self, num_classes, num_votes, trainId_2_catName):
        self.num_votes = num_votes
        super().__init__(num_classes, trainId_2_catName)

    def init_state(self):
        self.scores = dict()
        self.sem_confusion = np.zeros(
            shape=(self.num_classes, self.num_classes), dtype=np.int64
        )
        self.vote_confusion = np.zeros(
            shape=(self.num_votes, self.num_votes), dtype=np.int64
        )

    def update(self, sem_pred, vote_pred, sem_gt, vote_gt):
        """
        Args:
            sem_pred:  [N, H, W] torch tsr or ndarray
            vote_pred: [N, H, W] torch tsr or ndarray
            sem_gt:    [N, H, W] torch tsr or ndarray
            vote_gt:   [N, H, W] torch tsr or ndarray
        """
        self._update_pair(
            sem_pred, sem_gt, 'sem', self.sem_confusion, self.num_classes)
        self._update_pair(
            vote_pred, vote_gt, 'vote', self.vote_confusion, self.num_votes)

    def _update_pair(self, pred, gt, key, confusion_matrix, num_cats):
        if len(pred.shape) == 4:
            pred = pred.argmax(dim=1)
        if isinstance(pred, torch.Tensor):
            pred, gt = pred.cpu().numpy(), gt.cpu().numpy()
        hist = self.fast_hist(pred, gt, num_cats)
        confusion_matrix += hist
        trainId_2_catName = self.trainId_2_catName if key == 'sem' else None
        self.scores[key] = self.compute_scores(
            confusion_matrix, trainId_2_catName
        )

    def __str__(self):
        sem_str = self.display(self.scores['sem'])
        vote_str = self.display(self.scores['vote'])
        combined = "sem: \n{} \nvote: \n{}".format(sem_str, vote_str)
        return combined


class PQMetric():
    def __init__(self, dset_meta):
        self.cats = dset_meta['cats']
        self.score = PQStat()
        self.metrics = [("All", None), ("Things", True), ("Stuff", False)]
        self.results = {}

    def update(self, gt_ann, gt, pred_ann, pred):
        assert gt.shape == pred.shape
        stat = pq_compute_single_img(self.cats, gt_ann, gt, pred_ann, pred)
        self.score += stat

    def state_dict(self):
        self.aggregate_results()
        return self.results

    def aggregate_results(self):
        for name, isthing in self.metrics:
            self.results[name], per_class_results = self.score.pq_average(
                self.cats, isthing=isthing
            )
            if name == 'All':
                self.results['per_class'] = per_class_results

    def __str__(self):
        self.aggregate_results()
        headers = [ m[0] for m in self.metrics ]
        keys = ['pq', 'sq', 'rq']
        data = []
        for tranche in headers:
            row = [100 * self.results[tranche][k] for k in keys]
            row = [tranche] + row + [self.results[tranche]['n']]
            data.append(row)
        table = tabulate(
            tabular_data=data,
            headers=([''] + keys + ['n_cats']),
            floatfmt=".2f", tablefmt='fancy_grid',
        )
        return table
