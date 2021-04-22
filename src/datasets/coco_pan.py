from .base import PanopticSeg


class COCO_Pan(PanopticSeg):
    aspect_grouping = True

    def __init__(self, *args, **kwargs):
        super().__init__(name='coco', *args, **kwargs)
        self.meta['stuff_pred_thresh'] = 4096
