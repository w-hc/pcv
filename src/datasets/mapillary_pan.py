from .base import PanopticSeg


class Mapillary_Pan(PanopticSeg):
    aspect_grouping = False

    def __init__(self, *args, **kwargs):
        super().__init__(name='mapillary', *args, **kwargs)
        self.meta['stuff_pred_thresh'] = 2048
