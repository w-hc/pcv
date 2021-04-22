import numpy as np
from .base import PanopticSeg


class Cityscapes_Pan(PanopticSeg):
    aspect_grouping = False

    def __init__(self, *args, **kwargs):
        super().__init__(name='cityscapes', *args, **kwargs)
        self.meta['stuff_pred_thresh'] = 2048
