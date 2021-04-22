from panoptic.datasets.base import SemanticSeg
import os.path as osp


class Cityscapes_Sem(SemanticSeg):
    def __init__(self, split, transforms=None):
        super().__init__(name='cityscapes', split=split, transforms=transforms)
        self.vanilla_lbl_root = osp.join(
            '/share/data/vision-greg/cityscapes/gtFine', split
        )
