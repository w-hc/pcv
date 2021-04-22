import torch
import torch.nn as nn
import torch.nn.functional as F
from panoptic.models.panoptic_base import PanopticBase, AbstractNet
from panoptic.models.components.resnet import (
    ResNetFeatureExtractor, resnet50, resnet101, resnet50_gn, resnet101_gn
)
from panoptic.models.components.FPN import FPN
import detectron2.config
import detectron2.modeling
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.semantic_seg import build_sem_seg_head
from detectron2.checkpoint import DetectionCheckpointer

from panoptic import cfg

primary_backbones = {
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet50_gn': resnet50_gn,
    'resnet101_gn': resnet101_gn
}


class PFPN_d2(PanopticBase):
    def instantiate_network(self, cfg):
        self.net = Net(
            num_classes=self.dset_meta['num_classes'], num_votes=self.pcv.num_votes,
            **cfg.model.params
        )


class Net(AbstractNet):
    def __init__(self, num_classes, num_votes,
                 fix_bn=True,
                 freeze_at=2,
                 norm='GN',
                 fpn_norm='',
                 conv_dims=128,
                 **kwargs):  # FPN_C, backbone='resnet50'):
        super().__init__()
        d2_cfg = detectron2.config.get_cfg()
        d2_cfg.merge_from_file('/home-nfs/whc/glab/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml')
        if fix_bn:
            d2_cfg.MODEL.RESNETS.NORM = "FrozenBN"
        else:
            d2_cfg.MODEL.RESNETS.NORM = "BN"
        d2_cfg.MODEL.BACKBONE.FREEZE_AT = freeze_at

        d2_cfg.MODEL.FPN.NORM = 'BN'
        self.backbone = build_backbone(d2_cfg)

        d2_cfg.MODEL.SEM_SEG_HEAD.NORM = norm
        d2_cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = conv_dims

        d2_cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
        self.sem_seg_head = build_sem_seg_head(d2_cfg, self.backbone.output_shape())
        self.sem_classifier = self.sem_seg_head.predictor
        del self.sem_seg_head.predictor

        d2_cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_votes
        tmp = build_sem_seg_head(d2_cfg, self.backbone.output_shape())
        self.vote_classifier = tmp.predictor
        assert cfg.data.dataset.params.caffe_mode == True

        checkpointer = DetectionCheckpointer(self)
        checkpointer.load(d2_cfg.MODEL.WEIGHTS)

    def stage1(self, x):
        return self.backbone(x)

    def stage2(self, features):
        # copy from detectron2/modeling/meta_arch/semantic_seg.py
        # why? because segFPNHead doesn't accept training==True and targets==None
        for i, f in enumerate(self.sem_seg_head.in_features):
            if i == 0:
                x = self.sem_seg_head.scale_heads[i](features[f])
            else:
                x = x + self.sem_seg_head.scale_heads[i](features[f])
        # x = self.sem_seg_head.predictor(x)
        return x, x
        # x = F.interpolate(x, scale_factor=self.common_stride, mode="bilinear", align_corners=False)


if __name__ == '__main__':
    model = Net(10,10)
    model.eval()
    input = torch.rand(size=(1, 3, 224, 224))
    with torch.no_grad():
        out = model(input)
        print(out[0].shape, out[1].shape)
