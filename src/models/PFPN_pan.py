import torch
import torch.nn as nn
import torch.nn.functional as F
from panoptic.models.panoptic_base import PanopticBase
from panoptic.models.components.resnet import (
    ResNetFeatureExtractor, resnet50, resnet101, resnet50_gn, resnet101_gn
)
from panoptic.models.components.FPN import FPN


primary_backbones = {
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet50_gn': resnet50_gn,
    'resnet101_gn': resnet101_gn
}


class PFPN_pan(PanopticBase):
    def instantiate_network(self, cfg):
        self.net = Net(
            num_classes=self.dset_meta['num_classes'], num_votes=self.pcv.num_votes,
            **cfg.model.params
        )


class Net(nn.Module):
    def __init__(self, num_classes, num_votes, FPN_C, backbone='resnet50', deep_classifier=False):
        super().__init__()
        backbone_f = primary_backbones[backbone]
        fea_extractor = ResNetFeatureExtractor(backbone_f(pretrained=True))
        self.distilled_fpn = FPN(fea_extractor, FPN_C)
        if deep_classifier:
            self.sem_classifier = nn.Sequential(
                nn.Conv2d(FPN_C, 256, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
            )
            self.vote_classifier = nn.Sequential(
                nn.Conv2d(FPN_C, 256, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(256, num_votes, kernel_size=1, bias=True)
            )
        else:
            self.sem_classifier = nn.Conv2d(
                FPN_C, num_classes, kernel_size=1, bias=True
            )
            self.vote_classifier = nn.Conv2d(
                FPN_C, num_votes, kernel_size=1, bias=True
            )
        # self.sem_classifier = nn.Conv2d(
        #     FPN_C, num_classes, kernel_size=1, bias=True
        # )
        # self.vote_classifier = nn.Conv2d(
        #     FPN_C, num_votes, kernel_size=1, bias=True
        # )
        self.loss_module = None

    def forward(self, *inputs):
        """
        If gt is supplied, then compute loss
        """
        x = inputs[0]
        x = self.distilled_fpn(x)
        sem_pred = self.sem_classifier(x)
        vote_pred = self.vote_classifier(x)

        if len(inputs) > 1:
            assert self.loss_module is not None
            loss = self.loss_module(sem_pred, vote_pred, *inputs[1:])
            return loss
        else:
            return sem_pred, vote_pred


if __name__ == '__main__':
    model = Net()
    input = torch.rand(size=(1, 3, 64, 64))
    with torch.no_grad():
        out = model(input)
        print(out.shape)
