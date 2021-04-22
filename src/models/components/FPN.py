import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_gn_relu(in_C, out_C, kernel_size, use_relu=True):
    assert kernel_size in (3, 1)
    pad = (kernel_size - 1) // 2
    num_groups = out_C // 16  # note this is hardcoded
    module = [
        nn.Conv2d(in_C, out_C, kernel_size, padding=pad, bias=False),
        # nn.GroupNorm(num_groups, num_channels=out_C),
        nn.BatchNorm2d(out_C)
    ]
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    return nn.Sequential(*module)


class FPN(nn.Module):
    def __init__(self, resnet_extractor, FPN_distill_C):
        super().__init__()
        self.strides = [32, 16, 8, 4]
        self.dims = [2048, 1024, 512, 256]  # assume resnet 50 and above
        self.FPN_feature_C = 256
        self.FPN_distill_C = FPN_distill_C

        self.backbone = resnet_extractor
        self.FPN_create_modules = nn.ModuleList()
        self.FPN_distill_modules = nn.ModuleList()

        for in_dim, stride in zip(self.dims, self.strides):
            self.FPN_create_modules.append(
                nn.ModuleDict({
                    'lateral': conv_gn_relu(
                        in_dim, self.FPN_feature_C,
                        kernel_size=1, use_relu=False
                    ),
                    'refine': conv_gn_relu(
                        self.FPN_feature_C, self.FPN_feature_C,
                        kernel_size=3, use_relu=False
                    )
                })
            )

        for stride in self.strides:
            self.FPN_distill_modules.append(
                self.get_distill_module(
                    stride / 4, self.FPN_feature_C, self.FPN_distill_C
                )
            )

    @staticmethod
    def get_distill_module(upsample_ratio, in_C, out_C):
        levels = math.log(upsample_ratio, 2)
        assert levels.is_integer()
        levels = int(levels)
        if levels == 0:
            return conv_gn_relu(in_C, out_C, kernel_size=3)

        module = []
        for _ in range(levels):
            module.append(
                # maybe use ordered dict?
                nn.Sequential(
                    conv_gn_relu(in_C, out_C, kernel_size=3),
                    nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                )
            )
            in_C = out_C
        return nn.Sequential(*module)

    def collect_resnet_features(self, x):
        acc = []
        for layer in self.backbone.layers:
            x = layer(x)
            acc.append(x)
        acc = acc[1:]  # throw away layer0 output
        return acc[::-1]  # high level features first

    def create_FPN(self, res_features):
        FPN_features = []
        prev = None
        for tsr, curr_module in zip(res_features, self.FPN_create_modules):
            tsr = curr_module['lateral'](tsr)
            if prev is not None:
                prev = F.interpolate(prev, scale_factor=2, mode='nearest')
                tsr = tsr + prev
            prev = tsr
            refined = curr_module['refine'](tsr)
            FPN_features.append(refined)
        return FPN_features

    def distill_FPN(self, FPN_features):
        acc = []
        for tsr, curr_module in zip(FPN_features, self.FPN_distill_modules):
            tsr = curr_module(tsr)
            acc.append(tsr)
        return sum(acc)

    def forward(self, x):
        res_features = self.collect_resnet_features(x)
        FPN_features = self.create_FPN(res_features)
        final = self.distill_FPN(FPN_features)
        return final


if __name__ == '__main__':
    from panoptic.models.components.resnet import ResNetFeatureExtractor
    from torchvision.models import resnet50
    extractor = ResNetFeatureExtractor(resnet50(pretrained=False))
    fpn = FPN(extractor)
    input = torch.rand(size=(1, 3, 64, 64))
    with torch.no_grad():
        output = fpn(input)
        print(output.shape)
