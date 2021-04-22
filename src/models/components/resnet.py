import torch
from torch import nn
from torchvision.models import resnet50, resnet101, resnext101_32x8d

def set_bn_eval(m):
    """freeze batch norms, per RT's method"""
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        for _p in m.parameters():
            _p.requires_grad = False


def set_conv_stride_to_1(m):
    """
    A ResNet Bottleneck has 2 types of modules potentially with stride 2
    conv2: 3x3 conv used to reduce the spatial dimension of features
    downsample[0]: 1x1 conv used to change shape of original in case the learnt
        residual has incompatible channel or spatial dim
    It does not make sense to give this 1x1 downsample conv dilation 2.
    Hence the if condition testing for conv kernel size
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.stride == (2, 2):
        m.stride = (1, 1)
        if m.kernel_size == (3, 3):
            m.dilation = (2, 2)
            m.padding = (2, 2)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layers = nn.ModuleList([
            layer0,
            resnet.layer1, resnet.layer2,
            resnet.layer3, resnet.layer4
        ])
        self.extractor = nn.Sequential(*self.layers)

    def forward(self, input):
        return self.extractor(input)

    def train(self, mode=True):
        if mode:
            super().train(True)
            # self.apply(set_bn_eval)
        else:
            super().train(False)


def resnet50_gn(pretrained=True):
    model = resnet50(norm_layer=lambda x: nn.GroupNorm(32, x))
    model.load_state_dict(torch.load('/share/data/vision-greg/rluo/model/pytorch-resnet/resnet_gn50-pth.pth'))
    return model


def resnet101_gn(pretrained=True):
    model = resnet101(norm_layer=lambda x: nn.GroupNorm(32, x))
    model.load_state_dict(torch.load('/share/data/vision-greg/rluo/model/pytorch-resnet/resnet_gn101-pth.pth'))
    return model


if __name__ == '__main__':
    from torchvision.models import resnet18
    model = ResNetFeatureExtractor(resnet18(pretrained=False))
    input = torch.rand(size=(1, 3, 64, 64))
    with torch.no_grad():
        out = model(input)
        print(out.shape)
