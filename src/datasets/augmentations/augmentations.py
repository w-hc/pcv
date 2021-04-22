import math
import random

# import torchvision.transforms as tv_trans
from torchvision.transforms import ColorJitter as tv_color_jitter
import torchvision.transforms.functional as tv_f
from PIL import Image


def _pad_to_sizes(mask, tw, th):
    w, h = mask.size
    right = max(tw - w, 0)
    bottom = max(th - h, 0)
    mask = tv_f.pad(mask, fill=0, padding=(0, 0, right, bottom))
    return mask


class RandomCrop():
    def __init__(self, size):
        if isinstance(size, (tuple, list)):
            assert len(size) == 2
            self.size = size
        else:
            self.size = [size, size]

    def __call__(self, img, mask):
        assert img.size == mask.size
        tw, th = self.size

        img = _pad_to_sizes(img, tw, th)
        mask = _pad_to_sizes(mask, tw, th)
        w, h = img.size

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        square = (x1, y1, x1 + tw, y1 + th)
        return img.crop(square), mask.crop(square)


class RandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                mask.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return img, mask


class Scale():
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, img, mask):
        w, h = img.size
        ratio = self.ratio
        target_size = (int(ratio * w), int(ratio * h))
        img = img.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)
        return (img, mask)


class Resize():
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        target_size = self.size
        img = img.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)
        return (img, mask)


class RandomSized():
    def __init__(self, jitter=(0.5, 2)):
        self.jitter = jitter

    def __call__(self, img, mask):
        assert img.size == mask.size

        scale = random.uniform(self.jitter[0], self.jitter[1])

        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = (
            img.resize((w, h), Image.BILINEAR),
            mask.resize((w, h), Image.NEAREST),
        )

        return img, mask


class RoundToMultiple():
    def __init__(self, stride, method='pad'):
        assert method in ('pad', 'resize')
        self.stride = stride
        self.method = method

    def __call__(self, img, mask):
        assert img.size == mask.size
        stride = self.stride
        if stride > 0:
            w, h = img.size
            w = int(math.ceil(w / stride) * stride)
            h = int(math.ceil(h / stride) * stride)
            if self.method == 'pad':
                img = _pad_to_sizes(img, w, h)
                mask = _pad_to_sizes(mask, w, h)
            else:
                img = img.resize((w, h), Image.BILINEAR)
                mask = mask.resize((w, h), Image.NEAREST)
        return img, mask


class COCOResize():
    '''
    adapted from maskrcnn_benchmark/data/transforms/transforms.py
    '''
    def __init__(self, min_size, max_size, round_to_divisble):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        assert round_to_divisble in ('pad', 'resize')
        self.min_size = min_size
        self.max_size = max_size
        self.round_to_divisble = round_to_divisble

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (ow, oh)

    def __call__(self, img, mask):
        assert img.size == mask.size
        size = self.get_size(img.size)
        img = img.resize(size)
        mask = mask.resize(size)
        padder = RoundToMultiple(32, self.round_to_divisble)
        img, mask = padder(img, mask)
        return img, mask


class ColorJitter():
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.tv_jitter = tv_color_jitter(
            brightness, contrast, saturation, hue
        )

    def __call__(self, img, mask):
        assert img.size == mask.size
        img = self.tv_jitter(img)
        return img, mask
