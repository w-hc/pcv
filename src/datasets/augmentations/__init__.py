from panoptic.utils import dynamic_load_py_object


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):

        # assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)

        return img, mask


def get_composed_augmentations(aug_list):
    if aug_list is None:
        print("Using No Augmentations")
        return None

    augmentations = []
    for aug_meta in aug_list:
        name = aug_meta.name
        params = aug_meta.params
        aug = dynamic_load_py_object(
            package_name=__name__,
            module_name='augmentations', obj_name=name
        )
        instance = aug(**params)
        augmentations.append(instance)

    return Compose(augmentations)
