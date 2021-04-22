from panoptic.utils import dynamic_load_py_object


def get_dataset_module(dset_name):
    dataset = dynamic_load_py_object(
        package_name=__name__, module_name=dset_name
    )
    return dataset
