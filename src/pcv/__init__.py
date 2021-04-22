from ..utils import dynamic_load_py_object


def get_pcv(pcv_cfg):
    module = dynamic_load_py_object(__name__, pcv_cfg.name)
    return module()
