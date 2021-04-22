import importlib
import datetime
from fabric.utils.timer import Timer


def dynamic_load_py_object(
    package_name, module_name, obj_name=None
):
    '''Dynamically import an object.
    Assumes that the object lives at module_name.py:obj_name
    If obj_name is not given, assume it shares the same name as the module.
    obj_name is case insensitive e.g. kitti.py/KiTTI is valid
    '''
    if obj_name is None:
        obj_name = module_name
    # use relative import syntax .targt_name
    target_module = importlib.import_module(
        '.{}'.format(module_name), package=package_name
    )
    target_obj = None
    for name, cls in target_module.__dict__.items():
        if name.lower() == obj_name.lower():
            target_obj = cls

    if target_obj is None:
        raise ValueError(
            "No object in {}.{}.py whose lower-case name matches {}".format(
                package_name, module_name, obj_name)
        )

    return target_obj


class CompoundTimer():
    def __init__(self):
        self.data, self.compute = Timer(), Timer()

    def eta(self, curr_step, tot_step):
        remaining_steps = tot_step - curr_step
        avg = self.data.avg + self.compute.avg
        eta_seconds = avg * remaining_steps
        return str(datetime.timedelta(seconds=int(eta_seconds)))

    def __repr__(self):
        return "data avg {:.2f}, compute avg {:.2f}".format(
            self.data.avg, self.compute.avg)
