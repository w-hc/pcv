"""
The final assembled model is in assembled.py
The rest of the modules contain helpers and modular components
"""
import torch
import torch.nn
from panoptic.utils import dynamic_load_py_object


def get_model_module(model_name):
    return dynamic_load_py_object(__name__, model_name)


def convert_inplace_sync_batchnorm(module, process_group=None):
    r"""Helper function to convert `torch.nn.BatchNormND` layer in the model to
    `torch.nn.SyncBatchNorm` layer.
    Args:
        module (nn.Module): containing module
        process_group (optional): process group to scope synchronization,
    default is the whole world
    Returns:
        The original module with the converted `torch.nn.SyncBatchNorm` layer
    Example::
        >>> # Network with nn.BatchNorm layer
        >>> module = torch.nn.Sequential(
        >>>            torch.nn.Linear(20, 100),
        >>>            torch.nn.BatchNorm1d(100)
        >>>          ).cuda()
        >>> # creating process group (optional)
        >>> # process_ids is a list of int identifying rank ids.
        >>> process_group = torch.distributed.new_group(process_ids)
        >>> sync_bn_module = convert_sync_batchnorm(module, process_group)
    """
    from inplace_abn import InPlaceABNSync
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = InPlaceABNSync(module.num_features,
                                        module.eps, module.momentum,
                                        module.affine,
                                        activation='identity',
                                        group=process_group)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep reuqires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    if isinstance(module, torch.nn.ReLU):
        module_output = torch.nn.ReLU()
    for name, child in module.named_children():
        module_output.add_module(name, convert_inplace_sync_batchnorm(child, process_group))
    del module
    return module_output

def convert_naive_sync_batchnorm(module, process_group=None):
    from detectron2.layers import NaiveSyncBatchNorm
    return convert_xbatchnorm(module, NaiveSyncBatchNorm, process_group=None)

def convert_apex_sync_batchnorm(module, process_group=None):
    from apex.parallel import SyncBatchNorm
    return convert_xbatchnorm(module, SyncBatchNorm, process_group=None)


def convert_xbatchnorm(module, bn_module, process_group=None):
    r"""Helper function to convert `torch.nn.BatchNormND` layer in the model to
    `torch.nn.SyncBatchNorm` layer.
    Args:
        module (nn.Module): containing module
        process_group (optional): process group to scope synchronization,
    default is the whole world
    Returns:
        The original module with the converted `torch.nn.SyncBatchNorm` layer
    Example::
        >>> # Network with nn.BatchNorm layer
        >>> module = torch.nn.Sequential(
        >>>            torch.nn.Linear(20, 100),
        >>>            torch.nn.BatchNorm1d(100)
        >>>          ).cuda()
        >>> # creating process group (optional)
        >>> # process_ids is a list of int identifying rank ids.
        >>> process_group = torch.distributed.new_group(process_ids)
        >>> sync_bn_module = convert_sync_batchnorm(module, process_group)
    """
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = bn_module(module.num_features,
                                                module.eps, module.momentum,
                                                module.affine,
                                                module.track_running_stats,
                                                process_group)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep reuqires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, convert_xbatchnorm(child, bn_module, process_group))
    del module
    return module_output