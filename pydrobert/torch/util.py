'''Utility functions'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


def optimizer_to(optimizer, to):
    '''Move tensors in an optimizer to another device

    This function traverses the state dictionary of an `optimizer` and pushes
    any tensors it finds to the device implied by `to` (as per the semantics
    of ``torch.tensor.to``)
    '''
    state_dict = optimizer.state_dict()
    key_stack = [(x,) for x in state_dict.keys()]
    new_device = False
    while key_stack:
        last_dict = None
        val = state_dict
        keys = key_stack.pop()
        for key in keys:
            last_dict = val
            val = val[key]
        if isinstance(val, torch.Tensor):
            try:
                last_dict[keys[-1]] = val.to(to)
                if last_dict[keys[-1]].device != val.device:
                    new_device = True
            except Exception as e:
                print(e)
                pass
        elif hasattr(val, 'keys'):
            key_stack += [keys + (x,) for x in val.keys()]
    if new_device:
        optimizer.load_state_dict(state_dict)
