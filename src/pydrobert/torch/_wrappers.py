# Copyright 2022 Sean Robertson
#
# proxy(...) is from
# https://medium.com/@ppeetteerrs/adding-type-hints-to-pytorch-call-function-30728a972392

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, TypeVar, cast
from inspect import signature

_FUNCTIONAL_DOC_TEMPLATE = """Functional version of {module_name}

This function accepts both the arguments initializing a :class:`{module_name}` instance
and the inputs to its call and outputs the return value of the call.

Parameters
----------
{parameters}
{returns}
See Also
--------
pydrobert.torch.modules.{module_name}
    For a description of what this does, its inputs, and its outputs.
"""


def functional_wrapper(module_name: str):
    def decorator(func):
        sig = signature(func)
        parameters = "\n".join(sig.parameters)
        returns = ""
        if sig.return_annotation is not sig.empty:
            returns = f"""

Returns
-------
{str(sig.return_annotation)}

"""
        func.__doc__ = _FUNCTIONAL_DOC_TEMPLATE.format(
            parameters=parameters, module_name=decorator.__modname, returns=returns
        )
        return func

    decorator.__modname = module_name

    return decorator


C = TypeVar("C", bound=Callable)


def proxy(f: C) -> C:
    return cast(C, lambda self, *x, **y: super(self.__class__, self).__call__(*x, **y))
