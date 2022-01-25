# Copyright 2022 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Package constants used throughout pydrobert.torch

If this submodule is imported first in :mod:`pydrobert.torch` and any values are
changed, the changes will propagate to any submodules.
"""

import os

__all__ = ["INDEX_PAD_VALUE", "USE_JIT"]


INDEX_PAD_VALUE = -100
"""The value to pad index-based tensors with

Batched operations often involve variable-width input. This value is used to
right-pad indexed-based tensors with to indicate that this element should be
ignored.

The default value (:obj:`-100`) was chosen to coincide with the PyTorch 1.0 default
for ``ignore_index`` in the likelihood losses
"""


USE_JIT = os.environ.get("PYTORCH_JIT", None) == "1"
"""Whether to eagerly compile functions with JIT

If :obj:`True`, :mod:`pydrobert.torch` compile all functions it can with JIT on import.
Otherwise, if using PyTorch >= 1.8.0, relevant items will be decorated with
:func:`torch.jit.script_if_tracing`. The default is :obj:`True` if and only if the
environment variable ``PYTORCH_JIT=1``.
"""