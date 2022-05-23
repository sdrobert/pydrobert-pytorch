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
import math

__all__ = ["INDEX_PAD_VALUE", "USE_JIT", "EPS_NINF", "EPS_INF", "EPS_0"]


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

EPS_NINF = math.log(1.1754943508222875e-38) / 2
"""A small enough value in log space that exponentiating it is very close to zero

This number is sometimes used in place of -infinity in log-space values to avoid
masking. Increasing it will decrease the accuracy of computations, but may avoid NaNs.
"""

EPS_0 = math.log1p(-2 * 1.1920928955078125e-07)
"""A large enough value in log space that exponentiating it is very close to 1

This number is sometimes used in place of 0 in log-space values to avoid masking.
Decreasing it will decrease the accuracy of computations, but may avoid NaNs.
"""

EPS_INF = math.log(3.4028234663852886e38) / 2
"""A large enough value in log space that exponentiating it is near infinity

This number is sometimes used in place of infinity in log-space values to avoid masking.
Decreasing it will decrease the accuracy of computations, but may avoid NaNs.
"""
