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

This list is non-exhaustive; types or functions may have their own defaults.
"""

import os
import math

__all__ = [
    "DEFT_ALI_SUBDIR",
    "DEFT_CHUNK_SIZE",
    "DEFT_CTM_CHANNEL",
    "DEFT_DEL_COST",
    "DEFT_FEAT_SUBDIR",
    "DEFT_FILE_PREFIX",
    "DEFT_FILE_SUFFIX",
    "DEFT_FRAME_SHIFT_MS",
    "DEFT_HYP_SUBDIR",
    "DEFT_INS_COST",
    "DEFT_PAD_VALUE",
    "DEFT_PDFS_SUBDIR",
    "DEFT_REF_SUBDIR",
    "DEFT_SUB_COST",
    "DEFT_FLOAT_PRINT_PRECISION",
    "DEFT_TEXTGRID_SUFFIX",
    "DEFT_TEXTGRID_TIER_ID",
    "DEFT_TEXTGRID_TIER_NAME",
    "EPS_0",
    "EPS_INF",
    "EPS_NINF",
    "INDEX_PAD_VALUE",
    "TINY",
    "USE_JIT",
]


INDEX_PAD_VALUE = -100
"""The value to pad index-based tensors with

Batched operations often involve variable-width input. This value is used to
right-pad indexed-based tensors with to indicate that this element should be
ignored.

The default value (:obj:`-100`) was chosen to coincide with the PyTorch 1.0 default
for ``ignore_index`` in the likelihood losses
"""

TINY = 1.1754943508222875e-38
"""Smallest representable floating-point integer"""


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

DEFT_FRAME_SHIFT_MS = 10.0
"""The default frame shift in milliseconds for commands"""

DEFT_TEXTGRID_SUFFIX = ".TextGrid"
"""The default suffix indicating TextGrid files for commands"""

DEFT_CHUNK_SIZE = 1000
"""Default number of units to process at once when performing multiprocessing"""


def _cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    cpu_count = os.cpu_count()
    return 0 if cpu_count is None else cpu_count


DEFT_NUM_WORKERS = _cpu_count()
"""Default number of workers when performing multiprocessing"""

DEFT_FILE_PREFIX = ""
"""Default prefix of a torch data file"""

DEFT_FILE_SUFFIX = ".pt"
"""Default suffix of a torch data file"""

DEFT_FLOAT_PRINT_PRECISION = 3
"""Default precision to write floating point values to file with"""

DEFT_CTM_CHANNEL = "A"
"""Default channel to write to CTM files"""

DEFT_TEXTGRID_TIER_ID = 0
"""Default TextGrid tier to read transcripts from"""

DEFT_TEXTGRID_TIER_NAME = "transcript"
"""Default TextGrid tiear to write transcripts to"""

DEFT_FEAT_SUBDIR = "feat"
"""Default subdirectory of a torch data directory containing features"""

DEFT_ALI_SUBDIR = "ali"
"""Default subdirectory of a torch data directory containing alignments"""

DEFT_REF_SUBDIR = "ref"
"""Default subdirectory of a torch data directory containing reference tokens"""

DEFT_PDFS_SUBDIR = "pdfs"
"""Default subdirectory of a torch data directory to write pdfs to"""

DEFT_HYP_SUBDIR = "hyp"
"""Default subdirectory of a torch data directory to write hypothesis tokens to"""

DEFT_PAD_VALUE = 0.0
"""Default value to pad floating-point tensors with"""

DEFT_INS_COST = 1.0
"""Default insertion cost in error rate/distance computations"""

DEFT_DEL_COST = 1.0
"""Default deletion cost in error rate/distance computations"""

DEFT_SUB_COST = 1.0
"""Default substitution cost in error rate/distance computations"""
