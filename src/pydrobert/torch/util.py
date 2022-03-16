# Copyright 2022 Sean Robertson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

warnings.warn(
    "pydrobert.torch.util is deprecated. Use pydrobert.torch.functional for "
    "functions. parse_arpa_lm has been moved to pydrobert.torch.data.",
    DeprecationWarning,
    2,
)

import warnings

from .functional import (
    beam_search_advance,
    ctc_greedy_search,
    ctc_prefix_search_advance,
    dense_image_warp,
    edit_distance,
    error_rate,
    optimal_completion,
    pad_variable,
    polyharmonic_spline,
    prefix_edit_distances,
    prefix_error_rates,
    random_walk_advance,
    sequence_log_probs,
    sparse_image_warp,
    time_distributed_return,
    warp_1d_grid,
)
from .data import parse_arpa_lm
