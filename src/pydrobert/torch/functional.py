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

"""Pytorch functions"""

from ._combinatorics import (
    binomial_coefficient,
    enumerate_vocab_sequences,
    enumerate_binary_sequences,
    enumerate_binary_sequences_with_cardinality,
    simple_random_sampling_without_replacement,
)
from ._decoding import (
    beam_search_advance,
    ctc_greedy_search,
    ctc_prefix_search_advance,
    random_walk_advance,
    sequence_log_probs,
)
from ._feats import mean_var_norm
from ._img import (
    dense_image_warp,
    pad_variable,
    polyharmonic_spline,
    random_shift,
    sparse_image_warp,
    spec_augment_apply_parameters,
    spec_augment_draw_parameters,
    spec_augment,
    warp_1d_grid,
)
from ._rl import time_distributed_return
from ._string import (
    edit_distance,
    error_rate,
    fill_after_eos,
    hard_optimal_completion_distillation_loss,
    minimum_error_rate_loss,
    optimal_completion,
    prefix_edit_distances,
    prefix_error_rates,
)


__all__ = [
    "beam_search_advance",
    "binomial_coefficient",
    "ctc_greedy_search",
    "ctc_prefix_search_advance",
    "dense_image_warp",
    "edit_distance",
    "enumerate_binary_sequences_with_cardinality",
    "enumerate_binary_sequences",
    "enumerate_vocab_sequences",
    "error_rate",
    "fill_after_eos",
    "hard_optimal_completion_distillation_loss",
    "mean_var_norm",
    "minimum_error_rate_loss",
    "optimal_completion",
    "pad_variable",
    "polyharmonic_spline",
    "prefix_edit_distances",
    "prefix_error_rates",
    "random_shift",
    "random_walk_advance",
    "sequence_log_probs",
    "simple_random_sampling_without_replacement",
    "sparse_image_warp",
    "spec_augment_apply_parameters",
    "spec_augment_draw_parameters",
    "spec_augment",
    "time_distributed_return",
    "warp_1d_grid",
]
