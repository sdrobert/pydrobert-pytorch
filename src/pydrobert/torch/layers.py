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

import warnings

warnings.warn(
    "pydrobert.torch.layers is deprecated. Use pydrobert.torch.functional for "
    "functions and pydrobert.torch.modules for modules",
    DeprecationWarning,
    2,
)

from ._attn import (
    ConcatSoftAttention,
    DotProductSoftAttention,
    GeneralizedDotProductSoftAttention,
    GlobalSoftAttention,
    MultiHeadedAttention,
)
from ._decoding import BeamSearch, CTCPrefixSearch, SequenceLogProbabilities
from ._img import (
    DenseImageWarp,
    PadVariable,
    PolyharmonicSpline,
    random_shift,
    RandomShift,
    SparseImageWarp,
    spec_augment_apply_parameters,
    spec_augment_draw_parameters,
    spec_augment,
    SpecAugment,
    Warp1DGrid,
)
from ._lm import (
    ExtractableSequentialLanguageModel,
    LookupLanguageModel,
    MixableSequentialLanguageModel,
    SequentialLanguageModel,
)
from ._rl import TimeDistributedReturn
from ._string import (
    EditDistance,
    ErrorRate,
    hard_optimal_completion_distillation_loss,
    HardOptimalCompletionDistillationLoss,
    minimum_error_rate_loss,
    MinimumErrorRateLoss,
    OptimalCompletion,
    PrefixEditDistances,
    PrefixErrorRates,
)
