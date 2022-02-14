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

"""Pytorch modules"""

__all__ = [
    "BeamSearch",
    "ConcatSoftAttention",
    "CTCGreedySearch",
    "CTCPrefixSearch",
    "DenseImageWarp",
    "DotProductSoftAttention",
    "EditDistance",
    "ErrorRate",
    "ExtractableSequentialLanguageModel",
    "GeneralizedDotProductSoftAttention",
    "GlobalSoftAttention",
    "HardOptimalCompletionDistillationLoss",
    "LookupLanguageModel",
    "MinimumErrorRateLoss",
    "MixableSequentialLanguageModel",
    "MultiHeadedAttention",
    "OptimalCompletion",
    "PadVariable",
    "PolyharmonicSpline",
    "PrefixEditDistances",
    "PrefixErrorRates",
    "RandomShift",
    "SequenceLogProbabilities",
    "SequentialLanguageModel",
    "SparseImageWarp",
    "SpecAugment",
    "TimeDistributedReturn",
    "Warp1DGrid",
]

from ._attn import (
    ConcatSoftAttention,
    DotProductSoftAttention,
    GeneralizedDotProductSoftAttention,
    GlobalSoftAttention,
    MultiHeadedAttention,
)
from ._decoding import (
    BeamSearch,
    CTCGreedySearch,
    CTCPrefixSearch,
    SequenceLogProbabilities,
)
from ._img import (
    DenseImageWarp,
    PolyharmonicSpline,
    Warp1DGrid,
    SparseImageWarp,
    PadVariable,
    RandomShift,
    SpecAugment,
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
    HardOptimalCompletionDistillationLoss,
    MinimumErrorRateLoss,
    OptimalCompletion,
    PrefixEditDistances,
    PrefixErrorRates,
)