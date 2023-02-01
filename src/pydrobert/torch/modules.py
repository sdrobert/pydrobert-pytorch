# Copyright 2023 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom PyTorch modules

Notes
-----
To document :class:`torch.nn.Module` subclasses, we add a special heading called "Call
Parameters" to the docstring which, along with "Returns", specify the signature of the
module's :func:`__call__` method. The header "Parameters" refers to what values the
module are initialized with. The general usage pattern is:

>>> module = Module(*params)
>>> returns = module(*call_params)
"""

__all__ = [
    "BeamSearch",
    "ChunkBySlices",
    "ChunkTokenSequencesBySlices",
    "ConcatSoftAttention",
    "CTCGreedySearch",
    "CTCPrefixSearch",
    "DenseImageWarp",
    "DotProductSoftAttention",
    "EditDistance",
    "ErrorRate",
    "ExtractableSequentialLanguageModel",
    "FeatureDeltas",
    "FillAfterEndOfSequence",
    "GeneralizedDotProductSoftAttention",
    "GlobalSoftAttention",
    "GumbelOneHotCategoricalRebarControlVariate",
    "HardOptimalCompletionDistillationLoss",
    "LogisticBernoulliRebarControlVariate",
    "LookupLanguageModel",
    "MeanVarianceNormalization",
    "MinimumErrorRateLoss",
    "MixableSequentialLanguageModel",
    "MultiHeadedAttention",
    "OptimalCompletion",
    "PadMaskedSequence",
    "PadVariable",
    "PolyharmonicSpline",
    "PrefixEditDistances",
    "PrefixErrorRates",
    "RandomShift",
    "RandomWalk",
    "SequenceLogProbabilities",
    "SequentialLanguageModel",
    "SliceSpectData",
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
    RandomWalk,
    SequenceLogProbabilities,
)
from ._feats import (
    ChunkTokenSequencesBySlices,
    FeatureDeltas,
    MeanVarianceNormalization,
    SliceSpectData,
)
from ._img import (
    DenseImageWarp,
    PolyharmonicSpline,
    Warp1DGrid,
    SparseImageWarp,
    RandomShift,
    SpecAugment,
)
from ._lm import (
    ExtractableSequentialLanguageModel,
    LookupLanguageModel,
    MixableSequentialLanguageModel,
    SequentialLanguageModel,
)
from ._mc import (
    LogisticBernoulliRebarControlVariate,
    GumbelOneHotCategoricalRebarControlVariate,
)
from ._pad import ChunkBySlices, PadVariable, PadMaskedSequence
from ._rl import TimeDistributedReturn
from ._string import (
    EditDistance,
    ErrorRate,
    FillAfterEndOfSequence,
    HardOptimalCompletionDistillationLoss,
    MinimumErrorRateLoss,
    OptimalCompletion,
    PrefixEditDistances,
    PrefixErrorRates,
)
