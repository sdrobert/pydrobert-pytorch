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

"""PyTorch distributions and interfaces

Warnings
--------
Distributions cannot be JIT scripted or traced.
"""

from ._combinatorics import (
    BinaryCardinalityConstraint,
    SimpleRandomSamplingWithoutReplacement,
)
from ._decoding import SequentialLanguageModelDistribution, TokenSequenceConstraint
from ._straight_through import (
    ConditionalStraightThrough,
    Density,
    GumbelOneHotCategorical,
    LogisticBernoulli,
    StraightThrough,
)

__all__ = [
    "BinaryCardinalityConstraint",
    "ConditionalStraightThrough",
    "Density",
    "GumbelOneHotCategorical",
    "LogisticBernoulli",
    "SequentialLanguageModelDistribution",
    "SimpleRandomSamplingWithoutReplacement",
    "StraightThrough",
    "TokenSequenceConstraint",
]
