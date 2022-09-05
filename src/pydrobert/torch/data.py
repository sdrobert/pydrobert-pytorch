# Copyright 2022 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classes and functions related to storing/retrieving speech data"""

import functools
import warnings

from ._datasets import (
    ContextWindowDataParams,
    ContextWindowDataSet,
    extract_window,
    SpectDataParams,
    SpectDataSet,
    validate_spect_data_set,
)
from ._dataloaders import (
    BucketBatchSampler,
    context_window_seq_to_batch,
    ContextWindowDataLoader,
    ContextWindowDataLoaderParams,
    ContextWindowEvaluationDataLoader,  # deprecated
    ContextWindowTrainingDataLoader,  # deprecated
    DataLoaderParams,
    DistributableSequentialSampler,
    DynamicLengthDataLoaderParams,
    EpochRandomSampler,
    spect_seq_to_batch,
    SpectDataLoader,
    SpectDataLoaderParams,
    SpectEvaluationDataLoader,  # deprecated
    SpectTrainingDataLoader,  # deprecated
)
from ._parsing import (
    parse_arpa_lm,
    read_ctm,
    read_trn_iter,
    read_trn,
    write_ctm,
    write_trn,
    transcript_to_token,
    token_to_transcript,
)

__all__ = [
    "BucketBatchSampler",
    "context_window_seq_to_batch",
    "ContextWindowDataLoaderParams",
    "ContextWindowDataParams",
    "ContextWindowDataSet",
    "DataLoaderParams",
    "DistributableSequentialSampler",
    "DynamicLengthDataLoaderParams",
    "EpochRandomSampler",
    "extract_window",
    "parse_arpa_lm",
    "read_ctm",
    "read_trn_iter",
    "read_trn",
    "spect_seq_to_batch",
    "SpectDataLoaderParams",
    "SpectDataParams",
    "SpectDataSet",
    "SpectDataLoader",
    "token_to_transcript",
    "transcript_to_token",
    "validate_spect_data_set",
    "write_ctm",
    "write_trn",
]


def import_and_deprecate(cls):

    from . import _dataloaders

    old_name = cls.__name__
    new_name = old_name.replace("DataSet", "DataLoader")
    cls = getattr(_dataloaders, new_name)

    @functools.wraps(cls)
    def wraps(*args, **kwargs):
        warnings.warn(
            f"The name '{wraps.__old}' is deprecated. Please swith to '{wraps.__new}'",
            DeprecationWarning,
        )
        return wraps.__cls(*args, **kwargs)

    wraps.__old = old_name
    wraps.__new = new_name
    wraps.__cls = cls

    return wraps


@import_and_deprecate
class DataSetParams:
    pass


@import_and_deprecate
class SpectDataSetParams:
    pass


@import_and_deprecate
class ContextWindowDataSetParams:
    pass
