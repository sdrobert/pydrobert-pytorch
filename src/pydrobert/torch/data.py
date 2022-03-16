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

from ._datasets import (
    ContextWindowDataParams,
    ContextWindowDataSet,
    extract_window,
    SpectDataParams,
    SpectDataSet,
    validate_spect_data_set,
)
from ._dataloaders import (
    context_window_seq_to_batch,
    ContextWindowDataSetParams,
    ContextWindowEvaluationDataLoader,
    ContextWindowTrainingDataLoader,
    DataSetParams,
    EpochRandomSampler,
    spect_seq_to_batch,
    SpectDataSetParams,
    SpectEvaluationDataLoader,
    SpectTrainingDataLoader,
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
    "context_window_seq_to_batch",
    "ContextWindowDataParams",
    "ContextWindowDataSet",
    "ContextWindowDataSetParams",
    "ContextWindowEvaluationDataLoader",
    "ContextWindowTrainingDataLoader",
    "DataSetParams",
    "EpochRandomSampler",
    "extract_window",
    "parse_arpa_lm",
    "read_ctm",
    "read_trn_iter",
    "read_trn",
    "spect_seq_to_batch",
    "SpectDataParams",
    "SpectDataSet",
    "SpectDataSetParams",
    "SpectEvaluationDataLoader",
    "SpectTrainingDataLoader",
    "token_to_transcript",
    "transcript_to_token",
    "validate_spect_data_set",
    "write_ctm",
    "write_trn",
]
