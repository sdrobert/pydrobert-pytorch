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

"""Utility functions which are neither pytorch modules nor functions"""

import functools
import warnings

__all__ = ["parse_arpa_lm"]

from ._lm import parse_arpa_lm


def import_and_deprecate(func):
    from . import functional

    name = func.__name__
    func = getattr(functional, name)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            "pytorch function access through pydrobert.torch.util is deprecated. "
            "Use pydrobert.torch.functional instead"
        )
        return wrapper.__func(*args, **kwargs)

    wrapper.__func = func
    return wrapper


@import_and_deprecate
def beam_search_advance():
    pass


@import_and_deprecate
def ctc_greedy_search():
    pass


@import_and_deprecate
def ctc_prefix_search_advance():
    pass


@import_and_deprecate
def dense_image_warp():
    pass


@import_and_deprecate
def edit_distance():
    pass


@import_and_deprecate
def error_rate():
    pass


@import_and_deprecate
def optimal_completion():
    pass


@import_and_deprecate
def pad_variable():
    pass


@import_and_deprecate
def polyharmonic_spline():
    pass


@import_and_deprecate
def prefix_edit_distances():
    pass


@import_and_deprecate
def prefix_error_rates():
    pass


@import_and_deprecate
def random_walk_advance():
    pass


@import_and_deprecate
def sequence_log_probs():
    pass


@import_and_deprecate
def sparse_image_warp():
    pass


@import_and_deprecate
def time_distributed_return():
    pass


@import_and_deprecate
def warp_1d_grid():
    pass
