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

from typing import Optional
import torch

from ._estimators import FunctionOnSample


class EnumerateEstimator:
    r"""Calculate expectation exactly by enumerating the support of the distribution

    An unbiased, zero-variance "estimate" of an expectation over a discrete variable
    may be calculated brute force by enumerating the support and taking the product of
    function values with their probabilities under the distribution.

    .. math::

        \mathbb{E}_{b \sim P}[f(b)] = \sum_b P(b) f(b).
    
    When the instance's :func:`estimate` function is called, it does just that.

    Parameters
    ----------
    proposal : torch.distributions.Distribution
        The distribution :math:`P`. Must be able to enumerate its support through
        ``proposal.enumerate_support()`` (``proposal.has_enumerate_support == True``).
    func : FunctionOnSample
        The function :math:`f`.
    is_log : bool, optional
        If :obj:`True`, `func` is :math:`\log f`. Results may be more numerically
        stable than if `func` were pre-exponentiated.
    return_log : bool, optional
        If :obj:`True`, the log of the expectation is returned instead of the
        expectation. Results may be more numerically stable if ``return_log == is_log``.
        If unset, `return_log` defaults to the value of `is_log`.
    
    Warnings
    --------
    :func:`estimate` may be both compute- and memory-intensive, depending on the size of
    the support.
    """

    proposal: torch.distributions.Distribution
    func: FunctionOnSample
    is_log: bool
    return_log: bool

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        func: FunctionOnSample,
        is_log: bool = False,
        return_log: Optional[bool] = None,
    ) -> None:
        if not proposal.has_enumerate_support:
            raise ValueError(
                "proposal must be able to enumerate its support "
                "(proposal.has_enumerate_support == True)"
            )
        self.proposal = proposal
        self.func = func
        self.is_log = is_log
        self.return_log = is_log if return_log is None else return_log

    def estimate(self) -> torch.Tensor:
        b = self.proposal.enumerate_support()
        log_pb = self.proposal.log_prob(b)
        fb = self.func(b)
        if self.is_log:
            v = fb + log_pb
            v = v.logsumexp(0) if self.return_log else v.exp().sum(0)
        elif self.return_log:
            v = (fb.log() + log_pb).logsumexp(0)
        else:
            v = (fb * log_pb.exp()).sum(0)
        return v

