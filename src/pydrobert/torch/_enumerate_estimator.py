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

import torch

from ._estimators import Estimator, FunctionOnSample


class EnumerateEstimator(Estimator):
    r"""Calculate expectation exactly by enumerating the support of the distribution

    An unbiased, zero-variance "estimate" of an expectation over a discrete variable
    may be calculated brute force by enumerating the support and taking the product of
    function values with their probabilities under the distribution.

    .. math::

        v = \mathbb{E}_{b \sim P}[f(b)] = \sum_b P(b) f(b).
    
    When called, the instance does just that.

    Parameters
    ----------
    proposal
        The distribution over which the expectation is taken, :math:`P`. Must be able to
        enumerate its support through
        :func:`torch.distributions.Distribution.enumerate_support`
        (``proposal.has_enumerate_support == True``).
    func
    is_log
    
    Returns
    -------
    v : torch.Tensor

    Warnings
    --------
    The call may be both compute- and memory-intensive, depending on the size of the
    support.
    """

    return_log: bool

    def __init__(
        self,
        proposal: torch.distributions.distribution.Distribution,
        func: FunctionOnSample,
        is_log: bool = False,
    ) -> None:
        if not proposal.has_enumerate_support:
            raise ValueError(
                "proposal must be able to enumerate its support "
                "(proposal.has_enumerate_support == True)"
            )
        super().__init__(proposal, func, is_log)

    def __call__(self) -> torch.Tensor:
        b = self.proposal.enumerate_support()
        log_pb = self.proposal.log_prob(b)
        fb = self.func(b)
        if self.is_log:
            v = fb + log_pb
            v = v.logsumexp(0)
        else:
            v = (fb * log_pb.exp()).sum(0)
        return v

