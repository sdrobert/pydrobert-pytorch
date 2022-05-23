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

import abc

from typing import Callable

import torch

from ._compat import TypeAlias

FunctionOnSample: TypeAlias = Callable[[torch.Tensor], torch.Tensor]


class Estimator(metaclass=abc.ABCMeta):
    r"""Computes an estimate of an expectation

    An estimator estimates the value of a function :math:`f` integrated over a
    probability density :math:`P`

    .. math::

        v = \mathbb{E}_{b \sim P}\left[f(b)\right]
          = \int_{b \in \mathrm{supp}(P)} f(b) \mathrm{d}P(b).
    
    The value of :math:`v` can be estimated in many ways. This base class serves as the
    common foundation for those estimators. The usage pattern is as follows:

    .. code-block:: python

        def func(b):
            # return the value of f(b) here
        
        # ...
        # training loop
        for epoch in range(num_epochs):
            # ...
            # 1. Determine parameterization (e.g. logits) from inputs.
            # 2. Initialize the distribution and estimator in the training loop.
            dist = torch.distributions.SomeDistribution(logits=logits)
            estimator = pydrobert.torch.estimators.SomeEstimator(dist, func, ...)
            v = estimator()  # of shape dist.batch_shape
            # 3. calculate loss as a function of v
            loss.backwards()
            # ...

    Parameters
    ----------
    proposal
        The distribution over which the expectation is taken. This is usually but not
        always :math:`P` (see :class:`ImportanceSamplingEstimator` for a
        counterexample).
    func
        The function :math:`f`. A callable (such as a :class:`pydrobert.torch.Module`)
        which accepts a sample tensor as input of shape ``(num_samples,) +
        proposal.batch_shape + proposal.event_shape`` and returns a tensor of shape
        ``(num_samples,) + proposal.batch_shape``.
    is_log
        If :obj:`True`, the estimator operates in log space. `func` defines :math:`\log
        f` instead of :math:`f` and the return value `v` represents an estimate of
        :math:`\log v`. Estimators will often be more numerically stable in log space.
    
    Returns
    -------
    v : torch.Tensor
        An estimate of :math:`v`. Of shape ``proposal.batch_shape``.
    
    Notes
    -----
    An estimator is not a :class:`torch.nn.Module` and is not in general safe to be
    JIT scripted or traced. The parameterization of the proposal distribution is usually
    output 
    """

    proposal: torch.distributions.distribution.Distribution
    func: FunctionOnSample
    is_log: bool

    def __init__(
        self,
        proposal: torch.distributions.distribution.Distribution,
        func: FunctionOnSample,
        is_log: bool = False,
    ):
        super().__init__()
        self.proposal = proposal
        self.func = func
        self.is_log = is_log

    @abc.abstractmethod
    def __call__(self) -> torch.Tensor:
        raise NotImplementedError
