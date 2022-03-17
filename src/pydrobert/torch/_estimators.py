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
"""Type for functions of samples used in estimators

This type is intended for use in estimators subclassing :class:`Estimator`.

A `FunctionOnSample` is a callable which accepts a :class:`torch.Tensor` and returns a
:class:`torch.Tensor`. The input is of shape ``(N,) + batch_size + event_size``, where
``N`` is some number of samples and ``batch_size`` and ``event_size`` are determined by
the proposal distribution. The return value is a tensor which broadcasts with ``(N,) +
batch_size``, usually of that shape, storing the values of the function evaluated on
each sample.

`FunctionOnSample` can be a :class:`torch.nn.Module`.
"""


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
        The function :math:`f`.
    is_log
        If :obj:`True`, `func` defines :math:`\log f` instead of :math:`f`. Unless
        otherwise specified, `is_log` being true is semantically identical to redefining
        `func` as::

            def new_func(b):
                return func(b).exp()
        
        and setting `is_log` to :obj:`False`. Practically, `is_log` may improve the
        numerical stability of certain estimators.
    
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
