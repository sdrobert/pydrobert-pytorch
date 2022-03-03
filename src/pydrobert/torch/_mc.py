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

import math
import abc
from typing import Callable, Optional

import torch

from torch.distributions import constraints

from . import config
from .distributions import Density, ConditionalStraightThrough

BERNOULLI_SYNONYMS = {"bern", "Bern", "bernoulli", "Bernoulli"}
CATEGORICAL_SYNONYMS = {"cat", "Cat", "categorical", "Categorical"}
ONEHOT_SYNONYMS = {"onehot", "OneHotCategorical"}

FunctionOnSample = Callable[[torch.Tensor], torch.Tensor]
"""Type for functions of samples used in MC estimation

This type is intended for use in subclasses implementing :class:`MonteCarloEstimator`.

A `FunctionOnSample` is a callable which accepts a :class:`torch.Tensor` and returns a
:class:`torch.Tensor`. The input is of shape ``(mc_samples,) + batch_size +
event_size``, where ``mc_samples`` is some number of Monte Carlo samples and
``batch_size`` and ``event_size`` are determined by the proposal distribution. The
return value is of shape ``(mc_samples, *)`` representing the values of the function
evaluated on all the MC samples.
"""


class MonteCarloEstimator(metaclass=abc.ABCMeta):
    r"""A Monte Carlo estimator base class

    A Monte Carlo estimator estimates the quantity

    .. math::

        z = \mathbb{E}_{b \sim P}[f(b)]
    
    by taking the sample average of :math:`N` samples from some distribution
    (usually :math:`P`) and optionally weighing them with :math:`W(b)` (usually 1):

    .. math::

        z \approx  \frac{1}{N} \sum_{n=1}^N W(b^{(n)}) f(b^{(n)})

    Parameters
    ----------
    proposal : torch.distributions.Distribution
        The distribution which is sampled from, usually identical to the distribution of
        the expectation.
    func : FunctionOnSample
        The function :math:`f` or :math:`\log f`, depending on the value of `is_log`.
    is_log : bool, optional
        If :obj:`True`, evaluations of any :class:`FunctionOnSample` return log values.
        The return values should be the same as if `is_log` were false and the function
        values were exponentiated, and vice-versa. Some estimators will be more
        numerically stable using log values.
    """

    proposal: torch.distributions.Distribution
    func: FunctionOnSample
    is_log: bool

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        func: FunctionOnSample,
        is_log: bool = False,
    ) -> None:
        self.proposal = proposal
        self.func = func
        self.is_log = is_log

    @abc.abstractmethod
    def estimate(self, mc_samples: int) -> torch.Tensor:
        """Perform the MC estimation
        
        Parameters
        ----------
        mc_samples : int
            The number of Monte Carlo samples to estimate with.
        
        Returns
        -------
        z : torch.Tensor
            The MC estimate. The MC dimension should already be reduced.
        """
        raise NotImplementedError


class ReinforceEstimator(MonteCarloEstimator):

    cv: Optional[FunctionOnSample]
    cv_mean: Optional[torch.Tensor]

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        func: FunctionOnSample,
        cv: Optional[FunctionOnSample] = None,
        cv_mean: Optional[torch.Tensor] = None,
        is_log: bool = False,
    ):
        super().__init__(proposal, func, is_log)
        if (cv is None) != (cv_mean is None):
            raise ValueError("Either both cv and cv_mean is specified or neither")
        self.cv = cv
        self.cv_mean = cv_mean

    def estimate(self, mc_samples: int) -> torch.Tensor:
        b = self.proposal.sample([mc_samples])
        fb = self.func(b)
        if self.is_log:
            fb = fb.exp()
        if self.cv is not None:
            c = self.cv_mean
            cvb = self.cv(b)
            if self.is_log:
                c, cvb = c.exp(), cvb.exp()
            fb = fb - cvb + c
        log_pb = self.proposal.log_prob(b)
        dlog_pb = fb.detach() * log_pb
        z = fb + dlog_pb - dlog_pb.detach()
        return z.mean(0)


class ImportanceSamplingEstimator(MonteCarloEstimator):

    density: Density
    self_normalize: bool

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        func: FunctionOnSample,
        density: Density,
        self_normalize: bool = False,
        is_log: bool = False,
    ) -> None:
        super().__init__(proposal, func, is_log)
        self.density = density
        self.self_normalize = self_normalize

    def estimate(self, mc_samples: int) -> torch.Tensor:
        with torch.no_grad():
            b = self.proposal.sample([mc_samples])
            lqb = self.proposal.log_prob(b)
        lpb = self.density.log_prob(b)
        llr = lpb - lqb
        fb = self.func(b)
        if self.self_normalize:
            llr = llr.log_softmax(0) + math.log(mc_samples)
        if self.is_log:
            z = (fb.clamp_min(config.EPS_NINF) + llr).exp()
        else:
            z = fb * llr.exp()
        return z.mean(0)


class RELAXEstimator(MonteCarloEstimator):

    cv: FunctionOnSample

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        func: FunctionOnSample,
        cv: FunctionOnSample,
        is_log: bool = False,
    ) -> None:
        if not isinstance(proposal, ConditionalStraightThrough):
            raise ValueError(f"proposal must implement ConditionalStraightThrough")
        super().__init__(proposal, func, is_log)
        self.cv = cv

    def estimate(self, mc_samples: int) -> torch.Tensor:
        z = self.proposal.rsample([mc_samples])
        b = self.proposal.threshold(z)
        zcond = self.proposal.csample(b)
        log_pb = self.proposal.tlog_prob(b)
        fb = self.func(b)
        cvz = self.cv(z)
        cvzcond = self.cv(zcond)
        if self.is_log:
            fb, cvz, cvzcond = fb.exp(), cvz.exp(), cvzcond.exp()
        fb = fb - cvzcond
        dlog_pb = fb.detach() * log_pb
        z = fb + cvz + dlog_pb - dlog_pb.detach()
        return z.mean(0)


class _REBARControlVariate(torch.nn.Module):

    __constants__ = ["func", "start_temp", "start_eta"]
    func: FunctionOnSample
    start_temp: float
    start_eta: float

    def __init__(self, func, start_temp: float = 0.1, start_eta: float = 1.0) -> None:
        if start_temp <= 0:
            raise ValueError(f"start_temp must be positive, got {start_temp}")
        super().__init__()
        self.func = func
        self.start_temp = start_temp
        self.start_eta = start_eta
        self.log_temp = torch.nn.Parameter(torch.Tensor(1))
        self.eta = torch.nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.log_temp.data.fill_(self.start_temp).log_()
        self.eta.data.fill_(self.start_eta)

    def extra_repr(self) -> str:
        return f"start_temp={self.start_temp},start_eta={self.start_eta}"


_REBAR_DOCS = """REBAR control variate for {dist} relaxation

REBAR [tucker2017]_ is a special case of the RELAX estimator [grathwohl2017]_ with
a control variate that passes a temperature-based transformation of the relaxed sample
to the function :math:`f` the expectation is being taken over. That is:

.. math::

    c_{{\\lambda,\\eta}}(z) = \eta * f(\\sigma(z / \\lambda))

For the {dist} distribution, :math:`\\sigma` is the {sigma} function.

Parameters
----------
func : pydrobert.torch.estimators.FunctionOnSample
    The function :math:`f`. Must be able to accept relaxed samples.
start_temp : float, optional
    The temperature the :math:`\\lambda` parameter is initialized to.
start_eta : float, optional
    The coefficient the :math:`\\eta` parameter is initialzied to.

See Also
--------
pydrobert.torch.estimators.RELAXEstimator
    For where to use this control variate.
"""


class LogisticBernoulliREBARControlVariate(_REBARControlVariate):
    __doc__ = _REBAR_DOCS.format(dist="LogisticBernoulli", sigma="sigmoid")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.eta * self.func((z / self.log_temp.exp()).sigmoid())


class GumbelOneHotCategoricalREBARControlVariate(_REBARControlVariate):
    __doc__ = _REBAR_DOCS.format(dist="GumbelOneHotCategorical", sigma="softmax")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.eta * self.func((z / self.log_temp.exp()).softmax(-1))
