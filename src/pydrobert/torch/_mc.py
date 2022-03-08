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
from typing import Callable, Optional, Sequence, Tuple

import torch

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
return value is a tensor which broadcasts with ``(mc_samples,) + batch_size``,
usually of that shape, storing the values of the function evaluated on each
sample.
"""


class MonteCarloEstimator(metaclass=abc.ABCMeta):
    r"""A Monte Carlo estimator base class

    A Monte Carlo estimator estimates the quantity

    .. math::

        z = \mathbb{E}_{b \sim P}[f(b)]
    
    by replacing the expectation with some form of random sampling. Letting :math:`N`
    represent some number of Monte Carlo samples, :math:`b^{(1:N)}` represent `N`
    samples drawn from a proposal (usually but not necessarily :math:`P`), the estimator
    :func:`G` is unbiased iff

    .. math::

        \mathbb{E}_{b^{(1:N)}}[G(b)] = z
    
    The general pattern for using an estimator is as follows:

    .. code-block:: python

        def func(b):
            # return the value of f(b) here
        
        # ...
        # training loop
        for epoch in range(num_epochs):
            # ...
            # 1. Determine parameterization (logits) from inputs.
            # 2. Initialize the distribution and estimator in the training loop. Avoids
            #    issues with missing computation graphs.
            dist = torch.distributions.SomeDistribution(logits=logits)
            estimator = pydrobert.torch.estimators.SomeEstimator(dist, func, ...)
            z = estimator.estimate(num_mc_samples)
            # 3. calculate loss as a function of z
            loss.backwards()
            # ...
    
    A toy example can be found in the `source repository
    <https://github.com/sdrobert/pydrobert-pytorch/blob/master/tests/test_mc.py>`_ under
    the test name ``test_benchmark``. It can be run from the repository root as

    .. code-block:: shell

        DO_MC_BENCHMARK=1 pytest tests/test_mc.py -k benchmark -s
    
    Warnings
    --------
    Subclasses of :class:`MonteCarloEstimator` cannot be JIT scripted or traced.
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
    r"""Direct MC estimate using REINFORCE gradient estimate

    The expectation :math:`z = \mathbb{E}_{b \sim P}[f(b)]` is estimated by drawing
    :math:`N` samples :math:`b^{(1:N)}` i.i.d. from :math:`P` and taking the sample
    average:

    .. math::

        z \approx \frac{1}{N} \sum_{n=1}^N f\left(b^{(n)}\right).
    
    An optional control variate :math:`c` can be specified:

        z \approx \frac{1}{N} \sum_{n=1}^N
            f\left(b^{(n)}\right) - c\left(b^{(n)}\right) + \mu_c
    
    which is unbiased when :math:`\mathbb{E}_{b \sim P}[c(b)] = \mu_c`.

    In the backward pass, the gradient of the expectation is estimated using REINFORCE
    [williams1992]_:

        \nabla z \approx \frac{1}{N} \sum_{n=1}^N \nabla
            \left(f\left(b^{(n)}\right) - c\left(b^{(n)}\right) + \mu_c\right)\log P(b).
    
    With the control variate terms excluded if they were not specified.

    Parameters
    ----------
    proposal : torch.distributions.Distribution
        The distribution to sample from, :math:`P`.
        The function :math:`f`.
    cv : FunctionOnSample or None, optional
        The function :math:`c`.
    cv_mean : torch.Tensor or None, optional
        The value :math:`\mu_c`.
    is_log : bool, optional
        If :obj:`True`, `func` and `c` are :math:`\log f` and :math:`\log c`
        respectively. Their return values will be exponentiated inside the call
        to :func:`estimate`. There will be little difference from pre-exponentiating
        the return values inside the respective functions/tensors.
    
    See Also
    --------
    pydrobert.torch.estimators.MonteCarloEstimator
        For general information on how to use Monte Carlo estimators.
    """

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
    r"""Importance Sampling MC estimate
    
    The expectation :math:`z = \mathbb{E}_{b \sim P}[f(b)]` is estimated by drawing
    :math:`N` samples :math:`b^{(1:N)}` i.i.d. from the proposal distribution :math:`Q`,
    weighing the values :math:`f(b)` according to the likelihood ratio of
    :math:`P(b)` over :math:`Q(b)`, and taking the sample average:

    .. math::

        z \approx \frac{1}{N} \sum_{n=1}^N w_n f\left(b^{(n)}\right) \\
        w_n = \frac{P\left(b^{(n)}\right)}{Q\left(b^{(n)}\right)}.
    
    The estimate is unbiased iff :math:`Q` dominates :math:`P`, that is

    .. math::

        \forall b \quad P(b) > 0 \implies Q(b) > 0.

    .. math::

        \nabla z \approx \frac{1}{N} \sum_{n=1}^N \frac{1}{Q\left(b^{(n)}\right)
            \nabla P\left(b^{(n)}\right)f\left(b^{(n)}\right).
    
    Note that the gradient with respect to parameters of :math:`Q` will be defined but
    set to zero.
    
    If `self_normalized` is set to :obj:`True`, :math:`z` is instead estimated as:

    .. math::

        z \approx \frac{1}{N} \sum_{n=1}^N \omega_n f\left(b^{(n)}\right) \\
        \omega_n = \frac{P\left(b^{(n)}\right)}{Q\left(b^{(n)}\right)}

    with gradients defined for :math:`Q(b)` using the log trick from REINFORCE
    [williams1992]_:
    
    .. math::
        \nabla z \approx \frac{1}{N} \sum_{n=1}^N
            \nabla \omega_n f\left(b^{(n)}\right) \log Q(b).
        
    
    In this case the gradient of :math:`Q` can be nonzero. The self-normalized estimate
    is biased but with decreasing bias (assuming the proposal dominates) as :math:`N \to
    \infty`. This property holds even if :math:`P` is not a probability density (i.e.
    :math:`\sum_b P(b) \neq 1`).

    Parameters
    ----------
    proposal : torch.distributions.Distribution
        The distribution :math:`Q`.
    func : FunctionOnSample
        The function :math:`f`.
    density : pydrobert.torch.distributions.Density
        The density :math:`P`. Can be unnormalized.
    self_normalize : bool, optional
        Whether to use the self-normalized estimator.
    is_log : bool, optional
        If :obj:`True`, `func` is :math:`\log f`. Its return values will be
        exponentiated inside the call to :func:`estimate`. Results might be more
        numerically stable than if return values are pre-exponentiated inside `func`.

    See Also
    --------
    pydrobert.torch.estimators.MonteCarloEstimator
        For general information on how to use Monte Carlo estimators.
    """

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
        b = self.proposal.sample([mc_samples])
        lqb = self.proposal.log_prob(b)
        lpb = self.density.log_prob(b)
        fb = self.func(b)
        if self.self_normalize:
            llr = lpb - lqb
            llr = llr.log_softmax(0) + math.log(mc_samples)
        else:
            llr = lpb - lqb.detach()
        if self.is_log:
            z = (fb.clamp_min(config.EPS_NINF) + llr).exp()
        else:
            z = fb * llr.exp()
        if self.self_normalize:
            dlqb = z.detach() * lqb
        else:
            dlqb = 0 * lqb  # ensure we get a zero gradient
        z = z + dlqb - dlqb.detach()
        return z.mean(0)


class RelaxEstimator(MonteCarloEstimator):
    r"""RELAX estimator

    The RELAX estimator [grathwohl2017]_ estimates the expectation :math:`z =
    \mathbb{E}_{b \sim P}[f(b)]` for discrete :math:`b` via MC estimation, attempting to
    minimize the variance of the estimator using control variates over their continuous
    relaxations.
    
    Let :math:`z^{(1:N)}` be :math:`N` continous relaxation variables drawn i.i.d.
    :math:`z^{(n)} \sim P(\cdot)` and :math:`b^{(1:N)}` be their discretizations
    :math:`H(z^{(n)}) = b^{(n)}`. Let :math:`P(z|b)` be a conditional s.t. the joint
    distribution satisfies :math:`P(b)P(z|b) = P(z, b) = P(z)I[H(z) = b]` and
    :math:`\tilde{z}^{(1:N)}` be samples drawn from those conditionals
    :math:`\tilde{z}^{(n)} \sim P(\cdot|b^{(n)})`. Let :math:`c` be some control variate
    accepting relaxed samples. Then the (unbiased) RELAX estimator is:

    .. math::

        z \approx \frac{1}{N} \sum^N_{n=1}
            f\left(b^{(n)}\right) - c\left(\tilde{z}^{(n)}\right)
                                  + c\left(z^{(n)}\right).
    
    Pairing this estimator with one of the REBAR control variates from
    :mod:`pydrobert.torch.modules` yields the REBAR estimator [tucker2017]_.

    We offer two ways of estimating the gradient :math:`\nabla z`. The first is a
    REINFORCE-style estimate:

    .. math::

        \nabla z \approx \frac{1}{N} \sum^N_{n=1} \nabla \left(
            \left(f\left(b^{(n)}\right) - c\left(\tilde{z}^{(n)}\right)\right)\log P(b)
                                  + c\left(z^{(n)}\right)\right).
    
    The above estimate requires no special consideration for any variable for which the
    gradient is being calculated. The second, following [grathwohl2017]_, specially
    optimizes the control variate parameters to minimize the variance of the gradient
    estimates of the parameters involved in drawing :math:`z`. Let :math:`\theta_{1:K}`
    be the set of such parameters, :math:`g_{\theta_k} \approx \nabla_{\theta_k}` be a
    REINFORCE-style estimate of the :math:`k`-th :math:`z` parameter using the equation
    above, and let :math:`\gamma` be a control variate parameter. Then the
    variance-minimizing loss can be approximated by:

    .. math::

        \nabla_\gamma \mathrm{Var}(z) \approx \frac{1}{K} \nabla_\gamma
            \left(\sum_{k=1}^K g^2_{\theta_k}\right).
    
    The remaining parameters are calculated with the REINFORCE-style estimator above.
    The proposal parameters `proposal_params` and control variate parameters `cv_params`
    must be specified to use this loss function.

    Parameters
    ----------
    proposal : torch.distributions.Distribution
        The distribution :math:`P`. Must implement
        :class:`pydrobert.torch.distributions.ConditionalStraightThrough`.
    func : FunctionOnSample
        The function :math:`f`.
    cv : FunctionOnSample
        The control variate :math:`c`.
    proposal_params : sequence of torch.Tensor, optional
        A sequence of parameters used in the computation of :math:`z` and
        :math:`P(H(z)`. Does not have to be specified unless using the
        variance-minimizing control variate objective. If non-empty, `cv_params` must
        be non-empty as well.
    cv_params : sequence of torch.Tensor, optional
        A sequence of parameters used in the computation of control variate values. Does
        not have to be specified unless using the variance-minimizing control variate
        objective. If non-empty, `proposal_params` must be non-empty as well.
    is_log : bool, optional
        If :obj:`True`, `func` and `c` are :math:`\log f` and :math:`\log c`
        respectively. Their return values will be exponentiated inside the call to
        :func:`estimate`. There will be little difference from pre-exponentiating the
        return values inside the respective functions/tensors.
    """

    cv: FunctionOnSample
    proposal_params: Tuple[torch.Tensor, ...]
    cv_params: Tuple[torch.Tensor, ...]

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        func: FunctionOnSample,
        cv: FunctionOnSample,
        proposal_params: Sequence[torch.Tensor] = tuple(),
        cv_params: Sequence[torch.Tensor] = tuple(),
        is_log: bool = False,
    ) -> None:
        if not isinstance(proposal, ConditionalStraightThrough):
            raise ValueError(f"proposal must implement ConditionalStraightThrough")
        proposal_params = tuple(proposal_params)
        cv_params = tuple(cv_params)
        if (len(proposal_params) > 0) != (len(cv_params) > 0):
            raise ValueError(
                "either both proposal_params and cv_params must be specified or neither"
            )
        super().__init__(proposal, func, is_log)
        self.cv = cv
        self.proposal_params = proposal_params
        self.cv_params = cv_params

    def estimate(self, mc_samples: int,) -> torch.Tensor:
        z = self.proposal.rsample([mc_samples])
        b = self.proposal.threshold(z)
        zcond = self.proposal.csample(b)
        log_pb = self.proposal.tlog_prob(b)
        fb = self.func(b)
        cvz = self.cv(z)
        cvzcond = self.cv(zcond)
        if self.is_log:
            fb, cvz, cvzcond = fb.exp(), cvz.exp(), cvzcond.exp()
        if self.cv_params:
            z_ = ((fb - cvzcond) * log_pb + cvz).mean(0)
            gs_proposal = torch.autograd.grad(
                z_,
                self.proposal_params,
                torch.ones_like(z_),
                create_graph=True,
                retain_graph=True,
            )
            gs_cv = [0.0] * len(self.cv_params)
            for gp in gs_proposal:
                gp = gp.norm(2)
                gs_cv = [
                    x + y
                    for (x, y) in zip(
                        gs_cv,
                        torch.autograd.grad(gp, self.cv_params, retain_graph=True),
                    )
                ]

            for gc, c in zip(gs_cv, self.cv_params):
                _attach_grad(c, gc.detach())

        fb_cvzcond = fb - cvzcond
        dlog_pb = fb_cvzcond.detach() * log_pb
        z = fb_cvzcond + cvz + dlog_pb - dlog_pb.detach()
        return z.mean(0)


def _attach_grad(x: torch.tensor, g: torch.Tensor):
    def hook(grad):
        hook.__handle.remove()
        hook.__handle = None  # dies if this hook is called again
        return hook.__grad

    hook.__grad = g
    hook.__handle = x.register_hook(hook)


class _RebarControlVariate(torch.nn.Module):

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

    c_{{\\lambda,\\eta}}(z) = \\eta * f(\\sigma(z / \\lambda))

For the {dist} distribution, :math:`\\sigma` is the {sigma} function.

Parameters
----------
func : pydrobert.torch.estimators.FunctionOnSample
    The function :math:`f`. Must be able to accept relaxed samples.
start_temp : float, optional
    The temperature the :math:`\\lambda` parameter is initialized to.
start_eta : float, optional
    The coefficient the :math:`\\eta` parameter is initialzied to.

Warnings
--------
This control variate can be traced but not scripted. Note that the MC estimator this
is passed to is likely unable to be scripted or traced.

See Also
--------
pydrobert.torch.estimators.RelaxEstimator
    For where to use this control variate.
"""


class LogisticBernoulliRebarControlVariate(_RebarControlVariate):
    __doc__ = _REBAR_DOCS.format(dist="LogisticBernoulli", sigma="sigmoid")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.eta * self.func((z / self.log_temp.exp()).sigmoid())


class GumbelOneHotCategoricalRebarControlVariate(_RebarControlVariate):
    __doc__ = _REBAR_DOCS.format(dist="GumbelOneHotCategorical", sigma="softmax")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.eta * self.func((z / self.log_temp.exp()).softmax(-1))
