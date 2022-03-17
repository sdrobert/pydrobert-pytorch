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
from typing import Optional, Sequence, Tuple

import torch

from . import config
from .distributions import Density, StraightThrough, ConditionalStraightThrough
from ._estimators import Estimator, FunctionOnSample


class MonteCarloEstimator(Estimator, metaclass=abc.ABCMeta):
    r"""A Monte Carlo estimator base class

    A Monte Carlo estimator estimates an expectation with some form of random sampling
    from a proposal. Letting :math:`N` represent some number of Monte Carlo samples,
    :math:`b^{(1:N)}` represent `N` samples drawn from a proposal (usually but not
    necessarily :math:`P`), the estimator :func:`G` is unbiased iff

    .. math::

        \mathbb{E}_{b^{(1:N)} \sim P}[G(b)] = \mathbb{E}_{b \sim P}[f(b)].
    
    A toy example can be found in the `source repository
    <https://github.com/sdrobert/pydrobert-pytorch/blob/master/tests/test_mc.py>`_ under
    the test name ``test_benchmark``. It can be run from the repository root as

    .. code-block:: shell

        DO_MC_BENCHMARK=1 pytest tests/test_mc.py -k benchmark -s
    
    Parameters
    ----------
    proposal
    func
    mc_samples
        The number of samples to draw from `proposal`, :math:`N`.
    is_log
    """

    mc_samples: int

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        func: FunctionOnSample,
        mc_samples: int,
        is_log: bool = False,
    ) -> None:
        if mc_samples <= 0:
            raise ValueError(f"mc_samples must be a natural number, got {mc_samples}")
        super().__init__(proposal, func, is_log)
        self.mc_samples = mc_samples


class DirectEstimator(MonteCarloEstimator):
    r"""Direct MC estimate using REINFORCE gradient estimate

    The expectation :math:`v = \mathbb{E}_{b \sim P}[f(b)]` is estimated by drawing
    :math:`N` samples :math:`b^{(1:N)}` i.i.d. from :math:`P` and taking the sample
    average:

    .. math::

        v \approx \frac{1}{N} \sum_{n=1}^N f\left(b^{(n)}\right).
    
    An optional control variate :math:`c` can be specified:

    .. math::

        v \approx \frac{1}{N} \sum_{n=1}^N
            f\left(b^{(n)}\right) - c\left(b^{(n)}\right) + \mu_c
    
    which is unbiased when :math:`\mathbb{E}_{b \sim P}[c(b)] = \mu_c`.

    In the backward pass, the gradient of the expectation is estimated using REINFORCE
    [williams1992]_:

    .. math::

        \nabla v \approx \frac{1}{N} \sum_{n=1}^N \nabla
            \left(f\left(b^{(n)}\right) - c\left(b^{(n)}\right) + \mu_c\right)\log P(b).
    
    With the control variate terms excluded if they were not specified.

    Parameters
    ----------
    proposal
    func
    mc_samples
        The number of samples to draw from `proposal`, :math:`N`.
    cv
        The function :math:`c`.
    cv_mean
        The value :math:`\mu_c`.
    is_log
    """

    cv: Optional[FunctionOnSample]
    cv_mean: Optional[torch.Tensor]

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        func: FunctionOnSample,
        mc_samples: int,
        cv: Optional[FunctionOnSample] = None,
        cv_mean: Optional[torch.Tensor] = None,
        is_log: bool = False,
    ):
        super().__init__(proposal, func, mc_samples, is_log)
        self.cv = cv
        self.cv_mean = cv_mean

    def __call__(self) -> torch.Tensor:
        b = self.proposal.sample([self.mc_samples])
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
        v = fb + dlog_pb - dlog_pb.detach()
        return v.mean(0)


class ReparameterizationEstimator(MonteCarloEstimator):
    r"""MC estimation of continuous variables with reparameterization gradient
    
    This estimator applies to distributions over continuous random variables :math:`z
    \sim P` whose values can be decomposed into the sum of a deterministic, learnable
    (i.e. with gradient) part :math:`\theta` and a random, unlearnable part
    :math:`\epsilon`:

    .. math::

        z = \theta + \epsilon,\>\epsilon \sim P' \\
        \nabla P(z) = \nabla P'(\epsilon) = 0
    
    The expectation :math:`v = \mathbb{E}_{z \sim P}[f(z)]` is estimated by drawing
    :math:`N` samples :math:`z^{(1:N)}` i.i.d. from :math:`P` and taking the sample
    average:

    .. math::
    
        v \approx \frac{1}{N} \sum^N_{n=1} f\left(z^{(n)}\right).
    
    We can ignore the probabilities in the bacward direction because :math:`\nabla P(z)
    = 0`, leaving the unbiased estimate of the gradient:

    .. math::

        \nabla v \approx \frac{1}{N} \sum^N_{n=1} \nabla f\left(z^{(n)}\right).
    
    Parameters
    ----------
    proposal
        The distribution over which the expectation is taken, :math:`P` (not
        :math:`P'`). `proposal` must implement the
        :func:`torch.distributions.distribution.Distribution.rsample` method
        (``proposal.has_rsample == True``).
    func
    mc_samples
    is_log
    """

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        func: FunctionOnSample,
        mc_samples: int,
        is_log: bool = False,
    ) -> None:
        if not proposal.has_rsample:
            raise ValueError("proposal must implement rsample")
        super().__init__(proposal, func, mc_samples, is_log)

    def __call__(self) -> torch.Tensor:
        z = self.proposal.rsample([self.mc_samples])
        fz = self.func(z)
        if self.is_log:
            fz = fz.exp()
        return fz.mean(0)


class StraightThroughEstimator(MonteCarloEstimator):
    r"""MC estimation of discrete variables with continuous relaxation's reparam grad
    
    A straight-through estimator [bengio2013]_ is like a
    :class:`ReparameterizationEstimator` but fudges the fact that the samples are
    actually discrete to compute the gradient. To estimate :math:`v = \mathbb{E}_{b \sim
    P}[f(b)]`, we need a distribution over discrete samples' continuous relaxations,

    .. math::

        z = \theta + \epsilon,\>\epsilon \sim P'
    
    and a threshold function :math:`H(z) = b` such that :math:`P(H(z)) = P(b)`. The
    estimate of :math:`v` is computed by drawing :math:`N` relaxed values
    :math:`z^{(1:N)}` and taking the sample average on thresholded values:

    .. math::

        v \approx \frac{1}{N} \sum^N_{n=1} f\left(H\left(z^{(n)}\right)\right).
    
    This estimate is unbiased. In the backward direction, we approximate

    .. math::

        \nabla P(H(z)) \approx \nabla P(z) = \nabla P'(\epsilon) = 0
    
    and end up with a biased estimate of the gradient resembling that of
    :class:`ReparameterizationEstimator`:

    .. math::

        \nabla v \approx \frac{1}{N} \sum^N_{n=1}
            \nabla \left(H\left(z^{(n)}\right)\right).
    
    Parameters
    ----------
    proposal
        The distribution over which the expectation is taken, :math:`P` (not
        :math:`P'`). `proposal` must implement :class:`StraightThrough`.
    func
    mc_samples
    is_log
    """

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        func: FunctionOnSample,
        mc_samples: int,
        is_log: bool = False,
    ) -> None:
        if not isinstance(proposal, StraightThrough):
            raise ValueError("proposal must implement StraightThrough")
        super().__init__(proposal, func, mc_samples, is_log)

    def __call__(self) -> torch.Tensor:
        z = self.proposal.rsample([self.mc_samples])
        b = self.proposal.threshold(z, True)
        fb = self.func(b)
        if self.is_log:
            fb = fb.exp()
        return fb.mean(0)


class ImportanceSamplingEstimator(MonteCarloEstimator):
    r"""Importance Sampling MC estimate
    
    The expectation :math:`v = \mathbb{E}_{b \sim P}[f(b)]` is estimated by drawing
    :math:`N` samples :math:`b^{(1:N)}` i.i.d. from the proposal distribution :math:`Q`,
    weighing the values :math:`f(b)` according to the likelihood ratio of
    :math:`P(b)` over :math:`Q(b)`, and taking the sample average:

    .. math::

        v \approx \frac{1}{N} \sum_{n=1}^N w_n f\left(b^{(n)}\right) \\
        w_n = \frac{P\left(b^{(n)}\right)}{Q\left(b^{(n)}\right)}.
    
    The estimate is unbiased iff :math:`Q` dominates :math:`P`, that is

    .. math::

        \forall b \quad P(b) > 0 \implies Q(b) > 0.
    
    The gradient is estimated as

    .. math::

        \nabla v \approx \frac{1}{N} \sum_{n=1}^N \frac{1}{Q\left(b^{(n)}\right)
            \nabla P\left(b^{(n)}\right)f\left(b^{(n)}\right).
    
    Note that the gradient with respect to parameters of :math:`Q` will be defined but
    set to zero.
    
    If `self_normalized` is set to :obj:`True`, :math:`v` is instead estimated as:

    .. math::

        v \approx \frac{1}{N} \sum_{n=1}^N \omega_n f\left(b^{(n)}\right) \\
        \omega_n = \frac{P\left(b^{(n)}\right)}{Q\left(b^{(n)}\right)}

    with gradients defined for :math:`Q(b)` using the log trick from REINFORCE
    [williams1992]_:
    
    .. math::
        \nabla v \approx \frac{1}{N} \sum_{n=1}^N
            \nabla \omega_n f\left(b^{(n)}\right) \log Q(b).
        
    
    In this case the gradient of :math:`Q` can be nonzero. The self-normalized estimate
    is biased but with decreasing bias (assuming the proposal dominates) as :math:`N \to
    \infty`. This property holds even if :math:`P` is not a probability density (i.e.
    :math:`\sum_b P(b) \neq 1`).

    Parameters
    ----------
    proposal
        The distribution over which the expectation is taken. In this case, `proposal`
        has probability density :math:`Q`, not :math:`P`.
    func
    mc_samples
    density
        The density :math:`P`. Can be unnormalized.
    self_normalize
        Whether to use the self-normalized estimator.
    is_log
        If :obj:`True`, `func` and `c` are :math:`\log f` and :math:`\log c`
        respectively. Their return values will be exponentiated inside the call to
        :func:`estimate`. There will be little difference from pre-exponentiating the
        return values inside the respective functions/tensors.
    """

    density: Density
    self_normalize: bool

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        func: FunctionOnSample,
        mc_samples: int,
        density: Density,
        self_normalize: bool = False,
        is_log: bool = False,
    ) -> None:
        super().__init__(proposal, func, mc_samples, is_log)
        self.density = density
        self.self_normalize = self_normalize

    def __call__(self) -> torch.Tensor:
        b = self.proposal.sample([self.mc_samples])
        lqb = self.proposal.log_prob(b)
        lpb = self.density.log_prob(b)
        fb = self.func(b)
        if self.self_normalize:
            llr = lpb - lqb
            llr = llr.log_softmax(0) + math.log(self.mc_samples)
        else:
            llr = lpb - lqb.detach()
        if self.is_log:
            v = (fb.clamp_min(config.EPS_NINF) + llr).exp()
        else:
            v = fb * llr.exp()
        if self.self_normalize:
            dlqb = v.detach() * lqb
        else:
            dlqb = 0 * lqb  # ensure we get a zero gradient
        v = v + dlqb - dlqb.detach()
        return v.mean(0)


class RelaxEstimator(MonteCarloEstimator):
    r"""RELAX estimator

    The RELAX estimator [grathwohl2017]_ estimates the expectation :math:`v =
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

        v \approx \frac{1}{N} \sum^N_{n=1}
            f\left(b^{(n)}\right) - c\left(\tilde{z}^{(n)}\right)
                                  + c\left(z^{(n)}\right).
    
    Pairing this estimator with one of the REBAR control variates from
    :mod:`pydrobert.torch.modules` yields the REBAR estimator [tucker2017]_.

    We offer two ways of estimating the gradient :math:`\nabla z`. The first is a
    REINFORCE-style estimate:

    .. math::

        \nabla v \approx \frac{1}{N} \sum^N_{n=1} \nabla \left(
            \left(f\left(b^{(n)}\right) - c\left(\tilde{z}^{(n)}\right)\right)\log P(b)
                                  + c\left(z^{(n)}\right)\right).
    
    The above estimate requires no special consideration for any variable for which the
    gradient is being calculated. The second, following [grathwohl2017]_, specially
    optimizes the control variate parameters to minimize the variance of the gradient
    estimates of the parameters involved in drawing :math:`z`. Let :math:`\theta_{1:K}`
    be the set of such parameters, :math:`g_{\theta_k} \approx \nabla_{\theta_k} v` be a
    REINFORCE-style estimate of the :math:`k`-th :math:`z` parameter using the equation
    above, and let :math:`\gamma` be a control variate parameter. Then the
    variance-minimizing loss can be approximated by:

    .. math::

        \nabla_\gamma \mathrm{Var}(v) \approx \frac{1}{K} \nabla_\gamma
            \left(\sum_{k=1}^K g^2_{\theta_k}\right).
    
    The remaining parameters are calculated with the REINFORCE-style estimator above.
    The proposal parameters `proposal_params` and control variate parameters `cv_params`
    must be specified to use this loss function.

    Parameters
    ----------
    proposal
        The distribution over which the expectation is taken, :math:`P`. Must implement
        :class:`pydrobert.torch.distributions.ConditionalStraightThrough`.
    func
    mc_samples
    cv
    proposal_params
        A sequence of parameters used in the computation of :math:`z` and
        :math:`P(H(z)`. Does not have to be specified unless using the
        variance-minimizing control variate objective. If non-empty, `cv_params` must be
        non-empty as well.
    cv_params
        A sequence of parameters used in the computation of control variate values. Does
        not have to be specified unless using the variance-minimizing control variate
        objective. If non-empty, `proposal_params` must be non-empty as well.
    is_log
        If :obj:`True`, `func` and `c` are :math:`\log f` and :math:`\log c`
        respectively. Their return values will be exponentiated inside the call to
        :func:`estimate`. There will be little difference from pre-exponentiating the
        return values inside the respective functions/tensors.
    
    Warnings
    --------
    The current implmentation does not support auxiliary loss functions for the
    control variate parameters when the variance-minimizing objective is used
    (`proposal_params` and `cv_params` are specified). Auxiliary loss functions for
    parameters other than `cv_params` are fine.
    """

    cv: FunctionOnSample
    proposal_params: Tuple[torch.Tensor, ...]
    cv_params: Tuple[torch.Tensor, ...]

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        func: FunctionOnSample,
        mc_samples: int,
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
        super().__init__(proposal, func, mc_samples, is_log)
        self.cv = cv
        self.proposal_params = proposal_params
        self.cv_params = cv_params

    def __call__(self) -> torch.Tensor:
        z = self.proposal.rsample([self.mc_samples])
        b = self.proposal.threshold(z)
        zcond = self.proposal.csample(b)
        log_pb = self.proposal.tlog_prob(b)
        fb = self.func(b)
        cvz = self.cv(z)
        cvzcond = self.cv(zcond)
        if self.is_log:
            fb, cvz, cvzcond = fb.exp(), cvz.exp(), cvzcond.exp()
        if self.cv_params:
            v_ = ((fb - cvzcond) * log_pb + cvz).mean(0)
            gs_proposal = torch.autograd.grad(
                v_,
                self.proposal_params,
                torch.ones_like(v_),
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
        v = fb_cvzcond + cvz + dlog_pb - dlog_pb.detach()
        return v.mean(0)


class IndependentMetropolisHastingsEstimator(MonteCarloEstimator):
    r"""Independent Metropolis Hastings MCMC estimator

    Independent Metropolis Hastings (IMH) is a Markov Chain Monte Carlo (MCMC) technique
    for estimating the value :math:`v = \mathbb{E}_{b \sim P}[f(b)] < \infty`. A
    Markov Chain of :math:`N` samples :math:`b^{(1:N)}` is contructed sequentially by
    iteratively drawing samples from a proposal :math:`b' \sim Q` and either accepting
    it or rejecting it and taking :math:`b^{(n-1)}` as the next sample in the chain,
    :math:`b^{(n)}`, according to the following rules:

    .. math::

        u \sim \mathrm{Uniform}([0, 1]) \\
        b^{(n)} = \begin{cases}
            b' & \alpha(b', b^{(n-1)}) > u \\
            b^{(n-1)} & \mathrm{otherwise}
        \end{cases} \\
        \alpha(b', b^{(n-1)}) = \min\left(
            \frac{P(b')Q(b^{(n-1)})}{P(b^{(n-1)}Q(b'))}, 1\right).
    
    The sample estimate from the Markov Chain

    .. math::

        v \approx \frac{1}{N - M} \sum_{n=M + 1}^N  f\left(b^{(n)}\right)
    
    for a fixed number of burn-in samples :math:`M \in [0, N)` is biased but converges
    asymptotically (:math:`\lim N \to \infty`) to :math:`v` with strong guarantees
    as long as there exists some constant :math:`\epsilon` such that [mengerson1996]_

    .. math::

        P(b) > 0 \implies \frac{P(b)}{Q(b)} \leq \epsilon.
    
    Parameters
    ----------
    proposal
        The proposal distribution :math:`Q`.
    func
    mc_samples
    density
        The density :math:`P`. Does not have to be a probability distribution (can be
        unnormalized).
    burn_in
        The number of samples in the chain discarded from the estimate, :math:`M`.
    initial_sample
        If specified, `initial_sample` is used as the value :math:`b^{(0)}` to start the
        chain. Of size either ``proposal.batch_size + proposal.event_size`` or ``(1,) +
        proposal.batch_size + proposal.event_size``. A :class:`ValueError` will be
        thrown if any elements are outside the support of :math:`P` (`density`). If
        unspecified, :math:`b^{(0)}` will be decided by randomly drawing from `proposal`
        until all elements are in the support of `density`.
    initial_sample_tries
        If `initial_sample` is unspecified, `initial_sample_tries` dictates the
        maximum number of draws from `proposal` allowed in order to find elements in
        the support of `density` before a :class:`RuntimeError` is thrown.

    Warnings
    --------
    The resulting estimate has no gradient attached to it and therefore cannot be
    backpropagated through.
    """

    density: Density
    initial_sample: Optional[torch.Tensor]
    initial_sample_tries: int
    burn_in: int

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        func: FunctionOnSample,
        mc_samples: int,
        density: Density,
        burn_in: int = 0,
        initial_sample: Optional[torch.Tensor] = None,
        initial_sample_tries: int = 1000,
        is_log: bool = False,
    ) -> None:
        super().__init__(proposal, func, mc_samples, is_log)
        if burn_in < 0 or burn_in >= mc_samples:
            raise ValueError(f"burn_in must be between [0, mc_samples={mc_samples}")
        if initial_sample is not None:
            sample_shape = self.proposal.batch_shape + self.proposal.event_shape
            if initial_sample.shape == sample_shape:
                initial_sample = initial_sample.unsqueeze(0)
            elif initial_sample.shape != (1,) + sample_shape:
                raise ValueError(
                    f"Expected initial_sample to have shape {(1,) + sample_shape} or "
                    f"{sample_shape} "
                )
            if not torch.isfinite(self.density.log_prob(initial_sample)).all():
                raise ValueError(
                    "all values in initial_sample must lie in the support of density"
                )
        elif initial_sample_tries < 1:
            raise ValueError(
                "initial_sample_tries must be positive when initial_sample is None"
            )
        self.density = density
        self.initial_sample = initial_sample
        self.initial_sample_tries = initial_sample_tries
        self.burn_in = burn_in

    @torch.no_grad()
    def find_initial_sample(self, tries: Optional[int] = None) -> torch.Tensor:
        """Find an initial sample by randomly sampling from the proposal"""
        if tries is None:
            tries = self.initial_sample_tries
        if tries < 1:
            raise ValueError("tries must be positive")
        sample = self.proposal.sample([1])
        keep = torch.isfinite(self.density.log_prob(sample))
        if keep.all():
            return sample
        for _ in range(tries - 1):
            cur_sample = self.proposal.sample([1])
            while keep.dim() < cur_sample.dim():
                keep = keep.unsqueeze(-1)  # event dims
            sample = torch.where(keep, sample, cur_sample)
            keep = torch.isfinite(self.density.log_prob(sample))
            if keep.all():
                return sample
        raise RuntimeError(
            f"Unable to find initial sample in {tries} draws. "
            f"Either specify initial_sample on instantiation or increase "
            f"initial_sample_tries."
        )

    @torch.no_grad()
    def __call__(self) -> torch.Tensor:
        if self.initial_sample is None:
            last_sample = self.find_initial_sample()
        else:
            last_sample = self.initial_sample
        v = 0
        num_kept = self.mc_samples - self.burn_in
        last_ratio = self.density.log_prob(last_sample) - self.proposal.log_prob(
            last_sample
        )
        uniform_draws = torch.rand(
            (self.mc_samples,) + self.proposal.batch_shape, device=last_sample.device
        ).log()
        for n in range(self.mc_samples):
            cur_sample = self.proposal.sample([1])
            cur_ratio = self.density.log_prob(cur_sample) - self.proposal.log_prob(
                cur_sample
            )
            accept = (cur_ratio - last_ratio) > uniform_draws[n]
            cur_ratio = accept * cur_ratio + (~accept) * last_ratio
            while accept.dim() < cur_sample.dim():
                accept = accept.unsqueeze(-1)  # event dims
            cur_sample = torch.where(accept, cur_sample, last_sample)
            if n >= self.burn_in:
                fb = self.func(cur_sample).squeeze(0)
                if self.is_log:
                    fb = (fb - math.log(num_kept)).exp()
                else:
                    fb /= num_kept
                v += fb
            last_sample, last_ratio = cur_sample, cur_ratio
        return v


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
func
    The function :math:`f`. Must be able to accept relaxed samples.
start_temp
    The temperature the :math:`\\lambda` parameter is initialized to.
start_eta
    The coefficient the :math:`\\eta` parameter is initialzied to.

Variables
---------
log_temp
    A scalar initialized to ``log(start_temp)``.
eta
    A scalar initialized to ``start_eta``.

Warnings
--------
This control variate can be traced but not scripted. Note that
:class:`pydrobert.torch.estimators.RelaxEstimator` is unable to be traced or scripted.

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
