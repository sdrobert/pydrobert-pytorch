# Copyright 2022 Sean Robertson

# The _validate_thresholded_sample method was adapted from PyTorch's _validate_sample
# method
# https://github.com/pytorch/pytorch/blob/201f7d330ac8c33a7bedb8f0a66954415d1d27db/torch/distributions/distribution.py
# GumbelOneHotCategorical is based on Pytorch's OneHotCategorical, Categorical, and
# Gumbel. LogisticBernoulli is based on Pytorch's Bernoulli and Gumbel
# https://github.com/pytorch/pytorch/blob/201f7d330ac8c33a7bedb8f0a66954415d1d27db/torch/distributions/one_hot_categorical.py
# https://github.com/pytorch/pytorch/blob/201f7d330ac8c33a7bedb8f0a66954415d1d27db/torch/distributions/categorical.py
# https://github.com/pytorch/pytorch/blob/201f7d330ac8c33a7bedb8f0a66954415d1d27db/torch/distributions/gumbel.py
# https://github.com/pytorch/pytorch/blob/201f7d330ac8c33a7bedb8f0a66954415d1d27db/torch/distributions/bernoulli.py
# See LICENSE_pytorch in project root directory for PyTorch license.

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
from typing import Optional, Sequence
import warnings
import math

import torch

from torch.distributions import constraints
from torch.distributions.utils import (
    broadcast_all,
    clamp_probs,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)

from ._compat import check_methods, euler_constant, one_hot


class StraightThrough(metaclass=abc.ABCMeta):
    """Interface for distributions for which a straight through estimate is possible

    Classes implementing this interface supply both a method for drawing a relaxed
    sample :func:`rsample` and a method for thresholding it into a discrete sample
    :func:`threshold`.
    """

    @abc.abstractmethod
    def rsample(self, sample_shape: Sequence = torch.Size()) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def threshold(
        self, z: torch.Tensor, straight_through: bool = False
    ) -> torch.Tensor:
        """Convert a relaxed sample into a discrete sample
        
        Parameters
        ----------
        z : torch.Tensor
            A relaxed sample, usually drawn via this instance's :func:`rsample` method.
        straight_through : bool, optional
            If true, attach the gradient of `z` to the discrete sample.
        
        Returns
        -------
        b : torch.Tensor
            The discrete sample acquired by applying a threshold function to `z`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def tlog_prob(self, b: torch.Tensor) -> torch.Tensor:
        """The log probability of a thresholded sample
        
        Parameters
        ----------
        b : torch.Tensor
            A discrete sample. Usually the result of drawing a relaxed sample from
            this instance's :func:`rsample` method, then applying a discrete threshold
            to it via :func:`threshold`.
        
        Returns
        -------
        lp : torch.Tensor
            The log probability of the sample.
        """
        raise NotImplementedError

    def _validate_thresholded_sample(self, value: torch.Tensor):
        """Argument validation for methods with a thresholded (discrete) sample arg
        
        Akin to :func:`torch.distributions.Distribution._validate_sample`
        """
        if not isinstance(value, torch.Tensor):
            raise ValueError("The b argument must be a Tensor")

        event_dim_start = len(value.size()) - len(self.event_shape)
        if value.size()[event_dim_start:] != self.event_shape:
            raise ValueError(
                "The right-most size of b must match event_shape:"
                f"{value.size()} vs {self.event_shape}."
            )

        actual_shape = value.size()
        expected_shape = self.batch_shape + self.event_shape
        for i, j in zip(reversed(actual_shape), reversed(expected_shape)):
            if i != 1 and j != 1 and i != j:
                raise ValueError(
                    "Value is not broadcastable with batch_shape+event_shape: "
                    f"{actual_shape} vs {expected_shape}."
                )

        try:
            support = self.thresholded_support
        except NotImplementedError:
            warnings.warn(
                f"{self.__class__} does not define `thresholded_support` to enable "
                "sample validation. Please initialize the distribution with "
                "`validate_args=False` to turn off validation."
            )
            return
        assert support is not None
        valid = support.check(value)
        if not valid.all():
            raise ValueError(
                "Expected b argument "
                f"({type(value).__name__} of shape {tuple(value.shape)}) "
                f"to be within the support ({repr(support)}) "
                f"of the distribution {repr(self)}, "
                f"but found invalid values:\n{value}"
            )

    @classmethod
    def __subclasscheck__(cls, C) -> bool:
        if cls is StraightThrough:
            return check_methods(C, "rsample", "threshold", "tlog_prob")
        return NotImplemented


class ConditionalStraightThrough(StraightThrough, metaclass=abc.ABCMeta):
    """Straight-throughs with a conditional dist on relaxed samples given discrete ones

    In addition to the methods of :class:`StraightThrough`, classes implementing this
    interface additionally allow for relaxed sampling given its discrete image
    :func:`csample`, and a method for determining the log probability of that
    conditional :func:`clog_prob`.
    """

    @abc.abstractmethod
    def csample(self, b: torch.Tensor) -> torch.Tensor:
        """Draw a relaxed sample conditioned on its thresholded (discrete) image

        Parameters
        ----------
        b : torch.Tensor
            A discrete sample. Usually the result of drawing a relaxed sample from
            this instance's :func:`rsample` method, then applying a discrete threshold
            to it via :func:`threshold`.
        
        Returns
        -------
        zcond : torch.Tensor
            A relaxed sample such that ``threshold(zcond) == b``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def clog_prob(self, zcond: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        r"""Return the log probability of a relaxed sample conditioned on a discrete one

        Returns :math:`lp = log P(z^{cond}|b)`, where the conditional obeys the
        following equality:

        .. math::

            P(z^{cond}|b)P(b) = P(z^{cond}, b) = \begin{cases}
                P(z^{cond}) & H(z^{cond}) = b \\
                0           & \mathrm{otherwise}
            \end{cases}
        
        where :math:`H` is the threshold function. In other words, given a discrete
        sample `b` which is the output of some thresholded relaxed sample, what is the
        probability that `zcond` is that sample?

        Parameters
        ----------
        zcond : torch.Tensor
            A relaxed sample.
        b : torch.Tensor
            A discrete sample. Usually the result of drawing a relaxed sample from
            this instance's :func:`rsample` method, then applying a discrete threshold
            to it via :func:`threshold`.
        
        Returns
        -------
        lp : torch.Tensor
        """
        raise NotImplementedError

    @classmethod
    def __subclasscheck__(cls, C) -> bool:
        if cls is ConditionalStraightThrough:
            return check_methods(
                C, "rsample", "threshold", "tlog_prob", "csample", "clog_prob"
            )
        return NotImplemented


class Density(metaclass=abc.ABCMeta):
    """Interface for a density function

    A density is a non-negative function over some domain. A density implements the
    method :func:`log_prob` which returns the log of the density applied to that sample. 
    
    While :func:`log_prob` is not necessarily a log probability for all densities, the
    name was chosen to match the method of :class:`torch.distributions.Distribution`.
    All probability densities are densities.
    """

    @abc.abstractmethod
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LogisticBernoulli(torch.distributions.Distribution, ConditionalStraightThrough):
    r"""A Logistic distribution which can be thresholded to Bernoulli samples
    
    This distribution should be treated as a (normalized) `Logistic distribution
    <https://en.wikipedia.org/wiki/Logistic_distribution>`__ with the option to
    discretize to Bernoulli values, not the other way around. :func:`sample`,
    :func:`rsample`, and statistics like the mean and standard deviation are all
    relative to the relaxed sample.

   The relaxation, threshold, and conditional relaxed sample defined in
    [tucker2017]_. The relaxation :math:`z` is sampled as

    .. math::

        u_i \sim Uniform([0, 1]) \\
        z_i = logits_i + log(u_i) - log (1 - u_i)

    which can be transformed into a Bernoulli sample by threshold

    .. math::

        b_i = \begin{cases}
            1 & z_i >= 0 \\
            0 & z_i < 0
        \end{cases}.
    
    A relaxed sample :math:`z^{cond}` conditioned on the Bernoulli sample :math:`b` can
    be drawn by

    .. math::

        v_i \sim Uniform([0, 1]) \\
        z^{cond}_i = \begin{cases}
            \log\left(\frac{v_i}{(1 - v_i)(1 - probs_i)} + 1 \right) & b_i = 1 \\
            -\log\left(\frac{v_i}{(1 - v_i)probs_i} + 1\right) & b_i = 0
        \end{cases}.
    """

    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.real
    thresholded_support = constraints.boolean
    has_enumerate_support = False
    has_rsample = True

    def __init__(
        self,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Either probs or logits must be specified, not both")
        if probs is not None:
            self._param = self.probs = probs
        else:
            self._param = self.logits = logits
        super(LogisticBernoulli, self).__init__(
            self._param.shape, validate_args=validate_args
        )

    @lazy_property
    def logits(self) -> torch.Tensor:
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self) -> torch.Tensor:
        return logits_to_probs(self.logits, is_binary=True)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogisticBernoulli, _instance)
        batch_shape = torch.Size(batch_shape)
        if "probs" in self.__dict__:
            new._param = new.probs = self.probs.expand(batch_shape)
        if "logits" in self.__dict__:
            new._param = new.logits = self.logits.expand(batch_shape)
        super(LogisticBernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    def rsample(self, sample_shape: Sequence = torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        logits = self.logits
        u = clamp_probs(torch.rand(shape, device=logits.device, dtype=logits.dtype))
        z = logits + u.log() - (-u).log1p()
        return z

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(z)
        # G(z) = (1 + exp(logits - z))^{-1}
        # g(z) = exp(logits - z) G(z)^2
        Ginv = self.logits - z
        g = Ginv - 2 * Ginv.exp().log1p()
        return g

    def threshold(
        self, z: torch.Tensor, straight_through: bool = False
    ) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(z)
        with torch.no_grad():
            b = (z >= 0.0).to(z)
        if straight_through:
            b = b + z - z.detach()
        return b

    def tlog_prob(self, b: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_thresholded_sample(b)
        logits, b = broadcast_all(self.logits, b)
        return -torch.nn.functional.binary_cross_entropy_with_logits(
            logits, b, reduction="none"
        )

    def csample(self, b: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_thresholded_sample(b)
        v = clamp_probs(torch.rand_like(b))
        probs = clamp_probs(self.probs)
        zcond = v / ((1 - v) * ((1 - b) * probs + b * (1 - probs))) + 1
        zcond = (2 * b - 1) * zcond.log()
        return zcond + b * torch.finfo(b.dtype).eps

    def clog_prob(self, zcond: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        bcond = self.threshold(zcond)  # validates zcond
        if self._validate_args:
            self._validate_thresholded_sample(b)
        zero_prob = bcond != b
        logits = self.logits
        # P(z,b=0) = g(z) / G(0) I[z <= 0]
        #          = exp(logits - z) (1 + exp(logits)) (1 + exp(logits  - z))^{-2} I[.]
        # P(z,b=1) = g(z) / (1 - G(0)) I[z > 0]
        #          = exp(-z) (1 + exp(logits)) (1 + exp(logits - z))^{-2} I[.]
        lp = (
            -zcond
            + (1 - b) * logits
            + logits.exp().log1p()
            - 2 * (logits - zcond).exp().log1p()
        )
        return lp.masked_fill(zero_prob, -float("inf"))

    @property
    def mean(self) -> torch.Tensor:
        return self.logits

    @property
    def stddev(self) -> torch.Tensor:
        return torch.tensor(
            math.pi / math.sqrt(3), device=self._param.device, dtype=self._param.dtype
        ).expand(self.batch_shape)

    @property
    def variance(self) -> torch.Tensor:
        return self.stddev.pow(2)

    def entropy(self) -> torch.Tensor:
        return torch.tensor(
            2, device=self._param.device, dtype=self._param.dtype
        ).expand(self.batch_shape)


class GumbelOneHotCategorical(
    torch.distributions.Distribution, ConditionalStraightThrough
):
    r"""Gumbel distributions with a categorical relaxation

    This distribution should be treated as a series of independent `Gumbel distributions
    <https://en.wikipedia.org/wiki/Gumbel_distribution>`__, normalized along the final
    dimension of `logits` or `probs`. Samples can optionally be discretized to draws
    from a (one-hot) categorical distribution by taking the max Gumbel variable along
    the final axis. :func:`sample`, :func:`rsample`, and statistics like the mean and
    standard deviation are all relative to the Gumbel samples.
    
    The relaxation, threshold, and conditional relaxed sample defined in [tucker2017]_.
    The relaxation :math:`z` is sampled as

    .. math::

        u_{i,j} \sim Uniform([0, 1]) \\
        z_{i,j} = \log probs_{i,j} - \log(-\log u_{i,j})
    
    which can be transformed into a (one-hot) categorical sample via by threshold

    .. math::

        b_{i,j} = \begin{cases}
            1 & j' \neq j \implies z_{i,j} > z_{i,j'} \\
            0 & \mathrm{otherwise}
        \end{cases}.

    A relaxed sample :math:`z^{cond}` conditioned on categorical sample :math:`b` can be
    drawn by

    .. math::

        v_{i,j} \sim Uniform([0, 1]) \\
        z^{cond}_{i,j} = \begin{cases}
            -\log(-\log v_{i,j}) & b_{i,j} = 1 \\
            -\log\left(
                -\frac{\log v_{i,j}}{probs_{i,j}} - \log \sum_{j'} b_{i,j'} v_{i,j'}
            \right) & b_{i,j} = 0
        \end{cases}.
    """

    arg_constraints = {
        "probs": constraints.simplex,
        "logits": constraints.real_vector,
    }
    support = constraints.real_vector
    thresholded_support = one_hot
    has_enumerate_support = False
    has_rsample = True

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Either probs or logits must be specified, not both")
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("probs must be at least 1 dimensional")
            self._param = self.probs = probs / probs.sum(-1, keepdim=True)
        else:
            if logits.dim() < 1:
                raise ValueError("logits must be at least 1 dimensional")
            self._param = self.logits = logits.log_softmax(-1)
        shape = self._param.shape
        batch_shape, event_shape = shape[:-1], shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @lazy_property
    def logits(self) -> torch.Tensor:
        return probs_to_logits(self.probs)

    @lazy_property
    def probs(self) -> torch.Tensor:
        return logits_to_probs(self.logits)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GumbelOneHotCategorical, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + self.event_shape
        if "probs" in self.__dict__:
            new._param = new.probs = self.probs.expand(param_shape)
        if "logits" in self.__dict__:
            new._param = new.logits = self.logits.expand(param_shape)
        super(GumbelOneHotCategorical, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    def rsample(self, sample_shape: Sequence = torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        logits = self.logits
        u = clamp_probs(torch.rand(shape, device=logits.device, dtype=logits.dtype))
        z = logits - (-u.log()).log()
        return z

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(z)
        g = self.logits - z
        return (g - g.exp()).sum(-1)

    def threshold(
        self, z: torch.Tensor, straight_through: bool = False
    ) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(z)
        with torch.no_grad():
            b_ = z.argmax(-1)
            b = torch.nn.functional.one_hot(b_, z.size(-1)).to(z)
        if straight_through:
            b = b + z - z.detach()
        return b

    def tlog_prob(self, b: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_thresholded_sample(b)
        lp_shape = b.shape[:-1]
        return self.logits.expand_as(b).masked_select(b.bool()).view(lp_shape)

    def csample(self, b: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_thresholded_sample(b)
        probs = clamp_probs(self.probs)
        log_v = clamp_probs(torch.rand_like(b)).log()
        zcond_match = -(-log_v).log() * b
        zcond_match_k = zcond_match.sum(-1, keepdim=True)
        zcond_nomatch = -(-log_v / probs - (log_v * b).sum(-1, keepdim=True)).log()
        # this reparameterization isn't very stable, so there's a small chance
        # zcond_nomatch ends up with the same value as zcond_match_k
        zcond_nomatch = torch.min(
            zcond_match_k - torch.finfo(b.dtype).eps, zcond_nomatch
        ) * (1 - b)
        return zcond_match + zcond_nomatch

    def clog_prob(self, zcond: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        bcond = self.threshold(zcond)  # validates zcond
        if self._validate_args:
            self._validate_thresholded_sample(b)
        zero_prob = (bcond != b).any(-1)
        # G(z) = exp(-exp((logits - z)/temp))
        # log G(z) = -exp((logits - z)/temp)
        # g(z) = 1/temp exp((logits - z)/temp)G(z)
        # log g(z) = (logits - z)/temp - exp((logits - z)/temp) - log temp
        # log P(z|b) = -inf * I[H(z) = b] + log g_0(z_k)
        #              + sum_{j != k} log g_{loc_j}(z_j) - log G_{loc_j}(z_k)
        neg_b = 1 - b
        logits = self.logits * neg_b
        g = logits - zcond
        g = g - g.exp()
        z_k = (zcond * b).sum(-1, keepdim=True)
        G = logits - z_k
        G = -G.exp() * neg_b
        log_prob = (g - G).sum(-1)
        return log_prob.masked_fill(zero_prob, -float("inf"))

    @property
    def mean(self) -> torch.Tensor:
        return self.logits + euler_constant

    @property
    def stddev(self) -> torch.Tensor:
        return torch.tensor(
            math.pi / math.sqrt(6), device=self._param.device, dtype=self._param.dtype
        ).expand(self._extended_shape())

    @property
    def variance(self) -> torch.Tensor:
        return self.stddev.pow(2)

    def entropy(self) -> torch.Tensor:
        return torch.tensor(
            self.event_shape[0] * (1 + euler_constant),
            device=self._param.device,
            dtype=self._param.dtype,
        ).expand(self.batch_shape)

