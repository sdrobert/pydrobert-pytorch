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

from typing import Optional, Union

import torch

from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from ._compat import script, trunc_divide


@script
@torch.no_grad()
def simple_random_sampling_without_replacement(
    total_count: torch.Tensor,
    given_count: torch.Tensor,
    out_size: Optional[int] = None,
) -> torch.Tensor:
    """Draw a binary vector with uniform probabilities but fixed cardinality

    Uses the algorithm proposed in [fan1962]_.

    Parameters
    ----------
    total_count
        The nonnegative sizes of the individual binary vectors. Must broadcast with
        `given_count`.
    given_count
        The cardinalities of the individual binary vectors. Must broadcast with
        and not exceed the values of `total_count`.
    out_size : int or None, optional
        The vector size. Must be at least the value of ``total_count.max()``. If unset,
        will default to that value.
    
    Returns
    -------
    b : torch.Tensor
        A sample tensor of shape ``(*, out_size)``, where ``(*,)`` is the broadcasted
        shape of `total_count` and `given_count`. The final dimension is the vector
        dimension. The ``n``-th vector of `b` is right-padded with zero for all values
        exceeding ``total_count[n]``, i.e. ``b[n, total_count[n]:].sum() == 0``. The
        remaining ``total_count[n]`` elements of the vector sum to associated given
        count, i.e. ``b[n, :total_count[n]].sum() == given_count[n]``.

    See Also
    --------
    pydrobert.torch.distributions.SimpleRandomSamplingWithoutReplacement
        For information on the distribution.
    """
    total_count_max = int(total_count.max().item())
    if out_size is None:
        out_size = total_count_max
    total_count, given_count = torch.broadcast_tensors(total_count, given_count)
    if (given_count > total_count).any():
        raise RuntimeError("given_count cannot exceed total_count")
    if out_size < total_count_max:
        raise RuntimeError(
            f"out_size ({out_size}) must not be less than max of total_count "
            f"({total_count_max})"
        )
    b = torch.empty(
        torch.Size([out_size]) + total_count.shape, device=total_count.device
    )
    remainder_ell = given_count
    remainder_t = total_count.clamp_min(1)
    for t in range(out_size):
        p = remainder_ell / remainder_t
        b_t = torch.bernoulli(p)
        b[t] = b_t
        remainder_ell = remainder_ell - b_t
        remainder_t = (remainder_t - 1).clamp_min_(1)
    return b.view(out_size, -1).T.view(total_count.shape + torch.Size([out_size]))


class BinaryCardinalityConstraint(constraints.Constraint):
    """Ensures a vector of binary values sums to the required cardinality"""

    is_discrete = True
    event_dim = 1

    def __init__(
        self, total_count: torch.Tensor, given_count: torch.Tensor, tmax: int
    ) -> None:
        self.given_count = given_count
        self.total_count_mask = total_count.unsqueeze(-1) <= torch.arange(
            tmax, device=total_count.device
        )
        super().__init__()

    def check(self, value: torch.Tensor) -> torch.Tensor:
        is_bool = ((value == 0) | (value == 1)).all(-1)
        isnt_gte_tc = (self.total_count_mask.expand_as(value) * value).sum(-1) == 0
        value_sum = value.sum(-1)
        matches_count = value_sum == self.given_count.expand_as(value_sum)
        return is_bool & isnt_gte_tc & matches_count


@script
def binomial_coefficient(length: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    r"""Compute the binomial coefficients (length choose count)
    
    The binomial coefficient "`length` choose `count`" is calculated as

    .. math::

        \binom{length}{count} = \frac{length!}{length!(length - count)!} \\
        x! = \begin{cases}
            \prod_{x'=1}^x x' & x > 0 \\
            1 & x = 0 \\
            0 & x < 0
        \end{cases}

    Parameters
    ----------
    length
        A long tensor of the upper terms in the coefficient. Must broadcast with
        `count`.
    count
        A long tensor of the lower terms in the coefficient. Must broadcast with
        `length`.
    
    Returns
    -------
    binom : torch.Tensor
        A long tensor of the broadcasted shape of `length` and `count`. The value
        at multi-index ``n``, ``binom[n]``, stores the binomial coefficient
        ``length[n]`` choose ``count[n]``, assuming `length` and `count` have already
        been broadcast together.
    
    Warnings
    --------
    As the values in `binom` can get very large, this function is susceptible to
    overflow. For example, :math:`\binom{67}{33}` exceeds the long's maximum. Overflow
    will be avoided by ensuring `length` does not exceed :obj:`66`. The binomial
    coefficient is at its highest when ``count = length // 2`` and at its lowest when
    ``count == length`` or ``count == 0``.
    """
    device = length.device
    if ((count < 0) | (length < 0)).any():
        raise RuntimeError("length and count must be non-negative")
    length_ = int(length.max().item())
    if length_ > 20:
        count_ = int(count.max().item())
        binom = torch.empty((count_ + 1, length_ + 1), device=device, dtype=torch.long)
        binom[..., 0] = 0
        binom[0] = 1
        for c in range(1, count_ + 1):
            binom[c, 1:] = binom[c - 1, :-1].cumsum(0)
        binom = binom.flatten()[length + count * (length_ + 1)]
    else:
        # the factorials are guaranteed to lie within long precision; this algorithm
        # saves some time
        length_m_count = (length - count).clamp_min_(-1)
        count = count.clamp_max(length_)
        x = torch.arange(length_ + 2, device=device)
        x[0] = 1
        x = x.cumprod_(0)
        binom = trunc_divide(x[length], x[count] * x[length_m_count])
        binom.masked_fill_(length_m_count == -1, 0)
    return binom


class SimpleRandomSamplingWithoutReplacement(torch.distributions.ExponentialFamily):
    r"""Draw binary vectors with uniform probability but fixed cardinality
    
    `Simple Random Sampling Without Replacement
    <https://en.wikipedia.org/wiki/Simple_random_sample>`__ (SRSWOR) is a uniform
    distribution over binary vectors of length :math:`T` with a fixed sum :math:`L`:

    .. math::

        P(b|L) = I\left[\sum^T_{t=1} b_t = L\right] \frac{1}{T \mathrm{\>choose\>} L}
    
    where :math:`I[\cdot]` is the indicator function. The distribution is a special
    case of the Conditional Bernoulli [chen1994]_ and a member of the
    `Exponential Family <https://en.wikipedia.org/wiki/Exponential_family>`__.

    Parameters
    ----------
    total_count
        The value(s) :math:`T`. Must broadcast with `given_count`. Represents the sizes
        of the sample vectors. If not all equal or less than `out_size`, samples will be
        right-padded with zeros.
    given_count
        The value(s) :math:`L`. Must broadcast with and have values no greater than
        `total_count`. Represents the cardinality constraints of the sample vectors.
    out_size
        The length of the binary vectors. If it exceeds some value of `total_count`,
        that sample will be right-padded with zeros. Must be no less than
        ``total_count.max()``. If unset, defaults to that value.        
    """

    arg_constraints = {
        "total_count": constraints.nonnegative_integer,
        "given_count": constraints.nonnegative_integer,
    }
    _mean_carrier_measure = 0

    def __init__(
        self,
        given_count: Union[int, torch.Tensor],
        total_count: Union[int, torch.Tensor],
        out_size: Optional[int] = None,
        validate_args=None,
    ):
        given_count = torch.as_tensor(given_count)
        total_count = torch.as_tensor(total_count)
        total_count_max = int(total_count.max().item())
        if out_size is None:
            out_size = total_count_max
        given_count, total_count = torch.broadcast_tensors(given_count, total_count)
        batch_shape = given_count.size()
        event_shape = torch.Size([out_size])
        self.total_count, self.given_count = total_count, given_count
        super().__init__(batch_shape, event_shape, validate_args)
        if self._validate_args:
            if (total_count < 0).any():
                raise ValueError("total_count must be nonnegative")
            if (given_count > total_count).any():
                raise ValueError("given_count cannot exceed total count")
            if out_size < total_count_max:
                raise ValueError(
                    f"out_size ({out_size}) must not be less than max of total_count "
                    f"({total_count_max})"
                )

    @constraints.dependent_property
    def support(self) -> torch.Tensor:
        return BinaryCardinalityConstraint(
            self.total_count, self.given_count, self.event_shape[0]
        )

    @lazy_property
    def log_partition(self) -> torch.Tensor:
        # log total_count choose given_count
        log_factorial = (
            torch.arange(
                1,
                self.event_shape[0] + 1,
                device=self.total_count.device,
                dtype=torch.float,
            )
            .log()
            .cumsum(0)
        )
        t_idx = (self.total_count.long() - 1).clamp_min(0)
        g_idx = (self.given_count.long() - 1).clamp_min(0)
        tmg_idx = (self.total_count.long() - self.given_count.long() - 1).clamp_min(0)
        return log_factorial[t_idx] - log_factorial[g_idx] - log_factorial[tmg_idx]

    @property
    def mean(self):
        len_mask = self.total_count.unsqueeze(-1) <= torch.arange(
            self.event_shape[0], device=self.total_count.device
        )
        return (
            (self.given_count / self.total_count.clamp_min(1.0))
            .unsqueeze(-1)
            .expand(self.batch_shape + self.event_shape)
        ).masked_fill(len_mask, 0.0)

    @property
    def variance(self):
        return self.mean * (1 - self.mean)

    @property
    def _natural_params(self):
        return self.total_count.new_zeros(self.batch_shape + self.event_shape)

    def _log_normalizer(self, logits):
        if (logits == 0).all():
            raise RuntimeError("Logits invalid")
        return self.log_partition

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(
            SimpleRandomSamplingWithoutReplacement, _instance
        )
        batch_shape = list(batch_shape)
        new.given_count = self.given_count.expand(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)

        if "log_partition" in self.__dict__:
            new.log_partition = self.log_partition.expand(batch_shape)

        super(SimpleRandomSamplingWithoutReplacement, new).__init__(
            torch.Size(batch_shape), self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size([])):
        sample_shape = torch.Size(sample_shape)
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            total_count = self.total_count.expand(shape[:-1])
            given_count = self.given_count.expand(shape[:-1])
            b = simple_random_sampling_without_replacement(
                total_count, given_count, self.event_shape[0]
            )
        return b

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (-self.log_partition).expand(value.shape[:-1])

