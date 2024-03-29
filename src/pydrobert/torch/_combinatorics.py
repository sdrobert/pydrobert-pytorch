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

from typing import Any, Optional, Tuple, Union, overload

import torch

from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from . import argcheck
from ._compat import script, trunc_divide


@script
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
    out_size
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
        self,
        given_count: torch.Tensor,
        tmax: int,
        total_count: Optional[torch.Tensor] = None,
    ) -> None:
        tmax = argcheck.is_nat(tmax, "tmax")
        given_count = argcheck.is_nonnegt(given_count, "given_count")
        if total_count is None:
            total_count_mask = torch.zeros(
                1, dtype=torch.bool, device=given_count.device
            )
        else:
            total_count = argcheck.is_nonnegt(total_count, "total_count")
            total_count_mask = total_count.unsqueeze(-1) <= torch.arange(
                tmax, device=total_count.device
            )
        super().__init__()
        self.given_count, self.total_count_mask = given_count, total_count_mask

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

    Notes
    -----
    When the maximum `length` exceeds :obj:`20`, the implementation uses the recursion
    defined in [howard1972]_.
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
        x = x.cumprod(0)
        binom = trunc_divide(x[length], x[count] * x[length_m_count])
        binom.masked_fill_(length_m_count == -1, 0)
    return binom


# why bother with overloads? JIT. torch.dtype is technically an int right now. Later
# versions of pytorch are able to quietly convert the type, but not 1.5.1
# https://github.com/pytorch/pytorch/issues/65607


@overload
def enumerate_vocab_sequences(
    length: int,
    vocab_size: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    ...


@script
def enumerate_vocab_sequences(
    length: int,
    vocab_size: int,
    device: torch.device = torch.device("cpu"),
    dtype: int = torch.long,
) -> torch.Tensor:
    """Enumerate all sequences of a finite range of values of a fixed length
    
    This function generalizes :func:`enumerate_binary_sequences` to any positive
    vocabulary size. Each step in each sequence takes on a value from 0-`vocab_size`

    Parameters
    ----------
    length
        The non-negative length of the vocab sequence.
    vocab_size
        The positive number of values in the vocabulary.
    device
        What device to return the tensor on.
    dtype
        The data type of the returned tensor.
    
    Returns
    -------
    support : torch.Tensor
        A tensor of shape ``(vocab_size ** length, length)`` of all possible sequences
        with that vocabulary. The sequences are ordered such that all configurations
        where ``support[s, t] > 0`` must follow those where ``support[s', t] == 0``
        (i.e. it implies ``s' < s``). Therefore all sequences of length ``length
        - x`` are contained in ``support[2 ** (length - x), :length - x]``.
    """
    if length < 0:
        raise RuntimeError(f"length must be non-negative, got {length}")
    if vocab_size <= 0:
        raise RuntimeError(f"vocab_size must be positive, got {vocab_size}")
    support = torch.empty(
        (length, int(vocab_size ** length)), device=device, dtype=dtype
    )
    range_ = torch.arange(vocab_size, device=device, dtype=dtype).view(1, vocab_size, 1)
    for t in range(length):
        support.view(length, int(vocab_size ** t), vocab_size, -1)[
            length - t - 1
        ] = range_
    return support.T.contiguous()


@overload
def enumerate_binary_sequences(
    length: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    ...


def enumerate_binary_sequences(
    length: int, device: torch.device = torch.device("cpu"), dtype: int = torch.long,
) -> torch.Tensor:
    """Enumerate all binary sequences of a fixed length
    
    Parameters
    ----------
    length 
        The non-negative length of the binary sequences.
    device
        What device to return the tensor on.
    dtype
        The data type of the returned tensor.
    
    Returns
    -------
    support : torch.Tensor
        A tensor of shape ``(2 ** length, length)`` of all possible binary sequences of
        length `length`. The sequences are ordered such that all configurations where
        ``support[s, t] == 1`` must follow those where ``support[s', t] == 0`` (i.e. it
        implies ``s' < s``). Therefore all binary sequences of length ``length - x`` are
        contained in ``support[2 ** (length - x), :length - x]``.
    
    Examples
    --------
    >>> support = enumerate_binary_sequences(3)
    >>> print(support)
    tensor([[0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]])
    >>> print(support[:4, :2])
    tensor([[0, 0],
        [1, 0],
        [0, 1],
        [1, 1]])
    """
    return enumerate_vocab_sequences(length, 2, device, dtype)


@overload
def enumerate_binary_sequences_with_cardinality(
    length: int,
    count: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    ...


@overload
def enumerate_binary_sequences_with_cardinality(
    length: torch.Tensor, count: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...


@script
def _enumerate_binary_sequences_with_cardinality_int(
    length: int, count: int, device: torch.device, dtype: int
) -> torch.Tensor:
    support = enumerate_binary_sequences(length, device, dtype)
    support = support[support.sum(1) == count]
    return support


@script
def _enumerate_binary_sequences_with_cardinality_tensor(
    length: torch.Tensor, count: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = length.device
    length_ = int(length.max().item())
    length, count = torch.broadcast_tensors(length, count)
    binom = binomial_coefficient(length, count)
    binom_ = int(binom.max().item())
    # _enumerate_binary_sequences outputs sequences with b_t = 1 only after all
    # sequences with b_t = 0. We therefore capture all the combos for a given length
    # by limiting ourselves to the indices up to 2 ** length.
    N = int(2 ** length_)
    support = enumerate_binary_sequences(length_, device, length.dtype)
    support = torch.cat([support, torch.empty_like(support)])
    range_ = torch.arange(2 * N, device=device).expand(binom.shape + (2 * N,))
    pad = (range_ >= N) & (range_ < N + (binom_ - binom).unsqueeze(-1))
    keep = (range_ < (2 ** length).unsqueeze(-1)) & (
        support.sum(-1).expand(binom.shape + (2 * N,)) == count.unsqueeze(-1)
    )
    support = support.expand(binom.shape + (-1, -1))[pad | keep]
    support = support.view(binom.shape + (binom_, length_))
    return support, binom


def enumerate_binary_sequences_with_cardinality(
    length: Any,
    count: Any,
    device: torch.device = torch.device("cpu"),
    dtype: int = torch.long,
) -> Any:
    r"""Enumerate the configurations of binary sequences with fixed sum
    
    Parameters
    ----------
    length
        The number of elements in the binary sequence. Either a tensor or an int. Must
        be the same type as `count`. If a tensor, must broadcast with `count`.
    count
        The number of elements with value 1. Either a tensor or an int. Must be the same
        type as `length`. If a tensor, must broadcast with `length`.
    device
        If `length` and `count` are integers, `device` specifies the device to return
        the tensor on. Otherwise the device of `length` is used.
    dtype
        If `length` and `count` are integers, `dtype` specifies the return type of
        the tensor. Otherwise the type of `length` is used.
    
    
    Returns
    -------
    support : torch.Tensor or tuple of torch.Tensor
        If `length` and `count` are both integers, `support` is a tensor of shape
        ``(N, length)`` where :math:`N = \binom{length}{count}` is the number of unique
        binary sequence configurations of length `length` such that for any ``n``,
        ``support[n].sum() == count``.

        If `length` and `count` are both long tensors, `support` is a tuple of tensors
        ``support_, binom`` where `support_` is of shape ``(B*, N_, length_)`` and
        `binom` is of shape ``(B*)``. ``B*`` refers to the broadcasted shape of `length`
        and `count`, ``N_`` is the maximum value in `binom`, and ``length_`` is the
        maximum value in ``length_``. For multi-index ``b``, ``support[b]`` stores the
        unique binary sequence configurations for ``length[b]`` and ``count[b]``.
        ``binom[b]`` stores the number of unique configurations for ``length[b]`` and
        ``count[b]``, which is always :math:`\binom{length[b]}{count[b]}`. Sequences
        are right-padded to the maximum length and count: for index ``b``, only values
        in ``support[b, :binom[b], :length[b]]`` are valid.
    
    Warnings
    --------
    The size of the returned support grows exponentially with `length`.
    """
    if isinstance(length, torch.Tensor) and isinstance(count, torch.Tensor):
        return _enumerate_binary_sequences_with_cardinality_tensor(length, count)
    elif isinstance(length, int) and isinstance(count, int):
        return _enumerate_binary_sequences_with_cardinality_int(
            length, count, device, dtype
        )
    else:
        raise RuntimeError("length and count must both be tensors or ints")


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
    validate_args
    
    Notes
    -----
    The support can only be enumerated if all elements of `total_count` are equal;
    likewise for `given_count`.
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
        validate_args: Optional[bool] = None,
    ):
        device = None
        if isinstance(given_count, torch.Tensor):
            device = given_count.device
            if (
                isinstance(total_count, torch.Tensor)
                and given_count.device != total_count.device
            ):
                raise ValueError(
                    "given_count and total_count must be on the same device"
                )
        elif isinstance(total_count, torch.Tensor):
            device = total_count.device
        given_count = torch.as_tensor(given_count, device=device)
        total_count = torch.as_tensor(total_count, device=device)
        total_count_max = int(total_count.max().item())
        if out_size is None:
            out_size = total_count_max
        given_count, total_count = torch.broadcast_tensors(given_count, total_count)
        batch_shape = given_count.size()
        event_shape = torch.Size([out_size])
        self.total_count, self.given_count = total_count, given_count
        super().__init__(batch_shape, event_shape, validate_args)
        if self._validate_args:
            given_count = argcheck.is_nonnegt(given_count, "given_count")
            total_count = argcheck.is_nonnegt(total_count, "total_count")
            argcheck.is_lte(given_count, total_count, "given_count", "total_count")
            argcheck.is_gte(out_size, total_count, "out_size", "total_count")

    @constraints.dependent_property
    def support(self):
        return BinaryCardinalityConstraint(
            self.given_count, self.event_shape[0], self.total_count
        )

    @property
    def has_enumerate_support(self) -> bool:
        return (
            (self.total_count == self.total_count.flatten()[0]).all()
            & (self.given_count == self.given_count.flatten()[0]).all()
        ).item()

    def enumerate_support(self, expand=True) -> torch.Tensor:
        if not self.has_enumerate_support:
            raise NotImplementedError(
                "total_count must all be equal and given_count must all be equal to "
                "enumerate support"
            )
        total = self.total_count.flatten()[0].item()
        given = self.given_count.flatten()[0].item()
        support = enumerate_binary_sequences_with_cardinality(
            total, given, self.total_count.device, dtype=torch.float
        )
        out_size = self.event_shape[0]
        if out_size != total:
            support = torch.nn.functional.pad(support, (0, out_size - total))
        support = support.view((-1,) + (1,) * len(self.batch_shape) + (out_size,))
        if expand:
            support = support.expand((-1,) + self.batch_shape + (out_size,))
        return support

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

