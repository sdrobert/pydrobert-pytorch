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

from typing import Optional

import torch

from . import argcheck
from ._compat import broadcast_shapes, script, unflatten
from ._wrappers import proxy


class GlobalSoftAttention(torch.nn.Module, metaclass=abc.ABCMeta):
    r"""Parent class for soft attention mechanisms on an entire input sequence

    Global soft attention mechansims [bahdanau2015]_ are a way of getting rid of one
    variable-length sequence dimension ``T`` in an input `key` using a weighted sum of a
    tensor `value` that is informed by some other tensor, `query`. The weights are
    dictated by the function :func:`score`. Usually, this is in the context of
    encoder-decoder architectures, which is explained here.

    Assume `query` is a tensor of shape ``(batch_size, query_size)`` representing a
    single hidden state of a decoder RNN. Assume `key` is a tensor of shape ``(T,
    batch_size, key_size)`` representing the encoder output, ``dim == 0`` to specify
    that the variable-length dimension of `key` is the zero-th dimension, and ``value ==
    key``. The output `out` will be a tensor of shape ``(batch_size, key_size)``.
    Letting :math:`t` index the `dim`-th dimension:

        .. math::

            out = \sum_t a_t value_t

    ``a`` is the attention vector. In our example, ``a`` will be of shape ``(T,
    batch_size)``. ``a`` is the result of a softmax over the `dim`-th dimension of
    another tensor ``e`` of shape ``(T, batch_size)`` with an optional `mask`

    .. math::

        a = softmax(e * mask - (1 - mask) \infty, dim)

    `mask` (if specified) is of shape ``(T, batch_size)`` and will set ``a`` to zero
    wherever the mask is zero. `mask` can be used to indicate padded values when `key`
    consists of variable-length sequences.

    ``e`` is the result of a score function over `key` and `query`

    .. math::

        e = score(query, key)

    :func:`score` is implemented by subclasses of :class:`GlobalSoftAttention`.

    Parameters
    ----------
    query_size : int
        The length of the last dimension of the `query` argument
    key_size : int
        The length of the last dimension of the `key` argument
    dim : int, optional
        The sequence dimension of the `key` argument
    
    Call Parameters
    ---------------
    query : torch.Tensor
        A tensor of shape ``(A*, query_size)`` representing the queries. ``(A*)`` must
        broadcast with ``(B*, C*)`` from `key`, `value`, and `mask`.
    key : torch.Tensor
        A tensor of shape ``(B*, T, C*, key_size)`` representing the keys. ``(B*, C*)``
        must broadcast with ``(A*)`` from `query`.
    value : torch.Tensor
        A tensor of shape ``(B*, T, C*, D*)`` representing the values. ``(B*, C*)``
        must broadcast with ``(A*)`` from `query`.
    mask : Optional[torch.Tensor]
        An optional boolean tensor of shape ``(B*, T, C*)`` which indicates which values
        of the key should be kept (:obj:`False` means zero-out). If unset, assumed to
        be entirely :obj:`True`.
    
    Returns
    -------
    out : torch.Tensor
        The output tensor of shape ``(E*, D*)``, where ``(E*)`` is the result of
        broadcasting ``(A*)`` with ``(B*, C*)``.

    Examples
    --------
    A simple auto-regressive decoder using soft attention on encoder outputs
    with "concat"-style attention

    >>> T, batch_size, encoded_size, hidden_size = 100, 5, 30, 124
    >>> num_classes, start, eos, max_decoder_steps = 20, -1, 0, 100
    >>> encoded_lens = torch.randint(1, T + 1, (batch_size,))
    >>> len_mask = torch.where(
    ...     torch.arange(T).unsqueeze(-1) < encoded_lens,
    ...     torch.tensor(1),
    ...     torch.tensor(0),
    ... )
    >>> encoded = torch.randn(T, batch_size, encoded_size)
    >>> rnn = torch.nn.RNNCell(encoded_size + 1, hidden_size)
    >>> ff = torch.nn.Linear(hidden_size, num_classes)
    >>> attention = ConcatSoftAttention(hidden_size, encoded_size)
    >>> h = torch.zeros((batch_size, hidden_size))
    >>> y = torch.full((1, batch_size), -1, dtype=torch.long)
    >>> for _ in range(max_decoder_steps):
    >>>     if y[-1].eq(eos).all():
    >>>         break
    >>>     context = attention(h, encoded, encoded, len_mask)
    >>>     cat = torch.cat([context, y[-1].unsqueeze(-1).float()], 1)
    >>>     h = rnn(cat)
    >>>     logit = ff(h)
    >>>     y_next = logit.argmax(-1).masked_fill(y[-1].eq(eos), eos)
    >>>     y = torch.cat([y, y_next.unsqueeze(0)], 0)

    See Also
    --------
    :ref:`advanced-attn`
        :class:`GlobalSoftAttention` is compatible with a variety of inputs.
        This tutorial gives a toy transformer network to illustrate
        broadcasting semantics.
    """

    __constants__ = ["query_size", "key_size", "dim"]

    query_size: int
    key_size: int
    dim: int

    def __init__(self, query_size: int, key_size: int, dim: int = 0):
        query_size = argcheck.is_nat(query_size, name="query_size")
        key_size = argcheck.is_nat(key_size, name="key_size")
        dim = argcheck.is_int(dim, name="dim")
        super().__init__()
        self.query_size, self.key_size, self.dim = query_size, key_size, dim

    @abc.abstractmethod
    def score(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Calculate the score function over the entire input

        This is implemented by subclasses of :class:`GlobalSoftAttention`. Computes::

            e = score(query, key)
        
        from the class description.

        Parameters
        ----------
        query
            A tensor of shape ``(A*, query_size)`` representing the queries. ``(A*)``
            must broadcast with ``(B*, C*)`` from `key`.
        key
            A tensor of shape ``(B*, T, C*, key_size)`` representing the keys. ``(B*,
            C*)`` must broadcast with ``(A*)`` from `query`.

        Returns
        -------
        torch.Tensor
            A tensor of scores of shape ``(E*, T, F*)``, where ``(E*)`` is the
            result of broadcasting ``(A*)`` with ``(B*, C*)``.
        """
        ...

    def check_input(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Check if input is properly formatted, RuntimeError otherwise"""
        key_dim = key.dim()
        if query.dim() != key_dim - 1:
            raise ValueError("query must have one fewer dimension than key")
        if key_dim != value.dim():
            raise ValueError("key must have same number of dimensions as value")
        if query.shape[-1] != self.query_size:
            raise ValueError("Last dimension of query must match query_size")
        if key.shape[-1] != self.key_size:
            raise ValueError("Last dimension of key must match key_size")
        if self.dim > key_dim - 2 or key_dim == -1 or self.dim < -key_dim + 1:
            raise ValueError(
                f"dim must be in the range [{-key_dim + 1}, {key_dim - 2}] and not -1"
            )
        e_shape = broadcast_shapes(query.unsqueeze(self.dim).shape[:-1], key.shape[:-1])
        if mask is not None:
            broadcast_shapes(e_shape, mask.shape)
        broadcast_shapes(e_shape + (1,), value.shape)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            # tracing can't handle calls with None arguments, so we make a
            # non-threatening mask to call with
            mask_ = torch.ones((1,), device=query.device, dtype=torch.bool)
            self.check_input(query, key, value, mask_)
        else:
            self.check_input(query, key, value, mask)
        e = self.score(query, key)
        if mask is not None:
            e = e.masked_fill(~mask, -float("inf"))
        a = torch.nn.functional.softmax(e, self.dim)
        return (a.unsqueeze(-1) * value).sum(self.dim)

    def extra_repr(self) -> str:
        return "query_size={}, key_size={}, dim={}".format(
            self.query_size, self.key_size, self.dim
        )

    def reset_parameters(self) -> None:
        pass


class DotProductSoftAttention(GlobalSoftAttention):
    r"""Global soft attention with dot product score function

    From [luong2015]_, the score function for this attention mechanism is

    .. math::

        e = scale\_factor \sum_i query_i key_i

    Where :math:`i` indexes the last dimension of both the query and key

    Parameters
    ----------
    size
        The size of the final dimension of both `query` and `key`.
    dim
    scale_factor
        A floating point to multiply the each :math:`e` with. Usually :obj:`1`, but if
        set to :math:`1 / size`, you'll get the scaled dot-product attention of
        [vaswani2017]_.
    
    Call Parameters
    ---------------
    query : torch.Tensor
    key : torch.Tensor
    value : torch.Tensor
    mask : Optional[torch.Tensor]
    
    Returns
    -------
    out : torch.Tensor

    See Also
    --------
    GlobalSoftAttention
        For a description of how to call this module, how it works, etc.
    """

    __constants__ = "query_size", "key_size", "dim", "scale_factor"

    scale_factor: float

    def __init__(self, size: int, dim: int = 0, scale_factor: float = 1.0):
        scale_factor = argcheck.is_float(scale_factor, name="scale_factor")
        super().__init__(size, size, dim)
        self.scale_factor = scale_factor

    def score(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        query = query.unsqueeze(self.dim)
        return (query * key).sum(-1) * self.scale_factor

    def extra_repr(self) -> str:
        return super().extra_repr() + f", scale_factor={self.scale_factor}"

    __call__ = proxy(GlobalSoftAttention.forward)


class GeneralizedDotProductSoftAttention(GlobalSoftAttention):
    r"""Dot product soft attention with a learned matrix in between

    The "general" score function from [luong2015]_, the score function for this
    attention mechanism is

    .. math::

        e = \sum_q query_q \sum_k W_{qk} key_k

    For some learned matrix :math:`W`. :math:`q` indexes the last dimension of `query`
    and :math:`k` the last dimension of `key`

    Parameters
    ----------
    query_size
    key_size
    dim
    bias
        Whether to add a bias term ``b``: :math:`W key + b`
    
    Call Parameters
    ---------------
    query : torch.Tensor
    key : torch.Tensor
    value : torch.Tensor
    mask : Optional[torch.Tensor]
    
    Returns
    -------
    out : torch.Tensor

    See Also
    --------
    GlobalSoftAttention
        For a description of how to call this module, how it works, etc.
    """

    def __init__(
        self, query_size: int, key_size: int, dim: int = 0, bias: bool = False
    ):
        bias = argcheck.is_bool(bias, "bias")
        super().__init__(query_size, key_size, dim)
        self.weight = torch.nn.parameter.Parameter(torch.empty(query_size, key_size))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.empty(query_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def score(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        Wkey = torch.nn.functional.linear(key, self.weight, self.bias)
        query = query.unsqueeze(self.dim)
        return (query * Wkey).sum(-1)

    reset_parameters = torch.jit.unused(torch.nn.Linear.reset_parameters)

    __call__ = proxy(GlobalSoftAttention.forward)


@script
def _concat_soft_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    v: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    query = query.unsqueeze(dim)
    shape = list(broadcast_shapes(query.shape[:-1], key.shape[:-1]))
    query = query.expand(shape + [query.size(-1)])
    key = key.expand(shape + [key.size(-1)])
    cat = torch.cat([query, key], -1)
    Wcat = torch.nn.functional.linear(cat, weight, bias)
    tanhWcat = torch.tanh(Wcat)
    return torch.nn.functional.linear(tanhWcat, v.unsqueeze(0), None).squeeze(-1)


class ConcatSoftAttention(GlobalSoftAttention):
    r"""Attention where query and key are concatenated, then fed into an MLP

    Proposed in [luong2015]_, though quite similar to that proposed in [bahdanau2015]_,
    the score function for this layer is:

    .. math::

        e = \sum_i v_i \tanh(\sum_c W_{ic} [query, key]_c)

    For some learned matrix :math:`W` and vector :math:`v`, where :math:`[query, key]`
    indicates concatenation along the last axis. `query` and `key` will be expanded to
    fit their broadcast dimensions. :math:`W` has shape ``(inter_size, key_size)`` and
    :math:`v` has shape ``(hidden_size,)``

    Parameters
    ----------
    query_size
    key_size
    dim
    bias
        Whether to add bias term ``b`` :math:`W [query, key] + b`
    hidden_size
        
    Call Parameters
    ---------------
    query : torch.Tensor
    key : torch.Tensor
    value : torch.Tensor
    mask : Optional[torch.Tensor]
    
    Returns
    -------
    out : torch.Tensor

    See Also
    --------
    GlobalSoftAttention
        For a description of how to call this module, how it works, etc.
    """

    def __init__(
        self,
        query_size: int,
        key_size: int,
        dim: int = 0,
        bias: bool = False,
        hidden_size: int = 1000,
    ):
        hidden_size = argcheck.is_nat(hidden_size, name="hidden_size")
        bias = argcheck.is_bool(bias, name="bias")
        super().__init__(query_size, key_size, dim)
        self.weight = torch.nn.parameter.Parameter(
            torch.empty(hidden_size, query_size + key_size)
        )
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter("bias", None)
        # there's no point in a bias for v. It'll just be absorbed by the
        # softmax later. You could add a bias after the tanh layer, though...
        self.v = torch.nn.parameter.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def score(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return _concat_soft_attention(
            query, key, self.weight, self.bias, self.v, self.dim
        )

    def reset_parameters(self) -> None:
        torch.nn.Linear.reset_parameters(self)
        torch.nn.init.normal_(self.v)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", hidden_size={self.v.size(0)}"

    __call__ = proxy(GlobalSoftAttention.forward)


class MultiHeadedAttention(GlobalSoftAttention):
    r"""Perform attention over a number of heads, concatenate, and project

    Multi-headed attention was proposed in [vaswani2017]_. It can be considered a
    wrapper around standard :class:`GlobalSoftAttention` that also performs
    :class:`GlobalSoftAttention`, but with more parameters. The idea is to replicate
    transformed versions of the `query`, `key`, and `value` `num_heads` times. Letting
    :math:`h` index the head:

    .. math::

        query_h = W^Q_h query \\
        key_h = W^K_h key \\
        value_h = W^V_h value

    If `query` is of shape ``(..., query_size)``, :math:`W^Q_h` is a learned matrix of
    shape ``(query_size, d_q)`` that acts on the final dimension of `query`. Likewise,
    :math:`W^K_h` is of shape ``(key_size, d_k)`` and :math:`W^V_h` is of shape
    ``(value_size, d_v)``. Note here that the last dimension of `value` must also be
    provided in `value_size`, unlike in other attention layers.

    Each head is then determined via a wrapped :class:`GlobalSoftAttention` instance,
    `single_head_attention`:

    .. math::

        head_h = single\_head\_attention(query_h, key_h, value_h, mask)

    Where `mask` is repeated over all :math:`h`.

    Since each :math:`head_h` has the same shape, they can be concatenated along the
    last dimension to get the tensor :math:`cat` of shape ``(..., d_v * num_heads)``,
    which is linearly transformed into the output

    .. math::

        out = W^C cat

    With a learnable matrix :math:`W^C` of shape ``(d_v * num_heads, out_size)``. `out`
    has a shape ``(..., out_size)``

    Parameters
    ----------
    query_size
        The size of the last dimension of the `query` being passed to this module (not
        the size of a head's query).
    key_size
        The size of the last dimension of the `key` being passed to this module (not the
        size of a head's key).
    value_size
        The size of the last dimension of the `value` being passed to this module (not
        the size of a head's value).
    num_heads
        The number of heads to spawn.
    single_head_attention
        An instance of a subclass of :class:`GlobalSoftAttention` responsible for
        processing a head. `single_head_attention` attention will be used to derive the
        sequence dimension (``dim``) of `key` via ``single_head_attention.dim``, the
        size of a head's query ``d_k`` via ``single_head_attention.query_size``, and the
        size of a head's key via ``single_head_attention.key_size``.
    out_size
        The size of the last dimension of `out`. If unset, the default is to match
        `value_size`.
    d_v
        The size of the last dimension of a head's value. If unset, will default to
        ``max(1, value_size // num_heads)``.
    bias_WQ
        Whether to add a bias term to :math:`W^Q`.
    bias_WK
        Whether to add a bias term to :math:`W^K`.
    bias_WV
        Whether to add a bias term to :math:`W^V`.
    bias_WC
        Whether to add a bias term to :math:`W^C`.

    Call Parameters
    ---------------
    query : torch.Tensor
    key : torch.Tensor
    value : torch.Tensor
    mask : Optional[torch.Tensor]
    
    Returns
    -------
    out : torch.Tensor
        The output tensor of shape ``(E*, D*, value_size)``, where ``(E*)``
        is the result of broadcasting ``(A*)`` with ``(B*, C*)``.
    """

    __constants__ = (
        "query_size",
        "key_size",
        "dim",
        "value_size",
        "num_heads",
        "out_size",
        "d_v",
    )

    value_size: int
    num_heads: int
    out_size: int
    d_v: int

    def __init__(
        self,
        query_size: int,
        key_size: int,
        value_size: int,
        num_heads: int,
        single_head_attention: GlobalSoftAttention,
        out_size: Optional[int] = None,
        d_v: Optional[int] = None,
        bias_WQ: bool = False,
        bias_WK: bool = False,
        bias_WV: bool = False,
        bias_WC: bool = False,
    ):
        value_size = argcheck.is_nat(value_size, "value_size")
        if out_size is None:
            out_size = value_size
        else:
            out_size = argcheck.is_nat(out_size, "out_size")
        num_heads = argcheck.is_nat(num_heads, "num_heads")
        if single_head_attention.dim < 0:
            raise ValueError(
                "Negative dimensions are ambiguous for multi-headed attention"
            )
        if d_v is None:
            d_v = max(1, value_size // num_heads)
        else:
            d_v = argcheck.is_nat(d_v, "d_v")
        bias_WQ = argcheck.is_bool(bias_WQ, "bias_WQ")
        bias_WK = argcheck.is_bool(bias_WQ, "bias_WK")
        bias_WV = argcheck.is_bool(bias_WQ, "bias_WV")
        bias_WC = argcheck.is_bool(bias_WC, "bias_WC")
        super().__init__(query_size, key_size, dim=single_head_attention.dim)
        self.value_size, self.out_size, self.num_heads = value_size, out_size, num_heads
        self.single_head_attention = single_head_attention
        self.single_head_attention = single_head_attention
        # we don't keep these in sync in case someone's using
        # single_head_attention
        self.d_q = single_head_attention.query_size
        self.d_k = single_head_attention.key_size
        self.d_v = d_v
        self.WQ = torch.nn.Linear(query_size, num_heads * self.d_q, bias=bias_WQ)
        self.WK = torch.nn.Linear(key_size, num_heads * self.d_k, bias=bias_WK)
        self.WV = torch.nn.Linear(value_size, num_heads * d_v, bias=bias_WV)
        self.WC = torch.nn.Linear(d_v * num_heads, out_size, bias=bias_WC)
        single_head_attention.reset_parameters()

    def check_input(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        # FIXME(sdrobert): TorchScript doesn't currently support calls to super().
        # Replace this when it does. Also surround broadcast_shapes with try/catch
        # when supported
        key_dim = key.dim()
        if query.dim() != key_dim - 1:
            raise RuntimeError("query must have one fewer dimension than key")
        if key_dim != value.dim():
            raise RuntimeError("key must have same number of dimensions as value")
        if query.shape[-1] != self.query_size:
            raise RuntimeError("Last dimension of query must match query_size")
        if key.shape[-1] != self.key_size:
            raise RuntimeError("Last dimension of key must match key_size")
        if self.dim > key_dim - 2 or key_dim == -1 or self.dim < -key_dim + 1:
            raise RuntimeError(
                f"dim must be in the range [{-key_dim + 1}, {key_dim - 2}] and not -1"
            )
        e_shape = broadcast_shapes(query.unsqueeze(self.dim).shape[:-1], key.shape[:-1])
        if mask is not None:
            broadcast_shapes(e_shape, mask.shape)
        broadcast_shapes(e_shape + (1,), value.shape)
        if value.size(-1) != self.value_size:
            raise RuntimeError("Last dimension of value must match value_size")

    @torch.jit.unused
    def score(self, query: torch.Tensor, key: torch.Tensor) -> None:
        raise NotImplementedError(
            "In MultiHeadedAttention, score() is handled by single_head_attention"
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            # avoid issues with calls with None
            # if the dimension is correct, a tensor of shape (1, ...) should always
            # broadcast
            mask_ = torch.ones((1,), device=query.device, dtype=torch.bool)
            if not torch.jit.is_scripting():
                self.check_input(query, key, value, mask_)
        elif not torch.jit.is_scripting():
            self.check_input(query, key, value, mask)
        query_heads = self.WQ(query)
        query_heads = unflatten(query_heads, -1, [self.num_heads, self.d_q])
        key_heads = self.WK(key)
        key_heads = unflatten(key_heads, -1, [self.num_heads, self.d_k])
        value_heads = self.WV(value)
        value_heads = unflatten(value_heads, -1, [self.num_heads, self.d_v])
        if mask is not None:
            mask = mask.unsqueeze(-2)
        cat = self.single_head_attention(query_heads, key_heads, value_heads, mask)
        cat = cat.flatten(-2)
        return self.WC(cat)

    def reset_parameters(self) -> None:
        self.WQ.reset_parameters()
        self.WK.reset_parameters()
        self.WV.reset_parameters()
        self.WC.reset_parameters()
        self.single_head_attention.reset_parameters()

    def extra_repr(self) -> str:
        s = super().extra_repr()
        # rest of info in single_head_attention submodule
        s += ", value_size={}, out_size={}, num_heads={}".format(
            self.value_size, self.out_size, self.num_heads
        )
        return s

    __call__ = proxy(GlobalSoftAttention.forward)
