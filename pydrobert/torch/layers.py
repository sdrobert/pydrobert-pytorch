# Copyright 2019 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Common neural layers from the literature not included in pytorch.nn

Notes
-----
Though loss functions could be considered neural layers, because they are
specific to training, they are included in ``pydrobert.torch.training`` instead
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import torch

from future.utils import with_metaclass

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"
__all__ = [
    'ConcatSoftAttention',
    'DotProductSoftAttention',
    'GeneralizedDotProductSoftAttention',
    'GlobalSoftAttention',
    'MultiHeadedAttention',
]

# XXX(sdrobert): a quick note on style. pytorch doesn't tend to protect its
# read-only attributes using private members, so we do the same


class GlobalSoftAttention(with_metaclass(abc.ABCMeta, torch.nn.Module)):
    r'''Parent class for soft attention mechanisms on an entire input sequence

    Global soft attention mechansims [bahdanau2015]_ are a way of getting rid
    of one variable-length sequence dimension ``T`` in an input `key` using a
    weighted sum of a tensor `value` that is informed by some other tensor,
    `query`. The weights are dictated by the function ``score(query, key)``.
    Usually, this is in the context of encoder-decoder architectures, which is
    explained here.

    Assume `query` is a tensor of shape ``(num_batch, query_size)``
    representing a single hidden state of a decoder RNN. Assume `key` is a
    tensor of shape ``(T, num_batch, key_size)`` representing the encoder
    output, ``dim == 0`` to specify that the variable-length dimension of `key`
    is the zero-th dimension, and ``value == key``. The output `out` will be a
    tensor of shape ``(num_batch, key_size)``. Letting :math:`t` index the
    `dim`-th dimension:

        .. math::

            out = \sum_t a_t value_t

    ``a`` is the attention vector. In our example, ``a`` will be of shape
    ``(T, num_batch)``. ``a`` is the result of a softmax over the `dim`-th
    dimension of another tensor ``e`` of shape ``(T, num_batch)`` with an
    optional `mask`

    .. math::

        a = softmax(e * mask - (1 - mask) \infty, dim)

    `mask` (if specified) is of shape ``(T, num_batch)`` and will set ``a`` to
    zero wherever the mask is zero. `mask` can be used to indicate padded
    values when `key` consists of variable-length sequences.

    ``e`` is the result of a score function over `key` and `query`

    .. math::

        e = score(query, key)

    ``score()`` is implemented by subclasses of :class:`GlobalSoftAttention`

    The signature when calling an instance this module is:

        attention(query, key, value[, mask])

    Parameters
    ----------
    query_size : int
        The length of the last dimension of the `query` argument
    key_size : int
        The length of the last dimension of the `key` argument
    dim : int, optional
        The sequence dimension of the `key` argument

    Attributes
    ----------
    query_size, key_size, dim : int

    Examples
    --------

    A simple auto-regressive decoder using soft attention on encoder outputs
    with "concat"-style attention

    >>> T, num_batch, encoded_size, hidden_size = 100, 5, 30, 124
    >>> num_classes, start, eos, max_decoder_steps = 20, -1, 0, 100
    >>> encoded_lens = torch.randint(1, T + 1, (num_batch,))
    >>> len_mask = torch.where(
    ...     torch.arange(T).unsqueeze(-1) < encoded_lens,
    ...     torch.tensor(1),
    ...     torch.tensor(0),
    ... )
    >>> encoded = torch.randn(T, num_batch, encoded_size)
    >>> rnn = torch.nn.RNNCell(encoded_size + 1, hidden_size)
    >>> ff = torch.nn.Linear(hidden_size, num_classes)
    >>> attention = ConcatSoftAttention(hidden_size, encoded_size)
    >>> h = torch.zeros((num_batch, hidden_size))
    >>> y = torch.full((1, num_batch), -1, dtype=torch.long)
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
    :ref:`Advanced Attention and Transformer Networks`
        :class:`GlobalSoftAttention` is compatible with a variety of inputs.
        This tutorial gives a toy transformer network to illustrate
        broadcasting semantics
    '''

    def __init__(self, query_size, key_size, dim=0):
        super(GlobalSoftAttention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.dim = dim

    @abc.abstractmethod
    def score(self, query, key):
        '''Calculate the score function over the entire input

        This is implemented by subclasses of :class:`GlobalSoftAttention`

        ``query.unsqueeze(self.dim)[..., 0]`` broadcasts with ``value[...,
        0]``. The final dimension of `query` is of length ``self.query_size``
        and the final dimension of `key` should be of length ``self.key_size``

        Parameters
        ----------
        query : torch.FloatTensor
        key : torch.FloatTensor

        Returns
        -------
        e : torch.FloatTensor
            Of the same shape as the above broadcasted tensor
        '''
        raise NotImplementedError()

    def forward(self, query, key, value, mask=None):
        if query.dim() + 1 != key.dim():
            raise ValueError('query must have one fewer dimension than key')
        if key.dim() != value.dim():
            raise ValueError(
                "key must have same number of dimensions as value")
        if query.shape[-1] != self.query_size:
            raise ValueError('Last dimension of query must match query_size')
        if key.shape[-1] != self.key_size:
            raise ValueError('Last dimension of key must match key_size')
        key_dim = key.dim()
        if self.dim > key_dim - 2 or self.dim < -key_dim + 1:
            raise ValueError(
                'dim must be in the range [{}, {}]'
                ''.format(-key_dim + 1, key_dim - 2)
            )
        e = self.score(query, key)
        if mask is not None:
            e = e.masked_fill(mask.eq(0), -float('inf'))
        a = torch.nn.functional.softmax(e, self.dim)
        c = (a.unsqueeze(-1) * value).sum(self.dim)
        return c

    def extra_repr(self):
        return 'query_size={}, key_size={}, dim={}'.format(
            self.query_size, self.key_size, self.dim)

    def reset_parameters(self):
        pass


class DotProductSoftAttention(GlobalSoftAttention):
    r'''Global soft attention with dot product score function

    From [luong2015]_, the score function for this attention mechanism is

    .. math::

        e = scale\_factor \sum_i query_i key_i

    Where :math:`i` indexes the last dimension of both the query and key

    Parameters
    ----------
    size : int
        Both the query and key size
    dim : int, optional
    scale_factor : float, optional
        A floating point to multiply the each :math:`e` with. Usually
        1, but if set to :math:`1 / size`, you'll get the scaled dot-product
        attention of [vaswani2017]_

    Attributes
    ----------
    query_size, key_size, dim : int
    scale_factor : float

    See Also
    --------
    GlobalSoftAttention
        For a description of how to call this module, how it works, etc.
    '''

    def __init__(self, size, dim=0, scale_factor=1.):
        super(DotProductSoftAttention, self).__init__(size, size, dim)
        self.scale_factor = scale_factor

    def score(self, query, key):
        query = query.unsqueeze(self.dim)
        return (query * key).sum(-1) * self.scale_factor

    def extra_repr(self):
        return 'size={}, dim={}'.format(self.query_size, self.dim)


class GeneralizedDotProductSoftAttention(GlobalSoftAttention):
    r'''Dot product soft attention with a learned matrix in between

    The "general" score function from [luong2015]_, the score function for this
    attention mechanism is

    .. math::

        e = \sum_q query_q \sum_k W_{qk} key_k

    For some learned matrix :math:`W`. :math:`q` indexes the last dimension
    of `query` and :math:`k` the last dimension of `key`

    Parameters
    ----------
    query_size : int
    key_size : int
    dim : int, optional
    bias : bool, optional
        Whether to add a bias term ``b``: :math:`W key + b`

    Attributes
    ----------
    query_size, key_size, dim : int
    W : torch.nn.Linear
        The matrix :math:`W`

    See Also
    --------
    GlobalSoftAttention
        For a description of how to call this module, how it works, etc.
    '''

    def __init__(self, query_size, key_size, dim=0, bias=False):
        super(GeneralizedDotProductSoftAttention, self).__init__(
            query_size, key_size, dim)
        self.W = torch.nn.Linear(key_size, query_size, bias=bias)

    def score(self, query, key):
        Wkey = self.W(key)
        query = query.unsqueeze(self.dim)
        return (query * Wkey).sum(-1)

    def reset_parameters(self):
        self.W.reset_parameters()


class ConcatSoftAttention(GlobalSoftAttention):
    r'''Attention where query and key are concatenated, then fed into an MLP

    Proposed in [luong2015]_, though quite similar to that proposed in
    [bahdanau2015]_, the score function for this layer is:

    .. math::

        e = \sum_i v_i \tanh(\sum_c W_{ic} [query, key]_c)

    For some learned matrix :math:`W` and vector :math:`v`, where
    :math:`[query, key]` indicates concatenation along the last axis. `query`
    and `key` will be expanded to fit their broadcast dimensions. :math:`W`
    has shape ``(inter_size, key_size)`` and :math:`v` has shape
    ``(hidden_size,)``

    Parameters
    ----------
    query_size : int
    key_size : int
    dim : int, optional
    bias : bool, optional
        Whether to add bias term ``b`` :math:`W [query, key] + b`
    hidden_size : int, optional

    Attributes
    ----------
    query_size, key_size, dim, hidden_size : int
    W : torch.nn.Linear
        The matrix :math:`W`
    v : torch.nn.Linear
        The vector :math:`v` as a single-row matrix

    See Also
    --------
    GlobalSoftAttention
        For a description of how to call this module, how it works, etc.
    '''

    def __init__(
            self, query_size, key_size, dim=0, bias=False, hidden_size=1000):
        super(ConcatSoftAttention, self).__init__(query_size, key_size, dim)
        self.hidden_size = hidden_size
        self.W = torch.nn.Linear(
            query_size + key_size, hidden_size, bias=bias)
        # there's no point in a bias for v. It'll just be absorbed by the
        # softmax later. You could add a bias after the tanh layer, though...
        self.v = torch.nn.Linear(hidden_size, 1, bias=False)

    def score(self, query, key):
        query = query.unsqueeze(self.dim)
        query_wo_last, key_wo_last = torch.broadcast_tensors(
            query[..., 0], key[..., 0])
        query, _ = torch.broadcast_tensors(query, query_wo_last.unsqueeze(-1))
        key, _ = torch.broadcast_tensors(key, key_wo_last.unsqueeze(-1))
        cat = torch.cat([query, key], -1)
        Wcat = self.W(cat)
        return self.v(Wcat).squeeze(-1)

    def reset_parameters(self):
        self.W.reset_parameters()
        self.v.reset_parameters()

    def extra_repr(self):
        s = super(ConcatSoftAttention, self).extra_repr()
        s += ', hidden_size={}'.format(self.hidden_size)
        return s


class MultiHeadedAttention(torch.nn.Module):
    r'''Perform attention over a number of heads, concatenate, and project

    Multi-headed attention was proposed in [vaswani2017]_. It can be considered
    a wrapper around standard :class:`GlobalSoftAttention` that results in a
    similar output as a regular attention layer. The idea is to replicate
    transformed versions of the `query`, `key`, and `value` `num_heads` times.
    Letting :math:`h` index the head:

    .. math::

        query_h = W^Q_h query \\
        key_h = W^K_h key \\
        value_h = W^V_h value

    If `query` is of shape ``(..., query_size)``, :math:`W^Q_h` is a learned
    matrix of shape ``(query_size, d_q)`` that acts on the final dimension of
    `query`. Likewise, :math:`W^K_h` is of shape ``(key_size, d_k)`` and
    :math:`W^V_h` is of shape ``(value_size, d_v)``.

    Each head is then determined via a wrapped :class:`GlobalSoftAttention`
    instance, `single_head_attention`:

    .. math::

        head_h = single\_head\_attention(query_h, key_h, value_h, mask)

    Where `mask` is repeated over all :math:`h`.

    Since each :math:`head_h` has the same shape, they can be concatenated
    along the last dimension to get the tensor :math:`cat` of shape
    ``(..., d_v * num_heads)``, which is linearly transformed into the output

    .. math::

        out = W^C cat

    With a learnable matrix :math:`W^C` of shape ``(d_v * num_heads,
    out_size)``. `out` has a shape ``(..., out_size)``

    This module has the following signature when called

        attention(query, key, value[, mask])

    Parameters
    ----------
    query_size : int
        The size of the last dimension of the `query` being passed to this
        module (not the size of a head's query)
    key_size : int
        The size of the last dimension of the `key` being passed to this
        module (not the size of a head's key)
    value_size : int
        The size of the last dimension of the `value` being passed to this
        module (not the size of a head's value)
    num_heads : int
        The number of heads to spawn
    single_head_attention : GlobalSoftAttention
        An instance of a subclass of :class:`GlobalSoftAttention` responsible
        for processing a head. `single_head_attention` attention will be used
        to derive the sequence dimension (``dim``) of `key` via
        ``single_head_attention.dim``, the size of a head's query ``d_k`` via
        ``single_head_attention.query_size``, and the size of a head's key via
        ``single_head_attention.key_size``
    out_size : int, optional
        The size of the last dimension of `out`. If unset, the default is to
        match `value_size`
    d_v : int, optional
        The size of the last dimension of a head's value. If unset, will
        default to ``max(1, value_size // num_heads)``
    bias_WQ : bool, optional
        Whether to add a bias term to :math:`W^Q`
    bias_WK : bool, optional
        Whether to add a bias term to :math:`W^K`
    bias_WV : bool, optional
        Whether to add a bias term to :math:`W^V`
    bias_WC : bool, optional
        Whether to add a bias term to :math:`W^C`

    Attributes
    ----------
    query_size, key_size, value_size, out_size, num_heads, dim : int
    d_q, d_k, d_v : int
    single_head_attention : GlobalSoftAttention
    WQ, WK, WV, WC : torch.nn.Linear
        Matrices :math:`W^Q`, :math:`W^K`, :math:`W^V`, and :math:`W^C`
    '''

    def __init__(
            self, query_size, key_size, value_size, num_heads,
            single_head_attention, out_size=None, d_v=None,
            bias_WQ=False, bias_WK=False, bias_WV=False, bias_WC=False):
        super(MultiHeadedAttention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.out_size = value_size if out_size is None else out_size
        self.num_heads = num_heads
        self.single_head_attention = single_head_attention
        # we don't keep these in sync in case someone's using
        # single_head_attention
        self.dim = single_head_attention.dim
        self.d_q = single_head_attention.query_size
        self.d_k = single_head_attention.key_size
        self.d_v = max(1, value_size // num_heads) if d_v is None else d_v
        self.WQ = torch.nn.Linear(
            query_size, num_heads * self.d_q, bias=bias_WQ)
        self.WK = torch.nn.Linear(
            key_size, num_heads * self.d_k, bias=bias_WK)
        self.WV = torch.nn.Linear(
            value_size, num_heads * self.d_v, bias=bias_WV)
        self.WC = torch.nn.Linear(
            self.d_v * num_heads, self.out_size, bias=bias_WC)
        single_head_attention.reset_parameters()

    def forward(self, query, key, value, mask=None):
        query_shape = tuple(query.shape)
        key_shape = tuple(key.shape)
        value_shape = tuple(value.shape)
        # we check dim here because we want to turn it into a positive
        # index before passing it to the head. This is because there's going
        # to be one more dimension to query, key, and value at the end that
        # represents the `num_heads` dimension, and we don't want negative
        # indices being wrongly offset because of it
        key_dim = len(key_shape)
        if self.dim > key_dim - 2 or self.dim < -key_dim + 1:
            raise ValuerError(
                'dim must be in the range [{}, {}]'
                ''.format(-key_dim + 1, key_dim - 2)
            )
        dim = (self.dim + key_dim) % key_dim
        query_heads = (
            self.WQ(query)
                .view(*(query_shape[:-1] + (self.num_heads, self.d_q)))
        )
        key_heads = (
            self.WK(key)
                .view(*(key_shape[:-1] + (self.num_heads, self.d_k)))
        )
        value_heads = (
            self.WV(value)
                .view(*(value_shape[:-1] + (self.num_heads, self.d_v)))
        )
        if mask is not None:
            mask = mask.unsqueeze(-2)
        old_dim = self.single_head_attention.dim
        try:
            self.single_head_attention.dim = dim
            cat = self.single_head_attention(
                query_heads, key_heads, value_heads, mask)
        finally:
            self.single_head_attention.dim = old_dim
        cat = cat.view(*(tuple(cat.shape[:-2]) + (self.num_heads * self.d_v,)))
        return self.WC(cat)

    def reset_parameters(self):
        self.WQ.reset_parameters()
        self.WK.reset_parameters()
        self.WV.reset_parameters()
        self.WC.reset_parameters()
        self.single_head_attention.reset_parameters()

    def extra_repr(self):
        return (
            'query_size={}, key_size={}, value_size={}, out_size={}, dim={}, '
            'num_heads={}, d_q={}, d_k={}, d_v={}'.format(
                self.query_size, self.key_size, self.value_size, self.out_size,
                self.dim, self.num_heads, self.d_q, self.d_k, self.d_v)
        )
