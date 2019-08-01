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

References
----------
.. [bahdanau2015] D. Bahdanau, K. Cho, and Y. Bengio, "Neural Machine
   Translation by Jointly Learning to Align and Translate.," in 3rd
   International Conference on Learning Representations, ICLR 2015, San Diego,
   CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015.
.. [luong2015] T. Luong, H. Pham, and C. D. Manning, "Effective Approaches to
   Attention-based Neural Machine Translation," in Proceedings of the 2015
   Conference on Empirical Methods in Natural Language Processing, Lisbon,
   Portugal, 2015, pp. 1412-1421.
.. [vaswani2017] A. Vaswani et al., "Attention is All you Need," in Advances in
   Neural Information Processing Systems 30, I. Guyon, U. V. Luxburg, S.
   Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds. Curran
   Associates, Inc., 2017, pp. 5998-6008.
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
    `query`. The weights are dictated by the function ``score(query, key)`.
    Usually, this is in the context of encoder-decoder architectures, which
    is explained here.

    For now, assume `query` is a tensor of shape ``(num_batch, query_size)``
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

        a = softmax(e * mask - (1 - mask) \inf, dim)

    `mask` (if specified) is of shape ``(T, num_batch)`` and will set ``a`` to
    zero wherever the mask is zero. `mask` can be used to indicate padded
    values when `key` consists of variable-length sequences.

    ``e`` is the result of a score function over `key` and `query`

    .. math::

        e = score(query, key)

    ``score()`` is implemented by subclasses of ``GlobalSoftAttention``

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
    query_size : int
    key_size : int
    dim : int

    Extended Summary
    ----------------

    For more complicated attention mechanisms, such as Transformer Networks
    [vaswani2017]_, `query`, `key`, `value`, and `mask` can take on more
    complicated shapes. For Transformer networks, `query` could be of shape
    ``(S, num_batch, query_size)``. `key` and `value` are equal, as before, but
    we've added an extra first dimension to both s.t. their shape is ``(T, 1,
    num_batch, key_size)``. Finally, `mask` is of shape ``(T, S, num_batch)``.
    The result `out` is a tensor of shape ``(S, num_batch, key_size)``.

    `query` is an (n - 1)-dimensional tensor for ``n > 1``. `key` is an
    n-dimensional tensor, and `value` is some n-dimensional tensor. Letting
    :math:`t` index the `dim`-th dimension of `key`, :math:`q` index the last
    dimension of `query`, and :math:`k` index the last index of `key`. Let
    :math:`query_{t=0}` indicate the "unsqueezed" version of `query` where
    :math:`t` is inserted as the `dim`-th dimension. Then :math:`query_{t=0,q}`
    must `broadcast
    <https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics>`__
    with :math:`key_k`. If specified, `mask` should broadcast with :math:`e`,
    that is, broadcast with a tensor of the same shape as :math:`key_k` after
    it has been broadcast to :math:`query_{t=0,q}`. Finally, `value` must
    broadcast with :math:`a_{k=0}`, that is, :math:`a` with an unsqueezed final
    dimension.

    To make this concrete, consider the shapes of the Transformer arguments.
    :math:`query_{t=0}` broadcasts with :math:`key_k` as

        query_t=0.shape 1   S   num_batch
        key_k.shape     T   1   num_batch
        ---------------------------------
        e.shape         T   S   num_batch

    `mask` clearly broadcasts with :math:`e` since they are the same size.
    :math:`e` is the same shape as :math:`a`. Finally, `value` broadcasts with
    :math:`a_k=0` as

        a_k=0.shape     T   S   num_batch   1
        value.shape     T   1   num_batch   key_size
        --------------------------------------------
        out.shape       T   S   num_batch   key_size

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

    A single-headed transformer network with one layer each in the encoder and
    decoder. We forego much of the complexity of the original model, showcasing
    the "attention is all you need" bit

    >>> class Encoder(torch.nn.Module):
    >>>     def __init__(self, model_size, num_classes, padding_idx=-1):
    >>>         super(Encoder, self).__init__()
    >>>         self.model_size = model_size
    >>>         self.num_classes = num_classes
    >>>         self.embedder = torch.nn.Embedding(
    ...             num_classes, model_size, padding_idx=padding_idx)
    >>>         self.attention = DotProductSoftAttention(
    ...             model_size, scale_factor=model_size ** -.5)
    >>>         self.norm = torch.nn.LayerNorm(model_size)
    >>>
    >>>     def forward(self, inp):
    >>>         embedding = self.embedder(inp)
    >>>         query = embedding  # (T, num_batch, model_size)
    >>>         kv = embedding.unsqueeze(1)  # (T, 1, num_batch, model_size)
    >>>         mask = inp.ne(self.embedder.padding_idx)
    >>>         mask = mask.unsqueeze(1)  # (T, 1, num_batch)
    >>>         out = self.attention(query, kv, kv, mask)
    >>>         out = self.norm(out + embedding) # (T, num_batch, model_s)
    >>>         return out, mask

    >>> class Decoder(torch.nn.Module):
    >>>     def __init__(self, model_size, num_classes, padding_idx=-2):
    >>>         super(Decoder, self).__init__()
    >>>         self.model_size = model_size
    >>>         self.num_classes = num_classes
    >>>         self.embedder = torch.nn.Embedding(
    ...             num_classes, model_size, padding_idx=padding_idx)
    >>>         self.attention = DotProductSoftAttention(
    ...             model_size, scale_factor=model_size ** -.5)
    >>>         self.norm1 = torch.nn.LayerNorm(model_size)
    >>>         self.norm2 = torch.nn.LayerNorm(model_size)
    >>>         self.ff = torch.nn.Linear(model_size, num_classes)
    >>>
    >>>     def forward(self, enc_out, dec_in, enc_mask=None):
    >>>         embedding = self.embedder(dec_in)
    >>>         query = embedding  # (S, num_batch, model_size)
    >>>         kv = embedding.unsqueeze(1)  # (S, 1, num_batch, model_size)
    >>>         pad_mask = dec_in.ne(self.embedder.padding_idx)
    >>>         pad_mask = pad_mask.unsqueeze(1)  # (S, 1, num_batch)
    >>>         auto_mask = torch.ones(
    ...             query.shape[0], query.shape[0], dtype=torch.uint8)
    >>>         # why upper and not lower? The mask is an inclusion mask.
    >>>         # The dimension we're summing out is dim=0, so the dim 1
    >>>         # will remain. So
    >>>         # auto_mask[:, 0] = [1, 0, 0, ...]  (at t=0)
    >>>         # auto_mask[:, 1] = [1, 1, 0, ...]  (at t=1)
    >>>         auto_mask = torch.triu(auto_mask)
    >>>         auto_mask = auto_mask.unsqueeze(-1)  # (S, S, 1)
    >>>         dec_mask = pad_mask & auto_mask  # (S, S, num_batch)
    >>>         dec_out = self.attention(query, kv, kv, dec_mask)
    >>>         dec_out = self.norm1(dec_out + embedding)
    >>>         query = dec_out  # (S, num_batch, model_size)
    >>>         kv = enc_out.unsqueeze(1)  # (T, 1, num_batch, model_size)
    >>>         out = self.attention(query, kv, kv, enc_mask)
    >>>         out = self.ff(self.norm2(out + query))
    >>>         return out, pad_mask

    Prep

    >>> T, num_batch, model_size = 100, 5, 1000
    >>> num_classes, start, eos = 20, 0, 1
    >>> padding = num_classes - 1
    >>> inp_lens = torch.randint(1, T + 1, (num_batch,))
    >>> inp = torch.nn.utils.rnn.pad_sequence(
    ...     [
    ...         torch.randint(2, num_classes - 1, (x + 1,))
    ...         for x in inp_lens
    ...     ],
    ...     padding_value=padding,
    ... )
    >>> inp[inp_lens, range(num_batch)] = eos
    >>> target_lens = torch.randint(1, T + 1, (num_batch,))
    >>> y = torch.nn.utils.rnn.pad_sequence(
    ...     [
    ...         torch.randint(2, num_classes - 1, (x + 2,))
    ...         for x in target_lens
    ...     ],
    ...     padding_value=padding,
    ... )
    >>> y[0] = start
    >>> y[target_lens + 1, range(num_batch)] = eos
    >>> dec_inp, targets = y[:-1], y[1:]
    >>> encoder = Encoder(model_size, num_classes, padding_idx=padding)
    >>> decoder = Decoder(model_size, num_classes, padding_idx=padding)
    >>> loss = torch.nn.CrossEntropyLoss(ignore_index=padding)
    >>> optim = torch.optim.Adam(
    ...     list(encoder.parameters()) + list(decoder.parameters()))

    Training a batch (you'lll have to do this a whole lot of times to get
    it to converge)

    >>> optim.zero_grad()
    >>> enc_out, enc_mask = encoder(inp)
    >>> logits, _ = decoder(enc_out, dec_inp, enc_mask)
    >>> logits = logits[..., :-1]  # get rid of padding logit
    >>> l = loss(logits.view(-1, num_classes - 1), targets.flatten())
    >>> l.backward()
    >>> optim.step()

    Decoding a batch (test time) using greedy search

    >>> enc_out, enc_mask = encoder(inp)
    >>> dec_hyp = torch.full((1, num_batch), start, dtype=torch.long)
    >>> enc_out, enc_mask = encoder(inp)
    >>> done_mask = torch.zeros(num_batch, dtype=torch.uint8)
    >>> while not done_mask.all():
    >>>     logits, _ = decoder(enc_out, dec_hyp, enc_mask)
    >>>     logits = logits[..., :-1]  # get rid of padding logit
    >>>     pred = logits[-1].argmax(1)
    >>>     pred.masked_fill_(done_mask, eos)
    >>>     done_mask = pred.eq(eos)
    >>>     dec_hyp = torch.cat([dec_hyp, pred.unsqueeze(0)], 0)
    >>> dec_hyp = dec_hyp[1:]
    '''

    def __init__(self, query_size, key_size, dim=0):
        super(GlobalSoftAttention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.dim = dim

    @abc.abstractmethod
    def score(self, query, key):
        '''Calculate the score function over the entire input

        This is implemented by subclasses of ``GlobalSoftAttention``

        ``query.unsqueeze(self.dim)[..., 0]`` broadcasts with ``value[...,
        0]``. The final dimension of `query` is of length ``self.query_size``
        and the final dimension of `key` should be of length ``self.key_size``

        Parameters
        ----------
        query : torch.tensor
        key : torch.tensor

        Returns
        -------
        e : torch.tensor
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
    query_size : int
    key_size : int
    dim : int
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

        e = \sum_q query_q, \sum_k W_{qk} key_k

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
    query_size : int
    key_size : int
    dim : int
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

        e = \sum_i v_i tanh(\sum_c W_{ic} [query, key]_c)

    For some learned matrix :math:`W` and vector :math:`v`, where
    :math:`[query, key]` indicates concatenation along the last axis. `query`
    and `key` will be expanded to fit their broadcast dimensions. :math:`W`
    has shape ``(inter_size, key_size)`` and :math:`v` has shape
    ``(hidden_size,)``

    Parameters
    ----------
    query_size : int
    key_size : int
    bias : bool, optional
        Whether to add bias term ``b`` :math:`W [query, key] + b`
    hidden_size : int, optional

    Attributes
    ----------
    query_size : int
    key_size : int
    dim : int
    hidden_size : int
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
    a wrapper around standard ``GlobalSoftAttention`` that results in a similar
    output as a regular attention layer. The idea is to replicate transformed
    versions of the `query`, `key`, and `value` `num_heads` times. Letting
    :math:`h` index the head:

    .. math::

        query_h = W^Q_h query \\
        key_h = W^K_h key \\
        value_h = W^V_h value

    If `query` is of shape ``(..., query_size)``, :math:`W^Q_h` is a learned
    matrix of shape ``(query_size, d_q)`` that acts on the final dimension of
    `query`. Likewise, :math:`W^K_h` is of shape ``(key_size, d_k)`` and
    :math:`W^V_h` is of shape ``(value_size, d_v)``.

    Each head is then determined via a wrapped ``GlobalSoftAttention``
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
        An instance of a subclass of ``GlobalSoftAttention`` responsible for
        processing a head. `single_head_attention` attention will be used to
        derive the sequence dimension (``dim``) of `key` via
        ``single_head_attention.dim``, the size of a head's query ``d_k``
        via ``single_head_attention.query_size``, and the size of a head's key
        via ``single_head_attention.key_size``
    out_size : int, optional
        The size of the last dimension of `out`. If unset, the default is to
        match `value_size`
    d_v : int, optional
        The size of the last dimension of a head's value. If unset, will
        default to ``max(1, value_size // num_heads)``
    bias_WQ, bias_WK, bias_WV, bias_WC : bool, optional
        Whether to add a bias term in each of the linear transformations
        :math:`W^Q`, :math:`W^K`, :math:`W^V`, :math:`W^C`

    Attributes
    ----------
    query_size : int
    key_size : int
    value_size : int
    out_size : int
    num_heads : int
    single_head_attention : GlobalSoftAttention
    dim : int
    d_q, d_k, d_v : int
    WQ, WK, WV, WC : torch.nn.Linear
        Matrices :math:`W^Q`, :math:`W^K`, :math:`W^V`, and :math:`W^C`

    See Also
    --------
    GlobalSoftAttention
        For more information on the shape restrictions of query, key, and
        value
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
