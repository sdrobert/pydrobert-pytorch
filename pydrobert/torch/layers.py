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
]


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
    '''

    def __init__(self, query_size, key_size, dim=0):
        super(GlobalSoftAttention, self).__init__()
        self._query_size = query_size
        self._key_size = key_size
        self.dim = dim

    @property
    def query_size(self):
        return self._query_size

    @property
    def key_size(self):
        return self._key_size

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
        if self.dim > key.dim() - 2:
            raise ValuerError(
                'dim must refer to a valid dimension of key after its last '
                'dimension has been removed'
            )
        e = self.score(query, key)
        if mask is not None:
            e = e.masked_fill(mask.eq(0), -float('inf'))
        a = torch.nn.functional.softmax(e, self.dim)
        c = (a.unsqueeze(-1) * value).sum(self.dim)
        return c

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

    See Also
    --------
    GlobalSoftAttention
        For a description of how to call this module, how it works, etc.
    '''

    def __init__(self, query_size, key_size, dim=0, bias=False):
        super(GeneralizedDotProductSoftAttention, self).__init__(
            query_size, key_size, dim)
        self._bias = bias
        self._W = torch.nn.Linear(key_size, query_size, bias=bias)

    @property
    def bias(self):
        return self._bias

    def score(self, query, key):
        Wkey = self._W(key)
        query = query.unsqueeze(self.dim)
        return (query * Wkey).sum(-1)

    def reset_parameters(self):
        self._W.reset_parameters()


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

    See Also
    --------
    GlobalSoftAttention
        For a description of how to call this module, how it works, etc.
    '''

    def __init__(
            self, query_size, key_size, dim=0, bias=False, hidden_size=1000):
        super(ConcatSoftAttention, self).__init__(query_size, key_size, dim)
        self._bias = bias
        self._hidden_size = hidden_size
        self._W = torch.nn.Linear(
            query_size + key_size, hidden_size, bias=bias)
        # there's no point in a bias for v. It'll just be absorbed by the
        # softmax later. You could add a bias after the tanh layer, though...
        self._v = torch.nn.Linear(hidden_size, 1, bias=False)

    @property
    def bias(self):
        return self._bias

    @property
    def hidden_size(self):
        return self._hidden_size

    def score(self, query, key):
        query = query.unsqueeze(self.dim)
        query_wo_last, key_wo_last = torch.broadcast_tensors(
            query[..., 0], key[..., 0])
        query, _ = torch.broadcast_tensors(query, query_wo_last.unsqueeze(-1))
        key, _ = torch.broadcast_tensors(key, key_wo_last.unsqueeze(-1))
        cat = torch.cat([query, key], -1)
        Wcat = self._W(cat)
        return self._v(Wcat).squeeze(-1)

    def reset_parameters(self):
        self._W.reset_parameters()
        self._v.reset_parameters()
