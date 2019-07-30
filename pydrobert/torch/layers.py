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
    r'''Parent class for soft attention mechanisms on an entire sequence

    Global soft attention mechanisms, introduced in [bahdanau2015]_, provide a
    way to condition an RNN on an entire (transformed) input sequence at once
    without the output sequence being of the same length. The RNN is called a
    decoder, and the decoder's state at a time step `h_t` is going to be passed
    into this layer along with the entire input to get a context vector `c_t`,
    i.e. ``c_t = Attention(h_t, x)``. Note the index :math:`t` is used to
    indicate that `c_t` and `h_t` are tensors sliced from some greater tensors
    arrayed over the decoder's time steps, e.g. ``h_t = h[t]`` for some ``h``
    of shape ``(T, num_batch, hidden_size)``.

    Suppose the input sequence is `x` of shape ``(S, num_batch, input_size)``
    if `batch_first` is ``True`` (``(num_batch, S, input_size)`` otherwise),
    `h_t` is of shape ``(num_batch, hidden_size)``, and `c_t` is of shape
    ``(num_batch, input_size)``. Then `c_t` is a weighted sum of `x`

    .. math::

        c_t = \sum_{s=1}^{S} a_{t, s} x_s

    where ``s`` indexes the ``S`` (step) dimension. :math:`a_t` is a tensor of
    shape ``(S, num_batch)`` s.t. :math:`\sum_s a_{t, s, bt} = 1` for any
    batch index :math:`bt`. :math:`a_{t,s}` is the result of a softmax
    distribution over the :math:`s` dimension:

    .. math::

        a_{t,s} = \frac{\exp(e_{t,s})}{\sum_{s'}^{S}\exp(e_{t,s'})}

    Where :math:`e_{t,s}` is the output of some score function:

    .. math::

        e_{t,s} = score(h_t, x_s)

    Subclasses are expected to implement the ``score()`` function.

    When called, this layer has the signature:

        attention(h_t, x, mask=None, val=None)

    Where `h_t` and `x` are as previously discussed. `mask` is a byte
    mask of shape ``(S, num_batch)`` if `batch_first` is ``False``
    (``(num_batch, S)`` otherwise). `mask` can be used to ensure that
    :math:`a_{t,s} == 0` whenever :math:`mask_{t,s} == 0`. This is useful to
    ensure correct calculations when `x` consists of variable-length sequences.

    If `val` is specified, `val` replaces `x` in :math:`\sum_s a_{t,s} x_s`
    when calculating :math:`c_t`. "val" refers to "value" in the
    "query-key-value" construction of [vaswani2017]_ (where `h_t` becomes the
    "query" and `x` becomse the "key"). `val` can have an arbitrary number of
    dimensions, as long as the first two match `x`. In this case, `c_t` will
    have shape ``(num_batches,) + tuple(val.shape[2:])``

    Parameters
    ----------
    input_size : int
        The non-time, non-batch dimension of the input, `x`. If the input
        comes from an encoder, `input_size` is likely the size of the encoder's
        output per time slice.
    hidden_size : int
        The non-time, non-batch dimension of the decoder's hidden state. This
        should not be confused with the hidden state size of the encoder, if
        any.
    batch_first : bool, optional
        If the first dimension of `x` is the batch dimension. Otherwise, it's
        the time dimension

    Attributes
    ----------
    input_size : int
    hidden_size : int
    batch_first : bool
    '''

    def __init__(self, input_size, hidden_size, batch_first=False):
        super(GlobalSoftAttention, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self.batch_first = batch_first

    @property
    def input_size(self):
        return self._input_size

    @property
    def hidden_size(self):
        return self._hidden_size

    @abc.abstractmethod
    def score(self, h_t, x):
        '''Calculate the score function over the entire input

        This is implemented by subclasses of ``GlobalSoftAttention``. It is
        usually the case that ``e_t[s]`` only uses ``h_t`` and ``x[s]`` in
        computations

        Parameters
        ----------
        h_t : torch.tensor
            Decoder states. Tensor of shape ``(num_batch, self.hidden_size)``
        x : torch.tensor
            Input. Tensor of shape ``(S, num_batch, self.input_size)`` if
            ``self.batch_first`` is ``False``, ``(num_batch, S,
            self.input_size)`` otherwise

        Returns
        -------
        e_t : torch.tensor
            Tensor of shape ``(S, num_batch)`` if ``self.batch_first`` is
            ``False``, ``(num_batch, S)`` otherwise
        '''
        raise NotImplementedError()

    def forward(self, h_t, x, mask=None, val=None):
        if h_t.dim() != 2:
            raise ValueError('Expected h_t to have 2 dimensions')
        if h_t.shape[1] != self.hidden_size:
            raise ValueError(
                'Expected h_t to have hidden size of {}'
                ''.format(self.hidden_size))
        if x.dim() != 3:
            raise ValueError('Expected x to have 3 dimensions')
        if self.batch_first:
            num_batches, S, input_size = x.shape
        else:
            S, num_batches, input_size = x.shape
        if h_t.shape[0] != num_batches:
            raise ValueError(
                'Expected batch dim of h_t ({}) to match batch dim of x ({})'
                ''.format(h_t.shape[0], num_batches))
        if input_size != self.input_size:
            raise ValueError(
                'Expected x to have input size of {}'.format(self.input_size))
        if mask is not None and mask.shape != x.shape[:-1]:
            raise ValueError(
                'Expected mask to have shape {}'.format(tuple(x.shape[:-1])))
        if val is None:
            val = x
        elif val.shape[:2] != x.shape[:2]:
            raise ValueError(
                'Expected val to have shape ({}, {}, ...)'
                ''.format(*tuple(x.shape[:2])))
        e_t = self.score(h_t, x)
        if mask is not None:
            # we perform on e_t instead of a_t to ensure sum_s a_{t,s} = 1
            e_t = e_t.masked_fill(mask.eq(0), -float('inf'))
        if self.batch_first:
            a_t = torch.nn.functional.softmax(e_t, 1)
            c_t = (a_t.unsqueeze(2) * val.view(num_batches, S, -1)).sum(1)
            c_t = c_t.view_as(val[:, 0])
        else:
            a_t = torch.nn.functional.softmax(e_t, 0)
            c_t = (a_t.unsqueeze(2) * val.view(S, num_batches, -1)).sum(0)
            c_t = c_t.view_as(val[0])
        return c_t

    def reset_parameters(self):
        pass


class DotProductSoftAttention(GlobalSoftAttention):
    r'''Global soft attention with dot product score function

    From [luong2015]_, the score function for this attention mechanism is

    .. math::

        e_{t,s} = scale\_factor \langle h_t, x_s \rangle

    Parameters
    ----------
    size : int
        Both the input size and hidden size
    batch_first : bool, optional
    scale_factor : float, optional
        A floating point to multiply the each :math:`e_{t,s}` with. Usually
        1, but if set to :math:`1 / size`, you'll get the scaled dot-product
        attention of [vaswani2017]_

    See Also
    --------
    GlobalSoftAttention
        For a description of how to call this module, how it works, etc.
    '''

    def __init__(self, size, batch_first=False, scale_factor=1.):
        super(DotProductSoftAttention, self).__init__(size, size, batch_first)
        self.scale_factor = scale_factor

    def score(self, h_t, x):
        h_t = h_t.unsqueeze(1 if self.batch_first else 0)
        return (h_t * x).sum(2) * self.scale_factor


class GeneralizedDotProductSoftAttention(GlobalSoftAttention):
    r'''Dot product soft attention with a learned matrix in between

    The "general" score function from [luong2015]_, the score function for this
    attention mechanism is

    .. math::

        e_{t, s} = \langle h_t, W x_s \rangle

    For some learned matrix :math:`W`

    Parameters
    ----------
    input_size : int
    hidden_size : int
    batch_first : bool, optional
    bias : bool, optional
        Whether to add a bias term ``b``: :math:`W x_s + b`

    See Also
    --------
    GlobalSoftAttention
        For a description of how to call this module, how it works, etc.
    '''

    def __init__(self, input_size, hidden_size, batch_first=False, bias=False):
        super(GeneralizedDotProductSoftAttention, self).__init__(
            input_size, hidden_size, batch_first)
        self._bias = bias
        self._W = torch.nn.Linear(input_size, hidden_size, bias=bias)

    @property
    def bias(self):
        return self._bias

    def score(self, h_t, x):
        Wx = self._W(x)
        h_t = h_t.unsqueeze(1 if self.batch_first else 0)
        return (h_t * Wx).sum(2)

    def reset_parameters(self):
        self._W.reset_parameters()


class ConcatSoftAttention(GlobalSoftAttention):
    r'''Attention where input and hidden are concatenated, then fed into an MLP

    Proposed in [luong2015]_, though quite similar to that proposed in
    [bahdanau2015]_, the score function for this layer is:

    .. math::

        e_{t, s} = \langle v, tanh(W [x_s, h_t]) \rangle

    For some learned matrix :math:`W`, where :math:`[x_s, h_t]` indicates
    concatenation. :math:`W` is of shape
    ``(input_size + hidden_size, intermediate_size)`` and :math:`v` is of
    shape ``(intermediate_size,)``

    Parameters
    ----------
    input_size : int
    hidden_size : int
    batch_first : bool, optional
    bias : bool, optional
        Whether to add bias term ``b`` :math:`W x_s + b`
    intermediate_size : int, optional

    See Also
    --------
    GlobalSoftAttention
        For a description of how to call this module, how it works, etc.
    '''

    def __init__(
            self, input_size, hidden_size, batch_first=False, bias=False,
            intermediate_size=1000):
        super(ConcatSoftAttention, self).__init__(
            input_size, hidden_size, batch_first)
        self._bias = bias
        self._intermediate_size = intermediate_size
        self._W = torch.nn.Linear(
            input_size + hidden_size, intermediate_size, bias=bias)
        # there's no point in a bias for v. It'll just be absorbed by the
        # softmax later. You could add a bias after the tanh layer, though...
        self._v = torch.nn.Linear(intermediate_size, 1, bias=False)

    @property
    def bias(self):
        return self._bias

    @property
    def intermediate_size(self):
        return self._intermediate_size

    def score(self, h_t, x):
        if self.batch_first:
            h_t = h_t.unsqueeze(1).expand(-1, x.shape[1], -1)
        else:
            h_t = h_t.unsqueeze(0).expand(x.shape[0], -1, -1)
        xh = torch.cat([x, h_t], 2)
        Wxh = self._W(xh)
        return self._v(Wxh).squeeze(2)

    def reset_parameters(self):
        self._W.reset_parameters()
        self._v.reset_parameters()
