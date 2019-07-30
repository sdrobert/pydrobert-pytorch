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

        attention(h_t, x[, mask])

    Where `h_t` and `x` are as previously discussed. `mask` is a byte
    mask of shape ``(S, num_batch)`` if `batch_first` is ``False``
    (``(num_batch, S)`` otherwise). `mask` can be used to ensure that
    :math:`a_{t,s} == 0` whenever :math:`mask_{t,s} == 0`. This is useful to
    ensure correct calculations when `x` consists of variable-length sequences

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

    def forward(self, h_t, x, mask=None):
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
        e_t = self.score(h_t, x)
        if mask is not None:
            # we perform on e_t instead of a_t to ensure sum_s a_{t,s} = 1
            e_t = e_t.masked_fill(mask.eq(0), -float('inf'))
        if self.batch_first:
            a_t = torch.nn.functional.softmax(e_t, 1)
            c_t = (a_t.unsqueeze(2) * x).sum(1)
        else:
            a_t = torch.nn.functional.softmax(e_t, 0)
            c_t = (a_t.unsqueeze(2) * x).sum(0)
        return c_t

    def reset_parameters(self):
        pass
