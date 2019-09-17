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
The loss functions :class:`HardOptimalCompletionDistillationLoss` and
:class:`MinimumErrorRateLoss` have been moved here from
:mod:`pydrobert.torch.training`
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import torch

from pydrobert.torch.util import error_rate, optimal_completion
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
    'HardOptimalCompletionDistillationLoss',
    'MimimumErrorRateLoss',
    'MultiHeadedAttention',
]

# XXX(sdrobert): a quick note on style. pytorch doesn't tend to protect its
# read-only attributes using private members, so we do the same


class HardOptimalCompletionDistillationLoss(torch.nn.Module):
    r'''A categorical loss function over optimal next tokens

    Optimal Completion Distillation (OCD) [sabour2018]_ tries to minimize the
    train/test discrepancy in transcriptions by allowing seq2seq models to
    generate whatever sequences they want, then assigns a per-step loss
    according to whatever next token would set the model on a path that
    minimizes the edit distance in the future.

    In its "hard" version, the version used in the paper, the OCD loss function
    is simply a categorical cross-entropy loss of each hypothesis token's
    distribution versus those optimal next tokens, averaged over the number of
    optimal next tokens:

    .. math::

        loss(logits_t) = \frac{-\log Pr(s_t|logits_t)}{|S_t|}

    Where :math:`s_t \in S_t` are tokens from the set of optimal next tokens
    given :math:`hyp_{\leq t}` and `ref`. The loss is decoupled from an exact
    prefix of `ref`, meaning that `hyp` can be longer or shorter than `ref`.

    When called, this loss function has the signature::

        loss(logits, ref, hyp)

    `hyp` is a long tensor of shape ``(max_hyp_steps, batch_size)`` if
    `batch_first` is :obj:`False`, otherwise ``(batch_size, max_hyp_steps)``
    that provides the hypothesis transcriptions. Likewise, `ref` of shape
    ``(max_ref_steps, batch_size)`` or ``(batch_size, max_ref_steps)``
    providing reference transcriptions. `logits` is a 4-dimensional tensor of
    shape ``(max_hyp_steps, batch_size, num_classes)`` if `batch_first` is
    :obj:`False`, ``(batch_size, max_hyp_steps, num_classes)`` otherwise. A
    softmax over the step dimension defines the per-step distribution over
    class labels.

    Parameters
    ----------
    eos : int, optional
        A special token in `ref` and `hyp` whose first occurrence in each
        batch indicates the end of a transcript
    include_eos : bool, optional
        Whether to include the first instance of `eos` found in both `ref` and
        `hyp` as valid tokens to be computed as part of the distance. `eos`
        must be a valid class index if `include_eos` is :obj:`True`
    batch_first : bool, optional
        Whether the batch dimension comes first, or the step dimension
    ins_cost : float, optional
        The cost of an adding a superfluous token to a transcript in `hyp`
    del_cost : float, optional
        The cost of missing a token from `ref`
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`
    weight : torch.FloatTensor, optional
        A manual rescaling weight given to each class
    reduction : {'mean', 'none', 'sum'}, optional
        Specifies the reduction to be applied to the output. 'none': no
        reduction will be applied. 'sum': the output will be summed. 'mean':
        the output will be averaged.

    Attributes
    ----------
    eos : int
    include_eos, batch_first : bool
    ins_cost, del_cost, sub_cost : float
    reduction : {'mean', 'none', 'sum'}
    weight : torch.FloatTensor or None

    See Also
    --------
    pydrobert.torch.util.optimal_completion
        Used to determine the optimal next token set :math:`S`
    pydrobert.torch.util.random_walk_advance
        For producing a random `hyp` based on `logits` if the underlying
        model producing `logits` is auto-regressive. Also provides an example
        of sampling non-auto-regressive models
    '''

    def __init__(
            self, eos=None, include_eos=True, batch_first=False, ins_cost=1.,
            del_cost=1., sub_cost=1., weight=None, reduction='mean'):
        super(HardOptimalCompletionDistillationLoss, self).__init__()
        self.eos = eos
        self.include_eos = include_eos
        self.batch_first = batch_first
        self.ins_cost = ins_cost
        self.del_cost = del_cost
        self.sub_cost = sub_cost
        self.reduction = reduction
        self._cross_ent = torch.nn.CrossEntropyLoss(
            weight=weight, reduction='none'
        )

    @property
    def weight(self):
        return self._cross_ent.weight

    @weight.setter
    def weight(self, value):
        self._cross_ent.weight = value

    def check_input(self, logits, ref, hyp):
        '''Check if input formatted correctly, otherwise RuntimeError'''
        if logits.dim() != 3:
            raise RuntimeError('logits must be 3 dimensional')
        if logits.shape[:-1] != hyp.shape:
            raise RuntimeError('first two dims of logits must match hyp shape')
        if self.include_eos and self.eos is not None and (
                (self.eos < 0) or (self.eos >= logits.shape[-1])):
            raise RuntimeError(
                'if include_eos=True, eos ({}) must be a class idx'.format(
                    self.eos))
        if self.reduction not in {'mean', 'sum', 'none'}:
            raise RuntimeError(
                '"{}" is not a valid value for reduction'
                ''.format(self.reduction))

    def forward(self, logits, ref, hyp, warn=True):
        self.check_input(logits, ref, hyp)
        num_classes = logits.shape[-1]
        # the padding we use will never be exposed to the user, so we merely
        # ensure we're not trampling the eos
        padding = -2 if self.eos == -1 else -1
        self._cross_ent.ignore_index = padding
        optimals = optimal_completion(
            ref, hyp, eos=self.eos, include_eos=self.include_eos,
            batch_first=self.batch_first, ins_cost=self.ins_cost,
            del_cost=self.del_cost, sub_cost=self.sub_cost,
            padding=padding, exclude_last=True, warn=warn,
        )
        max_unique_next = optimals.shape[-1]
        logits = logits.unsqueeze(2).expand(-1, -1, max_unique_next, -1)
        logits = logits.contiguous()
        loss = self._cross_ent(
            logits.view(-1, logits.shape[-1]), optimals.flatten()
        ).view_as(optimals)
        padding_mask = optimals.eq(padding)
        no_padding_mask = padding_mask.eq(0)
        loss = loss.masked_fill(padding_mask, 0.).sum(2)
        loss = torch.where(
            no_padding_mask.any(2),
            loss / no_padding_mask.float().sum(2),
            loss,
        )
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class MinimumErrorRateLoss(torch.nn.Module):
    r'''Error rate expectation normalized over some number of transcripts

    Proposed in [prabhavalkar2018]_ though similar ideas had been explored
    previously. Given a subset of all possible token sequences and their
    associated probability mass over that population, this loss calculates the
    probability mass normalized over the subset, then calculates the
    expected error rate over that normalized distribution. That is, given some
    sequences :math:`s \in S \subseteq P`, the loss for a given reference
    transcription :math:`s^*` is

    .. math::

        \mathcal{L}(s, s^*) = \frac{Pr(s) ER(s, s^*)}{\sum_{s'} Pr(s')}

    This is an exact expectation over :math:`S` but not over :math:`P`. The
    larger the mass covered by :math:`S`, the closer the expectation is to the
    population - especially so for an n-best list (though it would be biased).

    This loss function has the following signature::

        loss(log_probs, ref, hyp)

    `log_probs` is a tensor of shape ``(batch_size, samples)`` providing the
    log joint probabilities of every path. `hyp` is a long tensor of shape
    ``(max_hyp_steps, batch_size, samples)`` if `batch_first` is :obj:`False`
    otherwise ``(batch_size, samples, max_hyp_steps)`` that provides the
    hypothesis transcriptions. `ref` is a 2- or 3-dimensional tensor. If 2D, it
    is of shape ``(max_ref_steps, batch_size)`` (or ``(batch_size,
    max_ref_steps)``). Alternatively, `ref` can be of shape ``(max_ref_steps,
    batch_size, samples)`` or ``(batch_size, samples, max_ref_steps)``.

    If `ref` is 2D, the loss is calculated as

    .. math::

        loss_{MER} = SoftMax(log\_probs)[ER(hyp_i, ref) - \mu_i]

    where :math:`\mu_i` is the average error rate along paths in the batch
    element :math:`i`. :math:`mu_i` can be removed by setting `sub_avg` to
    :obj:`False`. Note that each hypothesis is compared against the same
    reference as long as the batch element remains the same

    If `ref` is 3D, the loss is calculated as

    .. math::

        loss_{MER} = SoftMax(log\_probs)[ER(hyp_i, ref_i) - \mu_i]

    In this version, each hypothesis is compared against a unique reference

    Parameters
    ----------
    eos : int, optional
        A special token in `ref` and `hyp` whose first occurrence in each
        batch indicates the end of a transcript
    include_eos : bool, optional
        Whether to include the first instance of `eos` found in both `ref` and
        `hyp` as valid tokens to be computed as part of the distance.
    sub_avg : bool, optional
        Whether to subtract the average error rate from each pathwise error
        rate
    batch_first : bool, optional
        Whether batch/path dimensions come first, or the step dimension
    norm : bool, optional
        If :obj:`False`, will use edit distances instead of error rates
    ins_cost : float, optional
        The cost of an adding a superfluous token to a transcript in `hyp`
    del_cost : float, optional
        The cost of missing a token from `ref`
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`
    reduction : {'mean', 'none', 'sum'}, optional
        Specifies the reduction to be applied to the output. 'none': no
        reduction will be applied. 'sum': the output will be summed. 'mean':
        the output will be averaged.

    Attributes
    ----------
    eos, ignore_index : int
    include_eos, sub_avg, batch_first, norm : bool
    ins_cost, del_cost, sub_cost : float
    reduction : {'mean', 'none', 'sum'}

    Notes
    -----

    A previous version of this module incorporated a Maximum Likelihood
    Estimate (MLE) into the loss as in [prabhavalkar2018]_, which required
    `logits` instead of `log_probs`. This was overly complicated, given the
    user can easily incorporate the additional loss term herself by using
    :class:`torch.nn.CrossEntropyLoss`. Take a look at the example below for
    how to recreate this

    Examples
    --------

    Assume here that `logits` is the output of some neural network, and that
    `hyp` has somehow been produced from that (e.g. a beam search or random
    walk). We combine this loss function with a cross-entropy/MLE term to
    sort-of recreate [prabhavalkar2018]_.

    >>> from pydrobert.torch.util import sequence_log_probs
    >>> steps, batch_size, num_classes, eos, padding = 30, 20, 10, 0, -1
    >>> samples, lmb = 10, .01
    >>> logits = torch.randn(
    ...     steps, samples, batch_size, num_classes, requires_grad=True)
    >>> hyp = torch.randint(num_classes, (steps, samples, batch_size))
    >>> ref_lens = torch.randint(1, steps + 1, (batch_size,))
    >>> ref_lens[0] = steps
    >>> ref = torch.nn.utils.rnn.pad_sequence(
    ...     [torch.randint(1, num_classes, (x,)) for x in ref_lens],
    ...     padding_value=padding,
    ... )
    >>> ref[ref_lens - 1, range(batch_size)] = eos
    >>> ref = ref.unsqueeze(1).repeat(1, samples, 1)
    >>> mer = MinimumErrorRateLoss(eos=eos)
    >>> mle = torch.nn.CrossEntropyLoss(ignore_index=padding)
    >>> log_probs = sequence_log_probs(logits, hyp, eos=eos)
    >>> l = mer(log_probs, ref, hyp)
    >>> l = l + lmb * mle(logits.view(-1, num_classes), ref.flatten())
    >>> l.backward()

    See Also
    --------
    pydrobert.torch.util.beam_search_advance
        For getting an n-best list into `hyp` and some `log_probs`.
    pydrobert.torch.util.random_walk_advance
        For getting a random sample into `hyp`
    pydrobert.torch.util.sequence_log_probs
        For converting token log probs (or logits) to sequence log probs
    '''

    def __init__(
            self, eos=None, include_eos=True, sub_avg=True, batch_first=False,
            norm=True, ins_cost=1., del_cost=1., sub_cost=1.,
            reduction='mean'):
        super(MinimumErrorRateLoss, self).__init__()
        self.eos = eos
        self.include_eos = include_eos
        self.sub_avg = sub_avg
        self.batch_first = batch_first
        self.norm = norm
        self.ins_cost = ins_cost
        self.del_cost = del_cost
        self.sub_cost = sub_cost
        self.reduction = reduction

    def check_input(self, log_probs, ref, hyp):
        '''Check if the input is formatted correctly, otherwise RuntimeError'''
        if log_probs.dim() != 2:
            raise RuntimeError('log_probs must be 2 dimensional')
        if hyp.dim() != 3:
            raise RuntimeError('hyp must be 3 dimensional')
        if ref.dim() not in {2, 3}:
            raise RuntimeError('ref must be 2 or 3 dimensional')
        if self.batch_first:
            if ref.dim() == 2:
                ref = ref.unsqueeze(1).expand(-1, hyp.shape[1], -1)
            if (
                    (ref.shape[:2] != hyp.shape[:2]) or
                    (ref.shape[:2] != log_probs.shape)):
                raise RuntimeError(
                    'ref and hyp batch_size and sample dimensions must match')
            if ref.shape[1] < 2:
                raise RuntimeError(
                    'Batch must have at least two samples, got {}'
                    ''.format(ref.shape[1]))
        else:
            if ref.dim() == 2:
                ref = ref.unsqueeze(-1).expand(-1, -1, hyp.shape[-1])
            if (
                    (ref.shape[1:] != hyp.shape[1:]) or
                    (ref.shape[1:] != log_probs.shape)):
                raise RuntimeError(
                    'ref and hyp batch_size and sample dimensions must match')
            if ref.shape[2] < 2:
                raise RuntimeError(
                    'Batch must have at least two samples, got {}'
                    ''.format(ref.shape[2]))
        if self.reduction not in {'mean', 'sum', 'none'}:
            raise RuntimeError(
                '"{}" is not a valid value for reduction'
                ''.format(self.reduction))

    def forward(self, log_probs, ref, hyp, warn=True):
        self.check_input(log_probs, ref, hyp)
        if self.batch_first:
            batch_size, samples, max_hyp_steps = hyp.shape
            max_ref_steps = ref.shape[-1]
            if ref.dim() == 2:
                ref = ref.unsqueeze(1).repeat(1, samples, 1)
            ref = ref.view(-1, max_ref_steps)
            hyp = hyp.view(-1, max_hyp_steps)
        else:
            max_hyp_steps, batch_size, samples = hyp.shape
            max_ref_steps = ref.shape[0]
            if ref.dim() == 2:
                ref = ref.unsqueeze(-1).repeat(1, 1, samples)
            ref = ref.view(max_ref_steps, -1)
            hyp = hyp.view(max_hyp_steps, -1)
        er = error_rate(
            ref, hyp, eos=self.eos, include_eos=self.include_eos,
            norm=self.norm, batch_first=self.batch_first,
            ins_cost=self.ins_cost, del_cost=self.del_cost,
            sub_cost=self.sub_cost, warn=warn,
        ).view(batch_size, samples)
        if self.sub_avg:
            er = er - er.mean(1, keepdim=True)
        loss = er * torch.nn.functional.softmax(log_probs, 1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class GlobalSoftAttention(with_metaclass(abc.ABCMeta, torch.nn.Module)):
    r'''Parent class for soft attention mechanisms on an entire input sequence

    Global soft attention mechansims [bahdanau2015]_ are a way of getting rid
    of one variable-length sequence dimension ``T`` in an input `key` using a
    weighted sum of a tensor `value` that is informed by some other tensor,
    `query`. The weights are dictated by the function ``score(query, key)``.
    Usually, this is in the context of encoder-decoder architectures, which is
    explained here.

    Assume `query` is a tensor of shape ``(batch_size, query_size)``
    representing a single hidden state of a decoder RNN. Assume `key` is a
    tensor of shape ``(T, batch_size, key_size)`` representing the encoder
    output, ``dim == 0`` to specify that the variable-length dimension of `key`
    is the zero-th dimension, and ``value == key``. The output `out` will be a
    tensor of shape ``(batch_size, key_size)``. Letting :math:`t` index the
    `dim`-th dimension:

        .. math::

            out = \sum_t a_t value_t

    ``a`` is the attention vector. In our example, ``a`` will be of shape
    ``(T, batch_size)``. ``a`` is the result of a softmax over the `dim`-th
    dimension of another tensor ``e`` of shape ``(T, batch_size)`` with an
    optional `mask`

    .. math::

        a = softmax(e * mask - (1 - mask) \infty, dim)

    `mask` (if specified) is of shape ``(T, batch_size)`` and will set ``a`` to
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

    def check_input(self, query, key, value, mask=None):
        '''Check if input is properly formatted, RuntimeError otherwise

        Warnings
        --------
        This method doesn't check that the tensors properly broadcast. If they
        don't, they will fail later on. It only ensures the proper sizes and
        that the final dimensions are appropriately sized where applicable

        See Also
        --------
        :ref:`Advanced Attention and Transformer Networks`
            For full broadcasting rules
        '''
        key_dim = key.dim()
        if query.dim() != key_dim - 1:
            raise RuntimeError('query must have one fewer dimension than key')
        if key_dim != value.dim():
            raise RuntimeError(
                "key must have same number of dimensions as value")
        if query.shape[-1] != self.query_size:
            raise RuntimeError('Last dimension of query must match query_size')
        if key.shape[-1] != self.key_size:
            raise RuntimeError('Last dimension of key must match key_size')
        if self.dim > key_dim - 2 or self.dim < -key_dim + 1:
            raise RuntimeError(
                'dim must be in the range [{}, {}]'
                ''.format(-key_dim + 1, key_dim - 2)
            )
        if mask is not None and mask.dim() != key_dim - 1:
            raise RuntimeError('mask must have one fewer dimension than key')

    def forward(self, query, key, value, mask=None):
        self.check_input(query, key, value, mask)
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


class MultiHeadedAttention(GlobalSoftAttention):
    r'''Perform attention over a number of heads, concatenate, and project

    Multi-headed attention was proposed in [vaswani2017]_. It can be considered
    a wrapper around standard :class:`GlobalSoftAttention` that also performs
    :class:`GlobalSoftAttention`, but with more parameters. The idea is to
    replicate transformed versions of the `query`, `key`, and `value`
    `num_heads` times. Letting :math:`h` index the head:

    .. math::

        query_h = W^Q_h query \\
        key_h = W^K_h key \\
        value_h = W^V_h value

    If `query` is of shape ``(..., query_size)``, :math:`W^Q_h` is a learned
    matrix of shape ``(query_size, d_q)`` that acts on the final dimension of
    `query`. Likewise, :math:`W^K_h` is of shape ``(key_size, d_k)`` and
    :math:`W^V_h` is of shape ``(value_size, d_v)``. Note here that the last
    dimension of `value` must also be provided in `value_size`, unlike in
    other attention layers.

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
        super(MultiHeadedAttention, self).__init__(
            query_size, key_size, dim=single_head_attention.dim)
        self.value_size = value_size
        self.out_size = value_size if out_size is None else out_size
        self.num_heads = num_heads
        self.single_head_attention = single_head_attention
        # we don't keep these in sync in case someone's using
        # single_head_attention
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

    def check_input(self, query, key, value, mask=None):
        '''Check that input is formatted correctly, RuntimeError otherwise'''
        super(MultiHeadedAttention, self).check_input(query, key, value, mask)
        if value.shape[-1] != self.value_size:
            raise RuntimeError('Last dimension of value must match value_size')

    def score(self, query, key):
        raise NotImplementedError(
            'In MultiHeadedAttention, score() is handled by '
            'single_head_attention')

    def forward(self, query, key, value, mask=None):
        self.check_input(query, key, value, mask)
        query_shape = tuple(query.shape)
        key_shape = tuple(key.shape)
        value_shape = tuple(value.shape)
        key_dim = key.dim()
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
        s = super(MultiHeadedAttention, self).extra_repr()
        # rest of info in single_head_attention submodule
        s += ', value_size={}, out_size={}, num_heads={}'.format(
            self.value_size, self.out_size, self.num_heads)
        return s
