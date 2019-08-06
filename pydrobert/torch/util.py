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

'''Utility functions'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import torch
import pydrobert.torch

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"
__all__ = [
    'beam_search_advance',
    'error_rate',
    'optimal_completion',
    'optimizer_to',
    'prefix_error_rates',
    'random_walk_advance',
]


def beam_search_advance(
        logits_t, width, log_prior=None, y_prev=None,
        eos=pydrobert.torch.INDEX_PAD_VALUE, lens=None, prevent_eos=False):
    r'''Advance a beam search

    Suppose a model outputs a un-normalized log-probability distribution over
    the next element of a sequence in `logits_t` s.t.

    .. math::

        Pr(y_t = c; log\_prior) = exp(logits_{t,c}) / \sum_k exp(logits_{t,k})

    We assume :math:`logits_t` is a function of what comes before
    :math:`logits_t = f(logits_{<t}, y_{<t})`. Alternatively, letting
    :math:`s_t = (logits_t, y_t)`, :math:`s` is a Markov Chain. A model is
    auto-regressive if :math:`f` depends on :math:`y_{<t}`, and is not
    auto-regressive if :math:`logits_t = f(logits_{<t})`.

    Beam search is a heuristic mechanism for determining a best path, i.e.
    :math:`\arg \max_y Pr(y)` that maximizes the probability of the best path
    by keeping track of `width` high probability paths called "beams" (the
    aggregate of which for a given batch element is named, unfortunately, "the
    beam"). If the model is auto-regressive, beam search is only approximate.
    However, if the model is not auto-regressive, beam search gives an exact
    n-best list.

    This function is called at every time step. It updates old beam
    log-probabilities (`log_prior`) with new ones (`score`) from the joint
    distribution with `logits`, and updates us the class indices emitted
    between them (`y`). See the examples section for how this might work.

    Parameters
    ----------
    logits : torch.FloatTensor
        The conditional probabilities over class labels for the current time
        step. Either of shape ``(num_batches, old_width, num_classes)``,
        where ``old_width`` is the number of beams in the previous time step,
        or ``(num_batches, num_classes)``, where it is assumed that
        ``old_width == 1``
    width : int
        The number of beams in the beam to produce for the current time step.
        ``width <= num_classes``
    log_prior : torch.FloatTensor, optional
        A tensor of (or proportional to) log prior probabilities of beams up
        to the previous time step. Either of shape ``(num_batches, old_width)``
        or ``(num_batches,)``. If unspecified, a uniform log prior will be used
    y_prev : torch.LongTensor, optional
        A tensor of shape ``(t - 1, num_batches, old_width)`` or
        ``(t - 1, num_batches)`` specifying :math:`y_{<t}`. If unspecified,
        it is assumed that ``t == 1``
    eos : int, optional
        A special end-of-sequence symbol indicating that the beam has ended.
        Can be a class index. If this value occurs in in ``y_prev[-1, bt, bm]``
        for some batch ``bt`` and beam ``bm``, that beam will be padded with
        an `eos` token and the score for that beam won't change
    lens : torch.LongTensor, optional
        A tensor of shape ``(num_batches,)``. If ``t > lens[bt]`` for some
        batch ``bt``, all beams for ``bt`` will be considered finished. All
        scores will be fixed and `eos` will be appended to `y_prev`
    prevent_eos : bool, optional
        Setting this flag to :obj:`True` will keep `eos` targets from entering
        a beam unless it has finished (either with a prior `eos` or through
        `lens`). Note that this will only have an effect when ``0 <= eos <=
        num_classes``

    Returns
    -------
    score : torch.FloatTensor
        Of shape ``(num_batches, width)`` of the log-joint probabilities of the
        new beams in the beam
    y : torch.LongTensor
        Of shape ``(t, num_batches, width)`` of indices of the class labels
        generated up to this point
    s : torch.LongTensor
        Of shape ``(num_batches, width)`` of indices of beams in the old beam
        which prefix the new beam. Note that beams in the new beam are sorted
        by descending probability

    Examples
    --------

    Auto-regressive decoding with beam search. We assume that all input have
    the same number of steps

    >>> N, I, C, T, W, H, eos, start = 5, 5, 10, 100, 5, 10, 0, -1
    >>> cell = torch.nn.RNNCell(I + 1, H)
    >>> ff = torch.nn.Linear(H, C)
    >>> inp = torch.rand(T, N, I)
    >>> y = torch.full((1, N, 1), start, dtype=torch.long)
    >>> h_t = torch.zeros(N, 1, H)
    >>> score = None
    >>> for inp_t in inp:
    >>>     y_tm1 = y[-1]
    >>>     old_width = y_tm1.shape[-1]
    >>>     inp_t = inp_t.unsqueeze(1).expand(N, old_width, I)
    >>>     x_t = torch.cat([inp_t, y_tm1.unsqueeze(2).float()], -1)
    >>>     h_t = cell(
    ...         x_t.view(N * old_width, I + 1),
    ...         h_t.view(N * old_width, H)
    ...     ).view(N, old_width, H)
    >>>     logits_t = ff(h_t)
    >>>     score, y, s_t = beam_search_advance(logits_t, W, score, y, eos)
    >>>     h_t = h_t.gather(1, s_t.unsqueeze(-1).expand(N, W, H))
    >>> bests = []
    >>> for batch_idx in range(N):
    >>>     best_beam_path = y[1:, batch_idx, 0]
    >>>     not_special_mask = best_beam_path.ne(eos)
    >>>     best_beam_path = best_beam_path.masked_select(not_special_mask)
    >>>     bests.append(best_beam_path)

    ``W``-best list for non-auto-regressive model. We don't emit an `eos`,
    instead completing the sequence when we've hit the target length via `lens`

    >>> N, I, C, T, W, H = 5, 5, 10, 100, 5, 10
    >>> rnn = torch.nn.RNN(I, H)
    >>> ff = torch.nn.Linear(H, C)
    >>> inp = torch.rand(T, N, I)
    >>> lens = torch.randint(1, T + 1, (N,)).sort(descending=True)[0]
    >>> packed_inp = torch.nn.utils.rnn.pack_padded_sequence(inp, lens)
    >>> packed_h, _ = rnn(packed_inp)
    >>> packed_logits = ff(packed_h[0])
    >>> logits = torch.nn.utils.rnn.pad_packed_sequence(
    ...     torch.nn.utils.rnn.PackedSequence(
    ...         packed_logits, batch_sizes=packed_h[1]),
    ...     total_length=T,
    ... )[0]
    >>> y = score = None
    >>> for t, logits_t in enumerate(logits):
    >>>     if t:
    >>>         logits_t = logits_t.unsqueeze(1).expand(-1, W, -1)
    >>>     score, y, _ = beam_search_advance(logits_t, W, score, y, lens=lens)
    '''
    if logits_t.dim() == 2:
        logits_t = logits_t.unsqueeze(1)
    elif logits_t.dim() != 3:
        raise ValueError('logits_t must have dimension of either 2 or 3')
    logits_t = torch.nn.functional.log_softmax(logits_t, 2)
    neg_inf = torch.tensor(-float('inf'), device=logits_t.device)
    num_batches, old_width, num_classes = logits_t.shape
    if log_prior is None:
        log_prior = torch.full(
            (num_batches, old_width),
            -torch.log(torch.tensor(float(num_classes))),
            dtype=logits_t.dtype, device=logits_t.device,
        )
    elif tuple(log_prior.shape) == (num_batches,) and old_width == 1:
        log_prior = log_prior.unsqueeze(1)
    elif log_prior.shape != logits_t.shape[:-1]:
        raise ValueError(
            'If logits_t of shape {} then log_prior must have shape {}'.format(
                (num_batches, old_width, num_classes),
                (num_batches, old_width),
            ))
    if prevent_eos and 0 <= eos < num_classes:
        # we have to put this before the num_done check so that it'll be
        # overwritten for paths that have finished already
        logits_t[..., eos] = neg_inf
    eos_set = None
    if y_prev is not None:
        if y_prev.dim() == 2:
            y_prev = y_prev.unsqueeze(2)
        if y_prev.shape[1:] != log_prior.shape:
            raise ValueError(
                'If logits_t of shape {} then y_prev must have shape '
                '(*, {}, {})'.format(
                    (num_batches, old_width, num_classes),
                    num_batches, old_width,
                )
            )
        eos_mask = y_prev[-1].eq(eos)
        num_done = eos_mask.sum(1)
        if num_done.sum().item():
            if old_width < width and torch.any(num_done == old_width):
                raise ValueError(
                    'New beam width ({}) is wider than old beam width '
                    '({}), but all paths are already done in one or more '
                    'batch elements'.format(width, old_width))
            # we're going to treat class 0 as the sentinel for eos (even if
            # eos is a legit class label)
            done_classes = torch.full_like(logits_t[0, 0], neg_inf)
            done_classes[0] = 0.
            logits_t = torch.where(
                eos_mask.unsqueeze(2),
                done_classes,
                logits_t,
            )
            # If eos_mask looks like this (vertical batch, horizontal beam):
            #   1 0 0 1 0
            #   0 1 0 0 0
            #   0 0 0 0 0
            # then eos_set will be
            #    0 -1 -1  3 -1
            #   -1  1 -1 -1 -1
            #   -1 -1 -1 -1 -1
            # s might look like
            #    1 2 3 3
            #    2 2 4 1
            #    1 2 3 4
            # we'll compare a single value from a row of s to a matched row
            # of eos_set. Any match means the beam had finished already. The
            # mask on y will be
            #    0 0 1 1
            #    0 0 0 1
            #    0 0 0 0
            # pretty funky
            eos_set = torch.where(
                eos_mask,
                torch.arange(old_width, device=logits_t.device),
                torch.tensor(-1, device=logits_t.device).expand(old_width),
            )
        t = y_prev.shape[0] + 1
    else:
        t = 1
    len_mask = None
    if lens is not None:
        if lens.shape != logits_t.shape[:1]:
            raise ValueError('lens must be of shape ({},)'.format(num_batches))
        len_mask = lens.lt(t)
        if torch.any(len_mask):
            if old_width < width:
                raise ValueError(
                    'New beam width ({}) is wider than old beam width '
                    '({}), but all paths are already done in one or more '
                    'batch elements'.format(width, old_width))
        else:
            len_mask = None
    joint = log_prior.unsqueeze(2) + logits_t
    score, idxs = torch.topk(joint.view(num_batches, -1), width, dim=1)
    s = idxs // num_classes
    y = (idxs % num_classes).unsqueeze(0)
    if eos_set is not None:
        y_mask = (s.unsqueeze(2) == eos_set.unsqueeze(1)).any(2)
        y = y.masked_fill(y_mask, eos)
    if len_mask is not None:
        score = torch.where(
            len_mask.unsqueeze(1),
            log_prior[..., :width],
            score,
        )
        y = y.masked_fill(len_mask.unsqueeze(1).unsqueeze(0), eos)
        s = torch.where(
            len_mask.unsqueeze(1),
            torch.arange(width, device=logits_t.device),
            s,
        )
    if y_prev is not None:
        y_prev = y_prev.gather(
            2, s.unsqueeze(0).expand(t - 1, num_batches, width))
        y = torch.cat([y_prev, y], 0)
    return score, y, s


def error_rate(
        ref, hyp, eos=None, include_eos=False, norm=True, batch_first=False,
        ins_cost=1., del_cost=1., sub_cost=1., warn=True):
    '''Calculate error rates over a batch

    An error rate is merely a `Levenshtein (edit) distance
    <https://en.wikipedia.org/wiki/Levenshtein_distance>`__ normalized over
    reference sequence lengths.

    Given a reference (gold-standard) transcript tensor `ref` of size
    ``(max_ref_steps, batch_size)`` if ``batch_first == False`` or
    ``(batch_size, max_hyp_steps)`` otherwise, and a similarly shaped tensor of
    hypothesis transcripts `hyp`, this function produces a tensor `er` of shape
    ``(batch_size,)`` storing the associated error rates.

    `er` will not have a gradient, and is thus not directly suited to being a
    loss function

    Parameters
    ----------
    ref : torch.LongTensor
    hyp : torch.LongTensor
    eos : int, optional
        A special token in `ref` and `hyp` whose first occurrence in each
        batch indicates the end of a transcript. This allows for
        variable-length transcripts in the batch
    include_eos : bool, optional
        Whether to include the first instance of `eos` found in both `ref` and
        `hyp` as valid tokens to be computed as part of the distance. This is
        useful when gauging if a model is learning to emit the `eos` properly,
        but is not usually included in an evaluation. Only the first `eos` per
        transcript is included
    norm : bool, optional
        If :obj:`False`, will return edit distances instead of error rates
    batch_first : bool, optional
    ins_cost : float, optional
        The cost of an adding a superfluous token to a transcript in `hyp`
    del_cost : float, optional
        The cost of missing a token from `ref`
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`
    warn : bool, optional
        Whether to display warnings on irregularities. Currently, this can
        happen in two ways:

        1. If :obj:`True` and `norm` is :obj:`True`, will warn when a reference
           transcription has zero length
        2. If `eos` is set and `include_eos` is :obj:`True`, will warn when a
           transcript does not include an `eos` symbol

    Returns
    -------
    er : torch.FloatTensor
    '''
    er = _levenshtein(
        ref, hyp, eos, include_eos, batch_first, ins_cost,
        del_cost, sub_cost, warn, norm=norm)
    return er


def optimal_completion(
        ref, hyp, eos=None, include_eos=True, batch_first=False, ins_cost=1.,
        del_cost=1., sub_cost=1., padding=pydrobert.torch.INDEX_PAD_VALUE,
        exclude_last=False, warn=True):
    r'''Return a mask of next tokens of a minimum edit distance prefix

    Given a reference transcript `ref` of shape ``(max_ref_steps, batch_size)``
    (or ``(batch_size, max_ref_steps)`` if `batch_first` is :obj:`True`) and a
    hypothesis transcript `hyp` of shape ``(max_hyp_steps, batch_size)``, (or
    ``(batch_size, max_hyp_steps)``), this function produces a long tensor
    `optimals` of shape ``(max_hyp_steps + 1, batch_size, max_unique_next)``
    (or ``(batch_size, max_hyp_steps + 1, max_unique_next)``), where
    ``max_unique_next <= max_ref_steps``, of the unique tokens that could be
    added to the hypothesis prefix ``hyp[:prefix_len, batch]`` such that some
    remaining suffix concatenated to the prefix would result in a minimal edit
    distance. See below for an example

    Parameters
    ----------
    ref : torch.LongTensor
    hyp : torch.LongTensor
    eos : int, optional
        A special token in `ref` and `hyp` whose first occurrence in each
        batch indicates the end of a transcript. This allows for
        variable-length transcripts in the batch
    include_eos : bool, optional
        Whether to include the first instance of `eos` found in both `ref` and
        `hyp` as valid tokens to be computed as part of the distance and next
        tokens for a suffix. Only the first `eos` per transcript is included
    batch_first : bool, optional
    ins_cost : float, optional
        The cost of an adding a superfluous token to a transcript in `hyp`
    del_cost : float, optional
        The cost of missing a token from `ref`
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`
    padding : int, optional
        The value to right-pad unequal-length sequences with
    exclude_last : bool, optional
        If true, will exclude the final prefix, consisting of the entire
        transcript, from the returned `optimals`. Optimals will be of shape
        ``(max_hyp_steps, batch_size, max_unique_next)``
    warn : bool, optional
        Whether to display warnings on irregularities. Currently, this only
        occurs when `eos` is set, `include_eos` is :obj:`True`, and a
        transcript does not contain the `eos` symbol

    Returns
    -------
    optimals : torch.LongTensor

    Examples
    --------

    Consider the reference text "foot" and the hypothesis text "bot". The below
    shows the matrix used to calculate edit distances between them::

        \ _ f o o t
        _ 0 1 2 3 4
        b 1 1 2 3 4
        o 2 2 1 2 3
        t 3 3 2 2 2

    If ``prefix_len == 0``, then the prefix is "", and "f" (from the suffix
    "foot") is the only subsequent token that would not increase the edit
    distance from that of the prefix (0). If ``prefix_len == 1``, then the
    prefix is "b". To arrive at the minimum edit distance for "b", one either
    treats "b" as an insertion or a substitution for "f", yielding suffixes
    "foot" and "oot". Thus, the subsequent token could be "f" or "o". For the
    prefix "bo", the minimum edit distance is achieved by first substituting
    "f" for "b", then substituting "o" for "o", resulting in the suffix "ot"
    and the next optimal character "o". Finally, for ``prefix_len == 3`` and
    prefix "bot", there are many operations that can produce the minimum edit
    distance of 2, resulting in one of the suffixes "ot", "t", and "". The
    latter suffix requires no more tokens and so any operation would increase
    the edit distance. Thus the optimal next tokens could be "o" or "t".

    Plugging "foot" and "bot" into this function, we get the prefixes:

    >>> ref_text, hyp_text = "foot", "bot"
    >>> ref = torch.tensor([ord(c) for c in ref_text]).unsqueeze(1)
    >>> hyp = torch.tensor([ord(c) for c in hyp_text]).unsqueeze(1)
    >>> optimal = optimal_completion(ref, hyp).squeeze(1)
    >>> for prefix_len, o_for_pr in enumerate(optimal):
    ...     o_for_pr = o_for_pr.masked_select(o_for_pr.ge(0)).tolist()
    ...     print('prefix={}: {}'.format(
    ...         hyp_text[:prefix_len], ','.join([chr(i) for i in o_for_pr])))
    prefix=: f
    prefix=b: f,o
    prefix=bo: o
    prefix=bot: o,t

    See Also
    --------
    pydrobert.torch.training.HardOptimalCompletionDistillationLoss
        A loss function that uses these optimal completions to train a model
    '''
    mask = _levenshtein(
        ref, hyp, eos, include_eos, batch_first, ins_cost, del_cost,
        sub_cost, warn, return_mask=True, exclude_last=exclude_last,
    )
    max_hyp_steps_p1, max_ref_steps, batch_size = mask.shape
    targets = []
    if batch_first:
        for mask_bt, ref_bt in zip(mask.transpose(0, 2), ref):
            for mask_bt_hyp in mask_bt.t():
                targets.append(torch.unique(ref_bt.masked_select(mask_bt_hyp)))
    else:
        for mask_hyp in mask:
            for mask_hyp_bt, ref_bt in zip(mask_hyp.t(), ref.t()):
                targets.append(torch.unique(ref_bt.masked_select(mask_hyp_bt)))
    targets = torch.nn.utils.rnn.pad_sequence(
        targets, padding_value=padding, batch_first=True)
    if batch_first:
        targets = targets.view(batch_size, max_hyp_steps_p1, -1)
    else:
        targets = targets.view(max_hyp_steps_p1, batch_size, -1)
    return targets


def optimizer_to(optimizer, to):
    '''Move tensors in an optimizer to another device

    This function traverses the state dictionary of an `optimizer` and pushes
    any tensors it finds to the device implied by `to` (as per the semantics
    of :func:`torch.Tensor.to`)
    '''
    state_dict = optimizer.state_dict()
    key_stack = [(x,) for x in state_dict.keys()]
    new_device = False
    while key_stack:
        last_dict = None
        val = state_dict
        keys = key_stack.pop()
        for key in keys:
            last_dict = val
            val = val[key]
        if isinstance(val, torch.Tensor):
            try:
                last_dict[keys[-1]] = val.to(to)
                if last_dict[keys[-1]].device != val.device:
                    new_device = True
            except Exception as e:
                print(e)
                pass
        elif hasattr(val, 'keys'):
            key_stack += [keys + (x,) for x in val.keys()]
    if new_device:
        optimizer.load_state_dict(state_dict)


def prefix_error_rates(
        ref, hyp, eos=None, include_eos=True, norm=True, batch_first=False,
        ins_cost=1., del_cost=1., sub_cost=1.,
        padding=pydrobert.torch.INDEX_PAD_VALUE, exclude_last=False,
        warn=True):
    '''Compute the error rate between ref and each prefix of hyp

    Given a reference transcript `ref` of shape ``(max_ref_steps, batch_size)``
    (or ``(batch_size, max_ref_steps)`` if `batch_first` is :obj:`True`) and a
    hypothesis transcript `hyp` of shape ``(max_hyp_steps, batch_size)``, (or
    ``(batch_size, max_hyp_steps)``), this function produces a tensor
    `prefix_ers` of shape ``(max_hyp_steps + 1, batch_size)`` (or
    ``(batch_size, max_hyp_steps + 1))`` which contains the error rates for
    each prefix of each hypothesis, starting from the empty prefix

    Parameters
    ----------
    ref : torch.LongTensor
    hyp : torch.LongTensor
    eos : int, optional
        A special token in `ref` and `hyp` whose first occurrence in each
        batch indicates the end of a transcript. This allows for
        variable-length transcripts in the batch
    include_eos : bool, optional
        Whether to include the first instance of `eos` found in both `ref` and
        `hyp` as valid tokens to be computed as part of the distance.
        Only the first `eos` per transcript is included
    norm : bool, optional
        If :obj:`False`, will return edit distances instead of error rates
    batch_first : bool, optional
    ins_cost : float, optional
        The cost of an adding a superfluous token to a transcript in `hyp`
    del_cost : float, optional
        The cost of missing a token from `ref`
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`
    padding : int, optional
        The value to right-pad the error rates of unequal-length sequences with
        in `prefix_ers`
    exclude_last : bool, optional
        If true, will exclude the final prefix, consisting of the entire
        transcript, from the returned `dists`. `dists` will be of shape
        ``(max_hyp_steps, batch_size, max_unique_next)``
    warn : bool, optional
        Whether to display warnings on irregularities. Currently, this only
        occurs when `eos` is set, `include_eos` is :obj:`True`, and a
        transcript does not contain the `eos` symbol

    Returns
    -------
    prefix_ers : torch.tensor

    See Also
    --------
    :ref:`Gradient Estimators`
        Provides an example where this function is used to determine a reward
        function for reinforcement learning
    '''
    prefix_ers = _levenshtein(
        ref, hyp, eos, include_eos, batch_first, ins_cost,
        del_cost, sub_cost, warn, norm=norm, return_prf_dsts=True,
        exclude_last=exclude_last, padding=padding,
    )
    return prefix_ers


def random_walk_advance(
        logits_t, num_samp, y_prev=None, eos=pydrobert.torch.INDEX_PAD_VALUE,
        lens=None, prevent_eos=False, include_relaxation=False):
    r'''Advance a random walk of sequences

    Suppose a model outputs a un-normalized log-probability distribution over
    the next element of a sequence in `logits_t` s.t.

    .. math::

        Pr(y_t = c) = exp(logits_{t,c}) / \sum_k exp(logits_k)

    We assume :math:`logits_t` is a function of what comes before
    :math:`logits_t = f(logits_{<t}, y_{<t})`. Alternatively, letting
    :math:`s_t = (logits_t, y_t)`, :math:`s` is a Markov Chain. A model is
    auto-regressive if :math:`f` depends on :math:`y_{<t}`, and is not
    auto-regressive if :math:`logits_t = f(logits_{<t})`.

    A random walk can be performed over a Markov Chain by sampling the elements
    :math:`y_t` of the greater sequence `y` one at a time, according to
    :math:`Pr(y_t = c)`. This allows us to sample the distribution
    :math:`Pr(Y)`.

    This function is called at every time step. It updates the sequences
    being built (`y_prev`) with one additional token and returns `y`. This
    function is intended to be coupled with an auto-regressive model, where
    `logits_t` is not known until :math:`y_t` is known. If the model is
    not auto-regressive, it is much more efficient to gather all `logits_t`
    into one :math:`logits` and sample all at once. See the examples section
    below for both behaviours

    Parameters
    ----------
    logits_t : torch.FloatTensor
        The conditional probabilities over class labels for the current time
        step. Either of shape ``(num_batches, old_samp, num_classes)``,
        where ``old_samp`` is the number of samples in the previous time
        step, or ``(num_batches, num_classes)``, where it is assumed that
        ``old_samp == 1``
    num_samp : int
        The number of samples to be drawn. Either ``old_samp == 1`` and/or
        ``num_samp <= old_samp`` must be :obj:`True`. That is, either all
        samples will share the same prefix, or we are building off a subset of
        the samples from ``y_prev`` (in this case, always the first `num_samp`)
    y_prev : torch.LongTensor, optional
        A tensor of shape ``(t - 1, num_batches, old_samp)`` or
        ``(t - 1, num_batches)`` specifying :math:`y_{<t}`. If unspecified,
        it is assumed that ``t == 1``
    eos : int, optional
        A special end-of-sequence symbol indicating that the beam has ended.
        Can be a class index. If this value occurs in in
        ``y_prev[-1, bt, smp]`` for some batch ``bt`` and sample ``smp``,
        `eos` will be appended to ``y_prev[:, bt, smp]``
    lens : torch.LongTensor, optional
        A tensor of shape ``(num_batches,)``. If ``t > lens[bt]`` for some
        batch ``bt``, all samples for ``bt`` will be considered finished. `eos`
        will be appended to `y_prev`
    prevent_eos : bool, optional
        Setting this flag to :obj:`True` will keep `eos` targets from being
        drawn unless a sample has finished (either with a prior `eos` or
        through `lens`). Note that this will only have an effect when ``0 <=
        eos <= num_classes``
    include_relaxation : bool, optional
        If :obj:`True`, a tuple will be returned whose second element is `z`,
        see below


    Returns
    -------
    y : torch.LongTensor
        A long tensor of shape ``(t, num_batches, num_samp)`` of the sampled
        sequences so far. Note that, since :math:`y_t` are drawn `i.i.d.`,
        there is no guarantee of the uniqueness of each `num_samp` samples
    z : torch.FloatTensor
        Only included if `include_relaxation` is :obj:`True`. `z` is a sample
        of a continuous relaxation of the categorical distribution of `logits`
        of shape ``(num_batches, num_samp, num_classes). Assuming ``y_prev[-1,
        bt, smp] != eos``, ``y[-1, bt, smp] == z[bt, smp].argmax(dim-1)``. If
        ``y_prev[-1, bt, smp] == eos``, ``z[bt, smp, :] = -infinity``. The
        primary purpose of `z` is to be used as an argument (alongside `y`) in
        more complicated gradient estimators from
        :mod:`pydrobert.torch.estimators`

    Examples
    --------

    Here is an example of random path sampling with a non-auto-regressive
    RNN. It does not need this function, and can take advantage of packed
    sequences for efficiency and gradient validity.

    >>> N, I, C, T, W, H, eos = 5, 4, 10, 100, 6, 15, 0
    >>> rnn = torch.nn.RNN(I, H)
    >>> ff = torch.nn.Linear(H, C)
    >>> inp = torch.rand(T, N, I)
    >>> lens = torch.randint(1, T + 1, (N,)).sort(descending=True)[0]
    >>> packed_inp = torch.nn.utils.rnn.pack_padded_sequence(inp, lens)
    >>> packed_h, _ = rnn(packed_inp)
    >>> packed_logits = ff(packed_h[0])
    >>> packed_logits_dup = packed_logits.detach().unsqueeze(1)
    >>> packed_logits_dup = packed_logits_dup.expand(-1, W, -1)  # (flat, W, C)
    >>> packed_y = torch.distributions.Categorical(
    ...     logits=packed_logits_dup).sample()  # (flat, W)
    >>> # we pad y with "eos" to ensure each sample is done by its length,
    >>> # but "eos" may have occurred beforehand
    >>> y = torch.nn.utils.rnn.pad_packed_sequence(
    ...     torch.nn.utils.rnn.PackedSequence(
    ...         packed_y, batch_sizes=packed_h[1]),
    ...     padding_value=eos, total_length=T,
    ... )[0]  # (T, N, W) (batch index gets inserted as 2nd dim)

    Here is an auto-regressive RNN that uses this function to build partial
    samples into `y`

    >>> N, I, C, T, W, H, eos, start = 5, 5, 10, 100, 5, 10, 0, -1
    >>> cell = torch.nn.RNNCell(I + 1, H)
    >>> ff = torch.nn.Linear(H, C)
    >>> inp = torch.rand(T, N, I)
    >>> y = torch.full((1, N, 1), start, dtype=torch.long)
    >>> h_t = torch.zeros(N, 1, H)
    >>> for inp_t in inp:
    >>>     y_tm1 = y[-1]
    >>>     old_samp = y_tm1.shape[-1]
    >>>     inp_t = inp_t.unsqueeze(1).expand(N, old_samp, I)
    >>>     x_t = torch.cat([inp_t, y_tm1.unsqueeze(2).float()], -1)
    >>>     h_t = cell(
    ...         x_t.view(N * old_samp, I + 1),
    ...         h_t.view(N * old_samp, H),
    ...     ).view(N, old_samp, H)
    >>>     logits_t = ff(h_t)
    >>>     y = random_walk_advance(logits_t, W, y, eos)
    >>>     if old_samp == 1:
    >>>         h_t = h_t.expand(-1, W, H).contiguous()

    See Also
    --------
    :ref:`Gradient Estimators`
        Includes a use case for `include_relaxation`
    '''
    if logits_t.dim() == 2:
        logits_t = logits_t.unsqueeze(1)
    elif logits_t.dim() != 3:
        raise ValueError('logits_t must have dimension of either 2 or 3')
    num_batches, old_samp, num_classes = logits_t.shape
    if prevent_eos and 0 <= eos < num_classes:
        logits_t[..., eos] = torch.tensor(
            -float('inf'), device=logits_t.device)
    if old_samp != 1 and num_samp > old_samp:
        raise ValueError(
            'either old_samp == 1 or num_samp <= old_samp must be true')
    eos_mask = None
    if y_prev is not None:
        if y_prev.dim() == 2:
            y_prev = y_prev.unsqueeze(2)
        if y_prev.shape[1:] != logits_t.shape[:-1]:
            raise ValueError(
                'If logits_t of shape {} then y_prev must have shape '
                '(*, {}, {})'.format(
                    (num_batches, old_samp, num_classes),
                    num_batches, old_samp,
                )
            )
        y_prev = y_prev.expand(-1, -1, num_samp)
        eos_mask = y_prev[-1].eq(eos)
        if eos_mask.any():
            eos_mask = eos_mask[..., :num_samp]
        else:
            eos_mask = None
        t = y_prev.shape[0] + 1
    else:
        t = 1
    logits_t = logits_t.expand(-1, num_samp, -1)
    if lens is not None:
        if lens.shape != logits_t.shape[:1]:
            raise ValueError('lens must be of shape ({},)'.format(num_batches))
        len_mask = lens.lt(t)
        if torch.any(len_mask):
            len_mask = len_mask.unsqueeze(1).expand(-1, num_samp)
            eos_mask = len_mask if eos_mask is None else (eos_mask | len_mask)
    u = torch.distributions.utils.clamp_probs(torch.rand_like(logits_t))
    log_theta = torch.nn.functional.log_softmax(logits_t, dim=-1)
    z = log_theta - torch.log(-torch.log(u))
    y = z.argmax(dim=-1)
    if eos_mask is not None:
        y = y.masked_fill(eos_mask, eos)
        z = z.masked_fill(eos_mask.unsqueeze(-1), -float('inf'))
    y = y.unsqueeze(0)
    if y_prev is not None:
        y = torch.cat([y_prev, y], 0)
    if include_relaxation:
        return y, z
    else:
        return y


def _levenshtein(
        ref, hyp, eos, include_eos, batch_first, ins_cost, del_cost,
        sub_cost, warn, norm=False, return_mask=False,
        return_prf_dsts=False, exclude_last=False, padding=None):
    assert not return_mask or not return_prf_dsts
    if ref.dim() != 2 or hyp.dim() != 2:
        raise ValueError('ref and hyp must be 2 dimensional')
    if batch_first:
        ref = ref.t()
        hyp = hyp.t()
    ref = ref.detach()
    hyp = hyp.detach()
    max_ref_steps, batch_size = ref.shape
    max_hyp_steps, batch_size_ = hyp.shape
    device = ref.device
    if batch_size != batch_size_:
        raise ValueError(
            'ref has batch size {}, but hyp has {}'.format(
                batch_size, batch_size_))
    if eos is not None:
        ref_lens = torch.full_like(ref[0], max_ref_steps)
        hyp_lens = torch.full_like(hyp[0], max_hyp_steps)
        for coord in ref.eq(eos).nonzero():
            ref_lens[..., coord[1]] = torch.min(ref_lens[coord[1]], coord[0])
        for coord in hyp.eq(eos).nonzero():
            hyp_lens[..., coord[1]] = torch.min(hyp_lens[coord[1]], coord[0])
        if include_eos:
            ref_eq_mask = ref_lens == max_ref_steps
            ref_lens = ref_lens + 1
            if ref_eq_mask.any():
                if warn:
                    warnings.warn(
                        "include_eos=True, but a transcription in ref did not "
                        "contain the eos symbol ({}). To suppress this "
                        "warning, set warn=False".format(eos))
                ref_lens = torch.where(
                    ref_eq_mask,
                    ref_lens - 1,
                    ref_lens
                )
            hyp_eq_mask = hyp_lens == max_hyp_steps
            hyp_lens = hyp_lens + 1
            if hyp_eq_mask.any():
                if warn:
                    warnings.warn(
                        "include_eos=True, but a transcription in hyp did not "
                        "contain the eos symbol ({}). To suppress this "
                        "warning, set warn=False".format(eos))
                hyp_lens = torch.where(
                    hyp_eq_mask,
                    hyp_lens - 1,
                    hyp_lens
                )
            del ref_eq_mask, hyp_eq_mask
    else:
        ref_lens = torch.full_like(ref[0], max_ref_steps)
        hyp_lens = torch.full_like(hyp[0], max_hyp_steps)
    ins_cost = torch.tensor(float(ins_cost), device=device)
    del_cost = torch.tensor(float(del_cost), device=device)
    sub_cost = torch.tensor(float(sub_cost), device=device)
    zero = torch.tensor(0., device=device)
    batch_range = torch.arange(batch_size, device=device)
    if return_mask:
        mask = torch.empty(
            (
                max_hyp_steps + (0 if exclude_last else 1),
                max_ref_steps, batch_size),
            device=device, dtype=torch.uint8)
        mask[0, 0] = 1
        mask[0, 1:] = 0
    elif return_prf_dsts:
        assert padding is not None
        prefix_ers = torch.empty(
            (max_hyp_steps + (0 if exclude_last else 1), batch_size),
            device=device, dtype=torch.float,
        )
        prefix_ers[0] = ref_lens
    # direct row down corresponds to insertion
    # direct col right corresponds to a deletion
    row = torch.arange(
        max_ref_steps + 1, device=device, dtype=torch.float
    ).unsqueeze(1).expand(max_ref_steps + 1, batch_size)
    last_row = torch.empty_like(row)
    # we vectorize as much as we can. Neither substitutions nor insertions
    # require values from the current row to be computed, and since the last
    # row can't be altered, we can easily vectorize there. We can't do the same
    # with deletions because they rely on what came before in the row
    for hyp_idx in range(1, max_hyp_steps + 1):
        last_row = row
        row = torch.where(
            hyp_lens < hyp_idx,
            last_row,
            last_row + ins_cost
        )
        sub_row = torch.where(
            ref == hyp[hyp_idx - 1],
            last_row[:-1],
            last_row[:-1] + sub_cost,
        )
        row[1:] = torch.min(row[1:], sub_row)
        for ref_idx in range(1, max_ref_steps + 1):
            row[ref_idx] = torch.min(row[ref_idx], row[ref_idx - 1] + del_cost)
        if return_mask and (hyp_idx < max_hyp_steps or not exclude_last):
            # As proven in the OCD paper, the optimal targets are always the
            # first character of a suffix of the reference transcript that
            # remains to be aligned. The levenshtein operation
            # corresponding to what we do with that target would be a matched
            # substitution (i.e. hyp's next token is the OCD target, resulting
            # in no change in cost from the prefix). Thus, given a levenshtein
            # matrix for one of these OCD targets (which is this matrix,
            # except for the final row), the minimal values on the final row
            # sit on a diagonal from the minimal values of the current row.
            mins = row.min(0, keepdim=True)[0]
            row_mask = (row[:-1] == mins)
            if exclude_last:
                row_mask = row_mask & (hyp_idx < hyp_lens)
            else:
                row_mask = row_mask & (hyp_idx <= hyp_lens)
            mask[hyp_idx] = row_mask
        elif return_prf_dsts and (hyp_idx < max_hyp_steps or not exclude_last):
            prefix_ers[hyp_idx] = row[ref_lens, batch_range]
    if return_mask:
        mask = mask & (
            (torch.arange(max_ref_steps, device=device)
                .unsqueeze(1)
                .expand(max_ref_steps, batch_size) < ref_lens)
            .unsqueeze(0)
        )
        return mask
    elif return_prf_dsts:
        if norm:
            prefix_ers = prefix_ers / ref_lens.float()
            zero_mask = ref_lens.eq(0).unsqueeze(0)
            if zero_mask.any():
                if warn:
                    warnings.warn(
                        "ref contains empty transcripts. Error rates will be "
                        "0 for prefixes of length 0, 1 otherwise. To suppress "
                        "this warning, set warn=False"
                    )
                prefix_ers = torch.where(
                    zero_mask,
                    (
                        torch.arange(prefix_ers.shape[0], device=device)
                        .gt(0).float().unsqueeze(1).expand_as(prefix_ers)
                    ),
                    prefix_ers,
                )
        prefix_ers = prefix_ers.masked_fill(
            (
                torch.arange(prefix_ers.shape[0], device=device).unsqueeze(1)
                .ge(hyp_lens + (0 if exclude_last else 1))
            ),
            padding,
        )
        if batch_first:
            prefix_ers = prefix_ers.t()
        return prefix_ers
    er = row[ref_lens, batch_range]
    if norm:
        er = er / ref_lens.float()
        zero_mask = ref_lens.eq(0)
        if zero_mask.any():
            if warn:
                warnings.warn(
                    "ref contains empty transcripts. Error rates for entries "
                    "will be 1 if any insertion and 0 otherwise. To suppress "
                    "this warning, set warn=False")
            er = torch.where(
                zero_mask,
                hyp_lens.gt(0).float(),
                er,
            )
    return er
