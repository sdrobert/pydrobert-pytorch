# Copyright 2021 Sean Robertson
#
# Code for polyharmonic_spline is converted from tensorflow code
# https://github.com/tensorflow/addons/blob/v0.11.2/tensorflow_addons/image/interpolate_spline.py
# code for sparse_image_warp is derived from tensorflow code, though it's not identical
# https://github.com/tensorflow/addons/blob/v0.11.2/tensorflow_addons/image/sparse_image_warp.py
#
# Which are also Apache 2.0 Licensed:
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Utility functions"""

import re
from typing import Optional, TextIO, Tuple, Union
import warnings

import torch
import pydrobert.torch

__all__ = [
    "beam_search_advance",
    "ctc_greedy_search",
    "ctc_prefix_search_advance",
    "dense_image_warp",
    "edit_distance",
    "error_rate",
    "optimal_completion",
    "parse_arpa_lm",
    "polyharmonic_spline",
    "prefix_edit_distances",
    "prefix_error_rates",
    "random_walk_advance",
    "sequence_log_probs",
    "sparse_image_warp",
    "time_distributed_return",
]


def parse_arpa_lm(file_: Union[TextIO, str], token2id: Optional[dict] = None) -> list:
    r"""Parse an ARPA statistical language model

    An `ARPA language model <https://cmusphinx.github.io/wiki/arpaformat/>`__
    is an n-gram model with back-off probabilities. It is formatted as

    ::

        \data\
        ngram 1=<count>
        ngram 2=<count>
        ...
        ngram <N>=<count>

        \1-grams:
        <logp> <token[t]> <logb>
        <logp> <token[t]> <logb>
        ...

        \2-grams:
        <logp> <token[t-1]> <token[t]> <logb>
        ...

        \<N>-grams:
        <logp> <token[t-<N>+1]> ... <token[t]>
        ...

        \end\

    Parameters
    ----------
    file_ : str or file
        Either the path or a file pointer to the file
    token2id : dict, optional
        A dictionary whose keys are token strings and values are ids. If set,
        tokens will be replaced with ids on read

    Returns
    -------
    prob_list : list
        A list of the same length as there are orders of n-grams in the
        file (e.g. if the file contains up to tri-gram probabilities then
        `prob_list` will be of length 3). Each element is a dictionary whose
        key is the word sequence (earliest word first). For 1-grams, this is
        just the word. For n > 1, this is a tuple of words. Values are either
        a tuple of ``logp, logb`` of the log-probability and backoff
        log-probability, or, in the case of the highest-order n-grams that
        don't need a backoff, just the log probability
    """
    if isinstance(file_, str):
        with open(file_) as f:
            return parse_arpa_lm(f, token2id=token2id)
    line = ""
    for line in file_:
        if line.strip() == "\\data\\":
            break
    if line.strip() != "\\data\\":
        raise IOError("Could not find \\data\\ line. Is this an ARPA file?")
    ngram_counts = []
    count_pattern = re.compile(r"^ngram\s+(\d+)\s*=\s*(\d+)$")
    for line in file_:
        line = line.strip()
        if not line:
            continue
        match = count_pattern.match(line)
        if match is None:
            break
        n, count = (int(x) for x in match.groups())
        if len(ngram_counts) < n:
            ngram_counts.extend(0 for _ in range(n - len(ngram_counts)))
        ngram_counts[n - 1] = count
    prob_list = [dict() for _ in ngram_counts]
    ngram_header_pattern = re.compile(r"^\\(\d+)-grams:$")
    ngram_entry_pattern = re.compile(r"^(-?\d+(?:\.\d+)?)\s+(.*)$")
    while line != "\\end\\":
        match = ngram_header_pattern.match(line)
        if match is None:
            raise IOError('line "{}" is not valid'.format(line))
        ngram = int(match.group(1))
        if ngram > len(ngram_counts):
            raise IOError(
                "{}-grams count was not listed, but found entry" "".format(ngram)
            )
        dict_ = prob_list[ngram - 1]
        for line in file_:
            line = line.strip()
            if not line:
                continue
            match = ngram_entry_pattern.match(line)
            if match is None:
                break
            logp, rest = match.groups()
            tokens = tuple(rest.strip().split())
            # IRSTLM and SRILM allow for implicit backoffs on non-final
            # n-grams, but final n-grams must not have backoffs
            logb = 0.0
            if len(tokens) == ngram + 1 and ngram < len(prob_list):
                try:
                    logb = float(tokens[-1])
                    tokens = tokens[:-1]
                except ValueError:
                    pass
            if len(tokens) != ngram:
                raise IOError(
                    'expected line "{}" to be a(n) {}-gram' "".format(line, ngram)
                )
            if token2id is not None:
                tokens = tuple(token2id[tok] for tok in tokens)
            if ngram == 1:
                tokens = tokens[0]
            if ngram != len(ngram_counts):
                dict_[tokens] = (float(logp), logb)
            else:
                dict_[tokens] = float(logp)
    if line != "\\end\\":
        raise IOError("Could not find \\end\\ line")
    for ngram_m1, (ngram_count, dict_) in enumerate(zip(ngram_counts, prob_list)):
        if len(dict_) != ngram_count:
            raise IOError(
                "Expected {} {}-grams, got {}".format(ngram_count, ngram_m1, len(dict_))
            )
    return prob_list


def beam_search_advance(
    logits_t: torch.Tensor,
    width: int,
    log_prior: Optional[torch.Tensor] = None,
    y_prev: Optional[torch.Tensor] = None,
    eos: int = pydrobert.torch.INDEX_PAD_VALUE,
    lens: Optional[torch.Tensor] = None,
    prevent_eos: bool = False,
    distribution: bool = True,
):
    r"""Advance a beam search

    Suppose a model outputs a un-normalized log-probability distribution over the next
    element of a sequence in `logits_t` s.t.

    .. math::

        Pr(y_t = c ; log\_prior) = exp(logits_{t,c}) / \sum_k exp(logits_{t,k})

    We assume :math:`logits_t` is a function of what comes before :math:`logits_t =
    f(logits_{<t}, y_{<t})`. Alternatively, letting :math:`s_t = (logits_t, y_t)`,
    :math:`s` is a Markov Chain. A model is auto-regressive if :math:`f` depends on
    :math:`y_{<t}`, and is not auto-regressive if :math:`logits_t = f(logits_{<t})`.

    Beam search is a heuristic mechanism for determining a best path, i.e. :math:`\arg
    \max_y Pr(y)` that maximizes the probability of the best path by keeping track of
    `width` high probability paths called "beams" (the aggregate of which for a given
    batch element is named, unfortunately, "the beam"). If the model is auto-regressive,
    beam search is only approximate. However, if the model is not auto-regressive, beam
    search gives an exact n-best list.

    This function is called at every time step. It updates old beam log-probabilities
    (`log_prior`) with new ones (`score`) from the conditional derived from `logits_t`,
    and updates us the class indices emitted between them (`y`). See the examples
    section for how this might work.

    Parameters
    ----------
    logits_t : torch.Tensor
        A float tensor of conditional probabilities over class labels for the current
        time step. Either of shape ``(batch_size, old_width, num_classes)``, where
        ``old_width`` is the number of beams in the previous time step, or
        ``(batch_size, num_classes)``, where it is assumed that ``old_width == 1``
    width : int
        The number of beams in the beam to produce for the current time step. ``width
        <= num_classes``
    log_prior : torch.Tensor, optional
        A float tensor of (or proportional to) log prior probabilities of beams up
        to the previous time step. Either of shape ``(batch_size, old_width)``
        or ``(batch_size,)``. If unspecified, a uniform log prior will be used
    y_prev : torch.Tensor, optional
        A long tensor of shape ``(t - 1, batch_size, old_width)`` or ``(t - 1,
        batch_size)`` specifying :math:`y_{<t}`. If unspecified, it is assumed that ``t
        == 1``
    eos : int, optional
        A special end-of-sequence symbol indicating that the beam has ended. Can be a
        class index. If this value occurs in in ``y_prev[-1, bt, bm]`` for some batch
        ``bt`` and beam ``bm``, that beam will be padded with an `eos` token and the
        score for that beam won't change
    lens : torch.Tensor, optional
        A long tensor of shape ``(batch_size,)``. If ``t > lens[bt]`` for some batch
        ``bt``, all beams for ``bt`` will be considered finished. All scores will be
        fixed and `eos` will be appended to `y_prev`
    prevent_eos : bool, optional
        Setting this flag to :obj:`True` will keep `eos` targets from entering a beam
        unless it has finished (either with a prior `eos` or through `lens`). Note that
        this will only have an effect when ``0 <= eos <= num_classes``
    distribution : bool, optional
        If :obj:`False`, a log-softmax will not be applied to `logits_t` prior to
        calculating the beams. Disabling `distribution` should be avoided when the beam
        search is being performed directly on model output. Setting `distribution` to
        :obj:`False` is useful when bootstrapping a language model probability
        distribution to the search

    Returns
    -------
    score : torch.Tensor
        A float tensor of shape ``(batch_size, width)`` of the log-joint probabilitie
         of the new beams in the beam
    y : torch.Tensor
        A long tensor of shape ``(t, batch_size, width)`` of indices of the class
        labels generated up to this point
    s : torch.Tensor
        A long tensor shape ``(batch_size, width)`` of indices of beams in the old beam
        which prefix the new beam. Note that beams in the new beam are sorted by
        descending probability

    Examples
    --------

    Auto-regressive decoding with beam search. We assume that all input have the same
    number of steps

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

    Were we to have a :class:`pydrobert.torch.layers.SequentialLanguageModel`,
    we could modify `logits_t`, and thus the search, to account for it:

    >>> # ... same as before
    >>> logits_t = torch.nn.functional.log_softmax(logits_t, -1)
    >>> logits_t = logits_t + lmb * lm(y)  # lmb is some constant
    >>> score, y, s_t = beam_search_advance(
    ...     logits_t, W, score, y, eos, distribution=False)
    >>> # ... same as before

    Note that `score` would no longer reflect a log-joint probability.

    The following produces a ``W``-best list for non-auto-regressive model. We don't
    emit an `eos`, instead completing the sequence when we've hit the target length via
    `lens`

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
    """
    if logits_t.dim() == 2:
        logits_t = logits_t.unsqueeze(1)
    elif logits_t.dim() != 3:
        raise RuntimeError("logits_t must have dimension of either 2 or 3")
    if distribution:
        logits_t = torch.nn.functional.log_softmax(logits_t, 2)
    neg_inf = torch.tensor(-float("inf"), device=logits_t.device)
    batch_size, old_width, num_classes = logits_t.shape
    if log_prior is None:
        log_prior = torch.full(
            (batch_size, old_width),
            -torch.log(torch.tensor(float(num_classes))),
            dtype=logits_t.dtype,
            device=logits_t.device,
        )
    elif tuple(log_prior.shape) == (batch_size,) and old_width == 1:
        log_prior = log_prior.unsqueeze(1)
    elif log_prior.shape != logits_t.shape[:-1]:
        raise RuntimeError(
            "If logits_t of shape {} then log_prior must have shape {}".format(
                (batch_size, old_width, num_classes),
                (batch_size, old_width),
            )
        )
    if prevent_eos and 0 <= eos < num_classes:
        # we have to put this before the num_done check so that it'll be
        # overwritten for paths that have finished already
        logits_t[..., eos] = neg_inf
    eos_set = None
    if y_prev is not None:
        if y_prev.dim() == 2:
            y_prev = y_prev.unsqueeze(2)
        if y_prev.shape[1:] != log_prior.shape:
            raise RuntimeError(
                "If logits_t of shape {} then y_prev must have shape "
                "(*, {}, {})".format(
                    (batch_size, old_width, num_classes),
                    batch_size,
                    old_width,
                )
            )
        eos_mask = y_prev[-1].eq(eos)
        num_done = eos_mask.long().sum(1)
        if num_done.sum().item():
            if old_width < width and torch.any(num_done == old_width):
                raise RuntimeError(
                    "New beam width ({}) is wider than old beam width "
                    "({}), but all paths are already done in one or more "
                    "batch elements".format(width, old_width)
                )
            # we're going to treat class 0 as the sentinel for eos (even if
            # eos is a legit class label)
            done_classes = torch.full_like(logits_t[0, 0], neg_inf)
            done_classes[0] = 0.0
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
            raise RuntimeError("lens must be of shape ({},)".format(batch_size))
        len_mask = lens.lt(t)
        if torch.any(len_mask):
            if old_width < width:
                raise RuntimeError(
                    "New beam width ({}) is wider than old beam width "
                    "({}), but all paths are already done in one or more "
                    "batch elements".format(width, old_width)
                )
        else:
            len_mask = None
    joint = log_prior.unsqueeze(2) + logits_t
    score, idxs = torch.topk(joint.view(batch_size, -1), width, dim=1)
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
        y_prev = y_prev.gather(2, s.unsqueeze(0).expand(t - 1, batch_size, width))
        y = torch.cat([y_prev, y], 0)
    return score, y, s


def ctc_greedy_search(
    logits: torch.Tensor,
    in_lens: Optional[torch.Tensor] = None,
    blank_idx: int = -1,
    batch_first: bool = False,
    is_probs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CTC greedy search

    The CTC greedy search picks the path with the highest probability class in
    `logits` for each element in the sequence. The path (log-)probability is the (sum)
    product of the chosen type (log-probabilities). The output sequence are the
    resulting sequence of class labels with blanks and duplicates removed.

    Parameters
    ----------
    logits : torch.Tensor
        A float tensor of shape ``(N, T, V)`` where ``T`` is the sequence dimension,
        ``N`` is the batch dimension, and ``V`` is the number of classes including the
        blank label. ``logits[n, t, :]`` represent the unnormalized log-probabilities
        of the labels at time ``t`` in batch element ``n``.
    in_lens : torch.Tensor or None, optional
        If specified, a long tensor of shape ``(N,)`` providing the lengths of the
        sequence in the batch. For a given batch element ``n``, only the values of
        `logits` in the slice ``logits[n, :in_lens[n]]`` will be considered valid.
    blank_idx : int, optional
        Which index along the class dimension specifices the blank label
    batch_first : bool, optional
        If :obj:`False`, `logits` is of shape ``(T, N, V)`` and `paths` is of shape
        ``(T, N)``.
    is_probs : bool, optional
        If :obj:`True`, `logits` will be considered a normalized probability
        distribution instead of an un-normalized log-probability distribution. The
        return value `max_` will take the product of sequence probabilities instead of
        the sum.

    Returns
    -------
    max_, paths, out_lens : torch.Tensor, torch.Tensor, torch.Tensor
        `max_` is a float tensor of shape ``(N,)`` of the total probability of the
        greedy path. `paths` is a long tensor of shape ``(N, T`` which stores the
        reduced greedy paths. `out_lens` is a long tensor of shape ``(N,)`` which
        specifies the lengths of the greedy paths within `paths`: for a given batch
        element ``n``, the reduced greedy path is the sequence in the range
        ``paths[n, :out_lens[n]]``. The values of `paths` outside this range are
        undefined.
    """
    if logits.dim() != 3:
        raise RuntimeError("logits must be 3-dimensional")
    V = logits.size(2)
    if blank_idx < -V or blank_idx > (V - 1):
        raise IndexError(
            "Blank index out of range (expected to be in the range of "
            f"[-{V},{V-1}], but got {blank_idx})"
        )
    blank_idx = (blank_idx + V) % V
    if not is_probs:
        # normalize
        logits = logits.log_softmax(2)
    if not batch_first:
        # the masked_fill/scatter_ logic won't work if it isn't batch_first
        logits = logits.transpose(0, 1)
    max_, argmax = logits.max(2)
    keep_mask = argmax != blank_idx
    keep_mask[:, 1:] &= argmax[:, 1:] != argmax[:, :-1]
    seq_size = argmax.size(1)
    if in_lens is not None:
        in_len_mask = torch.arange(seq_size, device=argmax.device).unsqueeze(
            0
        ) < in_lens.unsqueeze(1)
        keep_mask = keep_mask & in_len_mask
        if is_probs:
            max_ = max_.masked_fill(~in_len_mask, 1.0)
        else:
            max_ = max_.masked_fill(~in_len_mask, 0.0)
        del in_len_mask
    out_lens = keep_mask.long().sum(1)
    data = argmax.masked_select(keep_mask)
    out_len_mask = torch.arange(seq_size, device=argmax.device).unsqueeze(
        0
    ) < out_lens.unsqueeze(1)
    if is_probs:
        max_ = max_.prod(1)
    else:
        max_ = max_.sum(1)
    argmax = argmax.masked_scatter_(out_len_mask, data)
    if not batch_first:
        argmax = argmax.t()
    return max_, argmax, out_lens


def ctc_prefix_search_advance(
    probs_t: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # ((N,K',V), (N,V), (N))
    width: int,  # K
    probs_prev: Tuple[torch.Tensor, torch.Tensor],  # (N,K'), (N,K')
    y_prev: torch.Tensor,  # (N, K', t-1)
    y_prev_last: torch.Tensor,  # (N,K')
    y_prev_lens: torch.Tensor,  # (N, K')
    prev_is_prefix: torch.Tensor,  # (N, K', K')  # [n, k, k'] iff k prefix of k'
    needs_sorted: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """CTC prefix search step function

    The step function of the CTC prefix search.

    Parameters
    ----------
    probs_t : (torch.Tensor, torch.Tensor, torch.Tensor)
        A triple of ``ext_probs_t, nonext_probs_t, blank_probs_t``. `ext_probs_t` is
        of shape ``(N, old_width, V)`` containing the probabilities of extending a
        prefix with a token of each type (resulting in a new token being added to the
        reduced transcription). `nonext_probs_t` has shape ``(N, V)`` and contains the
        probabilities of adding a token that does not extend a given prefix (i.e. when
        a token immediately follows another of the same type with no blanks in between).
        `blank_probs_t` is of shape ``(N)`` and contains the blank label probabilities.
    width : int
        The beam width
    probs_prev : (torch.Tensor, torch.Tensor)
        A pair of ``nb_probs_prev, b_probs_prev``. Each is a tensor of shape
        ``(N, old_width)``. `nb_probs_prev` contains the summed mass of the paths
        reducing to the given prefix which end in a non-blank token. `b_probs_prev`
        is the summed mass of the paths reducing to the given prefix which end in a
        blank token. ``nb_probs_prev + b_probs_prev = probs_prev``, the total mass of
        each prefix.
    y_prev : torch.Tensor
        A long tensor of shape ``(N, old_width, S)`` containing the (reduced) prefixes
        of each path.
    y_prev_last : torch.Tensor
        A long tensor of shape ``(N, old_width)`` containing the last token in each
        prefix. Arbitrary when the prefix is length 0.
    y_prev_lens: torch.Tensor
        A long tensor of shape ``(N, old_width)`` specifying the length of each prefix.
        For batch element ``n`` and prefix ``k``, only the tokens in
        ``y_prev[n, k, :y_prev_lens[n, k]]`` are valid.
    prev_is_prefix : torch.Tensor
        A boolean tensor of shape ``(N, old_width, old_width)``. ``prev_is_prefix[n, k,
        k']`` if and only if prefix ``k`` is a (non-strict) prefix of ``k'``
    needs_sorted : bool, optional
        If :obj:`False` the prefixes in the return value need not be sorted by
        decreasing probability (the top-`width`-probability prefixes will still be
        returned).

    Returns
    -------
    y_next, y_next_last, y_next_lens, probs_next, next_is_prefix, next_src,
    next_is_nonext : torch.Tensor, torch.Tensor, torch.Tensor, (torch.Tensor,
    torch.Tensor), torch.Tensor, torch.Tensor
        The first five are analogous to the ``*prev*`` arguments, but after the step
        has completed. `next_src` is a long tensor of shape ``(N, width)`` such that
        the value ``k_old = next_src[n, k_new]`` is the index from the previous step
        (over ``old_width``) that is a prefix of the new prefix at ``k_new`` (i.e. its
        source). `next_is_nonext` is a boolean tensor indicating if the new prefix
        did _not_ extend its source. If true, the new prefix is identical to the source.
        If false, it has one token more.

    See Also
    --------
    pydrobert.torch.layers.CTCPrefixSearch
        Performs the entirety of the search.

    Notes
    -----
    If an extending prefix matches a previous nonextending prefix, the former's mass
    is absorbed into the latter's.
    """

    ext_probs_t, nonext_probs_t, blank_probs_t = probs_t
    del probs_t

    nb_probs_prev, b_probs_prev = probs_prev
    tot_probs_prev = nb_probs_prev + b_probs_prev
    del probs_prev

    N, Kp, V = ext_probs_t.shape
    tm1 = y_prev.size(2)

    # b_ext_probs_cand is all zeros
    # nonblank extensions include blank prefix + extension and non-blank, non-matching
    # prefixes
    nb_ext_probs_cand = (
        nb_probs_prev.unsqueeze(2)
        .expand(N, Kp, V)
        .scatter(2, y_prev_last.unsqueeze(2), 0.0)
        + b_probs_prev.unsqueeze(2)
    ) * ext_probs_t  # (N, K', V)
    # blank non-extensions are all previous paths plus a blank
    b_nonext_probs_cand = tot_probs_prev * blank_probs_t.unsqueeze(1)  # (N, K')
    # nonblank non-extensions are non-blank, matching prefixes and final matching token
    nb_nonext_probs_cand = nb_probs_prev * nonext_probs_t.gather(1, y_prev_last)  # N,K'
    del nb_probs_prev, b_probs_prev, tot_probs_prev

    # An extending candidate can match an existing non-extending candidate.
    # We'll dump the extending candidate's probability mass into the non-extending
    # candidate's mass if they're equal.
    #
    # let's assume path k is a prefix of path k'. What's the token that we'd have to
    # match if we wanted to extend path k while remaining a prefix of k'?
    # y_prev[n, k', y_prev_lens[n, k]] = to_match[n, k, k']
    if tm1:
        to_match = (
            y_prev.gather(
                2, y_prev_lens.clamp(max=tm1 - 1).unsqueeze(1).expand(N, Kp, Kp)
            )
            .transpose(1, 2)
            .clamp(0, V - 1)
        )  # (N, K', K')
        # for k in range(Kp):
        #     for kp in range(Kp):
        #         print(f"k={k}, k'={kp}, to_match={to_match[0, k, kp].item()}")
    else:
        to_match = torch.zeros((N, Kp, Kp), device=y_prev.device, dtype=y_prev.dtype)
    # if we match to_match, will we be an exact match?
    ext_is_exact = (
        (y_prev_lens + 1).unsqueeze(2) == y_prev_lens.unsqueeze(1)
    ) & prev_is_prefix  # (N, K', K')
    # gather the extensions that match to_match, multiply with zero if k won't exactly
    # match k', and sum into k'. Then sum those into the relevant non-extension
    # candidates
    nb_nonext_probs_cand = nb_nonext_probs_cand + (
        nb_ext_probs_cand.gather(2, to_match) * ext_is_exact
    ).sum(1)
    # clear the probabilities of extensions k->v that exactly matched some k' for v
    has_match = (
        torch.nn.functional.one_hot(to_match, V).bool() & ext_is_exact.unsqueeze(2)
    ).any(2)
    nb_ext_probs_cand = nb_ext_probs_cand.masked_fill(has_match, -float("inf"))
    del has_match, ext_is_exact

    # we can finally determine the top k paths. Put the non-extending candidates after
    # the extending candidates (the last K' elements of the second dimension)
    tot_probs_cand = torch.cat(
        [nb_ext_probs_cand.view(N, Kp * V), nb_nonext_probs_cand + b_nonext_probs_cand],
        1,
    )  # (N, K' * (V + 1))
    tot_probs_next, next_ind = tot_probs_cand.topk(width, 1, sorted=needs_sorted)
    del tot_probs_cand, tot_probs_next

    next_is_nonext = next_ind >= (Kp * V)
    next_src = (next_ind - (Kp * V)) * next_is_nonext + (next_ind // V) * (
        ~next_is_nonext
    )
    next_ext = next_ind % V

    y_next_prefix_lens = y_prev_lens.gather(1, next_src)  # (N, K)
    y_next = torch.cat(
        [
            y_prev.gather(1, next_src.unsqueeze(2).expand(N, width, tm1)),
            torch.empty((N, width, 1), device=y_prev.device, dtype=y_prev.dtype),
        ],
        2,
    ).scatter(
        2, y_next_prefix_lens.unsqueeze(2), next_ext.unsqueeze(2)
    )  # (N, K, t)
    y_next_lens = y_next_prefix_lens + (~next_is_nonext)
    del y_next_prefix_lens

    nb_ext_probs_next = nb_ext_probs_cand.view(N, Kp * V).gather(
        1, next_ind.clamp(max=Kp * V - 1)
    )  # (N, K)
    nb_nonext_probs_next = nb_nonext_probs_cand.gather(1, next_src)  # (N, K)
    nb_probs_next = (
        nb_ext_probs_next * (~next_is_nonext) + nb_nonext_probs_next * next_is_nonext
    )
    del nb_ext_probs_next, nb_nonext_probs_next, nb_nonext_probs_cand, nb_ext_probs_cand

    b_probs_next = b_nonext_probs_cand.gather(1, next_src) * next_is_nonext  # (N, K)
    del b_nonext_probs_cand

    y_next_last = y_prev_last.gather(1, next_src) * next_is_nonext + next_ext * (
        ~next_is_nonext
    )
    del y_prev_last

    next_prefix_is_prefix = prev_is_prefix.gather(
        1, next_src.unsqueeze(2).expand(N, width, Kp)
    ).gather(2, next_src.unsqueeze(1).expand(N, width, width))
    next_len_leq = y_next_lens.unsqueeze(2) <= y_next_lens.unsqueeze(1)
    next_to_match = y_next.gather(
        2, (y_next_lens - 1).clamp(min=0).unsqueeze(1).expand(N, width, width)
    ).transpose(1, 2)
    next_ext_matches = next_to_match == next_ext.unsqueeze(2)
    next_is_prefix = (
        next_prefix_is_prefix
        & next_len_leq
        & (
            next_is_nonext.unsqueeze(2)
            | (~next_is_nonext.unsqueeze(2) & next_ext_matches)
        )
    )
    del (
        next_prefix_is_prefix,
        next_len_leq,
        next_to_match,
        next_ext_matches,
        next_ext,
        next_ind,
    )

    return (
        y_next,  # (N, K, t)
        y_next_last,  # (N, K)
        y_next_lens,  # (N, K)
        (nb_probs_next, b_probs_next),  # (N, K), (N, K)
        next_is_prefix,  # (N, K, K)
        next_src,  # (N, K)
        next_is_nonext,  # (N, K)
    )


def time_distributed_return(
    r: torch.Tensor, gamma: float, batch_first: bool = False
) -> torch.Tensor:
    r"""Accumulate future local rewards at every time step

    In `reinforcement learning
    <https://en.wikipedia.org/wiki/Reinforcement_learning>`__, the return is defined as
    the sum of discounted future rewards. This function calculates the return for a
    given time step :math:`t` as

    .. math::

        R_t = \sum_{t'=t} \gamma^(t' - t) r_{t'}

    Where :math:`r_{t'}` gives the (local) reward at time :math:`t'` and
    :math:`\gamma` is the discount factor. :math:`\gamma \in [0, 1)` implies
    convergence, but this is not enforced here

    Parameters
    ----------
    r : torch.Tensor
        A two-dimensional float tensor of shape ``(steps, batch_size)`` (or
        ``(batch_size, steps)`` if `batch_first` is :obj:`True`) of local rewards. The
        :math:`t` dimension is the step dimension
    gamma : float
        The discount factor
    batch_first : bool, optional

    Returns
    -------
    `R` : torch.Tensor
        Of the same shape as `r`

    See Also
    --------
    :ref:`Gradient Estimators`
        Provides an example of reinforcement learning that uses this function
    """
    if r.dim() != 2:
        raise RuntimeError("r must be 2 dimensional")
    if not gamma:
        return r
    if batch_first:
        exp = torch.arange(r.shape[-1], device=r.device, dtype=r.dtype)
        discount = torch.pow(gamma, exp)
        discount = (discount.unsqueeze(1) / discount.unsqueeze(0)).tril()
        R = torch.matmul(r, discount)
    else:
        exp = torch.arange(r.shape[0], device=r.device, dtype=r.dtype)
        discount = torch.pow(gamma, exp)
        discount = (discount.unsqueeze(0) / discount.unsqueeze(1)).triu()
        R = torch.matmul(discount, r)
    return R


def error_rate(
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int] = None,
    include_eos: bool = False,
    norm: bool = True,
    batch_first: bool = False,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
    sub_cost: float = 1.0,
    warn: bool = True,
) -> torch.Tensor:
    """Calculate error rates over a batch of references and hypotheses

    An error rate is the total number of insertions, deletions, and substitutions
    between a reference (gold-standard) and hypothesis (generated) transcription,
    normalized by the number of elements in a reference. Consult the Wikipedia article
    on the `Levenshtein distance <https://en.wikipedia.org/wiki/Levenshtein_distance>`__
    for more information.

    Given a reference (gold-standard) transcript long tensor `ref` of size
    ``(max_ref_steps, batch_size)`` if `batch_first` is :obj:`False` or ``(batch_size,
    max_ref_steps)`` otherwise, and a long tensor `hyp` of shape ``(max_hyp_steps,
    batch_size)`` or ``(batch_size, max_hyp_steps)``, this function produces a tensor
    `er` of shape ``(batch_size,)`` storing the associated error rates.

    `er` will not have a gradient, and is thus not directly suited to being a loss
    function.

    Parameters
    ----------
    ref : torch.Tensor
    hyp : torch.Tensor
    eos : int or None, optional
        A special token in `ref` and `hyp` whose first occurrence in each batch
        indicates the end of a transcript. This allows for variable-length transcripts
        in the batch
    include_eos : bool, optional
        Whether to include the first instance of `eos` found in both `ref` and `hyp` as
        valid tokens to be computed as part of the rate. This is useful when gauging
        if a model is learning to emit the `eos` properly, but is not usually included
        in an evaluation. Only the first `eos` per transcript is included
    norm : bool, optional
        If :obj:`False`, will return the number of mistakes (rather than the number
        of mistakes over the total number of referene tokens)
    batch_first : bool, optional
    ins_cost : float, optional
        The cost of an adding a superfluous token to a transcript in `hyp`
    del_cost : float, optional
        The cost of missing a token from `ref`
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`
    warn : bool, optional
        Whether to display warnings on irregularities. Currently, this can happen in
        three ways.

        1. If :obj:`True` and `ins_cost`, `del_cost`, or `sub_cost` is not 1, a warning
           about a difference in computations will be raised. See the below warning for
           more info.
        2. If :obj:`True` and `norm` is :obj:`True`, will warn when a reference
           transcription has zero length
        3. If `eos` is set and `include_eos` is :obj:`True`, will warn when a transcript
           does not include an `eos` symbol

    Returns
    -------
    er : torch.Tensor
        The error rates in `er` will always be floating-point, regardless of whether
        they are normalized or not

    Warnings
    --------
    Up to and including `v0.3.0`, `error_rate` computed a normalized `Edit distance
    <https://en.wikipedia.org/wiki/Edit_distance>`__ instead of an error rate. The
    latter can be considered the total weighted cost of insertions, deletions, and
    substitutions (as per `ins_cost`, `del_cost`, and `sub_cost`), whereas the former is
    the sum of the number of mistakes. The old behaviour of returning the cost is now in
    :func:`edit_distance` (though `norm` is :obj:`False` by default). For speech
    recognition evaluation, `error_rate` is the function to use. However, if you are
    using the default costs, ``ins_cost == del_cost == sub_cost == 1``, there should be
    no numerical difference between the two.

    While `error_rate` does not report the total cost, `ins_cost`, `del_cost`, and
    `sub_cost` impact how references are aligned to hypotheses. For example, setting
    ``sub_cost == 0`` will not remove the count of substitutions from the error rate.
    In fact, it's more likely to do the opposite: since the cost for a substitution is
    lower, the underlying algorithm is more likely to align with substitutions,
    increasing the contribution of substitutions to the error rate.
    """
    er = _string_matching(
        ref,
        hyp,
        eos,
        include_eos,
        batch_first,
        ins_cost,
        del_cost,
        sub_cost,
        warn,
        norm=norm,
        return_mistakes=True,
    )
    return er


def edit_distance(
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int] = None,
    include_eos: bool = False,
    norm: bool = False,
    batch_first: bool = False,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
    sub_cost: float = 1.0,
    warn: bool = True,
) -> torch.Tensor:
    """Compute an edit distance over a batch of references and hypotheses

    An `Edit Distance <https://en.wikipedia.org/wiki/Edit_distance>`__ quantifies
    how dissimilar two token sequences are as the total cost of transforming a
    reference sequence into a hypothesis sequence. There are three operations that can
    be performed, each with an associated cost: adding an extra token to the reference,
    removing a token from the reference, or swapping a token in the reference with a
    token in the hypothesis.

    Given a reference (gold-standard) transcript long tensor `ref` of size
    ``(max_ref_steps, batch_size)`` if `batch_first` is :obj:`False` or ``(batch_size,
    max_ref_steps)`` otherwise, and a long tensor `hyp` of shape ``(max_hyp_steps,
    batch_size)`` or ``(batch_size, max_hyp_steps)``, this function produces a tensor
    `er` of shape ``(batch_size,)`` storing the associated edit distances.

    Parameters
    ----------
    ref : torch.Tensor
    hyp : torch.Tensor
    eos : int or None, optional
        A special token in `ref` and `hyp` whose first occurrence in each batch
        indicates the end of a transcript. This allows for variable-length transcripts
        in the batch
    include_eos : bool, optional
        Whether to include the first instance of `eos` found in both `ref` and `hyp` as
        valid tokens to be computed as part of the rate. This is useful when gauging
        if a model is learning to emit the `eos` properly, but is not usually included
        in an evaluation. Only the first `eos` per transcript is included
    norm : bool, optional
        If :obj:`True`, will normalize the distance by the number of tokens in the
        reference sequence (making the returned value a divergence)
    batch_first : bool, optional
    ins_cost : float, optional
        The cost of an adding an extra token to a sequence in `ref`
    del_cost : float, optional
        The cost of removing a token from a sequence in `ref`
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`
    warn : bool, optional
        Whether to display warnings on irregularities. Currently, this can happen in
        two ways.

        1. If :obj:`True` and `norm` is :obj:`True`, will warn when a reference
           transcription has zero length
        2. If `eos` is set and `include_eos` is :obj:`True`, will warn when a transcript
           does not include an `eos` symbol

    Returns
    -------
    ed : torch.Tensor
        The error rates in `ed` will always be floating-point, regardless of whether
        they are normalized or not

    Notes
    -----
    This function returns identical values (modulo a bug fix) to :func:`error_rate` up
    to `v0.3.0` (though the default of `norm` has changed to :obj:`False`). For more
    details on the distinction between this function and the new :func:`error_rate`,
    please see that function's documentation.
    """
    ed = _string_matching(
        ref,
        hyp,
        eos,
        include_eos,
        batch_first,
        ins_cost,
        del_cost,
        sub_cost,
        warn,
        norm=norm,
    )
    return ed


def optimal_completion(
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int] = None,
    include_eos: bool = True,
    batch_first: bool = False,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
    sub_cost: float = 1.0,
    padding: int = pydrobert.torch.INDEX_PAD_VALUE,
    exclude_last: bool = False,
    warn: bool = True,
) -> torch.Tensor:
    r"""Return a mask of next tokens of a minimum edit distance prefix

    Given a reference transcript `ref` of shape ``(max_ref_steps, batch_size)`` (or
    ``(batch_size, max_ref_steps)`` if `batch_first` is :obj:`True`) and a hypothesis
    transcript `hyp` of shape ``(max_hyp_steps, batch_size)`` (or ``(batch_size,
    max_hyp_steps)``), this function produces a long tensor `optimals` of shape
    ``(max_hyp_steps + 1, batch_size, max_unique_next)`` (or ``(batch_size,
    max_hyp_steps + 1, max_unique_next)``), where ``max_unique_next <= max_ref_steps``,
    of the unique tokens that could be added to the hypothesis prefix ``hyp[:prefix_len,
    batch]`` such that some remaining suffix concatenated to the prefix would result in
    a minimal edit distance. See below for an example.

    Parameters
    ----------
    ref : torch.Tensor
    hyp : torch.Tensor
    eos : int or None, optional
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
    optimals : torch.Tensor

    Examples
    --------

    Consider the reference text "foot" and the hypothesis text "bot". The below shows
    the matrix used to calculate edit distances between them::

        \ _ f o o t
        _ 0 1 2 3 4
        b 1 1 2 3 4
        o 2 2 1 2 3
        t 3 3 2 2 2

    If ``prefix_len == 0``, then the prefix is "", and "f" (from the suffix "foot") is
    the only subsequent token that would not increase the edit distance from that of the
    prefix (0). If ``prefix_len == 1``, then the prefix is "b". To arrive at the minimum
    edit distance for "b", one either treats "b" as an insertion or a substitution for
    "f", yielding suffixes "foot" and "oot". Thus, the subsequent token could be "f" or
    "o". For the prefix "bo", the minimum edit distance is achieved by first
    substituting "f" for "b", then substituting "o" for "o", resulting in the suffix
    "ot" and the next optimal character "o". Finally, for ``prefix_len == 3`` and prefix
    "bot", there are many operations that can produce the minimum edit distance of 2,
    resulting in one of the suffixes "ot", "t", and "". The latter suffix requires no
    more tokens and so any operation would increase the edit distance. Thus the optimal
    next tokens could be "o" or "t".

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
    pydrobert.torch.layers.HardOptimalCompletionDistillationLoss
        A loss function that uses these optimal completions to train a model
    """
    mask = _string_matching(
        ref,
        hyp,
        eos,
        include_eos,
        batch_first,
        ins_cost,
        del_cost,
        sub_cost,
        warn,
        return_mask=True,
        exclude_last=exclude_last,
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
        targets, padding_value=padding, batch_first=True
    )
    if batch_first:
        targets = targets.view(batch_size, max_hyp_steps_p1, -1)
    else:
        targets = targets.view(max_hyp_steps_p1, batch_size, -1)
    return targets


def prefix_error_rates(
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int] = None,
    include_eos: bool = True,
    norm: bool = True,
    batch_first: bool = False,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
    sub_cost: float = 1.0,
    padding: int = pydrobert.torch.INDEX_PAD_VALUE,
    exclude_last: bool = False,
    warn: bool = True,
) -> torch.Tensor:
    """Compute the error rate between ref and each prefix of hyp

    Given a reference transcript `ref` of shape ``(max_ref_steps, batch_size)`` (or
    ``(batch_size, max_ref_steps)`` if `batch_first` is :obj:`True`) and a hypothesis
    transcript `hyp` of shape ``(max_hyp_steps, batch_size)`` (or ``(batch_size,
    max_hyp_steps)``), this function produces a tensor `prefix_ers` of shape
    ``(max_hyp_steps + 1, batch_size)`` (or ``(batch_size, max_hyp_steps + 1))`` which
    contains the error rates for each prefix of each hypothesis, starting from the empty
    prefix.

    Parameters
    ----------
    ref : torch.Tensor
    hyp : torch.Tensor
    eos : int or None, optional
        A special token in `ref` and `hyp` whose first occurrence in each batch
        indicates the end of a transcript. This allows for variable-length transcripts
        in the batch.
    include_eos : bool, optional
        Whether to include the first instance of `eos` found in both `ref` and `hyp` as
        valid tokens to be computed as part of the distance. Only the first `eos` per
        transcript is included.
    norm : bool, optional
        If :obj:`False`, will return the numbers of mistakes (rather than the numbers
        of mistakes over the total number of referene tokens)
    batch_first : bool, optional
    ins_cost : float, optional
        The cost of an adding a superfluous token to a transcript in `hyp`
    del_cost : float, optional
        The cost of missing a token from `ref`
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`
    padding : int, optional
        The value to right-pad the error rates of unequal-length sequences with in
        `prefix_ers`
    exclude_last : bool, optional
        If true, will exclude the final prefix, consisting of the entire transcript,
        from the returned `dists`. `dists` will be of shape ``(max_hyp_steps,
        batch_size, max_unique_next)``
    warn : bool, optional
        Whether to display warnings on irregularities. Currently, this can happen in
        three ways.

        1. If :obj:`True` and `ins_cost`, `del_cost`, or `sub_cost` is not 1, a warning
           about a difference in computations will be raised. See the below warning for
           more info.
        2. If :obj:`True` and `norm` is :obj:`True`, will warn when a reference
           transcription has zero length
        3. If `eos` is set and `include_eos` is :obj:`True`, will warn when a transcript
           does not include an `eos` symbol

    Returns
    -------
    prefix_ers : torch.Tensor

    See Also
    --------
    :ref:`Gradient Estimators`
        Provides an example where this function is used to determine a reward
        function for reinforcement learning

    Warnings
    --------
    The values returned by this function changed after `v0.3.0`. The old behaviour
    can be found in :func:`prefix_edit_distances` (though with `norm` defaulting to
    :obj:`False`). Consult the warning in :func:`error_rate` for more info.
    """
    prefix_ers = _string_matching(
        ref,
        hyp,
        eos,
        include_eos,
        batch_first,
        ins_cost,
        del_cost,
        sub_cost,
        warn,
        norm=norm,
        return_prf_dsts=True,
        exclude_last=exclude_last,
        padding=padding,
        return_mistakes=True,
    )
    return prefix_ers


def prefix_edit_distances(
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int] = None,
    include_eos: bool = True,
    norm: bool = False,
    batch_first: bool = False,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
    sub_cost: float = 1.0,
    padding: int = pydrobert.torch.INDEX_PAD_VALUE,
    exclude_last: bool = False,
    warn: bool = True,
) -> torch.Tensor:
    """Compute the edit distance between ref and each prefix of hyp

    Given a reference transcript `ref` of shape ``(max_ref_steps, batch_size)`` (or
    ``(batch_size, max_ref_steps)`` if `batch_first` is :obj:`True`) and a hypothesis
    transcript `hyp` of shape ``(max_hyp_steps, batch_size)`` (or ``(batch_size,
    max_hyp_steps)``), this function produces a tensor `prefix_eds` of shape
    ``(max_hyp_steps + 1, batch_size)`` (or ``(batch_size, max_hyp_steps + 1))`` which
    contains the edit distance between the reference for each prefix of each hypothesis,
    starting from the empty prefix.

    Parameters
    ----------
    ref : torch.Tensor
    hyp : torch.Tensor
    eos : int or None, optional
        A special token in `ref` and `hyp` whose first occurrence in each batch
        indicates the end of a transcript. This allows for variable-length transcripts
        in the batch.
    include_eos : bool, optional
        Whether to include the first instance of `eos` found in both `ref` and `hyp` as
        valid tokens to be computed as part of the distance. Only the first `eos` per
        transcript is included.
    norm : bool, optional
        If :obj:`True`, will normalize the distances by the number of tokens in the
        reference sequence (making the returned values divergences)
    batch_first : bool, optional
    ins_cost : float, optional
        The cost of an adding an extra token to a sequence in `ref`
    del_cost : float, optional
        The cost of removing a token from a sequence in `ref`
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`
    padding : int, optional
        The value to right-pad the edit distance of unequal-length sequences with in
        `prefix_eds`
    exclude_last : bool, optional
        If true, will exclude the final prefix, consisting of the entire transcript,
        from the returned `dists`. `dists` will be of shape ``(max_hyp_steps,
        batch_size, max_unique_next)``
    warn : bool, optional
        Whether to display warnings on irregularities. Currently, this can happen in
        two ways.

        1. If :obj:`True` and `norm` is :obj:`True`, will warn when a reference
           transcription has zero length
        2. If `eos` is set and `include_eos` is :obj:`True`, will warn when a transcript
           does not include an `eos` symbol

    Returns
    -------
    prefix_eds : torch.Tensor

    Notes
    -----
    This function returns identical values (modulo a bug fix) to
    :func:`prefix_error_rates` up to `v0.3.0` (though the default of `norm` has changed
    to :obj:`False`). For more details on the distinction between this function and the
    new :func:`prefix_error_rates`, please consult the documentation of
    :func:`error_rate`.
    """
    prefix_eds = _string_matching(
        ref,
        hyp,
        eos,
        include_eos,
        batch_first,
        ins_cost,
        del_cost,
        sub_cost,
        warn,
        norm=norm,
        return_prf_dsts=True,
        exclude_last=exclude_last,
        padding=padding,
        return_mistakes=False,
    )
    return prefix_eds


def random_walk_advance(
    logits_t: torch.Tensor,
    num_samp: int,
    y_prev: Optional[torch.Tensor] = None,
    eos: int = pydrobert.torch.INDEX_PAD_VALUE,
    lens: Optional[torch.Tensor] = None,
    prevent_eos: bool = False,
    include_relaxation: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Advance a random walk of sequences

    Suppose a model outputs a un-normalized log-probability distribution over the next
    element of a sequence in `logits_t` s.t.

    .. math::

        Pr(y_t = c) = exp(logits_{t,c}) / \sum_k exp(logits_k)

    We assume :math:`logits_t` is a function of what comes before :math:`logits_t =
    f(logits_{<t}, y_{<t})`. Alternatively, letting :math:`s_t = (logits_t, y_t)`,
    :math:`s` is a Markov Chain. A model is auto-regressive if :math:`f` depends on
    :math:`y_{<t}`, and is not auto-regressive if :math:`logits_t = f(logits_{<t})`.

    A random walk can be performed over a Markov Chain by sampling the elements
    :math:`y_t` of the greater sequence `y` one at a time, according to :math:`Pr(y_t =
    c)`. This allows us to sample the distribution :math:`Pr(Y)`.

    This function is called at every time step. It updates the sequences being built
    (`y_prev`) with one additional token and returns `y`. This function is intended to
    be coupled with an auto-regressive model, where `logits_t` is not known until
    :math:`y_t` is known. If the model is not auto-regressive, it is much more efficient
    to gather all `logits_t` into one :math:`logits` and sample all at once. See the
    examples section below for both behaviours

    Parameters
    ----------
    logits_t : torch.Tensor
        The conditional probabilities over class labels for the current time step.
        Either of shape ``(batch_size, old_samp, num_classes)``, where ``old_samp`` is
        the number of samples in the previous time step, or ``(batch_size,
        num_classes)``, where it is assumed that ``old_samp == 1``
    num_samp : int
        The number of samples to be drawn. Either ``old_samp == 1`` and/or ``num_samp <=
        old_samp`` must be :obj:`True`. That is, either all samples will share the same
        prefix, or we are building off a subset of the samples from ``y_prev`` (in this
        case, always the first `num_samp`)
    y_prev : torch.Tensor, optional
        A long tensor of shape ``(t - 1, batch_size, old_samp)`` or ``(t - 1,
        batch_size)`` specifying :math:`y_{<t}`. If unspecified, it is assumed that
        ``t == 1``
    eos : int, optional
        A special end-of-sequence symbol indicating that the beam has ended. Can be a
        class index. If this value occurs in in ``y_prev[-1, bt, smp]`` for some batch
        ``bt`` and sample ``smp``, `eos` will be appended to ``y_prev[:, bt, smp]``
    lens : torch.Tensor, optional
        A long tensor of shape ``(batch_size,)``. If ``t > lens[bt]`` for some batch
        ``bt``, all samples for ``bt`` will be considered finished. `eos` will be
        appended to `y_prev`
    prevent_eos : bool, optional
        Setting this flag to :obj:`True` will keep `eos` targets from being drawn unless
        a sample has finished (either with a prior `eos` or through `lens`). Note that
        this will only have an effect when ``0 <= eos <= num_classes``
    include_relaxation : bool, optional
        If :obj:`True`, a tuple will be returned whose second element is `z`, see below

    Returns
    -------
    y : torch.Tensor
        A long tensor of shape ``(t, batch_size, num_samp)`` of the sampled
        sequences so far. Note that, since :math:`y_t` are drawn `i.i.d.`,
        there is no guarantee of the uniqueness of each `num_samp` samples
    z : torch.Tensor
        Only included if `include_relaxation` is :obj:`True`. `z` is a sample
        of a continuous relaxation of the categorical distribution of `logits`
        of shape ``(batch_size, num_samp, num_classes). Assuming ``y_prev[-1,
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

    Notes
    -----

    Unlike in the beam search, `logits_t` must be transformed into a probability
    distribution. Otherwise, we would not be able to sample the next step

    See Also
    --------
    :ref:`Gradient Estimators`
        Includes a use case for `include_relaxation`
    """
    if logits_t.dim() == 2:
        logits_t = logits_t.unsqueeze(1)
    elif logits_t.dim() != 3:
        raise RuntimeError("logits_t must have dimension of either 2 or 3")
    batch_size, old_samp, num_classes = logits_t.shape
    if prevent_eos and 0 <= eos < num_classes:
        logits_t[..., eos] = torch.tensor(-float("inf"), device=logits_t.device)
    if old_samp != 1 and num_samp > old_samp:
        raise RuntimeError("either old_samp == 1 or num_samp <= old_samp must be true")
    eos_mask = None
    if y_prev is not None:
        if y_prev.dim() == 2:
            y_prev = y_prev.unsqueeze(2)
        if y_prev.shape[1:] != logits_t.shape[:-1]:
            raise RuntimeError(
                "If logits_t of shape {} then y_prev must have shape "
                "(*, {}, {})".format(
                    (batch_size, old_samp, num_classes),
                    batch_size,
                    old_samp,
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
            raise RuntimeError("lens must be of shape ({},)".format(batch_size))
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
        z = z.masked_fill(eos_mask.unsqueeze(-1), -float("inf"))
    y = y.unsqueeze(0)
    if y_prev is not None:
        y = torch.cat([y_prev, y], 0)
    if include_relaxation:
        return y, z
    else:
        return y


def sequence_log_probs(
    logits: torch.Tensor, hyp: torch.Tensor, dim: int = 0, eos: Optional[int] = None
) -> torch.Tensor:
    r"""Calculate joint log probability of sequences

    `logits` is a tensor of shape ``(..., steps, ..., num_classes)`` where ``steps``
    enumerates the time/step `dim` -th dimension. `hyp` is a long tensor of shape
    ``(..., steps, ...)`` matching the shape of `logits` minus the last dimension.
    Letting :math:`t` index the step dimension and :math:`b` index all other shared
    dimensions of `logits` and `hyp`, this function outputs a tensor `log_probs` of the
    log-joint probability of sequences in the batch:

    .. math::

        \log Pr(samp_b = hyp_b) = \log \left(
            \prod_t Pr(samp_{b,t} == hyp_{b,t}; logits_{b,t})\right)

    :math:`logits_{b,t}` (with the last dimension free) characterizes a categorical
    distribution over ``num_classes`` tokens via a softmax function. We assume
    :math:`samp_{b,t}` is independent of :math:`samp_{b',t'}` given :math:`logits_t`.

    The resulting tensor `log_probs` is matches the shape of `logits` or
    `hyp` without the ``step`` and ``num_classes`` dimensions.

    This function can handle variable-length sequences in three ways:

    1. Any values of `hyp` not in ``[0, num_classes)`` will be considered
       padding and ignored.
    2. If `eos` (end-of-sentence) is set, the first occurrence at :math:`b,t`
       is included in the sequence, but all :math:`b,>t` are ignored. This is
       in addition to 1.
    3. `logits` and `hyp` may be :class:`torch.nn.utils.rnn.PackedSequence`
       objects. In this case, the packed sequence dimension is assumed to
       index ``steps`` (`dim` is ignored). The remaining batch dimension will
       always be stacked into dimension 0. This is also in addition to 1.

    Parameters
    ----------
    logits : torch.Tensor or torch.nn.utils.rnn.PackedSequence
    hyp : torch.Tensor or torch.nn.utils.rnn.PackedSequence
    dim : int, optional
    eos : int or :obj:`None`, optional

    Returns
    -------
    log_prob : torch.Tensor

    Notes
    -----
    `dim` is relative to ``hyp.shape``, not ``logits.shape``

    See Also
    --------
    pydrobert.torch.layers.MinimumErrorRateLoss
        An example training regime that uses this function
    """
    if isinstance(logits, torch.nn.utils.rnn.PackedSequence):
        return _sequence_log_probs_packed(logits, hyp)
    if isinstance(hyp, torch.nn.utils.rnn.PackedSequence):
        raise RuntimeError("both hyp and logits must be packed sequences, or neither")
    if logits.shape[:-1] != hyp.shape:
        raise RuntimeError(
            "logits and hyp must have same shape (minus last dimension of " "logits)"
        )
    hyp_dim = hyp.dim()
    if dim < -hyp_dim or dim > hyp_dim - 1:
        raise RuntimeError(
            "Dimension out of range (expected to be in range of [{}, {}], but "
            "got {})".format(-hyp_dim, hyp_dim - 1, dim)
        )
    dim = (hyp_dim + dim) % hyp_dim
    steps = hyp.shape[dim]
    num_classes = logits.shape[-1]
    logits = torch.nn.functional.log_softmax(logits, -1)
    mask = hyp.lt(0) | hyp.ge(num_classes)
    if eos is not None:
        hyp_lens = _lens_from_eos(hyp, eos, dim) + 1
        if dim:
            hyp_lens = hyp_lens.unsqueeze(dim)
            if dim == hyp_dim - 1:
                hyp_lens = hyp_lens.unsqueeze(-1)
            else:
                hyp_lens = hyp_lens.flatten(dim + 1)
        else:
            hyp_lens = hyp_lens.view(1, -1)
        len_mask = torch.arange(steps, device=logits.device).unsqueeze(-1)
        len_mask = len_mask >= hyp_lens
        len_mask = len_mask.view_as(mask)
        mask = mask | len_mask
    hyp = hyp.masked_fill(mask, 0)
    logits = logits.gather(-1, hyp.unsqueeze(-1)).squeeze(-1)
    logits = logits.masked_fill(mask, 0.0)
    return logits.sum(dim)


def polyharmonic_spline(
    train_points: torch.Tensor,
    train_values: torch.Tensor,
    query_points: torch.Tensor,
    order: int,
    regularization_weight: float = 0.0,
    full_matrix: bool = True,
) -> torch.Tensor:
    """Guess values at query points using a learned polyharmonic spline

    A spline estimates a function ``f : points -> values`` from a fixed number of
    training points/knots and the values of ``f`` at those points. It does that by
    solving a series of piecewise linear equations between knots such that the values at
    the knots match the given values (and some additional constraints depending on the
    spline).

    This function based on the `interpolate_spline
    <https://www.tensorflow.org/addons/api_docs/python/tfa/image/interpolate_spline>`__
    function from Tensorflow, which implements a `Polyharmonic Spline
    <https://en.wikipedia.org/wiki/Polyharmonic_spline>`__. For technical details,
    consult the TF documentation.

    Parameters
    ----------
    train_points : torch.Tensor
        A tensor of shape ``(N, T, I)`` representing the training points/knots for
        ``N`` different functions. ``N`` is the batch dimension, ``T`` is the number
        of training points, and ``I`` is the size of the vector input to ``f``. Cast to
        float
    train_values : torch.Tensor
        A float tensor of shape ``(N, T, O)`` of ``f`` evaluated on `train_points`.
        ``O`` is the size of the output vector of ``f``.
    query_points : torch.Tensor
        A tensor of shape ``(N, Q, I)`` representing the points you wish to have
        estimates for. ``Q`` is the number of such points. Cast to float
    order : int
        Order of the spline (> 0). 1 = linear. 2 = thin plate spline.
    regularization_weight : float, optional
        Weight placed on the regularization term. See TF for more info.
    full_matrix : bool, optional
        Whether to solve linear equations via a full concatenated matrix or a block
        decomposition. Setting to :obj:`True` better matches TF and appears to slightly
        improve numerical accuracy at the cost of twice the run time and more memory
        usage.

    Throws
    ------
    RuntimeError
        This function can return a :class`RuntimeError` when no unique spline can be
        estimated. In general, the spline will require at least ``I+1`` non-degenerate
        points (linearly independent). See the Wikipedia entry on splnes for more info.

    Returns
    -------
    query_values : torch.Tensor
        A tensor of shape ``(N, Q, O)`` of the values estimated by the spline
    """
    train_points = train_points.float()
    query_points = query_points.float()

    w, v = _solve_interpolation(
        train_points, train_values, order, regularization_weight, full_matrix
    )

    query_values = _apply_interpolation(w, v, train_points, query_points, order)
    return query_values


def dense_image_warp(
    image: torch.Tensor,
    flow: torch.Tensor,
    indexing: str = "hw",
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """Warp an input image with per-pixel flow vectors

    Given an `image` and a `flow` field, generates a new image `warped` such that

    ::
        warped[n, c, h, w] = image[n, c, h - flow[n, h, w, 0], w - flow[n, h, w, 1]]

    If the reference indices ``h - ...`` and ``w - ...`` are not integers, the value is
    interpolated from the neighboring pixel values.

    This reproduces the functionality of Tensorflow's `dense_image_warp
    <https://www.tensorflow.org/addons/api_docs/python/tfa/image/dense_image_warp>`__,
    except `image` is in ``NCHW`` order instead of ``NHWC`` order. It wraps
    `torch.nn.functional.grid_sample`.

    Warning
    -------
    `flow` is not an optical flow. Please consult the TF documentation for more details.

    Parameters
    ----------
    image : torch.Tensor
        A float tensor of shape ``(N, C, H, W)``, where ``N`` is the batch dimension,
        ``C`` is the channel dimension, ``H`` is the height dimension, and ``W`` is the
        width dimension.
    flow : torch.Tensor
        A float tensor of shape ``(N, H, W, 2)``.
    indexing : {'hw', 'wh'}, optional
        If `indexing` is ``"hw"``, ``flow[..., 0] = h``, the height index, and
        ``flow[..., 1] = w`` is the width index. If ``"wh"``, ``flow[..., 0] = w``
        and ``flow[..., 1] = h``. The default in TF is ``"hw"``, whereas torch's
        `grid_sample` is ``"wh"``
    mode : {'bilinear', 'nearest'}
        The method of interpolation. Either use bilinear interpolation or the nearest
        pixel value. The TF default is ``"bilinear"``
    padding_mode : {"border", "zeros", "reflection"}
        Controls how points outside of the image boundaries are interpreted.
        ``"border"``: copy points at around the border of the image. ``"zero"``:
        use zero-valued pixels. ``"reflection"``: reflect pixels into the image starting
        from the boundaries.

    Returns
    -------
    warped : torch.FloatTensor
        The warped image of shape ``(N, C, H, W)``.
    """

    # from tfa.image.dense_image_warp
    # output[n, c, h, w] = image[n, c, h - flow[n, h, w, 0], w - flow[n, h, w, 1]]
    # outside of image uses border

    # from torch.nn.functional.grid_sample
    # output[n, c, h, w] = image[n, c, h, f(grid[n, h, w, 1], H),
    # f(grid[n, h, w, 0], W)]
    # where
    # f(x, X) = ((x + 1) * X - 1) / 2
    # therefore
    # output[n, c, h, w] = image[n, c, ((grid[n, h, w, 1] + 1) * H - 1) / 2,
    #                                  ((grid[n, h, w, 0] + 1) * W - 1) / 2]
    #
    # ((grid[n, h, w, 1] + 1) * H - 1) / 2 = h - flow[n, h, w, 0]
    # grid[n, h, w, 1] = (2 * h - 2 * flow[n, h, w, 0] + 1) / H - 1
    # likewise
    # grid[n, h, w, 0] = (2 * w - 2 * flow[n, h, w, 1] + 1) / W - 1

    flow = flow.float()

    N, C, H, W = image.shape
    h = torch.arange(H, dtype=image.dtype, device=image.device)  # (H,)
    w = torch.arange(W, dtype=image.dtype, device=image.device)  # (W,)
    h, w = torch.meshgrid(h, w)  # (H, W), (H, W)
    if indexing == "hw":
        # grid_sample uses wh sampling, so we flip both the flow and hw along final axis
        hw = torch.stack((w, h), 2).unsqueeze(0)  # (1, H, W, 2)
        flow = flow.flip(-1)
    elif indexing == "wh":
        hw = torch.stack((w, h), 2).unsqueeze(0)  # (1, H, W, 2)
    else:
        raise ValueError("Invalid indexing! must be one of 'wh' or 'hw'")
    HW = torch.tensor([[[[W, H]]]], dtype=image.dtype, device=image.device)  # (1,1,1,2)
    grid = (2 * hw - 2 * flow + 1.0) / HW - 1.0

    return torch.nn.functional.grid_sample(
        image, grid, mode=mode, padding_mode=padding_mode, align_corners=False
    )


def sparse_image_warp(
    image: torch.Tensor,
    source_points: torch.Tensor,
    dest_points: torch.Tensor,
    indexing: str = "hw",
    field_interpolation_order: int = 2,
    field_regularization_weight: float = 0.0,
    field_full_matrix: bool = True,
    pinned_boundary_points: int = 0,
    dense_interpolation_mode: str = "bilinear",
    dense_padding_mode: str = "border",
    include_flow: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Warp an image by specifying mappings between few control points

    Given a source image `image`, a few source coordinates `source_points` and their
    corresponding positions `dest_points` in the warped image, this function
    interpolates the remainder of the map with a polyharmonic spline and produces a
    warped image `warped`.

    This function mirrors the behaviour of Tensorflow's `sparse_image_warp
    <https://www.tensorflow.org/addons/api_docs/python/tfa/image/sparse_image_warp>`__,
    except `image` is in ``NCHW`` order instead of ``NHWC`` order. For more details,
    please consult their documentation.

    Parameters
    ----------
    image : torch.Tensor
        A float tensor of shape ``(N, C, H, W)``, where ``N`` is the batch dimension,
        ``C`` the channel dimension, ``H`` the image height, and ``W`` the image width.
    source_points : torch.Tensor
        A tensor of shape ``(N, M, 2)``, where ``M`` is the number of control points
        and the final dimension stores the coordinates of the control point in `image`.
        Cast to float.
    dest_points : torch.Tensor
        A tensor of shape ``(N, M, 2)`` such that the point ``source_points[n, m, :]``
        in `image` will be mapped to ``dest_points[n, m, :]`` in `warped`.
        Cast to float.
    indexing : {'hw', 'wh'}, optional
        If `indexing` is ``"hw"``, ``source_points[n, m, 0]`` and
        ``dest_points[n, m, 0]`` index the height dimension in `image` and `warped`,
        respectively, and ``source_points[n, m, 1]`` and ``dest_points[n, m, 1]`` the
        width dimension. If `indexing` is ``"wh"``, the width dimension is the 0-index
        and height the 1.
    field_interpolation_order : int, optional
        The order of the polyharmonic spline used to interpolate the rest of the points
        from the control. See :func:`polyharmonic_spline` for more info.
    field_regularization_weight : int, optional
        The regularization weight of the polyharmonic spline used to interpolate the
        rest of the points from the control. See :func:`polyharmonic_spline` for more
        info.
    field_full_matrix : bool, optional
        Determines the method of calculating the polyharmonic spline used to interpolate
        the rest of the points from the control. See :func:`polyharmonic_spline` for
        more info.
    pinned_boundary_points : int, optional
        Dictates whether and how many points along the boundary of `image` are mapped
        identically to points in `warped`. This keeps the boundary of the `image` from
        being pulled into the interior of `warped`. When :obj:`0`, no points are added.
        When :obj:`1`, four points are added, one in each corner of the image. When
        ``k > 2``, one point in each corner of the image is added, then ``k - 1``
        equidistant points along each of the four edges, totaling ``4 * k`` points.
    dense_interpolation_mode : {'bilinear', 'nearest'}, optional
        The method with which partial indices in the derived mapping are interpolated.
        See :func:`dense_image_warp` for more info.
    dense_padding_mode : {'border', 'zero', 'reflection'}, optional
        What to do when points in the derived mapping fall outside of the boundaries.
        See :func:`dense_image_warp` for more info.
    include_flow : bool, optional
        If :obj:`True`, include the flow field `flow` interpolated from the control
        points in the return value.

    Returns
    -------
    warped[, flow] : torch.Tensor[, torch.Tensor]
        `warped` is a float tensor of shape ``(N, C, H, W)`` containing the warped
        images. If `include_flow` is :obj:`True`, `flow`, a float tensor of shape
        ``(N, H, W, 2)``. ``flow[n, h, w, :]`` is the flow for coordinates ``h, w``
        in whatever order was specified by `indexing`. See :func:`dense_image_warp`
        for more details.
    """

    # all our computations assume "wh" ordering, so we flip it here if necessary.
    # Though unintuitive, we need this for our call to grid_sample
    if indexing == "hw":
        source_points = source_points.flip(-1)
        dest_points = dest_points.flip(-1)

    source_points = source_points.float()
    dest_points = dest_points.float()

    N, C, H, W = image.shape
    WH = torch.tensor([[W, H]] * N, dtype=image.dtype, device=image.device)

    M = source_points.shape[1]
    if not M:
        return image

    if pinned_boundary_points > 0:
        pinned_points = _deterimine_pinned_points(pinned_boundary_points, WH)
        source_points = torch.cat([source_points, pinned_points], 1)  # (N,M',2)
        dest_points = torch.cat([dest_points, pinned_points], 1)  # (N,M+4k=M',2)
        # now just pretend M' was M all along

    H_range = torch.arange(H, dtype=image.dtype, device=image.device)  # (H,)
    W_range = torch.arange(W, dtype=image.dtype, device=image.device)  # (W,)
    h, w = torch.meshgrid(H_range, W_range)  # (H, W), (H, W)
    query_points = torch.stack([w.flatten(), h.flatten()], 1)  # (H * W, 2)

    if include_flow:
        train_points = dest_points
        train_values = dest_points - source_points
        flow = polyharmonic_spline(
            train_points,
            train_values,
            query_points.unsqueeze(0).expand(N, H * W, 2),
            field_interpolation_order,
            regularization_weight=field_regularization_weight,
            full_matrix=field_full_matrix,
        )

        flow = flow.view(N, H, W, 2)

        warped = dense_image_warp(
            image,
            flow,
            indexing="wh",
            mode=dense_interpolation_mode,
            padding_mode=dense_padding_mode,
        )

        if indexing == "hw":
            flow = flow.flip(-1)

        return warped, flow
    else:
        # If we can return just the warped image, we can bypass our call to
        # dense_image_warp by interpolating the 'grid' parameter of 'grid_sample'
        # instead of the 'flow' parameter of 'dense_image_warp'
        # coord = ((grid + 1) * size - 1) / 2
        # grid = (2 coord + 1) / size - 1
        train_points = dest_points  # (N, M, 2)
        train_values = (2.0 * source_points + 1.0) / WH.unsqueeze(1) - 1.0  # (N, M, 2)

        grid = polyharmonic_spline(
            train_points,
            train_values,
            query_points.unsqueeze(0).expand(N, H * W, 2),
            field_interpolation_order,
            regularization_weight=field_regularization_weight,
            full_matrix=field_full_matrix,
        )

        grid = grid.view(N, H, W, 2)

        warped = torch.nn.functional.grid_sample(
            image,
            grid,
            mode=dense_interpolation_mode,
            padding_mode=dense_padding_mode,
            align_corners=False,
        )

        return warped


def _deterimine_pinned_points(k, sizes):

    w_max = (sizes[:, :1] - 1).expand(-1, k + 1)  # (N, k+1)
    h_max = (sizes[:, 1:] - 1).expand(-1, k + 1)  # (N, k+1)
    range_ = torch.linspace(
        0.0, 1.0, k + 1, dtype=sizes.dtype, device=sizes.device
    )  # (k+1,)
    w_range = w_max * range_  # (N, k+1)
    h_range = h_max * range_  # (N, k+1)
    zeros = torch.zeros_like(w_range)  # (N, k+1)

    # (0, 0) -> (W - 1, 0) inclusive
    bottom_edge = torch.stack([w_range, zeros], 2)  # (N, k+1, 2)
    # (0, 0) -> (0, H - 1) exclusive
    left_edge = torch.stack([zeros[:, 1:-1], h_range[:, 1:-1]], 2)  # (N, k-1, 2)
    # (0, H - 1) -> (W - 1, H - 1) inclusive
    top_edge = torch.stack([w_range, h_max], 2)  # (N, k+1, 2)
    # (W - 1, 0) -> (W - 1, H - 1) exclusive
    right_edge = torch.stack([w_max[:, 1:-1], h_range[:, 1:-1]], 2)  # (N, k-1, 2)

    return torch.cat([bottom_edge, left_edge, top_edge, right_edge], 1)  # (N, 4k, 2)


def _solve_interpolation(c, f, k, reg, full):
    # based on
    # https://mathematica.stackexchange.com/questions/65763/understanding-polyharmonic-splines
    # Symbol map (theirs => ours)
    # x,y => c  (N, T, I)
    # A => A    (N, T, T)
    # W => B    (N, T, I+1)
    # v => w    (N, T, O)
    # bb => v   (N, I+1, O)
    # wa => f   (N, T, O)
    r_cc = torch.cdist(c, c)  # (N, T, T)
    A = _phi(r_cc, k)  # (N, T, T)
    if reg > 0.0:
        A = A + torch.eye(A.shape[1], dtype=A.dtype, device=A.device).unsqueeze(0) * reg
    B = torch.cat([c, torch.ones_like(c[..., :1])], 2)  # (N, T, I+1)

    if full:
        # full matrix method (TF)
        ABt = torch.cat([A, B.transpose(1, 2)], 1)  # (N, T+I+1, T)
        zeros = torch.zeros(
            (B.shape[0], B.shape[2], B.shape[2]), device=B.device, dtype=B.dtype
        )
        B0 = torch.cat(
            [B, zeros],
            1,
        )  # (N, T+I+1, I+1)
        ABtB0 = torch.cat([ABt, B0], 2)  # (N, T+I+1, T+I+1)
        zeros = torch.zeros(
            (B.shape[0], B.shape[2], f.shape[2]), device=f.device, dtype=f.dtype
        )
        f0 = torch.cat(
            [f, zeros],
            1,
        )  # (N, T+I+1, O)
        wv, _ = torch.solve(f0, ABtB0)
        w, v = wv[:, : B.shape[1]], wv[:, B.shape[1] :]
    else:
        # block decomposition
        Ainv = torch.inverse(A)  # (N, T, T)
        Ainv_f = torch.bmm(Ainv, f)  # (N, T, O)
        Ainv_B = torch.bmm(Ainv, B)  # (N, T, I+1)
        Bt = B.transpose(1, 2)  # (N, I+1, T)
        Bt_Ainv_B = torch.bmm(Bt, Ainv_B)  # (N, I+1, I+1)
        Bt_Ainv_f = torch.bmm(Bt, Ainv_f)  # (N, I+1, O)
        v, _ = torch.solve(Bt_Ainv_f, Bt_Ainv_B)  # (N, I+1, O)
        Ainv_B_v = torch.bmm(Ainv_B, v)  # (N, T, O)
        w = Ainv_f - Ainv_B_v  # (N, T, O)

    # orthagonality constraints
    # assert torch.allclose(w.sum(1), torch.tensor(0.0, device=w.device)), w.sum()
    # assert torch.allclose(
    #     torch.bmm(w.transpose(1, 2), c), torch.tensor(0.0, device=w.device)
    # ), torch.bmm(w.transpose(1, 2), c).sum()

    return w, v


def _apply_interpolation(w, v, c, x, k):
    r = torch.cdist(x, c)  # (N, Q, T)
    phi_r = _phi(r, k)  # (N, Q, T)
    phi_r_w = torch.bmm(phi_r, w)  # (N, Q, O)
    x1 = torch.cat([x, torch.ones_like(x[..., :1])], 2)  # (N, Q, I+1)
    x1_v = torch.bmm(x1, v)  # (N, Q, O)
    return phi_r_w + x1_v


def _phi(r, k):
    if k % 2:
        return r ** k
    else:
        return (r ** k) * (torch.clamp(r, min=torch.finfo(r.dtype).eps)).log()


def _lens_from_eos(tok, eos, dim):
    # length to first eos (exclusive)
    mask = tok.eq(eos)
    x = torch.cumsum(mask, dim, dtype=torch.long)
    max_, argmax = (x.eq(1) & mask).max(dim)
    return argmax.masked_fill(max_.eq(0), tok.shape[dim])


def _sequence_log_probs_packed(logits, hyp):
    if not isinstance(hyp, torch.nn.utils.rnn.PackedSequence):
        raise RuntimeError("both hyp and logits must be packed sequences, or neither")
    logits, logits_lens = logits.data, logits.batch_sizes
    hyp, hyp_lens = hyp.data, hyp.batch_sizes
    if (hyp_lens != logits_lens).any():
        raise RuntimeError("hyp and logits must have the same sequence lengths")
    if logits.shape[:-1] != hyp.shape:
        raise RuntimeError(
            "logits and hyp must have same shape (minus last dimension of " "logits)"
        )
    num_classes = logits.shape[-1]
    logits = torch.nn.functional.log_softmax(logits, -1)
    not_class_mask = hyp.lt(0) | hyp.ge(num_classes)
    hyp = hyp.masked_fill(not_class_mask, 0)
    logits = logits.gather(-1, hyp.unsqueeze(-1)).squeeze(-1)
    logits = logits.masked_fill(not_class_mask, 0.0)
    logits_split = logits.split(logits_lens.tolist(), 0)
    return torch.stack(tuple(x.sum(0) for x in logits_split), 0)


def _string_matching(
    ref,
    hyp,
    eos,
    include_eos,
    batch_first,
    ins_cost,
    del_cost,
    sub_cost,
    warn,
    norm=False,
    return_mask=False,
    return_prf_dsts=False,
    exclude_last=False,
    padding=None,
    return_mistakes=False,
):
    assert not return_mask or not return_prf_dsts
    assert not exclude_last or (return_mask or return_prf_dsts)
    if ref.dim() != 2 or hyp.dim() != 2:
        raise RuntimeError("ref and hyp must be 2 dimensional")
    if ins_cost == del_cost == sub_cost > 0.0:
        # results are equivalent and faster to return
        ins_cost = del_cost = sub_cost = 1.0
        return_mistakes = False
    elif return_mistakes and warn:
        warnings.warn(
            "The behaviour for non-uniform error rates has changed after v0.3.0. "
            "Please switch to edit_distance functions for old behaviour. Set "
            "warn=False to suppress this warning"
        )
    if batch_first:
        ref = ref.t()
        hyp = hyp.t()
    ref = ref.detach()
    hyp = hyp.detach()
    max_ref_steps, batch_size = ref.shape
    max_hyp_steps, batch_size_ = hyp.shape
    device = ref.device
    if batch_size != batch_size_:
        raise RuntimeError(
            "ref has batch size {}, but hyp has {}".format(batch_size, batch_size_)
        )
    if eos is not None:
        ref_lens = _lens_from_eos(ref, eos, 0)
        hyp_lens = _lens_from_eos(hyp, eos, 0)
        if include_eos:
            ref_eq_mask = ref_lens == max_ref_steps
            ref_lens = ref_lens + 1
            if ref_eq_mask.any():
                if warn:
                    warnings.warn(
                        "include_eos=True, but a transcription in ref did not "
                        "contain the eos symbol ({}). To suppress this "
                        "warning, set warn=False".format(eos)
                    )
                ref_lens = ref_lens - ref_eq_mask.to(ref_lens.dtype)
            hyp_eq_mask = hyp_lens == max_hyp_steps
            hyp_lens = hyp_lens + 1
            if hyp_eq_mask.any():
                if warn:
                    warnings.warn(
                        "include_eos=True, but a transcription in hyp did not "
                        "contain the eos symbol ({}). To suppress this "
                        "warning, set warn=False".format(eos)
                    )
                hyp_lens = hyp_lens - hyp_eq_mask.to(hyp_lens.dtype)
            del ref_eq_mask, hyp_eq_mask
    else:
        ref_lens = torch.full((batch_size,), max_ref_steps, device=ref.device)
        hyp_lens = torch.full((batch_size,), max_hyp_steps, device=ref.device)
    ins_cost = torch.tensor(float(ins_cost), device=device)
    del_cost = torch.tensor(float(del_cost), device=device)
    sub_cost = torch.tensor(float(sub_cost), device=device)
    if return_mask:
        # this dtype business is a workaround for different default mask
        # types < 1.2.0 and > 1.2.0
        mask = torch.empty(
            (max_hyp_steps + (0 if exclude_last else 1), max_ref_steps, batch_size),
            device=device,
            dtype=ins_cost.eq(ins_cost).dtype,
        )
        mask[0, 0] = 1
        mask[0, 1:] = 0
    elif return_prf_dsts:
        assert padding is not None
        prefix_ers = torch.empty(
            (max_hyp_steps + (0 if exclude_last else 1), batch_size),
            device=device,
            dtype=torch.float,
        )
        prefix_ers[0] = ref_lens * (1.0 if return_mistakes else del_cost)
    # direct row down corresponds to insertion
    # direct col right corresponds to a deletion
    #
    # we vectorize as much as we can. Neither substitutions nor insertions require
    # values from the current row to be computed, and since the last row can't be
    # altered, we can easily vectorize there. To vectorize deletions, we use del_matrix.
    # It has entries
    #
    # 0   inf inf inf ...
    # d   0   inf inf ...
    # 2d  d   0   inf ...
    # ...
    #
    # Where "d" is del_cost. When we sum with the intermediate values of the next row
    # "v" (containing the minimum of insertion and subs costs), we get
    #
    # v[0]    inf     inf     inf ...
    # v[0]+d  v[1]    inf     inf ...
    # v[0]+2d v[1]+d  v[2]    inf ...
    # ...
    #
    # And we take the minimum of each row. The dynamic programming algorithm for
    # levenshtein would usually handle deletions as:
    #
    # for i=1..|v|:
    #     v[i] = min(v[i], v[i-1]+d)
    #
    # if we unroll the loop, we get the minimum of the elements of each row of the above
    # matrix
    row = torch.arange(max_ref_steps + 1, device=device, dtype=torch.float)  # (R+1, N)
    if return_mistakes:
        mistakes = row.unsqueeze(1).expand(max_ref_steps + 1, batch_size)
        row = row * del_cost
    else:
        row *= del_cost
        del_mat = row.unsqueeze(1) - row
        del_mat = del_mat + torch.full_like(del_mat, float("inf")).triu(1)
        del_mat = del_mat.unsqueeze(-1)  # (R + 1, R + 1, 1)
    row = row.unsqueeze(1).expand(max_ref_steps + 1, batch_size)
    for hyp_idx in range(1, max_hyp_steps + (0 if exclude_last else 1)):
        not_done = (hyp_idx - (0 if exclude_last else 1)) < hyp_lens
        last_row = row
        ins_mask = (hyp_lens >= hyp_idx).float()  # (N,)
        neq_mask = (ref != hyp[hyp_idx - 1]).float()  # (R + 1, N)
        row = last_row + ins_cost * ins_mask
        sub_row = last_row[:-1] + sub_cost * neq_mask
        if return_mistakes:
            # The kicker is substitutions over insertions or deletions.
            pick_sub = row[1:] >= sub_row
            row[1:] = torch.where(pick_sub, sub_row, row[1:])
            last_mistakes = mistakes
            mistakes = last_mistakes + ins_mask
            msub_row = last_mistakes[:-1] + neq_mask
            mistakes[1:] = torch.where(pick_sub, msub_row, mistakes[1:])
            # FIXME(sdrobert): the min function behaves non-determinically r.n.
            # (regardless of what the 1.7.0 docs say!) so techniques for extracting
            # indices from the min are a wash. If we can get determinism, we can flip
            # the 1 dimension if (del_mat + row) before the min and get the least idx
            # min, which should have the fewest number of deletions.
            for ref_idx in range(1, max_ref_steps + 1):
                del_ = row[ref_idx - 1] + del_cost
                pick_sub = del_ >= row[ref_idx]
                row[ref_idx] = torch.where(pick_sub, row[ref_idx], del_)
                mistakes[ref_idx] = torch.where(
                    pick_sub, mistakes[ref_idx], mistakes[ref_idx - 1] + 1.0
                )
            mistakes = torch.where(not_done, mistakes, last_mistakes)
        else:
            row[1:] = torch.min(row[1:], sub_row)
            row, _ = (del_mat + row).min(1)
        row = torch.where(not_done, row, last_row)
        if return_mask:
            # As proven in the OCD paper, the optimal targets are always the first
            # character of a suffix of the reference transcript that remains to be
            # aligned. The levenshtein operation corresponding to what we do with that
            # target would be a matched substitution (i.e. hyp's next token is the OCD
            # target, resulting in no change in cost from the prefix). Thus, given a
            # levenshtein matrix for one of these OCD targets (which is this matrix,
            # except for the final row), the minimal values on the final row sit on a
            # diagonal from the minimal values of the current row.
            mins = row.min(0, keepdim=True)[0]
            row_mask = row[:-1] == mins
            row_mask &= not_done
            mask[hyp_idx] = row_mask
        elif return_prf_dsts:
            if return_mistakes:
                prefix_ers[hyp_idx] = mistakes.gather(0, ref_lens.unsqueeze(0)).squeeze(
                    0
                )
            else:
                prefix_ers[hyp_idx] = row.gather(0, ref_lens.unsqueeze(0)).squeeze(0)
    if return_mask:
        mask = mask & (
            (
                torch.arange(max_ref_steps, device=device)
                .unsqueeze(1)
                .expand(max_ref_steps, batch_size)
                < ref_lens
            ).unsqueeze(0)
        )
        return mask
    elif return_prf_dsts:
        if norm:
            prefix_ers = prefix_ers / ref_lens.to(row.dtype)
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
                        torch.arange(prefix_ers.size(0), device=device)
                        .gt(0)
                        .to(row.dtype)
                        .unsqueeze(1)
                        .expand_as(prefix_ers)
                    ),
                    prefix_ers,
                )
        prefix_ers = prefix_ers.masked_fill(
            (
                torch.arange(prefix_ers.size(0), device=device)
                .unsqueeze(1)
                .ge(hyp_lens + (0 if exclude_last else 1))
            ),
            padding,
        )
        if batch_first:
            prefix_ers = prefix_ers.t()
        return prefix_ers
    if return_mistakes:
        er = mistakes.gather(0, ref_lens.unsqueeze(0)).squeeze(0)
    else:
        er = row.gather(0, ref_lens.unsqueeze(0)).squeeze(0)
    if norm:
        er = er / ref_lens.to(er.dtype)
        zero_mask = ref_lens.eq(0)
        if zero_mask.any():
            if warn:
                warnings.warn(
                    "ref contains empty transcripts. Error rates for entries "
                    "will be 1 if any insertion and 0 otherwise. To suppress "
                    "this warning, set warn=False"
                )
            er = torch.where(zero_mask, hyp_lens.gt(0).to(er.dtype), er)
    return er
