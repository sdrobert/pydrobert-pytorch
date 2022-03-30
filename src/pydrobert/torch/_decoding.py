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

import math

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union

import torch

from ._lm import (
    ExtractableSequentialLanguageModel,
    MixableSequentialLanguageModel,
    SequentialLanguageModel,
)
from ._compat import script, trunc_divide, jit_isinstance, SpoofPackedSequence
from ._string import _lens_from_eos
from ._wrappers import functional_wrapper


@script
def beam_search_advance(
    log_probs_t: torch.Tensor,
    width: int,
    log_probs_prev: torch.Tensor,
    y_prev: torch.Tensor,
    y_prev_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Beam search step function

    Parameters
    ----------
    log_probs_t
        A tensor of shape ``(N, old_width, V)`` containing the log probabilities of
        extending a given path with a token of a given type in the vocabulary.
    width : int
        The beam width
    log_probs_prev
        A tensor of shape ``(N, old_width)`` containing the log probabilities of
        the paths so far.
    y_prev
        A tensor of shape ``(S, N, old_width)`` containing the path prefixes.
    y_prev_lens
        A tensor of shape ``(N, old_width)`` specifying the lengths of the prefixes. For
        batch element ``n`` and path ``k`` in the beam, only the values
        ``y_prev[:y_prev_lens[n, k], n, k]`` are valid. If unspecified, it is assumed
        ``y_prev_lens[:, :] == S``.

    Returns
    -------
    y_next, y_next_lens, log_probs_next : torch.Tensor
        The ``*next*`` tensors can be interpreted in the same way as their ``*prev*``
        counterparts, but after the step. The ``old_width`` dimension has been replaced
        with `width`. `y_next` is of shape either ``(S, N, width)`` or ``(S + 1, N,
        width)``, depending on whether `y_prev` needed to grow in order to accommodate
        the newest tokens in the path.
    next_src : torch.Tensor
        A long tensor of shape ``(N, width)`` such that the value ``k_old = next_src[n,
        k_new]`` is the index from the previous step (over ``old_width``) that is a
        prefix of the new path at ``k_new`` (i.e. its source).

    Warnings
    --------
    This function has been drastically simplified after v0.3.0. The logic for
    end-of-sequence handling has been punted to the encapsulating search module.

    If there are too few possible extensions to fill the beam, undefined paths will be
    added to the end of the beam with probability :obj:`-float('inf')`. This means that
    an invalid path cannot be distibguished from a 0-probability path. Consider using a
    very negative value as a replacement for ``log 0``, e.g. ``log_probs_t =
    log_probs_t.clamp(min=torch.finfo(torch.float).min / 2)``.

    See Also
    --------
    pydrobert.torch.modules.BeamSearch
        For the full beam search.
    """
    if log_probs_t.dim() != 3:
        raise RuntimeError("log_probs_t must be 3 dimensional")
    N, Kp, V = log_probs_t.shape
    if width < 1:
        raise RuntimeError(f"Expected width to be >= 1, got {width}")
    if log_probs_prev.shape != (N, Kp):
        raise RuntimeError(
            f"Expected log_probs_prev to be of shape {(N, Kp)}, got "
            f"{log_probs_prev.shape}"
        )
    if y_prev.dim() != 3:
        raise RuntimeError("y_prev must be 3 dimensional")
    if y_prev.shape[1:] != (N, Kp):
        raise RuntimeError(
            f"Expected the last two dimensions of y_prev to be {(N, Kp)}, "
            f"got {y_prev.shape[1:]}"
        )
    tm1 = y_prev.size(0)
    if y_prev_lens is not None and y_prev_lens.shape != (N, Kp):
        raise RuntimeError(
            f"Expected y_prev_lens to have shape {(N, Kp)}, got {y_prev_lens.shape}"
        )

    K = min(width, Kp * V)
    cand_log_probs = (log_probs_prev.unsqueeze(2) + log_probs_t).flatten(1)
    log_probs_next, next_ind = cand_log_probs.topk(K, 1)
    next_src = trunc_divide(next_ind, V)
    y_t = (next_ind % V).unsqueeze(0)  # (1, N, K)

    if tm1:
        y_next = y_prev.gather(2, next_src.unsqueeze(0).expand(tm1, N, K))
        if y_prev_lens is None:
            y_next = torch.cat([y_next, y_t], 0)
            y_next_lens = y_t.new_full((N, K), tm1 + 1)
        else:
            # don't make y bigger unless we have to
            if int(y_prev_lens.max().item()) >= tm1:
                y_next = torch.cat([y_next, y_t], 0)
            y_prev_lens_prefix = y_prev_lens.gather(1, next_src)
            y_next = y_next.scatter(0, y_prev_lens_prefix.unsqueeze(0), y_t)
            y_next_lens = y_prev_lens_prefix + 1
    elif y_prev_lens is not None and (y_prev_lens != 0).any():
        raise RuntimeError("Invalid lengths for t=0")
    else:
        y_next = y_t
        y_next_lens = torch.ones((N, K), dtype=y_t.dtype, device=y_t.device)

    if K < width:
        rem = width - K
        y_next = torch.cat([y_next, y_next.new_empty(tm1 + 1, N, rem)], 2)
        log_probs_next = torch.cat(
            [log_probs_next, log_probs_next.new_full((N, rem), -float("inf"))], 1
        )
        zeros = y_next_lens.new_zeros(N, rem)
        y_next_lens = torch.cat([y_next_lens, zeros], 1)
        next_src = torch.cat([next_src, zeros], 1)

    return y_next, y_next_lens, log_probs_next, next_src


class BeamSearch(torch.nn.Module):
    """Perform beam search on the outputs of a SequentialLanguageModel

    Beam search is a heuristic algorithm that keeps track of `width` most promising
    paths in the beam by probability, distributed by the language model `lm`.

    A path continues to be extended until it is either pruned or emits an
    end-of-sequence (`eos`) symbol (if set). The search ends for a batch element when
    its highest probability path ends with an `eos` or all paths end with an `eos`
    (depending on the setting of `finish_all_paths`). The search ends for the entire
    batch either when the search for all batch elements have ended or `max_iters` steps
    has been reached, whichever comes first. It is therefore necessary to set at least
    one of `eos` or `max_iters`.

    Parameters
    ----------
    lm
        The language model responsible for producing distributions over the next token
        type
    width
        The beam width
    eos
        The end of sequence type. If set, must be in-vocabulary (according to
        ``lm.vocab_size``). Either `eos` or `max_iters` must be set.
    max_iters
        The maximum number of tokens to generate in the paths before returning. Either
        `eos` or `max_iters` must be set.
    finish_all_paths
        Applicable only when `eos` is set. If :obj:`True`, waits for all paths in all
        batches' beams to emit an `eos` symbol before stopping. If :obj:`False`, only
        the highest probability path need end with an `eos` before stopping.
    
    Call Parameters
    ---------------
    y_prev : torch.Tensor
        A long tensor of shape ``(S*, N[, old_width])``. In most cases, `y_prev` should
        be an empty tensor of shape ``(0, N[, 1])``, though it can be used start the
        search with different prefixes.
    prev : Dict[str, torch.Tensor], optional
        Whatever state info must be initially passed to the `lm`.
    
    Returns
    -------
    y : torch.Tensor
        A long tensor of shape ``(S, N, width)`` where ``S >= S*`` containing the
        `width` paths per batch element.
    y_lens: torch.Tensor
        A long tensor of shape ``(N, width)`` of the lengths of the corresponding paths
        including the first instance of `eos`, if it exists. For batch element ``n`` and
        path ``k``, only the tokens in ``y[:y_lens[n, k], n, k]`` are valid.
    y_log_probs : torch.Tensor
        A tensor of shape ``(N, width)`` containing the log probabilities of the paths.

    Warnings
    --------
    Return values will always contain `width` prefixes, regardless of whether this is
    possible. The log probabilities of invalid prefixes will be set to
    :obj:`-float("inf")` and will populate the latter indices of the beam. Since this
    cannot be distinguished from a zero-probability path (``log 0 = -inf``), care must
    be taken by the user to avoid confusing them.

    As soon as a batch element reaches its completion condition the search is frozen for
    that batch element, even if the search continues for other batch elements. This is
    in order to produce consistent results across batch sizes.

    Notes
    -----
    While the core operations of beam search - extending existing paths and pruning the
    low scoring ones - are generally constant, the details will vary between
    implementations. This no-frills implementation is best considered a starting point.
    """

    __constants__ = ["width", "eos", "max_iters", "finish_all_paths"]

    width: int
    eos: Optional[int]
    max_iters: Optional[int]
    finish_all_paths: bool

    def __init__(
        self,
        lm: ExtractableSequentialLanguageModel,
        width: int,
        eos: Optional[int] = None,
        max_iters: Optional[int] = None,
        finish_all_paths: bool = False,
    ):
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive")
        if eos is not None:
            if eos < -lm.vocab_size or eos > lm.vocab_size - 1:
                raise ValueError(
                    f"Expected eos to be in the range [{-lm.vocab_size}, "
                    f"{lm.vocab_size - 1}], got {eos}"
                )
            eos = (eos + lm.vocab_size) % lm.vocab_size
        if max_iters is not None and max_iters < 0:
            raise ValueError("max_iters must be non-negative")
        if eos is None and max_iters is None:
            raise ValueError("at least one of eos or max_iters must be set")
        self.lm = lm
        self.width = width
        self.eos = eos
        self.max_iters = max_iters
        self.finish_all_paths = finish_all_paths

    def reset_parameters(self) -> None:
        if hasattr(self.lm, "reset_parameters"):
            self.lm.reset_parameters()

    @torch.jit.export
    def update_log_probs_for_step(
        self,
        log_probs_prev: torch.Tensor,
        log_probs_t: torch.Tensor,
        y_prev: torch.Tensor,
        y_prev_lens: torch.Tensor,
        eos_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update log_probs_prev and log_probs_t for a step of the beam search

        Subclasses may overload this method to modify the log-probabilities of the paths
        in the beam as well as the log-probabilities of the tokens extending each path.

        Parameters
        ----------
        log_probs_prev
            Of shape ``(N, K)`` containing the log probabilities of paths up to the
            current step.
        log_probs_t
            Of shape ``(N, K, V)`` containing the log probabilities of extending each
            path with a token of a given type.
        y_prev
            Of shape ``(S, N, K)`` containing the paths in the beam up to the current
            step.
        y_prev_lens
            Of shape ``(N, K)`` containing the lengths of the paths up to the current
            step (including the first `eos`, if any). For batch element ``n`` and path
            ``k``, only the tokens in the range ``y_prev[:y_prev_lens[n, k], n, k]`` are
            valid.
        eos_mask
            A boolean tensor of shape ``(N, K)`` which is true when a path has already
            ended. Will be all :obj:`False` when `eos` is unset or there is no history.

        Returns
        -------
        tuple of torch.Tensor
            A tuple `(log_probs_prev_new, log_probs_t_new)`. The modified versions of
            the associated arguments.

        Notes
        -----
        Modifications mean that the results will no longer be interpreted as log
        probabilities, but scores.
        """
        return log_probs_prev, log_probs_t

    def _to_width(
        self,
        y_prev: torch.Tensor,
        log_probs_prev: torch.Tensor,
        y_prev_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        S, N, prev_width = y_prev.shape
        if prev_width < self.width:
            # fill with invalid paths
            rem = self.width - prev_width
            log_probs_prev = torch.cat(
                [log_probs_prev, log_probs_prev.new_full((N, rem), -float("inf"))], 1
            )
            y_prev = torch.cat([y_prev, y_prev.new_zeros(S, N, rem)], 2)
            y_prev_lens = torch.cat([y_prev_lens, y_prev_lens.new_zeros(N, rem)], 1)
        elif prev_width > self.width:
            # get the highest probability prefixes of what we've got
            log_probs_prev, src = log_probs_prev.topk(self.width, 1)
            y_prev = y_prev.gather(2, src.unsqueeze(0).expand(S, N, self.width))
            y_prev_lens = y_prev_lens.gather(1, src)
        return y_prev, log_probs_prev, y_prev_lens

    def forward(
        self, y_prev: torch.Tensor, _prev: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _prev is None:
            prev = dict()
        else:
            prev = _prev
        if y_prev.dim() == 2:
            prev_width = 1
        elif y_prev.dim() == 3:
            if not y_prev.size(0):
                raise RuntimeError(
                    "Cannot start with empty prefix when y_prev is 3 dimensional"
                )
            prev_width = y_prev.size(2)
            if prev_width < 1:
                raise RuntimeError("dim 3 in y_prev must be positive")
            y_prev = y_prev.flatten(1)
        else:
            raise RuntimeError("y_prev must be 2 or 3 dimensional")

        device = y_prev.device
        S_prev, N = y_prev.size(0), y_prev.size(1) // prev_width
        prev = self.lm.update_input(prev, y_prev)
        y_prev = y_prev.view(S_prev, N, prev_width)

        if self.eos is not None and S_prev:
            y_prev_lens = (
                -((y_prev == self.eos).cumsum(0).clamp(max=1).sum(0) - 1).clamp(min=0)
                + S_prev
            )

            len_eq_mask = y_prev_lens.unsqueeze(1) == y_prev_lens.unsqueeze(2)  # NKK
            tok_ge_len_mask = (
                torch.arange(S_prev, device=device).view(S_prev, 1, 1) >= y_prev_lens
            )  # SNK
            eq_mask = (
                y_prev.unsqueeze(2) == y_prev.unsqueeze(3)
            ) | tok_ge_len_mask.unsqueeze(
                3
            )  # SNKK
            eq_mask = (
                eq_mask.all(0)
                & len_eq_mask
                & ~torch.eye(prev_width, dtype=torch.bool, device=device)
            )  # NKK
            if eq_mask.any():
                raise RuntimeError(
                    "y_prev was equivalent for the following (batch_idx, path_idx) "
                    f"paths: {torch.nonzero(eq_mask)}"
                )
        else:
            y_prev_lens = torch.full(
                (N, prev_width), S_prev, dtype=torch.long, device=device
            )
        log_probs_prev = torch.full(
            (N, prev_width), -math.log(prev_width), device=device
        )

        if self.max_iters is None:
            max_iters = 1024 * 1024 * 1024 * 1024
        else:
            max_iters = self.max_iters
        for t in range(S_prev, max_iters + S_prev):
            t = torch.tensor(t, device=device)

            if self.eos is not None and t:
                # determine which paths have already finished (and whether we should
                # stop)
                eos_mask = (
                    y_prev.permute(1, 2, 0)
                    .gather(2, (y_prev_lens - 1).clamp(min=0).unsqueeze(2))
                    .squeeze(2)
                    == self.eos
                ) & (y_prev_lens > 0)
                if self.finish_all_paths:
                    done_mask = eos_mask.all(1, keepdim=True)
                else:
                    done_mask = eos_mask[..., :1]
                if done_mask.all():
                    break
            else:
                eos_mask = torch.full(
                    (N, prev_width), 0, device=device, dtype=torch.bool
                )
                done_mask = eos_mask[..., :1]

            # determine extension probabilities
            log_probs_t, in_next = self.lm.calc_idx_log_probs(
                y_prev.flatten(1), prev, t
            )
            log_probs_t = log_probs_t.reshape(N, prev_width, self.lm.vocab_size)

            # update probabilities if the subclass so desires
            log_probs_prev, log_probs_t = self.update_log_probs_for_step(
                log_probs_prev, log_probs_t, y_prev, y_prev_lens, eos_mask
            )

            if self.eos is not None:
                # if a path has finished, we allocate the entire probability mass to the
                # eos token
                log_probs_t = log_probs_t.masked_fill(
                    eos_mask.unsqueeze(2), -float("inf")
                )
                eos_mask_ = eos_mask.unsqueeze(2).repeat(1, 1, self.lm.vocab_size)
                eos_mask_[..., : self.eos] = False
                eos_mask_[..., self.eos + 1 :] = False
                log_probs_t = log_probs_t.masked_fill(eos_mask_, 0.0)

            # extend + prune
            (y_next, y_next_lens, log_probs_next, next_src) = beam_search_advance(
                log_probs_t, self.width, log_probs_prev, y_prev, y_prev_lens
            )

            if self.eos is not None:
                # beam_search_advance always increments the length. Decrement for the
                # paths which had completed before the step
                y_next_lens = y_next_lens - eos_mask.gather(1, next_src).to(y_next_lens)

            # update lm intermediate values
            next_src = (
                torch.arange(
                    0, prev_width * N, prev_width, device=next_src.device
                ).unsqueeze(1)
                + next_src
            )
            prev = self.lm.extract_by_src(in_next, next_src.flatten())

            if self.eos is not None and done_mask.any():
                y_prev, log_probs_prev, y_prev_lens = self._to_width(
                    y_prev, log_probs_prev, y_prev_lens
                )
                y_next[:-1] = torch.where(done_mask.unsqueeze(0), y_prev, y_next[:-1])
                log_probs_next = torch.where(done_mask, log_probs_prev, log_probs_next)
                y_next_lens = torch.where(done_mask, y_prev_lens, y_next_lens)

            y_prev = y_next
            y_prev_lens = y_next_lens
            log_probs_prev = log_probs_next
            prev_width = self.width

        y_prev, log_probs_prev, y_prev_lens = self._to_width(
            y_prev, log_probs_prev, y_prev_lens
        )

        return y_prev, y_prev_lens, log_probs_prev


@functional_wrapper("CTCGreedySearch")
@script
def ctc_greedy_search(
    logits: torch.Tensor,
    in_lens: Optional[torch.Tensor] = None,
    blank_idx: int = -1,
    batch_first: bool = False,
    is_probs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    keep_mask[:, 1:] = keep_mask[:, 1:] & (argmax[:, 1:] != argmax[:, :-1])
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
        # del in_len_mask
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


class CTCGreedySearch(torch.nn.Module):
    """CTC greedy search

    The CTC greedy search picks the path with the highest probability class in `logits`
    for each element in the sequence. The path (log-)probability is the (sum) product of
    the chosen type (log-probabilities). The output sequences are the resulting sequence
    of class labels with blanks and duplicates removed.

    Parameters
    ----------
    blank_idx
        Which index along the class dimension specifices the blank label
    batch_first
        If :obj:`False`, `logits` is of shape ``(T, N, V)`` and `paths` is of shape
        ``(T, N)``.
    is_probs
        If :obj:`True`, `logits` will be considered a normalized probability
        distribution instead of an un-normalized log-probability distribution. The
        return value `max_` will take the product of sequence probabilities instead of
        the sum.
    
    Call Parameters
    ---------------
    logits : torch.Tensor
        A tensor of shape ``(N, T, V)`` where ``T`` is the sequence dimension, ``N``
        is the batch dimension, and ``V`` is the number of classes including the blank
        label. ``logits[n, t, :]`` represent the unnormalized log-probabilities of the
        labels at time ``t`` in batch element ``n``.
    in_len : torch.Tensor, optional
        A long tensor of shape ``(N,)`` providing the lengths of the sequence in the
        batch. For a given batch element ``n``, only the values of `logits` in the slice
        ``logits[n, :in_lens[n]]`` will be considered valid.
    
    Returns
    -------
    max_ : torch.Tensor
        A tensor of shape ``(N,)`` containing the total log-probability of the greedy
        path. 
    paths : torch.Tensor
        A long tensor of shape ``(N, T)`` which stores the reduced greedy paths.
    out_lens : torch.Tensor
        A long tensor of shape ``(N,)`` which specifies the lengths of the greedy paths
        within `paths`: for a given batch element ``n``, the reduced greedy path is the
        sequence in the range ``paths[n, :out_lens[n]]``. The values of `paths` outside
        this range are undefined.
    """

    __constants__ = ["blank_idx", "batch_first", "is_probs"]

    blank_idx: int
    batch_first: bool
    is_probs: bool

    def __init__(
        self, blank_idx: int = -1, batch_first: bool = False, is_probs: bool = False
    ):
        super().__init__()
        self.blank_idx = blank_idx
        self.batch_first = batch_first
        self.is_probs = is_probs

    def extra_repr(self) -> str:
        return ", ".join(f"{x}={getattr(self, x)}" for x in self.__constants__)

    def forward(
        self, logits: torch.Tensor, in_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return ctc_greedy_search(
            logits, in_lens, self.blank_idx, self.batch_first, self.is_probs
        )


@script
def ctc_prefix_search_advance(
    probs_t: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # ((N,K',V), (N,V), (N))
    width: int,  # K
    probs_prev: Tuple[torch.Tensor, torch.Tensor],  # (N,K'), (N,K')
    y_prev: torch.Tensor,  # (t - 1, N, K')
    y_prev_last: torch.Tensor,  # (N,K')
    y_prev_lens: torch.Tensor,  # (N, K')
    prev_is_prefix: torch.Tensor,  # (N, K', K')  # [n, k, k'] iff k prefix of k'
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
    probs_t
        A triple of ``ext_probs_t, nonext_probs_t, blank_probs_t``. `ext_probs_t` is of
        shape ``(N, old_width, V)`` containing the probabilities of extending a prefix
        with a token of each type (resulting in a new token being added to the reduced
        transcription). `nonext_probs_t` has shape ``(N, V)`` and contains the
        probabilities of adding a token that does not extend a given prefix (i.e. when a
        token immediately follows another of the same type with no blanks in between).
        `blank_probs_t` is of shape ``(N)`` and contains the blank label probabilities.
    width
        The beam width.
    probs_prev
        A pair of ``nb_probs_prev, b_probs_prev``. Each is a tensor of shape ``(N,
        old_width)``. `nb_probs_prev` contains the summed mass of the paths reducing to
        the given prefix which end in a non-blank token. `b_probs_prev` is the summed
        mass of the paths reducing to the given prefix which end in a blank token.
        ``nb_probs_prev + b_probs_prev = probs_prev``, the total mass of each prefix.
    y_prev
        A long tensor of shape ``(S, N, old_width)`` containing the (reduced) prefixes
        of each path.
    y_prev_last
        A long tensor of shape ``(N, old_width)`` containing the last token in each
        prefix. Arbitrary when the prefix is length 0.
    y_prev_lens
        A long tensor of shape ``(N, old_width)`` specifying the length of each prefix.
        For batch element ``n`` and prefix ``k``, only the tokens in
        ``y_prev[:y_prev_lens[n, k], n, k]`` are valid.
    prev_is_prefix
        A boolean tensor of shape ``(N, old_width, old_width)``. ``prev_is_prefix[n, k,
        k']`` if and only if prefix ``k`` is a (non-strict) prefix of ``k'``

    Returns
    -------
    y_next, y_next_last, y_next_lens, probs_next, next_is_prefix : torch.Tensor
        The ``*next*`` tensors are analogous to the ``*prev*`` arguments, but after
        the step is completed.
    next_src : torch.Tensor
        A long tensor of shape ``(N, width)`` such that the value ``k_old = next_src[n,
        k_new]`` is the index from the previous step (over ``old_width``) that is a
        prefix of the new prefix at ``k_new`` (i.e. its source).
    next_is_nonext : torch.Tensor
        A boolean tensor indicating if the new prefix did _not_ extend its source. If
        true, the new prefix is identical to the source. If false, it has one token
        more.

    See Also
    --------
    pydrobert.torch.modules.CTCPrefixSearch
        Performs the entirety of the search.

    Warnings
    --------
    This function treats large widths the same as
    :func:`pydrobert.torch.modules.CTCPrefixSearch`: the beam will be filled to `width`
    but invalid prefixes will be assigned a total probability of :obj:`-float("inf")`.
    However, this function will only set the non-blank probabilities of invalid prefixes
    to negative infinity; blank probabilities of invalid prefixes may be :obj:`0`
    instead. The total (summed) mass will still be :obj:`-float("inf")`.

    Notes
    -----
    If an extending prefix matches a previous nonextending prefix, the former's mass
    is absorbed into the latter's and the latter's path is invalidated (see warning).
    """

    if width < 1:
        raise RuntimeError("width must be positive")
    ext_probs_t = probs_t[0]
    nonext_probs_t = probs_t[1]
    blank_probs_t = probs_t[2]
    device = ext_probs_t.device
    dtype = ext_probs_t.dtype
    # del probs_t
    if ext_probs_t.dim() != 3:
        raise RuntimeError("ext_probs_t must be 3 dimensional")
    N, Kp, V = ext_probs_t.shape
    if nonext_probs_t.shape != (N, V):
        raise RuntimeError(
            f"expected nonext_probs_t to have shape {(N, V)}, got {nonext_probs_t.shape}"
        )
    if blank_probs_t.shape != (N,):
        raise RuntimeError(
            f"expected blank_probs_t to have shape {(N,)}, got {blank_probs_t.shape}"
        )
    nb_probs_prev = probs_prev[0]
    b_probs_prev = probs_prev[1]
    # del probs_prev
    if nb_probs_prev.shape != (N, Kp):
        raise RuntimeError(
            f"expected nb_probs_prev to have shape {(N, Kp)}, got {nb_probs_prev.shape}"
        )
    if b_probs_prev.shape != (N, Kp):
        raise RuntimeError(
            f"expected b_probs_prev to have shape {(N, Kp)}, got {b_probs_prev.shape}"
        )
    if y_prev.dim() != 3:
        raise RuntimeError("y_prev must be 3 dimensional")
    if y_prev.shape[1:] != (N, Kp):
        raise RuntimeError(
            f"expected last two dimensions of y_prev to be {(N, Kp)}, "
            f"got {y_prev.shape[1:]}"
        )
    tm1 = y_prev.size(0)
    if y_prev_last.shape != (N, Kp):
        raise RuntimeError(
            f"expected y_prev_last to have shape {(N, Kp)}, got {y_prev_last.shape}"
        )
    if y_prev_lens.shape != (N, Kp):
        raise RuntimeError(
            f"expected y_prev_lens to have shape {(N, Kp)}, got {y_prev_lens.shape}"
        )
    if prev_is_prefix.shape != (N, Kp, Kp):
        raise RuntimeError(
            f"expected prev_is_prefix to have shape {(N, Kp, Kp)}, "
            f"got {prev_is_prefix.shape}"
        )
    K = min(width, Kp * (V + 1))  # the maximum number of legitimate paths

    tot_probs_prev = nb_probs_prev + b_probs_prev
    # this is to ensure invalid or empty paths don't mess up our gather/scatter
    y_prev_last = y_prev_last.clamp(0, V - 1)

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
    # (N.B. y_prev_last may be garbage for invalid or empty paths, hence the clamp)
    nb_nonext_probs_cand = nb_probs_prev * nonext_probs_t.gather(1, y_prev_last)  # N,K'
    # del nb_probs_prev, b_probs_prev, tot_probs_prev

    # An extending candidate can match an existing non-extending candidate.
    # We'll dump the extending candidate's probability mass into the non-extending
    # candidate's mass if they're equal.
    #
    # let's assume path k is a strict prefix of path k'. What's the token that we'd have
    # to match if we wanted to extend path k while remaining a prefix of k'?
    # y_prev[y_prev_lens[n, k], n, k'] = to_match[n, k, k']
    if tm1:
        to_match = (
            y_prev.gather(
                0,
                y_prev_lens.clamp(max=tm1 - 1)
                .unsqueeze(2)
                .expand(N, Kp, Kp)
                .transpose(0, 1),
            )
            .transpose(0, 1)
            .clamp(0, V - 1)
        )  # (N, K', K')
        # print(y_prev[:, 0].t(), y_prev_lens[0])
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
        nb_ext_probs_cand.gather(2, to_match).masked_fill(~ext_is_exact, 0.0)
    ).sum(1)
    # clear the probabilities of extensions k->v that exactly matched some k' for v
    has_match = (
        torch.nn.functional.one_hot(to_match, V).to(torch.bool)
        & ext_is_exact.unsqueeze(3)
    ).any(2)
    nb_ext_probs_cand = nb_ext_probs_cand.masked_fill(has_match, -float("inf"))
    # del has_match, ext_is_exact

    # we can finally determine the top k paths. Put the non-extending candidates after
    # the extending candidates (the last K' elements of the second dimension)
    tot_probs_cand = torch.cat(
        [nb_ext_probs_cand.view(N, Kp * V), nb_nonext_probs_cand + b_nonext_probs_cand],
        1,
    )  # (N, K' * (V + 1))
    next_ind = tot_probs_cand.topk(K, 1)[1]  # (N, K)
    # del tot_probs_cand

    next_is_nonext = next_ind >= (Kp * V)
    next_src = torch.where(
        next_is_nonext, next_ind - (Kp * V), trunc_divide(next_ind, V)
    )
    next_ext = next_ind % V

    y_next_prefix_lens = y_prev_lens.gather(1, next_src)  # (N, K)
    y_next = torch.cat(
        [
            y_prev.gather(2, next_src.unsqueeze(0).expand(tm1, N, K)),
            torch.empty((1, N, K), device=y_prev.device, dtype=y_prev.dtype),
        ],
        0,
    ).scatter(
        0, y_next_prefix_lens.unsqueeze(0), next_ext.unsqueeze(0)
    )  # (t, N, K)
    y_next_lens = y_next_prefix_lens + (~next_is_nonext)
    # del y_next_prefix_lens

    nb_ext_probs_next = nb_ext_probs_cand.view(N, Kp * V).gather(
        1, next_ind.clamp(max=Kp * V - 1)
    )  # (N, K)
    nb_nonext_probs_next = nb_nonext_probs_cand.gather(1, next_src)  # (N, K)
    nb_probs_next = torch.where(next_is_nonext, nb_nonext_probs_next, nb_ext_probs_next)
    # del nb_ext_probs_next, nb_nonext_probs_next, nb_nonext_probs_cand, nb_ext_probs_cand

    b_probs_next = b_nonext_probs_cand.gather(1, next_src) * next_is_nonext  # (N, K)
    # del b_nonext_probs_cand

    y_next_last = y_prev_last.gather(1, next_src) * next_is_nonext + next_ext * (
        ~next_is_nonext
    )
    # del y_prev_last

    next_prefix_is_prefix = prev_is_prefix.gather(
        1, next_src.unsqueeze(2).expand(N, K, Kp)
    ).gather(2, next_src.unsqueeze(1).expand(N, K, K))
    next_len_leq = y_next_lens.unsqueeze(2) <= y_next_lens.unsqueeze(1)
    next_to_match = y_next.gather(
        0, (y_next_lens - 1).clamp(min=0).unsqueeze(2).expand(N, K, K).transpose(0, 1)
    ).transpose(0, 1)
    next_ext_matches = next_to_match == next_ext.unsqueeze(2)
    next_is_prefix = (
        next_prefix_is_prefix
        & next_len_leq
        & (
            next_is_nonext.unsqueeze(2)
            | (~next_is_nonext.unsqueeze(2) & next_ext_matches)
        )
    )
    # del next_prefix_is_prefix, next_len_leq, next_to_match, next_ext_matches
    # del next_ext, next_ind

    if K < width:
        # we've exceeded the possible number of legitimate paths. Append up to the
        # width but set their probabilities to -inf and make sure they aren't
        # considered a prefix of anything
        # This should only happen once in a given rollout assuming width stays
        # constant, so it's ok to be a bit expensive.
        rem = width - K
        y_next = torch.cat([y_next, y_next.new_empty(tm1 + 1, N, rem)], 2)
        zeros = torch.zeros((N, rem), device=device, dtype=y_next_last.dtype)
        y_next_last = torch.cat([y_next_last, zeros], 1)
        y_next_lens = torch.cat([y_next_lens, zeros], 1)
        neg_inf = torch.full((N, rem), -float("inf"), device=device, dtype=dtype)
        nb_probs_next = torch.cat([nb_probs_next, neg_inf], 1)
        b_probs_next = torch.cat([b_probs_next, neg_inf], 1)
        false_ = torch.zeros((N, rem), device=device, dtype=torch.bool)
        next_is_nonext = torch.cat([next_is_nonext, false_], 1)
        next_is_prefix = torch.cat(
            [next_is_prefix, false_.unsqueeze(1).expand(N, K, rem)], 2
        )
        next_is_prefix = torch.cat(
            [next_is_prefix, false_.unsqueeze(2).expand(N, rem, width)], 1
        )
        next_src = torch.cat([next_src, zeros], 1)

    return (
        y_next,  # (t, N, K)
        y_next_last,  # (N, K)
        y_next_lens,  # (N, K)
        (nb_probs_next, b_probs_next),  # (N, K), (N, K)
        next_is_prefix,  # (N, K, K)
        next_src,  # (N, K)
        next_is_nonext,  # (N, K)
    )


class CTCPrefixSearch(torch.nn.Module):
    r"""Perform a CTC prefix search with optional shallow fusion

    A Connectionist Temporal Classification [graves2006]_ prefix search is similar to a
    beam search, but a fixed number of (reduced) prefixes are maintained in the beam
    rather than a fixed number of paths. Reduced paths contain no blank labels.

    Shallow fusion [gulcehre2015]_ is enabled by initializing this module with `lm`.
    Shallow fusion updates the probability of extending a prefix :math:`y_{1..t-1}` with
    a new token math:`v` (:math:`v` is not blank) with the following equation

    .. math::
        \log S(y_t=v|y_{1..t-1}) = \log P_{logits}(y_t=v) +
                                                \beta \log P_{lm}(y_t = v|y_{1..t-1})

    The resulting value :math:`log S(y_t=v)` is not technically a probability.

    Parameters
    ----------
    width
        The number of prefixes to keep track of per step.
    beta
        The mixing coefficient :math:`\beta` used when performing shallow fusion.
    lm
        If set, the language model used in shallow fusion. Specifying `lm` will
        restrict the extended vocabulary size of `logits` to be one more than that
        of `lm`: ``lm.vocab_size == V``.
    
    Call Parameters
    ---------------
    logits : torch.Tensor
        A tensor of shape ``(T, N, V + 1)`` s.t. ``logits[t, n]`` represents the
        unnormalized log-probabilities over the extended vocabulary (including blanks)
        at step ``t`` of batch element ``n``. The blank type logits are assumed to be
        stored in the final index of the vocabulary: ``logits[..., V]``.
    logit_lens : torch.Tensor or None, optional
        An optional tensor of shape ``(N,)`` s.t., for a given batch index ``n``, only
        the values in the slice ``logits[:lens[n], n]`` are valid. If unset then all
        sequences are assumed to be of length ``T``.
    prev : Dict[str, torch.Tensor] or None, optional
        Whatever state info must be initially passed to the `lm`, if specified.
    
    Returns
    -------
    y : torch.Tensor
        A long tensor of shape ``(S, N, width)`` containing the `width` prefixes per
        batch element, ``S <= T``.
    y_lens : torch.Tensor
        A long tensor of shape ``(N, width)`` of the lengths of the corresponding
        prefixes: for each batch element ``n`` and prefix ``k``, only the tokens
        ``y[:y_lens[n, k], n, k]`` are valid.  Note that for all ``k``, ``y_lens[n, k]
        <= logit_lens[n]``.
    y_probs : torch.Tensor
        A tensor of shape ``(N, width)`` containing those prefix's estimated (not log)
        probabilities. Prefixes are ordered in decreasing probability (``y_probs[n, k]
        >= y_probs[n, k + 1]``).

    Warnings
    --------
    The blank index, effectively ``V``, is different from the default index of
    :class:`torch.nn.CTCLoss`, ``0``. We chose this in order to avoid confusion between
    the index set of `logits` and the index set of `lm`: this way, the interpretation of
    the indices up to but excluding ``V`` in both refer to the same type/label.

    Return values will always contain `width` prefixes, regardless of whether this is
    possible. The probabilities of invalid prefixes will be set to :obj:`-float("inf")`
    and will populate the latter indices of the beam.

    Notes
    -----
    The CTC prefix search is often called a beam search in the literature. We stick with
    the name from [graves2006]_ as it is entirely possible to apply a normal beam search
    to CTC logits, only removing blank labels after the search. Doing so would be faster
    and may not lead to much decrease in performance if `logits` is sufficiently
    "peaky".
    """

    __constants__ = ["width", "beta"]

    width: int
    beta: float

    def __init__(
        self,
        width: int,
        beta: float = 0.2,
        lm: Optional[MixableSequentialLanguageModel] = None,
    ):
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive")
        self.width = width
        self.beta = beta
        if lm is None:
            self.add_module("lm", None)
        else:
            self.lm = lm

    def reset_parameters(self) -> None:
        if self.lm is not None and hasattr(self.lm, "reset_parameters"):
            self.lm.reset_parameters()

    def forward(
        self,
        logits: torch.Tensor,
        lens: Optional[torch.Tensor] = None,
        prev_: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prev_ is None:
            prev: Dict[str, torch.Tensor] = dict()
        else:
            prev = prev_
        if logits.dim() != 3:
            raise RuntimeError("logits must be 3 dimensional")
        T, N, Vp1 = logits.shape
        V = Vp1 - 1
        device, dtype = logits.device, logits.dtype
        if self.lm is not None and self.lm.vocab_size != V:
            raise RuntimeError(
                f"Expected dim 2 of logits to be {self.lm.vocab_size + 1}, got {Vp1}"
            )
        if lens is None:
            lens = torch.full((N,), T, device=logits.device, dtype=torch.long)
            len_min = len_max = T
        elif lens.dim() != 1:
            raise RuntimeError("lens must be 1 dimensional")
        elif lens.size(0) != N:
            raise RuntimeError(f"expected dim 0 of lens to be {N}, got {lens.size(0)}")
        else:
            len_min, len_max = int(lens.min().item()), int(lens.max().item())

        probs = logits.softmax(2)
        blank_probs = probs[..., V]  # (T, N)
        nonext_probs = probs[..., :V]  # (T, N, V)

        nb_probs_prev = torch.zeros((N, 1), device=device, dtype=dtype)
        b_probs_prev = torch.ones((N, 1), device=device, dtype=dtype)
        y_prev = torch.empty((0, N, 1), dtype=torch.long, device=logits.device)
        y_prev_lens = y_prev_last = torch.zeros(
            (N, 1), dtype=torch.long, device=logits.device
        )
        prev_is_prefix = torch.full(
            (N, 1, 1), 1, device=logits.device, dtype=torch.bool
        )
        if self.lm is not None:
            prev = self.lm.update_input(prev, y_prev)
        prev_width = 1
        for t in range(len_max):
            valid_mask = None if t < len_min else (t < lens).unsqueeze(1)  # (N, 1)
            nonext_probs_t, blank_probs_t = nonext_probs[t], blank_probs[t]
            if self.lm is None or not self.beta:
                ext_probs_t = nonext_probs_t.unsqueeze(1).expand(N, prev_width, V)
                in_next = dict()
            else:
                lm_log_probs_t, in_next = self.lm.calc_idx_log_probs(
                    y_prev.flatten(1), prev, y_prev_lens.flatten()
                )
                lm_probs_t = (self.beta * lm_log_probs_t).exp().view(N, prev_width, V)
                # note we're no longer in log space, so it's a product
                ext_probs_t = lm_probs_t * nonext_probs_t.unsqueeze(1)
            (
                y_next,
                y_next_last,
                y_next_lens,
                (nb_probs_next, b_probs_next),
                next_is_prefix,
                next_src,
                next_is_nonext,
            ) = ctc_prefix_search_advance(
                (ext_probs_t, nonext_probs_t, blank_probs_t),
                self.width,
                (nb_probs_prev, b_probs_prev),
                y_prev,
                y_prev_last,
                y_prev_lens,
                prev_is_prefix,
            )

            if self.lm is not None and self.beta:
                next_src = (
                    torch.arange(
                        0, prev_width * N, prev_width, device=next_src.device
                    ).unsqueeze(1)
                    + next_src
                )
                prev = self.lm.extract_by_src(prev, next_src.flatten())
                in_next = self.lm.extract_by_src(in_next, next_src.flatten())
                prev = self.lm.mix_by_mask(prev, in_next, next_is_nonext.flatten())

            if valid_mask is None:
                y_prev_lens = y_next_lens
                nb_probs_prev, b_probs_prev = nb_probs_next, b_probs_next
            else:
                y_next[:-1] = torch.where(valid_mask.unsqueeze(0), y_next[:-1], y_prev)
                y_prev_lens = torch.where(valid_mask, y_next_lens, y_prev_lens)
                if prev_width < self.width:
                    assert prev_width == 1  # otherwise advance would've padded it
                    # add invalid path probs rather than broadcast the one good one
                    neg_inf = nb_probs_prev.new_full(
                        (N, self.width - prev_width), -float("inf")
                    )
                    nb_probs_prev = torch.cat([nb_probs_prev, neg_inf], 1)
                    b_probs_prev = torch.cat([b_probs_prev, neg_inf], 1)
                nb_probs_prev = torch.where(valid_mask, nb_probs_next, nb_probs_prev)
                b_probs_prev = torch.where(valid_mask, b_probs_next, b_probs_prev)
            y_prev = y_next
            # we can let y_next_last and next_is_prefix continue spinning after t passes
            # the length
            y_prev_last, prev_is_prefix = y_next_last, next_is_prefix
            prev_width = self.width

        probs_prev = nb_probs_prev + b_probs_prev

        if prev_width == 1 != self.width:
            # fill the shape, but only the first (empty path is valid)
            y_prev = y_prev.repeat(1, 1, self.width)
            y_prev_lens = y_prev_lens.repeat(1, self.width)
            probs_prev = torch.cat(
                [
                    probs_prev,
                    probs_prev.new_full((N, self.width - prev_width), -float("inf")),
                ],
                1,
            )
        # now we zero out the probabilities of duplicate paths which could've arisen
        return y_prev, y_prev_lens, probs_prev


@script
def random_walk_advance(
    log_probs_t: torch.Tensor,
    log_probs_prev: torch.Tensor,
    y_prev: torch.Tensor,
    y_prev_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random walk step function
    
    Parameters
    ----------
    log_probs_t
        A tensor of shape ``(N, V)`` containing the log probabilities of extending a
        given path with a token of a given type in the vocabulary.
    log_probs_prev
        A tensor of shape ``(N,)`` containing the log probablities of the paths so far.
    y_prev
        A tensor of shape ``(S, N)`` containing the paths so far.
    y_prev_lens
        A tensor of shape ``(N,)`` specifying the lengths of the prefixes.
        For batch element ``n``, only the values  ``y_prev[:y_prev_lens[n], n]``
        are valid. If unspecified, it is assumed that ``y_prev_lens[:] == S``.
    
    Returns
    -------
    y_next, log_probs_next : torch.Tensor
        The ``*next**`` tensors can be interpreted in the same way as their ``*prev*``
        counterparts, but after the step. `y_next` is of shape either ``(S, N)`` or ``(S
        + 1, N)``, depending on whether the size of `y_prev` needed to grow in order to
        accommodate the newest token in the path. Note the next path lengths are always
        the previous path lengths plus one, i.e. ``y_next_lens = y_prev_lens + 1``.
    
    Warnings
    --------
    This function has been drastically simplified after v0.3.0. The logic for
    end-of-sequence handling has been punted to the encapsulating search module. The
    logic for the relaxation has been entirely removed given the revamp of
    :mod:`pydrobert.torch.estimators`.

    See Also
    --------
    pydrobert.torch.modules.RandomWalk
        For the full random walk.
    """
    if log_probs_t.dim() != 2:
        raise RuntimeError("log_probs_t must be 2-dimensional")
    N, V = log_probs_t.shape
    if log_probs_prev.shape != (N,):
        raise RuntimeError(
            f"Expected log_probs_prev to be of shape {(N,)}, got {log_probs_prev.shape}"
        )
    if y_prev.dim() != 2:
        raise RuntimeError("y_prev must be 2-dimensional")
    if y_prev.size(1) != N:
        raise RuntimeError(f"Expected dim 1 of y_prev to be {N}, got {y_prev.size(-1)}")
    tm1 = y_prev.size(0)
    if y_prev_lens is not None and y_prev_lens.shape != (N,):
        raise RuntimeError(
            f"Expected y_prev_lens to have shape {(N,)}, got {y_prev_lens.shape}"
        )
    y_t = torch.multinomial(log_probs_t.exp(), 1, True)  # (N, 1)
    log_probs_next = log_probs_prev + log_probs_t.gather(1, y_t).squeeze(1)
    y_t = y_t.T  # (1, N)
    if tm1:
        if y_prev_lens is None:
            y_next = torch.cat([y_prev, y_t], 0)
        else:
            # don't make y bigger unless we have to
            if int(y_prev_lens.max().item()) >= tm1:
                y_next = torch.cat([y_prev, y_t], 0)
            else:
                y_next = y_prev
            y_next = y_next.scatter(0, y_prev_lens.unsqueeze(0), y_t)
    else:
        y_next = y_t

    return y_next, log_probs_next


class RandomWalk(torch.nn.Module):
    """Perform a random walk on the outputs of a SequentialLanguageModel

    A random walk iteratively builds a sequence of tokens by sampling the next token
    given a prefix of tokens.

    A path continues to be extended until it emits an end-of-sequence (`eos`) symbol (if
    set). The walk ends for a batch as soon as all paths in the batch have ended or
    `max_iters` has been reached, whichever comes first. It is therefore necessary to
    set at least one of `eos` or `max_iters`.

    Parameters
    ----------
    lm
        The language model responsible for producing distributions over the next token
        type
    eos
        The end of sequence type. If set, must be in-vocabulary (according to
        ``lm.vocab_size``). Either `eos` or `max_iters` must be set.
    max_iters
        The maximum number of tokens to generate in the paths before returning. Either
        `eos` or `max_iters` must be set.
    
    Call Parameters
    ---------------
    y_prev : torch.Tensor
        A long tensor of shape ``(S*, N)`` containing the prefixes prior to the start of
        the search. `y_prev` may be an empty tensor of shape ``(0, N)`` indicating that
        each sequence in the batch should be completely generated by the walk. A
        non-empty tensor `y_prev` can be used to prime the generation process.
    prev : Dict[str, torch.Tensor], optional
        Whatever state info must be initially passed to the `lm`.
    
    Returns
    -------
    y : torch.Tensor
        A long tensor of shape ``(S, N)`` containing the paths.
    y_lens : torch.Tensor
        A long tensor of shape ``(N,)`` of the lengths of the corresponding paths
        including the first instance of `eos`, if it exists. For batch element ``n``,
        only the tokens in ``y[:y_lens[n], n]`` are valid.
    y_log_probs : torch.Tensor
        A tensor of shape ``(N,)`` containing the log probabilities of the paths.
    """

    __constants__ = ["eos", "max_iters"]

    eos: Optional[int]
    max_iters: Optional[int]

    def __init__(
        self,
        lm: SequentialLanguageModel,
        eos: Optional[int] = None,
        max_iters: Optional[int] = None,
    ):
        super().__init__()
        if eos is not None:
            if eos < -lm.vocab_size or eos > lm.vocab_size - 1:
                raise ValueError(
                    f"Expected eos to be in the range [{-lm.vocab_size}, "
                    f"{lm.vocab_size - 1}], got {eos}"
                )
            eos = (eos + lm.vocab_size) % lm.vocab_size
        if max_iters is not None and max_iters < 0:
            raise ValueError("max_iters must be non-negative")
        if eos is None and max_iters is None:
            raise ValueError("at least one of eos or max_iters must be set")
        self.lm = lm
        self.eos = eos
        self.max_iters = max_iters

    def reset_parameters(self) -> None:
        if hasattr(self.lm, "reset_parameters"):
            self.lm.reset_parameters()

    @torch.jit.export
    def update_log_probs_for_step(
        self,
        log_probs_prev: torch.Tensor,
        log_probs_t: torch.Tensor,
        y_prev: torch.Tensor,
        y_prev_lens: torch.Tensor,
        eos_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update log_probs_prev and log_probs_t for a step of the random walk

        Subclasses may overload this method to modify the log-probabilities of the
        prefixes as well as the log-probabilities of the tokens extending each path.

        Parameters
        ----------
        log_probs_prev
            Of shape ``(N,)`` containing the log probabilities of paths up to the
            current step.
        log_probs_t
            Of shape ``(N, V)`` containing the log probabilities of extending each path
            with a token of a given type.
        y_prev
            Of shape ``(S, N)`` containing the paths up to the current step
        y_prev_lens
            Of shape ``(N,)`` containing the lengths of the paths up to the current step
            (including the first `eos`, if any). For batch element ``n`` only the tokens
            in the range ``y_prev[:y_prev_lens[n], n]`` are valid.
        eos_mask
            A boolean tensor of shape ``(N)`` which is true when a path has already
            ended. Will be all :obj:`False` when `eos` is unset or there is no history.

        Returns
        -------
        log_probs_prev_new, log_probs_t_new : torch.Tensor
            The modified versions of the associated arguments

        Notes
        -----
        Modifications mean that the results will no longer be interpreted as log
        probabilities, but scores.
        """
        return log_probs_prev, log_probs_t

    if TYPE_CHECKING:

        def forward(
            self, y_prev: torch.Tensor, prev: Dict[str, torch.Tensor] = dict()
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            pass

    else:

        def forward(
            self, y: torch.Tensor, _prev: Optional[Dict[str, torch.Tensor]] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if _prev is None:
                prev = dict()
            else:
                prev = _prev
            if y.dim() != 2:
                raise RuntimeError("y_prev must be 2 dimensional")

            device = y.device
            S_prev, N = y.size(0), y.size(1)
            prev = self.lm.update_input(prev, y)

            if self.eos is not None and S_prev:
                y_lens = (
                    -((y == self.eos).cumsum(0).clamp(max=1).sum(0) - 1).clamp_min_(0)
                    + S_prev
                )
                eos_mask = (y == self.eos).any(0)
            else:
                y_lens = torch.full((N,), S_prev, dtype=torch.long, device=device)
                eos_mask = torch.zeros(N, device=device, dtype=torch.bool)
            log_probs = torch.zeros(N, device=device)

            if self.max_iters is None:
                max_iters = 1024 * 1024 * 1024 * 1024
            else:
                max_iters = self.max_iters
            for t in range(S_prev, max_iters + S_prev):
                if eos_mask.all():
                    break
                t = torch.tensor(t, device=device)

                # determine extension probabilities
                log_probs_t, prev = self.lm.calc_idx_log_probs(y[:t], prev, t)

                # update probabilities if the subclass so desires
                log_probs, log_probs_t = self.update_log_probs_for_step(
                    log_probs, log_probs_t, y[:t], y_lens, eos_mask
                )

                if self.eos is not None:
                    # if a path has finished, we allocate the entire probability mass to
                    # the eos token
                    log_probs_t = log_probs_t.masked_fill(
                        eos_mask.unsqueeze(1), -float("inf")
                    )
                    eos_mask_ = eos_mask.unsqueeze(1).repeat(1, self.lm.vocab_size)
                    eos_mask_[..., : self.eos] = False
                    eos_mask_[..., self.eos + 1 :] = False
                    log_probs_t = log_probs_t.masked_fill(eos_mask_, 0.0)

                y, log_probs = random_walk_advance(log_probs_t, log_probs, y, y_lens)

                if self.eos is not None:
                    # if the thing prior to this was not an eos, then either this isn't
                    # an eos or its the first eos. Both add to lens. This is why we
                    # accumulate using the previous eos mask
                    y_lens += ~eos_mask
                    eos_mask = (
                        y.gather(0, y_lens.unsqueeze(0) - 1).squeeze(0) == self.eos
                    )
                else:
                    y_lens += 1

            return y, y_lens, log_probs


@script
def _sequence_log_probs_tensor(
    logits: torch.Tensor, hyp: torch.Tensor, dim: int, eos: Optional[int]
) -> torch.Tensor:
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


@script
def _sequence_log_probs_ps(
    logits: Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ],
    hyp: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    hyp_dim = hyp.dim()
    if dim < -hyp_dim or dim > hyp_dim - 1:
        raise RuntimeError(
            "Dimension out of range (expected to be in range of [{}, {}], but "
            "got {})".format(-hyp_dim, hyp_dim - 1, dim)
        )
    logits, batch_sizes, sidxs, uidxs = logits
    if sidxs is not None:
        hyp = torch.index_select(hyp, 1 - dim, sidxs)  # sort hyp
    lens = (
        (torch.arange(hyp.size(1 - dim)).unsqueeze(1) < batch_sizes)
        .to(torch.long)
        .sum(1)
    )
    hyp = torch.nn.utils.rnn.pack_padded_sequence(hyp, lens, batch_first=bool(dim))[0]
    num_classes = logits.shape[1]
    logits = torch.nn.functional.log_softmax(logits, -1)
    mask = hyp.lt(0) | hyp.ge(num_classes)
    hyp = hyp.masked_fill(mask, 0)
    logits = logits.gather(1, hyp.unsqueeze(1)).squeeze(1)
    logits = logits.masked_fill(mask, 0.0)
    logits = torch.nn.utils.rnn.pad_packed_sequence(
        SpoofPackedSequence(logits, batch_sizes, None, None), batch_first=True,
    )[0].sum(1)
    if uidxs is not None:
        logits = logits[uidxs]
    return logits


@functional_wrapper("SequentialLogProbabilities")
def sequence_log_probs(
    logits: Any, hyp: torch.Tensor, dim: int = 0, eos: Optional[int] = None,
) -> torch.Tensor:
    if isinstance(logits, torch.Tensor):
        return _sequence_log_probs_tensor(logits, hyp, dim, eos)
    elif jit_isinstance(
        logits,
        Tuple[
            torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
        ],
    ):
        return _sequence_log_probs_ps(logits, hyp, dim)
    raise RuntimeError("logits must be either a Tensor or PackedSequence")


class SequenceLogProbabilities(torch.nn.Module):
    r"""Calculate joint log probability of sequences

    Letting :math:`t` index the step dimension and :math:`b` index all other shared
    dimensions of `logits` and `hyp`, this function outputs a tensor `log_probs` of the
    log-joint probability of sequences in the batch:

    .. math::

        \log Pr(samp_b = hyp_b) = \log \left(
            \prod_t Pr(samp_{b,t} == hyp_{b,t}; logits_{b,t})\right)

    :math:`logits_{b,t}` (with the last dimension free) characterizes a categorical
    distribution over ``num_classes`` tokens via a softmax function. We assume
    :math:`samp_{b,t}` is independent of :math:`samp_{b',t'}` given :math:`logits_t`.

    If `eos` (end-of-sentence) is set, the first occurrence at :math:`b,t` is included
    in the sequence, but all :math:`b,>t` are ignored.

    Parameters
    ----------
    dim
        The sequence dimension of `logits`.
    eos
        If set, specifies the end-of-sequence token index in the last dimension of
        `logits`.

    Call Parameters
    ---------------
    logits : torch.Tensor or torch.nn.utils.rnn.PackedSequence
        A tensor of shape ``(A*, T, B*, num_classes)`` where ``T`` enumerates the
        time/step `dim`-th dimension. The unnormalized log-probabilities over type.
        Alternatively, `logits` may be a packed sequence. In this case, `eos` is
        ignored.
    hyp : torch.Tensor
        A long tensor of shape ``(A*, T, B*)``. The token sequences over time. Any
        values of `hyp` not in ``[0, num_classes)`` will be considered padding and
        ignored.

    Returns
    -------
    log_probs : torch.Tensor
        A tensor of shape ``(A*, B*)`` containing the log probabilties of the sequences
        in `hyp`.

    Notes
    -----
    :class:`PackedSequence` instances with ``enforce_sorted=False`` first sort sequences
    by length. The sort is not guaranteed to be deterministic if some entries have equal
    length. To avoid the possibility that `logits` and `hyp` are sorted differently, we
    require `hyp` to always be a :class:`torch.Tensor`.
    """

    __constants__ = ["dim", "eos"]
    dim: int
    eos: Optional[int]

    def __init__(self, dim: int = 0, eos: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.eos = eos

    def extra_repr(self) -> str:
        s = f"dim={self.dim}"
        if self.eos is not None:
            s += f", eos={self.eos}"
        return s

    def forward(self, logits: Any, hyp: torch.Tensor) -> torch.Tensor:
        return sequence_log_probs(logits, hyp, self.dim, self.eos)
