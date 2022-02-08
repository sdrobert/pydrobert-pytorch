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
from optparse import Option

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union

import torch

from . import config
from ._lm import ExtractableSequentialLanguageModel, MixableSequentialLanguageModel
from ._compat import script, trunc_divide, jit_isinstance, SpoofPackedSequence
from ._string import _lens_from_eos


@script
def beam_search_advance(
    log_probs_t: torch.Tensor,
    width: int,
    log_probs_prev: torch.Tensor,
    y_prev: torch.Tensor,
    y_prev_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Beam search step function

    The step function of any beam search.

    Parameters
    ----------
    log_probs_t : torch.Tensor
        A tensor of shape ``(N, old_width, V)`` containing the probabilities of
        extending a given path with a token of a given type in the vocabulary.
    width : int
        The beam width
    log_probs_prev : torch.Tensor
        A tensor of shape ``(N, old_width)`` containing the log probabilities of
        the paths so far.
    y_prev : torch.Tensor
        A tensor of shape ``(S, N, old_width)`` containing the path prefixes.
    y_prev_lens : torch.Tensor or None, optional
        A tensor of shape ``(N, old_width)`` specifying the lengths of the prefixes.
        For batch element ``n``, only the values ``y_prev[:y_prev_lens[n, k], n, k]``
        are valid. If unspecified, it is assumed ``y_prev_lens[n, k] == S``.

    Returns
    -------
    y_next, y_next_lens, log_probs_next, next_src : torch.Tensor, torch.Tensor
        The ``*next*`` tensors can be interpreted in the same way as their ``*prev*``
        counterparts, but after the step. The ``old_width`` dimension has been replaced
        with `width`. ``next_src` is a long tensor of shape ``(N, width)`` such that the
        value ``k_old = next_src[n, k_new]`` is the index from the previous step (over
        ``old_width``) that is a prefix of the new path at ``k_new`` (i.e. its source).

    Warnings
    --------
    This function has been drastically simplified after v0.3.0. The logic for
    end-of-sequence handling has been punted to the encapsulating search module.

    If there are too few possible extensions to fill the beam, undefined paths will be
    added to the end of the beam with probability :obj:`-float('inf')`. This means that
    an invalid path cannot be distibguished from a 0-probability path. Consider using a
    very negative value as a replacement for ``log 0``, e.g. ``log_probs_t =
    log_probs_t.clamp(min=torch.finfo(torch.float).min / 2)``.
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
    next_token = (next_ind % V).unsqueeze(0)  # (1, N, K)

    if tm1:
        y_prev_prefix = y_prev.gather(2, next_src.unsqueeze(0).expand(tm1, N, K))
        y_next = torch.cat([y_prev_prefix, next_token], 0)
        if y_prev_lens is None:
            y_next_lens = next_token.new_full((N, K), tm1 + 1)
        else:
            y_prev_lens_prefix = y_prev_lens.gather(1, next_src)
            y_next = y_next.scatter(0, y_prev_lens_prefix.unsqueeze(0), next_token)
            y_next_lens = y_prev_lens_prefix + 1
    elif y_prev_lens is not None and (y_prev_lens != 0).any():
        raise RuntimeError("Invalid lengths for t=0")
    else:
        y_next = next_token
        y_next_lens = torch.ones(
            (N, K), dtype=next_token.dtype, device=next_token.device
        )

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

    This module has the following signature:

        search(y_prev, prev=dict())

    `y_prev` is long tensor of shape ``(S*, N[, old_width])``. In most cases, `y_prev`
    should be an empty tensor of shape ``(0, N[, 1])``, though it can be used start the
    search with different prefixes. `prev` is whatever input is initially passed into
    `lm`.

    A path continues to be extended until it is either pruned or emits an
    end-of-sequence (`eos`) symbol (if set). The search ends for a batch element when
    its highest probability path ends with an `eos` or all paths end with an `eos`
    (depending on the setting of `finish_all_paths`). The search ends for the entire
    batch either when the search for all batch elements have ended or `max_iters` steps
    has been reached, whichever comes first. It is therefore necessary to set at least
    one of `eos` or `max_iters`.

    The call returns a triple of tensors ``y, y_lens, y_log_probs``. ``y`` is a long
    tensor of shape ``(S, N, width)`` containing the `width` paths per batch element.
    `y_lens` is a long tensor of shape ``(N, width)`` of the lengths of the
    corresponding paths including the first instance of `eos`, if it exists. For batch
    element ``n`` and path ``k``, only the tokens in ``y[:y_lens[n, k], n, k]`` are
    valid.  `y_log_probs` is of shape ``(N, width)`` and contains the log probabilities
    of the paths.

    Parameters
    ----------
    lm : ExtractableSequentialLanguageModel
        The language model responsible for producing distributions over the next token
        type
    width : int
        The beam width
    eos : int or None, optional
        The end of sequence type. If set, must be in-vocabulary (according to
        ``lm.vocab_size``). Either `eos` or `max_iters` must be set.
    max_iters : int or None, optional
        The maximum number of tokens to generate in the paths before returning. Either
        `eos` or `max_iters` must be set.
    finish_all_paths : bool, optional
        Applicable only when `eos` is set. If :obj:`True`, waits for all paths in all
        batches' beams to emit an `eos` symbol before stopping. If :obj:`False`, only
        the highest probability path need end with an `eos` before stopping.

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
        log_probs_prev : torch.Tensor
            Of shape ``(N, K)`` containing the log probabilities of paths up to the
            current step.
        log_probs_t : torch.Tensor
            Of shape ``(N, K, V)`` containing the log probabilities of extending each
            path with a token of a given type.
        y_prev : torch.Tensor
            Of shape ``(S, N, K)`` containing the paths in the beam up to the current
            step.
        y_prev_lens : torch.Tensor
            Of shape ``(N, K)`` containing the lengths of the paths up to the current
            step (including the first `eos`, if any). For batch element ``n`` and path
            ``k``, only the tokens in the range ``y_prev[:y_prev_lens[n, k], n, k]`` are
            valid.
        eos_mask : torch.Tensor
            A boolean tensor of shape ``(N, K)`` which is true when a path has already
            ended. Will be all :obj:`False` when `eos` is unset or there is no history.

        Returns
        -------
        log_probs_prev_new, log_probs_t_new : torch.Tensor, torch.Tensor
            The modified versions of the associated arguments

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

    if TYPE_CHECKING:

        def forward(
            self, y_prev: torch.Tensor, prev: Dict[str, torch.Tensor] = dict()
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            pass

    else:

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
                    -((y_prev == self.eos).cumsum(0).clamp(max=1).sum(0) - 1).clamp(
                        min=0
                    )
                    + S_prev
                )

                len_eq_mask = y_prev_lens.unsqueeze(1) == y_prev_lens.unsqueeze(
                    2
                )  # NKK
                tok_ge_len_mask = (
                    torch.arange(S_prev, device=device).view(S_prev, 1, 1)
                    >= y_prev_lens
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
                    y_next_lens = y_next_lens - eos_mask.gather(1, next_src).to(
                        y_next_lens
                    )

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
                    y_next[:-1] = torch.where(
                        done_mask.unsqueeze(0), y_prev, y_next[:-1]
                    )
                    log_probs_next = torch.where(
                        done_mask, log_probs_prev, log_probs_next
                    )
                    y_next_lens = torch.where(done_mask, y_prev_lens, y_next_lens)

                y_prev = y_next
                y_prev_lens = y_next_lens
                log_probs_prev = log_probs_next
                prev_width = self.width

            y_prev, log_probs_prev, y_prev_lens = self._to_width(
                y_prev, log_probs_prev, y_prev_lens
            )

            return y_prev, y_prev_lens, log_probs_prev


@script
def ctc_greedy_search(
    logits: torch.Tensor,
    in_lens: Optional[torch.Tensor] = None,
    blank_idx: int = -1,
    batch_first: bool = False,
    is_probs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Functional version of CTCGreedySearch
    
    See Also
    --------
    pydrobert.torch.modules.CTCGreedySearch
        For more information on this function's parameters and return values
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

    When instantiated, this module has the signature::

        max_, paths, out_lens = ctc_greedy_search(logits[, in_lens])
    
    Where `logits` is a tensor of shape ``(N, T, V)`` where ``T`` is the sequence
    dimension, ``N`` is the batch dimension, and ``V`` is the number of classes
    including the blank label. ``logits[n, t, :]`` represent the unnormalized
    log-probabilities of the labels at time ``t`` in batch element ``n``. If specified,
    `in_lens` is a tensor of  shape ``(N,)`` providing the lengths of the sequence in
    the batch. For a given batch element ``n``, only the values of `logits` in the slice
    ``logits[n, :in_lens[n]]`` will be considered valid. The call returns a triple
    ``max_, paths, out_lens`` where `max_` is a tensor of shape ``(N,)`` containing the
    total log-probability of the greedy path.  `paths` is a long tensor of shape ``(N,
    T)`` which stores the reduced greedy paths. `out_lens` is a long tensor of shape
    ``(N,)`` which specifies the lengths of the greedy paths within `paths`: for a given
    batch element ``n``, the reduced greedy path is the sequence in the range ``paths[n,
    :out_lens[n]]``. The values of `paths` outside this range are undefined.

    Parameters
    ----------
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
        A long tensor of shape ``(S, N, old_width)`` containing the (reduced) prefixes
        of each path.
    y_prev_last : torch.Tensor
        A long tensor of shape ``(N, old_width)`` containing the last token in each
        prefix. Arbitrary when the prefix is length 0.
    y_prev_lens: torch.Tensor
        A long tensor of shape ``(N, old_width)`` specifying the length of each prefix.
        For batch element ``n`` and prefix ``k``, only the tokens in
        ``y_prev[:y_prev_lens[n, k], n, k]`` are valid.
    prev_is_prefix : torch.Tensor
        A boolean tensor of shape ``(N, old_width, old_width)``. ``prev_is_prefix[n, k,
        k']`` if and only if prefix ``k`` is a (non-strict) prefix of ``k'``

    Returns
    -------
    y_next, y_next_last, y_next_lens, probs_next, next_is_prefix, next_src,
    next_is_nonext : torch.Tensor, torch.Tensor, torch.Tensor,
                     (torch.Tensor, torch.Tensor), torch.Tensor, torch.Tensor
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

    Warnings
    --------
    This function treats large widths the same as
    :func:`pydrobert.torch.layers.CTCPrefixSearch`: the beam will be filled to `width`
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

    This module is called with the following signature:

        search(logits, logit_lens=None, prev=dict())

    where `logits` is a tensor of shape ``(T, N, V + 1)`` s.t. ``logits[t, n]``
    represents the unnormalized log-probabilities over the extended vocabulary
    (including blanks) at step ``t`` of batch element ``n``. The blank type logits are
    assumed to be stored in the final index of the vocabulary: ``logits[..., V]``.
    `logit_lens` is an optional tensor of shape ``(N,)`` s.t., for a given batch index
    ``n``, only the values in the slice ``logits[:lens[n], n]`` are valid. If
    `logit_lens` is not specified then all sequences are assumed to be of length ``T``.

    The call returns a triple of tensors ``y, y_lens, y_probs``. ``y`` is a long tensor
    of shape ``(S, N, width)`` containing the `width` prefixes per batch element, ``S <=
    T``. `y_lens` is a long tensor of shape ``(N, width)`` of the lengths of the
    corresponding prefixes: for each batch element ``n`` and prefix ``k``, only the
    tokens ``y[:y_lens[n, k], n, k]`` are valid. `y_probs` is a tensor of shape ``(N,
    width)`` containing those prefix's etimated (not log) probabilities. Note that for
    all ``k``, ``y_lens[n, k] <= logit_lens[n]``. Prefixes are ordered in decreasing
    probability (``y_probs[n, k] >= y_probs[n, k + 1]``).

    Shallow fusion [gulcehre2015]_ is enabled by initializing this module with `lm`.
    Shallow fusion updates the probability of extending a prefix :math:`y_{1..t-1}` with
    a new token math:`v` (:math:`v` is not blank) with the following equation

    .. math::
        \log S(y_t=v|y_{1..t-1}) = \log P_{logits}(y_t=v) +
                                                \beta \log P_{lm}(y_t = v|y_{1..t-1})

    The resulting value :math:`log S(y_t=v)` is not technically a probability. If the
    LM needs an initial input, it can be passed with the optional argument `prev`.

    Parameters
    ----------
    width : int
        The number of prefixes to keep track of per step.
    beta : float, optional
        The mixing coefficient :math:`\beta` used when performing shallow fusion.
    lm : MixableSequentialLanguageModel or None, optional
        If set, the language model used in shallow fusion. Specifying `lm` will
        restrict the extended vocabulary size of `logits` to be one more than that
        of `lm`: ``lm.vocab_size == V``.

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

    if TYPE_CHECKING:

        def forward(
            self,
            logits: torch.Tensor,
            lens: Optional[torch.Tensor] = None,
            prev: Dict[str, torch.Tensor] = dict(),
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            pass

    else:

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
                raise RuntimeError(
                    f"expected dim 0 of lens to be {N}, got {lens.size(0)}"
                )
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
                    lm_probs_t = (
                        (self.beta * lm_log_probs_t).exp().view(N, prev_width, V)
                    )
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
                    y_next[:-1] = torch.where(
                        valid_mask.unsqueeze(0), y_next[:-1], y_prev
                    )
                    y_prev_lens = torch.where(valid_mask, y_next_lens, y_prev_lens)
                    if prev_width < self.width:
                        assert prev_width == 1  # otherwise advance would've padded it
                        # add invalid path probs rather than broadcast the one good one
                        neg_inf = nb_probs_prev.new_full(
                            (N, self.width - prev_width), -float("inf")
                        )
                        nb_probs_prev = torch.cat([nb_probs_prev, neg_inf], 1)
                        b_probs_prev = torch.cat([b_probs_prev, neg_inf], 1)
                    nb_probs_prev = torch.where(
                        valid_mask, nb_probs_next, nb_probs_prev
                    )
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
                        probs_prev.new_full(
                            (N, self.width - prev_width), -float("inf")
                        ),
                    ],
                    1,
                )
            # now we zero out the probabilities of duplicate paths which could've arisen
            return y_prev, y_prev_lens, probs_prev


def random_walk_advance(
    logits_t: torch.Tensor,
    num_samp: int,
    y_prev: Optional[torch.Tensor] = None,
    eos: int = config.INDEX_PAD_VALUE,
    lens: Optional[torch.Tensor] = None,
    prevent_eos: bool = False,
    include_relaxation: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
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

    Warnings
    --------
    This function is not safe for JIT tracing or scripting.

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
    eos_mask: Optional[torch.Tensor] = None
    if y_prev is not None:
        if y_prev.dim() == 2:
            y_prev = y_prev.unsqueeze(2)
        if y_prev.shape[1:] != logits_t.shape[:-1]:
            raise RuntimeError(
                "If logits_t of shape {} then y_prev must have shape "
                "(*, {}, {})".format(
                    (batch_size, old_samp, num_classes), batch_size, old_samp,
                )
            )
        y_prev = y_prev.expand(-1, -1, num_samp)
        eos_mask = y_prev[-1].eq(eos)
        if eos_mask.any():
            eos_mask = eos_mask[..., :num_samp]
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


if TYPE_CHECKING:

    def sequence_log_probs(
        logits: Union[torch.Tensor, torch.nn.utils.rnn.PackedSequence],
        hyp: torch.Tensor,
        dim: int = 0,
        eos: Optional[int] = None,
    ) -> torch.Tensor:
        """Functional version of SequentialLogProbabilities

        Parameters
        ----------
        logits : torch.Tensor or torch.nn.utils.rnn.PackedSequence
        hyp : torch.Tensor
        dim : int, optional
        eos : int or `None`, optional

        See Also
        --------
        pydrobert.torch.layers.SequentialLogProbabilities
            For more details about the parameters
        """
        pass


else:

    def sequence_log_probs(
        logits: Any, hyp: torch.Tensor, dim: int = 0, eos: Optional[int] = None,
    ) -> torch.Tensor:
        hyp_dim = hyp.dim()
        if dim < -hyp_dim or dim > hyp_dim - 1:
            raise RuntimeError(
                "Dimension out of range (expected to be in range of [{}, {}], but "
                "got {})".format(-hyp_dim, hyp_dim - 1, dim)
            )
        if isinstance(logits, torch.Tensor):
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
        elif jit_isinstance(
            logits,
            Tuple[
                torch.Tensor,
                torch.Tensor,
                Optional[torch.Tensor],
                Optional[torch.Tensor],
            ],
        ):
            logits, batch_sizes, sidxs, uidxs = logits
            if sidxs is not None:
                hyp = torch.index_select(hyp, 1 - dim, sidxs)  # sort hyp
            lens = (
                (torch.arange(hyp.size(1 - dim)).unsqueeze(1) < batch_sizes)
                .to(torch.long)
                .sum(1)
            )
            hyp = torch.nn.utils.rnn.pack_padded_sequence(
                hyp, lens, batch_first=bool(dim)
            )[0]
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
        raise RuntimeError("logits must be either a Tensor or PackedSequence")


class SequenceLogProbabilities(torch.nn.Module):
    r"""Calculate joint log probability of sequences

    Once initialized, this module is called with the signature::

        log_probs = sequence_log_probs(logits, hyp)

    `logits` is a tensor of shape ``(..., steps, ..., num_classes)`` where ``steps``
    enumerates the time/step `dim`-th dimension. `hyp` is a long tensor of shape
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

    The resulting tensor `log_probs` is matches the shape of `logits` or `hyp` without
    the ``step`` and ``num_classes`` dimensions.

    Any values of `hyp` not in ``[0, num_classes)`` will be considered padding and
    ignored.

    If `eos` (end-of-sentence) is set, the first occurrence at :math:`b,t` is included
    in the sequence, but all :math:`b,>t` are ignored.
    
    `logits` may instead be a :class:`torch.nn.utils.rnn.PackedSequence`, though `hyp`
    must remain a tensor. `eos` is ignored in this case.

    Parameters
    ----------
    dim : int, optional
    eos : int or :obj:`None`, optional

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

    if TYPE_CHECKING:

        def forward(
            self,
            logits: Union[torch.Tensor, torch.nn.utils.rnn.PackedSequence],
            hyp: torch.Tensor,
        ) -> torch.Tensor:
            pass

    else:

        def forward(self, logits: Any, hyp: torch.Tensor) -> torch.Tensor:
            return sequence_log_probs(logits, hyp, self.dim, self.eos)
