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

import abc
import warnings

from typing import Any, Dict, Optional, Sequence, Tuple, Union, overload, List
from bisect import insort

import torch
import numpy as np

from . import argcheck
from ._compat import script
from ._wrappers import proxy

ProbDicts = List[
    Dict[Union[np.signedinteger, Tuple[np.signedinteger, ...]], np.floating]
]


class SequentialLanguageModel(torch.nn.Module, metaclass=abc.ABCMeta):
    r"""A language model whose sequence probability is built sequentially

    A language model provides the (log-)probability of a sequence of tokens. A
    sequential language model assumes that the probability distribution can be factored
    into a product of probabilities of the current token given the prior sequence, i.e.
    for token sequence :math:`\{w_s\}`

    .. math::

        P(w) = \prod_{s=1}^S P(w_s | w_{s - 1}, w_{s - 2}, \ldots w_1)

    This definition includes statistical language models, such as n-grams, where the
    probability of the current token is based only on a fixed-length history, as well as
    recurrent neural language models [mikolov2010]_.

    Parameters
    ----------
    vocab_size
        The vocabulary size. Controls the size of the final output dimension,
        as well as what values of `hist` are considered in-vocabulary

    Call Parameters
    ---------------
    hist : torch.Tensor
        A long tensor of shape ``(S, N)`` where ``S`` is the sequence dimension and
        ``N`` is the batch dimension. ``hist[:, n]`` is the n-th token prefix
        :math:`(w^{(n)}_0, w^{(n)}_1, \ldots, w^{(n)}_{S-1})`.
    prev : Dict[str, torch.Tensor], optional
        A dictionary of tensors which represents some additional state information which
        can be used in the computation. It may contain static input (e.g. a tensor of
        encoder output in neural machine translation) and/or dynamic input from prior
        calls to the LM (e.g. the previous hidden state in an RNN-based language model).
    idx : Optional[Union[int, torch.Tensor]], optional
        If specified, it is either a single integer or a long tensor of shape ``(N,)``
        specifying the indices of the tokens with which to return a distribution over.
        See the return value below.

    Returns
    -------
    log_probs : torch.Tensor or tuple of torch.Tensor
        The return value changes depending on whether `idx` was specified.

        If `idx` was not specified, the distributions over the next token over all
        prefixes in `hist` are returned. `log_probs` is a tensor of shape ``(S + 1, N,
        vocab_size)`` where each ``log_probs[s, n, v]`` equals :math:`\log P(w^{(n)}_{s}
        = v | w^{(n)}_{s - 1}, \ldots)`. That is, each distribution over types
        conditioned on each prefix of tokens (``:0``, ``:1``, ``:2``, etc.) is returned.

        If `idx` was specified, the distributions over only the token at those indices
        are returned. `log_probs` is a pair of tensors ``log_probs_idx, next_``.
        `log_probs_idx` is of shape ``(N, vocab_size)`` and ``log_probs[n, v]`` equals
        :math:`\log P(w^{(n)}_{idx[n]} = v | w^{(n)}_{idx[n]-1}, \ldots)`. That is, the
        distributions over the next type conditioned on token prefixes up to and
        excluding ``s = idx``. `next_` is a dictionary of tensors representing the
        updated state of the language model after computing these log probabilities,
        assuming `prev` represented the state at ``idx - 1``.

    Notes
    -----
    When this module is scripted, its return type will be :class:`typing.Any`. This
    reflects the fact that either `warped` is returned on its own (a tensor) or both
    `warped_` and `flow` (a tuple). Use :func:`torch.jit.isinstance` for type refinement
    in subsequent scripting. Tracing will infer the correct type. Alternatively, one can
    use the methods :func:`update_input`, :func:`calc_idx_log_probs`, and
    :func:`calc_full_log_probs` to avoid ambiguity in the return type altogether.

    This module has changed considerably since version 0.3.0. The primary changes are a)
    to replace the boolean switch `full` with `idx`; b) the inclusion of the `prev`
    argument for shared computations; c) the removal of `eos`, `sos`, and `oov`
    attributes; and d) replacing the more general signature of `hist`, ``(S, *)``, with
    ``(S, N)``. The former is strictly more powerful: the functionality of ``full=True``
    is replicated by setting ``idx=None`` and ``full=False`` by setting ``idx=-1``. The
    added functionality is intended to facilitate CTC decoding where prefixes stored in
    `hist` may be of different lengths. b) generalizes LMs by allowing additional input
    while also speeding up iterative computations. The removal of the `eos` and `sos`
    was due to a lack of generalizability. `oov` was removed because the user probably
    has to handle OOVs on her own when computing the loss.

    See Also
    --------
    Language Modelling and Decoding
        For a tutorial on how to build and use a language model.
    """

    __constants__ = ("vocab_size",)

    vocab_size: int

    def __init__(self, vocab_size: int):
        vocab_size = argcheck.is_posi(vocab_size, "vocab_size")
        super().__init__()
        self.vocab_size = vocab_size

    @torch.jit.export
    def update_input(
        self, prev: Dict[str, torch.Tensor], hist: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Update whatever is passed in as input to the language model

        Parameters
        ----------
        prev
            The initial `prev` dictionary passed prior to calculating any log
            probabilities.
        hist
            The initial `hist` tensor passed prior to calculating any log probabilites.

        Returns
        -------
        prev_ : Dict[str, torch.Tensor]
            The updated `prev`, populated with any additional information necessary to
            calculating log probabilities.

        Warnings
        --------
        This method should be robust to repeated calls prior to computing log
        probabilities. That is, the result of ``update_input(prev, hist)`` should
        be the same as ``update_input(update_input(prev, hist), hist)``.
        """
        return prev

    def extra_repr(self) -> str:
        s = "vocab_size={}".format(self.vocab_size)
        return s

    @abc.abstractmethod
    def calc_idx_log_probs(
        self,
        hist: torch.Tensor,
        prev: Dict[str, torch.Tensor],
        idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculates log_prob_idx over types at prefix up to and excluding idx

        Implements the :func:`forward` call when `idx` is specified. See the class
        description for more information on the parameters and returns. Note that `idx`
        is guaranteed to be a tensor, either of shape ``(,)`` (scalar) or ``(N,)``, with
        values in the range ``[0, hist.size(0)]``. `prev` can also be assumed to have
        been initialized using :func:`update_input` when the index is zero.

        Parameters
        ----------
        hist
        prev
        idx

        Returns
        -------
        log_probs_idx : torch.Tensor
        next_ : Dict[str, torch.Tensor]
        """
        raise NotImplementedError()

    @torch.jit.export
    def calc_full_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculates log_prob over all prefixes

        Implements the :func:`forward` call when `idx` is not specified. See the class
        description for more information on the parameters and returns. `prev` can be
        assumed to have been initialized using :func:`update_input`.

        Parameters
        ----------
        hist
        prev

        Returns
        -------
        log_probs : torch.Tensor
        """
        log_probs = []
        for idx in torch.arange(hist.size(0) + 1, device=hist.device):
            log_probs_idx, prev = self.calc_idx_log_probs(hist, prev, idx)
            log_probs.append(log_probs_idx)
        return torch.stack(log_probs, 0)

    @overload
    def forward(
        self,
        hist: torch.Tensor,
        prev: Dict[str, torch.Tensor] = dict(),
        idx: Optional[Union[int, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        ...

    def forward(
        self,
        hist: torch.Tensor,
        prev: Optional[Dict[str, torch.Tensor]] = None,
        idx: Optional[Any] = None,
    ) -> Any:
        if prev is None:
            prev = dict()
        if hist.dim() != 2:
            raise RuntimeError("hist must be 2 dimensional")
        S, N = hist.shape
        idx_ = torch.empty(0)
        if idx is not None:
            if isinstance(idx, int):
                idx_ = torch.as_tensor(idx, dtype=torch.long, device=hist.device)
            elif isinstance(idx, torch.Tensor):
                idx_ = idx
            if not idx_.numel():
                raise RuntimeError("idx_ must be at least one element")
            if idx_.dim() == 1:
                if idx_.size(0) == 1:
                    idx_ = idx_.squeeze(0)
                elif idx_.size(0) != N:
                    raise RuntimeError(
                        f"Expected dim 0 of idx_ to be of size {N}, got {idx_.size(0)}"
                    )
            if ((idx_ < -S - 1) | (idx_ > S)).any():
                raise RuntimeError(
                    f"All values in idx_ must be between ({-S - 1}, {S})"
                )
            idx_ = (idx_ + S + 1) % (S + 1)
        prev = self.update_input(prev, hist)
        if idx is None:
            return self.calc_full_log_probs(hist, prev)
        else:
            return self.calc_idx_log_probs(hist, prev, idx_)


class ExtractableSequentialLanguageModel(
    SequentialLanguageModel, metaclass=abc.ABCMeta
):
    """A SequentialLanguageModel whose prev values can be reordered on the batch idx

    :class:`SequentialLanguageModel` calls are on batched histories of paths `hist`. A
    :class:`SequentialLanguageModel` which is also a
    :class:`ExtractableSequentialLanguageModel` promises that, were we to rearrange
    and/or choose only some of those batch elements in `hist` to continue computations
    with, we can call the model's :func:`extract_by_src` method to rearrange/extract the
    relevant values in `prev` or `next_` in the same way.
    """

    @abc.abstractmethod
    def extract_by_src(
        self, prev: Dict[str, torch.Tensor], src: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Replace values in prev with those indexed in src

        Assume the values in the path history `hist` of shape ``(S, N_old)`` have been
        transformed into `new_hist` of shape ``(S, N_new)`` according to the mapping
        ``new_hist[s, n] = hist[s, src[n]]``. This method should apply the same
        transformation to the contents of `prev` and return that dictionary.

        Parameters
        ----------
        prev
            An input/output value for a step of the language model.
        src
            A tensor of shape ``(N,)`` containing the indices of the old batch index
            (of possibly different size) to extract the new batch elements from.

        Returns
        -------
        new_prev : Dict[str, torch.Tensor]

        Examples
        --------
        If we have an LSTM-based model and ``prev = {'hidden_state' : h, 'cell_state'
        : c}`` for a hidden state tensor `h` and cell state tensor `c` both of shape
        ``(N_old, H)``, then the return value of this method would be computed as

        >>> return {
        ...     'hidden_state': prev['hidden_state'].gather(0, src),
        ...     'cell_state': prev['cell_state'].gather(0, src),
        ... }
        """
        raise NotImplementedError()


class MixableSequentialLanguageModel(
    ExtractableSequentialLanguageModel, metaclass=abc.ABCMeta
):
    """An ExtractableSequentialLanguageModel whose prev values can be mixed

    In addition to the functionality of :class:`ExtractableSequentialLanguageModel`, a
    :class:`MixableSequentialLanguageModel` can also account for transformations from
    pairs of histories `hist_a` and `hist_b` into one `new_hist` such that each path in
    the latter is either from `hist_a` or `hist_b`. :func:`mix_by_mask` accomplishes
    this for the dictionaries `prev` and `in_next`.
    """

    @abc.abstractmethod
    def mix_by_mask(
        self,
        prev_true: Dict[str, torch.Tensor],
        prev_false: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Populate a new prev by picking values from either of two others

        Assume we have three batched path history tensors `hist_true`, `hist_false`, and
        `hist_new` each of shape ``(S, N)``. We're also assuming that if the sequences
        in each are of different lengths, we've also padded them appropriately.
        ``hist_new[:, n] = hist_true[:, n]`` when ``mask[n] == True`` and ``hist_new[:,
        n] = hist_false[:, n]`` otherwise. This method should apply the same
        transformation between `prev_true` and `prev_false` to come up with `prev_new`.

        Parameters
        ----------
        prev_true
            The input/output dictionary for the true branch of `mask`
        prev_false
            The input/output dictionary for the false branch of `mask`
        mask
            A boolean tensor of shape ``(N,)``

        Returns
        -------
        prev_new : dict

        Examples
        --------
        Continuing with the LSTM example from
        :class:`ExtractableSequentialLanguageModel`, the hidden states and cell states
        of the LSTM should always be the same size regardless of the remaining history,
        making the implementation trivial:

        >>> return {
        ...     'hidden_state': torch.where(
        ...         mask.unsqueeze(1),
        ...         prev_true['hidden_state'],
        ...         prev_false['hidden_state']),
        ...     'cell_state': torch.where(
        ...         mask.unsqueeze(1),
        ...         prev_true['cell_state'],
        ...         prev_false['cell_state']),
        ... }
        """
        raise NotImplementedError()


@script
def _lookup_calc_idx_log_probs(
    hist: torch.Tensor,
    idx: torch.Tensor,
    pointers: torch.Tensor,
    ids: torch.Tensor,
    logs: torch.Tensor,
    shift: int,
    sos: int,
    V: int,
    N: int,
    G: int,
) -> torch.Tensor:
    # we produce two tries with the same node ids: one for logp and one for
    # logb. Let N be the maximal n-gram. The children of the root are
    # 1-grams, their children are 2-grams, etc. Thus, x-gram is synonymous
    # for level x of the trie. The logb trie does not have N-gram children
    # b/c there are no backoffs for the maximal n-gram.
    #
    # pointers is a flattened array of size X of pointers of internal
    # nodes. They are only populated when N > 1. pointers is arranged in
    # a breadth-first manner: levels = [
    #   1-grams + 1; 2-grams + 1; ...; (N - 1)-grams + 1]
    # pointers contain positive offsets from their current node to the
    # first index of its children. The immediately subsequent pointer is
    # the exclusive offset to the end of the range of children; if the
    # values of the pointer and subsequent pointer are equal, the node has
    # no children. The subsequent pointer is either the inclusive offset
    # of the start of a sibling's children, or a dummy pointer (the +1s
    # above) for the final child in a level.
    #
    # ids = [2-grams + 1; ...; N-grams], that is, remove the 1-grams
    # level from pointers and add the N-grams level. Thus, to convert from
    # a pointers index to an ids index, one need only subtract U
    # (vocab_size + shift + 1 % N). id values correspond to the last token
    # in a reverse n-gram produced by the path through the tree so far.
    #
    # logs = [
    #   1-grams + 1; 2-grams + 1; ...; N-grams;
    #   1-grams + 1; 2-grams + 1; ...; (N-1)-grams]. The first X values
    # are the log-probabilities. Letting G be the number of N-gram nodes,
    # the remaining X - G entries are the backoff probabilities
    B: int = hist.size(1)
    M, X = B * V, pointers.numel()
    U = V + shift + (1 % N)
    K, L = X + G - U, 2 * X + G
    device = hist.device
    assert ids.numel() == K
    assert logs.numel() == L
    if idx.numel() == 0:
        raise RuntimeError("idx cannot be empty")
    if idx.numel() == 1:
        hist = hist[:idx]
        if idx >= N - 1:
            hist = hist[hist.size(0) - (N - 1) :]
        else:
            hist = torch.cat(
                [
                    torch.full(
                        (N - 1 - hist.size(0), B),
                        sos,
                        dtype=torch.long,
                        device=device,
                    ),
                    hist,
                ],
                0,
            )
    else:
        min_idx = int(idx.min().item())  # parent ensures min_idx >=0
        if min_idx < N - 1:
            hist = torch.cat(
                [
                    torch.full(
                        (N - 1 - min_idx, B),
                        sos,
                        dtype=torch.long,
                        device=device,
                    ),
                    hist,
                ],
                0,
            )
            idx = idx + N - 1 - min_idx
        idx = torch.arange(-N + 1, 0, 1, device=idx.device).unsqueeze(1) + idx
        hist = hist.gather(0, idx)
    assert hist.size(0) == N - 1
    if shift:
        hist = hist.masked_fill(hist.eq(sos), V)
        # hist = hist + shift
    hist = hist.to(ids.dtype)

    # add the possible extensions to the history
    cur_step = vrange = torch.arange(V, dtype=torch.long, device=device)
    cur_step = cur_step.view(1, 1, V).expand(1, B, V)
    hist = torch.cat([hist.unsqueeze(2).expand(N - 1, B, V), cur_step], 0)

    if N == 1:
        # we're a unigram model, or we've only got unigram history
        logs_t = logs[:G].unsqueeze(0).expand(B, G)
        return logs_t.gather(1, hist[-1])  # (B, V)

    # we're now definitely not a unigram model w/ non-empty history
    hist = hist.view(-1, M)  # pretend M is batch; reshape at end
    out = torch.zeros(M, dtype=torch.float, device=device)
    running_mask = torch.full(out.shape, 1, dtype=torch.bool, device=device)
    children = tokens = hist[0]
    for Nn in range(N - 1):
        n = N - Nn
        offsets = pointers[children]  # (M,)
        # the +1 is because we've shifted over one, meaning the offset
        # pointing to the same location is one less
        num_children = pointers[children + 1] - offsets + 1  # (N,)
        first_children = children + offsets
        step_mask = running_mask
        for t in range(1, n):
            tokens = hist[Nn + t]
            # the max avoids working with empty tensors
            S = max(1, int(num_children.max().item()))
            all_children = first_children.unsqueeze(1) + vrange[:S].unsqueeze(0)
            matches = (
                (ids[all_children.clamp(max=K + U - 1) - U] == tokens.unsqueeze(1))
                & (vrange[:S].unsqueeze(0) < num_children.unsqueeze(1))
                & step_mask.unsqueeze(1)
            )
            next_step = matches.any(1)
            if t == n - 1:
                # we're last. Add probabilities
                logs_t = torch.where(
                    matches,
                    logs[all_children],
                    torch.zeros(
                        all_children.shape,
                        dtype=logs.dtype,
                        device=device,
                    ),
                ).sum(
                    1
                )  # (M,)
                # the trie has dummy lower-order n-grams. If there's
                # an (n+1) gram passing through it. We do not want to
                # match these - we will back off further
                finite = torch.isfinite(logs_t)
                out = torch.where(finite, out + logs_t, out)
                next_step = next_step & finite
                running_mask = running_mask & next_step.eq(0)
                new_backoff = step_mask & next_step.eq(0)
                # add backoff for newly failed paths
                out = torch.where(
                    new_backoff,
                    out + logs[X + G + children],
                    out,
                )
            else:
                # we're not last. Update children
                children = torch.where(
                    matches,
                    all_children,
                    torch.zeros(
                        all_children.shape,
                        dtype=all_children.dtype,
                        device=all_children.device,
                    ),
                ).sum(1)
                offsets = pointers[children]
                num_children = pointers[children + 1] - offsets + 1
                first_children = children + offsets
            step_mask = next_step
        children = tokens = hist[Nn + 1]
    # unigrams always exist. Add the log-probability and exit
    out = torch.where(running_mask, out + logs[tokens], out)
    return out.view(B, V)


class LookupLanguageModel(MixableSequentialLanguageModel):
    r"""Construct a backoff n-gram model from a fixed lookup table

    An instance of this model will search for a stored log-probability of the current
    token given a fixed-length history in a lookup table. If it can't find it, it backs
    off to a shorter length history and incurs a penalty:

    .. math::

        Pr(w_t|w_{t-1},\ldots,w_{t-(N-1)}) = \begin{cases}
            Entry(w_{t-(N-1)}, w_{t-(N-1)+1}, \ldots, w_t)
                & \text{if } Entry(w_{t-(N-1)}, \ldots) > 0 \\
            Backoff(w_{t-(N-1)}, \ldots, w_{t-1})
            Pr(w_t|w_{t-1},\ldots,w_{t-(N-1)+1}) & \text{else}
        \end{cases}

    Missing entries are assumed to have value 0 and missing backoff penalties are
    assumed to have value 1.

    Parameters
    ----------
    vocab_size
    sos
        The start of sequence token. If specified, any prefix with fewer tokens than the
        maximum order of n-grams minus 1 will be prepended up to that length with this
        token.
    prob_dicts
        A list of dictionaries whose entry at index ``i`` corresponds to a table of
        ``i+1``-gram probabilities. Keys must all be ids, not strings. Unigram keys are
        just ids; for n > 1 keys are tuples of ids with the latest word last. Values in
        the dictionary of the highest order n-gram dictionaries (last in `prob_dicts`)
        are the log-probabilities of the keys. Lower order dictionaries' values are
        pairs of log-probability and log-backoff penalty. If `prob_dicts` is not
        specified, a unigram model with a uniform prior will be built
    destructive
        If :obj:`True`, allows initialization to modify `prob_dicts` directly instead
        of making a fresh copy. Doing so can help reduce memory pressure
    
    Call Parameters
    ---------------
    hist : torch.Tensor
    prev : Dict[str, torch.Tensor], optional
    idx : Optional[Union[int, torch.Tensor]], optional

    Returns
    -------
    log_probs : torch.Tensor or tuple of torch.Tensor

    Notes
    -----
    Initializing an instance from an `prob_dicts` is expensive. `prob_dicts` is converted
    to a trie (something like [heafield2011]_) so that it takes up less space in memory,
    which can take some time.

    Rather than re-initializing repeatedly, it is recommended you save and load this
    module's state dict. :func:`load_state_dict` as been overridden to support loading
    different table sizes, avoiding the need for an accurate `prob_dicts` on
    initialization:

    >>> # first time
    >>> lm = LookupLanguageModel(vocab_size, sos, prob_dicts)  # slow
    >>> state_dict = lm.state_dict()
    >>> # save state dict, quit, startup, then reload state dict
    >>> lm = LookupLanguageModel(vocab_size, sos)  # fast!
    >>> lm.load_state_dict(state_dict)

    See Also
    --------
    pydrobert.util.parse_arpa_lm
        How to read a pretrained table of n-gram probabilities into
        `prob_dicts`. The parameter `token2id` should be specified to ensure
        id-based keys.

    Warnings
    --------
    After 0.3.0, `prob_list` was renamed to `prob_dicts`. `prob_list` is deprecated

    After 0.3.0, `sos` became no longer optional. `pad_sos_to_n` was removed as an
    argument (implicitly true now). `eos` and `oov` were also removed as part of updates
    to :obj:`SequentialLanguageModel`

    JIT scripting is possible with this module, but not tracing.
    """

    __constants__ = ("vocab_size", "sos", "max_ngram", "max_ngram_nodes", "shift")

    sos: int
    max_ngram: int
    max_ngram_nodes: int
    shift: int

    # XXX(sdrobert): as discussed in [heafield2011], we could potentially speed
    # up computations by keeping track of prefix probs and storing them in
    # case of backoff. This makes sense in a serial case, when we can choose to
    # explore or not explore a path. In a massively parallel environment, I'm
    # not sure it's worth the effort...

    @overload
    def __init__(
        self,
        vocab_size: int,
        sos: int,
        prob_dicts: Optional[ProbDicts] = None,
        destructive: bool = False,
    ):
        ...

    def __init__(
        self,
        vocab_size: int,
        sos: int,
        prob_dicts: Optional[ProbDicts] = None,
        destructive: bool = False,
        *,
        prob_list: Optional[ProbDicts] = None,
    ):
        sos = argcheck.is_int(sos, "sos")
        destructive = argcheck.is_bool(destructive)
        if prob_list is not None:
            if prob_dicts is None:
                warnings.warn(
                    "prob_list has been renamed to prob_dicts", DeprecationWarning
                )
                prob_dicts = prob_list
            else:
                raise ValueError(
                    "prob_list and prob_dicts cannot be specified simultaneously"
                )
        super().__init__(vocab_size)
        self.sos = sos
        if sos < 0 or sos > vocab_size:
            # we want sos to refer to an index but it's oov, so we'll shift all
            # indices in hyp up by one and fill the occurrences of sos with 0
            self.shift = 1
        else:
            self.shift = 0
        if prob_dicts is None:
            logs = -torch.full(
                (self.shift + vocab_size,), vocab_size, dtype=torch.float
            ).log()
            ids = pointers = torch.tensor([], dtype=torch.uint8)
            self.max_ngram = 1
            self.max_ngram_nodes = self.shift + vocab_size
        else:
            self.max_ngram = len(prob_dicts)
            self.max_ngram_nodes = -1  # changed by build_trie
            logs, ids, pointers = self._build_trie(prob_dicts, destructive)
        self.register_buffer("logs", logs)
        self.register_buffer("ids", ids)
        self.register_buffer("pointers", pointers)

    def extra_repr(self) -> str:
        s = super(LookupLanguageModel, self).extra_repr()
        s += ", max_ngram={}, sos={}".format(self.max_ngram, self.sos)
        return s

    @torch.jit.export
    def extract_by_src(
        self, prev: Dict[str, torch.Tensor], src: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return dict()

    @torch.jit.export
    def mix_by_mask(
        self,
        prev_true: Dict[str, torch.Tensor],
        prev_false: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return dict()

    def calc_idx_log_probs(
        self,
        hist: torch.Tensor,
        prev: Dict[str, torch.Tensor],
        idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return (
            _lookup_calc_idx_log_probs(
                hist,
                idx,
                self.pointers,
                self.ids,
                self.logs,
                self.shift,
                self.sos,
                self.vocab_size,
                self.max_ngram,
                self.max_ngram_nodes,
            ),
            prev,
        )

    def load_state_dict(self, state_dict: dict, **kwargs) -> None:
        error_prefix = "Error(s) in loading state_dict for {}:\n".format(
            self.__class__.__name__
        )
        missing_keys = {"pointers", "ids", "logs"} - set(state_dict)
        if missing_keys:
            raise RuntimeError(
                'Missing key(s) in state_dict: "{}".'.format('", "'.join(missing_keys))
            )
        pointers = state_dict["pointers"]
        ids = state_dict["ids"]
        logs = state_dict["logs"]
        if len(ids) and len(pointers):
            # n > 1
            if len(pointers) < self.vocab_size + self.shift + 1:
                raise RuntimeError(
                    error_prefix + "Expected {} unigram probabilities, got {} "
                    "(vocab_size and sos must be correct!)".format(
                        self.vocab_size + self.shift, len(pointers) - 1
                    )
                )
            X, K, L = len(pointers), len(ids), len(logs)
            U = self.vocab_size + self.shift + 1
            self.max_ngram = 1
            self.max_ngram_nodes = last_ptr = U - 1
            error = RuntimeError(
                error_prefix + "buffer contains unexpected value (are you sure "
                "you've set vocab_size and sos correctly?)"
            )
            while last_ptr < len(pointers):
                offset = pointers[last_ptr].item()
                if offset <= 0:
                    raise error
                last_ptr += offset
                self.max_ngram_nodes = offset - 1
                self.max_ngram += 1
            # last_ptr should be X + G
            if (last_ptr != K + U) or (last_ptr != L - X):
                raise RuntimeError(error_prefix + "Unexpected buffer length")
        else:  # n == 1
            if len(pointers) != len(ids):
                raise RuntimeError(error_prefix + "Incompatible trie buffers")
            if len(logs) != self.vocab_size + self.shift:
                raise RuntimeError(
                    error_prefix + "Expected {} unigram probabilities, got {} "
                    "(vocab_size and sos must be correct!)"
                    "".format(self.vocab_size + self.shift, len(logs))
                )
            self.max_ngram_nodes = self.vocab_size + self.shift
            self.max_ngram = 1
        # resize
        self.pointers = torch.empty_like(pointers, device=self.pointers.device)
        self.ids = torch.empty_like(ids, device=self.ids.device)
        self.logs = torch.empty_like(logs, device=self.logs.device)
        return super(LookupLanguageModel, self).load_state_dict(state_dict, **kwargs)

    def _build_trie(self, prob_dicts: ProbDicts, destructive: bool):
        if not len(prob_dicts):
            raise ValueError("prob_dicts must contain at least unigrams")
        if not destructive:
            prob_dicts = [prob_dict.copy() for prob_dict in prob_dicts]
        total_entries, nan, inf = 0, float("nan"), float("inf")
        unigrams = set(range(self.vocab_size))
        if self.shift:
            unigrams.add(self.sos)
        for n in range(self.max_ngram - 1, -1, -1):
            prob_dict = prob_dicts[n]
            is_last = n == self.max_ngram - 1
            if is_last and not prob_dict:
                raise ValueError("Final element in prob_dicts must not be empty")
            if is_last:
                dummy_value = -inf
            else:
                dummy_value = -inf, 0.0
            if not n:
                keys = set(prob_dict.keys())
                if keys - unigrams:
                    raise ValueError(
                        "Unexpected unigrams in prob_dicts: {} (are these "
                        "ids?)".format(keys - unigrams)
                    )
                prob_dict.update((key, dummy_value) for key in unigrams - keys)
            else:
                for seq in prob_dict:
                    if len(seq) != n + 1:
                        raise ValueError(
                            "Key {0} in {1}-gram is not a sequence of length "
                            "{1}".format(n + 1, seq)
                        )
                    if set(seq) - unigrams:
                        raise ValueError(
                            "Unexpected tokens in {}-gram in prob_dicts: {} ("
                            "are these ids?)"
                            "".format(n + 1, set(seq) - unigrams)
                        )
                    prefix = seq[:-1]
                    if len(prefix) == 1:
                        prefix = prefix[0]
                    if prefix not in prob_dicts[n - 1]:
                        prob_dicts[n - 1][prefix] = -inf, 0.0
            total_entries += len(prob_dict)
            if is_last:
                self.max_ngram_nodes = len(prob_dict)
        if self.shift:
            prob_dicts[0][self.vocab_size] = prob_dicts[0].pop(self.sos)
            for n in range(1, self.max_ngram):
                sos_keys = []
                prob_dict = prob_dicts[n]
                for key in prob_dict.keys():
                    if self.sos in key:
                        sos_keys.append(key)
                while len(sos_keys):
                    key = sos_keys.pop()
                    key_ = tuple(self.vocab_size if k == self.sos else k for k in key)
                    prob_dict[key_] = prob_dict.pop(key)

        N, G, V = self.max_ngram, self.max_ngram_nodes, self.vocab_size
        U, X = V + self.shift + (1 % N), total_entries - G + (N - 1)
        K, L = X + G - U, 2 * X + G
        if N > 1:
            # what's the maximum possible offset? It's the maximal possible
            # distance between a parent and child, or an n-gram and an
            # (n+1)-gram. Let the former have S nodes in the level, the latter
            # T nodes. Let a, b, and c correspond to offsets of distinct paths
            # through the trie and x be the dummy offset. The longest offset in
            # pointers is produced as a value of b like this:
            #
            #   abcccc...cxaaaa...bx
            #
            # i.e. there are a lot of branches of a in (n+1) but only one
            # parent, and there are a lot of branches of c in n but no
            # descendants. The hop from b to x is of size S - 1, and the hop
            # from x to the next b is of size T, so the worst potential hop is
            # S + T - 1
            max_potential_offset = max(
                len(prob_dicts[n]) + len(prob_dicts[n - 1]) - 1 for n in range(1, N)
            )
        else:
            max_potential_offset = 0  # no descendants
        for pointer_type in (torch.uint8, torch.int16, torch.int32, torch.int64):
            if torch.iinfo(pointer_type).max >= max_potential_offset:
                break
        if torch.iinfo(pointer_type).max < max_potential_offset:
            # should not happen
            raise ValueError("too many childen")
        for id_type in (torch.uint8, torch.int16, torch.int32, torch.int64):
            if torch.iinfo(id_type).max >= U:
                break
        if torch.iinfo(id_type).max < U:
            # should never happen in a practical situation
            raise ValueError("vocab too large")
        pointers = torch.zeros(X, dtype=pointer_type)
        ids = torch.zeros(K, dtype=id_type)
        logs = torch.zeros(L, dtype=torch.float)
        prob_dict = prob_dicts.pop(0)
        unigram_values = [prob_dict[x] for x in range(U - 1 % N)]
        allocated = U - 1 % N
        if N == 1:
            logs.copy_(torch.tensor(unigram_values, dtype=torch.float))
        else:
            logs[:allocated].copy_(
                torch.tensor([x[0] for x in unigram_values], dtype=torch.float)
            )
            logs[X + G : X + G + allocated].copy_(
                torch.tensor([x[1] for x in unigram_values], dtype=torch.float)
            )
        del unigram_values
        parents = dict(((x,), x) for x in range(U - 1))
        N -= 1
        while N:
            prob_dict = prob_dicts.pop(0)
            start = allocated
            pointers[allocated] = len(prob_dict) + 1
            logs[allocated] = logs[X + G + allocated] = nan
            allocated += 1
            prob_list = []
            while prob_dict:
                insort(prob_list, prob_dict.popitem())
            children = dict()
            for key, value in prob_list:
                children[key] = allocated
                ids[allocated - U] = key[-1]
                if N == 1:
                    logs[allocated] = float(value)
                else:
                    logs[allocated] = float(value[0])
                    logs[allocated + X + G] = float(value[1])
                prefix = key[:-1]
                parent = parents[prefix]
                while parent >= 0 and not pointers[parent]:
                    pointers[parent] = allocated - parent
                    parent -= 1
                allocated += 1
            while not pointers[start - 1]:
                pointers[start - 1] = pointers[start] + 1
                start -= 1
            N -= 1
            parents = children
        assert allocated == L - X
        # see if we can shrink the pointer size
        if len(pointers):
            max_offset = pointers.max().item()
            for pointer_type in (torch.uint8, torch.int16, torch.int32, torch.int64):
                if torch.iinfo(pointer_type).max >= max_offset:
                    break
            pointers = pointers.to(pointer_type)
        return logs, ids, pointers

    __call__ = proxy(SequentialLanguageModel.forward)


class ShallowFusionLanguageModel(SequentialLanguageModel):
    r"""Language model combining two language models with shallow fusion

    Shallow fusion [gulcehre2015]_ combines the predictions of two language models
    by taking the weighted sum of their log probabilities:

    .. math::
        \log S(y_t=v|...) = \log P_{first}(y_t=v|...) +
                                \beta \log P_{second}(y_t = v|...)

    The resulting value :math:`log S(y_t=v)` is not technically a probability.

    Parameters
    ----------
    first
        The first language model
    second
        The second language model, whose log probabilities multiply with `beta`
    beta
        The value :math:`\beta`
    first_prefix
        Elements of the state dict for `first` will have `first_prefix` prepended to
        their keys
    second_prefix
        Like `first_prefix`, but for `second`

    Warnings
    --------
    This class does not (and cannot) support JIT.

    Notes
    -----
    If you intend to perform shallow fusion between CTC logits and an external language
    model, you will not be able to do so via this class. CTC operates on an extended
    vocabulary while an external language model does not. Fortunately,
    :class:`CTCPrefixSearch` has built-in support for shallow fusion. Consult that
    class for more information.

    See Also
    --------
    MixableShallowFusionModel
        A mixable subclass of this class. Applicable only if `first` and `second`
        are both :class:`MixableSequentialLanguageModel` instances.
    ExtractableShallowFusionModel
        An extractable subclass of this class. Applicable only if `first` and `second`
        are both :class:`ExtractableSequentialLanguageModel` instances.
    """

    __constants__ = "beta", "first_prefix", "second_prefix"
    first: SequentialLanguageModel
    second: SequentialLanguageModel
    beta: float
    first_prefix: str
    second_prefix: str

    def __init__(
        self,
        first: SequentialLanguageModel,
        second: SequentialLanguageModel,
        beta: float = 0.0,
        first_prefix: str = "first.",
        second_prefix: str = "second.",
    ):
        beta = argcheck.is_float(beta, "beta")
        first_prefix = argcheck.is_str(first_prefix, "first_prefix")
        second_prefix = argcheck.is_str(second_prefix, "second_prefix")
        if first.vocab_size != second.vocab_size:
            raise ValueError(
                f"first's vocab_size ({first.vocab_size}) differs from second's "
                f"vocab_size ({second.vocab_size})"
            )
        if not len(first_prefix) or not len(second_prefix):
            raise ValueError(f"prefixes cannot be empty")
        if first_prefix == second_prefix:
            raise ValueError(f"first_prefix matches second_prefix ('{first_prefix}')")
        super().__init__(first.vocab_size)
        self.first, self.second, self.beta = first, second, beta
        self.first_prefix, self.second_prefix = first_prefix, second_prefix

    def extra_repr(self) -> str:
        return super().extra_repr() + (
            f", beta={self.beta}, first_prefix='{self.first_prefix}'"
            f", second_prefix='{self.second_prefix}'"
            f", first={self.first}, second={self.second}"
        )

    @torch.jit.export
    def split_dicts(
        self, prev: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Split state dicts into state dicts for first and second lms"""
        prev_first: Dict[str, torch.Tensor] = dict()
        prev_second: Dict[str, torch.Tensor] = dict()
        for k, v in prev.items():
            if k.startswith(self.first_prefix):
                prev_first[k[len(self.first_prefix) :]] = v
            elif k.startswith(self.second_prefix):
                prev_second[k[len(self.second_prefix) :]] = v
            else:
                raise RuntimeError(
                    f"key '{k}' from prev does not start with first_prefix "
                    f"'{self.first_prefix}' nor second_prefix '{self.second_prefix}'"
                )
        return prev_first, prev_second

    @torch.jit.export
    def merge_dicts(
        self, prev_first: Dict[str, torch.Tensor], prev_second: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Merge state dicts from first and second lms into state dict"""
        prev: Dict[str, torch.Tensor] = dict()
        prev.update((self.first_prefix + k, v) for (k, v) in prev_first.items())
        prev.update((self.second_prefix + k, v) for (k, v) in prev_second.items())
        return prev

    @torch.jit.export
    def update_input(
        self, prev: Dict[str, torch.Tensor], hist: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        prev_first, prev_second = self.split_dicts(prev)
        prev_first = self.first.update_input(prev_first, hist)
        prev_second = self.second.update_input(prev_second, hist)
        return self.merge_dicts(prev_first, prev_second)

    @torch.jit.export
    def calc_idx_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor], idx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prev_first, prev_second = self.split_dicts(prev)
        log_probs_first, cur_first = self.first.calc_idx_log_probs(
            hist, prev_first, idx
        )
        log_probs_second, cur_second = self.second.calc_idx_log_probs(
            hist, prev_second, idx
        )
        log_probs = log_probs_first + self.beta * log_probs_second
        cur = self.merge_dicts(cur_first, cur_second)
        return log_probs, cur

    @torch.jit.export
    def calc_full_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        prev_first, prev_second = self.split_dicts(prev)
        log_probs_first = self.first.calc_full_log_probs(hist, prev_first)
        log_probs_second = self.second.calc_full_log_probs(hist, prev_second)
        return log_probs_first + self.beta * log_probs_second


class ExtractableShallowFusionLanguageModel(
    ShallowFusionLanguageModel, ExtractableSequentialLanguageModel
):
    """ShallowFusionLanguageModel which is also an ExtractableSequentialLanguageModel

    Both `first` and `second` must be :class:`ExtractableSequentialLanguageModel`
    instances.

    See Also
    --------
    ShallowFusionLanguageModel
        For a description of shallow fusion and parameters. `first` and `second` may
        not be extractable, but neither is :class:`ShallowFusionLanguageModel`.
    MixableShallowFusionModel
        A mixable subclass of this class. Applicable only if `first` and `second`
        are both :class:`MixableSequentialLanguageModel` instances.
    """

    first: ExtractableSequentialLanguageModel
    second: ExtractableSequentialLanguageModel

    def __init__(
        self,
        first: ExtractableSequentialLanguageModel,
        second: ExtractableSequentialLanguageModel,
        beta: float = 0,
        first_prefix: str = "first.",
        second_prefix: str = "second.",
    ):
        super().__init__(first, second, beta, first_prefix, second_prefix)

    @torch.jit.export
    def extract_by_src(
        self, prev: Dict[str, torch.Tensor], src: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        prev_first, prev_second = self.split_dicts(prev)
        prev_first = self.first.extract_by_src(prev_first, src)
        prev_second = self.second.extract_by_src(prev_second, src)
        return self.merge_dicts(prev_first, prev_second)


class MixableShallowFusionLanguageModel(
    ExtractableShallowFusionLanguageModel, MixableSequentialLanguageModel
):
    """ShallowFusionLanguageModel which is also a MixableSequentialLanguageModel

    Both `first` and `second` must be :class:`ExtractableSequentialLanguageModel`
    instances.

    See Also
    --------
    ShallowFusionLanguageModel
        For a description of shallow fusion and parameters. `first` and `second` may
        not be mixable, but neither is :class:`ShallowFusionLanguageModel`.
    ExtractableSequentialLanguageModel
        An extractable superclass of this class. Applicable if `first` and `second`
        are both :class:`ExtractableSequentialLanguageModel` instances.
    """

    first: MixableSequentialLanguageModel
    second: MixableSequentialLanguageModel

    def __init__(
        self,
        first: MixableSequentialLanguageModel,
        second: MixableSequentialLanguageModel,
        beta: float = 0,
        first_prefix: str = "first.",
        second_prefix: str = "second.",
    ):
        super().__init__(first, second, beta, first_prefix, second_prefix)

    @torch.jit.export
    def mix_by_mask(
        self,
        prev_true: Dict[str, torch.Tensor],
        prev_false: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        prev_first_true, prev_second_true = self.split_dicts(prev_true)
        prev_first_false, prev_second_false = self.split_dicts(prev_false)
        prev_first = self.first.mix_by_mask(prev_first_true, prev_first_false, mask)
        prev_second = self.second.mix_by_mask(prev_second_true, prev_second_false, mask)
        return self.merge_dicts(prev_first, prev_second)
