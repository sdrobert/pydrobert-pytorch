# Copyright 2020 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common neural layers from the literature not included in pytorch.nn

Notes
-----
The loss functions :class:`HardOptimalCompletionDistillationLoss` and
:class:`MinimumErrorRateLoss` have been moved here from
:mod:`pydrobert.torch.training`
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import torch

from pydrobert.torch.util import error_rate, optimal_completion, polyharmonic_spline
from future.utils import with_metaclass

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2020 Sean Robertson"
__all__ = [
    "ConcatSoftAttention",
    "DotProductSoftAttention",
    "GeneralizedDotProductSoftAttention",
    "GlobalSoftAttention",
    "HardOptimalCompletionDistillationLoss",
    "LookupLanguageModel",
    "MinimumErrorRateLoss",
    "MultiHeadedAttention",
    "SequentialLanguageModel",
    "SpecAugment",
]

# XXX(sdrobert): a quick note on style. pytorch doesn't tend to protect its
# read-only attributes using private members, so neither do we


class SequentialLanguageModel(with_metaclass(abc.ABCMeta, torch.nn.Module)):
    r"""A language model whose sequence probability is built sequentially

    A language model provides the (log-)probability of a sequence of tokens. A
    sequential language model assumes that the probability distribution can be
    factored into a product of probabilities of the current token given the
    prior sequence, i.e. for token sequence :math:`\{w_s\}`

    .. math::

        P(w) = \prod_{s=1}^S P(w_s | w_{s - 1}, w_{s - 2}, \ldots w_1)

    This definition includes statistical language models, such as n-grams,
    where the probability of the current token is based only on a fixed-length
    history, as well as recurrent neural language models [mikolov2010]_.

    Subclasses have the following signature:

        lm(hist, full=False)

    Where `hist` is a :class:`torch.LongTensor` of shape ``(s - 1, *)``
    corresponding to the sequence up to but excluding the current step ``s``,
    where ``s >= 1``. Letting ``i`` be a multi-index of all but the first
    dimension of `hist`, ``hist[:, i]`` is the i-th sequence :math:`(w^{(i)}_1,
    w^{(i)}_2, \ldots, w^{(i)}_{s - 1})`.

    When `full` is :obj:`False`, it outputs a :class:`torch.FloatTensor`
    `log_probs` of shape ``(*, vocab_size)``, where ``log_probs[i, v]`` equals
    :math:`\log P(w^{(i)}_s = v | w^{(i)}_{s-1}, \ldots)`

    When `full` is :obj:`True`, `log_probs` is of shape ``(s, *, vocab_size)``
    where each ``log_probs[s', i, v]`` equals :math:`\log P(w^{(i)}_{s'} = v |
    w^{(i)}_{s' - 1}, \ldots)`

    Parameters
    ----------
    vocab_size : int
        The vocabulary size. Controls the size of the final output dimension,
        as well as what values of `hist` are considered in-vocabulary
    sos : int, optional
        An optional start-of-sequence token. Setting this option will prepend
        `hist` with a tensor full of `sos`. `sos` can be in- or
        out-of-vocabulary. Setting `sos` does not change the size of the
        output, regardless of whether `full` is :obj:`True`: the prepended
        tensor is considered context for ``s' == 0``
    eos : int, optional
        An optional end-of-sequence token. If this token is found in `hist`,
        values succeeding it in `log_prob` will be replaced with zero. `eos`
        need not be in-vocabulary
    oov : int, optional
        An optional out-of-vocabulary token. If any elements of `hist` are not
        `eos` or not within the range ``[0, vocab_size)``, they will be
        replaced with this token. `oov` must be in-vocabulary itself

    Attributes
    ----------
    vocab_size : int
    sos : int or :obj:`None`
    eos : int or :obj:`None`
    oov : int or :obj:`None`
    """

    def __init__(self, vocab_size, sos=None, eos=None, oov=None):
        super(SequentialLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.sos = sos
        self.eos = eos
        self.oov = oov
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if sos is not None and sos == eos:
            raise ValueError("sos cannot equal eos")
        if self.oov is not None and (self.oov < 0 or self.oov >= vocab_size):
            raise ValueError("oov must be within [0, vocab_size)")

    def extra_repr(self):
        s = "vocab_size={}".format(self.vocab_size)
        if self.sos is not None:
            s += ", sos={}".format(self.sos)
        if self.eos is not None:
            s += ", eos={}".format(self.eos)
        if self.oov is not None:
            s += ", oov={}".format(self.oov)
        return s

    def check_input(self, hist, **kwargs):
        """Check if the input is formatted correctly, otherwise RuntimeError"""
        if hist.dim() < 2:
            raise RuntimeError("hist must be at least 2-D")
        if self.oov is None:
            oov_mask = kwargs.get("oov_mask", None)
            if oov_mask is None:
                oov_mask = hist.ge(self.vocab_size) | hist.lt(0)
                eos_mask = kwargs.get("eos_mask", None)
                if eos_mask is None and self.eos is not None:
                    eos_mask = hist.eq(self.eos).cumsum(0, dtype=torch.long)
                    eos_mask = eos_mask.ne(0)
                if eos_mask is not None:
                    oov_mask = oov_mask & eos_mask.eq(0)
            if kwargs.get("skip_first", False):
                assert self.sos is not None
                oov_mask = oov_mask[1:]
            if oov_mask.any():
                raise RuntimeError(
                    "Found values in hist that were not eos and not between "
                    "[0, {})".format(self.vocab_size)
                )

    @abc.abstractmethod
    def calc_last_log_probs(self, hist, eos_mask):
        """Calculate log probabilities conditioned on history

        Do not call this directly; instead, call the instance.

        Subclasses implement this method. `hist` is of shape ``(s, N)``, where
        ``s`` is the sequence dimension and ``N`` is the batch dimension. ``s``
        can be zero, indicating no history is available. All out-of-vocabulary
        elements (except eos) have been replaced with the oov token (if `oov`
        is not :obj:`None`). If eos is not :obj:`None`, `eos_mask` is also not
        :obj:`None` and of size ``(s, N)``. ``hist[s', n] == eos`` iff
        ``eos_mask[s', n].ne(0)``. `hist` has been right-filled with eos s.t.
        if ``hist[s', n] == eos`` and ``s' < s - 1`` then
        ``hist[s' + 1, n] == eos``. If sos has been set, `hist` has been
        prepended with a vector of ``(N,)`` filled with the symbol, which may
        or may not be in-vocabulary
        """
        raise NotImplementedError()

    def calc_full_log_probs(self, hist, eos_mask):
        """Calculate log probabilities at each step of history and next

        Do not call this directly; instead, call the instance with the keyword
        argument `full` set to :obj:`True`.

        Subclasses may implement this method to avoid repeated computation.
        It should produce an output identical to the following
        default implementation

        >>> out = torch.stack([
        >>>     self.calc_last_log_probs(
        >>>         hist[:s], None if eos_mask is None else eos_mask[:s])
        >>>     for s in range(0 if self.sos is None else 1, hist.shape[0] + 1)
        >>> ], dim=0)

        If sos has been set, `hist` has been prepended with a vector of sos.
        In this case, the probability of the empty slice of `hist` should not
        be calculated
        """
        return torch.stack(
            [
                self.calc_last_log_probs(
                    hist[:s], None if eos_mask is None else eos_mask[:s]
                )
                for s in range(0 if self.sos is None else 1, hist.shape[0] + 1)
            ],
            dim=0,
        )

    def forward(self, hist, full=False):
        if self.sos is not None:
            if hist.dim() < 2:
                raise RuntimeError("hist must be at least 2-D")
            sos_prepend = torch.full(
                (1,) + hist.shape[1:], self.sos, dtype=hist.dtype, device=hist.device
            )
            if hist.shape[0]:
                hist = torch.cat([sos_prepend, hist], dim=0)
            else:
                hist = sos_prepend
            del sos_prepend
        if self.eos is not None:
            eos_mask = hist.eq(self.eos).cumsum(0, dtype=torch.long).ne(0)
        else:
            eos_mask = None
        if self.oov is not None:
            oov_mask = hist.ge(self.vocab_size) | hist.lt(0)
            if eos_mask is not None:
                oov_mask = oov_mask & eos_mask.eq(0)
        else:
            oov_mask = None
        self.check_input(
            hist, eos_mask=eos_mask, oov_mask=oov_mask, skip_first=self.sos is not None
        )
        if oov_mask is not None:
            hist = hist.masked_fill(oov_mask, self.oov)
        if eos_mask is not None:
            hist = hist.masked_fill(eos_mask, self.eos)
        hist_shape = hist.shape
        first, rest = hist_shape[0], hist_shape[1:].numel()
        hist = hist.view(first, rest)
        if full:
            out_shape = (-1,)
            out_shape += tuple(hist_shape[1:]) + (self.vocab_size,)
            out = self.calc_full_log_probs(
                hist, None if eos_mask is None else eos_mask.view(first, rest)
            )
            out = out.reshape(*out_shape)
            if eos_mask is not None:
                if self.sos is not None:
                    eos_mask = eos_mask[1:]
                if eos_mask.shape[0]:
                    eos_mask = torch.cat(
                        [eos_mask[:1].ne(eos_mask[:1]), eos_mask], dim=0
                    )
                    out = out.masked_fill(eos_mask.unsqueeze(-1), 0.0)
        else:
            out_shape = tuple(hist_shape[1:]) + (self.vocab_size,)
            out = self.calc_last_log_probs(
                hist, None if eos_mask is None else eos_mask.view(first, rest)
            )
            out = out.reshape(*out_shape)
            if eos_mask is not None and eos_mask.shape[0]:
                out = out.masked_fill(eos_mask[-1].unsqueeze(-1), 0.0)
        return out

    def reset_parameters(self):
        pass


class LookupLanguageModel(SequentialLanguageModel):
    r"""Construct a backoff n-gram model from a fixed lookup table

    An instance of this model will search for a stored log-probability of the
    current token given a fixed-length history in a lookup table. If it can't
    find it, it backs off to a shorter length history and incurs a penalty:

    .. math::

        Pr(w_t|w_{t-1},\ldots,w_{t-(N-1)}) = \begin{cases}
            Entry(w_{t-(N-1)}, w_{t-(N-1)+1}, \ldots, w_t)
                & \text{if } Entry(w_{t-(N-1)}, \ldots) > 0 \\
            Backoff(w_{t-(N-1)}, \ldots, w_{t-1})
            Pr(w_t|w_{t-1},\ldots,w_{t-(N-1)+1}) & \text{else}
        \end{cases}

    Missing entries are assumed to have value 0. and missing backoff penalties
    are assumed to have value 1.

    Parameters
    ----------
    vocab_size : int
    sos : int, optional
    eos : int, optional
    oov : int, optional
    prob_list : sequence, optional
        A list of dictionaries whose entry at index ``i`` corresponds to a
        table of ``i+1``-gram probabilities. Keys must all be ids, not strings.
        Unigram keys are just ids; for n > 1 keys are tuples of ids with the
        latest word last. Values in the dictionary of the highest order n-gram
        dictionaries (last in `prob_list`) are the log-probabilities of the
        keys. Lower order dictionaries' values are pairs of log-probability and
        log-backoff penalty. If `prob_list` is not specified, a unigram model
        with a uniform prior will be built
    pad_sos_to_n : bool, optional
        For backoff models, it is usually the case that the input sequence is
        pre-padded with `sos` (n - 1) times rather than just once so that the
        context window is always of size `n`. If `pad_sos_to_n` is
        :obj:`False`, we will not perform the additional padding (though there
        will still be a sequence-initial `sos`). If no `sos` token is
        specified, this option is moot. It is usually safe to keep this setting
        :obj:`True` since no language model should assign a backoff penalty
        to prefixes of `sos` symbols.

    Notes
    -----
    Initializing an instance from an `prob_list` is expensive. `prob_list` is
    converted to a trie (something like [heafield2011]_) so that it takes up
    less space in memory, which can take some time.

    Rather than re-initializing repeatedly, it is recommended you save and load
    this module's state dict. :func:`load_state_dict` as been overridden to
    support loading different table sizes, avoiding the need for an accurate
    `prob_list` on initialization:

    >>> # first time
    >>> lm = LookupLanguageModel(vocab_size, sos, eos, oov, prob_list)  # slow
    >>> state_dict = lm.state_dict()
    >>> # save state dict, quit, startup, then reload state dict
    >>> lm = LookupLanguageModel(vocab_size, sos, eos, oov)  # fast!
    >>> lm.load_state_dict(state_dict)

    See Also
    --------
    pydrobert.util.parse_arpa_lm
        How to read a pretrained table of n-gram probabilities into
        `prob_list`. The parameter `token2id` should be specified to ensure
        id-based keys.
    """

    # XXX(sdrobert): as discussed in [heafield2011], we could potentially speed
    # up computations by keeping track of prefix probs and storing them in
    # case of backoff. This makes sense in a serial case, when we can choose to
    # explore or not explore a path. In a massively parallel environment, I'm
    # not sure it's worth the effort...

    def __init__(
        self,
        vocab_size,
        sos=None,
        eos=None,
        oov=None,
        prob_list=None,
        pad_sos_to_n=True,
    ):
        super(LookupLanguageModel, self).__init__(vocab_size, sos=sos, eos=eos, oov=oov)
        self.pad_sos_to_n = pad_sos_to_n
        if sos is not None and (sos < 0 or sos > vocab_size):
            # we want sos to refer to an index but it's oov, so we'll shift all
            # indices in hyp up by one and fill the occurrences of sos with 0
            self.shift = 1
        else:
            self.shift = 0
        if prob_list is None:
            logs = -torch.full(
                (self.shift + vocab_size,), vocab_size, dtype=torch.float
            ).log()
            ids = pointers = torch.tensor([], dtype=torch.uint8)
            self.max_ngram = 1
            self.max_ngram_nodes = self.shift + vocab_size
        else:
            self.max_ngram = len(prob_list)
            self.max_ngram_nodes = None  # changed by build_trie
            logs, ids, pointers = self._build_trie(prob_list)
        self.register_buffer("logs", logs)
        self.register_buffer("ids", ids)
        self.register_buffer("pointers", pointers)

    def extra_repr(self):
        s = super(LookupLanguageModel, self).extra_repr()
        s += ", max_ngram={}".format(self.max_ngram)
        if not self.pad_sos_to_n:
            s += ", pad_sos_to_n=False"
        return s

    def calc_last_log_probs(self, hist, eos_mask):
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
        B, V, N = hist.shape[1], self.vocab_size, self.max_ngram
        M, G, X = B * V, self.max_ngram_nodes, len(self.pointers)
        U = V + self.shift + (1 % N)
        K, L = X + G - U, 2 * X + G
        device = hist.device
        assert len(self.ids) == K
        assert len(self.logs) == L
        if self.eos is not None and (self.eos < 0 or self.eos >= self.vocab_size):
            # eos is out-of-vocabulary. Replace with in-vocabulary (it'll be
            # zero-filled by the parent class)
            hist = hist.masked_fill(hist.eq(self.eos), 0)
        hist = hist[max(0, hist.shape[0] - (N - 1)) :]
        if self.pad_sos_to_n and self.sos is not None and hist.shape[0] != N - 1:
            sos_prepend = torch.full(
                (N - 1 - hist.shape[0],) + hist.shape[1:],
                self.sos,
                dtype=hist.dtype,
                device=hist.device,
            )
            hist = torch.cat([sos_prepend, hist], dim=0)
            del sos_prepend
        if self.shift:
            hist = hist.masked_fill(hist.eq(self.sos), -self.shift)
            hist = hist + self.shift
        cur_step = torch.arange(
            self.shift, V + self.shift, dtype=hist.dtype, device=device
        )
        cur_step = cur_step.view(1, 1, V).expand(-1, B, -1)
        if hist.shape[0]:
            hist = hist.unsqueeze(-1).expand(-1, -1, V)
            hist = torch.cat([hist, cur_step], dim=0)
        else:
            hist = cur_step
        del cur_step
        if N == 1 or hist.shape[0] == 1:
            # we're a unigram model, or we've only got unigram history
            hist = hist[-1]  # (B, V)
            logs = self.logs[:G].unsqueeze(0).expand(B, G)
            return logs.gather(1, hist)  # (B, V)
        # we're now definitely not a unigram model w/ non-empty history
        assert X and K
        hist = hist.view(-1, M)  # pretend M is batch; reshape at end
        out = torch.zeros(M, dtype=torch.float, device=device)
        running_mask = torch.ones_like(out, dtype=torch.uint8).eq(1)
        vrange = torch.arange(V, dtype=torch.int32, device=device)
        while True:
            children = tokens = hist[0]
            if hist.shape[0] == 1:
                # unigrams always exist. Add the log-probability and exit
                out = torch.where(running_mask, out + self.logs[tokens], out)
                break
            offsets = self.pointers[children].to(torch.int32)  # (M,)
            num_children = self.pointers[children + 1].to(torch.int32)
            # the +1 is because we've shifted over one, meaning the offset
            # pointing to the same location is one less
            num_children = num_children - offsets + 1  # (M,)
            parents = children
            first_children = parents + offsets.long()
            step_mask = running_mask
            for t in range(1, hist.shape[0]):
                next_step = step_mask & num_children.ne(0)
                tokens = hist[t]
                S = num_children.max()
                all_children = (
                    first_children.unsqueeze(1) + vrange[:S].unsqueeze(0).long()
                )
                matches = self.ids[all_children.clamp(max=K + U - 1) - U].long()
                matches = matches == tokens.unsqueeze(1)
                matches = matches & (
                    vrange[:S].unsqueeze(0) < num_children.unsqueeze(1)
                )
                matches = matches & step_mask.unsqueeze(1)
                next_step = matches.any(1)
                if t == hist.shape[0] - 1:
                    # we're last. Add probabilities
                    logs = torch.where(
                        matches,
                        self.logs[all_children],
                        torch.zeros_like(all_children, dtype=torch.float),
                    ).sum(
                        1
                    )  # (M,)
                    # the trie has dummy lower-order n-grams. If there's
                    # an (n+1) gram passing through it. We do not want to
                    # match these - we will back off further
                    finite = torch.isfinite(logs)
                    out = torch.where(finite, out + logs, out)
                    next_step = next_step & finite
                    running_mask = running_mask & next_step.eq(0)
                    new_backoff = step_mask & next_step.eq(0)
                    # add backoff for newly failed paths
                    out = torch.where(
                        new_backoff, out + self.logs[X + G + parents], out,
                    )
                else:
                    # we're not last. Update children
                    children = torch.where(
                        matches, all_children, torch.zeros_like(all_children),
                    ).sum(1)
                # this'll be invalid for the last step, so don't re-use!
                step_mask = next_step
                if t != hist.shape[0] - 1:
                    offsets = self.pointers[children].to(torch.int32)
                    num_children = self.pointers[children + 1].to(torch.int32)
                    num_children = num_children - offsets + 1
                    parents = children
                    first_children = parents + offsets.long()
            hist = hist[1:]
        return out.view(B, V)

    def load_state_dict(self, state_dict, **kwargs):
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
                    "(vocab_size, eos, and sos must be correct!)".format(
                        self.vocab_size + self.shift, len(pointers) - 1
                    )
                )
            X, K, L = len(pointers), len(ids), len(logs)
            U = self.vocab_size + self.shift + 1
            self.max_ngram = 1
            self.max_ngram_nodes = last_ptr = U - 1
            error = RuntimeError(
                error_prefix + "buffer contains unexpected value (are you sure "
                "you've set vocab_size, eos, and sos correctly?)"
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
                    "(vocab_size, eos, and sos must be correct!)"
                    "".format(self.vocab_size + self.shift, len(logs))
                )
            self.max_ngram_nodes = self.vocab_size + self.shift
            self.max_ngram = 1
        # resize
        self.pointers = torch.empty_like(pointers, device=self.pointers.device)
        self.ids = torch.empty_like(ids, device=self.ids.device)
        self.logs = torch.empty_like(logs, device=self.logs.device)
        return super(LookupLanguageModel, self).load_state_dict(state_dict, **kwargs)

    def _build_trie(self, prob_list):
        if not len(prob_list):
            raise ValueError("prob_list must contain at least unigrams")
        prob_list = [x.copy() for x in prob_list]
        total_entries, nan, inf = 0, float("nan"), float("inf")
        unigrams = set(range(self.vocab_size))
        if self.shift:
            unigrams.add(self.sos)
        for n in range(self.max_ngram - 1, -1, -1):
            dict_ = prob_list[n]
            is_last = n == self.max_ngram - 1
            if is_last and not dict_:
                raise ValueError("Final element in prob_list must not be empty")
            if is_last:
                dummy_value = -inf
            else:
                dummy_value = -inf, 0.0
            if not n:
                keys = set(dict_.keys())
                if keys - unigrams:
                    raise ValueError(
                        "Unexpected unigrams in prob_list: {} (are these "
                        "ids?)".format(keys - unigrams)
                    )
                dict_.update((key, dummy_value) for key in unigrams - keys)
            else:
                for seq in dict_:
                    if len(seq) != n + 1:
                        raise ValueError(
                            "Key {0} in {1}-gram is not a sequence of length "
                            "{1}".format(n + 1, seq)
                        )
                    if set(seq) - unigrams:
                        raise ValueError(
                            "Unexpected tokens in {}-gram in prob_list: {} ("
                            "are these ids?)"
                            "".format(n + 1, set(seq) - unigrams)
                        )
                    prefix = seq[:-1]
                    if len(prefix) == 1:
                        prefix = prefix[0]
                    if prefix not in prob_list[n - 1]:
                        prob_list[n - 1][prefix] = -inf, 0.0
            total_entries += len(dict_)
            if is_last:
                self.max_ngram_nodes = len(dict_)
        if self.shift:
            prob_list[0] = dict(
                (0, v) if k == self.sos else (k + 1, v)
                for (k, v) in prob_list[0].items()
            )
            for n in range(1, self.max_ngram):
                prob_list[n] = dict(
                    (tuple(0 if t == self.eos else t + 1 for t in k), v)
                    for (k, v) in prob_list[n].items()
                )
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
                len(prob_list[n]) + len(prob_list[n - 1]) - 1 for n in range(1, N)
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
        dict_ = prob_list.pop(0)
        unigram_values = [dict_[x] for x in range(U - 1 % N)]
        allocated = U - 1 % N
        if N == 1:
            logs.copy_(torch.tensor(unigram_values))
        else:
            logs[:allocated].copy_(torch.tensor([x[0] for x in unigram_values]))
            logs[X + G : X + G + allocated].copy_(
                torch.tensor([x[1] for x in unigram_values])
            )
        del unigram_values
        parents = dict(((x,), x) for x in range(U - 1))
        N -= 1
        while N:
            dict_ = prob_list.pop(0)
            start = allocated
            pointers[allocated] = len(dict_) + 1
            logs[allocated] = logs[X + G + allocated] = nan
            allocated += 1
            keys = sorted(dict_.keys())
            children = dict()
            for key in keys:
                value = dict_[key]
                children[key] = allocated
                ids[allocated - U] = key[-1]
                if N == 1:
                    logs[allocated] = value
                else:
                    logs[allocated] = value[0]
                    logs[allocated + X + G] = value[1]
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


class HardOptimalCompletionDistillationLoss(torch.nn.Module):
    r"""A categorical loss function over optimal next tokens

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
    """

    def __init__(
        self,
        eos=None,
        include_eos=True,
        batch_first=False,
        ins_cost=1.0,
        del_cost=1.0,
        sub_cost=1.0,
        weight=None,
        reduction="mean",
    ):
        super(HardOptimalCompletionDistillationLoss, self).__init__()
        self.eos = eos
        self.include_eos = include_eos
        self.batch_first = batch_first
        self.ins_cost = ins_cost
        self.del_cost = del_cost
        self.sub_cost = sub_cost
        self.reduction = reduction
        self._cross_ent = torch.nn.CrossEntropyLoss(weight=weight, reduction="none")

    @property
    def weight(self):
        return self._cross_ent.weight

    @weight.setter
    def weight(self, value):
        self._cross_ent.weight = value

    def check_input(self, logits, ref, hyp):
        """Check if input formatted correctly, otherwise RuntimeError"""
        if logits.dim() != 3:
            raise RuntimeError("logits must be 3 dimensional")
        if logits.shape[:-1] != hyp.shape:
            raise RuntimeError("first two dims of logits must match hyp shape")
        if (
            self.include_eos
            and self.eos is not None
            and ((self.eos < 0) or (self.eos >= logits.shape[-1]))
        ):
            raise RuntimeError(
                "if include_eos=True, eos ({}) must be a class idx".format(self.eos)
            )
        if self.reduction not in {"mean", "sum", "none"}:
            raise RuntimeError(
                '"{}" is not a valid value for reduction' "".format(self.reduction)
            )

    def forward(self, logits, ref, hyp, warn=True):
        self.check_input(logits, ref, hyp)
        # the padding we use will never be exposed to the user, so we merely
        # ensure we're not trampling the eos
        padding = -2 if self.eos == -1 else -1
        self._cross_ent.ignore_index = padding
        optimals = optimal_completion(
            ref,
            hyp,
            eos=self.eos,
            include_eos=self.include_eos,
            batch_first=self.batch_first,
            ins_cost=self.ins_cost,
            del_cost=self.del_cost,
            sub_cost=self.sub_cost,
            padding=padding,
            exclude_last=True,
            warn=warn,
        )
        max_unique_next = optimals.shape[-1]
        logits = logits.unsqueeze(2).expand(-1, -1, max_unique_next, -1)
        logits = logits.contiguous()
        loss = self._cross_ent(
            logits.view(-1, logits.shape[-1]), optimals.flatten()
        ).view_as(optimals)
        padding_mask = optimals.eq(padding)
        no_padding_mask = padding_mask.eq(0)
        loss = loss.masked_fill(padding_mask, 0.0).sum(2)
        loss = torch.where(
            no_padding_mask.any(2), loss / no_padding_mask.float().sum(2), loss,
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class MinimumErrorRateLoss(torch.nn.Module):
    r"""Error rate expectation normalized over some number of transcripts

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
    """

    def __init__(
        self,
        eos=None,
        include_eos=True,
        sub_avg=True,
        batch_first=False,
        norm=True,
        ins_cost=1.0,
        del_cost=1.0,
        sub_cost=1.0,
        reduction="mean",
    ):
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
        """Check if the input is formatted correctly, otherwise RuntimeError"""
        if log_probs.dim() != 2:
            raise RuntimeError("log_probs must be 2 dimensional")
        if hyp.dim() != 3:
            raise RuntimeError("hyp must be 3 dimensional")
        if ref.dim() not in {2, 3}:
            raise RuntimeError("ref must be 2 or 3 dimensional")
        if self.batch_first:
            if ref.dim() == 2:
                ref = ref.unsqueeze(1).expand(-1, hyp.shape[1], -1)
            if (ref.shape[:2] != hyp.shape[:2]) or (ref.shape[:2] != log_probs.shape):
                raise RuntimeError(
                    "ref and hyp batch_size and sample dimensions must match"
                )
            if ref.shape[1] < 2:
                raise RuntimeError(
                    "Batch must have at least two samples, got {}"
                    "".format(ref.shape[1])
                )
        else:
            if ref.dim() == 2:
                ref = ref.unsqueeze(-1).expand(-1, -1, hyp.shape[-1])
            if (ref.shape[1:] != hyp.shape[1:]) or (ref.shape[1:] != log_probs.shape):
                raise RuntimeError(
                    "ref and hyp batch_size and sample dimensions must match"
                )
            if ref.shape[2] < 2:
                raise RuntimeError(
                    "Batch must have at least two samples, got {}"
                    "".format(ref.shape[2])
                )
        if self.reduction not in {"mean", "sum", "none"}:
            raise RuntimeError(
                '"{}" is not a valid value for reduction' "".format(self.reduction)
            )

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
            ref,
            hyp,
            eos=self.eos,
            include_eos=self.include_eos,
            norm=self.norm,
            batch_first=self.batch_first,
            ins_cost=self.ins_cost,
            del_cost=self.del_cost,
            sub_cost=self.sub_cost,
            warn=warn,
        ).view(batch_size, samples)
        if self.sub_avg:
            er = er - er.mean(1, keepdim=True)
        loss = er * torch.nn.functional.softmax(log_probs, 1)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class GlobalSoftAttention(with_metaclass(abc.ABCMeta, torch.nn.Module)):
    r"""Parent class for soft attention mechanisms on an entire input sequence

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
    """

    def __init__(self, query_size, key_size, dim=0):
        super(GlobalSoftAttention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.dim = dim

    @abc.abstractmethod
    def score(self, query, key):
        """Calculate the score function over the entire input

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
        """
        raise NotImplementedError()

    def check_input(self, query, key, value, mask=None):
        """Check if input is properly formatted, RuntimeError otherwise

        Warnings
        --------
        This method doesn't check that the tensors properly broadcast. If they
        don't, they will fail later on. It only ensures the proper sizes and
        that the final dimensions are appropriately sized where applicable

        See Also
        --------
        :ref:`Advanced Attention and Transformer Networks`
            For full broadcasting rules
        """
        key_dim = key.dim()
        if query.dim() != key_dim - 1:
            raise RuntimeError("query must have one fewer dimension than key")
        if key_dim != value.dim():
            raise RuntimeError("key must have same number of dimensions as value")
        if query.shape[-1] != self.query_size:
            raise RuntimeError("Last dimension of query must match query_size")
        if key.shape[-1] != self.key_size:
            raise RuntimeError("Last dimension of key must match key_size")
        if self.dim > key_dim - 2 or self.dim < -key_dim + 1:
            raise RuntimeError(
                "dim must be in the range [{}, {}]" "".format(-key_dim + 1, key_dim - 2)
            )
        if mask is not None and mask.dim() != key_dim - 1:
            raise RuntimeError("mask must have one fewer dimension than key")

    def forward(self, query, key, value, mask=None):
        self.check_input(query, key, value, mask)
        e = self.score(query, key)
        if mask is not None:
            e = e.masked_fill(mask.eq(0), -float("inf"))
        a = torch.nn.functional.softmax(e, self.dim)
        c = (a.unsqueeze(-1) * value).sum(self.dim)
        return c

    def extra_repr(self):
        return "query_size={}, key_size={}, dim={}".format(
            self.query_size, self.key_size, self.dim
        )

    def reset_parameters(self):
        pass


class DotProductSoftAttention(GlobalSoftAttention):
    r"""Global soft attention with dot product score function

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
    """

    def __init__(self, size, dim=0, scale_factor=1.0):
        super(DotProductSoftAttention, self).__init__(size, size, dim)
        self.scale_factor = scale_factor

    def score(self, query, key):
        query = query.unsqueeze(self.dim)
        return (query * key).sum(-1) * self.scale_factor

    def extra_repr(self):
        return "size={}, dim={}".format(self.query_size, self.dim)


class GeneralizedDotProductSoftAttention(GlobalSoftAttention):
    r"""Dot product soft attention with a learned matrix in between

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
    """

    def __init__(self, query_size, key_size, dim=0, bias=False):
        super(GeneralizedDotProductSoftAttention, self).__init__(
            query_size, key_size, dim
        )
        self.W = torch.nn.Linear(key_size, query_size, bias=bias)

    def score(self, query, key):
        Wkey = self.W(key)
        query = query.unsqueeze(self.dim)
        return (query * Wkey).sum(-1)

    def reset_parameters(self):
        self.W.reset_parameters()


class ConcatSoftAttention(GlobalSoftAttention):
    r"""Attention where query and key are concatenated, then fed into an MLP

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
    """

    def __init__(self, query_size, key_size, dim=0, bias=False, hidden_size=1000):
        super(ConcatSoftAttention, self).__init__(query_size, key_size, dim)
        self.hidden_size = hidden_size
        self.W = torch.nn.Linear(query_size + key_size, hidden_size, bias=bias)
        # there's no point in a bias for v. It'll just be absorbed by the
        # softmax later. You could add a bias after the tanh layer, though...
        self.v = torch.nn.Linear(hidden_size, 1, bias=False)

    def score(self, query, key):
        query = query.unsqueeze(self.dim)
        query_wo_last, key_wo_last = torch.broadcast_tensors(query[..., 0], key[..., 0])
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
        s += ", hidden_size={}".format(self.hidden_size)
        return s


class MultiHeadedAttention(GlobalSoftAttention):
    r"""Perform attention over a number of heads, concatenate, and project

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
    """

    def __init__(
        self,
        query_size,
        key_size,
        value_size,
        num_heads,
        single_head_attention,
        out_size=None,
        d_v=None,
        bias_WQ=False,
        bias_WK=False,
        bias_WV=False,
        bias_WC=False,
    ):
        super(MultiHeadedAttention, self).__init__(
            query_size, key_size, dim=single_head_attention.dim
        )
        self.value_size = value_size
        self.out_size = value_size if out_size is None else out_size
        self.num_heads = num_heads
        self.single_head_attention = single_head_attention
        # we don't keep these in sync in case someone's using
        # single_head_attention
        self.d_q = single_head_attention.query_size
        self.d_k = single_head_attention.key_size
        self.d_v = max(1, value_size // num_heads) if d_v is None else d_v
        self.WQ = torch.nn.Linear(query_size, num_heads * self.d_q, bias=bias_WQ)
        self.WK = torch.nn.Linear(key_size, num_heads * self.d_k, bias=bias_WK)
        self.WV = torch.nn.Linear(value_size, num_heads * self.d_v, bias=bias_WV)
        self.WC = torch.nn.Linear(self.d_v * num_heads, self.out_size, bias=bias_WC)
        single_head_attention.reset_parameters()

    def check_input(self, query, key, value, mask=None):
        """Check that input is formatted correctly, RuntimeError otherwise"""
        super(MultiHeadedAttention, self).check_input(query, key, value, mask)
        if value.shape[-1] != self.value_size:
            raise RuntimeError("Last dimension of value must match value_size")

    def score(self, query, key):
        raise NotImplementedError(
            "In MultiHeadedAttention, score() is handled by " "single_head_attention"
        )

    def forward(self, query, key, value, mask=None):
        self.check_input(query, key, value, mask)
        query_shape = tuple(query.shape)
        key_shape = tuple(key.shape)
        value_shape = tuple(value.shape)
        key_dim = key.dim()
        dim = (self.dim + key_dim) % key_dim
        query_heads = self.WQ(query).view(
            *(query_shape[:-1] + (self.num_heads, self.d_q))
        )
        key_heads = self.WK(key).view(*(key_shape[:-1] + (self.num_heads, self.d_k)))
        value_heads = self.WV(value).view(
            *(value_shape[:-1] + (self.num_heads, self.d_v))
        )
        if mask is not None:
            mask = mask.unsqueeze(-2)
        old_dim = self.single_head_attention.dim
        try:
            self.single_head_attention.dim = dim
            cat = self.single_head_attention(query_heads, key_heads, value_heads, mask)
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
        s += ", value_size={}, out_size={}, num_heads={}".format(
            self.value_size, self.out_size, self.num_heads
        )
        return s


class SpecAugment(torch.nn.Module):
    r"""Perform warping/masking of time/frequency dimensions of filter bank features

    SpecAugment [park2019]_ (and later [park2020]_) is a series of data transformations
    for training data augmentation of time-frequency features such as Mel-scaled
    triangular filter bank coefficients.

    An instance `spec_augment` of `SpecAugment` is called as

        new_feats = spec_augment(feats[, lengths])

    `feats` is a float tensor of shape ``(N, T, F)`` where ``N`` is the batch dimension,
    ``T`` is the time (frames) dimension, and ``F`` is the frequency (coefficients per
    frame) dimension. `lengths` is an optional long tensor of shape ``(N,)`` specifying
    the actual number of frames before right-padding per batch element. That is,
    for batch index ``n``, only ``feats[n, :lengths[n]]`` are valid. `new_feats` is
    of the same size as `feats` with some or all of the following operations performed
    in order independently per batch index:

    1. Choose a random frame along the time dimension. Warp `feats` such that ``feats[n,
       0]`` and feats[n, lengths[n] - 1]`` are fixed, but that random frame gets mapped
       to a random new location a few frames to the left or right.
    2. Do the same for the frequency dimension.
    3. Mask out (zero) one or more random-width ranges of frames in a random location
       along the time dimension.
    4. Do the same for the frequency dimension.

    The original SpecAugment implementation only performs steps 1, 3, and 4; step 2 is
    a trivial extension.

    Default parameter values are from [park2020]_.

    The `spec_augment` instance must be in training mode in order to apply any
    transformations; `spec_augment` always returns `feats` as-is in evaluation mode.

    Parameters
    ----------
    max_time_warp : float, optional
        A non-negative float specifying the maximum number of frames the chosen
        random frame can be shifted left or right by in step 1. Setting to :obj:`0`
        disables step 1.
    max_freq_warp : float, optional
        A non-negative float specifying the maximum number of coefficients the chosen
        random frequency coefficient index will be shifted up or down by in step 2.
        Setting to :obj:`0` disables step 2.
    max_time_mask : int, optional
        A non-negative integer specifying an absolute upper bound on the number of
        sequential frames in time that can be masked out by a single mask. The minimum
        of this upper bound and that from `max_time_mask_proportion` specifies the
        actual maximum. Setting this, `max_time_mask_proportion`, `num_time_mask`,
        or `num_time_mask_proportion` to :obj:`0` disables step 3.
    max_freq_mask : int, optional
        A non-negative integer specifying the maximum number of sequential coefficients
        in frequency that can be masked out by a single mask. Setting this or
        `num_freq_mask` to :obj:`0` disables step 4.
    max_time_mask_proportion : float, optional
        A value in the range :math:`[0, 1]` specifying a relative upper bound on the
        number of squential frames in time that can be masked out by a single mask. For
        batch element ``n``, the upper bound is ``int(max_time_mask_poportion *
        length[n])``. The minimum of this upper bound and that from `max_time_mask`
        specifies the actual maximum. Setting this, `max_time_mask`, `num_time_mask`,
        or `num_time_mask_proportion` to :obj:`0` disables step 4.
    num_time_mask : int, optional
        A non-negative integer specifying an absolute upper bound number of random masks
        in time per batch element to create. Setting this, `num_time_mask_proportion`,
        `max_time_mask`, or `max_time_mask_proportion` to :obj:`0` disables step 3.
        Drawn i.i.d. and may overlap.
    num_time_mask_proportion : float, optional
        A value in the range :math:`[0, 1]` specifying a relative upper bound on the
        number of time masks per element in the batch to create. For batch element
        ``n``, the upper bound is ``int(num_time_mask_proportion * length[n])``. The
        minimum of this upper bound and that from `num_time_mask` specifies the
        actual maximum. Setting this, `num_time_mask`, `max_time_mask`, or
        `max_time_mask_proportion` to :obj:`0` disables step 3. Drawn i.i.d. and may
        overlap.
    num_freq_mask : int, optional
        The total number of random masks in frequency per batch element to create.
        Setting this or `max_freq_mask` to :obj:`0` disables step 4. Drawn i.i.d. and
        may overlap.
    interpolation_order : int, optional
        Controls order of interpolation of warping. 1 = linear (default for
        [park2020]_). 2 = thin plate (default for [park2019]_). Higher orders are
        possible at increased computational cost.

    Attributes
    ----------
    max_time_warp : float
    max_freq_warp : float
    max_time_mask : int
    max_freq_mask : int
    max_time_mask_proportion : float
    num_time_mask : int
    num_freq_mask : int
    interpolation_order : int

    Notes
    -----
    There are a few differences between this implementation of warping and those you
    might find online or described in the source paper [park2019]_. These require some
    knowledge of what's happening under the hood and are unlikely to change the way you
    use this function. We assume we're warping in time, though the following applies to
    frequency warping as well.

    First, the warp parameters are real- rather than integer-valued. You can set
    `max_time_warp` or `max_freq_warp` to 0.5 if you'd like. The shift value drawn
    between ``[0, max_time_warp]`` is also real-valued. Since the underlying warp
    relies on interpolation between partial indices anyways (the vast majority of tensor
    values will be the result of interpolation), there is no preference for
    integer-valued parameters from a computational standpoint. Further, real-valued warp
    parameters allow for a virtually infinite number of warps instead of just a few.

    Second, the boundary points of the warp interpolation are :obj:`-0.5` and
    :obj:`length - 0.5` rather than :obj:`0` and :obj:`length - 1` (implied by
    :func:`sparse_image_warp`). In short, this ensures the distance between the boundary
    and the shifted value is at least half a sample. This change is mostly
    inconsequential as any interpolated values with indices outside of ``[0, length -
    1]`` will be filled with boundary values anyways.

    Finally, time warping is implemented by determining the transformation in one
    dimension (time) and broadcasting it across the other (frequency), rather than
    performing a two-dimensional warp. This is not in line with [park2019]_, but is
    with [park2020]_. I have confirmed with the first author that the slight warping
    of frequency that occurred due to the 2D warp was unintentional.
    """

    def __init__(
        self,
        max_time_warp=80.0,
        max_freq_warp=0.0,
        max_time_mask=100,
        max_freq_mask=27,
        max_time_mask_proportion=0.04,
        num_time_mask=20,
        num_time_mask_proportion=0.04,
        num_freq_mask=2,
        interpolation_order=1,
    ):
        super(SpecAugment, self).__init__()
        self.max_time_warp = max_time_warp
        self.max_freq_warp = max_freq_warp
        self.max_time_mask = max_time_mask
        self.max_freq_mask = max_freq_mask
        self.max_time_mask_proportion = max_time_mask_proportion
        self.num_time_mask = num_time_mask
        self.num_time_mask_proportion = num_time_mask_proportion
        self.num_freq_mask = num_freq_mask
        self.interpolation_order = interpolation_order

    def extra_repr(self):
        s = "warp_t={},max_f={},num_f={},max_t={},max_t_p={:.2f},num_t={}".format(
            self.max_time_warp,
            self.max_freq_mask,
            self.num_freq_mask,
            self.max_time_mask,
            self.max_time_mask_proportion,
            self.num_time_mask,
        )
        if self.max_freq_warp:
            s += ",warp_f={}".format(self.max_freq_warp)
        return s

    def check_input(self, feats, lengths=None):
        if feats.dim() != 3:
            raise RuntimeError(
                "Expected feats to have three dimensions, got {}".format(feats.dim())
            )
        if lengths is not None:
            if lengths.dim() != 1:
                raise RuntimeError(
                    "Expected lengths to be one dimensional, got {}"
                    "".format(lengths.dim())
                )
            N, T, _ = feats.shape
            if lengths.shape[0] != N:
                raise RuntimeError(
                    "Batch dimension of feats ({}) and lengths ({}) do not match"
                    "".format(N, lengths.shape[0])
                )
            if not torch.all((lengths <= T) & (lengths > 0)):
                raise RuntimeError(
                    "values of lengths must be between (1, {})".format(T)
                )

    def draw_parameters(self, feats, lengths=None):
        """Randomly draw parameterizations of augmentations

        Called as part of this layer's :func:`__call__` method.

        Parameters
        ----------
        feats : torch.FloatTensor
            Time-frequency features of shape ``(N, T, F)``.
        lengths : torch.LongTensor, optional
            Tensor of shape ``(N,)`` containing the number of frames before padding.

        Returns
        -------
        w_0 : torch.FloatTensor or :obj:`None`
            If step 1 is enabled, of shape ``(N,)`` containing the source points in the
            time warp.
        w : torch.FloatTensor or :obj:`None`
            If step 1 is enabled, of shape ``(N,)`` containing the number of frames to
            shift the source point by (positive or negative) in the destination in time.
            Positive values indicate a right shift.
        v_0 : torch.FloatTensor or :obj:`None`
            If step 2 is enabled, of shape ``(N,)`` containing the source points in the
            frequency warp.
        v : torch.FloatTensor or :obj:`None`
            If step 2 is enabled, of shape ``(N,)`` containing the number of
            coefficients to shift the source point by (positive or negative) in the
            destination in time. Positive values indicate a right shift.
        t_0 : torch.LongTensor or :obj:`None`
            If step 3 is enabled, of shape ``(N, M_T)`` where ``M_T`` is the number of
            time masks specifying the lower index (inclusive) of the time masks.
        t : torch.LongTensor or :obj:`None`
            If step 3 is enabled, of shape ``(N, M_T)`` specifying the number of frames
            per time mask.
        f_0 : torch.LongTensor or :obj:`None`
            If step 4 is enabled, of shape ``(N, M_F)`` where ``M_F`` is the number of
            frequency masks specifying the lower index (inclusive) of the frequency
            masks.
        f : torch.LongTensor or :obj:`None`
            If step 4 is enabled, of shape ``(N, M_F)`` specifying the number of
            frequency coefficients per frequency mask.
        """
        N, T, F = feats.shape
        device = feats.device
        eps = torch.finfo(torch.float).eps
        omeps = 1 - eps
        if lengths is None:
            lengths = torch.tensor([T] * N, device=device)
        lengths = lengths.to(device)
        # note that order matters slightly in whether we draw widths or positions first.
        # The paper specifies that position is drawn first for warps, whereas widths
        # are drawn first for masks
        if self.max_time_warp:
            # we want the range (W, length - W) exclusive to be where w_0 can come
            # from. If W >= length / 2, this is impossible. Rather than giving up,
            # we limit the maximum length to W < length / 2
            max_ = torch.clamp(lengths.float() / 2 - eps, max=self.max_time_warp)
            w_0 = (
                torch.rand([N], device=device) * (lengths - 2 * (max_ + eps))
                + max_
                + eps
            )
            w = torch.rand([N], device=device) * (2 * max_) - max_
        else:
            w_0 = w = None
        if self.max_freq_warp:
            max_ = min(self.max_freq_warp, F / 2 - eps)
            v_0 = torch.rand([N], device=device) * (F - 2 * (max_ + eps)) + max_ + eps
            v = torch.rand([N], device=device) * (2 * max_) - max_
        else:
            v_0 = v = None
        if (
            self.max_time_mask
            and self.max_time_mask_proportion
            and self.num_time_mask
            and self.num_time_mask_proportion
        ):
            lengths = lengths.float()
            max_ = (
                torch.clamp(
                    lengths * self.max_time_mask_proportion, max=self.max_time_mask,
                )
                .floor()
                .to(device)
            )
            nums_ = (
                torch.clamp(
                    lengths * self.num_time_mask_proportion, max=self.num_time_mask,
                )
                .floor()
                .to(device)
            )
            t = (
                (
                    torch.rand([N, self.num_time_mask], device=device)
                    * (max_ + omeps).unsqueeze(1)
                )
                .long()
                .masked_fill(
                    nums_.unsqueeze(1)
                    <= torch.arange(
                        self.num_time_mask, dtype=lengths.dtype, device=device
                    ),
                    0,
                )
            )
            t_0 = (
                torch.rand([N, self.num_time_mask], device=device)
                * (lengths.unsqueeze(1) - t + omeps)
            ).long()
        else:
            t = t_0 = None
        if self.max_freq_mask and self.num_freq_mask:
            max_ = min(self.max_freq_mask, F)
            f = (
                torch.rand([N, self.num_freq_mask], device=device) * (max_ + omeps)
            ).long()
            f_0 = (
                torch.rand([N, self.num_freq_mask], device=device) * (F - f + omeps)
            ).long()
        else:
            f = f_0 = None
        return w_0, w, v_0, v, t_0, t, f_0, f

    @staticmethod
    def warp_1d_grid(src, flow, lengths, max_length, interpolation_order):
        """Interpolate grid values for 1d of a grid_sample

        Called as part of this layer's :func:`__call__` method.

        Parameters
        ----------
        src : torch.LongTensor
            A tensor of shape ``(N,)`` containing random source points
        flow : torch.LongTensor
            A tensor of shape ``(N,)`` containing corresponding flow fields for
            ``src`` such that ``new_feats[n, * dst[n] *] =
            feats[n, * src[n] - flow[n] *]`` (for whichever dimension we're talking
            about)
        lengths : torch.LongTensor
            A tensor of shape ``(N,)`` specifying the number of valid indices along
            the dimension in question.
        max_length : int
            An integer s.t. ``max_length >= lengths[n]`` for all ``n``
        interpolation order : int
            Degree of warp.

        Returns
        -------
        grid : torch.FloatTensor
            A tensor of shape ``(N, max_length)`` providing coordinates for one
            dimension of :func:`torch.nn.functional.grid_sample` that will be used to
            warp the features
        """
        device = src.device
        # the interpolation has three points (per batch elem):
        # 1. t=-.5, flow=0
        # 2. t=src, flow=flow
        # 3. t=lengths -.5, flow=0
        # whatever happens after lengths -.5 is undefined
        # grid = (2 * dst - 2 * flow + 1) / max_length - 1
        N = src.shape[0]
        src, flow, lengths = src.float(), flow.float(), lengths.float()
        zeros = torch.zeros_like(src)
        src = torch.stack([zeros - 0.5, src, lengths - 0.5], 1)  # (N, 3)
        flow = torch.stack([zeros, flow, zeros], 1)  # (N, 3)
        sparse_grid = (2.0 * src + 1.0) / max_length - 1.0  # (N,3)
        t = torch.arange(max_length, device=device, dtype=torch.float)
        grid = polyharmonic_spline(
            (src + flow).unsqueeze(-1),  # dst (N, 3, 1)
            sparse_grid.unsqueeze(-1),  # (N, 3, 1)
            t.unsqueeze(0).expand(N, max_length).unsqueeze(-1),  # (N, T, 1)
            interpolation_order,
        ).squeeze(
            -1
        )  # (N, T)
        # we perform "boundary" interpolation, meaning any values past index length - 1
        # are assumed to be equal to the boundary and with zero gradient.
        boundary = (2.0 * lengths - 1.0) / max_length - 1.0
        grid = torch.min(grid, boundary.unsqueeze(-1))
        return grid

    def apply_parameters(self, feats, params, lengths=None):
        """Use drawn parameters to apply augmentations

        Called as part of this layer's :func:`__call__` method.

        Parameters
        ----------
        feats : torch.FloatTensor
            Time-frequency features of shape ``(N, T, F)``.
        params : tuple of torch.LongTensor
            All parameter tensors returned by :func:`draw_parameters`.
        lengths : torch.LongTensor, optional
            Tensor of shape ``(N,)`` containing the number of frames before padding.

        Returns
        -------
        new_feats : torch.FloatTensor
            Augmented time-frequency features of same shape as `feats`.
        """
        N, T, F = feats.shape
        device = feats.device
        if lengths is None:
            lengths = torch.tensor([T] * N, device=device)
        lengths = lengths.float()
        w_0, w, v_0, v, t_0, t, f_0, f = params
        new_feats = feats
        time_grid = freq_grid = None
        do_warp = False
        if w_0 is not None and w is not None:
            time_grid = self.warp_1d_grid(w_0, w, lengths, T, self.interpolation_order)
            do_warp = True
        if v_0 is not None and v is not None:
            freq_grid = self.warp_1d_grid(
                v_0,
                v,
                torch.tensor([F] * N, device=device),
                F,
                self.interpolation_order,
            )
            do_warp = True
        if do_warp:
            if time_grid is None:
                time_grid = torch.arange(T, device=device, dtype=torch.float)
                time_grid = (2 * time_grid + 1) / T - 1
                time_grid = time_grid.unsqueeze(0).expand(N, T)
            elif freq_grid is None:
                freq_grid = torch.arange(F, device=device, dtype=torch.float)
                freq_grid = (2 * freq_grid + 1) / F - 1
                freq_grid = freq_grid.unsqueeze(0).expand(N, F)
            time_grid = time_grid.unsqueeze(2).expand(N, T, F)
            freq_grid = freq_grid.unsqueeze(1).expand(N, T, F)
            # note: grid coordinate are (freq, time) rather than (time, freq)
            grid = torch.stack([freq_grid, time_grid], 3)  # (N, T, F, 2)
            new_feats = torch.nn.functional.grid_sample(
                new_feats.unsqueeze(1), grid, padding_mode="border", align_corners=False
            ).squeeze(1)
        tmask = fmask = None
        if t_0 is not None and t is not None:
            tmask = torch.arange(T, device=device).unsqueeze(0).unsqueeze(2)  # (1, T,1)
            t_1 = t_0 + t  # (N, MT)
            tmask = (tmask >= t_0.unsqueeze(1)) & (tmask < t_1.unsqueeze(1))  # (N,T,MT)
            tmask = tmask.any(2, keepdim=True)  # (N, T, 1)
        if f_0 is not None and f is not None:
            fmask = torch.arange(F, device=device).unsqueeze(0).unsqueeze(2)  # (1, F,1)
            f_1 = f_0 + f  # (N, MF)
            fmask = (fmask >= f_0.unsqueeze(1)) & (fmask < f_1.unsqueeze(1))  # (N,F,MF)
            fmask = fmask.any(2).unsqueeze(1)  # (N, 1, F)
        if tmask is not None:
            if fmask is not None:
                tmask = tmask | fmask
            new_feats = new_feats.masked_fill(tmask, 0.0)
        elif fmask is not None:
            new_feats = new_feats.masked_fill(fmask, 0.0)
        return new_feats

    def forward(self, feats, lengths=None):
        self.check_input(feats, lengths)
        if not self.training:
            return feats
        N, T, F = feats.shape
        if lengths is None:
            lengths = torch.tensor([T] * N, device=feats.device)
        params = self.draw_parameters(feats, lengths)
        return self.apply_parameters(feats, params, lengths)
