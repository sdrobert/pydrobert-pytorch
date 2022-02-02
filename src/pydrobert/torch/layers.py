# Copyright 2021 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common neural layers from the literature not included in pytorch.nn"""

import abc
import math
from turtle import forward
from typing import (
    Any,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Dict,
    Union,
    TYPE_CHECKING,
)

import torch

from pydrobert.torch.util import (
    beam_search_advance,
    dense_image_warp,
    edit_distance,
    error_rate,
    optimal_completion,
    ctc_prefix_search_advance,
    pad_variable,
    polyharmonic_spline,
    prefix_edit_distances,
    prefix_error_rates,
    sequence_log_probs,
    sparse_image_warp,
    time_distributed_return,
    warp_1d_grid,
    _get_tensor_eps,
)

import pydrobert.torch.config as config

from ._compat import broadcast_shapes, script

__all__ = [
    "BeamSearch",
    "ConcatSoftAttention",
    "CTCPrefixSearch",
    "DenseImageWarp",
    "DotProductSoftAttention",
    "EditDistance",
    "ErrorRate",
    "ExtractableSequentialLanguageModel",
    "GeneralizedDotProductSoftAttention",
    "GlobalSoftAttention",
    "hard_optimal_completion_distillation_loss",
    "HardOptimalCompletionDistillationLoss",
    "LookupLanguageModel",
    "minimum_error_rate_loss",
    "MinimumErrorRateLoss",
    "MixableSequentialLanguageModel",
    "MultiHeadedAttention",
    "OptimalCompletion",
    "PadVariable",
    "PolyharmonicSpline",
    "PrefixEditDistances",
    "PrefixErrorRates",
    "random_shift",
    "RandomShift",
    "SequentialLanguageModel",
    "SequentialLogProbabilities",
    "SparseImageWarp",
    "spec_augment_apply_parameters",
    "spec_augment_draw_parameters",
    "spec_augment",
    "SpecAugment",
    "TimeDistributedReturn",
    "Warp1DGrid",
]


class TimeDistributedReturn(torch.nn.Module):
    r"""Accumulate future local rewards at every time step

    In `reinforcement learning
    <https://en.wikipedia.org/wiki/Reinforcement_learning>`__, the return is defined as
    the sum of discounted future rewards. This function calculates the return for a
    given time step :math:`t` as

    .. math::

        R_t = \sum_{t'=t} \gamma^(t' - t) r_{t'}

    Where :math:`r_{t'}` gives the (local) reward at time :math:`t'` and :math:`\gamma`
    is the discount factor. :math:`\gamma \in [0, 1)` implies convergence, but this is
    not enforced here.

    When instantiated, this module has the signature::

        R = time_distributed_return(r)
    
    where `r` is a two-dimensional tensor of shape ``(steps, batch_size)``, ``r[t, n]``
    being the (0-indexed) ``t``-th element of the ``n``-th batch element sequence.
    The return value `R` is a tensor of the same shape.


    Parameters
    ----------
    r : torch.Tensor
        A two-dimensional float tensor of shape ``(steps, batch_size)`` (or
        ``(batch_size, steps)`` if `batch_first` is :obj:`True`) of local rewards. The
        :math:`t` dimension is the step dimension
    gamma : float
        The discount factor :math:`\gamma`.
    batch_first : bool, optional
        Transposes the dimensions of `r` and `R` if :obj:`True`.

    See Also
    --------
    :ref:`Gradient Estimators`
        Provides an example of reinforcement learning that uses this function
    """

    __constants__ = ["gamma", "batch_first"]

    gamma: float
    batch_first: bool

    def __init__(self, gamma: float, batch_first: bool):
        super().__init__()
        self.gamma = gamma
        self.batch_first = batch_first

    def extra_repr(self) -> str:
        return f"gamma={self.gamma},batch_first={self.batch_first}"

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return time_distributed_return(r, self.gamma, self.batch_first)


class Warp1DGrid(torch.nn.Module):
    """Interpolate grid values for a dimension of a grid_sample

    This module determines a grid along a single dimension of a signal,
    image, volume, whatever. 

    When instantiated, this method has the signature::

        grid = warp_1d_grid(src, flow, lengths)
    
    `src` is a tensor of shape ``(N,)`` containing source points. `flow` is
    a tensor of shape ``(N,)`` containing corresponding flow fields for `src`.
    `lengths` is a long tensor of shape ``(N,)`` specifying the number of
    valid indices along the dimension in question. The return value is a tensor
    `grid` of shape ``(N, max_length)`` which provides coodinates for one
    dimension of the grid passed to :func:`torch.nn.functional.grid_sample`.
    See the example below.

    Parameters
    ----------
    max_length : int or `None`, optional
        A maximum length to which the grid will be padded. If unspecified,
        it will be taken to be ``lengths.max().ceil()``.
    interpolation_order : int, optional
        The degree of the spline used ot interpolate the grid.
    """

    __constants__ = ["max_length", "interpolation_order"]

    interpolation_order: int
    max_length: Optional[int]

    def __init__(self, max_length: Optional[int] = None, interpolation_order: int = 1):
        super().__init__()
        if max_length is not None and max_length < 0:
            raise ValueError("max_length must be non-negative")
        self.max_length = max_length
        self.interpolation_order = interpolation_order

    def extra_repr(self) -> str:
        s = f"interpolation_order={self.interpolation_order}"
        if self.max_length is not None:
            s = f"max_length={self.max_length}, " + s
        return s

    def forward(
        self, src: torch.Tensor, flow: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        return warp_1d_grid(
            src, flow, lengths, self.max_length, self.interpolation_order
        )


class PadVariable(torch.nn.Module):
    """Pad variable-length input by a variable amount on each side

    This module attempts to replicate the behaviour of :func:`torch.nn.functional.pad`
    on a tensor containing variable sequence lengths with variable amounts of
    padding.

    When instantiated, this module has the signature::

        padded = pad_variable(x, lens, pad)

    `x` is a tensor of shape ``(N, T, *)`` where ``N`` is the batch index and ``T`` is
    the sequence index. `lens` is a long tensor of shape ``(N,)`` specifying the
    sequence lengths: only the values in the range ``x[n, :lens[n]]`` are considered
    part of the sequence of batch element ``n``. `pad` is a tensor of shape ``(2, N)``
    specifying how many elements at the start (``pad[0]``) and end (``pad[1]``) of each
    sequence. The return tensor `padded` will have shape ``(N, T', *)`` such that, for a
    given batch index ``n``::

        padded[n, :pad[0, n]] = left padding
        padded[n, pad[0,n]:pad[0,n] + lens[n]] = x[n, :lens[n]]
        padded[n, pad[0,n] + lens[n]:pad[0,n] + lens[n] + pad[1, n]] = right padding

    Parameters
    ----------
    mode : {'constant', 'reflect', 'replicate'}, optional
        How to pad the sequences. :obj:`'constant'`: fill the padding region with the
        value specified by `value`. :obj:`'reflect'`: padded values are reflections
        around the endpoints. For example, the first right-padded value of the ``n``-th
        sequence would be ``x[n, lens[n] - 2``, the third ``x[n, lens[n] - 3]``, and
        so on. :obj:`replicate`: padding duplicates the endpoints of each sequence.
        For example, the left-padded values of the ``n``-th sequence would all be
        ``x[n, 0]``; the right-padded values would be ``x[n, lens[n] - 1]``.
    value : scalar, optional
        The value to pad with when ``mode == 'constant'``.

    Raises
    ------
    NotImplementedError
        If any value in ``pad[:, n]`` equals or exceeds ``lens[n]`` when
        ``mode == 'reflect'``
    RuntimeError
        If any element in `lens` is less than 1 when ``mode == 'replicate'``

    Examples
    --------

    >>> x = torch.arange(10).view(2, 5)
    >>> x
    tensor([[0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]])
    >>> lens = torch.tensor([3, 4])
    >>> pad = torch.arange(4).view(2, 2)
    >>> pad.t()  # [[0_left, 0_right], [1_left, 1_right]]
    tensor([[0, 2],
            [1, 3]])
    >>> y = pad_variable(x, lens, pad)  # constant w/ value 0
    >>> y[0, :3 + 0 + 2]
    tensor([0, 1, 2, 0, 0])
    >>> y[1, :4 + 1 + 3]
    tensor([0, 5, 6, 7, 8, 0, 0, 0])
    >>> y = pad_variable(x, lens, pad, 'reflect')
    >>> y[0, :3 + 0 + 2]
    tensor([0, 1, 2, 1, 0])
    >>> y[1, :4 + 1 + 3]
    tensor([6, 5, 6, 7, 8, 7, 6, 5])
    >>> y = pad_variable(x, lens, pad, 'replicate')
    >>> y[0, :3 + 0 + 2]
    tensor([0, 1, 2, 2, 2])
    >>> y[1, :4 + 1 + 3]
    tensor([5, 5, 6, 7, 8, 8, 8, 8])
    """

    __constants__ = ["mode", "value"]

    mode: str
    value: float

    def __init__(self, mode: str = "constant", value: float = 0.0):
        super().__init__()
        if mode not in {"constant", "reflect", "replicate"}:
            raise ValueError(
                "mode should be one of 'constant', 'reflect', or 'replicate', got "
                f"'{mode}'"
            )
        self.mode = mode
        self.value = value

    def extra_repr(self) -> str:
        s = f"mode={self.mode}"
        if self.mode == "constant":
            s += f", value={self.value}"
        return s

    def forward(
        self, x: torch.Tensor, lens: torch.Tensor, pad: torch.Tensor
    ) -> torch.Tensor:
        return pad_variable(x, lens, pad, self.mode, self.value)


_SM_ARGS = """\
`ref` is a long tensor of shape ``(max_ref_steps, batch_size)`` such that the
``n``-th reference (gold-standard) token sequence is stored in ``ref[:, n]``. `hyp`
is a long tensor of shape ``(max_hyp_steps, batch_size)`` containing the hypothesis
(machine-generated) sequences."""

_SM_PARAM_DICT = {
    "eos": """\
    eos : int or None, optional
        A special token in `ref` and `hyp` whose first occurrence in each batch
        indicates the end of a transcript. This allows for variable-length transcripts
        in the batch.
    """,
    "include_eos": """\
    include_eos : bool, optional
        Whether to include the first instance of `eos` found in both `ref` and `hyp` as
        valid tokens to be computed as part of the rate. This is useful when gauging
        if a model is learning to emit the `eos` properly, but is not usually included
        in an evaluation. Only the first `eos` per transcript is included.
    """,
    "norm": """\
    norm : bool, optional
        If :obj:`True`, will normalize the distance by the number of tokens in the
        reference sequence (making the returned value a divergence)
    """,
    "batch_first": """\
    batch_first : bool, optional
        If :obj:`True`, the first two dimensions of `ref`, `hyp`, and the return value
        are transposed from those above.
    """,
    "ins_cost": """\
    ins_cost : float, optional
        The cost of an adding an extra token to a sequence in `ref`
    """,
    "del_cost": """\
    del_cost : float, optional
        The cost of removing a token from a sequence in `ref`
    """,
    "sub_cost": """\
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`
    """,
    "warn": """\
    warn : bool, optional
        Whether to display warnings on irregularities. Currently, this can happen in
        three ways:

        1. If :obj:`True` and `ins_cost`, `del_cost`, or `sub_cost` is not 1, a warning
           about a difference in computations will be raised. See the below warning for
           more info.
        2. If :obj:`True` and `norm` is :obj:`True`, will warn when a reference
           transcription has zero length
        3. If `eos` is set and `include_eos` is :obj:`True`, will warn when a transcript
           does not include an `eos` symbol
    """,
    "padding": """\
    padding : int, optional
        The value to right-pad unequal-length sequences with. Defauls to
        :obj:`pydrobert.torch.config.INDEX_PAD_VALUE`.
    """,
    "exclude_last": """\
    exclude_last : bool, optional
        If true, will exclude the final prefix, consisting of the entire transcript,
        from the return value. It will be of shape ``(max_hyp_steps, batch_size,
        max_unique_next)``
    """,
}


class _StringMatching(torch.nn.Module, metaclass=abc.ABCMeta):
    __constants__ = [
        "eos",
        "include_eos",
        "batch_first",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "warn",
    ]

    eos: Optional[int]
    include_eos: bool
    batch_first: bool
    ins_cost: float
    del_cost: float
    sub_cost: float
    warn: bool

    def __init__(
        self, eos, include_eos, batch_first, ins_cost, del_cost, sub_cost, warn
    ):
        super().__init__()
        self.eos = eos
        self.include_eos = include_eos
        self.batch_first = batch_first
        self.ins_cost = ins_cost
        self.del_cost = del_cost
        self.sub_cost = sub_cost
        self.warn = warn

    def extra_repr(self) -> str:
        return ", ".join(f"{x}={getattr(self, x)}" for x in self.__constants__)

    @abc.abstractmethod
    def forward(self, ref: torch.Tensor, hyp: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EditDistance(_StringMatching):
    __constants__ = [
        "eos",
        "include_eos",
        "norm",
        "batch_first",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "warn",
    ]

    norm: bool

    def __init__(
        self,
        eos: Optional[int] = None,
        include_eos: bool = False,
        norm: bool = False,
        batch_first: bool = False,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        sub_cost: float = 1.0,
        warn: bool = True,
    ):
        super().__init__(
            eos, include_eos, batch_first, ins_cost, del_cost, sub_cost, warn
        )
        self.norm = norm

    __doc__ = f"""Compute an edit distance over a batch of references and hypotheses

    An `Edit Distance <https://en.wikipedia.org/wiki/Edit_distance>`__ quantifies
    how dissimilar two token sequences are as the total cost of transforming a
    reference sequence into a hypothesis sequence. There are three operations that can
    be performed, each with an associated cost: adding an extra token to the reference,
    removing a token from the reference, or swapping a token in the reference with a
    token in the hypothesis.

    When instantiated, this module has the signature::

        ed = edit_distance(ref, hyp)

    {_SM_ARGS}  The return value `ed` is a tensor of shape ``(batch_size,)`` storing the
    associated edit distances.

    Parameters
    ----------
    {"".join(_SM_PARAM_DICT[c] for c in __constants__)}

    Notes
    -----
    This module returns identical values (modulo a bug fix) to :func:`error_rate` up
    to `v0.3.0` (though the default of `norm` has changed to :obj:`False`). For more
    details on the distinction between this module and the new :func:`ErrorRate`,
    please see that module's documentation.
    """

    def forward(self, ref: torch.Tensor, hyp: torch.Tensor) -> torch.Tensor:
        return edit_distance(
            ref,
            hyp,
            self.eos,
            self.include_eos,
            self.norm,
            self.batch_first,
            self.ins_cost,
            self.del_cost,
            self.sub_cost,
            self.warn,
        )


class PrefixEditDistances(_StringMatching):

    __constants__ = [
        "eos",
        "include_eos",
        "norm",
        "batch_first",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "padding",
        "exclude_last",
        "warn",
    ]

    norm: bool
    padding: int
    exclude_last: bool

    def __init__(
        self,
        eos: Optional[int] = None,
        include_eos: bool = True,
        norm: bool = False,
        batch_first: bool = False,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        sub_cost: float = 1.0,
        padding: int = config.INDEX_PAD_VALUE,
        exclude_last: bool = False,
        warn: bool = True,
    ):
        super().__init__(
            eos, include_eos, batch_first, ins_cost, del_cost, sub_cost, warn
        )
        self.norm = norm
        self.padding = padding
        self.exclude_last = exclude_last

    __doc__ = f"""Compute the edit distance between ref and each prefix of hyp

    When instantiated, this module has the signature::

        prefix_eds = prefix_edit_distances(ref, hyp)
    
    {_SM_ARGS} The return value `prefix_eds` is of shape ``(max_hyp_steps + 1,
    batch_size)`` and contains the edit distances for each prefix of each hypothesis,
    starting from the empty prefix.

    Parameters
    ----------
    {"".join(_SM_PARAM_DICT[c] for c in __constants__)}
    
    Notes
    -----
    This module returns identical values (modulo a bug fix) to
    :func:`prefix_error_rates` (and :class:`PrefixErrorRates`) up to `v0.3.0` (though
    the default of `norm` has changed to :obj:`False`). For more details on the
    distinction between this module and the new :func:`prefix_error_rates`, please
    consult the documentation of :class:`ErrorRate`.
    """

    def forward(self, ref: torch.Tensor, hyp: torch.Tensor) -> torch.Tensor:
        return prefix_edit_distances(
            ref,
            hyp,
            self.eos,
            self.include_eos,
            self.norm,
            self.batch_first,
            self.ins_cost,
            self.del_cost,
            self.sub_cost,
            self.padding,
            self.exclude_last,
            self.warn,
        )


class ErrorRate(_StringMatching):
    __constants__ = [
        "eos",
        "include_eos",
        "norm",
        "batch_first",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "warn",
    ]

    norm: bool

    def __init__(
        self,
        eos: Optional[int] = None,
        include_eos: bool = False,
        norm: bool = True,
        batch_first: bool = False,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        sub_cost: float = 1.0,
        warn: bool = True,
    ):
        super().__init__(
            eos, include_eos, batch_first, ins_cost, del_cost, sub_cost, warn
        )
        self.norm = norm

    __doc__ = f"""Calculate error rates over a batch of references and hypotheses

    An error rate is the total number of insertions, deletions, and substitutions
    between a reference (gold-standard) and hypothesis (generated) transcription,
    normalized by the number of elements in a reference. Consult the Wikipedia article
    on the `Levenshtein distance <https://en.wikipedia.org/wiki/Levenshtein_distance>`__
    for more information.

    When instantiated, this module has the signature::

        er = error_rate(ref, hyp)

    {_SM_ARGS} The return value `er` is a tensor of shape ``(batch_size,)`` storing the
    associated error rates. `er` will not have a gradient, and is thus not directly
    suited to being a loss function.

    Parameters
    ----------
    {"".join(_SM_PARAM_DICT[c] for c in __constants__)}

    Warnings
    --------
    Up to and including `v0.3.0`, :func:`error_rate` computed a normalized
    `Edit distance <https://en.wikipedia.org/wiki/Edit_distance>`__ instead of an error
    rate. The latter can be considered the total weighted cost of insertions, deletions,
    and substitutions (as per `ins_cost`, `del_cost`, and `sub_cost`), whereas the
    former is the sum of the number of mistakes. The old behaviour of returning the cost
    is now in :func:`edit_distance` and :class:`EditDistance` (though `norm` is
    :obj:`False` by default). For speech recognition evaluation, this module or
    :func:`error_rate` is the one to use. However, if you are using the default costs,
    ``ins_cost == del_cost == sub_cost == 1``, there should be no numerical difference
    between the two.
    """

    def forward(self, ref: torch.Tensor, hyp: torch.Tensor) -> torch.Tensor:
        return error_rate(
            ref,
            hyp,
            self.eos,
            self.include_eos,
            self.norm,
            self.batch_first,
            self.ins_cost,
            self.del_cost,
            self.sub_cost,
            self.warn,
        )


class PrefixErrorRates(_StringMatching):
    __constants__ = [
        "eos",
        "include_eos",
        "norm",
        "batch_first",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "padding",
        "exclude_last",
        "warn",
    ]

    norm: bool
    padding: int
    exclude_last: bool

    def __init__(
        self,
        eos: Optional[int] = None,
        include_eos: bool = True,
        norm: bool = True,
        batch_first: bool = False,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        sub_cost: float = 1.0,
        padding: int = config.INDEX_PAD_VALUE,
        exclude_last: bool = False,
        warn: bool = True,
    ):
        super().__init__(
            eos, include_eos, batch_first, ins_cost, del_cost, sub_cost, warn
        )
        self.norm = norm
        self.padding = padding
        self.exclude_last = exclude_last

    __doc__ = f"""Compute the error rate between ref and each prefix of hyp

    When instantiated, this module has the signature::

        prefix_ers = prefix_error_rates(ref, hyp)
    
    {_SM_ARGS} The return value `prefix_ers` is of shape ``(max_hyp_steps + 1,
    batch_size)`` and contains the error rates for each prefix of each hypothesis,
    starting from the empty prefix.

    Parameters
    ----------
    {"".join(_SM_PARAM_DICT[c] for c in __constants__)}

    Warnings
    --------
    The values returned by :func:`prefix_error_rates` (and thus this module) changed
    after `v0.3.0`. The old behaviour can be found in :class:`PrefixEditDistances`
    (though with `norm` defaulting to :obj:`False`). Consult the warning in
    :class:`ErrorRate` for more info.
    """

    def forward(self, ref: torch.Tensor, hyp: torch.Tensor) -> torch.Tensor:
        return prefix_error_rates(
            ref,
            hyp,
            self.eos,
            self.include_eos,
            self.norm,
            self.batch_first,
            self.ins_cost,
            self.del_cost,
            self.sub_cost,
            self.padding,
            self.exclude_last,
            self.warn,
        )


class OptimalCompletion(_StringMatching):
    __constants__ = [
        "eos",
        "include_eos",
        "batch_first",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "padding",
        "exclude_last",
        "warn",
    ]

    padding: int
    exclude_last: bool

    def __init__(
        self,
        eos: Optional[int] = None,
        include_eos: bool = True,
        batch_first: bool = False,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        sub_cost: float = 1.0,
        padding: int = config.INDEX_PAD_VALUE,
        exclude_last: bool = False,
        warn: bool = True,
    ):
        super().__init__(
            eos, include_eos, batch_first, ins_cost, del_cost, sub_cost, warn
        )
        self.padding = padding
        self.exclude_last = exclude_last

    __doc__ = f"""Return a mask of next tokens of a minimum edit distance prefix
    
    When instantiated, this module has the signature::

        optimals = optimal_completion(ref, hyp)
    
    {_SM_ARGS} The return value `optimals` is a long tensor of shape ``(max_hyp_steps +
    1, batch_size, max_unique_next)``, where ``max_unique_next <= max_ref_steps``, of
    the unique tokens that could be added to the hypothesis prefix ``hyp[:prefix_len,
    batch]`` such that some remaining suffix concatenated to the prefix would result in
    a minimal edit distance. See below for an example.

    Parameters
    ----------
    {"".join(_SM_PARAM_DICT[c] for c in __constants__)}

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
    ...     print('prefix={{}}: {{}}'.format(
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

    def forward(self, ref: torch.Tensor, hyp: torch.Tensor) -> torch.Tensor:
        return optimal_completion(
            ref,
            hyp,
            self.eos,
            self.include_eos,
            self.batch_first,
            self.ins_cost,
            self.del_cost,
            self.sub_cost,
            self.padding,
            self.exclude_last,
            self.warn,
        )


class SparseImageWarp(torch.nn.Module):
    r"""Warp an image by specifying mappings between few control points

    This module, when instantiated, has the signature::

        warped[, flow] = sparse_image_warp(image, source_points, dest_points)

    `image` is a source image of shape `(N, C, H, W)``, where ``N`` is the batch
    dimension, ``C`` the channel dimension, ``H`` the image height, and ``W`` the image
    width. `source_points` and `dest_points` are tensors of shape ``(N, M, 2)``, where
    ``M`` is the number of control points. `warped` is a float tensor of shape ``(N, C,
    H, W)`` containing the warped images. The point ``source_points[n, m, :]`` in
    `image` will be mapped to ``dest_points[n, m, :]`` in `warped`. If `include_flow` is
    :obj:`True`, `flow`, a float tensor of shape ``(N, H, W, 2)``. ``flow[n, h, w, :]``
    is the flow for coordinates ``h, w`` in whatever order was specified by `indexing`.
    See :class:`DenseImageWarp` for more details about `flow`.

    This module mirrors the behaviour of Tensorflow's `sparse_image_warp
    <https://www.tensorflow.org/addons/api_docs/python/tfa/image/sparse_image_warp>`__,
    except `image` is in ``NCHW`` order instead of ``NHWC`` order. For more details,
    please consult their documentation.

    Parameters
    ----------
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
    
    Warnings
    --------
    When this module is scripted, its return type will be :class:`typing.Any`. This
    reflects the fact that either `warn` is returned on its own (a tensor) or both
    `warn` and `flow` (a tuple). Use :func:`torch.jit.isinstance` for type refinement in
    subsequent scripting. Tracing will infer the correct type.
    """

    __constants__ = [
        "indexing",
        "field_interpolation_order",
        "field_regularization_weight",
        "field_full_matrix",
        "pinned_boundary_points",
        "dense_interpolation_mode",
        "dense_padding_mode",
        "include_flow",
    ]

    field_interpolation_order: int
    field_regularization_weight: float
    field_full_matrix: bool
    pinned_boundary_points: int
    dense_interpolation_mode: str
    dense_padding_mode: str
    include_flow: bool

    def __init__(
        self,
        indexing: str = "hw",
        field_interpolation_order: int = 2,
        field_regularization_weight: float = 0.0,
        field_full_matrix: bool = True,
        pinned_boundary_points: int = 0,
        dense_interpolation_mode: str = "bilinear",
        dense_padding_mode: str = "border",
        include_flow: bool = True,
    ):
        super().__init__()
        if field_interpolation_order <= 0:
            raise ValueError(
                "field_interpolation_order must be positive, got "
                f"{field_interpolation_order}"
            )
        if pinned_boundary_points < 0:
            raise ValueError(
                "pinned_boundary_points must be non-negative, got "
                f"{pinned_boundary_points}"
            )
        if indexing not in {"hw", "wh"}:
            raise ValueError(f"indexing must be either 'hw' or 'wh', got '{indexing}'")
        if dense_interpolation_mode not in {"bilinear", "nearest"}:
            raise ValueError(
                "dense_interpolation_mode must be either 'bilinear' or 'nearest', got "
                f"'{dense_interpolation_mode}'"
            )
        if dense_padding_mode not in {"border", "zeros", "reflection"}:
            raise ValueError(
                "dense_padding_mode must be one of 'border', 'zeros', or 'relection', "
                f"got '{dense_padding_mode}'"
            )
        self.indexing = indexing
        self.field_interpolation_order = field_interpolation_order
        self.field_regularization_weight = field_regularization_weight
        self.field_full_matrix = field_full_matrix
        self.pinned_boundary_points = pinned_boundary_points
        self.dense_interpolation_mode = dense_interpolation_mode
        self.dense_padding_mode = dense_padding_mode
        self.include_flow = include_flow

    def extra_repr(self) -> str:
        return ", ".join(f"{x}={getattr(self, x)}" for x in self.__constants__)

    if TYPE_CHECKING:

        def forward(
            self,
            image: torch.Tensor,
            source_points: torch.Tensor,
            dest_points: torch.Tensor,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            pass

    else:

        def forward(
            self,
            image: torch.Tensor,
            source_points: torch.Tensor,
            dest_points: torch.Tensor,
        ) -> Any:
            return sparse_image_warp(
                image,
                source_points,
                dest_points,
                self.indexing,
                self.field_interpolation_order,
                self.field_regularization_weight,
                self.field_full_matrix,
                self.pinned_boundary_points,
                self.dense_interpolation_mode,
                self.dense_padding_mode,
                self.include_flow,
            )


class PolyharmonicSpline(torch.nn.Module):
    """Guess values at query points using a learned polyharmonic spline

    A spline estimates a function ``f : points -> values`` from a fixed number of
    training points/knots and the values of ``f`` at those points. It does that by
    solving a series of piecewise linear equations between knots such that the values at
    the knots match the given values (and some additional constraints depending on the
    spline).

    This module is based on the `interpolate_spline
    <https://www.tensorflow.org/addons/api_docs/python/tfa/image/interpolate_spline>`__
    function from Tensorflow, which implements a `Polyharmonic Spline
    <https://en.wikipedia.org/wiki/Polyharmonic_spline>`__. For technical details,
    consult the TF documentation.

    The call signature of this module, once instantiated, is::

        query_values = polyharmonic_spline(
            train_points, train_values, query_points, query_values
        )
    
    `train_points` is tensor of shape ``(N, T, I)`` representing the training
    points/knots for ``N`` different functions. ``N`` is the batch dimension, ``T`` is
    the number of training points, and ``I`` is the size of the vector input to ``f``.
    `train_values` is a float tensor of shape ``(N, T, O)`` of ``f`` evaluated on
    `train_points`. ``O`` is the size of the output vector of ``f``. `query_points` is
    a tensor of shape ``(N, Q, I)`` representing the points you wish to have
    estimates for. ``Q`` is the number of such points. `query_values` is a tensor of
    shape ``(N, Q, O)`` consisting of the values estimated by the spline

    Parameters
    ----------
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
        This module can return a :class`RuntimeError` when no unique spline can be
        estimated. In general, the spline will require at least ``I+1`` non-degenerate
        points (linearly independent). See the Wikipedia entry on splnes for more info.
    """

    __constants__ = ["order", "regularization_weight", "full_matrix"]

    order: int
    regularization_weight: float
    full_matrix: bool

    def __init__(
        self, order: int, regularization_weight: float = 0.0, full_matrix: bool = True
    ):
        super().__init__()
        if order <= 0:
            raise ValueError(f"order must be positive, got {order}")
        self.order = order
        self.regularization_weight = regularization_weight
        self.full_matrix = full_matrix

    def forward(
        self,
        train_points: torch.Tensor,
        train_values: torch.Tensor,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        return polyharmonic_spline(
            train_points,
            train_values,
            query_points,
            self.order,
            self.regularization_weight,
            self.full_matrix,
        )


class DenseImageWarp(torch.nn.Module):
    """Warp an input image with per-pixel flow vectors

    Once initialized, this module is called with the signature::

        warped = dense_image_warp(image, flow)

    `image` is a float tensor of shape ``(N, C, H, W)``, where ``N`` is the batch
    dimension, ``C`` is the channel dimension, ``H`` is the height dimension, and ``W``
    is the width dimension. `flow` is a float tensor of shape ``(N, H, W, 2)``.
    It returns a new image `warped` of shape ``(N, C, H, W)`` such that

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
    indexing : {'hw', 'wh'}, optional
        If `indexing` is ``"hw"``, ``flow[..., 0] = h``, the height index, and
        ``flow[..., 1] = w`` is the width index. If ``"wh"``, ``flow[..., 0] = w``
        and ``flow[..., 1] = h``. The default in TF is ``"hw"``, whereas torch's
        `grid_sample` is ``"wh"``
    mode : {'bilinear', 'nearest'}, optional
        The method of interpolation. Either use bilinear interpolation or the nearest
        pixel value. The TF default is ``"bilinear"``
    padding_mode : {"border", "zeros", "reflection"}
        Controls how points outside of the image boundaries are interpreted.
        ``"border"``: copy points at around the border of the image. ``"zero"``:
        use zero-valued pixels. ``"reflection"``: reflect pixels into the image starting
        from the boundaries.
    """

    __constants__ = ["indexing", "mode", "padding_mode"]

    indexing: str
    mode: str
    padding_mode: str

    def __init__(
        self,
        indexing: str = "hw",
        mode: str = "bilinear",
        padding_mode: str = "border",
    ):
        super().__init__()
        if indexing not in {"hw", "wh"}:
            raise ValueError(f"indexing must be either 'hw' or 'wh', got '{indexing}'")
        if mode not in {"bilinear", "nearest"}:
            raise ValueError(
                f"mode must be either 'bilinear' or 'nearest', got '{mode}'"
            )
        if padding_mode not in {"border", "zeros", "reflection"}:
            raise ValueError(
                "padding_mode must be one of 'border', 'zeros', or 'relection', got "
                f"'{padding_mode}'"
            )
        self.indexing = indexing
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return dense_image_warp(
            image, flow, self.indexing, self.mode, self.padding_mode
        )


class SequentialLogProbabilities(torch.nn.Module):
    r"""Calculate joint log probability of sequences

    Once initialized, this module is called with the signature::

        log_probs = sequential_log_probs(logits, hyp)

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

    Subclasses are called with the following signature:

        lm(hist, prev=None, idx=None)

    `hist` is a long tensor of shape ``(S, N)`` consisting of prefixes up to length
    ``S``. ``hist[:, n]`` is the n-th prefix :math:`(w^{(n)}_0, w^{(n)}_1, \ldots,
    w^{(n)}_{S-1})`.

    If `idx` is not specified, it outputs a float tensor `log_probs` of shape ``(S + 1,
    N, vocab_size)`` where each ``log_probs[s, n, v]`` equals :math:`\log P(w^{(n)}_{s}
    = v | w^{(n)}_{s - 1}, \ldots)`. That is, each distribution over types conditioned
    on each prefix of tokens (``:0``, ``:1``, ``:2``, etc.) is returned.

    If `idx` is specified, it must etiher be an integer or a long tensor of shape
    ``(,)`` or ``(N,)``. The call returns a pair. The first element is `log_probs_idx`
    of shape ``(N, vocab_size)``, where ``log_probs[n, v]`` equals :math:`\log
    P(w^{(n)}_{idx[n]} = v | w^{(n)}_{idx[n]-1}, \ldots)`. That is, the distributions
    over the next type conditioned on token prefixes up to and excluding ``s = idx`` are
    returned. The second element, `in_next`, is discussed in relation to `prev` below.

    The `prev` argument is a dictionary of tensors which represents some additional
    input used in the computation. It may contain static input (e.g. a tensor of encoder
    output in neural machine translation) and/or dynamic input from prior calls to the
    LM (e.g. the previous hidden state in an RNN-based language model). `in_next`, the
    second element in the return pair, will be fed to the next forward call as the
    argument `prev` (assuming the new value for `idx` is `idx + 1`).

    Parameters
    ----------
    vocab_size : int
        The vocabulary size. Controls the size of the final output dimension,
        as well as what values of `hist` are considered in-vocabulary

    Notes
    -----
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
    """

    __constants__ = ["vocab_size"]

    vocab_size: int

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")

    @torch.jit.export
    def update_input(
        self, prev: Dict[str, torch.Tensor], hist: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Update whatever is passed in as input to the language model

        This method is called in the :func:`forward`. The return value should replace
        `prev` with whatever additional information is necessary before
        :func:`calc_idx_log_probs` if it is not already there, such as an initial hidden
        state. The implementation should be robust to repeated calls.
        """
        return prev

    def extra_repr(self) -> str:
        s = "vocab_size={}".format(self.vocab_size)
        return s

    @abc.abstractmethod
    def calc_idx_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor], idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculates log_prob_idx over types at prefix up to and excluding idx

        Subclasses implement this. Values in idx are guaranteed to be between ``[0,
        hist.size(0)]``. Return should be a pair of ``log_prob_idx, in_cur``. Note `idx`
        may be a scalar if all batch indices are the same.
        """
        raise NotImplementedError()

    @torch.jit.export
    def calc_full_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculates log_prob over all prefixes and stacks them on the first dim

        Implemented in :class:`SequentialLanguageModel` as a simple loop. Subclasses
        may overload this function if the result can be calculated more quickly.
        """
        log_probs = []
        for idx in torch.arange(hist.size(0) + 1, device=hist.device):
            log_probs_idx, prev = self.calc_idx_log_probs(hist, prev, idx)
            log_probs.append(log_probs_idx)
        return torch.stack(log_probs, 0)

    if TYPE_CHECKING:

        def forward(
            self,
            hist: torch.Tensor,
            prev: Optional[Dict[str, torch.Tensor]] = None,
            idx: Optional[Union[int, torch.Tensor]] = None,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
            pass

    else:

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
    with, we can call the model's :func:`extract_by_src` method to rearrange/extract
    the relevant values in `prev` or `in_next` in the same way.
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
        prev : dict
            An input/output value for a step of the lm
        src : torch.Tensor
            A tensor of shape ``(N,)`` containing the indices of the old batch index
            (of possibly different size) to extract the new batch elements from.

        Returns
        -------
        new_prev : dict

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
    pairs of histories `hist_a` and `hist_b` into one `new_hist` such that each path
    in the latter is either from `hist_a` or `hist_b`. :func:`mix_by_mask` accomplishes
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
        transformation between `prev_true` and `prev_false` to come up with
        `prev_new`.

        Parameters
        ----------
        prev_true : dict
            The input/output dictionary for the true branch of `mask`
        prev_false : dict
            The input/output dictionary for the false branch of `mask`
        mask : torch.Tensor
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
                        (N - 1 - hist.size(0), B), sos, dtype=torch.long, device=device,
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
                        (N - 1 - min_idx, B), sos, dtype=torch.long, device=device,
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
        hist = hist.masked_fill(hist.eq(sos), -shift)
        hist = hist + shift

    # add the possible extensions to the history
    cur_step = torch.arange(shift, V + shift, dtype=torch.long, device=device)
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
    vrange = torch.arange(V, dtype=torch.int32, device=device)
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
                    torch.zeros(all_children.shape, dtype=logs.dtype, device=device,),
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
                out = torch.where(new_backoff, out + logs[X + G + children], out,)
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

    Missing entries are assumed to have value 0 and missing backoff penalties are
    assumed to have value 1.

    Parameters
    ----------
    vocab_size : int
    sos : int or None, optional
        The start of sequence token. Any prefix with fewer tokens than the maximum order
        of n-grams minus 1 will be prepended up to that length with this token.
    prob_list : sequence or None, optional
        A list of dictionaries whose entry at index ``i`` corresponds to a
        table of ``i+1``-gram probabilities. Keys must all be ids, not strings.
        Unigram keys are just ids; for n > 1 keys are tuples of ids with the
        latest word last. Values in the dictionary of the highest order n-gram
        dictionaries (last in `prob_list`) are the log-probabilities of the
        keys. Lower order dictionaries' values are pairs of log-probability and
        log-backoff penalty. If `prob_list` is not specified, a unigram model
        with a uniform prior will be built

    Notes
    -----
    Initializing an instance from an `prob_list` is expensive. `prob_list` is converted
    to a trie (something like [heafield2011]_) so that it takes up less space in memory,
    which can take some time.

    Rather than re-initializing repeatedly, it is recommended you save and load this
    module's state dict. :func:`load_state_dict` as been overridden to support loading
    different table sizes, avoiding the need for an accurate `prob_list` on
    initialization:

    >>> # first time
    >>> lm = LookupLanguageModel(vocab_size, sos, prob_list)  # slow
    >>> state_dict = lm.state_dict()
    >>> # save state dict, quit, startup, then reload state dict
    >>> lm = LookupLanguageModel(vocab_size, sos)  # fast!
    >>> lm.load_state_dict(state_dict)

    See Also
    --------
    pydrobert.util.parse_arpa_lm
        How to read a pretrained table of n-gram probabilities into
        `prob_list`. The parameter `token2id` should be specified to ensure
        id-based keys.

    Warnings
    --------
    After 0.3.0, `sos` became no longer optional. `pad_sos_to_n` was removed as an
    argument (implicitly true now). `eos` and `oov` were also removed as part of updates
    to :obj:`SequentialLanguageModel`

    JIT scripting is possible with this module, but not tracing.
    """

    __constants__ = ["vocab_size", "sos", "max_ngram", "max_ngram_nodes", "shift"]

    sos: int
    max_ngram: int
    max_ngram_nodes: int
    shift: int

    # XXX(sdrobert): as discussed in [heafield2011], we could potentially speed
    # up computations by keeping track of prefix probs and storing them in
    # case of backoff. This makes sense in a serial case, when we can choose to
    # explore or not explore a path. In a massively parallel environment, I'm
    # not sure it's worth the effort...

    def __init__(
        self, vocab_size: int, sos: int, prob_list: Optional[Sequence[dict]] = None,
    ):
        super().__init__(vocab_size)
        self.sos = sos
        if sos < 0 or sos > vocab_size:
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
            self.max_ngram_nodes = -1  # changed by build_trie
            logs, ids, pointers = self._build_trie(prob_list)
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
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor], idx: torch.Tensor,
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
                for (k, v) in list(prob_list[0].items())
            )
            for n in range(1, self.max_ngram):
                prob_list[n] = dict(
                    (tuple(t + 1 for t in k), v)
                    for (k, v) in list(prob_list[n].items())
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


@script
def hard_optimal_completion_distillation_loss(
    logits: torch.Tensor,
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int] = None,
    include_eos: bool = True,
    batch_first: bool = False,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
    sub_cost: float = 1.0,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    ignore_index: int = -2,
    warn: bool = True,
) -> torch.Tensor:
    """Functional version of HardOptimalCompletionDistillationLoss

    See Also
    --------
    HardOptimalCompletionDistillationLoss
        The :class:`torch.nn.Module` version. Describes the arguments.
    """
    if logits.dim() != 3:
        raise RuntimeError("logits must be 3 dimensional")
    if logits.shape[:-1] != hyp.shape:
        raise RuntimeError("first two dims of logits must match hyp shape")
    if include_eos:
        if eos is not None and ((eos < 0) or (eos >= logits.size(-1))):
            raise RuntimeError(f"If include_eos=True, eos ({eos}) must be a class idx")
        if eos is not None and eos == ignore_index:
            raise RuntimeError(
                f"If include_eos=True, eos cannot equal ignore_index ({eos}"
            )
    optimals = optimal_completion(
        ref,
        hyp,
        eos=eos,
        include_eos=include_eos,
        batch_first=batch_first,
        ins_cost=ins_cost,
        del_cost=del_cost,
        sub_cost=sub_cost,
        padding=ignore_index,
        exclude_last=True,
        warn=warn,
    )
    max_unique_next = optimals.shape[-1]
    logits = logits.unsqueeze(2).expand(-1, -1, max_unique_next, -1)
    logits = logits.contiguous()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        optimals.flatten(),
        weight=weight,
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(optimals)
    padding_mask = optimals.eq(ignore_index)
    no_padding_mask = ~padding_mask
    loss = loss.masked_fill(padding_mask, 0.0).sum(2)
    loss = torch.where(
        no_padding_mask.any(2), loss / no_padding_mask.float().sum(2), loss,
    )
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction != "none":
        raise RuntimeError(f"'{reduction}' is not a valid value for reduction")
    return loss


class HardOptimalCompletionDistillationLoss(torch.nn.Module):
    r"""A categorical loss function over optimal next tokens

    Optimal Completion Distillation (OCD) [sabour2018]_ tries to minimize the train/test
    discrepancy in transcriptions by allowing seq2seq models to generate whatever
    sequences they want, then assigns a per-step loss according to whatever next token
    would set the model on a path that minimizes the edit distance in the future.

    In its "hard" version, the version used in the paper, the OCD loss function is
    simply a categorical cross-entropy loss of each hypothesis token's distribution
    versus those optimal next tokens, averaged over the number of optimal next tokens:

    .. math::

        loss(logits_t) = \frac{-\log Pr(s_t|logits_t)}{|S_t|}

    Where :math:`s_t \in S_t` are tokens from the set of optimal next tokens given
    :math:`hyp_{\leq t}` and `ref`. The loss is decoupled from an exact prefix of `ref`,
    meaning that `hyp` can be longer or shorter than `ref`.

    When called, this loss function has the signature::

        loss(logits, ref, hyp)

    `hyp` is a long tensor of shape ``(max_hyp_steps, batch_size)`` if `batch_first` is
    :obj:`False`, otherwise ``(batch_size, max_hyp_steps)`` that provides the hypothesis
    transcriptions. Likewise, `ref` of shape ``(max_ref_steps, batch_size)`` or
    ``(batch_size, max_ref_steps)`` providing reference transcriptions. `logits` is a
    4-dimensional tensor of shape ``(max_hyp_steps, batch_size, num_classes)`` if
    `batch_first` is :obj:`False`, ``(batch_size, max_hyp_steps, num_classes)``
    otherwise. A softmax over the step dimension defines the per-step distribution over
    class labels.

    Parameters
    ----------
    eos : int or None, optional
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
    weight : torch.Tensor or None, optional
        A float tensor of manual rescaling weight given to each class
    reduction : {'mean', 'none', 'sum'}, optional
        Specifies the reduction to be applied to the output. 'none': no
        reduction will be applied. 'sum': the output will be summed. 'mean':
        the output will be averaged.
    ignore_index : int, optional
        Specify a target value that is ignored and does not contribute to the input
        gradient. Should not be set to `eos` when `include_eos` is :obj:`True`.

    See Also
    --------
    pydrobert.torch.util.optimal_completion
        Used to determine the optimal next token set :math:`S`
    pydrobert.torch.util.random_walk_advance
        For producing a random `hyp` based on `logits` if the underlying
        model producing `logits` is auto-regressive. Also provides an example
        of sampling non-auto-regressive models
    """

    __constants__ = [
        "eos",
        "include_eos",
        "batch_first",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "reduction",
        "ignore_index",
    ]

    eos: Optional[int]
    include_eos: bool
    batch_first: bool
    ins_cost: float
    del_cost: float
    sub_cost: float
    reduction: str
    ignore_index: int

    def __init__(
        self,
        eos: Optional[int] = None,
        include_eos: bool = True,
        batch_first: bool = False,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        sub_cost: float = 1.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.eos = eos
        self.include_eos = include_eos
        self.batch_first = batch_first
        self.ins_cost = ins_cost
        self.del_cost = del_cost
        self.sub_cost = sub_cost
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight)

    def forward(
        self,
        logits: torch.Tensor,
        ref: torch.Tensor,
        hyp: torch.Tensor,
        warn: bool = True,
    ) -> torch.Tensor:
        return hard_optimal_completion_distillation_loss(
            logits,
            ref,
            hyp,
            self.eos,
            self.include_eos,
            self.batch_first,
            self.ins_cost,
            self.del_cost,
            self.sub_cost,
            self.weight,
            self.reduction,
            self.ignore_index,
            warn,
        )


@script
def minimum_error_rate_loss(
    log_probs: torch.Tensor,
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int] = None,
    include_eos: bool = True,
    sub_avg: bool = True,
    batch_first: bool = False,
    norm: bool = True,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
    sub_cost: float = 1.0,
    reduction: str = "mean",
    warn: bool = True,
) -> torch.Tensor:
    """Functional version of MinimumErrorRateLoss

    See Also
    --------
    MinimumErrorRateLoss
        The :class:`torch.nn.Module` version. Describes the arguments
    """
    if log_probs.dim() != 2:
        raise RuntimeError("log_probs must be 2 dimensional")
    if hyp.dim() != 3:
        raise RuntimeError("hyp must be 3 dimensional")
    if ref.dim() not in (2, 3):
        raise RuntimeError("ref must be 2 or 3 dimensional")
    if batch_first:
        batch_size, samples, max_hyp_steps = hyp.shape
        if ref.dim() == 2:
            ref = ref.unsqueeze(1).repeat(1, samples, 1)
        if (ref.shape[:2] != (batch_size, samples)) or (
            ref.shape[:2] != log_probs.shape
        ):
            raise RuntimeError(
                "ref and hyp batch_size and sample dimensions must match"
            )
        max_ref_steps = ref.size(-1)
        ref = ref.view(-1, max_ref_steps)
        hyp = hyp.view(-1, max_hyp_steps)
    else:
        max_hyp_steps, batch_size, samples = hyp.shape
        if ref.dim() == 2:
            ref = ref.unsqueeze(-1).repeat(1, 1, samples)
        if (ref.shape[1:] != (batch_size, samples)) or (
            ref.shape[1:] != log_probs.shape
        ):
            raise RuntimeError(
                "ref and hyp batch_size and sample dimensions must match"
            )
        max_ref_steps = ref.size(0)
        ref = ref.view(max_ref_steps, -1)
        hyp = hyp.view(max_hyp_steps, -1)
    if samples < 2:
        raise RuntimeError(f"Batch must have at least two samples, got {samples}")
    er = error_rate(
        ref,
        hyp,
        eos=eos,
        include_eos=include_eos,
        norm=norm,
        batch_first=batch_first,
        ins_cost=ins_cost,
        del_cost=del_cost,
        sub_cost=sub_cost,
        warn=warn,
    ).view(batch_size, samples)
    if sub_avg:
        er = er - er.mean(1, keepdim=True)
    loss = er * torch.nn.functional.softmax(log_probs, 1)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction != "none":
        raise RuntimeError(f"'{reduction}' is not a valid value for reduction")
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

        loss(log_probs, ref, hyp, warn=True)

    `log_probs` is a tensor of shape ``(batch_size, samples)`` providing the log joint
    probabilities of every path. `hyp` is a long tensor of shape ``(max_hyp_steps,
    batch_size, samples)`` if `batch_first` is :obj:`False` otherwise ``(batch_size,
    samples, max_hyp_steps)`` that provides the hypothesis transcriptions. `ref` is a 2-
    or 3-dimensional tensor. If 2D, it is of shape ``(max_ref_steps, batch_size)`` (or
    ``(batch_size, max_ref_steps)``). Alternatively, `ref` can be of shape
    ``(max_ref_steps, batch_size, samples)`` or ``(batch_size, samples,
    max_ref_steps)``.

    If `ref` is 2D, the loss is calculated as

    .. math::

        loss_{MER} = SoftMax(log\_probs)[ER(hyp_i, ref) - \mu_i]

    where :math:`\mu_i` is the average error rate along paths in the batch element
    :math:`i`. :math:`mu_i` can be removed by setting `sub_avg` to :obj:`False`. Note
    that each hypothesis is compared against the same reference as long as the batch
    element remains the same

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

    A previous version of this module incorporated a Maximum Likelihood Estimate (MLE)
    into the loss as in [prabhavalkar2018]_, which required `logits` instead of
    `log_probs`. This was overly complicated, given the user can easily incorporate the
    additional loss term herself by using :class:`torch.nn.CrossEntropyLoss`. Take a
    look at the example below for how to recreate this

    Examples
    --------

    Assume here that `logits` is the output of some neural network, and that `hyp` has
    somehow been produced from that (e.g. a beam search or random walk). We combine this
    loss function with a cross-entropy/MLE term to sort-of recreate [prabhavalkar2018]_.

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

    __constants__ = [
        "eos",
        "include_eos",
        "sub_avg",
        "batch_first",
        "norm",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "reduction",
    ]

    eos: Optional[int]
    include_eos: bool
    sub_avg: bool
    norm: bool
    ins_cost: float
    del_cost: float
    sub_cost: float
    reduction: str

    def __init__(
        self,
        eos: Optional[int] = None,
        include_eos: bool = True,
        sub_avg: bool = True,
        batch_first: bool = False,
        norm: bool = True,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        sub_cost: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.eos = eos
        self.include_eos = include_eos
        self.sub_avg = sub_avg
        self.batch_first = batch_first
        self.norm = norm
        self.ins_cost = ins_cost
        self.del_cost = del_cost
        self.sub_cost = sub_cost
        self.reduction = reduction

    def forward(
        self,
        log_probs: torch.Tensor,
        ref: torch.Tensor,
        hyp: torch.Tensor,
        warn: bool = True,
    ) -> torch.Tensor:
        return minimum_error_rate_loss(
            log_probs,
            ref,
            hyp,
            self.eos,
            self.include_eos,
            self.sub_avg,
            self.batch_first,
            self.norm,
            self.ins_cost,
            self.del_cost,
            self.sub_cost,
            self.reduction,
            warn,
        )


class GlobalSoftAttention(torch.nn.Module, metaclass=abc.ABCMeta):
    r"""Parent class for soft attention mechanisms on an entire input sequence

    Global soft attention mechansims [bahdanau2015]_ are a way of getting rid
    of one variable-length sequence dimension ``T`` in an input `key` using a
    weighted sum of a tensor `value` that is informed by some other tensor,
    `query`. The weights are dictated by the function ``score(query, key)``.
    Usually, this is in the context of encoder-decoder architectures, which is
    explained here.

    Assume `query` is a tensor of shape ``(batch_size, query_size)`` representing a
    single hidden state of a decoder RNN. Assume `key` is a tensor of shape ``(T,
    batch_size, key_size)`` representing the encoder output, ``dim == 0`` to specify
    that the variable-length dimension of `key` is the zero-th dimension, and ``value ==
    key``. The output `out` will be a tensor of shape ``(batch_size, key_size)``.
    Letting :math:`t` index the `dim`-th dimension:

        .. math::

            out = \sum_t a_t value_t

    ``a`` is the attention vector. In our example, ``a`` will be of shape ``(T,
    batch_size)``. ``a`` is the result of a softmax over the `dim`-th dimension of
    another tensor ``e`` of shape ``(T, batch_size)`` with an optional `mask`

    .. math::

        a = softmax(e * mask - (1 - mask) \infty, dim)

    `mask` (if specified) is of shape ``(T, batch_size)`` and will set ``a`` to zero
    wherever the mask is zero. `mask` can be used to indicate padded values when `key`
    consists of variable-length sequences.

    ``e`` is the result of a score function over `key` and `query`

    .. math::

        e = score(query, key)

    ``score()`` is implemented by subclasses of :class:`GlobalSoftAttention`

    The signature when calling an instance this module is::

        attention(query, key, value[, mask])

    Parameters
    ----------
    query_size : int
        The length of the last dimension of the `query` argument
    key_size : int
        The length of the last dimension of the `key` argument
    dim : int, optional
        The sequence dimension of the `key` argument

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

    __constants__ = ["query_size", "key_size", "dim"]

    query_size: int
    key_size: int
    dim: int

    def __init__(self, query_size: int, key_size: int, dim: int = 0):
        super().__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.dim = dim

    @abc.abstractmethod
    def score(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Calculate the score function over the entire input

        This is implemented by subclasses of :class:`GlobalSoftAttention`

        ``query.unsqueeze(self.dim)[..., 0]`` broadcasts with ``value[...,
        0]``. The final dimension of `query` is of length ``self.query_size``
        and the final dimension of `key` should be of length ``self.key_size``

        Parameters
        ----------
        query : torch.Tensor
        key : torch.Tensor

        Returns
        -------
        e : torch.Tensor
            Of the same shape as the above broadcasted tensor
        """
        raise NotImplementedError()

    @torch.jit.export
    def check_input(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Check if input is properly formatted, RuntimeError otherwise

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
        if self.dim > key_dim - 2 or key_dim == -1 or self.dim < -key_dim + 1:
            raise RuntimeError(
                f"dim must be in the range [{-key_dim + 1}, {key_dim - 2}] and not -1"
            )
        e_shape = broadcast_shapes(query.unsqueeze(self.dim).shape[:-1], key.shape[:-1])
        if mask is not None:
            broadcast_shapes(e_shape, mask.shape)
        broadcast_shapes(e_shape + (1,), value.shape)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            # tracing can't handle calls with None arguments, so we make a
            # non-threatening mask to call with
            mask_ = torch.ones(
                (1,) * (key.dim() - 1), device=query.device, dtype=torch.bool
            )
            self.check_input(query, key, value, mask_)
        else:
            self.check_input(query, key, value, mask)
        e = self.score(query, key)
        if mask is not None:
            e = e.masked_fill(~mask, -float("inf"))
        a = torch.nn.functional.softmax(e, self.dim)
        return (a.unsqueeze(-1) * value).sum(self.dim)

    def extra_repr(self) -> str:
        return "query_size={}, key_size={}, dim={}".format(
            self.query_size, self.key_size, self.dim
        )

    def reset_parameters(self) -> None:
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

    See Also
    --------
    GlobalSoftAttention
        For a description of how to call this module, how it works, etc.
    """

    __constants__ = ["query_size", "key_size", "dim", "scale_factor"]

    scale_factor: float

    def __init__(self, size: int, dim: int = 0, scale_factor: float = 1.0):
        super().__init__(size, size, dim)
        self.scale_factor = scale_factor

    def score(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        query = query.unsqueeze(self.dim)
        return (query * key).sum(-1) * self.scale_factor

    def extra_repr(self) -> str:
        return super().extra_repr() + f", scale_factor={self.scale_factor}"


class GeneralizedDotProductSoftAttention(GlobalSoftAttention):
    r"""Dot product soft attention with a learned matrix in between

    The "general" score function from [luong2015]_, the score function for this
    attention mechanism is

    .. math::

        e = \sum_q query_q \sum_k W_{qk} key_k

    For some learned matrix :math:`W`. :math:`q` indexes the last dimension of `query`
    and :math:`k` the last dimension of `key`

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
    """

    weight: torch.Tensor

    def __init__(
        self, query_size: int, key_size: int, dim: int = 0, bias: bool = False
    ):
        super().__init__(query_size, key_size, dim)
        self.weight = torch.nn.parameter.Parameter(torch.empty(query_size, key_size))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.empty(query_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def score(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        Wkey = torch.nn.functional.linear(key, self.weight, self.bias)
        query = query.unsqueeze(self.dim)
        return (query * Wkey).sum(-1)

    reset_parameters = torch.jit.unused(torch.nn.Linear.reset_parameters)


class ConcatSoftAttention(GlobalSoftAttention):
    r"""Attention where query and key are concatenated, then fed into an MLP

    Proposed in [luong2015]_, though quite similar to that proposed in [bahdanau2015]_,
    the score function for this layer is:

    .. math::

        e = \sum_i v_i \tanh(\sum_c W_{ic} [query, key]_c)

    For some learned matrix :math:`W` and vector :math:`v`, where :math:`[query, key]`
    indicates concatenation along the last axis. `query` and `key` will be expanded to
    fit their broadcast dimensions. :math:`W` has shape ``(inter_size, key_size)`` and
    :math:`v` has shape ``(hidden_size,)``

    Parameters
    ----------
    query_size : int
    key_size : int
    dim : int, optional
    bias : bool, optional
        Whether to add bias term ``b`` :math:`W [query, key] + b`
    hidden_size : int, optional

    See Also
    --------
    GlobalSoftAttention
        For a description of how to call this module, how it works, etc.
    """

    weight: torch.Tensor
    v: torch.Tensor

    def __init__(
        self,
        query_size: int,
        key_size: int,
        dim: int = 0,
        bias: bool = False,
        hidden_size: int = 1000,
    ):
        super().__init__(query_size, key_size, dim)
        self.weight = torch.nn.parameter.Parameter(
            torch.empty(hidden_size, query_size + key_size)
        )
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter("bias", None)
        # there's no point in a bias for v. It'll just be absorbed by the
        # softmax later. You could add a bias after the tanh layer, though...
        self.v = torch.nn.parameter.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def score(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        query = query.unsqueeze(self.dim)
        query_wo_last, key_wo_last = torch.broadcast_tensors(query[..., 0], key[..., 0])
        query, _ = torch.broadcast_tensors(query, query_wo_last.unsqueeze(-1))
        key, _ = torch.broadcast_tensors(key, key_wo_last.unsqueeze(-1))
        cat = torch.cat([query, key], -1)
        Wcat = torch.nn.functional.linear(cat, self.weight, self.bias)
        tanhWcat = torch.tanh(Wcat)
        return torch.nn.functional.linear(tanhWcat, self.v.unsqueeze(0), None).squeeze(
            -1
        )

    def reset_parameters(self) -> None:
        torch.nn.Linear.reset_parameters(self)
        torch.nn.init.normal_(self.v)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", hidden_size={self.v.size(0)}"


class MultiHeadedAttention(GlobalSoftAttention):
    r"""Perform attention over a number of heads, concatenate, and project

    Multi-headed attention was proposed in [vaswani2017]_. It can be considered a
    wrapper around standard :class:`GlobalSoftAttention` that also performs
    :class:`GlobalSoftAttention`, but with more parameters. The idea is to replicate
    transformed versions of the `query`, `key`, and `value` `num_heads` times. Letting
    :math:`h` index the head:

    .. math::

        query_h = W^Q_h query \\
        key_h = W^K_h key \\
        value_h = W^V_h value

    If `query` is of shape ``(..., query_size)``, :math:`W^Q_h` is a learned matrix of
    shape ``(query_size, d_q)`` that acts on the final dimension of `query`. Likewise,
    :math:`W^K_h` is of shape ``(key_size, d_k)`` and :math:`W^V_h` is of shape
    ``(value_size, d_v)``. Note here that the last dimension of `value` must also be
    provided in `value_size`, unlike in other attention layers.

    Each head is then determined via a wrapped :class:`GlobalSoftAttention` instance,
    `single_head_attention`:

    .. math::

        head_h = single\_head\_attention(query_h, key_h, value_h, mask)

    Where `mask` is repeated over all :math:`h`.

    Since each :math:`head_h` has the same shape, they can be concatenated along the
    last dimension to get the tensor :math:`cat` of shape ``(..., d_v * num_heads)``,
    which is linearly transformed into the output

    .. math::

        out = W^C cat

    With a learnable matrix :math:`W^C` of shape ``(d_v * num_heads, out_size)``. `out`
    has a shape ``(..., out_size)``

    This module has the following signature when called

        attention(query, key, value[, mask])

    Parameters
    ----------
    query_size : int
        The size of the last dimension of the `query` being passed to this module (not
        the size of a head's query).
    key_size : int
        The size of the last dimension of the `key` being passed to this module (not the
        size of a head's key).
    value_size : int
        The size of the last dimension of the `value` being passed to this module (not
        the size of a head's value).
    num_heads : int
        The number of heads to spawn.
    single_head_attention : GlobalSoftAttention
        An instance of a subclass of :class:`GlobalSoftAttention` responsible for
        processing a head. `single_head_attention` attention will be used to derive the
        sequence dimension (``dim``) of `key` via ``single_head_attention.dim``, the
        size of a head's query ``d_k`` via ``single_head_attention.query_size``, and the
        size of a head's key via ``single_head_attention.key_size``
    out_size : int, optional
        The size of the last dimension of `out`. If unset, the default is to match
        `value_size`
    d_v : int, optional
        The size of the last dimension of a head's value. If unset, will default to
        ``max(1, value_size // num_heads)``
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

    __constants__ = [
        "query_size",
        "key_size",
        "dim",
        "value_size",
        "num_heads",
        "out_size",
        "d_v",
    ]

    value_size: int
    num_heads: int
    out_size: int
    d_v: int

    def __init__(
        self,
        query_size: int,
        key_size: int,
        value_size: int,
        num_heads: int,
        single_head_attention: GlobalSoftAttention,
        out_size: Optional[int] = None,
        d_v: Optional[int] = None,
        bias_WQ: bool = False,
        bias_WK: bool = False,
        bias_WV: bool = False,
        bias_WC: bool = False,
    ):
        super().__init__(query_size, key_size, dim=single_head_attention.dim)
        if self.dim < 0:
            raise ValueError(
                "Negative dimensions are ambiguous for multi-headed attention"
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

    def check_input(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        # FIXME(sdrobert): TorchScript doesn't currently support calls to super().
        # Replace this when it does. Also surround broadcast_shapes with try/catch
        # when supported
        key_dim = key.dim()
        if query.dim() != key_dim - 1:
            raise RuntimeError("query must have one fewer dimension than key")
        if key_dim != value.dim():
            raise RuntimeError("key must have same number of dimensions as value")
        if query.shape[-1] != self.query_size:
            raise RuntimeError("Last dimension of query must match query_size")
        if key.shape[-1] != self.key_size:
            raise RuntimeError("Last dimension of key must match key_size")
        if self.dim > key_dim - 2 or key_dim == -1 or self.dim < -key_dim + 1:
            raise RuntimeError(
                f"dim must be in the range [{-key_dim + 1}, {key_dim - 2}] and not -1"
            )
        e_shape = broadcast_shapes(query.unsqueeze(self.dim).shape[:-1], key.shape[:-1])
        if mask is not None:
            broadcast_shapes(e_shape, mask.shape)
        broadcast_shapes(e_shape + (1,), value.shape)
        if value.size(-1) != self.value_size:
            raise RuntimeError("Last dimension of value must match value_size")

    def score(self, query: torch.Tensor, key: torch.Tensor) -> NoReturn:
        raise NotImplementedError(
            "In MultiHeadedAttention, score() is handled by single_head_attention"
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            # avoid issues with calls with None
            # if the dimension is correct, a tensor of shape (1, ...) should always
            # broadcast
            mask = torch.ones(
                (1,) * (key.dim() - 1), device=query.device, dtype=torch.bool
            )
        self.check_input(query, key, value, mask)
        query_shape = query.shape
        key_shape = key.shape
        value_shape = value.shape
        query_heads = self.WQ(query).view(
            (query_shape[:-1] + (self.num_heads, self.d_q))
        )
        key_heads = self.WK(key).view((key_shape[:-1] + (self.num_heads, self.d_k)))
        value_heads = self.WV(value).view(
            (value_shape[:-1] + (self.num_heads, self.d_v))
        )
        if mask is not None:
            mask = mask.unsqueeze(-2)
        cat = self.single_head_attention(query_heads, key_heads, value_heads, mask)
        cat = cat.view((cat.shape[:-2] + (self.num_heads * self.d_v,)))
        return self.WC(cat)

    def reset_parameters(self) -> None:
        self.WQ.reset_parameters()
        self.WK.reset_parameters()
        self.WV.reset_parameters()
        self.WC.reset_parameters()
        self.single_head_attention.reset_parameters()

    def extra_repr(self) -> str:
        s = super().extra_repr()
        # rest of info in single_head_attention submodule
        s += ", value_size={}, out_size={}, num_heads={}".format(
            self.value_size, self.out_size, self.num_heads
        )
        return s


@script
def _spec_augment_check_input(
    feats: torch.Tensor, lengths: Optional[torch.Tensor] = None
):
    if feats.dim() != 3:
        raise RuntimeError(
            f"Expected feats to have three dimensions, got {feats.dim()}"
        )
    N, T, _ = feats.shape
    if lengths is not None:
        if lengths.dim() != 1:
            raise RuntimeError(
                f"Expected lengths to be one dimensional, got {lengths.dim()}"
            )
        if lengths.size(0) != N:
            raise RuntimeError(
                f"Batch dimension of feats ({N}) and lengths ({lengths.size(0)}) "
                "do not match"
            )
        if not torch.all((lengths <= T) & (lengths > 0)):
            raise RuntimeError(f"values of lengths must be between (1, {T})")


SpecAugmentParams = Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]


@script
def spec_augment_draw_parameters(
    feats: torch.Tensor,
    max_time_warp: float,
    max_freq_warp: float,
    max_time_mask: int,
    max_freq_mask: int,
    max_time_mask_proportion: float,
    num_time_mask: int,
    num_time_mask_proportion: float,
    num_freq_mask: int,
    lengths: Optional[torch.Tensor] = None,
) -> SpecAugmentParams:
    """Functional version of SpecAugment.draw_parameters

    See Also
    --------
    SpecAugment
        For definitions of arguments and a description of this function.
    """
    _spec_augment_check_input(feats, lengths)
    N, T, F = feats.shape
    device = feats.device
    eps = _get_tensor_eps(feats)
    omeps = 1 - eps
    if lengths is None:
        lengths = torch.full((N,), T, dtype=torch.long, device=device)
    lengths = lengths.to(device)
    # note that order matters slightly in whether we draw widths or positions first.
    # The paper specifies that position is drawn first for warps, whereas widths
    # are drawn first for masks
    if max_time_warp:
        # we want the range (W, length - W) exclusive to be where w_0 can come
        # from. If W >= length / 2, this is impossible. Rather than giving up,
        # we limit the maximum length to W < length / 2
        max_ = torch.clamp(lengths.float() / 2 - eps, max=max_time_warp)
        w_0 = torch.rand([N], device=device) * (lengths - 2 * (max_ + eps)) + max_ + eps
        w = torch.rand([N], device=device) * (2 * max_) - max_
    else:
        w_0 = w = torch.empty(0)
    if max_freq_warp:
        max_ = min(max_freq_warp, F / 2 - eps)
        v_0 = torch.rand([N], device=device) * (F - 2 * (max_ + eps)) + max_ + eps
        v = torch.rand([N], device=device) * (2 * max_) - max_
    else:
        v_0 = v = torch.empty(0)
    if (
        max_time_mask
        and max_time_mask_proportion
        and num_time_mask
        and num_time_mask_proportion
    ):
        lengths = lengths.float()
        max_ = (
            torch.clamp(lengths * max_time_mask_proportion, max=max_time_mask,)
            .floor()
            .to(device)
        )
        nums_ = (
            torch.clamp(lengths * num_time_mask_proportion, max=num_time_mask,)
            .floor()
            .to(device)
        )
        t = (
            (
                torch.rand([N, num_time_mask], device=device)
                * (max_ + omeps).unsqueeze(1)
            )
            .long()
            .masked_fill(
                nums_.unsqueeze(1)
                <= torch.arange(num_time_mask, dtype=lengths.dtype, device=device),
                0,
            )
        )
        t_0 = (
            torch.rand([N, num_time_mask], device=device)
            * (lengths.unsqueeze(1) - t + omeps)
        ).long()
    else:
        t = t_0 = torch.empty(0)
    if max_freq_mask and num_freq_mask:
        max_ = min(max_freq_mask, F)
        f = (torch.rand([N, num_freq_mask], device=device) * (max_ + omeps)).long()
        f_0 = (torch.rand([N, num_freq_mask], device=device) * (F - f + omeps)).long()
    else:
        f = f_0 = torch.empty(0)
    return w_0, w, v_0, v, t_0, t, f_0, f


@script
def spec_augment_apply_parameters(
    feats: torch.Tensor,
    params: SpecAugmentParams,
    interpolation_order: int,
    lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Functional version of SpecAugment.apply_parameters

    See Also
    --------
    SpecAugment
        For definitions of arguments and a description of this function.
    """
    _spec_augment_check_input(feats, lengths)
    N, T, F = feats.shape
    device = feats.device
    if lengths is None:
        lengths = torch.full((N,), T, dtype=torch.long, device=device)
    lengths = lengths.to(feats.dtype)
    w_0, w, v_0, v, t_0, t, f_0, f = params
    new_feats = feats
    time_grid: Optional[torch.Tensor] = None
    freq_grid: Optional[torch.Tensor] = None
    do_warp = False
    if w_0 is not None and w_0.numel() and w is not None and w.numel():
        time_grid = warp_1d_grid(w_0, w, lengths, T, interpolation_order)
        do_warp = True
    if v_0 is not None and v_0.numel() and v is not None and v.numel():
        freq_grid = warp_1d_grid(
            v_0,
            v,
            torch.full((N,), F, dtype=torch.long, device=device),
            F,
            interpolation_order,
        )
        do_warp = True
    if do_warp:
        if time_grid is None:
            time_grid = torch.arange(T, device=device, dtype=torch.float)
            time_grid = (2 * time_grid + 1) / T - 1
            time_grid = time_grid.unsqueeze(0).expand(N, T)
        if freq_grid is None:
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
    tmask: Optional[torch.Tensor] = None
    fmask: Optional[torch.Tensor] = None
    if t_0 is not None and t_0.numel() and t is not None and t.numel():
        tmask = torch.arange(T, device=device).unsqueeze(0).unsqueeze(2)  # (1, T,1)
        t_1 = t_0 + t  # (N, MT)
        tmask = (tmask >= t_0.unsqueeze(1)) & (tmask < t_1.unsqueeze(1))  # (N,T,MT)
        tmask = tmask.any(2, keepdim=True)  # (N, T, 1)
    if f_0 is not None and f_0.numel() and f is not None and f.numel():
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


@script
def spec_augment(
    feats: torch.Tensor,
    max_time_warp: float,
    max_freq_warp: float,
    max_time_mask: int,
    max_freq_mask: int,
    max_time_mask_proportion: float,
    num_time_mask: int,
    num_time_mask_proportion: float,
    num_freq_mask: int,
    interpolation_order: int,
    lengths: Optional[torch.Tensor] = None,
    training: bool = True,
) -> torch.Tensor:
    """Functional version of SpecAugment

    See Also
    --------
    SpecAugment
        For definitions of arguments and a description of this function.
    """
    _spec_augment_check_input(feats, lengths)
    if not training:
        return feats
    params = spec_augment_draw_parameters(
        feats,
        max_time_warp,
        max_freq_warp,
        max_time_mask,
        max_freq_mask,
        max_time_mask_proportion,
        num_time_mask,
        num_time_mask_proportion,
        num_freq_mask,
        lengths,
    )
    return spec_augment_apply_parameters(feats, params, interpolation_order, lengths)


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

    The original SpecAugment implementation only performs steps 1, 3, and 4; step 2 is a
    trivial extension.

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

    __constants__ = [
        "max_time_warp",
        "max_freq_warp",
        "max_time_mask",
        "max_freq_mask",
        "max_time_mask_proportion",
        "num_time_mask",
        "num_time_mask_proportion",
        "num_freq_mask",
        "interpolation_order",
    ]

    max_time_warp: float
    max_freq_warp: float
    max_time_mask: int
    max_freq_mask: int
    max_time_mask_proportion: float
    num_time_mask: int
    num_time_mask_proportion: float
    num_freq_mask: int
    interpolation_order: int

    def __init__(
        self,
        max_time_warp: float = 80.0,
        max_freq_warp: float = 0.0,
        max_time_mask: int = 100,
        max_freq_mask: int = 27,
        max_time_mask_proportion: float = 0.04,
        num_time_mask: int = 20,
        num_time_mask_proportion: float = 0.04,
        num_freq_mask: int = 2,
        interpolation_order: int = 1,
    ):
        super().__init__()
        self.max_time_warp = float(max_time_warp)
        self.max_freq_warp = float(max_freq_warp)
        self.max_time_mask = max_time_mask
        self.max_freq_mask = max_freq_mask
        self.max_time_mask_proportion = max_time_mask_proportion
        self.num_time_mask = num_time_mask
        self.num_time_mask_proportion = num_time_mask_proportion
        self.num_freq_mask = num_freq_mask
        self.interpolation_order = interpolation_order

    def extra_repr(self) -> str:
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

    def draw_parameters(
        self, feats: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> SpecAugmentParams:
        """Randomly draw parameterizations of augmentations

        Called as part of this layer's :func:`__call__` method.

        Parameters
        ----------
        feats : torch.Tensor
            Time-frequency features of shape ``(N, T, F)``.
        lengths : torch.Tensor or None, optional
            Long tensor of shape ``(N,)`` containing the number of frames before
            padding.

        Returns
        -------
        w_0 : torch.Tensor
            If step 1 is enabled, of shape ``(N,)`` containing the source points in the
            time warp (floatint-point). Otherwise, is empty.
        w : torch.Tensor
            If step 1 is enabled, of shape ``(N,)`` containing the number of frames to
            shift the source point by (positive or negative) in the destination in time.
            Positive values indicate a right shift. Otherwise is empty.
        v_0 : torch.Tensor
            If step 2 is enabled, of shape ``(N,)`` containing the source points in the
            frequency warp (floating point). Otherwise is empty.
        v : torch.Tensor
            If step 2 is enabled, of shape ``(N,)`` containing the number of
            coefficients to shift the source point by (positive or negative) in the
            destination in time. Positive values indicate a right shift. Otherwise is
            empty.
        t_0 : torch.Tensor
            If step 3 is enabled, of shape ``(N, M_T)`` where ``M_T`` is the number of
            time masks specifying the lower index (inclusive) of the time masks.
            Otherwise is empty.
        t : torch.Tensor
            If step 3 is enabled, of shape ``(N, M_T)`` specifying the number of frames
            per time mask. Otherise is empty.
        f_0 : torch.Tensor
            If step 4 is enabled, of shape ``(N, M_F)`` where ``M_F`` is the number of
            frequency masks specifying the lower index (inclusive) of the frequency
            masks. Otherwise is empty.
        f : torch.Tensor
            If step 4 is enabled, of shape ``(N, M_F)`` specifying the number of
            frequency coefficients per frequency mask. Otherwise is empty.
        """
        return spec_augment_draw_parameters(
            feats,
            self.max_time_warp,
            self.max_freq_warp,
            self.max_time_mask,
            self.max_freq_mask,
            self.max_time_mask_proportion,
            self.num_time_mask,
            self.num_time_mask_proportion,
            self.num_freq_mask,
            lengths,
        )

    def apply_parameters(
        self,
        feats: torch.Tensor,
        params: SpecAugmentParams,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Use drawn parameters to apply augmentations

        Called as part of this layer's :func:`__call__` method.

        Parameters
        ----------
        feats : torch.Tensor
            Time-frequency features of shape ``(N, T, F)``.
        params : sequence of torch.Tensor
            All parameter tensors returned by :func:`draw_parameters`.
        lengths : torch.Tensor, optional
            Tensor of shape ``(N,)`` containing the number of frames before padding.

        Returns
        -------
        new_feats : torch.Tensor
            Augmented time-frequency features of same shape as `feats`.
        """
        return spec_augment_apply_parameters(
            feats, params, self.interpolation_order, lengths
        )

    def reset_parameters(self) -> None:
        pass

    def forward(
        self, feats: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if lengths is None:
            # _spec_augment_check_input(feats)
            lengths = torch.full(
                (feats.size(0),), feats.size(1), dtype=torch.long, device=feats.device
            )
        if not self.training:
            return feats
        params = self.draw_parameters(feats, lengths)
        return self.apply_parameters(feats, params, lengths)


@script
def random_shift(
    in_: torch.Tensor,
    in_lens: torch.Tensor,
    prop: Tuple[float, float],
    mode: str,
    value: float,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional version of RandomShift

    See Also
    --------
    RandomShift
        For definitions of arguments and a description of this function
    """
    if in_.dim() < 2:
        raise RuntimeError(f"in_ must be at least 2 dimensional")
    if in_lens.dim() != 1 or in_lens.size(0) != in_.size(0):
        raise RuntimeError(
            f"For in_ of shape {in_.shape}, expected in_lens to be of shape "
            f"({in_.size(0)}), got {in_lens.shape}"
        )
    if training:
        in_lens_ = in_lens.float()
        pad = torch.stack([prop[0] * in_lens_, prop[1] * in_lens_])
        pad *= torch.rand_like(pad)
        pad = pad.long()
        out_lens = in_lens + pad.sum(0)
        return pad_variable(in_, in_lens, pad, mode, value), out_lens
    else:
        return in_, in_lens


class RandomShift(torch.nn.Module):
    """Pad to the left and right of each sequence by a random amount

    This layer is intended for training models which are robust to small shifts in some
    variable-length sequence dimension (e.g. speech recognition). It pads each input
    sequence with some number of elements at its beginning and end. The number of
    elements is randomly chosen but bounded above by some proportion of the input length
    specified by the user. Its call signature is

        out, out_lens = layer(in_, in_lens)

    Where: `in_` is a tensor of shape ``(N, T, *)`` where ``N`` is the batch dimension
    and ``T`` is the sequence dimension; `in_lens` is a long tensor of shape ``(N,)``;
    `out` is a tensor of the same type as `in_` of shape ``(N, T', *)``; and `out_lens`
    is of shape ``(N,)``. The ``n``-th input sequence is stored in the range
    ``in_[n, :in_lens[n]]``. The padded ``n``-th sequence is stored in the range
    ``out[n, :out_lens[n]]``. Values outside of these ranges are undefined.

    The amount of padding is dictated by the parameter `prop` this layer is initialized
    with. A proportion is a non-negative float dictating the maximum ratio of the
    original sequence length which may be padded, exclusive. `prop` can be a pair
    ``left, right`` for separate ratios of padding at the beginning and end of a
    sequence, or just one float if the proportions are the same. For example,
    ``prop=0.5`` of a sequence of length ``10`` could result in a sequence of length
    between ``10`` and ``18`` inclusive since each side of the sequence could be padded
    with ``0-4`` elements (``0.5 * 10 = 5`` is an exclusive bound).

    Padding is only applied if this layer is in training mode. If testing,
    ``out, out_lens = in_, in_lens``.

    Parameters
    ----------
    prop : float or tuple
    mode : {'reflect', 'constant', 'replicate'}, optional
        The method with which to pad the input sequence.
    value : float, optional
        The constant with which to pad the sequence if `mode` is set to
        :obj:`'constant'`.

    Raises
    ------
    NotImplementedError
        On initialization if `mode` is :obj:`'reflect'` and a value in `prop` exceeds
        ``1.0``. Reflection currently requires the amount of padding does not exceed
        the original sequence length.

    See Also
    --------
    pydrobert.torch.util.pad_variable
        For more details on the different types of padding. Note the default `mode` is
        different between this and the function.
    """

    __constants__ = ["prop", "mode", "value"]

    prop: Tuple[float, float]
    mode: str
    value: float

    def __init__(
        self,
        prop: Union[float, Tuple[float, float]],
        mode: str = "reflect",
        value: float = 0.0,
    ):
        super().__init__()
        try:
            prop = (float(prop), float(prop))
        except TypeError:
            prop = tuple(prop)
        if len(prop) != 2:
            raise ValueError(
                f"prop must be a single or pair of floating points, got '{prop}'"
            )
        if prop[0] < 0.0 or prop[1] < 0.0:
            raise ValueError("prop values must be non-negative")
        if mode == "reflect":
            if prop[0] > 1.0 or prop[1] > 1.0:
                raise NotImplementedError(
                    "if 'mode' is 'reflect', values in 'prop' must be <= 1"
                )
        elif mode not in {"constant", "replicate"}:
            raise ValueError(
                "'mode' must be one of 'reflect', 'constant', or 'replicate', got "
                f"'{mode}'"
            )
        self.mode = mode
        self.prop = prop
        self.value = value

    def extra_repr(self) -> str:
        return f"prop={self.prop}, mode={self.mode}, value={self.value}"

    def reset_parameters(self) -> None:
        pass

    def forward(
        self, in_: torch.Tensor, in_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return random_shift(
            in_, in_lens, self.prop, self.mode, self.value, self.training
        )
