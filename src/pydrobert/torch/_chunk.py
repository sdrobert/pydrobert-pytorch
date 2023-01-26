# Copyright 2023 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, overload
from typing_extensions import Literal

import torch

from ._compat import script
from ._wrappers import functional_wrapper, proxy


@overload
def slice_spect_data(
    in_: torch.Tensor,
    in_lens: Optional[torch.Tensor] = None,
    other_lens: Optional[torch.Tensor] = None,
    policy: Literal["fixed", "ali", "ref"] = None,
    window_type: Literal["symmmetric", "causal", "future"] = "symmetric",
    valid_only: bool = True,
    lobe_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...


@script
@torch.no_grad()
@functional_wrapper("SliceSpectData")
def slice_spect_data(
    in_: torch.Tensor,
    in_lens: Optional[torch.Tensor] = None,
    other_lens: Optional[torch.Tensor] = None,
    policy: str = "fixed",
    window_type: str = "symmetric",
    valid_only: bool = True,
    lobe_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N, T = in_.shape[:2]
    device = in_.device
    if not T:
        return (
            torch.empty(0, 2, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    if lobe_size < 0:
        raise RuntimeError(f"Expected non-negative lobe_size, got {lobe_size}")
    if window_type not in ("symmetric", "causal", "future"):
        raise RuntimeError(
            "expected window_type to be one of 'symmetric', 'casual', or 'future'"
            f"got '{window_type}'"
        )
    if policy == "fixed":
        shift = lobe_size + 1
        if valid_only and window_type == "symmetric":
            window_size = 2 * lobe_size + 1
            starts = torch.arange(0, T - window_size, shift, device=device)
            ends = starts + window_size
            mids = ends - 1
        elif window_type == "symmetric":
            window_size = 2 * lobe_size + 1
            half_shift = shift // 2
            TT = (T + half_shift) // shift
            mids = torch.arange(TT, device=device) * shift + half_shift
            starts = mids - window_size // 2
            ends = starts + window_size
        elif valid_only:
            # the behaviour doesn't change with "causal" or "future" when valid_only
            starts = torch.arange(0, T - shift, shift, device=device)
            ends = starts + shift
            mids = ends - 1
        elif window_type == "causal":
            starts = torch.arange(-lobe_size, T - lobe_size, shift, device=device)
            ends = starts + shift
            mids = ends - 1
        else:  # future
            starts = mids = torch.arange(0, T, shift, device=device)
            ends = starts + shift
        starts, ends = starts.expand(N, -1), ends.expand(N, -1)
        # starts = starts.clamp_min_(0).expand(N, -1)
        # if in_lens is None:
        #     ends = ends.clamp_max_(T).expand(N, -1)
        # else:
        #     ends = torch.min(ends.unsqueeze(0), in_lens.unsqueeze(1))
        TT = starts.size(1)
        slices = torch.stack([starts, ends], 2).flatten(end_dim=1)
        sources = torch.arange(N, device=device).view(N, 1).expand(N, TT).flatten()
        if in_lens is not None:
            mask = (in_lens.unsqueeze(1) > mids).flatten()
            slices = slices[mask]
            sources = sources[mask]
    elif policy == "ali":
        if in_.ndim != 2:
            raise RuntimeError(f"expected tensor of dimension 2 with policy 'ali'")
        mask = in_[:, :-1] != in_[:, 1:]
        arange = torch.arange(T, device=device)
        if in_lens is not None:
            mask = mask & (in_lens.view(N, 1) > arange[1:])
        else:
            in_lens = torch.full((N,), T, device=device)
        nonempty = (in_lens > 0).view(N, 1)
        starts = torch.cat([nonempty, mask], 1).nonzero()
        mask = torch.cat([torch.zeros_like(nonempty), mask], 1)
        mask |= nonempty & (in_lens.view(N, 1) == arange)
        ends = mask.nonzero()
        sources = starts[:, 0]
        starts, ends = starts[:, 1], ends[:, 1]
        if lobe_size:
            NN = starts.size(0)
            do_left = window_type in ("symmetric", "causal")
            do_right = window_type in ("symmetric", "future")
            if valid_only:
                offs = (int(do_left) + int(do_right)) * lobe_size
                is_same = sources[: NN - offs] == sources[offs:]
                starts = starts[: NN - offs][is_same]
                ends = ends[offs:][is_same]
                sources = sources[: NN - offs][is_same]
            else:
                start_idx = torch.arange(NN, device=device)
                end_idx = start_idx.clone()
                for n in range(1, lobe_size + 1):
                    offs = (sources[n:] == sources[: NN - n]).long()
                    if do_left:
                        start_idx[n:] -= offs
                    if do_right:
                        end_idx[: NN - n] += offs
                starts = starts[start_idx]
                ends = ends[end_idx]
        slices = torch.stack([starts, ends], 1)
    elif policy == "ref":
        if in_.ndim != 3:
            raise RuntimeError(f"Expected in_ to be 3-dimensional, got {in_.ndim}")
        if in_.size(2) != 3:
            raise RuntimeError(
                f"Expected 3rd dimension of in_ to be of size 3, got {in_.size(2)}"
            )
        starts = in_[..., 1]
        ends = in_[..., 2]
        if in_lens is None:
            in_lens = torch.full((N,), T, device=device)
        if other_lens is None:
            # the final segment's end time
            other_lens = (
                ends[..., 1]
                .gather(1, (in_lens - 1).clamp_min_(0).view(N, 1))
                .squeeze(1)
                .masked_fill_(in_lens == 0, 0)
            )
        mask = in_lens.view(N, 1) > torch.arange(T, device=device)
        mask &= (in_[..., 1:] >= 0).all(2)
        if window_type in ("symmetric", "causal"):
            starts = starts - lobe_size
        if window_type in ("symmetric", "future"):
            ends = ends + lobe_size
        if valid_only:
            mask &= (starts >= 0) & (ends <= other_lens.view(N, 1))
        mask &= starts < ends
        starts, ends, mask = starts.flatten(), ends.flatten(), mask.flatten()
        sources = torch.arange(N, device=device).view(N, 1).expand(N, T).flatten()
        starts = starts[mask]
        ends = ends[mask]
        sources = sources[mask]
        slices = torch.stack([starts, ends], 1)
    else:
        raise RuntimeError(
            f"Expected policy to be one of 'fixed', 'ali', or 'ref'; got {policy}"
        )
    return slices, sources


class SliceSpectData(torch.nn.Module):
    """Determine slices of feature chunks according to a variety of policies
    
    This module helps to chunk :class:`pydrobert.data.SpectDataLoader` data (or other
    similarly-structured tensors) into smaller units by returning slices of that data.
    The input to this module and the means of determining those slices varies according
    to the `policy` specified (see below). The return values can then be used
    to slice the data. For example, feature chunks can be stored in a list as
    follows


    If `policy` is :obj:`'fixed'`, slices are extracted at fixed intervals (``lobe_size
    + 1``) along the length of the data. `in_` is assumed to be the data in question,
    e.g. the `feats` tensor in a :class:`pydrobert.data.SpectDataLoader`, in batch-first
    order (although any tensor which matches its first two dimensions will do).
    `in_lens` may be used to specify the actual lengths of the input sequences if they
    were padded to fit in the same batch element. If `window_type` is
    :obj:`'symmetric'`, windows are of size ``1 + 2 * lobe_size``; otherwise, windows
    are of size ``1 + lobe_size``. When `valid_only` is :obj:`True`, slices start at
    index :obj:`0` and as many slices as can be fit fully within the sequences are
    returned. When `valid_only` is :obj:`False` slices are kept if their "middle" index
    lies before the end of the sequence with lobes clamped within the sequence. The
    "middle" index for the symmetric window is at ``slice[0] + window_size // 2``; for
    the causal window it's the last index of the window, ``slice[1] - 1``; for the
    future window it's the first, ``slice[0]``. When `valid_only` is :obj:`False`, the
    initial slice's offsets differ as well: for the symmetric case, it's ``(lobe_size +
    1) // 2 - window_size // 2``; for the causal case, it's :obj:`-lobe_size`; and the
    future case it's still :obj:`0`. As an example, given a sequence of length :obj:`8`,
    the following are slices under different configurations of the :obj:`'fixed'`
    policy with a `lobe_size` of :obj:`2`::

        [[0, 5], [3, 8]]          # symmetric, valid_only
        [[0, 3], [3, 6]]          # not symmetric, valid_only
        [[0, 4], [2, 7], [5, 9]]  # symmetric, not valid_only
        [[0, 1], [1, 4]]

    Parameters
    ----------
    policy
        Specifies how to slice the data. If :obj:`'fixed'`, extract windows of fixed
        length at fixed intervals. If :obj:`'ali'`, use changes in frame-level
        alignments to determine segment boundaries and slice along those. If
        :obj:`'ref'`, use token segmentations as slices. See above for more info.
    window_type
        Specifies how the window will be constructed around the "middle unit" in the
        policy. In general :obj:`'symmetric'` adds lobes to either side of the middle
        unit, :obj:`'causal'` to the left (towards :obj:`0`), :obj:`'future'` to the
        right. See above for differences by policy.
    valid_only
        Dictates what to do when a would-be slice passes over the boundaries of a
        segment. If :obj:`True`, any such slices are thrown out. If :obj:`False`, slice
        boundaries are limited to the boundaries of the sequence.
    lobe_size
        Specifies the number of units to each side of the middle unit will be
        added to the chunk. Non-negative.

    Call Parameters
    ---------------
    in_ : torch.Tensor
        A tensor of shape ``(N, T, *)``, where ``N`` is the batch dimension and ``T`` is
        the (maximum) sequence dimension. What `in_` contains depends on `policy`. See
        above.
    in_lens : torch.Tensor, optional
        A long tensor of shape ``(N,)`` specifying the lengths of sequences in `in_`.
        For the ``n``-th batch element, only the elements ``in_[n, :in_lens[n]]`` are
        considered. If unspecified, all sequences are assumed to be of length ``T``.
    other_lens : torch.Tensor, optional
        An additional long tensor of shape ``(N,)`` specifying some other lengths,
        depending on the policy. It is currently only used in the :obj:`'ref'` policy
        (see above).

    Returns
    -------
    slices : torch.Tensor
        A long tensor of shape ``(M, 2)`` storing the slices of all batch elements.
        ``M`` is the total number of slices. ``slices[m, 0]`` is the ``m``-th slice's
        start index (inclusive), while ``slices[m, 1]`` is the ``m``-th slice's end
        index (exclusive).
    sources : torch.Tensor
        A long tensor of shape ``(M,)`` where ``sources[m]`` is the batch index of the
        ``m``-th slice.
    """

    __constants__ = ["policy", "window_type", "valid_only", "lobe_size"]

    policy: str
    window_type: str
    valid_only: bool
    lobe_size: int

    def __init__(
        self,
        policy: Literal["fixed", "ali", "ref"] = "fixed",
        window_type: Literal["symmetric", "causal", "future"] = "symmetric",
        valid_only: bool = True,
        lobe_size: int = 0,
    ) -> None:
        super().__init__()
        if policy not in {"fixed", "ali", "ref"}:
            raise ValueError(
                f"policy should be one of 'fixed', 'ali', or 'ref'. Got '{policy}'"
            )
        if window_type not in {"symmetric", "causal", "future"}:
            raise ValueError(
                "window_type should be one of 'symmetric', 'causal', or 'future'. "
                f"Got '{window_type}'"
            )
        if lobe_size < 0:
            raise ValueError(f"lobe_size should be non-negative, got {lobe_size}")
        self.policy, self.window_type, self.lobe_size = policy, window_type, lobe_size
        self.valid_only = valid_only

    def extra_repr(self) -> str:
        return (
            f"policy={self.policy}, window_type={self.window_type}, "
            f"lobe_size={self.lobe_size}, valid_only={self.valid_only}"
        )

    def forward(
        self,
        in_: torch.Tensor,
        in_lens: Optional[torch.Tensor] = None,
        other_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return slice_spect_data(
            in_,
            in_lens,
            other_lens,
            self.policy,
            self.window_type,
            self.valid_only,
            self.lobe_size,
        )

    __call__ = proxy(forward)

