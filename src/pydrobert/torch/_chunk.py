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
def extract_chunk_slices(
    in_: torch.Tensor,
    in_lens: Optional[torch.Tensor] = None,
    policy: Literal["fixed", "ali", "ref"] = None,
    window_type: Literal["symmmetric", "causal", "future", "valid"] = "symmetric",
    lobe_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...


@script
@functional_wrapper("ExtractChunkSlices")
def extract_chunk_slices(
    in_: torch.Tensor,
    in_lens: Optional[torch.Tensor] = None,
    policy: str = "fixed",
    window_type: str = "symmetric",
    lobe_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N, T = in_.shape[:2]
    device = in_.device
    if lobe_size < 0:
        raise RuntimeError(f"Expected non-negative lobe_size, got {lobe_size}")
    if window_type not in ("symmetric", "causal", "future", "valid"):
        raise NotImplementedError
    if policy == "fixed":
        shift = lobe_size + 1
        if window_type == "symmetric":
            window_size = 2 * lobe_size + 1
            half_shift = shift // 2
            TT = (T + half_shift) // shift
            mids = torch.arange(TT, device=device) * shift + half_shift
            starts = mids - window_size // 2
            ends = starts + window_size
        elif window_type == "causal":
            starts = torch.arange(-lobe_size, T - lobe_size, shift, device=device)
            ends = starts + shift
            mids = ends - 1
        elif window_type == "future":
            starts = mids = torch.arange(0, T, shift, device=device)
            ends = starts + shift
        else:
            starts = torch.arange(0, T - shift, shift, device=device)
            ends = starts + shift
            mids = ends - 1
        starts = starts.clamp_min_(0).expand(N, -1)
        if in_lens is None:
            ends = ends.clamp_max_(T).expand(N, -1)
        else:
            ends = torch.min(ends.unsqueeze(0), in_lens.unsqueeze(1))
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
        starts = torch.nonzero(torch.cat([nonempty, mask], 1))
        mask = torch.cat([torch.zeros_like(nonempty), mask], 1)
        mask |= nonempty & (in_lens.view(N, 1) == arange)
        ends = torch.nonzero(mask)
        # assert (starts[:, 0] == ends[:, 0]).all()
        sources = starts[:, 0]
        starts, ends = starts[:, 1], ends[:, 1]
        if lobe_size and window_type != "valid":
            NN = starts.size(0)
            start_idx = torch.arange(NN, device=device)
            end_idx = start_idx.clone()
            do_left = window_type in ("symmetric", "causal")
            do_right = window_type in ("symmetric", "future")
            for n in range(1, lobe_size + 1):
                offs = (sources[n:] == sources[: NN - n]).long()
                if do_left:
                    start_idx[n:] -= offs
                if do_right:
                    end_idx[: NN - n] += offs
            starts = starts[start_idx]
            ends = ends[end_idx]
        elif lobe_size:
            NN = starts.size(0)
            is_same = sources[: NN - lobe_size] == sources[lobe_size:]
            starts = starts[: NN - lobe_size][is_same]
            ends = ends[lobe_size:][is_same]
            sources = sources[: NN - lobe_size][is_same]
        slices = torch.stack([starts, ends], 1)
    else:
        raise NotImplementedError
    return slices, sources


class ExtractChunkSlices(torch.nn.Module):
    """Determine slices of feature chunks according to a variety of policies"""

    __constants__ = ["policy", "window_type", "lobe_size"]

    policy: str
    window_type: str
    lobe_size: int

    def __init__(
        self,
        policy: Literal["fixed", "ali"] = "fixed",
        window_type: Literal["symmetric", "causal", "future", "valid"] = "symmetric",
        lobe_size: int = 0,
    ) -> None:
        super().__init__()
        if policy not in {"fixed", "ali"}:
            raise ValueError(
                f"policy should be one of 'fixed' or 'ali'. Got '{policy}'"
            )
        if window_type not in {"symmetric", "causal", "future", "valid"}:
            raise ValueError(
                "window_type should be one of 'symmetric', 'causal', 'future', or "
                f"'valid'. Got '{window_type}'"
            )
        if lobe_size < 0:
            raise ValueError(f"lobe_size should be non-negative, got {lobe_size}")
        self.policy, self.window_type, self.lobe_size = policy, window_type, lobe_size

    def extra_repr(self) -> str:
        return (
            f"policy={self.policy}, window_type={self.window_type}, "
            f"lobe_size={self.lobe_size}"
        )

    def forward(
        self, in_: torch.Tensor, in_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return extract_chunk_slices(
            in_, in_lens, self.policy, self.window_type, self.lobe_size
        )

    __call__ = proxy(forward)

