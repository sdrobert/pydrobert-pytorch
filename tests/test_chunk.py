# Copyright 2023 Sean Robertson
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

import torch
import pytest
from pydrobert.torch.modules import ExtractChunkSlices


@pytest.mark.parametrize("policy", ["fixed", "ali"])
@pytest.mark.parametrize("window_type", ["symmetric", "causal", "future", "valid"])
@pytest.mark.parametrize("lobe_size", [0, 2])
def test_extract_chunk_sizes(device, policy, window_type, jit_type, lobe_size):
    if policy == "fixed":
        in_lens = torch.tensor([0, 8, 5], device=device)
        in_ = torch.empty((3, 11), device=device)
        if lobe_size == 0:
            # fmt: off
            slices_exp = torch.tensor([
                [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],  # n=1
                [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],  # n=2
            ], device=device)
            # fmt: on
            sources_exp = torch.tensor(
                [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], device=device
            )
        else:
            assert lobe_size == 2
            if window_type == "symmetric":
                # fmt: off
                slices_exp = torch.tensor([
                    [0, 4], [2, 7], [5, 8],  # n=1
                    [0, 4], [2, 5],  # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor([1, 1, 1, 2, 2], device=device)
            elif window_type == "causal":
                # fmt: off
                slices_exp = torch.tensor([
                    [0, 1], [1, 4], [4, 7],  # n=1
                    [0, 1], [1, 4],  # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor([1, 1, 1, 2, 2], device=device)
            elif window_type == "future":
                # fmt: off
                slices_exp = torch.tensor([
                    [0, 3], [3, 6], [6, 8],  # n=1
                    [0, 3], [3, 5],  # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor([1, 1, 1, 2, 2], device=device)
            else:
                # fmt: off
                slices_exp = torch.tensor([
                    [0, 3], [3, 6],  # n=1
                    [0, 3],  # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor([1, 1, 2], device=device)
    elif policy == "ali":
        in_lens = torch.tensor([7, 5, 9, 0], device=device)
        # fmt: off
        in_ = torch.tensor([
            [0, 0, 0, 1, 1, 0, 0, 5, 5, 5],  # n=0 t=7
            [1, 2, 2, 2, 2, 6, 6, 6, 6, 6],  # n=1 t=5
            [3, 3, 3, 3, 1, 2, 3, 4, 4, 4],  # n=2 t=9
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 1],  # n=3 t=0
        ], device=device)
        # fmt: on
        if lobe_size == 0:
            # fmt: off
            slices_exp = torch.tensor([
                [0, 3], [3, 5], [5, 7],  # n=0
                [0, 1], [1, 5],  # n=1
                [0, 4], [4, 5], [5, 6], [6, 7], [7, 9],  # n=2
            ], device=device)
            # fmt: on
            sources_exp = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2], device=device)
        else:
            assert lobe_size == 2
            if window_type == "symmetric":
                # fmt: off
                slices_exp = torch.tensor([
                    [0, 7], [0, 7], [0, 7],  # n=0
                    [0, 5], [0, 5],  # n=1
                    [0, 6], [0, 7], [0, 9], [4, 9], [5, 9],  # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor(
                    [0, 0, 0, 1, 1, 2, 2, 2, 2, 2], device=device
                )
            elif window_type == "causal":
                # fmt: off
                slices_exp = torch.tensor([
                    [0, 3], [0, 5], [0, 7],  # n=0
                    [0, 1], [0, 5],  # n=1
                    [0, 4], [0, 5], [0, 6], [4, 7], [5, 9],  # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor(
                    [0, 0, 0, 1, 1, 2, 2, 2, 2, 2], device=device
                )
            elif window_type == "future":
                # fmt: off
                slices_exp = torch.tensor([
                    [0, 7], [3, 7], [5, 7],  # n=0
                    [0, 5], [1, 5],  # n=1
                    [0, 6], [4, 7], [5, 9], [6, 9], [7, 9],  # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor(
                    [0, 0, 0, 1, 1, 2, 2, 2, 2, 2], device=device
                )
            else:
                # fmt: off
                slices_exp = torch.tensor([
                    [0, 7],  # n=0
                    [0, 6], [4, 7], [5, 9],  # n=2
                ])
                # fmt: on
                sources_exp = torch.tensor([0, 2, 2, 2], device=device)
    else:
        assert False
    extract_chunk_slices = ExtractChunkSlices(policy, window_type, lobe_size)
    if jit_type == "script":
        extract_chunk_slices = torch.jit.script(extract_chunk_slices)
    elif jit_type == "trace":
        extract_chunk_slices = torch.jit.trace(
            extract_chunk_slices, (torch.zeros_like(in_), torch.zeros_like(in_lens))
        )
    slices_act, sources_act = extract_chunk_slices(in_, in_lens)
    assert slices_exp.shape == slices_act.shape
    assert sources_exp.shape == sources_act.shape
    assert (slices_exp == slices_act).all()
    assert (sources_exp == sources_act).all()
