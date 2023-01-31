# Copyright 2022 Sean Robertson
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
from pydrobert.torch.modules import ChunkBySlices, PadVariable, PadMaskedSequence


@pytest.mark.parametrize("mode", ["constant", "reflect", "replicate"])
@pytest.mark.parametrize("another_dim", [True, False])
def test_pad_variable(device, mode, another_dim, jit_type):
    N, Tmax, Tmin, F = 10, 50, 5, 30 if another_dim else 1
    x = torch.rand((N, Tmax, F), device=device)
    lens = torch.randint(Tmin, Tmax + 1, (N,), device=device)
    pad = torch.randint(Tmin - 1, size=(2, N), device=device)
    exp_padded = []
    for x_n, lens_n, pad_n in zip(x, lens, pad.t()):
        x_n = x_n[:lens_n]
        padded_n = torch.nn.functional.pad(
            x_n.unsqueeze(0).unsqueeze(0), [0, 0] + pad_n.tolist(), mode
        ).view(-1, F)
        exp_padded.append(padded_n)
    pad_variable = PadVariable(mode)
    if jit_type == "script":
        pad_variable = torch.jit.script(pad_variable)
    elif jit_type == "trace":
        pad_variable = torch.jit.trace(
            pad_variable,
            (
                torch.ones(1, 2),
                torch.full((1,), 2, dtype=torch.long),
                torch.ones(2, 1, dtype=torch.long),
            ),
        )
    act_padded = pad_variable(x, lens, pad)
    for exp_padded_n, act_padded_n in zip(exp_padded, act_padded):
        assert torch.allclose(exp_padded_n, act_padded_n[: len(exp_padded_n)])
    # quick double-check that other types work
    for type_ in (torch.long, torch.bool):
        assert pad_variable(x.to(type_), lens, pad).dtype == type_


@pytest.mark.parametrize("batch_first", [True, False])
def test_pad_masked_sequence(device, batch_first, jit_type):
    N1, N2, N3, N4, p = 15, 3, 11, 17, -1
    x = torch.rand((N1, N2, N3, N4), device=device)
    mask = torch.randint(2, (N1, N2), device=device, dtype=torch.bool)
    T, N = (N2, N1) if batch_first else (N1, N2)
    exp_lens = torch.empty(N, dtype=torch.long, device=device)
    exp_x = torch.full_like(x, p)
    for n in range(N):
        if batch_first:
            x_n, mask_n, ex_n = x[n], mask[n], exp_x[n]
        else:
            x_n, mask_n, ex_n = x[:, n], mask[:, n], exp_x[:, n]
        i = 0
        for j in range(T):
            if mask_n[j]:
                ex_n[i] = x_n[j]
                i += 1
        exp_lens[n] = i
    pad_masked_sequence = PadMaskedSequence(batch_first, float(p))
    if jit_type == "script":
        pad_masked_sequence = torch.jit.script(pad_masked_sequence)
    elif jit_type == "trace":
        pad_masked_sequence = torch.jit.trace(
            pad_masked_sequence, (torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool))
        )
    act_x, act_lens = pad_masked_sequence(x, mask)
    assert act_x.shape == exp_x.shape
    assert act_lens.shape == exp_lens.shape
    assert (act_lens == exp_lens).all()
    assert (act_x == exp_x).all()


@pytest.mark.parametrize("mode", ["constant", "reflect", "replicate"])
@pytest.mark.parametrize("another_dim", [True, False])
def test_chunk_by_slice(device, mode, another_dim, jit_type):
    N, Tmax, Tmin, F = 30, 20, 5, 7 if another_dim else 1
    lens = torch.randint(Tmin, Tmax + 1, (N,), device=device)
    starts = torch.randint(-Tmax + 1, Tmax, (N,))
    starts = torch.max(starts, -lens + 1)
    ends = starts + torch.randint(-1, Tmax - 1, (N,))  # -1 to allow empty slices
    ends = torch.min(ends, 2 * lens - 1)
    slices = torch.stack([starts, ends], 1)
    x = torch.arange(N * Tmax * F, device=device, dtype=torch.float).view(N, Tmax, F)
    exp_chunks = []
    exp_chunk_lens = []
    for x_n, lens_n, starts_n, ends_n in zip(x, lens, starts, ends):
        chunk_lens_n = (ends_n - starts_n).clamp_min_(0).view(1)
        exp_chunk_lens.append(chunk_lens_n)
        if chunk_lens_n == 0:
            exp_chunks.append(x_n[:0])
            continue
        pad_left = (-starts_n).clamp_min_(0).item()
        pad_right = (ends_n - lens_n).clamp_min_(0).item()
        x_n = x_n[:lens_n]
        x_n = torch.nn.functional.pad(
            x_n.unsqueeze(0).unsqueeze(0), [0, 0] + [pad_left, pad_right], mode
        ).view(-1, F)
        chunks_n = x_n[starts_n + pad_left : ends_n + pad_left]
        exp_chunks.append(chunks_n)
    exp_chunk_lens = torch.cat(exp_chunk_lens)
    chunk_by_slices = ChunkBySlices(mode)
    if jit_type == "script":
        chunk_by_slices = torch.jit.script(chunk_by_slices)
    elif jit_type == "trace":
        chunk_by_slices = torch.jit.trace(
            chunk_by_slices,
            (
                torch.ones(1, 1),
                torch.full((1,), 2, dtype=torch.long),
                torch.zeros(1, 2, dtype=torch.long),
            ),
        )
    act_chunks, act_chunk_lens = chunk_by_slices(x, lens, slices)
    assert (exp_chunk_lens == act_chunk_lens).all()
    for n, (exp_chunks_n, act_chunks_n, chunk_lens_n) in enumerate(
        zip(exp_chunks, act_chunks, exp_chunk_lens)
    ):
        act_chunks_n = act_chunks_n[:chunk_lens_n]
        assert exp_chunks_n.shape == act_chunks_n.shape
        exp_chunks_n, act_chunks_n = exp_chunks_n.squeeze(1), act_chunks_n.squeeze(1)
        assert torch.allclose(exp_chunks_n, act_chunks_n[:chunk_lens_n]), (
            n,
            x[n].squeeze(1),
            starts[n],
            ends[n],
            lens[n],
        )
