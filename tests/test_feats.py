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

from pydrobert.torch.modules import ChunkTokenSequencesBySlices, MeanVarianceNormalization, FeatureDeltas, SliceSpectData


@pytest.mark.parametrize("style", ["given", "sample", "accum"])
def test_mean_var_norm(device, jit_type, style):
    N1, N2, N3, N4, eps = 100, 200, 5, 50, 1e-5
    mean = torch.randn(N3, device=device)
    std = torch.rand(N3, device=device).clamp_min_(eps)
    y_exp = torch.randn(N1, N2, N3, N4, device=device)
    x = y_exp * std.unsqueeze(1) + mean.unsqueeze(1)
    mvn = MeanVarianceNormalization(
        -2, mean if style == "given" else None, std if style == "given" else None, eps
    )
    if jit_type == "script":
        mvn = torch.jit.script(mvn)
    if style == "accum":
        for x_n in x:
            mvn.accumulate(x_n)
        mvn.store()
        assert torch.allclose(mean, mvn.mean.float(), atol=1e-2)
        assert torch.allclose(std, mvn.std.float(), atol=1e-2)
    if jit_type == "trace":
        mvn = torch.jit.trace(mvn, (torch.empty(1, 1, N3, 1, device=device),))
    y_act = mvn(x)
    assert torch.allclose(y_exp, y_act, atol=1e-2)


@pytest.mark.parametrize("order, width", [(0, 10), (1, 3), (2, 2)])
@pytest.mark.parametrize("dim", [-3, 0, 3])
def test_feat_deltas(device, jit_type, order, width, dim):
    N1, N2, N3, N4 = 10, 5, 4, 2
    post = pytest.importorskip("pydrobert.speech.post")
    x = torch.randn(N1, N2, N3, N4, device=device)
    op = post.Deltas(order, target_axis=dim, context_window=width)
    exp = torch.tensor(op.apply(x.numpy(), axis=-2, in_place=True)).to(device)
    exp_shape = [N1, N2, N3, N4]
    exp_shape[dim] *= order + 1
    assert exp.shape == tuple(exp_shape)
    feat_deltas = FeatureDeltas(dim, -2, True, order, width)
    if jit_type == "script":
        feat_deltas = torch.jit.script(feat_deltas)
    elif jit_type == "trace":
        feat_deltas = torch.jit.trace(feat_deltas, (torch.empty(1, 1, 1, 1),))
    act = feat_deltas(x)
    assert exp.shape == act.shape
    assert torch.allclose(exp, act, atol=1e-5)



import torch
import pytest
from pydrobert.torch.modules import SliceSpectData


@pytest.mark.parametrize("policy", ["fixed", "ali", "ref"])
@pytest.mark.parametrize("window_type", ["symmetric", "causal", "future"])
@pytest.mark.parametrize("valid_only", [True, False], ids=["valid", "invalid"])
@pytest.mark.parametrize("lobe_size", [0, 2])
def test_slice_spect_data(
    device, policy, window_type, valid_only, jit_type, lobe_size
):
    if policy == "fixed":
        in_lens = other_lens = torch.tensor([0, 8, 5], device=device)
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
            if valid_only and window_type == "symmetric":
                # fmt: on
                slices_exp = torch.tensor([
                    [0, 5], [3, 8],  # n=1
                    [0, 5],  # n=2
                ], device=device)
                # fmt: off
                sources_exp = torch.tensor([1, 1, 2], device=device)
            elif window_type == "symmetric":
                # fmt: off
                slices_exp = torch.tensor([
                    [-1, 4], [2, 7], [5, 10],  # n=1
                    [-1, 4], [2, 7],  # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor([1, 1, 1, 2, 2], device=device)
            elif valid_only:
                # fmt: off
                slices_exp = torch.tensor([
                    [0, 3], [3, 6],  # n=1
                    [0, 3],  # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor([1, 1, 2], device=device)
            elif window_type == "causal":
                # fmt: off
                slices_exp = torch.tensor([
                    [-2, 1], [1, 4], [4, 7],  # n=1
                    [-2, 1], [1, 4],  # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor([1, 1, 1, 2, 2], device=device)
            else:  # future
                # fmt: off
                slices_exp = torch.tensor([
                    [0, 3], [3, 6], [6, 9],  # n=1
                    [0, 3], [3, 6],  # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor([1, 1, 1, 2, 2], device=device)
    elif policy == "ali":
        in_lens = other_lens = torch.tensor([7, 5, 9, 0], device=device)
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
            if valid_only and window_type == "symmetric":
                slices_exp = torch.tensor([[0, 9]], device=device)
                sources_exp = torch.tensor([2], device=device)
            elif window_type == "symmetric":
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
            elif valid_only:
                # fmt: off
                slices_exp = torch.tensor([
                    [0, 7],  # n=0
                    [0, 6], [4, 7], [5, 9],  # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor([0, 2, 2, 2], device=device)
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
            else:
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
        assert policy == "ref"
        in_lens = torch.tensor([3, 0, 3], device=device)
        other_lens = torch.tensor([3, 10, 4], device=device)
        # fmt: off
        in_ = torch.tensor([
            [[0, 0, 1], [0, 0, 2], [1, 1, 3]],  # n=0 r=3 t=3
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # n=1 r=0 t=10
            [[1, 2, 2], [1, 2, 5], [1, 2, -1]],  # n=2 r=3 t=4
        ], device=device)
        # fmt: on
        if lobe_size == 0 and valid_only:
            # fmt: off
            slices_exp = torch.tensor([
                [0, 1], [0, 2], [1, 3],  # n=0
            ], device=device)
            # fmt: on
            sources_exp = torch.tensor([0, 0, 0], device=device)
        elif lobe_size == 0:
            # fmt: off
            slices_exp = torch.tensor([
                [0, 1], [0, 2], [1, 3],  # n=0
                [2, 5],  # n=2
            ], device=device)
            # fmt: on
            sources_exp = torch.tensor([0, 0, 0, 2], device=device)
        else:
            assert lobe_size == 2
            if valid_only and window_type == "symmetric":
                slices_exp = torch.tensor([[0, 4]], device=device)
                sources_exp = torch.tensor([2], device=device)
            elif window_type == "symmetric":
                # fmt: off
                slices_exp = torch.tensor([
                    [-2, 3], [-2, 4], [-1, 5],  # n=0
                    [0, 4], [0, 7],  # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor([0, 0, 0, 2, 2], device=device)
            elif valid_only and window_type == "causal":
                slices_exp = torch.tensor([[0, 2]], device=device)
                sources_exp = torch.tensor([2], device=device)
            elif window_type == "causal":
                # fmt: off
                slices_exp = torch.tensor([
                    [-2, 1], [-2, 2], [-1, 3],  # n=0
                    [0, 2], [0, 5],  # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor([0, 0, 0, 2, 2], device=device)
            elif valid_only:
                # fmt: off
                slices_exp = torch.tensor([
                    [0, 3],  # n=0
                    [2, 4],  # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor([0, 2], device=device)
            else:  # future
                # fmt: off
                slices_exp = torch.tensor([
                    [0, 3], [0, 4], [1, 5],  # n=2
                    [2, 4], [2, 7], # n=2
                ], device=device)
                # fmt: on
                sources_exp = torch.tensor([0, 0, 0, 2, 2], device=device)
    extract_chunk_slices = SliceSpectData(
        policy, window_type, valid_only, lobe_size
    )
    if jit_type == "script":
        extract_chunk_slices = torch.jit.script(extract_chunk_slices)
    elif jit_type == "trace":
        extract_chunk_slices = torch.jit.trace(
            extract_chunk_slices,
            (
                torch.zeros_like(in_),
                torch.zeros_like(in_lens),
                torch.zeros_like(other_lens),
            ),
        )
    slices_act, sources_act = extract_chunk_slices(in_, in_lens, other_lens)
    assert slices_exp.shape == slices_act.shape
    assert sources_exp.shape == sources_act.shape
    assert (slices_exp == slices_act).all()
    assert (sources_exp == sources_act).all()


@pytest.mark.parametrize("partial", [True, False], ids=['partial', 'full'])
@pytest.mark.parametrize("retain", [True, False], ids=["absolute", "relative"])
def test_chunk_token_sequences_by_slices(device, partial, jit_type, retain):
    ref_lens = torch.tensor([0, 5, 2], device=device)
    # fmt: off
    refs = torch.tensor([
        [[0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1], [4, 0, 1]],  # n=0
        [[0, 0, 2], [-1, 2, 4], [1, 4, 6], [2, -1, 7], [3, 5, 8]],  # n=1
        [[0, 5, 4], [0, 2, 2], [0, 2, 2], [1, 2, 2], [2, 2, 2]],  # n=2
    ], device=device)
    # fmt: on
    slices = torch.tensor([[0, 1], [3, 7], [-1, 3]], device=device)
    if partial:
        exp_chunks = [
            torch.empty((0, 3), device=device, dtype=torch.long),
            torch.tensor([[-1, 2, 4], [1, 4, 6], [3, 5, 8]], device=device),
            torch.tensor([[0, 2, 2]], device=device),
        ]
    else:
        exp_chunks = [
            torch.empty((0, 3), device=device, dtype=torch.long),
            torch.tensor([[1, 4, 6]], device=device),
            torch.tensor([[0, 2, 2]], device=device),
        ]
    if not retain:
        exp_chunks[1][:, 1:] += slices[1, 0]
        exp_chunks[2][:, 1:] += slices[2, 0]
    chunk_token_sequences_by_slices = ChunkTokenSequencesBySlices(partial, retain)
    if jit_type == "script":
        chunk_token_sequences_by_slices = torch.jit.script(chunk_token_sequences_by_slices)
    elif jit_type == "trace":
        chunk_token_sequences_by_slices = torch.jit.trace(
            chunk_token_sequences_by_slices,
            (
                torch.empty((1, 0, 3), dtype=torch.long),
                torch.zeros((1, 2), dtype=torch.long),
                torch.zeros((1,), dtype=torch.long),
            ),
        )
    act_chunks, act_lens = chunk_token_sequences_by_slices(refs, slices, ref_lens)
    assert len(act_lens) == len(exp_chunks)
    for act_chunk_n, act_lens_n, exp_chunk_n in zip(act_chunks, act_lens, exp_chunks):
        assert act_lens_n == exp_chunk_n.size(0)
        act_chunk_n = act_chunk_n[:act_lens_n]
        assert (act_chunk_n == exp_chunk_n).all()
