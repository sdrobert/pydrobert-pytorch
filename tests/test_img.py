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

import os

import torch
import pytest
import numpy as np

from pydrobert.torch.modules import (
    DenseImageWarp,
    PolyharmonicSpline,
    RandomShift,
    SparseImageWarp,
    SpecAugment,
    Warp1DGrid,
)
from pydrobert.torch._compat import meshgrid


def test_warp_1d_grid(device, jit_type):
    N, W = 5, 7
    src = torch.arange(N, device=device)
    lengths = src + W - N + 1
    flow = torch.ones(N, device=device)
    warp_1d_grid = Warp1DGrid()
    if jit_type == "script":
        warp_1d_grid = torch.jit.script(warp_1d_grid)
    else:
        warp_1d_grid = torch.jit.trace(
            warp_1d_grid, (torch.zeros(1), torch.zeros(1), torch.ones(1))
        )
    grid_W = warp_1d_grid(src, flow, lengths).view(N, 1, -1)
    grid_H = torch.zeros_like(grid_W) - 1
    grid = torch.stack([grid_W, grid_H], -1)
    feats = torch.eye(N, W, device=device)
    new_feats = torch.nn.functional.grid_sample(
        feats.view(N, 1, 1, W), grid, align_corners=False
    ).view(N, W)
    assert (new_feats.argmax(1) == (src + flow).long()).all()


def test_warp_1d_grid_batch(device):
    N, T = 12, 7
    src = torch.ones(N, device=device)
    lengths = torch.randint(2, T + 1, (N,), device=device)
    flow = torch.rand(N, device=device) - 0.5
    exp_grid = Warp1DGrid(max_length=T)(src, flow, lengths)
    exp_grid = ((exp_grid + 1) * T - 1) / 2
    for n, exp_grid_n in enumerate(exp_grid):
        src_n = src[n : n + 1]
        flow_n = flow[n : n + 1]
        lengths_n = lengths[n : n + 1]
        length_n = int(lengths_n.item())
        act_grid_n = Warp1DGrid()(src_n, flow_n, lengths_n)[0]
        # see note in Warp1DGrid for why we modify act_grid_n
        act_grid_n = ((act_grid_n + 1) * length_n - 1) / 2
        assert torch.allclose(exp_grid_n[: lengths_n[0]], act_grid_n, atol=1e-5)


@pytest.mark.parametrize("indexing", ["hw", "wh"])
def test_dense_image_warp_matches_tensorflow(device, indexing, jit_type):
    dir_ = os.path.join(os.path.dirname(__file__), "dense_image_warp")
    img = torch.tensor(np.load(os.path.join(dir_, "img.npy")), device=device)
    flow = torch.tensor(np.load(os.path.join(dir_, "flow.npy")), device=device)
    if indexing == "wh":
        flow = flow.flip(-1)
    dense_image_warp = DenseImageWarp(indexing=indexing)
    if jit_type == "script":
        dense_image_warp = torch.jit.script(dense_image_warp)
    elif jit_type == "trace":
        dense_image_warp = torch.jit.trace(
            dense_image_warp, (torch.empty(1, 1, 1, 1), torch.empty(1, 1, 1, 2))
        )
    exp = torch.tensor(np.load(os.path.join(dir_, "warped.npy")), device=device)
    act = dense_image_warp(img, flow)
    assert torch.allclose(exp, act), (exp - act).abs().max()


@pytest.mark.parametrize("order", [1, 2, 3])
def test_polyharmonic_interpolation_matches_tensorflow(order, device, jit_type):
    dir_ = os.path.join(os.path.dirname(__file__), "polyharmonic_spline")
    x = torch.tensor(np.load(os.path.join(dir_, "x.npy")), device=device)
    y = torch.tensor(np.load(os.path.join(dir_, "y.npy")), device=device)
    q = torch.tensor(np.load(os.path.join(dir_, "q.npy")), device=device)
    exp = torch.tensor(
        np.load(os.path.join(dir_, "o{}.npy".format(order))), device=device
    )
    polyharmonic_spline = PolyharmonicSpline(order, full_matrix=True)
    if jit_type == "script":
        polyharmonic_spline = torch.jit.script(polyharmonic_spline)
    elif jit_type == "trace":
        polyharmonic_spline = torch.jit.trace(
            polyharmonic_spline,
            (torch.rand(1, 2, 1), torch.rand(1, 2, 5), torch.rand(1, 1, 1)),
        )
    act = polyharmonic_spline(x, y, q)
    assert torch.allclose(exp, act, atol=1e-3), (exp - act).abs().max()


@pytest.mark.parametrize("pinned_boundary_points", [0, 1, 2])
def test_sparse_image_warp_identity(device, pinned_boundary_points, jit_type):
    N, C, H, W = 50, 12, 8, 3
    img = exp = torch.rand(N, C, H, W, device=device) * 255
    # we add 3 random control pointrs under the identity mapping to ensure a
    # non-degenerate interpolate
    src = dst = torch.rand(N, 3, 2, device=device) * min(H, W)
    sparse_image_warp = SparseImageWarp(
        dense_interpolation_mode="nearest",
        pinned_boundary_points=pinned_boundary_points,
    )
    if jit_type == "script":
        sparse_image_warp = torch.jit.script(sparse_image_warp)
    elif jit_type == "trace":
        sparse_image_warp = torch.jit.trace(
            sparse_image_warp,
            (torch.empty(1, 1, 2, 2), torch.empty(1, 0, 2), torch.empty(1, 0, 2)),
        )
    act, flow = sparse_image_warp(img, src, dst)
    assert torch.allclose(flow, torch.tensor(0.0, device=device))
    assert torch.allclose(exp, act), (exp - act).abs().max()


def test_spec_augment_compare_1d_warp_to_2d_warp(device):
    # it turns out that the 2D warp isn't fully agnostic to the control points in the
    # frequency dimension (as seen if you uncomment the final line in this test). It
    # appears that while the interpolated flow is zero along the frequency dimension,
    # there are some strange oscillations in the flow's time dimension in the 2D case.
    # I've verified with Daniel Park that this was not intended. In any event, it makes
    # this test flawed
    N, T, F, W = 12, 30, 20, 5
    feats = torch.rand(N, T, F, device=device)
    spec_augment = SpecAugment(
        max_time_warp=W, max_freq_warp=0, max_time_mask=0, max_freq_mask=0
    )  # no masking
    params = spec_augment.draw_parameters(feats)
    w_0, w = params[:2]

    # the 2D transform outlined in the paper features 6 zero-flow boundary points -
    # one in each corner and two at the mid-points of the left and right edges -
    # and a single point w_0 along the midpoint line being shifted to w_0 + w.
    # the corner boundary points are handled by pinned_boundary_points=1
    # note: for coordinates, it's (freq, time), not (time, freq), despite the order in
    # feats
    midF = F // 2
    zeros = torch.zeros_like(w_0)  # (N,)
    shift_src = torch.stack([zeros + midF, w_0], 1)  # (N, 2)
    shift_dst = torch.stack([zeros + midF, w_0 + w], 1)  # (N, 2)
    left_src = left_dst = torch.stack([zeros + midF, zeros], 1)  # (N, 2)
    right_src = right_dst = torch.stack([zeros + midF, zeros + T - 1], 1)  # (N, 2)
    src = torch.stack([left_src, shift_src, right_src], 1)  # (N, 3, 2)
    dst = torch.stack([left_dst, shift_dst, right_dst], 1)  # (N, 3, 2)
    exp, flow = SparseImageWarp("wh", pinned_boundary_points=1)(
        feats.unsqueeze(1), src, dst
    )
    exp = exp.squeeze(1)
    assert torch.allclose(flow[..., 0], torch.tensor(0.0, device=device))


def test_spec_augment_batch(device):
    N, T, F, W = 5, 40, 10, 5
    # we make feats move
    feats = torch.rand((N, T, F), device=device)
    lengths = torch.randint(1, T + 1, (N,), device=device)
    lengths[0] = T
    spec_augment = SpecAugment(
        max_time_warp=W, max_freq_warp=0, max_time_mask=0, max_freq_mask=0,
    )
    params = spec_augment.draw_parameters(feats, lengths)
    w_0, w = params[:2]
    feats.requires_grad = True
    act_feats = spec_augment.apply_parameters(feats, params, lengths)
    ones = (
        (lengths.unsqueeze(-1) > torch.arange(T, device=device))
        .float()
        .unsqueeze(-1)
        .expand(N, T, F)
    )
    (act_g,) = torch.autograd.grad([act_feats], [feats], ones)
    for n in range(N):
        feats_n = feats[n : n + 1, : lengths[n]]
        w_0_n, w_n = w_0[n : n + 1], w[n : n + 1]
        params_n = (w_0_n, w_n) + params[2:]
        exp_feats_n = spec_augment.apply_parameters(feats_n, params_n)
        act_feats_n = act_feats[n : n + 1, : lengths[n]]
        assert torch.allclose(exp_feats_n, act_feats_n, atol=1e-4), (
            (exp_feats_n - act_feats_n).abs().max()
        )
        (exp_g_n,) = torch.autograd.grad(
            [exp_feats_n], [feats_n], torch.ones_like(feats_n)
        )
        act_g_n = act_g[n : n + 1, : lengths[n]]
        assert torch.allclose(exp_g_n, act_g_n, atol=1e-4), (
            (exp_g_n - act_g_n).abs().max()
        )


def test_spec_augment_zero_params_is_identity(device):
    N, T, F = 50, 200, 80
    feats = exp = torch.rand(N, T, F, device=device)
    spec_augment = SpecAugment(
        max_time_warp=1000, max_freq_warp=1000, max_time_mask=1000, max_freq_mask=1000
    )
    params = spec_augment.draw_parameters(feats)
    # w_0 and v_0 must remain nonzero b/c otherwise interpolation would include the
    # border twice
    params[1].zero_()  # w
    params[3].zero_()  # v
    params[5].zero_()  # t
    params[7].zero_()  # f
    act = feats = spec_augment.apply_parameters(feats, params)
    assert torch.allclose(exp, act, atol=1e-4), (exp - act).abs().max()


def test_spec_augment_masking(device):
    N, T, F = 500, 200, 80
    max_time_mask = max_freq_mask = 20
    nT = nF = 2
    feats = torch.ones(N, T, F, device=device)
    spec_augment = SpecAugment(
        max_time_warp=0,
        max_freq_warp=0,
        max_time_mask=max_time_mask,
        max_freq_mask=max_freq_mask,
        max_time_mask_proportion=1.0,
        num_time_mask=nT,
        num_time_mask_proportion=1 / (100 * T),
        num_freq_mask=nF,
        interpolation_order=2,
    )

    assert nT == nF == 2  # logic below only works when 2

    # current setting shouldn't draw any time masks
    params = spec_augment.draw_parameters(feats)
    t = params[5]
    assert not (t > 0).any()

    spec_augment.num_time_mask_proportion = 1.0

    params = spec_augment.draw_parameters(feats)
    t_0, t, f_0, f = params[4:]
    assert (t > 0).any()  # some t could coincidentally land on zero
    t_1, f_1 = t_0 + t, f_0 + f  # (N, nT), (N, nF)

    max_t0s = torch.max(t_0.unsqueeze(1), t_0.unsqueeze(2))  # (N, nT, nT)
    min_t1s = torch.min(t_1.unsqueeze(1), t_1.unsqueeze(2))  # (N, nT, nT)
    diff_t = torch.clamp(min_t1s - max_t0s, min=0).tril()  # (N, nT, nT)
    diff_t = diff_t * torch.eye(nT, device=device, dtype=torch.long) - diff_t * (
        1 - torch.eye(nT, device=device, dtype=torch.long)
    )
    exp_masked_t = diff_t.sum(2).sum(1)  # (N,)
    assert torch.all(exp_masked_t <= t.sum(1))

    max_f0s = torch.max(f_0.unsqueeze(1), f_0.unsqueeze(2))  # (N, nF, nF)
    min_f1s = torch.min(f_1.unsqueeze(1), f_1.unsqueeze(2))  # (N, nF, nF)
    diff_f = torch.clamp(min_f1s - max_f0s, min=0).tril()  # (N, nF, nF)
    diff_f = diff_f * torch.eye(nF, device=device, dtype=torch.long) - diff_f * (
        1 - torch.eye(nF, device=device, dtype=torch.long)
    )
    exp_masked_f = diff_f.sum(2).sum(1)  # (N,)
    assert torch.all(exp_masked_f <= f.sum(1))

    act_feats = spec_augment.apply_parameters(feats, params)
    eq_0 = (act_feats == 0.0).long()
    act_masked_t = eq_0.prod(2).sum(1)
    act_masked_f = eq_0.prod(1).sum(1)

    assert torch.all(act_masked_t == exp_masked_t)
    assert torch.all(act_masked_f == exp_masked_f)


@pytest.mark.parametrize("use_lengths", [True, False], ids=["lengths", "nolengths"])
def test_spec_augment_call(device, use_lengths, jit_type):
    N, T, F = 30, 2048, 80
    max_time_warp, max_freq_warp = 15, 20
    max_time_mask, max_freq_mask = 30, 7
    num_time_mask, num_freq_mask = 2, 3
    max_time_mask_proportion = 0.2
    if use_lengths:
        lengths = torch.randint(1, T + 1, (N,), device=device)
    feats = torch.rand(N, T, F, device=device)
    spec_augment = SpecAugment(
        max_time_warp=max_time_warp,
        max_freq_warp=max_freq_warp,
        max_time_mask=max_time_mask if jit_type != "trace" else 0,
        max_freq_mask=max_freq_mask,
        max_time_mask_proportion=max_time_mask_proportion,
        num_time_mask=num_time_mask,
        num_freq_mask=num_freq_mask,
    ).to(device)
    if use_lengths:
        args = (feats, lengths)
        trace_args = (
            torch.empty(1, 1, 1, device=device),
            torch.ones(1, dtype=torch.long, device=device),
        )
    else:
        args = (feats,)
        trace_args = (torch.empty(1, 1, 1, device=device),)
    if jit_type == "trace":
        # spec_augment is nondeterministic, so we don't check repeat return values
        spec_augment = torch.jit.trace(spec_augment, trace_args, check_trace=False)
    elif jit_type == "script":
        spec_augment = torch.jit.script(spec_augment)
    spec_augment(*args)


@pytest.mark.parametrize("mode", ["reflect", "constant", "replicate"])
def test_random_shift_call(device, mode, jit_type):
    N, T, A, B = 50, 300, 13, 11
    in_ = torch.rand(N, T, A, B, device=device)
    in_lens = torch.randint(1, T + 1, (N,), device=device)
    rand_shift = RandomShift(1.0, mode).to(device)
    if jit_type == "trace":
        # random_shift is nondeterministic, so we don't check repeat return values
        rand_shift = torch.jit.trace(
            rand_shift,
            (
                torch.empty(1, 1, device=device),
                torch.ones(1, device=device, dtype=torch.long),
            ),
            check_trace=False,
        )
    elif jit_type == "script":
        rand_shift = torch.jit.script(rand_shift)
    out, out_lens = rand_shift(in_, in_lens)
    assert out.dim() == 4
    assert (out_lens >= in_lens).all()
    assert out.size(0) == N
    assert out.size(1) >= out_lens.max()
    assert out.size(2) == A
    assert out.size(3) == B


def test_polyharmonic_interpolation_linear(device):
    # when the order is 1, this should simply be linear interpolation
    x = torch.arange(3, device=device).unsqueeze(0).unsqueeze(-1).float()
    y = torch.tensor([[[0.0], [1.0], [0.0]]], device=device)
    y = torch.cat([y, 1.0 - y], 2)  # (1, 3, 2)
    q = torch.tensor([[[0.0], [0.5], [1.0], [1.6], [2.0]]], device=device)
    exp = torch.tensor(
        [[[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.4, 0.6], [0.0, 1.0]]], device=device
    )
    act = PolyharmonicSpline(1)(x, y, q)
    assert torch.allclose(exp, act)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_polyharmonic_interpolation_equal_on_knots(order, device):
    N, T, in_, out = 10, 11, 12, 13
    x = torch.rand(N, T, in_, device=device)
    y = torch.rand(N, T, out, device=device)
    act = PolyharmonicSpline(order)(x, y, x)
    # the high tolerance seems a numerical stability issue caused by polynomials in
    # the RBF
    assert torch.allclose(y, act, atol=1e-3), (y - act).abs().max()


@pytest.mark.parametrize("flip_h", [True, False])
@pytest.mark.parametrize("flip_w", [True, False])
def test_dense_image_warp_flow_flips(device, flip_h, flip_w):
    H, W = 30, 40
    img = torch.arange(H * W, dtype=torch.float32, device=device).view(1, 1, H, W)
    exp = img
    if flip_h:
        h = 2 * torch.arange(H, dtype=torch.float32, device=device) - H + 1
        exp = exp.flip(2)
    else:
        h = torch.zeros((H,), dtype=torch.float32, device=device)
    if flip_w:
        w = 2 * torch.arange(W, dtype=torch.float32, device=device) - W + 1
        exp = exp.flip(3)
    else:
        w = torch.zeros((W,), dtype=torch.float32, device=device)
    exp = exp.flatten()
    flow = torch.stack(meshgrid(h, w), 2)
    act = DenseImageWarp()(img, flow).flatten()
    assert torch.allclose(exp, act, atol=1e-4), (exp - act).abs().max()
    act = DenseImageWarp(mode="nearest")(img, flow).flatten()
    assert torch.allclose(exp, act), (exp - act).abs().max()


def test_dense_image_warp_shift_right(device):
    N, C, H, W = 11, 20, 50, 19
    img = torch.rand(N, C, H, W, device=device)
    flow = torch.ones(N, H, W, 2, device=device)
    exp = img[..., :-1, :-1]
    act = DenseImageWarp()(img, flow)[..., 1:, 1:]
    assert torch.allclose(exp, act, atol=1e-5), (exp - act).abs().max()
    act = DenseImageWarp(mode="nearest")(img, flow)[..., 1:, 1:]
    assert torch.allclose(exp, act), (exp - act).abs().max()


@pytest.mark.parametrize("include_flow", [True, False])
@pytest.mark.parametrize("pinned_boundary_points", [0, 2])
def test_sparse_image_warp_matches_tensorflow(
    device, include_flow, pinned_boundary_points
):
    dir_ = os.path.join(os.path.dirname(__file__), "sparse_image_warp")
    img = torch.tensor(np.load(os.path.join(dir_, "img.npy")), device=device)
    src = torch.tensor(np.load(os.path.join(dir_, "src.npy")), device=device)
    dst = torch.tensor(np.load(os.path.join(dir_, "dst.npy")), device=device)
    exp_warped = torch.tensor(
        np.load(os.path.join(dir_, "warped_{}.npy".format(pinned_boundary_points))),
        device=device,
    )
    sparse_image_warp = SparseImageWarp(
        pinned_boundary_points=pinned_boundary_points, include_flow=include_flow
    )
    if include_flow:
        exp_flow = torch.tensor(
            np.load(os.path.join(dir_, "flow_{}.npy".format(pinned_boundary_points))),
            device=device,
        )
        act_warped, act_flow = sparse_image_warp(img, src, dst)
        assert torch.allclose(exp_flow, act_flow, atol=1e-3), (
            (exp_flow - act_flow).abs().max()
        )
    else:
        act_warped = sparse_image_warp(img, src, dst)
    assert torch.allclose(exp_warped, act_warped, atol=1e-3), (
        (exp_warped - act_warped).abs().max()
    )
