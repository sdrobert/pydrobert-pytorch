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

from pydrobert.torch.modules import MeanVarianceNormalization, FeatureDeltas


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