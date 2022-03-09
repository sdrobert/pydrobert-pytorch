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

import torch
import pytest

import pydrobert.torch.estimators as estimators


@pytest.fixture(params=["log", "exp"])
def is_log(request):
    return request.param == "log"


def test_enumerate_estimator(device, is_log):
    T, V = 10, 6
    logits = torch.randn((T, V), device=device, requires_grad=True)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[..., 0] = True
    target = torch.randint(1, V, (T,), device=device)
    logits_ = logits.masked_fill(mask, -float("inf"))
    probs = logits_.softmax(-1)

    def func(b: torch.Tensor) -> torch.Tensor:
        target_ = target.expand(b.shape[:-1]).unsqueeze(-1)
        probs_ = b.gather(-1, target_).squeeze(-1)
        if is_log:
            return probs_.log()
        else:
            return probs_

    exp_loss = func(probs).mean()
    assert exp_loss != 0
    (exp_g,) = torch.autograd.grad(exp_loss, [logits])

    logits_ = logits.masked_fill(mask, -float("inf"))
    probs = logits_.softmax(-1)
    dist = torch.distributions.OneHotCategorical(probs=probs)
    estimator = estimators.EnumerateEstimator(dist, func, is_log)
    act_loss = estimator.estimate().mean()
    assert torch.allclose(exp_loss, act_loss)
    (act_g,) = torch.autograd.grad(act_loss, [logits])
    assert torch.allclose(exp_g, act_g)
