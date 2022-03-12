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

from pydrobert.torch.distributions import GumbelOneHotCategorical, LogisticBernoulli


def test_logistic_bernoulli(device):
    N, T = int(1e6), 10
    probs = torch.rand(T, device=device)
    probs[0] = 0.0  # make sure it doesn't NaN
    dist = LogisticBernoulli(probs=probs)
    z = dist.rsample([N])
    assert torch.allclose(z.mean(0), dist.mean, atol=1)
    assert torch.allclose(z.std(0), dist.stddev, atol=1e-2)
    b = dist.threshold(z)
    assert torch.allclose(b.mean(0), probs, atol=1e-3)
    zz = dist.csample(b)
    assert torch.allclose(dist.threshold(zz), b)
    # E_b[E_{z|b}[z]] = E_z[z]
    assert torch.allclose(zz.mean(0), dist.mean, atol=1)
    assert torch.allclose(zz.std(0), dist.stddev, atol=1e-2)
    exp_log_prob = dist.log_prob(zz)
    act_log_prob = dist.tlog_prob(b)
    assert exp_log_prob.shape == act_log_prob.shape
    act_log_prob += dist.clog_prob(zz, b)
    assert exp_log_prob.shape == act_log_prob.shape
    assert torch.allclose(exp_log_prob, act_log_prob), (
        (exp_log_prob - act_log_prob).abs().max()
    )


def test_gumbel_one_hot_categorical(device):
    N, T, V = int(1e6), 4, 3
    probs = torch.rand(T, V, device=device)
    probs[0, 0] = 0.0  # make sure it doesn't NaN
    probs /= probs.sum(-1, keepdim=True)
    dist = GumbelOneHotCategorical(probs=probs)
    z = dist.rsample([N])
    assert torch.allclose(z.mean(0), dist.mean, atol=1)
    assert torch.allclose(z.std(0), dist.stddev, atol=1e-2)
    b = dist.threshold(z)
    assert torch.allclose(b.mean(0), probs, atol=1e-3)
    zz = dist.csample(b)
    assert torch.allclose(dist.threshold(zz), b)
    assert torch.allclose(zz.mean(0), dist.mean, atol=1)
    assert torch.allclose(zz.std(0), dist.stddev, atol=1e-2)
    exp_log_prob = dist.log_prob(zz)
    act_log_prob = dist.tlog_prob(b)
    assert exp_log_prob.shape == act_log_prob.shape
    act_log_prob += dist.clog_prob(zz, b)
    assert exp_log_prob.shape == act_log_prob.shape
    assert torch.allclose(exp_log_prob, act_log_prob), (
        (exp_log_prob - act_log_prob).abs().max()
    )
