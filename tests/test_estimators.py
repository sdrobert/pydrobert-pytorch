# Copyright 2021 Sean Robertson
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

from itertools import product

import torch
import pytest
import pydrobert.torch.distributions as distributions
import pydrobert.torch.estimators as estimators
import pydrobert.torch.modules as modules


@pytest.fixture(params=["log", "exp"])
def is_log(request):
    return request.param == "log"


class Func(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.rand(1))

    def __call__(self, b):
        return b * self.theta


class LogFunc(Func):
    def __call__(self, b):
        return (b * self.theta).masked_fill(b == 0, 0.0).log()


def test_reinforce_estimator(device, is_log):
    N, T = int(1e5), 30
    logits = torch.randn(T, device=device, requires_grad=True)
    mask = torch.randint(2, (T,), device=device) == 1
    probs = logits.sigmoid().masked_fill(mask, 0)
    func = (LogFunc if is_log else Func)().to(device)
    v = (func.theta * probs).sum()
    exp_loss = (v - T / 2) ** 2
    exp_g_logits, exp_g_theta = torch.autograd.grad(exp_loss, [logits, func.theta])
    assert (exp_g_logits.masked_select(mask) == 0).all()

    probs = logits.sigmoid().masked_fill(mask, 0)

    if is_log:

        c = logits.detach()

        def cv(b):
            return logits.detach().expand_as(b)

    else:

        c = probs.detach()

        def cv(b):
            return probs.detach().expand_as(b)

    dist = torch.distributions.Bernoulli(probs=probs)
    estimator = estimators.ReinforceEstimator(dist, func, cv, c, is_log)
    v = estimator.estimate(N).sum()
    act_loss = (v - T / 2) ** 2
    assert torch.isclose(exp_loss, act_loss, atol=1)
    act_g_logits, act_g_theta = torch.autograd.grad(act_loss, [logits, func.theta])
    assert torch.isclose(exp_g_theta, act_g_theta, atol=1)
    assert torch.allclose(exp_g_logits, act_g_logits, atol=1e-1)


@pytest.mark.parametrize("self_normalize", [True, False], ids=["norm", "nonorm"])
def test_importance_sampling_estimator(device, self_normalize, is_log):
    N, T = int(1e6), 30
    logits = torch.randn(T, device=device, requires_grad=True)
    mask = torch.randint(2, (T,), device=device) == 1
    probs = logits.sigmoid().masked_fill(mask, 0)
    func = (LogFunc if is_log else Func)().to(device)
    v = (func.theta * probs).sum()
    exp_loss = (v - T / 2) ** 2
    exp_g_logits, exp_g_theta = torch.autograd.grad(exp_loss, [logits, func.theta])
    assert (exp_g_logits.masked_select(mask) == 0).all()

    probs = logits.sigmoid().masked_fill(mask, 0)

    if self_normalize:

        class MyDensity(torch.distributions.Bernoulli):
            def log_prob(self, value):
                return super().log_prob(value) - 1

        density = MyDensity(probs=probs)
    else:
        density = torch.distributions.Bernoulli(probs=probs)
    proposal = torch.distributions.Bernoulli(probs=torch.rand_like(probs))
    estimator = estimators.ImportanceSamplingEstimator(
        proposal, func, density, self_normalize, is_log
    )
    v = estimator.estimate(N).sum()
    act_loss = (v - T / 2) ** 2
    assert torch.isclose(exp_loss, act_loss, atol=1)
    act_g_logits, act_g_theta = torch.autograd.grad(act_loss, [logits, func.theta])
    assert torch.isclose(exp_g_theta, act_g_theta, atol=1)
    assert torch.allclose(exp_g_logits, act_g_logits, atol=1e-1)


def test_rebar_estimator_bernoulli(device, is_log):
    N, T = int(1e5), 30
    logits = torch.randn(T, device=device, requires_grad=True)
    mask = torch.randint(2, (T,), device=device) == 1
    probs = logits.sigmoid().masked_fill(mask, 0)
    func = (LogFunc if is_log else Func)().to(device)
    v = (func.theta * probs).sum()
    exp_loss = (v - T / 2) ** 2
    exp_g_logits, exp_g_theta = torch.autograd.grad(exp_loss, [logits, func.theta])
    assert (exp_g_logits.masked_select(mask) == 0).all()

    probs = logits.sigmoid().masked_fill(mask, 0)
    dist = distributions.LogisticBernoulli(probs=probs)
    cv = modules.LogisticBernoulliREBARControlVariate(func).to(device)
    estimator = estimators.RELAXEstimator(dist, func, cv, is_log)
    v = estimator.estimate(N).sum()
    act_loss = (v - T / 2) ** 2
    assert torch.isclose(exp_loss, act_loss, atol=1)
    act_g_logits, act_g_theta, g_log_temp, g_eta = torch.autograd.grad(
        act_loss, [logits, func.theta, cv.log_temp, cv.eta]
    )
    assert torch.isclose(exp_g_theta, act_g_theta, atol=1)
    assert torch.allclose(exp_g_logits, act_g_logits, atol=1e-1)
    zero_ = torch.tensor(0.0, device=device)
    assert not torch.isclose(g_log_temp, zero_)
    assert not torch.isclose(g_eta, zero_)


def test_rebar_estimator_categorical(device):
    N, T, V = int(1e5), 12, 5
    logits = torch.randn((T, V), device=device, requires_grad=True)
    mask = torch.randint_like(logits, 2) == 1
    mask[..., 0] = False
    logits_ = logits.masked_fill(mask, -float("inf"))
    probs = logits_.softmax(-1)
    theta = torch.arange(V, device=device, dtype=logits.dtype, requires_grad=True)
    v = (probs * theta / V).sum()
    exp_loss = (v - T / 2) ** 2
    exp_g_logits, exp_g_theta = torch.autograd.grad(exp_loss, [logits, theta])
    assert (exp_g_logits.masked_select(mask) == 0).all()

    def func(b):
        return (b * theta / V).sum(-1)

    logits_ = logits.masked_fill(mask, -float("inf"))
    probs = logits_.softmax(-1)
    dist = distributions.GumbelOneHotCategorical(probs=probs)
    cv = modules.GumbelOneHotCategoricalREBARControlVariate(func).to(device)
    estimator = estimators.RELAXEstimator(dist, func, cv)
    v = estimator.estimate(N).sum()
    act_loss = (v - T / 2) ** 2
    assert torch.isclose(exp_loss, act_loss, atol=1)
    act_g_logits, act_g_theta, g_log_temp, g_eta = torch.autograd.grad(
        act_loss, [logits, theta, cv.log_temp, cv.eta]
    )
    assert torch.allclose(exp_g_theta, act_g_theta, atol=1)
    assert torch.allclose(exp_g_logits, act_g_logits, atol=1e-1)
    zero_ = torch.tensor(0.0, device=device)
    assert not torch.isclose(g_log_temp, zero_)
    assert not torch.isclose(g_eta, zero_)


def _expectation(f, logits, dist):
    if dist == "bern":
        pop = (
            torch.FloatTensor(b).to(logits.device).view_as(logits)
            for b in product(range(2), repeat=logits.nelement())
        )
        d = torch.distributions.Bernoulli(logits=logits)
    elif dist == "cat":
        pop = (
            torch.FloatTensor(b).to(logits.device).view(logits.shape[:-1])
            for b in product(
                range(logits.shape[-1]), repeat=logits.nelement() // logits.shape[-1]
            )
        )
        d = torch.distributions.Categorical(logits=logits)
    else:
        pop = (
            torch.zeros_like(logits).scatter_(
                -1,
                torch.LongTensor(b).to(logits.device).view(logits.shape[:-1] + (1,)),
                1.0,
            )
            for b in product(
                range(logits.shape[-1]), repeat=logits.nelement() // logits.shape[-1]
            )
        )
        d = torch.distributions.OneHotCategorical(logits=logits)
    return sum(f(b) * d.log_prob(b).sum().exp() for b in pop)


@pytest.mark.parametrize("dist", ["bern", "cat", "onehot"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_z(dist, seed, device):
    torch.manual_seed(seed)
    logits = torch.randn(2, 2, 2).to(device)
    exp = _expectation(lambda b: b, logits, dist)
    logits = logits[None, ...].expand((10000,) + logits.shape)
    b = estimators.to_b(estimators.to_z(logits, dist, warn=False), dist)
    act = b.float().mean(0)
    assert act.device == exp.device
    assert exp.shape == act.shape
    assert torch.allclose(exp, act, atol=1e-1)


class ControlVariate(torch.nn.Module):
    def __init__(self, dist):
        super(ControlVariate, self).__init__()
        self.dist = dist
        self.weight = torch.nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-1.0, 1.0)

    def forward(self, inp, y=None):
        outp = inp * self.weight
        if self.dist != "bern":
            outp = outp.sum(-1)
        if y:
            return outp + y
        else:
            return outp


@pytest.mark.parametrize("seed", [4, 5, 6])
@pytest.mark.parametrize("dist", ["bern", "cat", "onehot"])
@pytest.mark.parametrize("est", ["reinforce", "relax"])
@pytest.mark.parametrize(
    "objective",
    [lambda b: b ** 2, lambda b: torch.exp(b)],
    ids=["squared error", "exponent"],
)
def test_bias(seed, dist, est, objective, device):
    torch.manual_seed(seed)
    logits = torch.randn(2, 4, requires_grad=True).to(device)

    def objective2(b):
        if dist == "onehot":
            return objective(b)[..., 0]
        else:
            return objective(b)

    exp = _expectation(objective2, logits, dist)
    (exp,) = torch.autograd.grad([exp], [logits], grad_outputs=torch.ones_like(exp))
    # if these tests fail, the number of markov samples might be too low. If
    # you keep raising this but it appears unable to meet the tolerance,
    # it's probably bias
    logits = logits[None, ...].expand((30000,) + logits.shape)
    z = estimators.to_z(logits, dist)
    b = estimators.to_b(z, dist)
    fb = estimators.to_fb(objective2, b)
    if est == "reinforce":
        g = estimators.reinforce(fb, b, logits, dist)
    elif est == "relax":
        g = estimators.relax(fb, b, logits, z, ControlVariate(dist).to(logits), dist)
    g = g.mean(0)
    assert exp.shape == g.shape
    assert torch.allclose(exp, g, atol=1e-1)


@pytest.mark.parametrize("dist", ["bern", "onehot"])
@pytest.mark.parametrize("est", ["rebar", "relax"])
def test_model_backprop(dist, device, est):
    torch.manual_seed(1)
    dim1, dim2, dim3 = 10, 5, 4
    model = torch.nn.Linear(dim3, dim3).to(device)
    inp = torch.randn(dim1, dim2, dim3, device=device)

    def f(x, y=None):
        if dist == "onehot":
            return x[..., -1] + y
        else:
            return x + y

    if est == "rebar":
        c = estimators.REBARControlVariate(f, dist).to(device)
    else:
        c = ControlVariate(dist).to(device)
    optim = torch.optim.Adam(list(model.parameters()) + list(c.parameters()))
    optim.zero_grad()
    logits = model(inp)
    z = estimators.to_z(logits, dist)
    b = estimators.to_b(z, dist)
    fb = f(b, y=1)
    diff, dlog_pb, dc_z, dc_z_tilde = estimators.relax(
        fb, b, logits, z, c, dist, components=True, y=1
    )
    g = diff * dlog_pb + dc_z - dc_z_tilde
    (g ** 2).sum().backward()
    if est == "rebar":
        assert c.log_temp.grad
    else:
        assert c.weight.grad
    assert model.weight.grad is None
    logits.backward(g)
    assert model.weight.grad.ne(0.0).any()


@pytest.mark.parametrize("markov", [10, 1000])
@pytest.mark.parametrize("num_latents", [2])
@pytest.mark.parametrize("dist,num_cat", [("bern", 2), ("onehot", 3)])
@pytest.mark.parametrize("est", ["reinforce", "rebar", "relax"])
def test_convergence(markov, num_latents, device, dist, num_cat, est):
    torch.manual_seed(7)
    # the objective is to minimize the expectation of the mean-squared error of
    # samples with the latent distribution parametrization. This will push
    # logits to maximize the chance of sampling the most likely value, not to
    # match the latents themselves
    max_iters = 20000
    if dist == "bern":
        latents = torch.rand(num_latents).to(device)
        mult_mask = torch.where(
            latents.gt(0.5), torch.ones(1).to(device), -torch.ones(1).to(device)
        )
        logits = torch.randn(num_latents, requires_grad=True, device=device)

        def f(b):
            return (b - latents) ** 2

        def convergence():
            return torch.all((mult_mask * logits.detach()).gt(1.0))

    else:
        latents = torch.rand(num_latents, num_cat).to(device)
        latents /= latents.sum(-1, keepdim=True)
        mask = (
            torch.zeros_like(latents)
            .scatter_(-1, latents.argmax(-1, keepdim=True), 1.0)
            .eq(1)
        )
        logits = torch.randn(num_latents, num_cat, requires_grad=True, device=device)

        def f(b):
            return ((b - latents) ** 2).sum(-1)

        def convergence():
            return torch.all(
                torch.log_softmax(logits.detach(), -1).masked_select(mask).gt(-0.05)
            )

    logit_optimizer = torch.optim.Adam([logits])
    if est == "rebar":
        c = estimators.REBARControlVariate(f, dist).to(device)
        tune_optimizer = torch.optim.Adam(c.parameters())
    elif est == "relax":
        c = torch.nn.Sequential(
            torch.nn.Linear(num_cat, 1), torch.nn.ReLU(), ControlVariate(dist)
        ).to(device)
        tune_optimizer = torch.optim.Adam(c.parameters())
    else:
        tune_optimizer = None
    for iter in range(1, max_iters + 1):
        logit_optimizer.zero_grad()
        if tune_optimizer:
            tune_optimizer.zero_grad()
        markov_logits = logits[None, ...].expand((markov,) + logits.shape)
        z = estimators.to_z(markov_logits, dist)
        b = estimators.to_b(z, dist)
        fb = f(b)
        if est == "reinforce":
            g = estimators.reinforce(fb, b, markov_logits, dist)
        else:
            g = estimators.relax(fb, b, markov_logits, z, c, dist)
        g = g.mean(0)
        logits.backward(g)
        if tune_optimizer:
            (g ** 2).sum().backward()
            tune_optimizer.step()
        logit_optimizer.step()
        del z, b, fb, markov_logits, g
        if convergence():
            print("Converged in {} iterations".format(iter))
            break
    print(latents, logits)
    if est == "rebar":
        print(c.log_temp, c.eta)
    assert convergence()
