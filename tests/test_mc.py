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

import time
import os

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
    q_probs = torch.rand_like(probs, requires_grad=True)
    proposal = torch.distributions.Bernoulli(probs=q_probs)
    estimator = estimators.ImportanceSamplingEstimator(
        proposal, func, density, self_normalize, is_log
    )
    v = estimator.estimate(N).sum()
    act_loss = (v - T / 2) ** 2
    assert torch.isclose(exp_loss, act_loss, atol=1)
    act_g_logits, act_g_theta, g_q_probs = torch.autograd.grad(
        act_loss, [logits, func.theta, q_probs]
    )
    assert torch.isclose(exp_g_theta, act_g_theta, atol=1)
    assert torch.allclose(exp_g_logits, act_g_logits, atol=1e-1)
    if not self_normalize:
        assert (g_q_probs == 0).all()


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
    cv = modules.LogisticBernoulliRebarControlVariate(func).to(device)
    estimator = estimators.RelaxEstimator(
        dist, func, cv, [probs], [cv.log_temp, cv.eta], is_log=is_log
    )
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
    cv = modules.GumbelOneHotCategoricalRebarControlVariate(func).to(device)
    estimator = estimators.RelaxEstimator(
        dist, func, cv, [probs], [cv.log_temp, cv.eta]
    )
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


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("DO_MC_BENCHMARK", None) != "1",
    reason="MC benchmark disabled by default. Set environment variable "
    "DO_MC_BENCHMARK=1 to enable and use flag -s to see output",
)
@pytest.mark.parametrize("mc_samples_per_iter", [1, 10, 100, 1000])
@pytest.mark.parametrize(
    "estimator",
    ["REINFORCE", "IS", "IS-sn", "REBAR", "REBAR-varmin", "RELAX", "RELAX-varmin"],
)
@pytest.mark.parametrize("num_bernoullis", [10])
def test_benchmark(mc_samples_per_iter, estimator, num_bernoullis):
    # The benchmark is simple, but convoluted: optimize a batch of independent Bernoulli
    # params towards their true values by using noisy counts. The total time and number
    # of iterations are reported.
    #
    # This is a very basic objective which is unlikely to be representative of your use
    # case. It is largely intended as a sanity check and an illustrative example.
    #
    # - REINFORCE:     REINFORCE estimate, no control variate.
    # - IS:            Importance Sampling estimate with proposal = density. Note using
    #                  the actual distribution as the proposal is nearly optimal from a
    #                  variance-minimizing perspective, but leads to an estimate not
    #                  very different from REINFORCE.
    # - IS-sn:         Same as IS but self-normalizing.
    # - REBAR:         REBAR estimate. Control variate is optimized with the gradient of
    #                  the MC estimate.
    # - REBAR-varmin:  Same as REBAR but control variate is optimized by minimizing the
    #                  variance of the gradient of the logits.
    # - RELAX:         RELAX estimate. The control variate is a learnable linear
    #                  transformation of each Bernoulli's relaxation. It is optimized
    #                  with the gradient of the MC estimate.
    # - RELAX-varmin:  Same as RELAX but control variate is optimized by minimizing the
    #                  variance of the gradient of the logits.

    torch.manual_seed(1)  # ensure same initialization for all estimators
    max_iters, num_bernoullis = 10000, 10
    ref_logits = torch.randn(num_bernoullis, device="cuda")
    hyp_logits = torch.nn.Parameter(torch.full((num_bernoullis,), 0.0, device="cuda"))

    def func(b):
        return b + (2 * torch.rand_like(b) - 1)

    if estimator == "RELAX":

        class ControlVariate(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.alpha = torch.nn.Parameter(torch.randn(1))
                self.beta = torch.nn.Parameter(torch.randn(1))

            def forward(self, x):
                return (self.alpha * x + self.beta).sigmoid()

        cv = ControlVariate().cuda()
    else:
        cv = modules.LogisticBernoulliRebarControlVariate(func).cuda()
    optimizer = torch.optim.Adam([hyp_logits] + list(cv.parameters()))
    loss_fn = torch.nn.MSELoss()

    print(
        f"Beginning benchmark for {estimator} w/ "
        f"mc_samples_per_iter = {mc_samples_per_iter}, "
        f"num_bernoullis = {num_bernoullis}, "
        f"and max_iters = {max_iters}"
    )
    atol = 1.0
    atol_scale_factor = 0.1
    converge_points = []
    start = time.time()
    for iter in range(1, max_iters + 1):
        optimizer.zero_grad()

        # We always have to reinitialize the estimator in the inner loop because the
        # distribution may instantiate tensors whose backwards graphs aren't kept.
        # Fortunately, the cost of reinitialization should be negligible.
        if estimator == "REINFORCE":
            dist = torch.distributions.Bernoulli(logits=hyp_logits)
            estimator_ = estimators.ReinforceEstimator(dist, func)
        elif estimator.startswith("IS"):
            dist = torch.distributions.Bernoulli(logits=hyp_logits)
            estimator_ = estimators.ImportanceSamplingEstimator(
                dist, func, dist, self_normalize=estimator.endswith("-sn")
            )
        else:  # REBAR or RELAX
            args = tuple()
            if estimator.endswith("-varmin"):
                args = ([hyp_logits], cv.parameters())
            dist = distributions.LogisticBernoulli(logits=hyp_logits)
            estimator_ = estimators.RelaxEstimator(dist, func, cv, *args)
        z = estimator_.estimate(mc_samples_per_iter)
        loss = loss_fn(z, ref_logits.sigmoid())
        loss.backward()
        optimizer.step()
        if torch.allclose(ref_logits, hyp_logits, atol=atol, rtol=0):
            elapsed = int(time.time() - start)
            converge_points.append((atol, iter, elapsed))
            atol *= atol_scale_factor
        # if not (iter % 100):
        # print(loss.item(), cv.log_temp.item(), cv.eta.item(), z.sum().item())
    elapsed = int(time.time() - start)
    diff = (ref_logits - hyp_logits).abs().max().item()
    print(f"Finished {max_iters} in about {elapsed}s.")
    print(f"Converged to within {diff} of the target.")
    for atol, iter, elapsed in converge_points:
        print(
            f"Estimator converged to within {atol} of target in {iter} iterations "
            f"({elapsed:e}s)"
        )

