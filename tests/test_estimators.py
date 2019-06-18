__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"

from itertools import product

import torch
import pytest
import pydrobert.torch.estimators as estimators


def _expectation(f, logits, dist):
    if dist == "bern":
        pop = (
            torch.FloatTensor(b).view_as(logits)
            for b in product(range(2), repeat=logits.nelement())
        )
        d = torch.distributions.Bernoulli(logits=logits)
    elif dist == "cat":
        pop = (
            torch.FloatTensor(b).view(logits.shape[:-1])
            for b in product(
                range(logits.shape[-1]),
                repeat=logits.nelement() // logits.shape[-1])
        )
        d = torch.distributions.Categorical(logits=logits)
    else:
        pop = (
            torch.zeros_like(logits).scatter_(
                -1, torch.LongTensor(b).view(logits.shape[:-1] + (1,)), 1.)
            for b in product(
                range(logits.shape[-1]),
                repeat=logits.nelement() // logits.shape[-1])
        )
        d = torch.distributions.OneHotCategorical(logits=logits)
    return sum(
        f(b) * d.log_prob(b).sum().exp()
        for b in pop
    )


@pytest.mark.cpu
@pytest.mark.parametrize("dist", ["bern", "cat", "onehot"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_z(dist, seed):
    torch.manual_seed(seed)
    logits = torch.randn(2, 2, 2)
    exp = _expectation(lambda b: b, logits, dist)
    logits = logits[None, ...].expand((10000,) + logits.shape)
    b = estimators.to_b(estimators.to_z(logits, dist, warn=False), dist)
    act = b.float().mean(0)
    assert exp.shape == act.shape
    assert torch.allclose(exp, act, atol=1e-1)


class ControlVariate(torch.nn.Module):

    def __init__(self, dist):
        super(ControlVariate, self).__init__()
        self.dist = dist
        self.weight = torch.nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-1., 1.)

    def forward(self, inp):
        outp = inp * self.weight
        if self.dist != "bern":
            outp = outp.sum(-1)
        return outp


@pytest.mark.cpu
@pytest.mark.parametrize("seed", [4, 5, 6])
@pytest.mark.parametrize("dist", ["bern", "cat", "onehot"])
@pytest.mark.parametrize("est", ["reinforce", "relax"])
@pytest.mark.parametrize("objective", [
    lambda b: b ** 2,
    lambda b: torch.exp(b),
], ids=[
    "squared error",
    "exponent"
])
def test_bias(seed, dist, est, objective):
    torch.manual_seed(seed)
    logits = torch.randn(2, 4, requires_grad=True)

    def objective2(b):
        if dist == "onehot":
            return objective(b)[..., 0]
        else:
            return objective(b)
    exp = _expectation(objective2, logits, dist)
    assert not torch.allclose(exp, torch.zeros(1))
    exp, = torch.autograd.grad(
        [exp], [logits], grad_outputs=torch.ones_like(exp))
    assert not torch.allclose(exp, torch.zeros(1))
    # if these tests fail, the number of markov samples might be too low. If
    # you keep raising this but it appears unable to meet the tolerance,
    # it's probably bias
    logits = logits[None, ...].expand((30000,) + logits.shape)
    z = estimators.to_z(logits, dist)
    b = estimators.to_b(z, dist)
    fb = estimators.to_fb(objective2, b)
    if est == 'reinforce':
        g = estimators.reinforce(fb, b, logits, dist)
    elif est == "relax":
        g = estimators.relax(fb, b, logits, z, ControlVariate(dist), dist)
    g = g.mean(0)
    assert exp.shape == g.shape
    assert torch.allclose(exp, g, atol=1e-1)


@pytest.mark.cpu
@pytest.mark.parametrize("dist", ["bern", "cat"])
def test_relax_c_backprop(dist):
    torch.manual_seed(1)
    logits = torch.randn(10, 5, 4, requires_grad=True)
    z = estimators.to_z(logits, dist)
    b = estimators.to_b(z, dist)
    fb = torch.rand_like(b)
    c = ControlVariate(dist)
    diff, dlog_pb, dc_z, dc_z_tilde = estimators.relax(
        fb, b, logits, z, c, dist, components=True)
    torch.autograd.grad(
        [diff], [logits], retain_graph=True,
        grad_outputs=torch.ones_like(diff))
    # for bernoulli, grad is
    torch.autograd.grad(
        [dc_z], [logits], retain_graph=True,
        grad_outputs=torch.ones_like(dc_z),
        allow_unused=True,
    )
    torch.autograd.grad(
        [dc_z_tilde], [logits], retain_graph=True,
        grad_outputs=torch.ones_like(dc_z_tilde))
    g = diff * dlog_pb + dc_z - dc_z_tilde
    (g ** 2).sum().backward()
    c.weight.grad
