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
    else:
        pop = (
            torch.FloatTensor(b).view(logits.shape[:-1])
            for b in product(
                range(logits.shape[-1]),
                repeat=logits.nelement() // logits.shape[-1])
        )
        d = torch.distributions.Categorical(logits=logits)
    return sum(
        f(b) * d.log_prob(b).sum().exp()
        for b in pop
    )


@pytest.mark.cpu
@pytest.mark.parametrize("dist", ["bern", "cat"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_z(dist, seed):
    torch.manual_seed(seed)
    logits = torch.randn(2, 2, 4)
    exp = _expectation(lambda b: b, logits, dist)
    logits = logits[None, ...].expand((1000000,) + logits.shape)
    b = estimators.to_b(estimators.to_z(logits, dist), dist)
    act = b.float().mean(0)
    assert exp.shape == act.shape
    assert torch.allclose(exp, act, atol=1e-1)


@pytest.mark.cpu
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("dist", ["bern", "cat"])
@pytest.mark.parametrize("est", ["reinforce"])
def test_bias(seed, dist, est):
    torch.manual_seed(seed)
    logits = torch.randn(1, 4)
    logits.requires_grad_(True)

    def f(b):
        return (b - 1) ** 2
    exp = _expectation(f, logits, dist)
    exp, = torch.autograd.grad(
        [exp], [logits], grad_outputs=torch.ones_like(exp))
    logits = logits[None, ...].expand((100000,) + logits.shape)
    z = estimators.to_z(logits, dist)
    b = estimators.to_b(z, dist)
    if est == 'reinforce':
        fb = torch.stack([
            f(b_i) for b_i in b
        ], dim=0)
        g = estimators.reinforce(fb, b, logits)
    g = g.mean(0)
    assert exp.shape == g.shape
    assert torch.allclose(exp, g, atol=1e-1)
