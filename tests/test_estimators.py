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
        theta = torch.sigmoid(logits)
        pop = (
            torch.LongTensor(b).view_as(logits)
            for b in product(range(2), repeat=logits.nelement())
        )
        return sum(
            f(b).to(theta) *
            (b.to(theta) * theta + (1 - b.to(theta)) * (1 - theta)).prod()
            for b in pop
        )
    else:
        pop = (
            torch.LongTensor(b).view(logits.shape[:-1])
            for b in product(
                range(logits.shape[-1]),
                repeat=logits.nelement() // logits.shape[-1])
        )
        return sum(
            f(b).to(logits) * (-torch.nn.functional.nll_loss(
                logits.log_softmax(-1).view(-1, logits.shape[-1]),
                b.flatten(), reduction='sum')).exp()
            for b in pop
        )


@pytest.mark.cpu
@pytest.mark.parametrize("dist", ["bern", "cat"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_z(dist, seed):
    torch.manual_seed(seed)
    theta = torch.distributions.utils.clamp_probs(torch.rand(2, 2, 4))
    if dist == "bern":
        logits = torch.log(theta) - torch.log1p(-theta)
    else:
        logits = torch.log(theta)
    exp = _expectation(lambda b: b, logits, dist)
    logits = logits[None, ...].expand((1000000, ) + logits.shape)
    b = estimators.to_b(estimators.to_z(logits, dist), dist)
    act = b.to(theta).mean(0)
    assert exp.shape == act.shape
    assert torch.allclose(exp, act, atol=1e-1)
