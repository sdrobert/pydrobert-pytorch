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

import math

import torch
import pytest

import pydrobert.torch.distributions as distributions
import pydrobert.torch.functional as functional


@pytest.mark.parametrize("tmax", [20, 66])
def test_binomial_coefficient(device, jit_type, tmax):
    T = torch.arange(tmax, device=device)
    binomial_coefficient = functional.binomial_coefficient
    if jit_type == "script":
        binomial_coefficient = torch.jit.script(binomial_coefficient)
    elif jit_type == "trace":
        binomial_coefficient = torch.jit.trace(
            binomial_coefficient, (torch.tensor(0), torch.tensor(0)),
        )
    binom = binomial_coefficient(T.unsqueeze(1), T)
    for length in range(tmax):
        for count in range(tmax):
            if count > length:
                N_exp = 0
            else:
                N_exp = math.factorial(length) // (
                    math.factorial(count) * math.factorial(length - count)
                )
            assert binom[length, count] == N_exp, (length, count)


def test_enumerate_binary_sequences(device, jit_type):
    tmax = 10
    enumerate_binary_sequences = functional.enumerate_binary_sequences
    if jit_type == "script":
        enumerate_binary_sequences = torch.jit.script(enumerate_binary_sequences)
    elif jit_type == "trace":
        pytest.xfail("trace unsupported for enumerate_binary_sequences")
    support = enumerate_binary_sequences(tmax, device)
    assert support.shape == (2 ** tmax, tmax)
    assert (support.sum(0) == 2 ** (tmax - 1)).all()
    half = tmax // 2
    assert (support[: 2 ** half, half:] == 0).all()
    assert (support[: 2 ** half, :half].sum(0) == 2 ** (half - 1)).all()


def test_enumerate_vocab_sequences(device, jit_type):
    tmax, vmax = 5, 4
    enumerate_vocab_sequences = functional.enumerate_vocab_sequences
    if jit_type == "script":
        enumerate_vocab_sequences = torch.jit.script(enumerate_vocab_sequences)
    elif jit_type == "trace":
        pytest.xfail("trace unsupported for enumerate_vocab_sequences")
    support = enumerate_vocab_sequences(tmax, vmax, device=device)
    assert support.shape == (vmax ** tmax, tmax)
    support_ = torch.unique(support, sorted=True, dim=0)
    assert support.shape == support_.shape
    nrange_exp = torch.arange(vmax, device=device)
    nrange_act, counts = support.flatten().unique(sorted=True, return_counts=True)
    assert counts.sum() == support.numel()
    assert (nrange_exp == nrange_act).all()
    assert (counts == support.numel() // vmax).all()
    for t in range(tmax):
        assert (support[: vmax ** t, t:] == 0).all()


def test_enumerate_binary_sequences_with_cardinality(device, jit_type):
    tmax = 10
    T = torch.arange(tmax - 1, -1, -1, device=device)
    eb = eb_ = functional.enumerate_binary_sequences_with_cardinality
    if jit_type == "script":
        eb = eb_ = torch.jit.script(eb)
    elif jit_type == "trace":
        eb = torch.jit.trace(eb, (torch.tensor(1), torch.tensor(1)))
    batched, binom = eb(T.unsqueeze(-1), T)
    for length in range(tmax):
        for count in range(tmax):
            nonbatched = eb_(length, count).to(device)
            if count > length:
                N_exp = M_exp = 0
            else:
                if count == 0:
                    M_exp, N_exp = 0, 1
                else:
                    M_exp = math.factorial(length - 1) // (
                        math.factorial(count - 1) * math.factorial(length - count)
                    )
                    N_exp = M_exp * length // count
            assert nonbatched.shape == (N_exp, length)
            assert (nonbatched.sum(1) == count).all()
            assert (nonbatched.sum(0) == M_exp).all()
            assert binom[tmax - length - 1, tmax - count - 1] == N_exp
            batched_elem = batched[tmax - length - 1, tmax - count - 1, :N_exp, :length]
            assert batched_elem.shape == nonbatched.shape
            assert (batched_elem == nonbatched).all()


def test_simple_random_sampling_without_replacement(device, jit_type):
    tmax_max, nmax, mmax = 16, 8, 2 ** 15
    tmax = torch.randint(tmax_max + 1, size=(nmax,), dtype=torch.float, device=device)
    lmax = (torch.rand(nmax, device=device) * (tmax + 1)).floor_()

    srswor = distributions.SimpleRandomSamplingWithoutReplacement(
        lmax, tmax, tmax_max, True
    )
    if jit_type == "script":
        srswor_ = torch.jit.script(
            functional.simple_random_sampling_without_replacement
        )
        b = srswor_(tmax.expand(mmax, nmax), lmax.expand(mmax, nmax), tmax_max)
    elif jit_type == "trace":
        # trace doesn't support integer parameters, so we'll redefine tmax_max to the
        # computed default
        tmax_max = int(tmax.max().item())
        srswor = distributions.SimpleRandomSamplingWithoutReplacement(
            lmax, tmax, tmax_max, True
        )
        srswor_ = torch.jit.trace(
            functional.simple_random_sampling_without_replacement,
            [torch.ones(1), torch.zeros(1)],
        )
        b = srswor_(tmax.expand(mmax, nmax), lmax.expand(mmax, nmax))
    else:
        b = srswor.sample([mmax])
    assert ((b == 0) | (b == 1)).all()
    assert (b.sum(-1) == lmax).all()
    tmax_mask = tmax.unsqueeze(1) > torch.arange(tmax_max, device=device)
    b = b * tmax_mask
    assert (b.sum(-1) == lmax).all()
    assert torch.allclose(b.float().mean(0), srswor.mean, atol=1e-2)

    lp_exp = []
    for n in range(nmax):
        tmax_n, lmax_n = int(tmax[n].item()), int(lmax[n].item())
        lp_exp.append(
            math.log(
                (math.factorial(tmax_n - lmax_n) * math.factorial(lmax_n))
                / math.factorial(tmax_n)
            )
        )
    lp_exp = torch.tensor(lp_exp, device=device).expand(mmax, nmax)
    lp_act = srswor.log_prob(b)
    assert torch.allclose(lp_exp, lp_act)


def test_simple_random_sampling_without_replacement_enumerate_support(device):
    tmax = 5
    given_count = 2
    total_count = torch.arange(1, tmax + 1, device=device).clamp_min_(given_count)
    dist = distributions.SimpleRandomSamplingWithoutReplacement(
        given_count, total_count
    )
    assert not dist.has_enumerate_support
    total_count.fill_(tmax)
    dist = distributions.SimpleRandomSamplingWithoutReplacement(
        given_count, total_count, tmax + 1
    )
    assert dist.has_enumerate_support
    support = dist.enumerate_support(True)
    M_exp = math.factorial(tmax - 1) // (
        math.factorial(given_count - 1) * math.factorial(tmax - given_count)
    )
    N_exp = M_exp * tmax // given_count
    assert support.shape == (N_exp, tmax, tmax + 1)
    assert (support[..., -1] == 0).all()
    support = support[..., :-1]
    assert (support.sum(-1) == given_count).all()
    assert (support.sum(0) == M_exp).all()
