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

import pydrobert.torch.distributions as distributions
import pydrobert.torch.functional as functional


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
    assert ((b == 0.0) | (b == 1.0)).all()
    assert (b.sum(-1) == lmax).all()
    tmax_mask = tmax.unsqueeze(1) > torch.arange(tmax_max, device=device)
    b = b * tmax_mask
    assert (b.sum(-1) == lmax).all()
    assert torch.allclose(b.mean(0), srswor.mean, atol=1e-2)

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

