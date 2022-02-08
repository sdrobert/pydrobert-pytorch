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

from pydrobert.torch.modules import TimeDistributedReturn


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("gamma", [0.0, 0.95])
def test_time_distributed_return(device, batch_first, gamma):
    steps, batch_size = 1000, 30
    r = torch.randn(steps, batch_size, device=device)
    exp = torch.empty_like(r)
    exp[-1] = r[-1]
    for step in range(steps - 2, -1, -1):
        exp[step] = r[step] + gamma * exp[step + 1]
    if batch_first:
        r = r.t().contiguous()
        exp = exp.t().contiguous()
    time_distributed_return = TimeDistributedReturn(gamma, batch_first)
    act = time_distributed_return(r)
    assert torch.allclose(exp, act, atol=1e-5)
