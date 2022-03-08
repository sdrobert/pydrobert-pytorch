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

from typing import Optional, Union

import torch

from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from ._compat import script


@script
@torch.no_grad()
def simple_random_sampling_without_replacement(
    total_count: torch.Tensor,
    given_count: torch.Tensor,
    total_count_max: Optional[int] = None,
) -> torch.Tensor:
    if total_count is None:
        total_count = int(total_count.max().item())
    total_count, given_count = torch.broadcast_tensors(total_count, given_count)
    if (given_count > total_count).any():
        raise RuntimeError("given_count cannot exceed total_count")
    b = torch.empty(
        torch.Size([total_count_max]) + total_count.shape, device=total_count.device
    )
    remainder_ell = given_count
    remainder_t = total_count
    for t in range(total_count_max):
        p = remainder_ell / remainder_t
        b_t = torch.bernoulli(p)
        b[t] = b_t
        remainder_ell = remainder_ell - b_t
        remainder_t = (remainder_t - 1).clamp_min_(1)
    return b.movedim(0, -1)


class CardinalityConstraint(constraints.Constraint):
    is_discrete = True
    event_dim = 1

    def __init__(
        self, total_count: torch.Tensor, given_count: torch.Tensor, tmax: int
    ) -> None:
        self.given_count = given_count
        self.total_count_mask = total_count.unsqueeze(-1) <= torch.arange(
            tmax, device=total_count.device
        )
        super().__init__()

    def check(self, value: torch.Tensor) -> torch.Tensor:
        is_bool = ((value == 0) | (value == 1)).all(-1)
        isnt_gte_tc = (self.total_count_mask.expand_as(value) * value).sum(-1) == 0
        value_sum = value.sum(-1)
        matches_count = value_sum == self.given_count.expand_as(value_sum)
        return is_bool & isnt_gte_tc & matches_count


class SimpleRandomSamplingWithoutReplacement(torch.distributions.ExponentialFamily):

    arg_constraints = {
        "total_count": constraints.nonnegative_integer,
        "given_count": constraints.nonnegative_integer,
    }
    _mean_carrier_measure = 0

    def __init__(
        self,
        given_count: Union[int, torch.Tensor],
        total_count: Union[int, torch.Tensor],
        out_size: Optional[int] = None,
        validate_args=None,
    ):
        given_count = torch.as_tensor(given_count)
        total_count = torch.as_tensor(total_count)
        if out_size is None:
            out_size = total_count.max()
        given_count, total_count = torch.broadcast_tensors(given_count, total_count)
        batch_shape = given_count.size()
        event_shape = torch.Size([out_size])
        self.total_count, self.given_count = total_count, given_count
        super().__init__(batch_shape, event_shape, validate_args)

    @constraints.dependent_property
    def support(self) -> torch.Tensor:
        return CardinalityConstraint(
            self.total_count, self.given_count, self.event_shape[0]
        )

    @lazy_property
    def log_partition(self) -> torch.Tensor:
        # log total_count choose given_count
        log_factorial = (
            torch.arange(
                1,
                self.event_shape[0] + 1,
                device=self.total_count.device,
                dtype=torch.float,
            )
            .log()
            .cumsum(0)
        )
        t_idx = (self.total_count.long() - 1).clamp_min(0)
        g_idx = (self.given_count.long() - 1).clamp_min(0)
        tmg_idx = (self.total_count.long() - self.given_count.long() - 1).clamp_min(0)
        return log_factorial[t_idx] - log_factorial[g_idx] - log_factorial[tmg_idx]

    @property
    def mean(self):
        len_mask = self.total_count.unsqueeze(-1) <= torch.arange(
            self.event_shape[0], device=self.total_count.device
        )
        return (
            (self.given_count / self.total_count.clamp_min(1.0))
            .unsqueeze(-1)
            .expand(self.batch_shape + self.event_shape)
        ).masked_fill(len_mask, 0.0)

    @property
    def variance(self):
        return self.mean * (1 - self.mean)

    @property
    def _natural_params(self):
        return self.total_count.new_zeros(self.batch_shape + self.event_shape)

    def _log_normalizer(self, logits):
        if (logits == 0).all():
            raise RuntimeError("Logits invalid")
        return self.log_partition

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(
            SimpleRandomSamplingWithoutReplacement, _instance
        )
        batch_shape = list(batch_shape)
        new.given_count = self.given_count.expand(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)

        if "log_partition" in self.__dict__:
            new.log_partition = self.log_partition.expand(batch_shape)

        super(SimpleRandomSamplingWithoutReplacement, new).__init__(
            torch.Size(batch_shape), self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size([])):
        sample_shape = torch.Size(sample_shape)
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            total_count = self.total_count.expand(shape[:-1])
            given_count = self.given_count.expand(shape[:-1])
            b = simple_random_sampling_without_replacement(
                total_count, given_count, self.event_shape[0]
            )
        return b

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (-self.log_partition).expand(value.shape[:-1])

