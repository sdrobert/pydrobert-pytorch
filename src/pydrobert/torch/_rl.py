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

import torch

from ._compat import script
from ._wrappers import functional_wrapper


@functional_wrapper("TimeDistributedReturn")
@script
def time_distributed_return(
    r: torch.Tensor, gamma: float, batch_first: bool = False
) -> torch.Tensor:
    if r.dim() != 2:
        raise RuntimeError("r must be 2 dimensional")
    if not gamma:
        return r
    if batch_first:
        exp = torch.arange(r.size(1), device=r.device, dtype=r.dtype)
        discount = torch.pow(gamma, exp)
        discount = (discount.unsqueeze(1) / discount.unsqueeze(0)).tril()
        R = torch.matmul(r, discount)
    else:
        exp = torch.arange(r.size(0), device=r.device, dtype=r.dtype)
        discount = torch.pow(gamma, exp)
        discount = (discount.unsqueeze(0) / discount.unsqueeze(1)).triu()
        R = torch.matmul(discount, r)
    return R


class TimeDistributedReturn(torch.nn.Module):
    r"""Accumulate future local rewards at every time step

    In `reinforcement learning
    <https://en.wikipedia.org/wiki/Reinforcement_learning>`__, the return is defined as
    the sum of discounted future rewards. This function calculates the return for a
    given time step :math:`t` as

    .. math::

        R_t = \sum_{t'=t} \gamma^(t' - t) r_{t'}

    Where :math:`r_{t'}` gives the (local) reward at time :math:`t'` and :math:`\gamma`
    is the discount factor. :math:`\gamma \in [0, 1)` implies convergence, but this is
    not enforced here.

    When instantiated, this module has the signature::

        R = time_distributed_return(r)
    
    where `r` is a two-dimensional tensor of shape ``(steps, batch_size)``, ``r[t, n]``
    being the (0-indexed) ``t``-th element of the ``n``-th batch element sequence.
    The return value `R` is a tensor of the same shape.


    Parameters
    ----------
    r : torch.Tensor
        A two-dimensional float tensor of shape ``(steps, batch_size)`` (or
        ``(batch_size, steps)`` if `batch_first` is :obj:`True`) of local rewards. The
        :math:`t` dimension is the step dimension
    gamma : float
        The discount factor :math:`\gamma`.
    batch_first : bool, optional
        Transposes the dimensions of `r` and `R` if :obj:`True`.

    See Also
    --------
    :ref:`Gradient Estimators`
        Provides an example of reinforcement learning that uses this function
    """

    __constants__ = ["gamma", "batch_first"]

    gamma: float
    batch_first: bool

    def __init__(self, gamma: float, batch_first: bool):
        super().__init__()
        self.gamma = gamma
        self.batch_first = batch_first

    def extra_repr(self) -> str:
        return f"gamma={self.gamma},batch_first={self.batch_first}"

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return time_distributed_return(r, self.gamma, self.batch_first)
