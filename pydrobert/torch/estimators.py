# Copyright 2019 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r'''Gradient estimators

Much of this code has been adapted from `David Duvenaud's repo
<https://github.com/duvenaud/relax>`_.

The goal of this module is to find some estimate

.. math:: g \approx d fb / d \theta

where `fb` is assumed not to be differentiable with respect to :math:`\theta`
as it relies on some :math:`b \sim \theta`. Instead of :math:`\theta`, we use
`logits`, where :math:`logits = \log(\theta / (1 - \theta))` when `\theta`
parameterizes Bernoullis in the usual way, and :math:`logits = \log(\theta)`
when `\theta` parameterizes categorical distributions in the usual way. `g` can
be plugged into backpropagation via something like ``logits.backward(g)``.
In this way, `logits` can be the output of a neural layer.

Different estimators require different arguments. In general, you'll need to
know what distribution you're parameterizing, the tensor containing the
parametrization, and have the function definition `f`. There are a number of
utility functions in this module with the template ``to_x`` that can convert
between these as necessary. This isn't done automatically, because `logits`,
`b`, and `fb` are not always available all at once.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"


def to_z(logits, dist):
    '''Samples random noise, then injects it into logits to produce z

    Parameters
    ----------
    logits : torch.FloatTensor
    dist : {"bern", "cat"}

    Returns
    -------
    z : torch.FloatTensor
    '''
    u = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
    if dist in {"bern", "Bern", "bernoulli", "Bernoulli"}:
        z = logits.detach() + torch.log(u) - torch.log1p(-u)
    elif dist in {"cat", "Cat", "categorical", "Categorical"}:
        z = logits.detach() - torch.log(-torch.log(u))
    else:
        raise ValueError("Unknown distribution {}".format(dist))
    z.requires_grad_(True)
    return z


def to_b(z, dist):
    '''Converts z to sample using a deterministic mapping

    Parameters
    ----------
    z : torch.FloatTensor
    dist : {"bern", "cat"}

    Returns
    -------
    b : torch.FloatTensor
    '''
    if dist in {"bern", "Bern", "bernoulli", "Bernoulli"}:
        b = z.gt(0.).to(z)
    elif dist in {"cat", "Cat", "categorical", "Categorical"}:
        b = z.argmax(dim=-1).to(z)
    else:
        raise ValueError("Unknown distribution {}".format(dist))
    return b


def to_fb(b, f):
    '''Simply call f(b)'''
    return f(b)


def reinforce(fb, b, logits):
    r'''Perform REINFORCE gradient estimation

    Arguments
    ---------
    fb : torch.Tensor
        The output of the function we're trying to optimize (`f(b)`). Should
        match the shape of b
    b : torch.Tensor
        The sample ``b \sim logits``
    logits : torch.Tensor
        The logit parameterization

    Returns
    -------
    g : torch.tensor
        A tensor with the same shape as `logits` representing the estimate
        of ``d fb / d logits``
    '''
    fb = fb.detach()
    b = b.detach()
    logits = logits.detach().requires_grad_(True)
    if fb.shape != b.shape:
        raise ValueError('fb does not have the same shape as b')
    if logits.shape == b.shape:
        log_pb = torch.distributions.Bernoulli(logits=logits).log_prob(
            b.float())
    elif b.shape == logits.shape[:-1]:
        log_pb = torch.distributions.Categorical(logits=logits).log_prob(
            b.float())
    else:
        raise ValueError('Do not know which distribution matches b and logits')
    g, = torch.autograd.grad([log_pb], [logits], grad_outputs=fb.float())
    return g
