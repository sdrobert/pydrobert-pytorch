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

.. math:: g \approx \partial fb / \partial \theta

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

BERNOULLI_SYNONYMS = {"bern", "Bern", "bernoulli", "Bernoulli"}
CATEGORICAL_SYNONYMS = {"cat", "Cat", "categorical", "Categorical"}


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
    # it's okay to detach logits here, since dz / dlogits = 1. No info
    # will be lost by just differentiating up to z
    if dist in BERNOULLI_SYNONYMS:
        z = logits.detach() + torch.log(u) - torch.log1p(-u)
    elif dist in CATEGORICAL_SYNONYMS:
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
    if dist in BERNOULLI_SYNONYMS:
        b = z.gt(0.).to(z)
    elif dist in CATEGORICAL_SYNONYMS:
        b = z.argmax(dim=-1).to(z)
    else:
        raise ValueError("Unknown distribution {}".format(dist))
    return b


def to_fb(f, b):
    '''Simply call f(b)'''
    return f(b)


def reinforce(fb, b, logits, dist):
    r'''Perform REINFORCE gradient estimation

    REINFORCE, or the score function, has a single-sample implementation as

    .. math:: g = f(b) \partial \log Pr(b; logits) / \partial logits

    It is an unbiased estimate of the derivative of the expectation.

    Arguments
    ---------
    fb : torch.Tensor
    b : torch.Tensor
    logits : torch.Tensor
    dist : {"bern", "cat"}

    Returns
    -------
    g : torch.tensor
        A tensor with the same shape as `logits` representing the estimate
        of ``d fb / d logits``

    Notes
    -----
    It is common (such as in A2C) to include a baseline to minimize the
    variance of the estimate. It's incorporated as `c` in

    .. math:: g = (f(b) - c) \log Pr(b; logits) / \partial logits

    Note that :math:`c_i` should be conditionally independent of :math:`b_i`
    for `g` to be unbiased. You can, however, condition on any preceding
    outputs :math:`b_{i - j}, j > 0` and all of `logits`.

    To get this functionality, simply subtract `c` from `fb` before passing it
    to this method. If `c` is the output of a neural network, a common (but
    sub-optimal) loss function is the mean-squared error between `fb` and `c`.
    '''
    fb = fb.detach()
    b = b.detach()
    logits = logits.detach().requires_grad_(True)
    if dist in BERNOULLI_SYNONYMS:
        log_pb = torch.distributions.Bernoulli(logits=logits).log_prob(b)
    elif dist in CATEGORICAL_SYNONYMS:
        log_pb = torch.distributions.Categorical(logits=logits).log_prob(b)
    else:
        raise ValueError("Unknown distribution {}".format(dist))
    g, = torch.autograd.grad([log_pb], [logits], grad_outputs=fb.float())
    return g


def _to_z_tilde(logits, b, dist):
    v = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
    # z_tilde ~ Pr(z|b, logits)
    # see REBAR paper for more details
    if dist in BERNOULLI_SYNONYMS:
        om_theta = torch.sigmoid(-logits)  # 1 - \theta
        v_prime = b * (v * (1 - om_theta) + om_theta) + (1. - b) * v * om_theta
        z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
    elif dist in CATEGORICAL_SYNONYMS:
        b = b.long()
        theta = torch.softmax(logits, dim=-1)
        mask = torch.zeros_like(logits, dtype=torch.uint8)
        mask.scatter_(-1, b[..., None], 1)
        log_v = v.log()
        z_tilde = torch.where(
            mask,
            -torch.log(-log_v),
            -torch.log(-log_v / theta - log_v.gather(-1, b[..., None])),
        )
    else:
        raise ValueError("Unknown distribution {}".format(dist))
    return z_tilde


def relax(fb, b, logits, z, surrogate, dist):
    fb = fb.detach()
    b = b.detach()
    logits = logits.detach().requires_grad_(True)
    # warning! d z_tilde / d logits is non-trivial. Needs graph from logits
    z_tilde = _to_z_tilde(logits, b, dist)
    c_z = surrogate(z)
    c_z_tilde = surrogate(z_tilde)
    diff = fb - c_z_tilde
    if dist in BERNOULLI_SYNONYMS:
        log_pb = torch.distributions.Bernoulli(logits=logits).log_prob(b)
    elif dist in CATEGORICAL_SYNONYMS:
        log_pb = torch.distributions.Categorical(logits=logits).log_prob(b)
        diff = diff.unsqueeze(-1)
    else:
        raise ValueError("Unknown distribution {}".format(dist))
    # reminder: d c_z / d logits = d c_z / d z * d z / d logits = d c_z / d z
    ones = torch.ones_like(c_z)
    dlog_pb, = torch.autograd.grad([log_pb], [logits], grad_outputs=ones)
    dc_z, = torch.autograd.grad(
        [c_z], [z], retain_graph=True, grad_outputs=ones)
    dc_z_tilde, = torch.autograd.grad(
        [c_z_tilde], [logits], retain_graph=True, grad_outputs=ones)
    g = diff * dlog_pb + dc_z - dc_z_tilde
    return g
