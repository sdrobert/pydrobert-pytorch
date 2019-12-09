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

See Also
--------
:ref:`Gradient Estimators`
    A description of how to use this module, as well as an example
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import torch

try:
    torch_bool = torch.bool
except AttributeError:
    torch_bool = torch.uint8

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"

__all__ = [
    "to_z",
    "to_b",
    "to_fb",
    "reinforce",
    "relax",
    "REBARControlVariate",
]

BERNOULLI_SYNONYMS = {"bern", "Bern", "bernoulli", "Bernoulli"}
CATEGORICAL_SYNONYMS = {"cat", "Cat", "categorical", "Categorical"}
ONEHOT_SYNONYMS = {"onehot", "OneHotCategorical"}

# XXX(sdrobert): Implementation detail
# The Gumbel relaxation does not *require* a softmax to be applied to the
# logits before normalization. You can sample
#
# z_i = logits_i - log(-log u_i)
#
# Then z' becomes (for sample b_k = 1)
#
# z'_i = logits_k - log(-log v_k)  for i == k
#      = -log((log v_k)/(sum_l exp logits_l) - (log v_i)/(exp logits_i))  else
#
# However, when I ran tests, I found this was slower to compute and less
# numerically stable (though unbiased) than just taking the softmax beforehand


def to_z(logits, dist, warn=True):
    '''Samples a continuous relaxation of `dist` parameterized by `logits`

    Parameters
    ----------
    logits : torch.FloatTensor
    dist : {"bern", "cat", "onehot"}
    warn : bool, optional
        Estimators that require `z` as an argument will likely need to
        propagate through `z` to get a derivative w.r.t. `logits`. If `warn` is
        true and ``not logits.requires_grad``, a warning will be issued through
        the :mod:`warnings` module.

    Returns
    -------
    z : torch.FloatTensor
    '''
    if warn and not logits.requires_grad:
        warnings.warn(
            "logits.requires_grad is False. This will likely cause an error "
            "with estimators that require z. To suppress this warning, set "
            "warn to False"
        )
    u = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
    if dist in BERNOULLI_SYNONYMS:
        z = logits + torch.log(u) - torch.log1p(-u)
    elif dist in CATEGORICAL_SYNONYMS | ONEHOT_SYNONYMS:
        log_theta = torch.nn.functional.log_softmax(logits, dim=-1)
        z = log_theta - torch.log(-torch.log(u))
    else:
        raise RuntimeError("Unknown distribution {}".format(dist))
    z.requires_grad_(True)
    return z


def to_b(z, dist):
    '''Converts z to a discrete sample using a deterministic mapping

    Parameters
    ----------
    z : torch.FloatTensor
    dist : {"bern", "cat", "onehot"}

    Returns
    -------
    b : torch.FloatTensor
    '''
    if dist in BERNOULLI_SYNONYMS:
        b = z.gt(0.).to(z)
    elif dist in CATEGORICAL_SYNONYMS:
        b = z.argmax(dim=-1).to(z)
    elif dist in ONEHOT_SYNONYMS:
        b = torch.zeros_like(z).scatter_(
            -1, z.argmax(dim=-1, keepdim=True), 1.)
    else:
        raise RuntimeError("Unknown distribution {}".format(dist))
    return b


def to_fb(f, b, **kwargs):
    '''Simply call f(b)'''
    return f(b)


def reinforce(fb, b, logits, dist):
    r'''Perform REINFORCE gradient estimation

    REINFORCE [williams1992]_, or the score function, has a single-sample
    implementation as

    .. math:: g = f(b) \frac{\partial \log Pr(b; logits)}{\partial logits}

    It is an unbiased estimate of the derivative of the expectation w.r.t
    `logits`.

    Though simple, it is often cited as high variance

    Parameters
    ----------
    fb : torch.Tensor
    b : torch.Tensor
    logits : torch.Tensor
    dist : {"bern", "cat", "onehot"}

    Returns
    -------
    g : torch.tensor
        A tensor with the same shape as `logits` representing the estimate
        of ``d fb / d logits``

    Notes
    -----
    It is common (such as in A2C) to include a baseline to minimize the
    variance of the estimate. It's incorporated as `c` in

    .. math:: g = (f(b) - c)\frac{\partial \log Pr(b; logits)}{\partial logits}

    Note that :math:`c_i` should be conditionally independent of :math:`b_i`
    for `g` to be unbiased. You can, however, condition on any preceding
    outputs :math:`b_{i - j}, j > 0` and all of `logits`.

    To get this functionality, simply subtract :math:`c` from `fb` before
    passing it to this method. If :math:`c` is the output of a neural network,
    a common (but sub-optimal) loss function is the mean-squared error between
    `fb` and :math:`c`
    '''
    if dist in BERNOULLI_SYNONYMS:
        log_pb = torch.distributions.Bernoulli(logits=logits).log_prob(b)
    elif dist in CATEGORICAL_SYNONYMS:
        log_pb = torch.distributions.Categorical(logits=logits).log_prob(b)
        fb = fb.unsqueeze(-1)
    elif dist in ONEHOT_SYNONYMS:
        log_pb = torch.distributions.OneHotCategorical(
            logits=logits).log_prob(b)
        fb = fb.unsqueeze(-1)
    else:
        raise RuntimeError("Unknown distribution {}".format(dist))
    g = fb * torch.autograd.grad(
        [log_pb], [logits], grad_outputs=torch.ones_like(log_pb))[0]
    return g


def relax(fb, b, logits, z, c, dist, components=False, **kwargs):
    r'''Perform RELAX gradient estimation

    RELAX [grathwohl2017]_ has a single-sample implementation as

    .. math::

        g = (f(b) - c(\widetilde{z}))
                \frac{\partial \log Pr(b; logits)}{\partial logits}
            + \frac{\partial c(z)}{\partial logits}
            - \frac{\partial c(\widetilde{z})}{\partial logits}

    where :math:`b = H(z)`, :math:`\widetilde{z} \sim Pr(z|b, logits)`, and `c`
    can be any differentiable function. It is an unbiased estimate of the
    derivative of the expectation w.r.t `logits`.

    `g` is itself differentiable with respect to the parameters of the control
    variate `c`. If the c is trainable, an easy choice for its loss is to
    minimize the variance of `g` via ``(g ** 2).sum().backward()``. Propagating
    directly from `g` should be suitable for most situations. Insofar as the
    loss cannot be directly computed from `g`, setting the argument for
    `components` to true will return a tuple containing the terms of `g`
    instead.

    Parameters
    ----------
    fb : torch.FloatTensor
    b : torch.Tensor
    logits : torch.FloatTensor
    z : torch.FloatTensor
    c : callable
        A module or function that accepts input of the shape of `z` and outputs
        a tensor of the same shape if modelling a Bernoulli, or of shape
        ``z[..., 0]`` (minus the last dimension) if Categorical.
    dist : {"bern", "cat"}
    components : bool, optional
    kwargs : keyword arguments, optional
        Additional keyword arguments to pass to `c`

    Returns
    -------
    g : torch.FloatTensor or tuple
        If `components` is :obj:`False`, `g` will be the gradient estimate with
        respect to `logits`. Otherwise, a tuple will be returned of ``(diff,
        dlog_pb, dc_z, dc_z_tilde)`` which correspond to the terms in the above
        equation and can reconstruct `g` as ``g = diff * dlog_pb + dc_z -
        dc_z_tilde``.

    Notes
    -----
    RELAX is a generalized version of REBAR [tucker2017]_. For the REBAR
    estimator, use an instance of :class:`REBARControlVariate` for `c`. See the
    class for more details.
    '''
    # in all computations (including those made by the user later), we don't
    # want to backpropagate past "logits" into the model. We make a detached
    # copy of logits and rebuild the graph from the detached copy to z
    logits = logits.detach().requires_grad_(True)
    z = _reattach_z_to_new_logits(logits, z, dist)
    z_tilde = _to_z_tilde(logits, b, dist)
    c_z = c(z, **kwargs)
    c_z_tilde = c(z_tilde, **kwargs)
    diff = fb - c_z_tilde
    if dist in BERNOULLI_SYNONYMS:
        log_pb = torch.distributions.Bernoulli(logits=logits).log_prob(b)
    elif dist in CATEGORICAL_SYNONYMS:
        log_pb = torch.distributions.Categorical(
            logits=logits).log_prob(b)
        diff = diff[..., None]
    elif dist in ONEHOT_SYNONYMS:
        log_pb = torch.distributions.OneHotCategorical(
            logits=logits).log_prob(b)
        diff = diff[..., None]
    else:
        raise RuntimeError("Unknown distribution {}".format(dist))
    dlog_pb, = torch.autograd.grad(
        [log_pb], [logits], grad_outputs=torch.ones_like(log_pb))
    # we need `create_graph` to be true here or backpropagation through the
    # control variate will not include the graphs of the derivative terms
    dc_z, = torch.autograd.grad(
        [c_z], [logits], create_graph=True, retain_graph=True,
        grad_outputs=torch.ones_like(c_z))
    dc_z_tilde, = torch.autograd.grad(
        [c_z_tilde], [logits], create_graph=True, retain_graph=True,
        grad_outputs=torch.ones_like(c_z_tilde))
    if components:
        return (diff, dlog_pb, dc_z, dc_z_tilde)
    else:
        return diff * dlog_pb + dc_z - dc_z_tilde


class REBARControlVariate(torch.nn.Module):
    r'''The REBAR control variate, for use in RELAX

    REBAR [tucker2017]_ has a single sample implementation as:

    .. math::

        g = (f(b) - \eta f(\sigma_\lambda(\widetilde{z})))
                \frac{\partial \log Pr(b; logits)}{\partial logits}
            + \eta \frac{\partial f(\sigma_\lambda(z))}{\partial logits}
            - \eta \frac{\partial f(\sigma_\lambda(\widetilde{z}))}
                        {\partial logits}

    where :math:`b = H(z)`, :math:`\widetilde{z} \sim Pr(z|b, logits)`, and
    :math:`\sigma` is the Concrete relaxation [maddison2016]_ of the discrete
    distribution. It is an unbiased estimate of the derivative of the
    expectation w.r.t `logits`.

    As remarked in [grathwohl2017]_, REBAR can be considered a special case of
    RELAX with :math:`c(x) = \eta f(\sigma_\lambda(x))`. An instance of this
    class can be fed to the ``relax`` function as the argument ``c``. To
    optimize the temperature :math:`\lambda` and importance :math:`\eta`
    simultaneously with :math:`logits`, one can take the output of ``g =
    relax(...)`` and call ``(g ** 2).sum().backward()``.

    Parameters
    ----------
    f : callable
        The objective whose expectation we seek to minimize. Any additional
        keyword arguments specified when calling this instance will be passed
        along to `f`
    dist: {"bern", "cat", "onehot"}
    start_temp : float, optional
        The initial value for :math:`\lambda \in (0,\infty)`
    start_eta : float, optional
        The initial value for :math:`\eta \in R`
    warn : bool, optional
        If ``True``, a warning will be issued when ``dist == "cat"``.
        :math:`z` will be continuous relaxations of one-hot samples of
        categorical distributions, but the discrete samples are index-based
        when ``dist == "cat"``. This might cause unexpected behaviours.

    Attributes
    ----------
    f : callable
    dist : {"bern", "cat", "onehot"}
    '''

    def __init__(self, f, dist, start_temp=0.1, start_eta=1.0, warn=True):
        if start_temp <= 0.:
            raise ValueError("start_temp must be positive")
        super(REBARControlVariate, self).__init__()
        self.dist = dist
        self.f = f
        if warn and dist in CATEGORICAL_SYNONYMS:
            warnings.warn(
                "'{}' implies categorical samples are index-based, but "
                "this instance will call 'f' with continuous relaxations "
                "of one-hot samples. It is likely that you want dist to "
                "be 'onehot' instead. To suppress this warning, set "
                "warn=False".format(dist))
        self.start_temp = start_temp
        self.start_eta = start_eta
        self.log_temp = torch.nn.Parameter(torch.Tensor(1))
        self.eta = torch.nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.start_temp <= 0.:
            raise RuntimeError("start_temp must be positive")
        self.log_temp.data.fill_(self.start_temp).log_()
        self.eta.data.fill_(self.start_eta)

    def check_input(self, z):
        if self.dist not in (
                CATEGORICAL_SYNONYMS | BERNOULLI_SYNONYMS | ONEHOT_SYNONYMS):
            raise RuntimeError(
                'dist "{}" is not a valid distribution name'.format(self.dist))

    def forward(self, z, **kwargs):
        self.check_input(z)
        z_temp = z / torch.exp(self.log_temp)
        if self.dist in BERNOULLI_SYNONYMS:
            return self.eta * self.f(torch.sigmoid(z_temp), **kwargs)
        else:
            return self.eta * self.f(torch.softmax(z_temp, -1), **kwargs)


def _reattach_z_to_new_logits(logits, z, dist):
    if dist in BERNOULLI_SYNONYMS:
        # z = logits + torch.log(u) - torch.log1p(-u)
        z = z.detach() + logits - logits.detach()
    elif dist in CATEGORICAL_SYNONYMS | ONEHOT_SYNONYMS:
        log_theta = torch.nn.functional.log_softmax(logits, dim=-1)
        # z = log_theta - torch.log(-torch.log(u))
        z = z.detach() + log_theta - log_theta.detach()
    else:
        raise RuntimeError("Unknown distribution {}".format(dist))
    return z


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
        mask = torch.arange(logits.shape[-1], device=b.device)
        mask = (b.unsqueeze(-1) == mask)
        log_v = v.log()
        z_tilde = torch.where(
            mask,
            -torch.log(-log_v),
            -torch.log(-log_v / theta - log_v.gather(-1, b[..., None])),
        )
    elif dist in ONEHOT_SYNONYMS:
        b = b.byte()
        theta = torch.softmax(logits, dim=-1)
        log_v = v.log()
        z_tilde = torch.where(
            b,
            -torch.log(-log_v),
            -torch.log(
                -log_v / theta - log_v.gather(-1, b.argmax(-1, keepdim=True))),
        )
    else:
        raise RuntimeError("Unknown distribution {}".format(dist))
    return z_tilde
