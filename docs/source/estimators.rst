Gradient Estimators
===================

Sometimes we wish to parameterize a discrete probability distribution and
backpropagate through it, and the loss/reward function we use :math:`f: R^D \to
R` is calculated on samples :math:`b \sim logits` instead of directly on the
parameterization `logits`, for example, in reinforcement learning. A reasonable
approach is to marginalize out the sample by optimizing the expectation

.. math::

    L = E_b[f] = \sum_b f(b) Pr(b ; logits)

If that sum is combinatorially infeasible, such as in a reinforcement learning
scenario when one can't enumerate all possible actions, one can use gradient
estimates to get an error signal for `logits`.

The goal of the functions in :mod:`pydrobert.torch.estimators` is to find some
estimate

.. math::

    g \approx \partial E_b[f(b)] / \partial logits

which can be plugged into the "backward" call to logits as a surrogate error
signal.

Different estimators require different arguments. The following are common to
most.

- `logits` is the distribution parameterization. `logits` are supposed to
  represent a parameterization with an unbounded domain.
- `b` is a tensor of samples drawn from the distribution parametrized by
  `logits`
- `dist` specifies the distribution that `logits` parameterizes. Currently,
  there are three.

  1. The value ``"bern"`` corresponds to the Bernoulli
     distribution, which, for parameterizations
     :math:`logits \in R^{A \times B \ldots}` produces samples
     :math:`b \in \{0,1\}^{A \times B \ldots}` whose individual elements
     :math:`b_i` are drawn i.i.d. from :math:`Pr(b_i;logits_i)`. The value
  2. ``"cat"`` corresponds to the Categorical distribution. If the last
     dimension of :math:`logits \in R^{A \times B \times \ldots \times D}`
     is of size :math:`D` and :math:`i` indexes all other dimensions, then
     :math:`b \in [0, D-1]^{A \times B \ldots}` whose individual elements
     are i.i.d. :math:`b_i \sim Pr(b_i = d; logits_{i,d})`
  3. ``"onehot"`` is also Categorical, but
     :math:`b' \in \{0,1\}^{A \times B \times \ldots \times D}` is a one-hot
     representation of the categorical :math:`b` s.t.
     `b'_{i,d} = 1 \Leftrightarrow b_i = d`.

- `fb` is a tensor of the values of :math:`f(b)`. In general, `fb` should be
  the same size as `b`, meaning one evaluation per sample. The exception is
  ``"onehot"``: `fb` should not have the final dimension of `b` as ``b[i, :]``
  corresponds to a single sample

`b` can be sampled by first calling ``z = to_z(logits, dist)``, then
``b = to_b(z, dist)``. Other arguments can be acquired using functions with
similar patterns.

The following is a very simple example of training a neural network using
gradient estimators for Bernoulli samples. We are trying to maximize the
per-sample reward via reward function `f`, which is simply going to reward a 1
and penalize a 0 with some random noise inserted.

>>> import torch
>>> from pydrobert.torch.estimators import *
>>> def f(b):
>>>     return b + torch.randn_like(b)
>>> batch_size, input_size, mc_samples = 31, 11, 100
>>> inp = torch.randn(batch_size, input_size)
>>> ff = torch.nn.Linear(input_size, input_size)

First we'll use the standard REINFORCE estimator [williams1992]_, but `Monte
Carlo <https://en.wikipedia.org/wiki/Monte_Carlo_method>`__ to reduce variance.

>>> optim = torch.optim.Adam(ff.parameters())
>>> optim.zero_grad()
>>> logits = ff(inp)
>>> logits_mc = logits.unsqueeze(0).expand(mc_samples, batch_size, input_size)
>>> z_mc = to_z(logits_mc, 'bern')
>>> b_mc = to_b(z_mc, 'bern')
>>> fb_mc = f(b_mc)
>>> g_mc = reinforce(fb_mc, b_mc, logits_mc, 'bern')
>>> g = g_mc.mean(0)
>>> logits.backward(-g)  # negative b/c we do gradient descent, not ascent
>>> optim.step()

`logits_mc` merely copies the parameterization `logits` `mc_samples` times
along its 0-th dimension, so we're effectively sampling from each categorical
`mc_samples` times. Taking the mean over the gradient estimate's 0-th dimension
will average the estimates across those samples.

Here's how to train the same neural network using the RELAX [grathwohl2017]_
gradient estimator. Here, we're training the control variate `c` by minimizing
the variance of the gradient estimate of `logits`, ``g = g_mc.mean(0)``.

>>> c = torch.nn.Linear(input_size, input_size)
>>> optim = torch.optim.Adam(tuple(ff.parameters()) + tuple(c.parameters()))
>>> optim.zero_grad()
>>> logits = ff(inp)
>>> logits_mc = logits.unsqueeze(0).expand(mc_samples, batch_size, input_size)
>>> z_mc = to_z(logits_mc, 'bern')
>>> b_mc = to_b(z_mc, 'bern')
>>> fb_mc = f(b_mc)
>>> g_mc = relax(fb_mc, b_mc, logits_mc, z_mc, c, 'bern')
>>> g = g_mc.mean(0)
>>> (g ** 2).sum().backward()  # error signal propagates through c
>>> logits.backward(-g)
>>> optim.step()

Last, we'll show how to train an auto-regressive RNN with gradient estimators.
We'll recreate the loss function introduced in [tjandra2018]_, which rewards
new output that reduce the edit distance of the utterance. There are a number
of complexities in the paper that we forego in order to illustrate the role of
the estimator clearly.

Keep in mind that, while there are `num_samps` samples per input, this is not
quite the same as a Markov Estimator. This is because the underlying
parameterization `logits` is dependent upon the sample prefix.

The following code generates random input and references for some number of
seeds and keeps track of the convergence of both the original REINFORCE
estimator and the RELAX estimator with a very small bidirectional RNN as a
control variate. Data are saved to a `Pandas <https://pandas.pydata.org/>`__
dataframe, then to CSV. Later, we plot the aggregated per-iteration
descriptive statistics using `Matplotlib <https://matplotlib.org/>`__. This
file is saved as ``estimator_convergence.py`` in this file's directory.

.. include:: estimator_convergence.py
   :code: python

For me, the results (saved in ``estimator_convergence.csv``) were as below

.. image:: estimator_convergence.png

The legend has the format ``<estimator>-{d,g}-<hidden_size>``, where ``d`` is
a difference-based loss function for the control variate, and ``g`` tries to
minimize the variance of the estimated gradient. Solid lines represent the
mean error rates, whereas dashed lines represent the loss of the control
variate.

In this situation, I found that the score function without a baseline did best.
The graph shows that all estimators that use a control variate quickly learn to
match the reward function (0 loss implies a perfect match). If the baseline
matches the reward function too well, then the estimate is very low variance
but approaches 0. Hence, we find that decreasing the hidden size of the control
variate increases the control variate loss, but improves the objective overall.
An earlier version of this script didn't properly handle padding in the control
variate, which seemed to add enough variance back that the model could converge
to a lower mean error rate. In other words, it's important that your control
variate isn't *too* good.
