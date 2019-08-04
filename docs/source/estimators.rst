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

Example TODO
