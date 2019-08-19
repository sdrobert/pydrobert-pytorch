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

'''Functions and classes involved in training'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import warnings

from csv import DictReader, writer
from string import Formatter
from collections import OrderedDict

import torch
import param
import pydrobert.torch

from pydrobert.torch.util import error_rate, optimal_completion

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"
__all__ = [
    'HardOptimalCompletionDistillationLoss',
    'MinimumErrorRateLoss',
    'TrainingStateParams',
    'TrainingStateController',
]


class HardOptimalCompletionDistillationLoss(torch.nn.Module):
    r'''A categorical loss function over optimal next tokens

    Optimal Completion Distillation (OCD) [sabour2018]_ tries to minimize the
    train/test discrepancy in transcriptions by allowing seq2seq models to
    generate whatever sequences they want, then assigns a per-step loss
    according to whatever next token would set the model on a path that
    minimizes the edit distance in the future.

    In its "hard" version, the version used in the paper, the OCD loss function
    is simply a categorical cross-entropy loss of each hypothesis token's
    distribution versus those optimal next tokens, averaged over the number of
    optimal next tokens:

    .. math::

        loss(logits_t) = \frac{-\log Pr(s_t|logits_t)}{|S_t|}

    Where :math:`s_t \in S_t` are tokens from the set of optimal next tokens
    given :math:`hyp_{\leq t}` and `ref`. The loss is decoupled from an exact
    prefix of `ref`, meaning that `hyp` can be longer or shorter than `ref`.

    When called, this loss function has the signature::

        loss(logits, ref, hyp)

    `hyp` is a long tensor of shape ``(max_hyp_steps, batch_size)`` if
    `batch_first` is :obj:`False`, otherwise ``(batch_size, max_hyp_steps)``
    that provides the hypothesis transcriptions. Likewise, `ref` of shape
    ``(max_ref_steps, batch_size)`` or ``(batch_size, max_ref_steps)``
    providing reference transcriptions. `logits` is a 4-dimensional tensor of
    shape ``(max_hyp_steps, batch_size, num_classes)`` if `batch_first` is
    :obj:`False`, ``(batch_size, max_hyp_steps, num_classes)`` otherwise. A
    softmax over the step dimension defines the per-step distribution over
    class labels.

    Parameters
    ----------
    eos : int, optional
        A special token in `ref` and `hyp` whose first occurrence in each
        batch indicates the end of a transcript
    include_eos : bool, optional
        Whether to include the first instance of `eos` found in both `ref` and
        `hyp` as valid tokens to be computed as part of the distance. `eos`
        must be a valid class index if `include_eos` is :obj:`True`
    batch_first : bool, optional
        Whether the batch dimension comes first, or the step dimension
    ins_cost : float, optional
        The cost of an adding a superfluous token to a transcript in `hyp`
    del_cost : float, optional
        The cost of missing a token from `ref`
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`
    weight : torch.FloatTensor, optional
        A manual rescaling weight given to each class
    reduction : {'mean', 'none', 'sum'}, optional
        Specifies the reduction to be applied to the output. 'none': no
        reduction will be applied. 'sum': the output will be summed. 'mean':
        the output will be averaged.

    Attributes
    ----------
    eos : int
    include_eos, batch_first : bool
    ins_cost, del_cost, sub_cost : float
    reduction : {'mean', 'none', 'sum'}
    weight : torch.FloatTensor or None

    See Also
    --------
    pydrobert.torch.util.optimal_completion
        Used to determine the optimal next token set :math:`S`
    pydrobert.torch.util.random_walk_advance
        For producing a random `hyp` based on `logits` if the underlying
        model producing `logits` is auto-regressive. Also provides an example
        of sampling non-auto-regressive models
    '''

    def __init__(
            self, eos=None, include_eos=True, batch_first=False, ins_cost=1.,
            del_cost=1., sub_cost=1., weight=None, reduction='mean'):
        super(HardOptimalCompletionDistillationLoss, self).__init__()
        self.eos = eos
        self.include_eos = include_eos
        self.batch_first = batch_first
        self.ins_cost = ins_cost
        self.del_cost = del_cost
        self.sub_cost = sub_cost
        self.reduction = reduction
        self._cross_ent = torch.nn.CrossEntropyLoss(
            weight=weight, reduction='none'
        )

    @property
    def weight(self):
        return self._cross_ent.weight

    @weight.setter
    def weight(self, value):
        self._cross_ent.weight = value

    def check_input(self, logits, ref, hyp):
        '''Check if input formatted correctly, otherwise RuntimeError'''
        if logits.dim() != 3:
            raise RuntimeError('logits must be 3 dimensional')
        if logits.shape[:-1] != hyp.shape:
            raise RuntimeError('first two dims of logits must match hyp shape')
        if self.include_eos and self.eos is not None and (
                (self.eos < 0) or (self.eos >= logits.shape[-1])):
            raise RuntimeError(
                'if include_eos=True, eos ({}) must be a class idx'.format(
                    self.eos))
        if self.reduction not in {'mean', 'sum', 'none'}:
            raise RuntimeError(
                '"{}" is not a valid value for reduction'
                ''.format(self.reduction))

    def forward(self, logits, ref, hyp, warn=True):
        self.check_input(logits, ref, hyp)
        num_classes = logits.shape[-1]
        # the padding we use will never be exposed to the user, so we merely
        # ensure we're not trampling the eos
        padding = -2 if self.eos == -1 else -1
        self._cross_ent.ignore_index = padding
        optimals = optimal_completion(
            ref, hyp, eos=self.eos, include_eos=self.include_eos,
            batch_first=self.batch_first, ins_cost=self.ins_cost,
            del_cost=self.del_cost, sub_cost=self.sub_cost,
            padding=padding, exclude_last=True, warn=warn,
        )
        max_unique_next = optimals.shape[-1]
        logits = logits.unsqueeze(2).expand(-1, -1, max_unique_next, -1)
        logits = logits.contiguous()
        loss = self._cross_ent(
            logits.view(-1, logits.shape[-1]), optimals.flatten()
        ).view_as(optimals)
        padding_mask = optimals.eq(padding)
        no_padding_mask = padding_mask.eq(0)
        loss = loss.masked_fill(padding_mask, 0.).sum(2)
        loss = torch.where(
            no_padding_mask.any(2),
            loss / no_padding_mask.float().sum(2),
            loss,
        )
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class MinimumErrorRateLoss(torch.nn.Module):
    r'''Error rate expectation normalized over some number of transcripts

    Proposed in [prabhavalkar2018]_ though similar ideas had been explored
    previously. Given a subset of all possible token sequences and their
    associated probability mass over that population, this loss calculates the
    probability mass normalized over the subset, then calculates the
    expected error rate over that normalized distribution. That is, given some
    sequences :math:`s \in S \subseteq P`, the loss for a given reference
    transcription :math:`s^*` is

    .. math::

        \mathcal{L}(s, s^*) = \frac{Pr(s) ER(s, s^*)}{\sum_{s'} Pr(s')}

    This is an exact expectation over :math:`S` but not over :math:`P`. The
    larger the mass covered by :math:`S`, the closer the expectation is to the
    population - especially so for an n-best list (though it would be biased).

    This loss function has the following signature::

        loss(log_probs, ref, hyp)

    `log_probs` is a tensor of shape ``(batch_size, samples)`` providing the
    log joint probabilities of every path. `hyp` is a long tensor of shape
    ``(max_hyp_steps, batch_size, samples)`` if `batch_first` is :obj:`False`
    otherwise ``(batch_size, samples, max_hyp_steps)`` that provides the
    hypothesis transcriptions. `ref` is a 2- or 3-dimensional tensor. If 2D, it
    is of shape ``(max_ref_steps, batch_size)`` (or ``(batch_size,
    max_ref_steps)``). Alternatively, `ref` can be of shape ``(max_ref_steps,
    batch_size, samples)`` or ``(batch_size, samples, max_ref_steps)``.

    If `ref` is 2D, the loss is calculated as

    .. math::

        loss_{MER} = SoftMax(log\_probs)[ER(hyp_i, ref) - \mu_i]

    where :math:`\mu_i` is the average error rate along paths in the batch
    element :math:`i`. :math:`mu_i` can be removed by setting `sub_avg` to
    :obj:`False`. Note that each hypothesis is compared against the same
    reference as long as the batch element remains the same

    If `ref` is 3D, the loss is calculated as

    .. math::

        loss_{MER} = SoftMax(log\_probs)[ER(hyp_i, ref_i) - \mu_i]

    In this version, each hypothesis is compared against a unique reference

    Parameters
    ----------
    eos : int, optional
        A special token in `ref` and `hyp` whose first occurrence in each
        batch indicates the end of a transcript
    include_eos : bool, optional
        Whether to include the first instance of `eos` found in both `ref` and
        `hyp` as valid tokens to be computed as part of the distance.
    sub_avg : bool, optional
        Whether to subtract the average error rate from each pathwise error
        rate
    batch_first : bool, optional
        Whether batch/path dimensions come first, or the step dimension
    norm : bool, optional
        If :obj:`False`, will use edit distances instead of error rates
    ins_cost : float, optional
        The cost of an adding a superfluous token to a transcript in `hyp`
    del_cost : float, optional
        The cost of missing a token from `ref`
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`
    reduction : {'mean', 'none', 'sum'}, optional
        Specifies the reduction to be applied to the output. 'none': no
        reduction will be applied. 'sum': the output will be summed. 'mean':
        the output will be averaged.

    Attributes
    ----------
    eos, ignore_index : int
    include_eos, sub_avg, batch_first, norm : bool
    ins_cost, del_cost, sub_cost : float
    reduction : {'mean', 'none', 'sum'}

    Notes
    -----

    A previous version of this module incorporated a Maximum Likelihood
    Estimate (MLE) into the loss as in [prabhavalkar2018]_, which required
    `logits` instead of `log_probs`. This was overly complicated, given the
    user can easily incorporate the additional loss term herself by using
    :class:`torch.nn.CrossEntropyLoss`. Take a look at the example below for
    how to recreate this

    Examples
    --------

    Assume here that `logits` is the output of some neural network, and that
    `hyp` has somehow been produced from that (e.g. a beam search or random
    walk). We combine this loss function with a cross-entropy/MLE term to
    sort-of recreate [prabhavalkar2018]_.

    >>> from pydrobert.torch.util import sequence_log_probs
    >>> steps, batch_size, num_classes, eos, padding = 30, 20, 10, 0, -1
    >>> samples, lmb = 10, .01
    >>> logits = torch.randn(
    ...     steps, samples, batch_size, num_classes, requires_grad=True)
    >>> hyp = torch.randint(num_classes, (steps, samples, batch_size))
    >>> ref_lens = torch.randint(1, steps + 1, (batch_size,))
    >>> ref_lens[0] = steps
    >>> ref = torch.nn.utils.rnn.pad_sequence(
    ...     [torch.randint(1, num_classes, (x,)) for x in ref_lens],
    ...     padding_value=padding,
    ... )
    >>> ref[ref_lens - 1, range(batch_size)] = eos
    >>> ref = ref.unsqueeze(1).repeat(1, samples, 1)
    >>> mer = MinimumErrorRateLoss(eos=eos)
    >>> mle = torch.nn.CrossEntropyLoss(ignore_index=padding)
    >>> log_probs = sequence_log_probs(logits, hyp, eos=eos)
    >>> l = mer(log_probs, ref, hyp)
    >>> l = l + lmb * mle(logits.view(-1, num_classes), ref.flatten())
    >>> l.backward()

    See Also
    --------
    pydrobert.torch.util.beam_search_advance
        For getting an n-best list into `hyp` and some `log_probs`.
    pydrobert.torch.util.random_walk_advance
        For getting a random sample into `hyp`
    pydrobert.torch.util.sequence_log_probs
        For converting token log probs (or logits) to sequence log probs
    '''

    def __init__(
            self, eos=None, include_eos=True, sub_avg=True, batch_first=False,
            norm=True, ins_cost=1., del_cost=1., sub_cost=1.,
            reduction='mean'):
        super(MinimumErrorRateLoss, self).__init__()
        self.eos = eos
        self.include_eos = include_eos
        self.sub_avg = sub_avg
        self.batch_first = batch_first
        self.norm = norm
        self.ins_cost = ins_cost
        self.del_cost = del_cost
        self.sub_cost = sub_cost
        self.reduction = reduction

    def check_input(self, log_probs, ref, hyp):
        '''Check if the input is formatted correctly, otherwise RuntimeError'''
        if log_probs.dim() != 2:
            raise RuntimeError('log_probs must be 2 dimensional')
        if hyp.dim() != 3:
            raise RuntimeError('hyp must be 3 dimensional')
        if ref.dim() not in {2, 3}:
            raise RuntimeError('ref must be 2 or 3 dimensional')
        if self.batch_first:
            if ref.dim() == 2:
                ref = ref.unsqueeze(1).expand(-1, hyp.shape[1], -1)
            if (
                    (ref.shape[:2] != hyp.shape[:2]) or
                    (ref.shape[:2] != log_probs.shape)):
                raise RuntimeError(
                    'ref and hyp batch_size and sample dimensions must match')
            if ref.shape[1] < 2:
                raise RuntimeError(
                    'Batch must have at least two samples, got {}'
                    ''.format(ref.shape[1]))
        else:
            if ref.dim() == 2:
                ref = ref.unsqueeze(-1).expand(-1, -1, hyp.shape[-1])
            if (
                    (ref.shape[1:] != hyp.shape[1:]) or
                    (ref.shape[1:] != log_probs.shape)):
                raise RuntimeError(
                    'ref and hyp batch_size and sample dimensions must match')
            if ref.shape[2] < 2:
                raise RuntimeError(
                    'Batch must have at least two samples, got {}'
                    ''.format(ref.shape[2]))
        if self.reduction not in {'mean', 'sum', 'none'}:
            raise RuntimeError(
                '"{}" is not a valid value for reduction'
                ''.format(self.reduction))

    def forward(self, log_probs, ref, hyp, warn=True):
        self.check_input(log_probs, ref, hyp)
        if self.batch_first:
            batch_size, samples, max_hyp_steps = hyp.shape
            max_ref_steps = ref.shape[-1]
            if ref.dim() == 2:
                ref = ref.unsqueeze(1).repeat(1, samples, 1)
            ref = ref.view(-1, max_ref_steps)
            hyp = hyp.view(-1, max_hyp_steps)
        else:
            max_hyp_steps, batch_size, samples = hyp.shape
            max_ref_steps = ref.shape[0]
            if ref.dim() == 2:
                ref = ref.unsqueeze(-1).repeat(1, 1, samples)
            ref = ref.view(max_ref_steps, -1)
            hyp = hyp.view(max_hyp_steps, -1)
        er = error_rate(
            ref, hyp, eos=self.eos, include_eos=self.include_eos,
            norm=self.norm, batch_first=self.batch_first,
            ins_cost=self.ins_cost, del_cost=self.del_cost,
            sub_cost=self.sub_cost, warn=warn,
        ).view(batch_size, samples)
        if self.sub_avg:
            er = er - er.mean(1, keepdim=True)
        loss = er * torch.nn.functional.softmax(log_probs, 1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class TrainingStateParams(param.Parameterized):
    '''Parameters controlling a TrainingStateController

    This class implements the
    :class:`pydrobert.param.optuna.TunableParameterized` interface
    '''
    num_epochs = param.Integer(
        None, bounds=(1, None), softbounds=(10, 100),
        doc='Total number of epochs to run for. If unspecified, runs '
        'until the early stopping criterion (or infinitely if disabled) '
    )
    log10_learning_rate = param.Number(
        None, softbounds=(-10, -2),
        doc='Initial optimizer log-learning rate. If unspecified, the initial '
        'learning rate of the optimizer instance remains unchanged'
    )
    early_stopping_threshold = param.Number(
        0.0, bounds=(0, None), softbounds=(0, 1.),
        doc='Minimum magnitude decrease in validation metric from the last '
        'best that resets the early stopping clock. If zero, the learning '
        'rate will never be reduced'
    )
    early_stopping_patience = param.Integer(
        1, bounds=(1, None), softbounds=(1, 30),
        doc='Number of epochs after which, if the classifier has failed to '
        'decrease its validation metric by a threshold, training is '
        'halted'
    )
    early_stopping_burnin = param.Integer(
        0, bounds=(0, None), softbounds=(0, 10),
        doc='Number of epochs before the early stopping criterion kicks in'
    )
    reduce_lr_threshold = param.Number(
        0.0, bounds=(0, None), softbounds=(0, 1.),
        doc='Minimum magnitude decrease in validation metric from the last '
        'best that resets the clock for reducing the learning rate. If zero, '
        'the learning rate will never be reduced'
    )
    reduce_lr_factor = param.Number(
        None, bounds=(0, 1), softbounds=(0, .5),
        inclusive_bounds=(False, False),
        doc='Factor by which to multiply the learning rate if there has '
        'been no improvement in the  after "reduce_lr_patience" '
        'epochs. If unset, uses the pytorch defaults'
    )
    reduce_lr_patience = param.Integer(
        1, bounds=(1, None), softbounds=(1, 30),
        doc='Number of epochs after which, if the classifier has failed to '
        'decrease its validation metric by a threshold, the learning rate is '
        'reduced'
    )
    reduce_lr_cooldown = param.Integer(
        0, bounds=(0, None), softbounds=(0, 10),
        doc='Number of epochs after reducing the learning rate before we '
        'resume checking improvements'
    )
    reduce_lr_log10_epsilon = param.Number(
        -8, bounds=(None, 0),
        doc='The log10 absolute difference between learning rates that, '
        'below which, reducing the learning rate is considered meaningless'
    )
    reduce_lr_burnin = param.Integer(
        0, bounds=(0, None), softbounds=(0, 10),
        doc='Number of epochs before the criterion for reducing the learning '
        'rate kicks in'
    )
    seed = param.Integer(
        None,
        doc='Seed used for training procedures (e.g. dropout). If '
        "unset, will not touch torch's seeding"
    )
    keep_last_and_best_only = param.Boolean(
        True,
        doc='If the model is being saved, keep only the model and optimizer '
        'parameters for the last and best epoch (in terms of validation loss).'
        ' If False, save every epoch. See also "saved_model_fmt" and '
        '"saved_optimizer_fmt"'
    )
    saved_model_fmt = param.String(
        'model_{epoch:03d}.pt',
        doc='The file name format string used to save model state information.'
        ' Entries from the state csv are used to format this string (see '
        'TrainingStateController)'
    )
    saved_optimizer_fmt = param.String(
        'optim_{epoch:03d}.pt',
        doc='The file name format string used to save optimizer state '
        'information. Entries from the state csv are used to format this '
        'string (see TrainingStateController)'
    )

    _tunable = (
        'num_epochs', 'log10_learning_rate',
        'early_stopping_threshold', 'early_stopping_patience',
        'early_stopping_burnin', 'reduce_lr_threshold',
        'reduce_lr_patience', 'reduce_lr_cooldown',
    )

    @classmethod
    def get_tunable(cls):
        return set(cls._tunable)

    @classmethod
    def suggest_params(cls, trial, base=None, only=None, prefix=''):
        if only is None:
            only = cls._tunable
        params = cls() if base is None else base
        pdict = params.params()
        eps = torch.finfo(torch.float).eps
        for name in only:
            pp = pdict.get(name, None)
            if pp is None:
                continue
            softbounds = pp.get_soft_bounds()
            if name in {
                    'num_epochs',
                    'early_stopping_patience', 'early_stopping_burnin',
                    'reduce_lr_patience', 'reduce_lr_cooldown'}:
                val = trial.suggest_int(prefix + name, *softbounds)
            elif name in {
                    'log10_learning_rate', 'early_stopping_threshold',
                    'reduce_lr_patience'}:
                softbounds = softbounds[0] + eps, softbounds[1]
                val = trial.suggest_uniform(prefix + name, *softbounds)
            setattr(params, name, val)
        return params


class TrainingStateController(object):
    '''Controls the state of training a model

    This class is used to help both control and persist experiment information
    like the current epoch, the model parameters, and model error. It assumes
    that the values stored in `params` have not changed when resuming a run.
    It is also used to control learning rates and early stopping.

    Examples
    --------

    >>> params = TrainingStateParams(num_epochs=5)
    >>> model = torch.nn.Linear(10, 1)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> controller = TrainingStateController(
    ...    params,
    ...    state_csv_path='log.csv',
    ...    state_dir='states')
    >>> controller.load_model_and_optimizer_for_epoch(
    ...     model, optimizer, controller.get_last_epoch())  # load previous
    >>> for epoch in range(params.num_epochs):
    >>>     # do training loop for epoch
    >>>     train_loss, val_loss = 0.1, 0.01
    >>>     if not controller.update_for_epoch(
    ...             model, optimizer, train_loss, val_loss):
    >>>         break  # early stopping

    Parameters
    ----------
    params : TrainingStateParams
    state_csv_path : str, optional
        A path to where training state information is stored. It stores in
        comma-separated-values format the following information. Note that
        stored values represent the state *after* updates due to epoch
        results, such as the learning rate. That way, an experiment can be
        resumed without worrying about updating the loaded results

        1. "epoch": the epoch associated with this row of information
        2. "es_resume_cd": the number of epochs left before the early
           stopping criterion begins/resumes
        3. es_patience_cd: the number of epochs left that must pass
           without much improvement before training halts due to early stopping
        4. "rlr_resume_cd": the number of epochs left before the
           criterion for reducing the learning rate begins/resumes
        5. "rlr_patience_cd": the number of epochs left that must pass
           without much improvement before the learning rate is reduced
        6. "lr": the learning rate of the optimizer after any updates
        7. "train_met": mean training metric in exponent format. The metric
           is assumed to be lower is better
        8. "val_met": mean validation metric in exponent format. The metric
           is assumed to be lower is better
        9. Any additional entries added through :func:`add_entry`

        If unset, the history will not be stored/loaded
    state_dir : str, optional
        A path to a directory to store/load model and optimizer states. If
        unset, the information will not be stored/loaded
    warn : bool, optional
        Whether to warn using :mod:`warnings` module when a format string does
        not contain the "epoch" field

    Attributes
    ----------
    params : TrainingStateParams
    state_csv_path : str or None
    state_dir : str or None
    user_entry_types : OrderedDict
        A collection of user entries specified by :func:`add_entry`
    cache_hist : dict
        A dictionary of cached results per epoch. Is not guaranteed to be
        up-to-date with `state_csv_path` unless :func:`update_cache` is called
    '''

    def __init__(self, params, state_csv_path=None, state_dir=None, warn=True):
        super(TrainingStateController, self).__init__()
        self.params = params
        if warn:
            for s in (
                    self.params.saved_model_fmt,
                    self.params.saved_optimizer_fmt):
                if not any(x[1] == 'epoch' for x in Formatter().parse(s)):
                    warnings.warn(
                        'State format string "{}" does not contain "epoch" '
                        'field, so is possibly not unique. In this case, only '
                        'the state of the last epoch will persist. To '
                        'suppress this warning, set warn=False'.format(s))
        self.state_csv_path = state_csv_path
        self.state_dir = state_dir
        self.cache_hist = dict()
        self.user_entry_types = OrderedDict()

    def add_entry(self, name, type_=str):
        '''Add an entry to to be stored and retrieved at every epoch

        This method is useful when training loops need specialized, persistent
        information on every epoch. Prior to the first time any information is
        saved via :func:`update_for_epoch`, this method can be called with an
        entry `name` and optional `type_`. The user is then expected to provide
        a keyword argument with that `name` every time :func:`update_for_epoch`
        is called. The values of those entries can be retrieved via
        :func:`get_info`, cast to `type_`, for any saved epoch

        Parameters
        ----------
        name : str
        type_ : type, optional
            `type_` should be a type that is serialized from a string via
            `type_(str_obj)` and serialized to a string via `str(type_obj)`

        Examples
        --------
        >>> params = TrainingStateParams()
        >>> controller = TrainingStateController(params)
        >>> model = torch.nn.Linear(10, 1)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> controller.add_entry('important_value', int)
        >>> controller.update_for_epoch(
        ...     model, optimizer, 0.1, 0.1, important_value=3)
        >>> controller.update_for_epoch(
        ...     model, optimizer, 0.2, 0.01, important_value=4)
        >>> assert controller[1]['important_value'] == 3
        >>> assert controller[2]['important_value'] == 4

        Notes
        -----
        :func:`add_entry` must be called prior to :func:`update_for_epoch`
        or :func:`save_info_to_hist`, or it may corrupt the experiment history.
        However, the controller can safely ignore additional entries when
        loading history from a CSV. Thus, there is no need to call
        :func:`add_entry` if no new training is to be done (unless those
        entries are needed outside of training)
        '''
        if name in {
                "epoch", "es_resume_cd", "es_patience_cd", "rlr_resume_cd",
                "rlr_patience_cd", "lr", "train_met", "val_met"}:
            raise ValueError('"{}" is a reserved entry name'.format(name))
        if not isinstance(type_, type):
            raise ValueError('type_ ({}) must be a type'.format(type_))
        self.user_entry_types[name] = type_
        self.update_cache()

    def update_cache(self):
        '''Update the cache with history stored in state_csv_path'''
        # add a dummy entry for epoch "0" just to make logic easier. We
        # won't save it
        self.cache_hist[0] = {
            'epoch': 0,
            'es_resume_cd': self.params.early_stopping_burnin,
            'es_patience_cd': self.params.early_stopping_patience,
            'rlr_resume_cd': self.params.reduce_lr_burnin,
            'rlr_patience_cd': self.params.reduce_lr_patience,
            'train_met': float('inf'),
            'val_met': float('inf'),
            'lr': None,
        }
        self.cache_hist[0].update(
            dict((key, None) for key in self.user_entry_types))
        if self.params.log10_learning_rate is not None:
            self.cache_hist[0]['lr'] = (
                10 ** self.params.log10_learning_rate)
        if (self.state_csv_path is None or
                not os.path.exists(self.state_csv_path)):
            return
        with open(self.state_csv_path) as f:
            reader = DictReader(f)
            for row in reader:
                epoch = int(row['epoch'])
                self.cache_hist[epoch] = {
                    'epoch': epoch,
                    'es_resume_cd': int(row['es_resume_cd']),
                    'es_patience_cd': int(row['es_patience_cd']),
                    'rlr_resume_cd': int(row['rlr_resume_cd']),
                    'rlr_patience_cd': int(row['rlr_patience_cd']),
                    'lr': float(row['lr']),
                    'train_met': float(row['train_met']),
                    'val_met': float(row['val_met']),
                }
                for name, type_ in self.user_entry_types.items():
                    self.cache_hist[epoch][name] = type_(row[name])

    def get_last_epoch(self):
        '''int : last finished epoch from training, or 0 if no history'''
        self.update_cache()
        return max(self.cache_hist)

    def get_best_epoch(self, train_met=False):
        '''Get the epoch that has lead to the best validation metric val so far

        The "best" is the lowest recorded.

        Parameters
        ----------
        train_met : bool, optional
            If :obj:`True` look for the best training metric value instead

        Returns
        -------
        epoch : int
            The corresponding 'best' epoch, or :obj:`0` if no epochs have run
        '''
        ent = 'train_met' if train_met else 'val_met'
        self.update_cache()
        min_epoch = 0
        min_met = self.cache_hist[0][ent]
        for info in self.cache_hist.values():
            if min_met > info[ent]:
                min_epoch = info['epoch']
                min_met = info[ent]
        return min_epoch

    def load_model_and_optimizer_for_epoch(self, model, optimizer, epoch=0):
        '''Load up model and optimizer states, or initialize them

        If `epoch` is 0, the model and optimizer are initialized with states
        for the beginning of the experiment. Otherwise, we look for
        appropriately named files in ``self.state_dir``
        '''
        if not epoch:
            if self.params.seed is not None:
                torch.manual_seed(self.params.seed)
            if hasattr(model, 'reset_parameters'):
                model.reset_parameters()
            else:
                warnings.warn(
                    'model has no reset_parameters() method, so cannot '
                    'reset parameters for epoch 0'
                )
            if self.params.log10_learning_rate is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 10 ** self.params.log10_learning_rate
            optimizer.state.clear()
        elif self.state_dir is not None:
            epoch_info = self[epoch]
            model_basename = self.params.saved_model_fmt.format(**epoch_info)
            optimizer_basename = self.params.saved_optimizer_fmt.format(
                **epoch_info)
            model_state_dict = torch.load(
                os.path.join(self.state_dir, model_basename),
                map_location='cpu',
            )
            model.load_state_dict(model_state_dict)
            optimizer_state_dict = torch.load(
                os.path.join(self.state_dir, optimizer_basename),
                map_location='cpu',
            )
            optimizer.load_state_dict(optimizer_state_dict)
        else:
            warnings.warn(
                'Unable to load optimizer for epoch {}. No state dict!'
                ''.format(epoch)
            )

    def delete_model_and_optimizer_for_epoch(self, epoch):
        '''Delete state dicts for model and epoch off of disk, if they exist

        This method does nothing if the epoch records or the files do not
        exist. It is called during :func:`update_for_epoch` if the parameter
        ``keep_last_and_best_only`` is :obj:`True`

        Parameters
        ----------
        epoch : int
            The epoch in question
        '''
        if self.state_dir is None:
            return
        epoch_info = self.get_info(epoch, None)
        if epoch_info is None:
            return
        model_basename = self.params.saved_model_fmt.format(**epoch_info)
        optimizer_basename = self.params.saved_optimizer_fmt.format(
            **epoch_info)
        try:
            os.remove(os.path.join(self.state_dir, model_basename))
        except OSError as e:
            pass
        try:
            os.remove(os.path.join(self.state_dir, optimizer_basename))
        except OSError:
            pass

    def get_info(self, epoch, *default):
        '''Get history entries for a specific epoch

        If there's an entry present for `epoch`, return it. The value is a
        dictionary with the keys "epoch", "es_resume_cd", "es_patience_cd",
        "rlr_resume_cd", "rlr_patience_cd", "lr", "train_met", and "val_met",
        as well as any additional entries specified through :func:`add_entry`.

        If there's no entry for `epoch`, and no additional arguments were
        passed to this method, it raises a :class:`KeyError`. If an additional
        argument was passed to this method, return it.
        '''
        if len(default) > 1:
            raise TypeError('expected at most 2 arguments, got 3')
        if epoch in self.cache_hist:
            return self.cache_hist[epoch]
        self.update_cache()
        return self.cache_hist.get(epoch, *default)

    def __getitem__(self, epoch):
        return self.get_info(epoch)

    def save_model_and_optimizer_with_info(self, model, optimizer, info):
        '''Save model and optimizer state dictionaries to file given epoch info

        This is called automatically during :func:`update_for_epoch`. Does not
        save if there is no directory to save to (i.e. ``self.state_dir is
        None``). Format strings from ``self.params`` are formatted with the
        values from `info` to construct the base names of each file

        Parameters
        ----------
        model : AcousticModel
        optimizer : torch.optim.Optimizer
        info : dict
            A dictionary with the entries "epoch", "es_resume_cd",
            "es_patience_cd", "rlr_resume_cd", "rlr_patience_cd", "lr",
            "train_met", "val_met", and any entries specified through
            :func:`add_entry`
        '''
        if self.state_dir is None:
            return
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        model_basename = self.params.saved_model_fmt.format(**info)
        optimizer_basename = self.params.saved_optimizer_fmt.format(
            **info)
        torch.save(
            model.state_dict(),
            os.path.join(self.state_dir, model_basename),
        )
        torch.save(
            optimizer.state_dict(),
            os.path.join(self.state_dir, optimizer_basename),
        )

    def save_info_to_hist(self, info):
        '''Append history entries to the history csv

        This is called automatically during :func:`update_for_epoch`. Does not
        save if there is no file to save to (i.e. ``self.state_csv_path is
        None``). Values are appended to the end of the csv file - no checking
        is performed for mantaining a valid history.

        Parameters
        ----------
        info : dict
            A dictionary with the entries "epoch", "es_resume_cd",
            "es_patience_cd", "rlr_resume_cd", "rlr_patience_cd", "lr",
            "train_met", "val_met", and any other entries specified via
            :func:`add_entry`
        '''
        self.cache_hist[info['epoch']] = info
        if self.state_csv_path is None:
            return
        if not self.params.num_epochs:
            epoch_fmt_str = '{:010d}'
        else:
            epoch_fmt_str = '{{:0{}d}}'.format(
                int(math.log10(self.params.num_epochs)) + 1)
        es_resume_cd_fmt_str = '{{:0{}d}}'.format(
            int(math.log10(max(
                self.params.early_stopping_burnin,
                1,
                ))) + 1
        )
        es_patience_cd_fmt_str = '{{:0{}d}}'.format(
            int(math.log10(max(
                self.params.early_stopping_patience,
                1,
                ))) + 1
        )
        rlr_resume_cd_fmt_str = '{{:0{}d}}'.format(
            int(math.log10(max(
                self.params.reduce_lr_cooldown,
                self.params.reduce_lr_burnin,
                1,
            ))) + 1
        )
        rlr_patience_cd_fmt_str = '{{:0{}d}}'.format(
            int(math.log10(max(
                self.params.reduce_lr_patience,
                1,
            ))) + 1
        )
        lr_fmt_str = train_met_fmt_str = val_met_fmt_str = '{:10e}'
        write_header = not os.path.exists(self.state_csv_path)
        with open(self.state_csv_path, 'a') as f:
            wr = writer(f)
            if write_header:
                wr.writerow([
                    'epoch',
                    'es_resume_cd',
                    'es_patience_cd',
                    'rlr_resume_cd',
                    'rlr_patience_cd',
                    'lr',
                    'train_met',
                    'val_met',
                ] + list(self.user_entry_types))
            wr.writerow([
                epoch_fmt_str.format(info['epoch']),
                es_resume_cd_fmt_str.format(info['es_resume_cd']),
                es_patience_cd_fmt_str.format(info['es_patience_cd']),
                rlr_resume_cd_fmt_str.format(info['rlr_resume_cd']),
                rlr_patience_cd_fmt_str.format(info['rlr_patience_cd']),
                lr_fmt_str.format(info['lr']),
                train_met_fmt_str.format(info['train_met']),
                val_met_fmt_str.format(info['val_met']),
            ] + [str(info[x]) for x in self.user_entry_types])

    def update_for_epoch(
            self, model, optimizer, train_met, val_met, epoch=None,
            **kwargs):
        '''Update history and optimizer after latest epoch results

        Parameters
        ----------
        model : AcousticModel
        optimizer : torch.optim.Optimizer
        train_met : float
            Mean value of metric on training set for epoch
        val_met : float
            Mean value of metric on validation set for epoch
        epoch : int, optional
            The epoch that just finished. If unset, it is inferred to be one
            after the last epoch in the history
        kwargs : Keyword arguments, optional
            Additional keyword arguments can be used to specify the values
            of entries specified via :func:`add_entry`

        Returns
        -------
        continue_training : bool
            Whether to continue training. This can be set to :obj:`False`
            either by hitting the max number of epochs or by early stopping
        '''
        if epoch is None:
            epoch = self.get_last_epoch() + 1
        if not self.params.num_epochs:
            continue_training = True
        else:
            continue_training = epoch < self.params.num_epochs
        info = dict(self.get_info(epoch - 1, None))
        if info is None:
            raise ValueError(
                'no entry for the previous epoch {}, so unable to update'
                ''.format(epoch))
        for key, value in kwargs.items():
            if key not in self.user_entry_types:
                raise TypeError(
                    'update_for_epoch() got an unexpected keyword argument '
                    '"{}" (did you forget to add_entry()?)'.format(key))
            elif not isinstance(value, self.user_entry_types[key]):
                raise ValueError(
                    'keyword argument "{}" value is not of type {}'
                    ''.format(key, self.user_entry_types[key]))
            info[key] = value
        remaining_user_entries = set(self.user_entry_types) - set(kwargs)
        if remaining_user_entries:
            raise TypeError(
                'The following keyword arguments were not provided as keyword '
                'arguments but were specified via add_entry(): {}'
                ''.format(sorted(remaining_user_entries)))
        last_best = self.get_best_epoch()
        best_info = self[last_best]
        if info['lr'] is None:
            # can only happen during the first epoch. We don't know the
            # optimizer defaults, so we get them now
            info['lr'] = optimizer.defaults['lr']
        if info["es_resume_cd"]:
            info["es_resume_cd"] -= 1
        elif (max(best_info['val_met'] - val_met, 0) <
                self.params.early_stopping_threshold):
            info["es_patience_cd"] -= 1
            if not info["es_patience_cd"]:
                continue_training = False
        else:
            info["es_patience_cd"] = self.params.early_stopping_patience
        if info["rlr_resume_cd"]:
            info["rlr_resume_cd"] -= 1
        elif (max(best_info['val_met'] - val_met, 0) <
                self.params.reduce_lr_threshold):
            info["rlr_patience_cd"] -= 1
            if not info["rlr_patience_cd"]:
                old_lr = info['lr']
                new_lr = old_lr * self.params.reduce_lr_factor
                rlr_epsilon = 10 ** self.params.reduce_lr_log10_epsilon
                if old_lr - new_lr > rlr_epsilon:
                    info['lr'] = new_lr
                    for param_group in optimizer.param_groups:
                        # just assume that the user knows what's what if
                        # the optimizer's lr doesn't match the old one
                        param_group['lr'] = new_lr
                info["rlr_resume_cd"] = self.params.reduce_lr_cooldown
                info["rlr_patience_cd"] = self.params.reduce_lr_patience
        else:
            info["rlr_patience_cd"] = self.params.reduce_lr_patience
        info["epoch"] = epoch
        info["val_met"] = val_met
        info["train_met"] = train_met
        # in the unlikely event that there's a SIGTERM here, this block tries
        # its best to maintain a valid history on exit. We have to delete the
        # old states first in case the file names match the new states
        self.cache_hist[info['epoch']] = info
        cur_best = self.get_best_epoch()
        try:
            if self.params.keep_last_and_best_only and cur_best != epoch - 1:
                self.delete_model_and_optimizer_for_epoch(epoch - 1)
            if self.params.keep_last_and_best_only and cur_best != last_best:
                self.delete_model_and_optimizer_for_epoch(last_best)
        finally:
            self.save_model_and_optimizer_with_info(model, optimizer, info)
            self.save_info_to_hist(info)
        return continue_training
