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

import torch
import param
import pydrobert.torch

from pydrobert.torch.util import optimizer_to, error_rate, optimal_completion

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

    `hyp` is a long tensor of shape ``(num_batches, max_hyp_steps)`` if
    `batch_first` is :obj:`False` otherwise ``(max_hyp_steps, num_batches)``
    that provides the hypothesis transcriptions. Likewise, `ref` of shape
    ``(num_batches, max_ref_steps)`` or ``(max_ref_steps, num_batches)``
    providing reference transcriptions. `logits` is a 4-dimensional tensor of
    shape ``(num_batches, max_hyp_steps, num_classes)`` if `batch_first` is
    :obj:`True`, ``(max_hyp_steps, num_batches, num_classes)`` otherwise. A
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

    def forward(self, logits, ref, hyp, warn=True):
        if logits.dim() != 3:
            raise ValueError('logits must be 3 dimensional')
        if logits.shape[:-1] != hyp.shape:
            raise ValueError('first two dims of logits must match hyp shape')
        num_classes = logits.shape[-1]
        if self.include_eos and self.eos is not None and (
                (self.eos < 0) or (self.eos >= num_classes)):
            raise ValueError(
                'if include_eos=True, eos ({}) must be a class idx'.format(
                    self.eos))
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
            logits.view(-1, num_classes), optimals.flatten()
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
        elif self.reduction != 'none':
            raise ValueError(
                '{} is not a valid value for reduction'.format(self.reduction))
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

    This loss function has one of the following signatures when called::

        loss(log_probs, ref, hyp)
        loss(logits, ref, hyp[, log_probs])

    `hyp` is a long tensor of shape ``(num_batches, num_paths, max_hyp_steps)``
    if `batch_first` is :obj:`False` otherwise ``(max_hyp_steps, num_batches,
    num_paths)`` that provides the hypothesis transcriptions. Likewise, `ref`
    of shape ``(num_batches, num_paths, max_ref_steps)`` or ``(max_ref_steps,
    num_batches, num_paths)`` providing reference transcriptions.
    ``num_batches`` enumerates the batches whereas ``num_paths`` enumerates
    the list of paths for a given batch element.

    `log_probs` is a two dimensional tensor of shape ``(num_batches,
    num_paths)`` providing the log joint probabilities of every path. Without
    `logits`, the loss is calculated as

    .. math::

        loss_{MER} = SoftMax(log\_probs)[ER(hyp_i, ref) - \mu_i]

    where :math:`\mu_i` is the average error rate along paths in the batch
    element :math:`i`. :math:`mu_i` can be removed by setting `sub_avg` to
    :obj:`False`.

    `logits` is a 4-dimensional tensor of shape ``(num_batches, num_paths,
    max_hyp_steps, num_classes)`` if `batch_first` is :obj:`True`,
    ``(max_hyp_steps, num_batches, num_paths, num_classes)`` otherwise.
    A softmax over the step dimension defines the per-step distribution over
    class labels. If `logits` is provided, an additional cross-entropy loss
    term comparing `logits` and `ref` will be added to the loss

    .. math::

        loss_{combined} = loss_{MER} + \lambda loss_{CE}

    If `logits` is provided, ``max_hyp_steps >= max_ref_steps``. Logits past
    reference boundaries will be ignored. Note that :math:`loss_{MER}` is
    derived from probability space, whereas :math:`loss_{CE}` is derived from
    log-probabilty space.

    If `log_probs` is provided in addition to `logits`, the former will be
    used in calculating :math:`loss_{MER}`. Otherwise, `log_prob` will be
    inferred from `logits` by assuming the Markov property and summing along
    the paths

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
        rate. Used in error rate term
    batch_first : bool, optional
        Whether batch/path dimensions come first, or the step dimension
    norm : bool, optional
        If :obj:`False`, will use edit distances instead of error rates. Used
        in error rate term
    ins_cost : float, optional
        The cost of an adding a superfluous token to a transcript in `hyp`.
        Used in error rate term
    del_cost : float, optional
        The cost of missing a token from `ref`. Used in error rate term
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`. Used in
        error rate term
    lmb : float, optional
        The contribution of the cross entropy term, when `logits` is passed
    ignore_index : int, optional
        A reference transcript symbol indicating that this index will be
        ignored. Used in cross entropy term only
    weight : torch.tensor, optional
        A manual rescaling weight given to each class. Used in cross entropy
        term only
    reduction : {'mean', 'none', 'sum'}, optional
        Specifies the reduction to be applied to the output. 'none': no
        reduction will be applied. 'sum': the output will be summed. 'mean':
        the output will be averaged.

    Attributes
    ----------
    eos, ignore_index : int
    include_eos, sub_avg, batch_first, norm : bool
    ins_cost, del_cost, sub_cost, lmb : float
    reduction : {'mean', 'none', 'sum'}

    Warnings
    --------
    The criteria for ignoring parts of `ref` differ between :math:`loss_{MER}`
    and :math:`loss_{CE}`, the former relying on `eos` and the latter relying
    on `ignore_index`. The distinction is made because the loss terms are
    ultimately doing different things. For example, :math:`loss_{MER}` might
    be calculated using the string that ends at the first occurence of `eos`,
    but the cross-entropy term might want to use tokens in `hyp` past it to
    match the underlying reference token length

    See Also
    --------
    pydrobert.torch.util.beam_search_advance
        For getting an n-best list into `hyp`
    '''

    def __init__(
            self, eos=None, include_eos=True, sub_avg=True, batch_first=False,
            norm=True, ins_cost=1., del_cost=1., sub_cost=1., lmb=0.01,
            ignore_index=pydrobert.torch.INDEX_PAD_VALUE, weight=None,
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
        self.lmb = lmb
        self.reduction = reduction
        self._cross_ent = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=weight, reduction='none'
        )

    @property
    def ignore_index(self):
        return self._cross_ent.ignore_index

    @ignore_index.setter
    def ignore_index(self, value):
        self._cross_ent.ignore_index = value

    @property
    def weight(self):
        return self._cross_ent.weight

    @weight.setter
    def weight(self, value):
        self._cross_ent.weight = value

    def forward(self, logits, ref, hyp, log_probs=None, warn=True):
        if hyp.dim() != 3:
            raise ValueError('hyp must be 3 dimensional')
        if ref.dim() != 3:
            raise ValueError('ref must be 3 dimensional')
        if log_probs is None:
            if logits.dim() == 2:
                log_probs, logits = logits, None
            elif logits.dim() != 4:
                raise ValueError(
                    'Expected first argument to have 2 or 4 dimensions')
        if logits is not None:
            if logits.dim() != 4:
                raise ValueError('Expected logits to have 4 dimensions')
            num_classes = logits.shape[-1]
            if logits.shape[:-1] != hyp.shape:
                raise ValueError(
                    'logits and hyp must agree on first three dimensions')
            if log_probs is None:
                dist = torch.nn.functional.log_softmax(logits, 3)
                logits_on_paths = dist.gather(3, hyp.unsqueeze(3)).squeeze(3)
                log_probs = logits_on_paths.sum(2 if self.batch_first else 0)
        if self.batch_first:
            num_batches, num_paths, max_ref_steps = ref.shape
            num_batches_, num_paths_, max_hyp_steps = hyp.shape
            flat_ref = ref.view(-1, max_ref_steps)
            flat_hyp = hyp.view(-1, max_hyp_steps)
            min_steps = min(max_ref_steps, max_hyp_steps)
            ref = ref[..., :min_steps]
            if logits is not None:
                logits = logits[..., :min_steps, :]
        else:
            max_ref_steps, num_batches, num_paths = ref.shape
            max_hyp_steps, num_batches_, num_paths_ = hyp.shape
            flat_ref = ref.view(max_ref_steps, -1)
            flat_hyp = hyp.view(max_hyp_steps, -1)
            min_steps = min(max_ref_steps, max_hyp_steps)
            ref = ref[:min_steps]
            if logits is not None:
                logits = logits[:min_steps]
        if (num_batches, num_paths) != (num_batches_, num_paths_):
            raise ValueError('batch and path dims must match btw ref and hyp')
        if num_paths < 2:
            raise ValueError('must be more than one path')
        er = error_rate(
            flat_ref, flat_hyp, eos=self.eos, include_eos=self.include_eos,
            norm=self.norm, batch_first=self.batch_first,
            ins_cost=self.ins_cost, del_cost=self.del_cost,
            sub_cost=self.sub_cost, warn=warn,
        ).view(num_batches, num_paths)
        if self.sub_avg:
            er = er - er.mean(1, keepdim=True)
        loss = er * torch.nn.functional.softmax(log_probs, 1)
        if self.lmb and logits is not None:
            # we always sum out the "steps" dim, which is why we don't do any
            # reduction.
            ref = ref.flatten()
            logits = logits.contiguous().view(-1, num_classes)
            ce_loss = self._cross_ent(logits, ref)
            ce_loss = ce_loss.masked_fill(ref == self.ignore_index, 0.)
            if self.batch_first:
                ce_loss = ce_loss.view(num_batches, num_paths, -1).sum(2)
            else:
                ce_loss = ce_loss.view(-1, num_batches, num_paths).sum(0)
            loss = loss + self.lmb * ce_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction != 'none':
            raise ValueError(
                '{} is not a valid value for reduction'.format(self.reduction))
        return loss


class TrainingStateParams(param.Parameterized):
    '''Parameters controlling a TrainingStateController'''
    num_epochs = param.Integer(
        None, bounds=(1, None),
        doc='Total number of epochs to run for. If unspecified, runs '
        'until the early stopping criterion (or infinitely if disabled) '
    )
    log10_learning_rate = param.Number(
        None, softbounds=(-10, -2),
        doc='Optimizer log-learning rate. If unspecified, uses the '
        'built-in rate'
    )
    early_stopping_threshold = param.Number(
        0., bounds=(0, None), softbounds=(0, 1.),
        doc='Minimum improvement in xent from the last best that resets the '
        'early stopping clock. If zero, early stopping will not be performed'
    )
    early_stopping_patience = param.Integer(
        1, bounds=(1, None), softbounds=(1, 30),
        doc='Number of epochs where, if the classifier has failed to '
        'improve it\'s error, training is halted'
    )
    early_stopping_burnin = param.Integer(
        0, bounds=(0, None), softbounds=(0, 10),
        doc='Number of epochs before the early stopping criterion kicks in'
    )
    reduce_lr_threshold = param.Number(
        0., bounds=(0, None), softbounds=(0, 1.),
        doc='Minimum improvement in xent from the last best that resets the '
        'clock for reducing the learning rate. If zero, the learning rate '
        'will not be reduced during training. Se'
    )
    reduce_lr_factor = param.Number(
        None, bounds=(0, 1), softbounds=(0, .5),
        inclusive_bounds=(False, False),
        doc='Factor by which to multiply the learning rate if there has '
        'been no improvement in the error after "reduce_lr_patience" '
        'epochs. If unset, uses the pytorch defaults'
    )
    reduce_lr_patience = param.Integer(
        1, bounds=(1, None), softbounds=(1, 30),
        doc='Number of epochs where, if the classifier has failed to '
        'improve it\'s error, the learning rate is reduced'
    )
    reduce_lr_cooldown = param.Integer(
        0, bounds=(0, None), softbounds=(0, 10),
        doc='Number of epochs after reducing the learning rate before we '
        'resume checking improvements'
    )
    reduce_lr_log10_epsilon = param.Number(
        -8, bounds=(None, 0),
        doc='The log10 absolute difference between error rates that, below '
        'which, reducing the error rate is considered meaningless'
    )
    reduce_lr_burnin = param.Integer(
        0, bounds=(0, None), softbounds=(0, 10),
        doc='Number of epochs before the criterion for reducing the learning '
        'rate kicks in'
    )
    seed = param.Integer(
        None,
        doc='Seed used for training procedures (e.g. dropout). If '
        'unset, will not touch torch\'s seeding'
    )
    dropout_prob = param.Magnitude(
        0.,
        doc='The probability of dropping a hidden unit during training'
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

        If unset, the history will not be stored/loaded
    state_dir : str, optional
        A path to a directory to store/load model and optimizer states. If
        unset, the information will not be stored/loaded

    Attributes
    ----------
    params : TrainingStateParams
    state_csv_path : str or None
    state_dir : str or None
    cache_hist : dict
        A dictionary of cached results per epoch. Is not guaranteed to be
        up-to-date with `state_csv_path` unless :func:`update_cache` is called
    '''

    def __init__(self, params, state_csv_path=None, state_dir=None):
        super(TrainingStateController, self).__init__()
        self.params = params
        for s in (
                self.params.saved_model_fmt, self.params.saved_optimizer_fmt):
            if not any(x[1] == 'epoch' for x in Formatter().parse(s)):
                warnings.warn(
                    'State format string "{}" does not contain "epoch" field, '
                    'so is possibly not unique. In this case, only the state '
                    'of the last epoch will persist'.format(s))
        self.state_csv_path = state_csv_path
        self.state_dir = state_dir
        self.cache_hist = dict()

    def update_cache(self):
        '''Update the cache with history stored in state_csv_path'''
        if 0 not in self.cache_hist:
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
            if self.params.log10_learning_rate is not None:
                self.cache_hist[0]['lr'] = (
                    10 ** self.params.log10_learning_rate)
        if (self.state_csv_path is None or
                not os.path.exists(self.state_csv_path)):
            return
        with open(self.state_csv_path) as f:
            reader = DictReader(f)
            for row in reader:
                self.cache_hist[int(row['epoch'])] = {
                    'epoch': int(row['epoch']),
                    'es_resume_cd': int(row['es_resume_cd']),
                    'es_patience_cd': int(row['es_patience_cd']),
                    'rlr_resume_cd': int(row['rlr_resume_cd']),
                    'rlr_patience_cd': int(row['rlr_patience_cd']),
                    'lr': float(row['lr']),
                    'train_met': float(row['train_met']),
                    'val_met': float(row['val_met']),
                }

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

        If `epoch` is not specified or 0, the model and optimizer are
        initialized with states for the beginning of the experiment. Otherwise,
        we look for appropriately named files in ``self.state_dir``
        '''
        model_device = next(model.parameters()).device
        if not epoch:
            # reset on cpu. Different devices can randomize differently
            model.cpu().reset_parameters()
            optim_defaults = dict(optimizer.defaults)
            if self.params.log10_learning_rate is not None:
                optim_defaults['lr'] = 10 ** self.params.log10_learning_rate
            else:
                del optim_defaults['lr']
            if self.params.seed is not None:
                torch.manual_seed(self.params.seed)
            new_optimizer = type(optimizer)(
                model.parameters(),
                **optim_defaults
            )
            model.to(model_device)
            optimizer_to(optimizer, model_device)
            optimizer.load_state_dict(new_optimizer.state_dict())
        elif self.state_dir is not None:
            epoch_info = self[epoch]
            model_basename = self.params.saved_model_fmt.format(**epoch_info)
            optimizer_basename = self.params.saved_optimizer_fmt.format(
                **epoch_info)
            model_state_dict = torch.load(
                os.path.join(self.state_dir, model_basename),
                map_location=model_device
            )
            model.load_state_dict(model_state_dict)
            optimizer_state_dict = torch.load(
                os.path.join(self.state_dir, optimizer_basename),
                map_location=model_device
            )
            optimizer.load_state_dict(optimizer_state_dict)
        else:
            print(
                'Unable to load optimizer for epoch {}. No state dict!'
                ''.format(epoch))

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
        "rlr_resume_cd", "rlr_patience_cd", "lr", "train_met", and "val_met".

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
            "train_met", and "val_met"
        '''
        if self.state_dir is None:
            return
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        model_basename = self.params.saved_model_fmt.format(**info)
        optimizer_basename = self.params.saved_optimizer_fmt.format(
            **info)
        model_state_dict = model.state_dict()
        # we always save on the cpu
        for key, val in model_state_dict.items():
            model_state_dict[key] = val.cpu()
        optimizer_to(optimizer, 'cpu')
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
            "train_met", and "val_met"
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
                ])
            wr.writerow([
                epoch_fmt_str.format(info['epoch']),
                es_resume_cd_fmt_str.format(info['es_resume_cd']),
                es_patience_cd_fmt_str.format(info['es_patience_cd']),
                rlr_resume_cd_fmt_str.format(info['rlr_resume_cd']),
                rlr_patience_cd_fmt_str.format(info['rlr_patience_cd']),
                lr_fmt_str.format(info['lr']),
                train_met_fmt_str.format(info['train_met']),
                val_met_fmt_str.format(info['val_met']),
            ])

    def update_for_epoch(
            self, model, optimizer, train_met, val_met, epoch=None):
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
