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

'''Functions and classes involved in training

Notes
-----
The loss functions :class:`HardOptimalCompletionDistillationLoss` and
:class:`MinimumErrorRateLoss` have been moved to :mod:`pydrobert.torch.layers`
'''

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


__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"
__all__ = [
    'TrainingStateParams',
    'TrainingStateController',
]


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
        'best that resets the early stopping clock. If zero, early stopping '
        'will never be performed'
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
    reduce_lr_factor = param.Magnitude(
        0.1, softbounds=(.1, .5), inclusive_bounds=(False, False),
        doc='Factor by which to multiply the learning rate if there has '
        'been no improvement in the  after "reduce_lr_patience" '
        'epochs'
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

    @classmethod
    def get_tunable(cls):
        '''Returns a set of tunable parameters'''
        return {
            'num_epochs', 'log10_learning_rate', 'early_stopping_threshold',
            'early_stopping_patience', 'early_stopping_burnin',
            'reduce_lr_factor', 'reduce_lr_threshold',
            'reduce_lr_patience', 'reduce_lr_cooldown',
            'reduce_lr_burnin',
        }

    @classmethod
    def suggest_params(cls, trial, base=None, only=None, prefix=''):
        '''Populate a parameterized instance with values from trial'''
        if only is None:
            only = cls.get_tunable()
        params = cls() if base is None else base
        pdict = params.param.params()
        if 'log10_learning_rate' in only:
            softbounds = pdict['log10_learning_rate'].get_soft_bounds()
            params.log10_learning_rate = trial.suggest_uniform(
                prefix + 'log10_learning_rate', *softbounds)
        if 'num_epochs' in only:
            softbounds = pdict['num_epochs'].get_soft_bounds()
            params.num_epochs = trial.suggest_int(
                prefix + 'num_epochs', *softbounds)
        if params.num_epochs is None:
            num_epochs = float('inf')
        else:
            num_epochs = params.num_epochs
        # if we sample patience and burnin so that their collective total
        # reaches or exceeds the number of epochs, they are effectively
        # disabled. Rather than allowing vast sums above the number of epochs,
        # we only allow the sum to reach the remaining epochs
        remaining_epochs = num_epochs
        if 'early_stopping_patience' not in only:
            remaining_epochs -= params.early_stopping_patience
        if 'early_stopping_burnin' not in only:
            remaining_epochs -= params.early_stopping_burnin
        remaining_epochs = max(0, remaining_epochs)
        if remaining_epochs and 'early_stopping_threshold' in only:
            softbounds = pdict['early_stopping_threshold'].get_soft_bounds()
            params.early_stopping_threshold = trial.suggest_uniform(
                prefix + 'early_stopping_threshold', *softbounds)
        if not params.early_stopping_threshold:
            remaining_epochs = 0
        if remaining_epochs and 'early_stopping_patience' in only:
            softbounds = pdict['early_stopping_patience'].get_soft_bounds()
            softbounds = tuple(min(x, remaining_epochs) for x in softbounds)
            params.early_stopping_patience = trial.suggest_int(
                prefix + 'early_stopping_patience', *softbounds)
            remaining_epochs -= params.early_stopping_patience
            assert remaining_epochs >= 0
        if remaining_epochs and 'early_stopping_burnin' in only:
            softbounds = pdict['early_stopping_burnin'].get_soft_bounds()
            softbounds = tuple(min(x, remaining_epochs) for x in softbounds)
            params.early_stopping_burnin = trial.suggest_int(
                prefix + 'early_stopping_burnin', *softbounds)
            remaining_epochs -= params.early_stopping_burnin
            assert remaining_epochs >= 0
        # we do the same thing, but for the learning rate scheduler
        remaining_epochs = num_epochs
        if 'reduce_lr_patience' not in only:
            remaining_epochs -= params.reduce_lr_patience
        if 'reduce_lr_burnin' not in only:
            remaining_epochs -= params.reduce_lr_burnin
        remaining_epochs = max(0, remaining_epochs)
        if remaining_epochs and 'reduce_lr_threshold' in only:
            softbounds = pdict['reduce_lr_threshold'].get_soft_bounds()
            params.reduce_lr_threshold = trial.suggest_uniform(
                prefix + 'reduce_lr_threshold', *softbounds)
        if not params.reduce_lr_threshold:
            remaining_epochs = 0
        if remaining_epochs and 'reduce_lr_patience' in only:
            softbounds = pdict['reduce_lr_patience'].get_soft_bounds()
            softbounds = tuple(min(x, remaining_epochs) for x in softbounds)
            params.reduce_lr_patience = trial.suggest_int(
                prefix + 'reduce_lr_patience', *softbounds)
            remaining_epochs -= params.reduce_lr_patience
        if remaining_epochs and 'reduce_lr_burnin' in only:
            softbounds = pdict['reduce_lr_burnin'].get_soft_bounds()
            softbounds = tuple(min(x, remaining_epochs) for x in softbounds)
            params.reduce_lr_burnin = trial.suggest_int(
                prefix + 'reduce_lr_burnin', *softbounds)
        if remaining_epochs and 'reduce_lr_factor' in only:
            softbounds = pdict['reduce_lr_factor'].get_soft_bounds()
            params.reduce_lr_factor = trial.suggest_uniform(
                prefix + 'reduce_lr_factor', *softbounds)
        if remaining_epochs and 'reduce_lr_cooldown' in only:
            softbounds = pdict['reduce_lr_cooldown'].get_soft_bounds()
            params.reduce_lr_cooldown = trial.suggest_int(
                prefix + 'reduce_lr_cooldown', *softbounds)
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
    >>> # load previous
    >>> controller.load_model_and_optimizer_for_epoch(model, optimizer)
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
    fmt_dict : dict
        A dictionary of format strings for the CSV entries
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
        self.fmt_dict = dict()
        if params.num_epochs is None:
            self.fmt_dict['epoch'] = '{:010d}'
        else:
            self.fmt_dict['epoch'] = '{{:0{}d}}'.format(
                int(math.log10(params.num_epochs)) + 1)
        self.fmt_dict['es_resume_cd'] = '{{:0{}d}}'.format(
            int(math.log10(max(params.early_stopping_burnin, 1))) + 1)
        self.fmt_dict['es_patience_cd'] = '{{:0{}d}}'.format(
            int(math.log10(max(params.early_stopping_patience, 1))) + 1)
        self.fmt_dict['rlr_resume_cd'] = '{{:0{}d}}'.format(
            int(math.log10(max(
                params.reduce_lr_cooldown,
                params.reduce_lr_burnin,
                1,
            ))) + 1
        )
        self.fmt_dict['rlr_patience_cd'] = '{{:0{}d}}'.format(
            int(math.log10(max(params.reduce_lr_patience, 1))) + 1)
        self.fmt_dict['lr'] = '{{:.{}e}}'.format(self.SCIENTIFIC_PRECISION - 1)
        self.fmt_dict['train_met'] = self.fmt_dict['lr']
        self.fmt_dict['val_met'] = self.fmt_dict['lr']

    '''The number of digits in significand of scientific notation

    Controls how many digits are saved when writing metrics and learning rate
    to disk (i.e. the ``x`` in ``x * 10^y``). Used when generating the format
    strings in ``self.fmt_dict`` on initialization
    '''
    SCIENTIFIC_PRECISION = 5

    def add_entry(self, name, type_=str, fmt='{}'):
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
            ``type_(str_obj)`` and serialized to a string via
            ``fmt.format(obj)``
        fmt : str, optional
            The format string used to serialize the objects into strings

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
        self.fmt_dict[name] = fmt
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

        The "best" is the lowest recorded validation metric. In the case of
        ties, the earlier epoch is chosen.

        Parameters
        ----------
        train_met : bool, optional
            If :obj:`True` look for the best training metric value instead

        Returns
        -------
        epoch : int
            The corresponding 'best' epoch, or :obj:`0` if no epochs have run

        Notes
        -----
        Negligible differences between epochs are determined by
        :obj:`TrainingStateController.METRIC_PRECISION`, which is relative
        to the metrics base 10. This is in contrast to early stopping criteria
        and learning rate annealing, whose thresholds are absolute.
        '''
        ent = 'train_met' if train_met else 'val_met'
        fmt = self.fmt_dict[ent]
        self.update_cache()
        min_epoch = 0
        min_met = self.cache_hist[0][ent]
        min_met = float(fmt.format(min_met))
        for info in self.cache_hist.values():
            cur = float(fmt.format(info[ent]))
            if cur < min_met:
                min_epoch = info['epoch']
                min_met = cur
        return min_epoch

    def load_model_for_epoch(self, model, epoch=None, strict=True):
        '''Load up just the model, or initialize it

        Parameters
        ----------
        model : torch.nn.Module
            Model state will be loaded into this
        epoch : int or :obj:`None`, optional
            The epoch from which the states should be loaded. We look for the
            appropriately named files in ``self.state_dir``. If `epoch` is
            :obj:`None`, the best epoch in recorded history will be loaded. If
            it's 0, the model is initialized with states from the beginning of
            the experiment
        strict : bool, optional
            Whether to strictly enforce that the keys in ``model.state_dict()``
            match those that were saved
        '''
        if epoch is None:
            epoch = self.get_best_epoch()
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
        elif self.state_dir is not None:
            epoch_info = self[epoch]
            model_basename = self.params.saved_model_fmt.format(**epoch_info)
            model_state_dict = torch.load(
                os.path.join(self.state_dir, model_basename),
                map_location='cpu',
            )
            model.load_state_dict(model_state_dict, strict=strict)
        else:
            warnings.warn(
                'Unable to load model for epoch {}. No state directory!'
                ''.format(epoch)
            )

    def load_model_and_optimizer_for_epoch(
            self, model, optimizer, epoch=None, strict=True):
        '''Load up model and optimizer states, or initialize them

        Parameters
        ----------
        model : torch.nn.Module
            Model state will be loaded into this
        optimizer : torch.optim.Optimizer
            Optimizer state will be loaded into this
        epoch : int or :obj:`None`, optional
            The epoch from which the states should be loaded. We look for the
            appropriately named files in ``self.state_dir``. If `epoch` is
            :obj:`None`, the last epoch in recorded history will be loaded. If
            it's 0, the model and optimizer are initialized with states
            for the beginning of the experiment.
        strict : bool, optional
            Whether to strictly enforce that the keys in ``model.state_dict()``
            match those that were saved
        '''
        if epoch is None:
            epoch = self.get_last_epoch()
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
            # there is no public API for resetting the state dictionary, so
            # we create a new instance as best as possible and copy the state
            # over from there. Note that settings like weight decay are already
            # part of the parameter group, so we don't need to worry about
            # initializing with them.
            brand_new_optimizer = type(optimizer)(optimizer.param_groups)
            optimizer.load_state_dict(brand_new_optimizer.state_dict())
            del brand_new_optimizer
        elif self.state_dir is not None:
            epoch_info = self[epoch]
            model_basename = self.params.saved_model_fmt.format(**epoch_info)
            optimizer_basename = self.params.saved_optimizer_fmt.format(
                **epoch_info)
            model_state_dict = torch.load(
                os.path.join(self.state_dir, model_basename),
                map_location='cpu',
            )
            model.load_state_dict(model_state_dict, strict=strict)
            optimizer_state_dict = torch.load(
                os.path.join(self.state_dir, optimizer_basename),
                map_location='cpu',
            )
            optimizer.load_state_dict(optimizer_state_dict)
        else:
            warnings.warn(
                'Unable to load model and optimizer for epoch {}. No state'
                'directory!'.format(epoch)
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
        except OSError:
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
        names = [
            'epoch',
            'es_resume_cd',
            'es_patience_cd',
            'rlr_resume_cd',
            'rlr_patience_cd',
            'lr',
            'train_met',
            'val_met',
        ]
        names += list(self.user_entry_types)
        write_header = not os.path.exists(self.state_csv_path)
        with open(self.state_csv_path, 'a') as f:
            wr = writer(f)
            if write_header:
                wr.writerow(names)
            wr.writerow([
                self.fmt_dict[k].format(info[k]) for k in names])

    def continue_training(self, epoch=None):
        '''Return a boolean on whether to continue training

        Useful when resuming training. Will check the training history at the
        target `epoch` and determine whether training should continue from that
        point, based on the total number of epochs and the early stopping
        criterion.

        Parameters
        ----------
        epoch : int or :obj:`None`, optional
            The epoch to check the history of. If :obj:`None`, the last epoch
            will be inferred

        Returns
        -------
        cont : bool
            :obj:`True` if training should continue
        '''
        if epoch is None:
            epoch = self.get_last_epoch()
        info = self[epoch]
        if not self.params.num_epochs:
            cont = True
        else:
            cont = epoch < self.params.num_epochs
        if self.params.early_stopping_threshold and not info["es_patience_cd"]:
            cont = False
        return cont

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
        cont : bool
            Whether to continue training. This can be set to :obj:`False`
            either by hitting the max number of epochs or by early stopping
        '''
        if epoch is None:
            epoch = self.get_last_epoch() + 1
        if not self.params.num_epochs:
            cont = True
        else:
            cont = epoch < self.params.num_epochs
            if epoch > self.params.num_epochs:
                warnings.warn(
                    'Training is continuing, despite passing num_epochs')
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
            if info["es_patience_cd"] < 0:
                warnings.warn(
                    "Early stopping criterion was already met, but training "
                    "has continued")
                info["es_patience_cd"] = 0
        else:
            info["es_patience_cd"] = self.params.early_stopping_patience
        # we do it this way in case someone continues training after early
        # stopping has been reached
        if self.params.early_stopping_threshold and not info["es_patience_cd"]:
            cont = False
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
        return cont
