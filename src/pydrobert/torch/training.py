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

"""Utilities for managing the training process"""

import os
import math
from typing import Optional
import warnings
import tempfile

from csv import DictReader, writer
from string import Formatter
from collections import OrderedDict

import torch
import torch.distributed
import param


__all__ = [
    "TrainingStateParams",
    "TrainingStateController",
]


class TrainingStateParams(param.Parameterized):
    """Parameters controlling a TrainingStateController

    This class implements the :class:`pydrobert.param.optuna.TunableParameterized`
    interface
    """

    num_epochs = param.Integer(
        None,
        bounds=(1, None),
        softbounds=(10, 100),
        doc="Total number of epochs to run for. If unspecified, runs "
        "until the early stopping criterion (or infinitely if disabled) ",
    )
    log10_learning_rate = param.Number(
        None,
        softbounds=(-10, -2),
        doc="Initial optimizer log-learning rate. If unspecified, the initial "
        "learning rate of the optimizer instance remains unchanged",
    )
    early_stopping_threshold = param.Number(
        0.0,
        bounds=(0, None),
        softbounds=(0, 1.0),
        doc="Minimum magnitude decrease in validation metric from the last "
        "best that resets the early stopping clock. If zero, early stopping "
        "will never be performed",
    )
    early_stopping_patience = param.Integer(
        1,
        bounds=(1, None),
        softbounds=(1, 30),
        doc="Number of epochs after which, if the classifier has failed to "
        "decrease its validation metric by a threshold, training is "
        "halted",
    )
    early_stopping_burnin = param.Integer(
        0,
        bounds=(0, None),
        softbounds=(0, 10),
        doc="Number of epochs before the early stopping criterion kicks in",
    )
    reduce_lr_threshold = param.Number(
        0.0,
        bounds=(0, None),
        softbounds=(0, 1.0),
        doc="Minimum magnitude decrease in validation metric from the last "
        "best that resets the clock for reducing the learning rate. If zero, "
        "the learning rate will never be reduced",
    )
    reduce_lr_factor = param.Magnitude(
        0.1,
        softbounds=(0.1, 0.5),
        inclusive_bounds=(False, False),
        doc="Factor by which to multiply the learning rate if there has "
        'been no improvement in the  after "reduce_lr_patience" '
        "epochs",
    )
    reduce_lr_patience = param.Integer(
        1,
        bounds=(1, None),
        softbounds=(1, 30),
        doc="Number of epochs after which, if the classifier has failed to "
        "decrease its validation metric by a threshold, the learning rate is "
        "reduced",
    )
    reduce_lr_cooldown = param.Integer(
        0,
        bounds=(0, None),
        softbounds=(0, 10),
        doc="Number of epochs after reducing the learning rate before we "
        "resume checking improvements",
    )
    reduce_lr_log10_epsilon = param.Number(
        -8,
        bounds=(None, 0),
        doc="The log10 absolute difference between learning rates that, "
        "below which, reducing the learning rate is considered meaningless",
    )
    reduce_lr_burnin = param.Integer(
        0,
        bounds=(0, None),
        softbounds=(0, 10),
        doc="Number of epochs before the criterion for reducing the learning "
        "rate kicks in",
    )
    seed = param.Integer(
        None,
        doc="Seed used for training procedures (e.g. dropout). If "
        "unset, will not touch torch's seeding",
    )
    keep_last_and_best_only = param.Boolean(
        True,
        doc="If the model is being saved, keep only the model and optimizer "
        "parameters for the last and best epoch (in terms of validation loss)."
        ' If False, save every epoch. See also "saved_model_fmt" and '
        '"saved_optimizer_fmt"',
    )
    saved_model_fmt = param.String(
        "model_{epoch:03d}.pt",
        doc="The file name format string used to save model state information."
        " Entries from the state csv are used to format this string (see "
        "TrainingStateController)",
    )
    saved_optimizer_fmt = param.String(
        "optim_{epoch:03d}.pt",
        doc="The file name format string used to save optimizer state "
        "information. Entries from the state csv are used to format this "
        "string (see TrainingStateController)",
    )

    @classmethod
    def get_tunable(cls):
        """Returns a set of tunable parameters"""
        return {
            "num_epochs",
            "log10_learning_rate",
            "early_stopping_threshold",
            "early_stopping_patience",
            "early_stopping_burnin",
            "reduce_lr_factor",
            "reduce_lr_threshold",
            "reduce_lr_patience",
            "reduce_lr_cooldown",
            "reduce_lr_burnin",
        }

    @classmethod
    def suggest_params(cls, trial, base=None, only=None, prefix=""):
        """Populate a parameterized instance with values from trial"""
        if only is None:
            only = cls.get_tunable()
        params = cls() if base is None else base
        pdict = params.param.params()
        if "log10_learning_rate" in only:
            softbounds = pdict["log10_learning_rate"].get_soft_bounds()
            params.log10_learning_rate = trial.suggest_uniform(
                prefix + "log10_learning_rate", *softbounds
            )
        if "num_epochs" in only:
            softbounds = pdict["num_epochs"].get_soft_bounds()
            params.num_epochs = trial.suggest_int(prefix + "num_epochs", *softbounds)
        if params.num_epochs is None:
            num_epochs = float("inf")
        else:
            num_epochs = params.num_epochs
        # if we sample patience and burnin so that their collective total
        # reaches or exceeds the number of epochs, they are effectively
        # disabled. Rather than allowing vast sums above the number of epochs,
        # we only allow the sum to reach the remaining epochs
        remaining_epochs = num_epochs
        if "early_stopping_patience" not in only:
            remaining_epochs -= params.early_stopping_patience
        if "early_stopping_burnin" not in only:
            remaining_epochs -= params.early_stopping_burnin
        remaining_epochs = max(0, remaining_epochs)
        if remaining_epochs and "early_stopping_threshold" in only:
            softbounds = pdict["early_stopping_threshold"].get_soft_bounds()
            params.early_stopping_threshold = trial.suggest_uniform(
                prefix + "early_stopping_threshold", *softbounds
            )
        if not params.early_stopping_threshold:
            remaining_epochs = 0
        if remaining_epochs and "early_stopping_patience" in only:
            softbounds = pdict["early_stopping_patience"].get_soft_bounds()
            softbounds = tuple(min(x, remaining_epochs) for x in softbounds)
            params.early_stopping_patience = trial.suggest_int(
                prefix + "early_stopping_patience", *softbounds
            )
            remaining_epochs -= params.early_stopping_patience
            assert remaining_epochs >= 0
        if remaining_epochs and "early_stopping_burnin" in only:
            softbounds = pdict["early_stopping_burnin"].get_soft_bounds()
            softbounds = tuple(min(x, remaining_epochs) for x in softbounds)
            params.early_stopping_burnin = trial.suggest_int(
                prefix + "early_stopping_burnin", *softbounds
            )
            remaining_epochs -= params.early_stopping_burnin
            assert remaining_epochs >= 0
        # we do the same thing, but for the learning rate scheduler
        remaining_epochs = num_epochs
        if "reduce_lr_patience" not in only:
            remaining_epochs -= params.reduce_lr_patience
        if "reduce_lr_burnin" not in only:
            remaining_epochs -= params.reduce_lr_burnin
        remaining_epochs = max(0, remaining_epochs)
        if remaining_epochs and "reduce_lr_threshold" in only:
            softbounds = pdict["reduce_lr_threshold"].get_soft_bounds()
            params.reduce_lr_threshold = trial.suggest_uniform(
                prefix + "reduce_lr_threshold", *softbounds
            )
        if not params.reduce_lr_threshold:
            remaining_epochs = 0
        if remaining_epochs and "reduce_lr_patience" in only:
            softbounds = pdict["reduce_lr_patience"].get_soft_bounds()
            softbounds = tuple(min(x, remaining_epochs) for x in softbounds)
            params.reduce_lr_patience = trial.suggest_int(
                prefix + "reduce_lr_patience", *softbounds
            )
            remaining_epochs -= params.reduce_lr_patience
        if remaining_epochs and "reduce_lr_burnin" in only:
            softbounds = pdict["reduce_lr_burnin"].get_soft_bounds()
            softbounds = tuple(min(x, remaining_epochs) for x in softbounds)
            params.reduce_lr_burnin = trial.suggest_int(
                prefix + "reduce_lr_burnin", *softbounds
            )
        if remaining_epochs and "reduce_lr_factor" in only:
            softbounds = pdict["reduce_lr_factor"].get_soft_bounds()
            params.reduce_lr_factor = trial.suggest_uniform(
                prefix + "reduce_lr_factor", *softbounds
            )
        if remaining_epochs and "reduce_lr_cooldown" in only:
            softbounds = pdict["reduce_lr_cooldown"].get_soft_bounds()
            params.reduce_lr_cooldown = trial.suggest_int(
                prefix + "reduce_lr_cooldown", *softbounds
            )
        return params


class TrainingStateController(object):
    """Controls the state of training a model

    This class is used to help both control and persist experiment information like the
    current epoch, the model parameters, and model error. It assumes that the values
    stored in `params` have not changed when resuming a run. It is also used to control
    learning rates and early stopping.

    Parameters
    ----------
    params
    state_csv_path
        A path to where training state information is stored. It stores in
        comma-separated-values format the following information. Note that stored values
        represent the state *after* updates due to epoch results, such as the learning
        rate. That way, an experiment can be resumed without worrying about updating the
        loaded results.

        1. "epoch": the epoch associated with this row of information
        2. "es_resume_cd": the number of epochs left before the early stopping criterion
           begins/resumes
        3. es_patience_cd: the number of epochs left that must pass without much
           improvement before training halts due to early stopping
        4. "rlr_resume_cd": the number of epochs left before the criterion for reducing
           the learning rate begins/resumes
        5. "rlr_patience_cd": the number of epochs left that must pass without much
           improvement before the learning rate is reduced
        6. "lr": the learning rate of the optimizer after any updates
        7. "train_met": mean training metric in exponent format. The metric is assumed
           to be lower is better
        8. "val_met": mean validation metric in exponent format. The metric is assumed
           to be lower is better
        9. Any additional entries added through :func:`add_entry`

        If unset, the history will not be stored/loaded.
    state_dir
        A path to a directory to store/load model and optimizer states. If unset, the
        information will not be stored/loaded.
    warn
        Whether to warn using :mod:`warnings` module when a format string does not
        contain the "epoch" field.
    reduce_op
        The op to combine metrics and other reducable ops in a distributed environment.
        See the note below for more details.

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

    Warnings
    --------
    Prior to `v0.4.0`, the cache of history was updated automatically (reading from
    `state_csv`) whenever :func:`get_last_epoch`, :func:`get_best_epoch`,
    :func:`add_entry`, or :func:`get_info` (when the info was missing from the cache)
    was called. Now, the cache is only updated automatically on initialization and with
    calls to :func:`add_entry`. The cache may still be updated manually via
    :func:`update_cache`. There was no good reason to continuously update the cache
    as any updates to the underlying file by other processes could ruin the control
    flow anyways.

    Notes
    -----
    :class:`TrainingStateController` has rudimentary support for distributed training
    via :class:`torch.nn.parallel.DistributedDataParallel`. Please read the `tutorial
    <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_ to understand the
    basics of the environment before continuing.

    Simple training loops involving a :class:`TrainingStateController`, like in the
    example above, should work with only the usual distributed boilerplate (spawning the
    pool, initializing process group, and wrapping the model with
    :class:`DistributedDataParallel`). The controller should be created and
    :func:`update_for_epoch` called in each worker.

    The only values which require coordinated over workers by default are the training
    and validation metrics; the rest -- early stopping, learning rate reduction, current
    epoch, and so on -- will stay in sync across workers. If a custom entry in the state
    history needs to be coordinated (i.e. it depends directly on the data seen over the
    epoch, not on the training or validation metrics), the `reduce` flag of
    :func:`add_entry` can be set to :obj:`True` and that value will likewise be
    coordinated on the call to :func:`update_for_epoch`. `reduce_op` determines how the
    relevant values are coordinated. An average is taken by default by first dividing
    each copy of the value by the world size and then summing the copies together via
    :obj:`torch.distributed.ReduceOp.SUM`. See :class:`torch.distributed.ReduceOp` for
    other options.
    """

    def __init__(
        self,
        params: TrainingStateParams,
        state_csv_path: Optional[str] = None,
        state_dir: Optional[str] = None,
        warn: bool = True,
        reduce_op: Optional[torch.distributed.ReduceOp] = None,
    ):
        super(TrainingStateController, self).__init__()
        self.params = params
        if warn:
            for s in (self.params.saved_model_fmt, self.params.saved_optimizer_fmt):
                if not any(x[1] == "epoch" for x in Formatter().parse(s)):
                    warnings.warn(
                        'State format string "{}" does not contain "epoch" '
                        "field, so is possibly not unique. In this case, only "
                        "the state of the last epoch will persist. To "
                        "suppress this warning, set warn=False".format(s)
                    )
        self.state_csv_path = state_csv_path
        self.state_dir = state_dir
        self.cache_hist = dict()
        self.user_entry_types = OrderedDict()
        self.fmt_dict = dict()
        self.reduce_op = reduce_op
        if params.num_epochs is None:
            self.fmt_dict["epoch"] = "{:010d}"
        else:
            self.fmt_dict["epoch"] = "{{:0{}d}}".format(
                int(math.log10(params.num_epochs)) + 1
            )
        self.fmt_dict["es_resume_cd"] = "{{:0{}d}}".format(
            int(math.log10(max(params.early_stopping_burnin, 1))) + 1
        )
        self.fmt_dict["es_patience_cd"] = "{{:0{}d}}".format(
            int(math.log10(max(params.early_stopping_patience, 1))) + 1
        )
        self.fmt_dict["rlr_resume_cd"] = "{{:0{}d}}".format(
            int(
                math.log10(
                    max(
                        params.reduce_lr_cooldown,
                        params.reduce_lr_burnin,
                        1,
                    )
                )
            )
            + 1
        )
        self.fmt_dict["rlr_patience_cd"] = "{{:0{}d}}".format(
            int(math.log10(max(params.reduce_lr_patience, 1))) + 1
        )
        self.fmt_dict["lr"] = "{{:.{}e}}".format(self.SCIENTIFIC_PRECISION - 1)
        self.fmt_dict["train_met"] = self.fmt_dict["lr"]
        self.fmt_dict["val_met"] = self.fmt_dict["lr"]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
        else:
            self._rank = -1
        self.reduced_entries = {"train_met", "val_met"}
        self.update_cache()

    """The number of digits in significand of scientific notation

    Controls how many digits are saved when writing metrics and learning rate
    to disk (i.e. the ``x`` in ``x * 10^y``). Used when generating the format
    strings in ``self.fmt_dict`` on initialization
    """
    SCIENTIFIC_PRECISION = 5

    # XXX(sdrobert): barriers are generally performed on both entry and exit of reads,
    # i.e. when the cache is updated by reading the history or a model/optimizer loaded.
    # Modification of disk isn't (i.e. writing history or adding/removing state dicts)
    # since those don't affect the state of the controller.
    def _barrier(self) -> None:
        if self._rank >= 0:
            torch.distributed.barrier()

    def update_cache(self) -> None:
        """Update the cache with history stored in state_csv_path"""
        # add a dummy entry for epoch "0" just to make logic easier. We
        # won't save it
        self.cache_hist[0] = {
            "epoch": 0,
            "es_resume_cd": self.params.early_stopping_burnin,
            "es_patience_cd": self.params.early_stopping_patience,
            "rlr_resume_cd": self.params.reduce_lr_burnin,
            "rlr_patience_cd": self.params.reduce_lr_patience,
            "train_met": float("inf"),
            "val_met": float("inf"),
            "lr": None,
        }
        self.cache_hist[0].update(dict((key, None) for key in self.user_entry_types))
        if self.params.log10_learning_rate is not None:
            self.cache_hist[0]["lr"] = 10**self.params.log10_learning_rate
        if self.state_csv_path is None:
            return
        self._barrier()
        if not os.path.exists(self.state_csv_path):
            self._barrier()
            return
        with open(self.state_csv_path) as f:
            reader = DictReader(f)
            for row in reader:
                epoch = int(row["epoch"])
                self.cache_hist[epoch] = {
                    "epoch": epoch,
                    "es_resume_cd": int(row["es_resume_cd"]),
                    "es_patience_cd": int(row["es_patience_cd"]),
                    "rlr_resume_cd": int(row["rlr_resume_cd"]),
                    "rlr_patience_cd": int(row["rlr_patience_cd"]),
                    "lr": float(row["lr"]),
                    "train_met": float(row["train_met"]),
                    "val_met": float(row["val_met"]),
                }
                for name, type_ in list(self.user_entry_types.items()):
                    self.cache_hist[epoch][name] = type_(row[name])
        self._barrier()

    def add_entry(
        self, name: str, typ: type = str, fmt: str = "{}", reduce: bool = False
    ) -> None:
        """Add an entry to to be stored and retrieved at every epoch

        This method is useful when training loops need specialized, persistent
        information on every epoch. Prior to the first time any information is saved via
        :func:`update_for_epoch`, this method can be called with an entry `name` and
        optional `typ`. The user is then expected to provide a keyword argument with
        that `name` every time :func:`update_for_epoch` is called. The values of those
        entries can be retrieved via :func:`get_info`, cast to `typ`, for any saved
        epoch

        Parameters
        ----------
        name
            The name/key of the entry.
        typ
            Should be a type that is serialized from a string via ``typ(str_obj)`` and
            serialized to a string via ``fmt.format(obj)``.
        fmt
            The format string used to serialize the objects into strings.
        reduce
            If :obj:`True` and in a distributed environment, the value will be
            synchronized across workers via a reduction op on each call to
            :func:`update_for_epoch` see the notes in the class documentation for
            more information.

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
        :func:`add_entry` must be called prior to :func:`update_for_epoch` or
        :func:`save_info_to_hist`, or it may corrupt the experiment history. However,
        the controller can safely ignore additional entries when loading history from a
        CSV. Thus, there is no need to call :func:`add_entry` if no new training is to
        be done (unless those entries are needed outside of training).
        """
        if name in {
            "epoch",
            "es_resume_cd",
            "es_patience_cd",
            "rlr_resume_cd",
            "rlr_patience_cd",
            "lr",
            "train_met",
            "val_met",
        }:
            raise ValueError('"{}" is a reserved entry name'.format(name))
        if not isinstance(typ, type):
            raise ValueError("typ ({}) must be a type".format(typ))
        self.user_entry_types[name] = typ
        self.fmt_dict[name] = fmt
        if reduce:
            self.reduced_entries.add(name)
        self.update_cache()

    def get_last_epoch(self) -> int:
        """Return the last finished epoch from training, or 0 if no history"""
        return max(self.cache_hist)

    def get_best_epoch(self, train_met: bool = False) -> int:
        """Get the epoch that has lead to the best validation metric val so far

        The "best" is the lowest recorded validation metric. In the case of ties, the
        earlier epoch is chosen.

        Parameters
        ----------
        train_met
            If :obj:`True` look for the best training metric value instead

        Returns
        -------
        epoch : int
            The corresponding 'best' epoch, or :obj:`0` if no epochs have run

        Notes
        -----
        Negligible differences between epochs are determined by
        :obj:`TrainingStateController.METRIC_PRECISION`, which is relative to the
        metrics base 10. This is in contrast to early stopping criteria and learning
        rate annealing, whose thresholds are absolute.
        """
        ent = "train_met" if train_met else "val_met"
        fmt = self.fmt_dict[ent]
        min_epoch = 0
        min_met = self.cache_hist[0][ent]
        min_met = float(fmt.format(min_met))
        for info in list(self.cache_hist.values()):
            cur = float(fmt.format(info[ent]))
            if cur < min_met:
                min_epoch = info["epoch"]
                min_met = cur
        return min_epoch

    def load_model_for_epoch(
        self, model: torch.nn.Module, epoch: Optional[int] = None, strict: bool = True
    ) -> None:
        """Load up just the model, or initialize it

        Parameters
        ----------
        model
            Model state will be loaded into this.
        epoch
            The epoch from which the states should be loaded. We look for the
            appropriately named files in ``self.state_dir``. If `epoch` is :obj:`None`,
            the best epoch in recorded history will be loaded. If it's 0, the model is
            initialized with states from the beginning of the experiment.
        strict
            Whether to strictly enforce that the keys in ``model.state_dict()`` match
            those that were saved.
        """
        self._barrier()
        if epoch is None:
            epoch = self.get_best_epoch()
        if not epoch:
            self._init_seed_and_model(model)
        elif self.state_dir is not None:
            model_pth = self.get_model_path_with_info(self.get_info(epoch))
            model_state_dict = torch.load(model_pth, map_location="cpu")
            model.load_state_dict(model_state_dict, strict=strict)
        else:
            warnings.warn(
                "Unable to load model for epoch {}. No state directory!"
                "".format(epoch)
            )
        self._barrier()

    def _init_seed_and_model(self, model):
        if self.params.seed is not None:
            torch.manual_seed(self.params.seed)
        if hasattr(model, "reset_parameters"):
            model.reset_parameters()
        elif self._rank >= 0 and hasattr(model, "module"):
            if hasattr(model.module, "reset_parameters"):
                if self.params.seed is not None:
                    model.module.reset_parameters()
                else:
                    warnings.warn(
                        "Not resetting parameters in distributed mode without seed"
                    )
        else:
            warnings.warn(
                "model has no reset_parameters() method, so cannot "
                "reset parameters for epoch 0",
            )

    def load_model_and_optimizer_for_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: Optional[int] = None,
        strict: bool = True,
    ) -> None:
        """Load up model and optimizer states, or initialize them

        Parameters
        ----------
        model
            Model state will be loaded into this.
        optimizer
            Optimizer state will be loaded into this.
        epoch
            The epoch from which the states should be loaded. We look for the
            appropriately named files in ``self.state_dir``. If `epoch` is unset, the
            last epoch in recorded history will be loaded. If it's 0, the model and
            optimizer are initialized with states for the beginning of the experiment.
        strict
            Whether to strictly enforce that the keys in ``model.state_dict()``
            match those that were saved.
        """
        self._barrier()
        if epoch is None:
            epoch = self.get_last_epoch()
        if not epoch:
            self._init_seed_and_model(model)
            if self.params.log10_learning_rate is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 10**self.params.log10_learning_rate
            # there is no public API for resetting the state dictionary, so
            # we create a new instance as best as possible and copy the state
            # over from there. Note that settings like weight decay are already
            # part of the parameter group, so we don't need to worry about
            # initializing with them.
            brand_new_optimizer = type(optimizer)(optimizer.param_groups)
            optimizer.load_state_dict(brand_new_optimizer.state_dict())
            del brand_new_optimizer
        elif self.state_dir is not None:
            info = self.get_info(epoch)
            model_pth = self.get_model_path_with_info(info)
            optim_pth = self.get_optimizer_path_with_info(info)
            model_state_dict = torch.load(model_pth, map_location="cpu")
            model.load_state_dict(model_state_dict, strict=strict)
            optimizer_state_dict = torch.load(optim_pth, map_location="cpu")
            optimizer.load_state_dict(optimizer_state_dict)
        else:
            warnings.warn(
                f"Unable to load model and optimizer for epoch {epoch}. No state_dir!"
            )
        self._barrier()

    def _clean_up_files(self, *pths):
        if self._rank <= 0:
            for pth in pths:
                if not os.path.exists(pth):
                    continue
                try:
                    os.remove(pth)
                except OSError:
                    warnings.warn(f"Failed to delete file '{pth}'")

    def delete_model_and_optimizer_for_epoch(self, epoch: int) -> None:
        """Delete state dicts for model and epoch off of disk, if they exist

        This method does nothing if the epoch records or the files do not exist.

        Parameters
        ----------
        epoch
        """
        if self.state_dir is None:
            return
        info = self.get_info(epoch, None)
        if info is None:
            return
        model_pth = self.get_model_path_with_info(info)
        optim_pth = self.get_optimizer_path_with_info(info)
        self._clean_up_files(model_pth, optim_pth)

    def get_info(self, epoch: int, *default) -> dict:
        """Get history entries for a specific epoch

        If there's an entry present for `epoch`, return it. The value is a dictionary
        with the keys "epoch", "es_resume_cd", "es_patience_cd", "rlr_resume_cd",
        "rlr_patience_cd", "lr", "train_met", and "val_met", as well as any additional
        entries specified through :func:`add_entry`.

        If there's no entry for `epoch`, and no additional arguments were passed to this
        method, it raises a :class:`KeyError`. If an additional argument was passed to
        this method, return it.
        """
        return self.cache_hist.get(epoch, *default)

    def __getitem__(self, epoch: int) -> dict:
        return self.get_info(epoch)

    def get_model_path_with_info(self, info: dict) -> str:
        return os.path.join(self.state_dir, self.params.saved_model_fmt.format(**info))

    def get_optimizer_path_with_info(self, info: dict) -> str:
        return os.path.join(
            self.state_dir, self.params.saved_optimizer_fmt.format(**info)
        )

    def save_model_and_optimizer_with_info(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, info: dict
    ) -> None:
        """Save model and optimizer state dictionaries to files given epoch info

        This is called automatically during :func:`update_for_epoch`. Does not save if
        there is no directory to save to (i.e. ``self.state_dir is None``). Format
        strings from ``self.params`` are formatted with the values from `info` to
        construct the base names of each file

        Parameters
        ----------
        model
            The model whose state dictionary will be saved.
        optimizer
            The optimizer whose state dictionary will be saved.
        info
            The history dictionary. Entries can be used in the state dict's path's
            format strings.
        """
        if self.state_dir is None:
            return
        if self._rank <= 0:
            # defensive write which makes sure we have enough space on the drive before
            # overwriting anything. Create in new file, then move into position
            write_pairs = (
                (model.state_dict(), self.get_model_path_with_info(info)),
                (optimizer.state_dict(), self.get_optimizer_path_with_info(info)),
            )
            replaces = []
            for obj, path in write_pairs:
                dir_ = os.path.dirname(path)
                os.makedirs(dir_, exist_ok=True)
                with tempfile.NamedTemporaryFile("wb", dir=dir_, delete=False) as f:
                    torch.save(obj, f)
                    replaces.append((f.name, path))
            for src, dst in replaces:
                os.replace(src, dst)
            del write_pairs, replaces

    def save_info_to_hist(self, info: dict):
        """Append history entries to the history csv

        This is called automatically during :func:`update_for_epoch`. Does not save if
        there is no file to save to (i.e. ``self.state_csv_path is None``). Values are
        appended to the end of the csv file - no checking is performed for mantaining a
        valid history.

        Parameters
        ----------
        info
        """
        epoch = info["epoch"]
        self.cache_hist[epoch] = info
        if self.state_csv_path is None:
            return
        if self._rank <= 0:
            names = [
                "epoch",
                "es_resume_cd",
                "es_patience_cd",
                "rlr_resume_cd",
                "rlr_patience_cd",
                "lr",
                "train_met",
                "val_met",
            ]
            names += list(self.user_entry_types)
            write_header = not os.path.exists(self.state_csv_path)
            with open(self.state_csv_path, "a") as f:
                wr = writer(f)
                if write_header:
                    wr.writerow(names)
                wr.writerow([self.fmt_dict[k].format(info[k]) for k in names])

    def continue_training(self, epoch: Optional[int] = None) -> bool:
        """Return a boolean on whether to continue training

        Useful when resuming training. Will check the training history at the
        target `epoch` and determine whether training should continue from that
        point, based on the total number of epochs and the early stopping
        criterion.

        Parameters
        ----------
        epoch
            The epoch to check the history of. If unset, the last epoch will be
            inferred.

        Returns
        -------
        cont : bool
            :obj:`True` if training should continue
        """
        if epoch is None:
            epoch = self.get_last_epoch()
        info = self.get_info(epoch)
        if not self.params.num_epochs:
            cont = True
        else:
            cont = epoch < self.params.num_epochs
        if self.params.early_stopping_threshold and not info["es_patience_cd"]:
            cont = False
        return cont

    def update_for_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_met: float,
        val_met: float,
        epoch: Optional[int] = None,
        best_is_train: bool = False,
        **kwargs,
    ) -> bool:
        """Update history, model, and optimizer after latest epoch results

        Parameters
        ----------
        model
            The model after the epoch that just finished.
        optimizer
            The optimizer after the epoch that just finished.
        train_met
            Mean value of metric on training set for epoch.
        val_met
            Mean value of metric on validation set for epoch.
        epoch
            The epoch that just finished. If unset, it is inferred to be one after the
            last epoch in the history.
        best_is_train
            Whether to just the best model in record by training set (:obj:`False` is
            validation)
        **kwargs
            Additional keyword arguments can be used to specify the values of entries
            specified via :func:`add_entry`.

        Returns
        -------
        cont : bool
            Whether to continue training. This can be set to :obj:`False` either by
            hitting the max number of epochs or by early stopping.
        """
        if self._rank >= 0:
            kwargs["train_met"] = train_met
            kwargs["val_met"] = val_met
            handles = []
            reduced_entries = sorted(self.reduced_entries)
            W = torch.distributed.get_world_size()
            to_gpu = torch.distributed.get_backend() == torch.distributed.Backend.NCCL
            for name in reduced_entries:
                kwargs[name] = torch.as_tensor(kwargs[name])
                if to_gpu and kwargs[name].device.type != "cuda":
                    kwargs[name] = kwargs[name].cuda()
                reduce_op = self.reduce_op
                if reduce_op is None:
                    kwargs[name] = kwargs[name] / W
                    reduce_op = torch.distributed.ReduceOp.SUM
                handles.append(
                    torch.distributed.all_reduce(kwargs[name], reduce_op, async_op=True)
                )
            for handle in handles:
                handle.wait()
            for name in reduced_entries:
                kwargs[name] = kwargs[name].item()
            train_met = kwargs.pop("train_met")
            val_met = kwargs.pop("val_met")
        if epoch is None:
            epoch = self.get_last_epoch() + 1
        last_best = self.get_best_epoch(best_is_train)
        if not self.params.num_epochs:
            cont = True
        else:
            cont = epoch < self.params.num_epochs
            if epoch > self.params.num_epochs:
                warnings.warn("Training is continuing, despite passing num_epochs")
        info = dict(self.get_info(epoch - 1, None))
        if info is None:
            raise ValueError(
                f"no entry for the previous epoch {epoch}, so unable to update"
            )
        for key, value in list(kwargs.items()):
            if key not in self.user_entry_types:
                raise TypeError(
                    "update_for_epoch() got an unexpected keyword argument "
                    f"'{key}' (did you forget to add_entry()?)"
                )
            elif not isinstance(value, self.user_entry_types[key]):
                raise ValueError(
                    'keyword argument "{}" value is not of type {}'
                    "".format(key, self.user_entry_types[key])
                )
            info[key] = value
        remaining_user_entries = set(self.user_entry_types) - set(kwargs)
        if remaining_user_entries:
            raise TypeError(
                "The following keyword arguments were not provided as keyword "
                "arguments but were specified via add_entry(): {}"
                "".format(sorted(remaining_user_entries))
            )
        if info["lr"] is None:
            # can only happen during the first epoch. We don't know the
            # optimizer defaults, so we get them now
            info["lr"] = optimizer.defaults["lr"]
        es_epoch = (
            epoch - self.params.early_stopping_patience + info["es_patience_cd"] - 1
        )
        es_info = self.get_info(es_epoch)
        if info["es_resume_cd"]:
            info["es_resume_cd"] -= 1
        elif (
            max(es_info["val_met"] - val_met, 0) < self.params.early_stopping_threshold
        ):
            info["es_patience_cd"] -= 1
            if info["es_patience_cd"] < 0:
                warnings.warn(
                    "Early stopping criterion was already met, but training "
                    "has continued"
                )
                info["es_patience_cd"] = 0
        else:
            info["es_patience_cd"] = self.params.early_stopping_patience
        # we do it this way in case someone continues training after early stopping has
        # been reached
        if self.params.early_stopping_threshold and not info["es_patience_cd"]:
            cont = False
        rlr_epoch = epoch - self.params.reduce_lr_patience + info["rlr_patience_cd"] - 1
        rlr_info = self.get_info(rlr_epoch)
        if info["rlr_resume_cd"]:
            info["rlr_resume_cd"] -= 1
        elif max(rlr_info["val_met"] - val_met, 0) < self.params.reduce_lr_threshold:
            info["rlr_patience_cd"] -= 1
            if not info["rlr_patience_cd"]:
                old_lr = info["lr"]
                new_lr = old_lr * self.params.reduce_lr_factor
                rlr_epsilon = 10**self.params.reduce_lr_log10_epsilon
                if old_lr - new_lr > rlr_epsilon:
                    info["lr"] = new_lr
                    for param_group in optimizer.param_groups:
                        # just assume that the user knows what's what if
                        # the optimizer's lr doesn't match the old one
                        param_group["lr"] = new_lr
                info["rlr_resume_cd"] = self.params.reduce_lr_cooldown
                info["rlr_patience_cd"] = self.params.reduce_lr_patience
        else:
            info["rlr_patience_cd"] = self.params.reduce_lr_patience
        info["epoch"] = epoch
        info["val_met"] = val_met
        info["train_met"] = train_met
        if self.state_dir is not None:
            model_pth = self.get_model_path_with_info(info)
            optim_pth = self.get_optimizer_path_with_info(info)
            wrote_info_warn = (
                f"Saving epoch {epoch} model and optimizer failed but write to "
                f"'{self.state_csv_path}' succeeded. You should delete that entry."
            )
            if self.params.keep_last_and_best_only:
                self.cache_hist[epoch] = info
                cur_best = self.get_best_epoch(best_is_train)

                if cur_best != epoch:
                    best_info = self.get_info(cur_best)
                    best_model_pth = self.get_model_path_with_info(best_info)
                    best_optim_pth = self.get_optimizer_path_with_info(best_info)
                    if model_pth == best_model_pth:
                        raise ValueError(
                            f"New model checkpoint '{model_pth}' would overwrite best "
                            "model checkpoint, so we raised instead. Either change the "
                            "model format string or set keep_last_and_best_only to "
                            "False"
                        )
                    elif optim_pth == best_optim_pth:
                        raise ValueError(
                            f"New optimizer checkpoint '{optim_pth}' would overwrite "
                            "best optimizer checkpoint, so we raised instead. Either "
                            "change the optimizer format string or set "
                            "keep_last_and_best_only to False"
                        )
                if cur_best == epoch - 1:
                    # no conflict. Keep everything. Save model and optimizer first so
                    # that user doesn't have to muck with history
                    self.save_model_and_optimizer_with_info(model, optimizer, info)
                    self.save_info_to_hist(info)
                else:
                    last_info = self.get_info(epoch - 1)
                    last_model_pth = self.get_model_path_with_info(last_info)
                    last_optim_pth = self.get_optimizer_path_with_info(last_info)
                    last_best_info = self.get_info(last_best)
                    last_best_model_pth = self.get_model_path_with_info(last_best_info)
                    last_best_optim_pth = self.get_optimizer_path_with_info(
                        last_best_info
                    )
                    save_info_first = {model_pth, optim_pth} & {
                        last_model_pth,
                        last_best_model_pth,
                        last_optim_pth,
                        last_best_optim_pth,
                    }
                    if save_info_first:
                        self.save_info_to_hist(info)
                    try:
                        self.save_model_and_optimizer_with_info(model, optimizer, info)
                    except:
                        if self._rank <= 0 and save_info_first and self.state_csv_path:
                            warnings.warn(wrote_info_warn)
                        raise
                    if not save_info_first:
                        self.save_info_to_hist(info)

                    clean_up = {last_model_pth, last_optim_pth}
                    if last_best != cur_best:
                        clean_up |= {last_best_model_pth, last_best_optim_pth}
                    clean_up -= {model_pth, optim_pth}
                    self._clean_up_files(*tuple(clean_up))
            else:
                save_info_first = os.path.exists(model_pth) or os.path.exists(optim_pth)
                if save_info_first:
                    self.save_info_to_hist(info)
                try:
                    self.save_model_and_optimizer_with_info(model, optimizer, info)
                except:
                    if self._rank <= 0 and save_info_first and self.state_csv_path:
                        warnings.warn(wrote_info_warn)
                    raise
                if not save_info_first:
                    self.save_info_to_hist(info)
        else:
            self.save_info_to_hist(info)
        return cont
