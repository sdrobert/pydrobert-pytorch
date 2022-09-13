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

import os
import argparse

from typing import Any, Dict, Optional
from typing_extensions import Literal

import torch
import param
import pytorch_lightning as pl

from ._datasets import SpectDataSet, SpectDataParams
from ._dataloaders import (
    DynamicLengthDataLoaderParams,
    SpectDataLoader,
    SpectDataLoaderParams,
)


try:
    import pydrobert.param.argparse as pargparse
except ImportError as _pargparse_error:
    pargparse = None


class LitSpectDataModuleParams(param.Parameterized):

    prefer_split = param.Boolean(True)

    common = param.ClassSelector(SpectDataLoaderParams, instantiate=False)
    train = param.ClassSelector(SpectDataLoaderParams, instantiate=False)
    val = param.ClassSelector(SpectDataLoaderParams, instantiate=False)
    test = param.ClassSelector(SpectDataLoaderParams, instantiate=False)
    predict = param.ClassSelector(SpectDataLoaderParams, instantiate=False)

    train_dir = param.Foldername(None)
    val_dir = param.Foldername(None)
    test_dir = param.Foldername(None)
    predict_dir = param.Foldername(None)

    info_path = param.Filename(None)

    mvn_path = param.Filename(None)
    warn_on_missing = param.Boolean(True)
    sort_batch = param.Boolean(False)
    suppress_alis = param.Boolean(True)
    tokens_only = param.Boolean(True)

    @property
    @param.depends("train", "val", "test", "predict")
    def loader_params_are_split(self) -> bool:
        return any(
            x is not None for x in (self.train, self.val, self.test, self.predict)
        )

    @property
    @param.depends("common")
    def loader_params_are_merged(self) -> bool:
        return self.common is not None

    @param.depends("common", "train", "val", "test", "predict", watch=True)
    def check_overlap(self):
        if self.loader_params_are_merged and self.loader_params_are_split:
            raise ValueError(
                "Cannot simultateously initialize 'common' and any of "
                "'train', 'val', 'test', or 'predict'"
            )

    @property
    @param.depends("common", "train", "val", "test", "predict")
    def train_params(self) -> SpectDataLoaderParams:
        if self._use_split():
            return self._get_parameterized_and_init("train")
        else:
            return self._get_parameterized_and_init("common")

    @property
    @param.depends("common", "train", "val", "test", "predict")
    def val_params(self) -> SpectDataLoaderParams:
        if self._use_split():
            return self._get_parameterized_and_init("val")
        else:
            return self._get_parameterized_and_init("common")

    @property
    @param.depends("common", "train", "val", "test", "predict")
    def test_params(self) -> SpectDataLoaderParams:
        if self._use_split():
            return self._get_parameterized_and_init("test")
        else:
            return self._get_parameterized_and_init("common")

    @property
    @param.depends("common", "train", "val", "test", "predict")
    def predict_params(self) -> SpectDataLoaderParams:
        if self._use_split():
            return self.predict
        else:
            return self._get_parameterized_and_init("common")

    def _get_parameterized_and_init(self, name: str):
        val = getattr(self, name)
        if val is None:
            val = self.param[name].class_(name=name)
            setattr(self, name, val)
        return val

    def _use_split(self):
        return self.loader_params_are_split or (
            self.prefer_split and not self.loader_params_are_merged
        )


def readable_dir(path: str) -> str:
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"'{path}' is not a directory")
    return path


def readable_file(path: str) -> str:
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"'{path}' is not a file")
    return path


def natural(v: str) -> int:
    try:
        v = int(v)
    except ValueError as e:
        raise argparse.ArgumentTypeError from e
    if v < 1:
        raise argparse.ArgumentTypeError(f"'{v}' is not a natural number")
    return v


class LitSpectDataModule(pl.LightningDataModule):

    train_set: Optional[SpectDataSet]
    predict_set: Optional[SpectDataSet]
    test_set: Optional[SpectDataSet]
    val_set: Optional[SpectDataSet]

    _num_workers: Optional[int]
    _pin_memory: Optional[bool]
    _num_filts: Optional[int]
    _max_ref_class: Optional[int]
    _max_ali_class: Optional[int]

    def __init__(
        self,
        params: LitSpectDataModuleParams,
        batch_first: bool = False,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        on_uneven_distributed: Literal["raise", "uneven", "ignore"] = "raise",
    ) -> None:
        super().__init__()

        # The member is a "local copy" of the hyperparameter. The default changes
        # depending on the node we're running on, so we don't want to propagate them
        # back to the module state
        self._num_workers = num_workers
        self._pin_memory = pin_memory

        self.train_set = self.predict_set = self.test_set = self.val_set = None
        self._num_filts = self._max_ref_class = self._max_ali_class = None
        self.save_hyperparameters()

    @property
    def params(self) -> LitSpectDataModuleParams:
        """LitSpectDataModuleParams : the data module parameters"""
        return self.hparams.params

    @property
    def batch_first(self) -> bool:
        """bool : whether dataloaders present data with the batch dimension first"""
        return self.hparams.batch_first

    @property
    def num_workers(self) -> Optional[int]:
        return self._num_workers

    @property
    def pin_memory(self) -> Optional[bool]:
        return self._pin_memory

    @property
    def on_uneven_distributed(self) -> Literal["raise", "uneven", "ignore"]:
        """str : how to handle uneven batch sizes in the distributed environment
        
        - 'raise': throw a :class:`ValueError`
        - 'uneven': allow some processes to make fewer or smaller batches.
        - 'ignore': ignore the distributed environment. Each process yields all batches.
        """
        return self.hparams.on_uneven_distributed

    @property
    def vocab_size(self) -> Optional[int]:
        """int : vocabulary size
        
        Alias of ``max_ref_class + 1``. Determined in :func:`prepare_data` if
        `params.info_path` is not :obj:`None`.
        """
        return None if self._max_ref_class is None else self._max_ref_class + 1

    @property
    def feat_size(self) -> Optional[int]:
        """int : feature vector size
        
        Alias of `num_filts`.  Determined in :func:`prepare_data` if `params.info_path`
        is not :obj:`None`.
        """
        return self._num_filts

    def max_ref_class(self) -> Optional[int]:
        """int : the maximum token id in the ref/ subdirectory (usually of training)
        
        Determined in :func:`prepare_data` if `params.info_path` is not :obj:`None`.
        """
        return self._max_ref_class

    def max_ali_class(self) -> Optional[int]:
        """int: the maximum token id in the ali/ subdirectory (usually of training)
        
        Determined in :func:`prepare_data` if `params.info_path` is not :obj:`None`.
        """
        return self._max_ali_class

    @property
    def num_filts(self) -> Optional[int]:
        """int : size of the last dimension of tensors in feat/
        
        Determined in :func:`prepare_data` if `params.info_path` is not :obj:`None`.
        """
        return self._num_filts

    def _get_mvn(self):
        if self.params.mvn_path is not None:
            dict_ = torch.load(self.params.mvn_path, "cpu")
            return dict_["mean"], dict_["std"]
        else:
            return None, None

    def _construct_dataset(
        self,
        stage: str,
        path: Optional[str],
        params: SpectDataParams,
        feat_mean: Optional[torch.Tensor],
        feat_std: Optional[torch.Tensor],
        **kwargs_,
    ):
        if path is None:
            raise ValueError(
                f"Cannot initialize datast for stage '{stage}': no data directory "
                "specified"
            )

        kwargs = {
            "warn_on_missing": self.params.warn_on_missing,
            "params": params,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "suppress_alis": self.params.suppress_alis,
            "tokens_only": self.params.tokens_only,
        }
        kwargs.update(kwargs_)

        return SpectDataSet(path, **kwargs)

    def prepare_data(self):

        if self.params.info_path is not None:
            with open(self.params.info_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    key, value = line.split()
                    value = int(value)
                    if key == "num_filts":
                        self._num_filts = value
                    elif key == "max_ali_class" and value != -1:
                        self._max_ali_class = value
                    elif key == "max_ref_class" and value != -1:
                        self._max_ref_class = value

    def setup(self, stage: Optional[str] = None):

        if self._num_workers is None:
            # FIXME(sdrobert): DDP and such
            self._num_workers = torch.multiprocessing.cpu_count()

        if self._pin_memory is None:
            if self.trainer is not None:
                self._pin_memory = isinstance(
                    self.trainer.accelerator,
                    pl.accelerators.accelerator.CUDAAccelerator,
                )
            else:
                self._pin_memory = True

        feat_mean, feat_std = self._get_mvn()

        if stage in {"fit", None}:
            self.train_set = self._construct_dataset(
                "fit",
                self.params.train_dir,
                self.params.train_params,
                feat_mean,
                feat_std,
            )
            if self.params.val_dir is not None:
                self.val_set = self._construct_dataset(
                    "fit",
                    self.params.val_dir,
                    self.params.val_params,
                    feat_mean,
                    feat_std,
                )

        if stage in {"test", None}:
            self.test_set = self._construct_dataset(
                "test",
                self.params.test_dir,
                self.params.test_params,
                feat_mean,
                feat_std,
            )

        if stage in {"predict", None}:
            if self.params.predict_dir is None:
                pred_dir = self.params.test_dir
            else:
                pred_dir = self.params.predict_dir
            if self.params.predict_params is None:
                pred_params = self.params.test_params
            else:
                pred_params = self.params.predict_params
            self.predict_set = self._construct_dataset(
                "predict",
                pred_dir,
                pred_params,
                feat_mean,
                feat_std,
                suppress_uttids=False,
            )

    def _construct_dataloader(
        self, ds: SpectDataSet, params: DynamicLengthDataLoaderParams, shuffle: bool,
    ) -> SpectDataLoader:

        return SpectDataLoader(
            ds,
            params,
            shuffle=shuffle,
            batch_first=self.batch_first,
            init_epoch=0 if self.trainer is None else self.trainer.current_epoch,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            on_uneven_distributed=self.on_uneven_distributed,
        )

    def train_dataloader(self):
        return self._construct_dataloader(
            self.train_set, self.params.train_params, True
        )

    def val_dataloader(self):
        return self._construct_dataloader(self.val_set, self.params.val_params, False)

    def test_dataloader(self):
        return self._construct_dataloader(self.test_set, self.params.test_params, False)

    def predict_dataloader(self):
        return self._construct_dataloader(
            self.predict_set, self.params.predict_params, False
        )

    @staticmethod
    def add_argparse_args(
        parser: argparse.ArgumentParser, include_overloads: bool = True
    ):
        if pargparse is None:
            raise _pargparse_error
        grp = pargparse.add_deserialization_group_to_parser(
            parser, LitSpectDataModuleParams, "data_params", reckless=True
        )

        if include_overloads:
            grp.add_argument("--train-dir", type=readable_dir, default=None)
            grp.add_argument("--val-dir", type=readable_dir, default=None)
            grp.add_argument("--test-dir", type=readable_dir, default=None)
            grp.add_argument("--predict-dir", type=readable_dir, default=None)
            grp.add_argument("--mvn-path", type=readable_file, default=None)
            grp.add_argument("--info-file", type=readable_file, default=None)

    @classmethod
    def from_argparse_args(
        cls,
        namespace: argparse.Namespace,
        batch_first: bool = False,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        on_uneven_distributed: Literal["raise", "uneven", "ignore"] = "raise",
        **overloads,
    ):
        data_params: LitSpectDataModuleParams = namespace.data_params

        for overload in (
            "train_dir",
            "val_dir",
            "test_dir",
            "predict_dir",
            "mvn_path",
            "info_file",
        ):
            val = getattr(namespace, overload, None)
            if val is not None:
                overloads[overload] = val

        data_params.update(**overloads)

        return cls(
            data_params, batch_first, num_workers, pin_memory, on_uneven_distributed
        )

    # def state_dict(self) -> Dict[str, Any]:
    #     return {
    #         "train_set": self.train_set,
    #         "predict_set": self.predict_set,
    #         "test_set": self.test_set,
    #         "val_set": self.val_set,
    #     }

    # def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    #     self.train_set = state_dict["train_set"]
    #     self.predict_set = state_dict["predict_set"]
    #     self.test_set = state_dict["test_set"]
    #     self.val_set = state_dict["val_set"]

