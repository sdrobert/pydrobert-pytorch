# Copyright 2023 Sean Robertson

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
import abc

from pathlib import PurePath
from typing import Dict, Optional, TypeVar, Generic, Type, Union
from typing_extensions import Literal, get_args

import torch
import param
import pytorch_lightning as pl
import pydrobert.param.abc as pabc
import pydrobert.param.argparse as pargparse

from . import argcheck
from ._datasets import SpectDataSet
from ._dataloaders import (
    SpectDataLoader,
    SpectDataLoaderParams,
    LangDataLoaderParams,
)


StrPath = Union[str, os.PathLike]
FileType = Literal["dir", "file"]
OnUnevenDistributed = Literal["raise", "uneven", "ignore"]
P = TypeVar("P", bound=param.Parameterized)


class PosixPath(param.Parameter):
    """POSIX-style path parameter

    Parameters
    ----------
    default
        Default value
    always_exists
        If :obj:`True`, setting to 
    """

    __slots__ = "always_exists", "type"

    always_exists: bool
    type: Optional[FileType]

    def __init__(
        self,
        default: Optional[StrPath] = None,
        always_exists: bool = True,
        type: Optional[FileType] = None,
        **params,
    ):
        default = argcheck.is_str(default, "default", True)
        type = argcheck.is_in(type, get_args(FileType), "type", True)
        always_exists = argcheck.is_bool(always_exists, "always_exists")
        super().__init__(default=default, **params)
        self.always_exists = always_exists
        self.type = type

    def __get__(self, obj, objtype) -> Optional[str]:
        path: Optional[StrPath] = super().__get__(obj, objtype)
        if path is not None:
            path = os.path.normpath(path)
            if os.path.exists(path):
                if self.type is not None and (
                    os.path.isdir(path) != (self.type == "dir")
                    or os.path.isfile(path) != (self.type == "file")
                ):
                    raise IOError(f"'{path}' is not a {self.type}")
            elif self.always_exists:
                raise IOError(f"'{path}' does not exist")
        return path

    @classmethod
    def serialize(cls, value: Optional[StrPath]) -> Optional[str]:
        if value is None:
            return value
        return PurePath(value).as_posix()

    @classmethod
    def deserialize(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value == "null":
            return None
        return os.path.normpath(value)


class LitDataModuleParamsMetaclass(pabc.AbstractParameterizedMetaclass):
    """ABC for LitDataModuleParams"""

    def __init__(mcs: "LitDataModuleParams", name, bases, dict_):
        pclass = dict_["pclass"]
        super().__init__(name, bases, dict_)
        mcs.param.params()["common"].class_ = pclass
        mcs.param.params()["train"].class_ = pclass
        mcs.param.params()["val"].class_ = pclass
        mcs.param.params()["test"].class_ = pclass
        mcs.param.params()["predict"].class_ = pclass


class LitDataModuleParams(
    param.Parameterized, Generic[P], metaclass=LitDataModuleParamsMetaclass
):
    """Base class LitDataModule parameters"""

    pclass: Type[P] = param.Parameterized

    prefer_split: bool = param.Boolean(True)

    common: Optional[P] = param.ClassSelector(
        param.Parameterized,
        instantiate=False,
        doc="Common data loader parameters. If set, cannot instantiate train, val, "
        "test, or predict",
    )
    train: Optional[P] = param.ClassSelector(
        param.Parameterized,
        instantiate=False,
        doc="Training data loader parameters. If set, cannot instantiate common",
    )
    val: Optional[P] = param.ClassSelector(
        param.Parameterized,
        instantiate=False,
        doc="Validation data loader parameters. If set, cannot instantiate common",
    )
    test: Optional[P] = param.ClassSelector(
        param.Parameterized,
        instantiate=False,
        doc="Test data loader parameters. If set, cannot instantiate common",
    )
    predict: Optional[P] = param.ClassSelector(
        param.Parameterized,
        instantiate=False,
        doc="Prediction data loader parameters. If set, cannot instantiate common",
    )

    train_dir: Optional[str] = PosixPath(None, doc="Path to training data directory")
    val_dir: Optional[str] = PosixPath(None, doc="Path to validation data directory")
    test_dir: Optional[str] = PosixPath(None, doc="Path to test data directory")
    predict_dir: Optional[str] = PosixPath(
        None,
        doc="Path to prediction data directory (leave empty to use test_dir if avail.)",
    )

    @property
    def loader_params_are_split(self) -> bool:
        return any(
            x is not None for x in (self.train, self.val, self.test, self.predict)
        )

    @property
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
    def train_params(self) -> Optional[P]:
        if self._use_split():
            return self.train
        else:
            return self.common

    @train_params.setter
    def train_params(self, params: Optional[P]):
        if self._use_split():
            self.train = params
        else:
            self.common = params

    @property
    def val_params(self) -> Optional[P]:
        if self._use_split():
            return self.val
        else:
            return self.common

    @val_params.setter
    def val_params(self, params: Optional[P]):
        if self._use_split():
            self.val = params
        else:
            self.common = params

    @property
    def test_params(self) -> Optional[P]:
        if self._use_split():
            return self.test
        else:
            return self.common

    @test_params.setter
    def test_params(self, params: Optional[P]):
        if self._use_split():
            self.train = params
        else:
            self.common = params

    @property
    def predict_params(self) -> Optional[P]:
        if self._use_split():
            return self.predict
        else:
            return self.common

    @test_params.setter
    def test_params(self, params: Optional[P]):
        if self._use_split():
            self.test = params
        else:
            self.common = params

    def initialize_missing(self, include_predict: bool = False):
        if self._use_split():
            with param.parameterized.batch_call_watchers(self):
                if self.train is None:
                    self.train = self.pclass(name="train")
                if self.val is None:
                    self.val = self.pclass(name="val")
                if self.test is None:
                    self.test = self.pclass(name="test")
                if include_predict:
                    self.predict = self.pclass(name="predict")
        elif self.common is None:
            self.common = self.pclass(name="common")

    @property
    def dev_dir(self) -> str:
        return self.val_dir

    @dev_dir.setter
    def dev_dir(self, val) -> str:
        self.val_dir = val

    @property
    def dev_params(self) -> Optional[P]:
        return self.val_params

    def _use_split(self) -> bool:
        return self.loader_params_are_split or (
            self.prefer_split and not self.loader_params_are_merged
        )


DS = TypeVar("DS", bound=torch.utils.data.Dataset)
DL = TypeVar("DL", bound=torch.utils.data.DataLoader)

Partition = Literal["train", "val", "test", "predict"]


class LitDataModule(pl.LightningDataModule, Generic[P, DS, DL], metaclass=abc.ABCMeta):
    """An ABC handling DataLoader parameterizations and partitions for lightning"""

    pclass: Type[LitDataModuleParams[P]]

    train_set: Optional[DS]
    predict_set: Optional[DS]
    test_set: Optional[DS]

    _num_workers: Optional[int]
    _pin_memory: Optional[bool]

    params: LitDataModuleParams[P]

    def __init__(
        self,
        params: LitDataModuleParams[P],
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ) -> None:
        params = argcheck.is_a(params, self.pclass, "params")
        num_workers = argcheck.is_nonnegi(num_workers, "num_workers", True)
        pin_memory = argcheck.is_bool(pin_memory, "pin_memory", True)
        super().__init__()

        self.params = params

        # The member is a "local copy" of the hyperparameter. The default changes
        # depending on the node we're running on, so we don't want to propagate them
        # back to the module state
        self._num_workers = num_workers
        self._pin_memory = pin_memory

        self.train_set = self.predict_set = self.test_set = self.val_set = None

    # @property
    # def params(self) -> LitDataModuleParams[P]:
    #     """the data module parameters"""
    #     return self.hparams.params

    @property
    def num_workers(self) -> Optional[int]:
        """Optional[int]: the number of parallel workers in a dataloader

        If initially unset, the value will be populated during :func:`setup` based on
        the number of CPU cores on the node.
        """
        return self._num_workers

    @property
    def pin_memory(self) -> Optional[bool]:
        """Optional[bool]: whether to pin memory to the cuda device
        
        If initially unset, the value will be populated during :func:`setup` based on
        whether the trainer's accelerator is on the GPU.
        """
        return self._pin_memory

    @abc.abstractmethod
    def construct_dataset(self, partition: Partition, path: str, params: P,) -> DS:
        ...

    def _construct_dataset_with_checks(
        self, partition: Partition, path: Optional[str], params: Optional[P],
    ) -> DS:
        if path is None:
            raise ValueError(f"Cannot construct {partition} dataset: no data directory")
        if params is None:
            raise ValueError(
                f"Cannot construct {partition} dataset: parameters not initialized"
            )
        return self.construct_dataset(partition, path, params)

    def setup(self, stage: Optional[str] = None):

        if self._num_workers is None:
            if self.trainer is not None and isinstance(
                self.trainer.strategy, pl.strategies.DDPSpawnStrategy
            ):
                self._num_workers = 1
            else:
                self._num_workers = torch.multiprocessing.cpu_count()

        if self._pin_memory is None:
            if self.trainer is not None:
                self._pin_memory = isinstance(
                    self.trainer.accelerator, pl.accelerators.CUDAAccelerator,
                )
            else:
                self._pin_memory = True

        if stage in {"fit", None}:
            self.train_set = self._construct_dataset_with_checks(
                "train", self.params.train_dir, self.params.train_params,
            )
            if self.params.val_dir is not None:
                self.val_set = self._construct_dataset_with_checks(
                    "val", self.params.val_dir, self.params.val_params,
                )

        if stage in {"test", None}:
            self.test_set = self._construct_dataset_with_checks(
                "test", self.params.test_dir, self.params.test_params,
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
            self.predict_set = self._construct_dataset_with_checks(
                "predict", pred_dir, pred_params
            )

    @abc.abstractmethod
    def construct_dataloader(self, partition: Partition, ds: DS, params: P) -> DL:
        ...

    def _construct_dataloader_with_checks(
        self, partition: Partition, ds: Optional[DS], params: Optional[P],
    ) -> DL:
        if params is None:
            raise ValueError(
                f"Cannot construct {partition} dataloader: parameters not "
                "initialized"
            )
        if ds is None:
            raise ValueError(
                f"Cannot construct {partition} dataloader: dataset not "
                "initialized (was setup performed?)"
            )
        return self.construct_dataloader(partition, ds, params)

    def train_dataloader(self) -> DL:
        return self._construct_dataloader_with_checks(
            "train", self.train_set, self.params.train_params
        )

    def val_dataloader(self) -> DL:
        return self._construct_dataloader_with_checks(
            "val", self.val_set, self.params.val_params
        )

    def dev_dataloader(self) -> DL:
        return self.val_dataloader()

    def test_dataloader(self) -> DL:
        return self._construct_dataloader_with_checks(
            "test", self.test_set, self.params.test_params,
        )

    def predict_dataloader(self) -> DL:
        params = self.params.predict_params
        if params is None:
            params = self.params.test_params
        return self._construct_dataloader_with_checks(
            "predict", self.predict_set, params,
        )

    @classmethod
    def add_argparse_args(
        cls,
        parser: argparse.ArgumentParser,
        split: bool = True,
        include_overloads: bool = True,
        read_format_str: str = "--read-data-{file_format}",
        print_format_str: Optional[str] = None,
    ):
        pobj = cls.pclass(name="data_params")
        pobj.prefer_split = split
        pobj.initialize_missing()

        if print_format_str is not None:
            pargparse.add_serialization_group_to_parser(
                parser, pobj, reckless=True, flag_format_str=print_format_str
            )

        grp = pargparse.add_deserialization_group_to_parser(
            parser, pobj, "data_params", reckless=True, flag_format_str=read_format_str,
        )

        if include_overloads:
            grp.add_argument(
                "--train-dir",
                metavar="DIR",
                type=readable_dir,
                default=None,
                help="Path to training directory. Clobbers value in data_params",
            )
            grp.add_argument(
                "--val-dir",
                "--dev-dir",
                metavar="DIR",
                type=readable_dir,
                default=None,
                help="Path to validation directory. Clobbers value in data_params",
            )
            grp.add_argument(
                "--test-dir",
                metavar="DIR",
                type=readable_dir,
                default=None,
                help="Path to test directory. Clobbers value in data_params",
            )
            grp.add_argument(
                "--predict-dir",
                metavar="DIR",
                type=readable_dir,
                default=None,
                help="Path to predict directory. Clobbers value in data_params",
            )

        return grp

    @classmethod
    def from_argparse_args(
        cls, namespace: argparse.Namespace, **kwargs,
    ):
        data_params = namespace.data_params
        data_params.initialize_missing()
        for overload in ("train_dir", "val_dir", "test_dir", "predict_dir"):
            value = getattr(namespace, overload, None)
            if value is not None:
                setattr(data_params, overload, value)

        return cls(data_params, **kwargs)


class LitLangDataModuleParams(LitDataModuleParams[LangDataLoaderParams]):
    """Params for a LitLangDataModule"""

    pclass = LangDataLoaderParams

    vocab_size: Optional[int] = param.Integer(
        None, bounds=(1, None), doc="Vocabulary size",
    )
    info_path: Optional[str] = PosixPath(
        None,
        doc="Path to output of get-torch-spect-data-dir-info command on train_dir/",
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


class LitSpectDataModuleParams(LitDataModuleParams[SpectDataLoaderParams]):
    """Parameters for LitSpectDataModule"""

    pclass = SpectDataLoaderParams

    info_path: Optional[str] = PosixPath(
        None, doc="Path to output of get-torch-spect-data-dir-info command on train_dir"
    )

    mvn_path: Optional[str] = PosixPath(
        None,
        doc="Path to output of compute-mvn-stats-for-torch-feat-data-dir on train_dir",
    )


class LitSpectDataModule(
    LitDataModule[SpectDataLoaderParams, SpectDataSet, SpectDataLoader]
):
    """A LitDataModule for SpectDataLoaders"""

    pclass = LitSpectDataModuleParams
    params: LitSpectDataModuleParams
    batch_first: bool
    sort_batch: bool
    suppress_alis: bool
    tokens_only: bool
    suppress_uttids: Optional[bool]
    shuffle: Optional[bool]
    warn_on_missing: bool
    on_uneven_distributed: OnUnevenDistributed

    _num_filts: Optional[int]
    _info_dict: Optional[Dict[str, int]]
    _mvn_mean: Optional[torch.Tensor]
    _mvn_std: Optional[torch.Tensor]

    def __init__(
        self,
        data_params: LitSpectDataModuleParams,
        batch_first: bool = False,
        sort_batch: bool = False,
        suppress_alis: bool = True,
        tokens_only: bool = True,
        suppress_uttids: Optional[bool] = None,
        shuffle: Optional[bool] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        warn_on_missing: bool = True,
        on_uneven_distributed: OnUnevenDistributed = "raise",
    ) -> None:
        data_params = argcheck.is_a(
            data_params, LitSpectDataModuleParams, "data_params"
        )
        batch_first = argcheck.is_bool(batch_first, "batch_first")
        sort_batch = argcheck.is_bool(sort_batch, "sort_batch")
        suppress_alis = argcheck.is_bool(suppress_alis, "suppress_alis")
        tokens_only = argcheck.is_bool(tokens_only, "tokens_only")
        suppress_uttids = argcheck.is_bool(suppress_uttids, "suppress_uttids", True)
        shuffle = argcheck.is_bool(shuffle, "shuffle", True)
        warn_on_missing = argcheck.is_bool(warn_on_missing, "warn_on_missing")
        on_uneven_distributed = argcheck.is_in(
            on_uneven_distributed,
            get_args(OnUnevenDistributed),
            "on_uneven_distributed",
        )
        super().__init__(data_params, num_workers, pin_memory)

        self.batch_first = batch_first
        self.sort_batch = sort_batch
        self.suppress_alis = suppress_alis
        self.tokens_only = tokens_only
        self.suppress_uttids = suppress_uttids
        self.shuffle = shuffle
        self.warn_on_missing = warn_on_missing
        self.on_uneven_distributed = on_uneven_distributed
        self._info_dict = self._mvn_mean = self._mvn_std = None

    def get_info_dict_value(
        self, key: str, default: Optional[int] = None
    ) -> Optional[int]:
        """Get a value from the info dict

        The info dict is gathered in :func:`setup` if ``params.info_path`` is not
        :obj:`None`.
        """
        return None if self._info_dict is None else self._info_dict.get(key, default)

    @property
    def vocab_size(self) -> Optional[int]:
        """int : vocabulary size
        
        Alias of ``max_ref_class + 1``.
        """
        return None if self.max_ref_class is None else self.max_ref_class + 1

    @property
    def batch_size(self) -> int:
        """int : training batch size
        
        This property is just the value of ``self.params.train_params.batch_size``.
        It is exposed in case ``auto_scale_batch_size`` is desired.
        """
        return self.params.train_params.batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self.params.train_params.batch_size = batch_size

    @property
    def feat_size(self) -> Optional[int]:
        """int : feature vector size
        
        Alias of `num_filts`.
        """
        return self.num_filts

    @property
    def max_ref_class(self) -> Optional[int]:
        """The maximum token id in the ref/ subdirectory (usually of training)
        
        Corresponds to the 
        """
        return self.get_info_dict_value("max_ref_class")

    def max_ali_class(self) -> Optional[int]:
        """int: the maximum token id in the ali/ subdirectory (usually of training)
        
        Determined in :func:`setup` if `params.info_path` is not :obj:`None`.
        """
        return self.get_info_dict_value("max_ali_class")

    @property
    def num_filts(self) -> Optional[int]:
        """int : size of the last dimension of tensors in feat/
        
        Determined in :func:`setup` if `params.info_path` is not :obj:`None`.
        """
        return None if self._info_dict is None else self._info_dict["num_filts"]

    def construct_dataset(
        self, partition: Partition, path: str, params: SpectDataLoaderParams,
    ) -> SpectDataSet:
        suppress_uttids = self.suppress_uttids
        if suppress_uttids is None:
            suppress_uttids = partition != "predict"
        return SpectDataSet(
            path,
            warn_on_missing=self.warn_on_missing,
            params=params,
            feat_mean=self._mvn_mean,
            feat_std=self._mvn_std,
            suppress_alis=self.suppress_alis,
            tokens_only=self.tokens_only,
            suppress_uttids=suppress_uttids,
        )

    def setup(self, stage: Optional[str] = None):

        if self.params.info_path is not None and self._info_dict is None:
            self._info_dict = dict()
            with open(self.params.info_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    key, value = line.split()
                    value = int(value)
                    if value != -1:
                        self._info_dict[key] = value

        if (
            self.params.mvn_path is not None
            and self._mvn_mean is None
            and self._mvn_std is None
        ):
            dict_ = torch.load(self.params.mvn_path, "cpu")

            self._mvn_mean = dict_["mean"]
            self._mvn_std = dict_["std"]

        super().setup(stage)

    def construct_dataloader(
        self, partition: Partition, ds: SpectDataSet, params: SpectDataLoaderParams,
    ) -> SpectDataLoader:
        shuffle = self.shuffle
        if shuffle is None:
            shuffle = partition == "train"
        return SpectDataLoader(
            ds,
            params,
            shuffle=shuffle,
            batch_first=self.batch_first,
            sort_batch=self.sort_batch,
            init_epoch=0 if self.trainer is None else self.trainer.current_epoch,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            on_uneven_distributed=self.on_uneven_distributed,
        )

    @classmethod
    def add_argparse_args(
        cls,
        parser: argparse.ArgumentParser,
        split: bool = True,
        include_overloads: bool = True,
        read_format_str: str = "--read-data-{file_format}",
        print_format_str: Optional[str] = None,
    ):
        grp = super().add_argparse_args(
            parser, split, include_overloads, read_format_str, print_format_str
        )
        if include_overloads:
            grp.add_argument(
                "--mvn-path",
                metavar="PTH",
                default=None,
                type=readable_file,
                help="Path to mean-variance normalization statistics. Clobbers value "
                "in data_params",
            )
            grp.add_argument(
                "--info-file",
                metavar="PTH",
                default=None,
                type=readable_file,
                help="Path to data set info file. Clobbers value in data_params",
            )
        return grp

    @classmethod
    def from_argparse_args(cls, namespace: argparse.Namespace, **kwargs):
        data_params: LitSpectDataModuleParams = namespace.data_params

        for overload in ("mvn_path", "info_file"):
            val = getattr(namespace, overload, None)
            if val is not None:
                setattr(data_params, overload, val)

        return super().from_argparse_args(namespace, **kwargs)
