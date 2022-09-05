# Copyright 2022 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import (
    Dict,
    List,
    Optional,
    Iterator,
    Container,
    Set,
    Sequence,
    Sized,
    Tuple,
    Hashable,
    Union,
)
import warnings

from collections import Counter

import param
import numpy as np
import torch

from . import config
from ._datasets import (
    ContextWindowDataParams,
    ContextWindowDataSet,
    SpectDataParams,
    SpectDataSet,
)

try:
    _BaseSampler = torch.utils.data.sampler.Sampler[int]
except TypeError:
    _BaseSampler = torch.utils.data.sampler.Sampler


class EpochRandomSampler(_BaseSampler):
    """Return random samples that are the same for a fixed epoch

    Parameters
    ----------
    data_source
        The dataset to draw the sample from.
    init_epoch
        The initial epoch
    base_seed
        Determines the starting seed of the sampler. Sampling is seeded with ``base_seed
        + epoch``. If unset, a seed is randomly generated from the default generator

    Attributes
    ----------
    epoch : int
        The current epoch. Responsible for seeding the upcoming samples.
    
    Yields
    ------
    sample_idx : int
        The current sample index of the current epoch.
    
    Notes
    -----
    This sampler can be used transparently with :mod:`torch.distributed`. On
    initialization, it checks if its process has been assigned a rank in the default
    group and, if so, yields a partition of the shuffled sample indices assigned to the
    rank. Unlike :class:`torch.utils.data.distributed.DistributedSampler`, this sampler
    will throw a :class:`RuntimeError` if the number of samples is not divisible by the
    number of processes.

    Examples
    --------
    >>> sampler = EpochRandomSampler(
    ...     torch.utils.data.TensorDataset(torch.arange(100)))
    >>> samples_ep0 = tuple(sampler)  # random
    >>> samples_ep1 = tuple(sampler)  # random, probably not same as first
    >>> assert tuple(sampler.get_samples_for_epoch(0)) == samples_ep0
    >>> assert tuple(sampler.get_samples_for_epoch(1)) == samples_ep1
    """

    epoch: int
    base_seed: int
    _offset: int
    _stride: int

    def __init__(
        self, data_source: Sized, init_epoch: int = 0, base_seed: Optional[int] = None
    ):
        self._len = len(data_source)
        self.epoch = init_epoch
        if base_seed is None:
            # we use numpy RandomState so that we can run in parallel with
            # torch's RandomState, but we acquire the initial random seed from
            # torch so that we'll be deterministic with a prior call to
            # torch.manual_seed(...)
            base_seed = torch.randint(np.iinfo(np.int32).max, (1,)).long().item()
        self.base_seed = base_seed
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() >= 0
        ):
            self._offset = torch.distributed.get_rank()
            self._stride = torch.distributed.get_world_size()
            if self._len % self._stride:
                raise RuntimeError(
                    f"dataset length ({self._len}) must be divisible by distributed "
                    f"world size ({self._stride})"
                )
        else:
            self._offset = 0
            self._stride = 1

    def __len__(self) -> int:
        return self._len

    def get_samples_for_epoch(self, epoch: int) -> np.ndarray:
        """np.ndarray : samples for a specific epoch"""
        rs = np.random.RandomState(self.base_seed + epoch)
        return rs.permutation(self._len)[self._offset :: self._stride]

    def __iter__(self) -> Iterator[int]:
        ret = iter(self.get_samples_for_epoch(self.epoch))
        self.epoch += 1
        return ret


class DistributableSequentialSampler(torch.utils.data.sampler.SequentialSampler):
    """A SequentialSampler which quietly handles :mod:`torch.distributed`

    A wrapper around :class:`torch.utils.data.sampler.SequentialSampler` which can be
    used transparently with :mod:`torch.distributed`.

    Parameters
    ----------
    data_source
        The dataset to draw the sample from.
    
    Notes
    -----
    On initialization, the sampler checks if its process has been assigned a rank in the
    default group and, if so, yields sample indices from a partition assigned to that
    rank. Each process yields strided indices of the form::

        r, r + R, r + 2R, ...
    
    where ``r`` is the rank and ``R`` is the world size. The original ordering can be
    recovered by interleaving reads from the processes. No process actually reads
    samples ``0, 1, 2, ...`` (unless the world size is ``1``), so any code which
    requires samples be presented in a global order should avoid this sampler.
    """

    epoch: int

    _offset: int
    _stride: int

    def __init__(self, data_source: Sized, init_epoch: int = 0):
        super().__init__(data_source)
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() >= 0
        ):
            self._offset = torch.distributed.get_rank()
            self._stride = torch.distributed.get_world_size()
            if len(self) % self._stride:
                raise RuntimeError(
                    f"dataset length ({len(self)}) must be divisible by distributed "
                    f"world size ({self._stride})"
                )
        else:
            self._offset = 0
            self._stride = 1
        self.epoch = init_epoch

    def __iter__(self) -> Iterator[int]:
        ret = iter(range(self._offset, len(self), self._stride))
        self.epoch += 1
        return ret


class BucketBatchSampler(_BaseSampler):
    """Batch samples into buckets, yielding as soon as the bucket is full
    
    Parameters
    ----------
    sampler
        Determines the order in which samples are put into buckets.
    idx2bucket
        A map specifying which bucket each sample belongs to. The keys are the indices
        yielded by `sampler`; the values are the ids of the corresponding buckets.
    bucket2size
        A map from the bucket ids (the values in `idx2bucket`) to the corresponding
        batch size. Values must be positive.
    drop_incomplete
        If :obj:`True`, any batches which are incomplete (smaller than the bucket's
        batch size) at the end of an epoch will be discarded. Otherwise, the incomplete
        batches will be yielded in the order of their bucket ids' hashes.
    
    Yields
    ------
    batch : list of int
        A list of indices from `sampler` all belonging to the same bucket. The batch is
        yielded as soon as it is full (or the epoch has ended with `drop_incomplete` set
        to :obj:`False`).
    
    Examples
    --------

    >>> N = 14
    >>> dataset = torch.utils.data.TensorDataset(torch.rand(N))
    >>> ssampler = torch.utils.data.SequentialSampler(dataset)
    >>> idx2bucket = dict((n, int(n % 3 == 0)) for n in range(N))
    >>> bucket2size = {0: 2, 1: 2}
    >>> bsampler = BucketBatchSampler(ssampler, idx2bucket, bucket2size, True)
    >>> print(list(bsampler))
    [[1, 2], [0, 3], [4, 5], [7, 8], [6, 9], [10, 11]]
    >>> bsampler = BucketBatchSampler(ssampler, idx2bucket, bucket2size, False)
    >>> print(list(bsampler))
    [[1, 2], [0, 3], [4, 5], [7, 8], [6, 9], [10, 11], [13], [12]]

    """

    sampler: _BaseSampler
    idx2bucket: Dict[int, Hashable]
    bucket2size: Dict[Hashable, int]
    drop_incomplete: bool
    _len: Optional[int]

    def __init__(
        self,
        sampler: _BaseSampler,
        idx2bucket: Dict[int, Hashable],
        bucket2size: Dict[Hashable, int],
        drop_incomplete: bool,
    ):
        self.sampler = sampler
        self.idx2bucket = idx2bucket
        self.bucket2size = bucket2size
        self.drop_incomplete = drop_incomplete
        self._len = None

    def __len__(self) -> int:
        if self._len is None:
            bucket2count = Counter(self.idx2bucket.values())
            self._len = 0
            for bucket, size in self.bucket2size.items():
                count = bucket2count.get(bucket, 0)
                self._len += count // size
                if not self.drop_incomplete and count % size:
                    self._len += 1
        return self._len

    def __iter__(self) -> Iterator[List[int]]:
        batches: Dict[Hashable, List[int]] = dict()
        for idx in self.sampler:
            hash_ = self.idx2bucket[idx]
            batch_size = self.bucket2size[hash_]
            batch = batches.setdefault(hash_, [])
            batch.append(idx)
            if batch_size == len(batch):
                yield batch
                del batches[hash_]
            elif batch_size < len(batch):
                raise RuntimeError(f"batch '{hash_}' has invalid size '{batch_size}'")
        if not self.drop_incomplete:
            for _, batch in sorted(batches.items(), key=lambda x: x[0]):
                yield batch


class DataLoaderParams(param.Parameterized):
    """General parameters for a DataSet from pydrobert.torch.data

    This implements the :class:`pydrobert.param.optuna.TunableParameterized`
    interface.
    """

    batch_size = param.Integer(
        10, bounds=(1, None), softbounds=(5, 10), doc="Number of elements in a batch.",
    )
    drop_last = param.Boolean(
        False, doc="Whether to drop the last batch if it does reach batch_size"
    )

    @classmethod
    def get_tunable(cls) -> Set[str]:
        """Returns a set of tunable parameters"""
        return {"batch_size"}

    @classmethod
    def suggest_params(
        cls, trial, base=None, only: Container[str] = None, prefix: str = ""
    ):
        """Populate a parameterized instance with values from trial"""
        params = cls() if base is None else base
        if only is None:
            only = cls.get_tunable()
        if "batch_size" in only:
            bounds = params.param.params()["batch_size"].get_soft_bounds()
            val = trial.suggest_int(prefix + "batch_size", *bounds)
            params.batch_size = val
        return params


class DynamicLengthDataLoaderParams(DataLoaderParams):
    """Parameters for a data loader whose elements have dynamic lengths"""

    num_length_buckets = param.Integer(
        1,
        bounds=(1, None),
        doc="If greater than 1, elements will be batched with other elements of "
        "similar length (along the feature time dimension). Elements will be "
        "partioned roughly evenly into num_length_buckets. Increasing "
        "num_length_buckets will usually decrease the total amount of padding "
        "per batch at the cost of fewer candidates to choose from within batches.",
    )
    size_batch_by_length = param.Boolean(
        False,
        doc="Only matters when num_length_buckets > 1. If false, all buckets have the "
        "same batch size of batch_size. If true, buckets with shorter-length "
        "utterances will contain greater than batch_size elements per batch. Letting "
        "x be the batch size of a bucket, y be the length of the largest element in "
        "the bucket, and Y be the length of the largest element in the corpus, x is "
        "the greatest value such that x * y <= Y * batch_size",
    )


class SpectDataLoaderParams(SpectDataParams, DynamicLengthDataLoaderParams):
    """Parameters for a :class:`SpectDataLoader`

    This implements the :class:`pydrobert.param.optuna.TunableParameterized` interface.
    """

    @classmethod
    def get_tunable(cls) -> Set[str]:
        return (
            SpectDataParams.get_tunable() | DynamicLengthDataLoaderParams.get_tunable()
        )

    @classmethod
    def suggest_params(
        cls, trial, base=None, only: Container[str] = None, prefix: str = ""
    ):
        params = cls() if base is None else base
        SpectDataParams.suggest_params(trial, params, only, prefix)
        DynamicLengthDataLoaderParams.suggest_params(trial, params, only, prefix)
        return params


def spect_seq_to_batch(
    seq: Sequence[Tuple[Union[torch.Tensor, str, None], ...]],
    batch_first: bool = True,
    sort: bool = True,
    has_alis: bool = True,
    has_uttids: bool = False,
) -> Tuple[Union[torch.Tensor, Tuple[str, ...], None], ...]:
    """Convert a sequence of spectral data to a batch

    This function is used to collate sequences of elements from a :class:`SpectDataSet`
    into batches.

    Parameters
    ----------
    seq
        A finite-length (``N``) sequence of tuples, each tuple corresponding to an
        utterance and containing, in order:

        1. `feat_n`, a tensor of size ``(T_n, F)`` representing per-frame spectral
           features.
        2. `ali_n` (if `has_alis` is :obj:`True)`, either :obj:`None` or a tensor
           of shape ``(T_n)`` representing per-frame alignment ids.
        3. `ref_n`, either :obj:`None` or a tensor of size ``(R_n[, 3])`` representing
           reference token sequences and optionally their frame shifts. Either all
           `ref_n` must contain the frame shift info (the ``3`` dimension) or none of
           them.
        4. `utt_n` (if `has_uttids` is :obj:`True`), the utterance id.

    batch_first
        If :obj:`True`, the batch dimension ``N`` comes before the sequence dimension
        ``T`` or ``R`` in the return values.
    sort
        If :obj:`True`, the tuples in `seq` are first sorted in descending order of
        ``T_n`` before being batched.
    has_alis
        Whether `ali_n` is part of the input values and `alis` is part of the output
        values. Note that `has_alis` should still be :obj:`True` if `ali_n` is present
        in `seq` but is :obj:`None`.
    has_uttids
        Whether `utt_n` is part of the input values and `uttids` is part of the output
        values.

    Returns
    -------
    batch
        A tuple containing the following elements:
        
        1. `feats`, a tensor of shape ``(max_n T_n, N, F)`` containing the right-padded
           sequences ``[feat_1, feat_2, ..., feat_N]``. Padded with zeros.
        2. `alis` (if `has_alis` is :obj:`True`), either :obj:`None` or a tensor of
           shape ``(max_n T_n, N)`` containing the right-padded sequence ``[ali_1,
           ali_2, ... ali_N]``. Padded with
           :const:`pydrobert.torch.config.INDEX_PAD_VALUE`.
        3. `refs`, either :obj:`None` or a tensor of shape ``(max_n R_n, N[, 3])``
            containing the right-padded sequences ``[ref_1, ref_2, ..., ref_N]``. 
            Padded with :const:`pydrobert.torch.config.INDEX_PAD_VALUE`.
        4. `feat_sizes`, a tensor of shape ``(N,)`` containing the sequence ``[T_1, T_2,
           ..., T_N]``.
        5. `ref_sizes`, a tensor of shape ``(N,)`` containing the sequence ``[R_1, R_2,
           ..., R_N]``.
        6. `uttids` (if `has_uttids` is :obj:`True`), an ``N``-tuple of the utterance
           ids.
    """
    if sort:
        seq = sorted(seq, key=lambda x: x[0].size(0), reverse=True)
    seq = list(zip(*seq))
    if has_alis:
        if has_uttids:
            feats, alis, refs, uttids = seq
        else:
            feats, alis, refs = seq
        ali_not_none = all(x is not None for x in alis)
    elif has_uttids:
        feats, refs, uttids = seq
        ali_not_none = False
    else:
        feats, refs = seq
        ali_not_none = False
    ref_not_none = all(x is not None for x in refs)
    feat_sizes = torch.tensor([x.size(0) for x in feats])
    feats = torch.nn.utils.rnn.pad_sequence(
        feats, padding_value=0, batch_first=batch_first
    )
    if ali_not_none:
        alis = torch.nn.utils.rnn.pad_sequence(
            alis, padding_value=config.INDEX_PAD_VALUE, batch_first=batch_first
        )
    else:
        alis = None
    if ref_not_none:
        ref_sizes = torch.tensor([len(x) for x in refs])
        refs = torch.nn.utils.rnn.pad_sequence(
            refs, padding_value=config.INDEX_PAD_VALUE, batch_first=batch_first
        )
    else:
        ref_sizes = refs = None
    if has_alis:
        if has_uttids:
            return feats, alis, refs, feat_sizes, ref_sizes, tuple(uttids)
        else:
            return feats, alis, refs, feat_sizes, ref_sizes
    elif has_uttids:
        return feats, refs, feat_sizes, ref_sizes, tuple(uttids)
    else:
        return feats, refs, feat_sizes, ref_sizes


def _get_bucket_batch_sampler_params(dataset, num_buckets, batch_size, dynamic):
    elem_per_bucket = len(dataset) // num_buckets
    if elem_per_bucket < batch_size:
        warnings.warn(
            f"The number of elements per bucket of the dataset ({elem_per_bucket}) "
            f"is less than batch_size ({batch_size}). Consider decreasing "
            "num_length_buckets"
        )
    len_idx = sorted((x[0].size(0), i) for (i, x) in enumerate(dataset))
    len_bounds = [len_idx[(n + 1) * elem_per_bucket - 1][0] for n in range(num_buckets)]
    len_bounds[-1] = len_idx[-1][0]
    len_bounds_ = sorted(set(len_bounds))
    if len_bounds_ != len_bounds:
        warnings.warn(
            f"Cannot evenly split dataset into {num_buckets} buckets. Decreasing to "
            f"{len(len_bounds_)}"
        )
        len_bounds = len_bounds_
    num_buckets = len(len_bounds)
    idx2bucket = dict((i, sum(int(l > b) for b in len_bounds)) for (l, i) in len_idx)
    if dynamic:
        m = len_bounds[-1] * batch_size
        bucket2size = dict((j, m // len_bounds[j]) for j in range(num_buckets))
    else:
        bucket2size = dict((j, batch_size) for j in range(num_buckets))
    return idx2bucket, bucket2size


class SpectDataLoader(torch.utils.data.DataLoader):
    """Dataloader for a :class:`SpectDataSet`

    Parameters
    ----------
    data
        Either a :class:`SpectDataSet` or a path to the data directory.
    params
        Contains at least the parameters specific to the loader. May also contain
        data set params -- see `data_params`.
    data_params
        Data set parameters. Relevant only when `data` is a path. Used to initialize
        the underlying :class:`SpectDataSet`. If :obj:`None`, `params` is assumed to
        also contain the data set parameters.
    shuffle
        Whether utterances are shuffled at every epoch or presented sequentially.
    batch_first
        Whether the batch dimension comes before the sequence dimension in `feats`
        and `refs`.
    sort_batch
        Whether utterances in a batch are sorted by feature length.
    init_epoch
        The epoch to resume from. When combined with a fixed `seed`, ensures the same
        batches are always delivered for a given epoch.
    seed
        The initial seed used for shuffling data. If unset, a random one will be
        generated.
    **kwargs
        Additional keyword arguments to initialize :class:`SpectDataSet` and
        :class:`torch.utils.data.DataLoader`. The former is only relevant when
        `data` is a path.
    
    Yields
    ------
    batch
        A tuple ``feats[, alis,] refs, feat_sizes, ref_sizes[, uttids]``, with `alis`
        included if `suppress_alis` is :obj:`False` and `uttids` included if
        `suppress_uttids` is :obj:`False`. See :func:`spect_seq_to_batch` for more
        information on the elements.
    """

    dataset: SpectDataSet
    batch_first: bool
    batch_sampler: Union[BucketBatchSampler, torch.utils.data.BatchSampler]
    sort_batch: bool

    def __init__(
        self,
        data: Union[str, SpectDataSet],
        params: Union[SpectDataLoaderParams, DynamicLengthDataLoaderParams],
        data_params: Optional[SpectDataParams] = None,
        shuffle: bool = True,
        batch_first: bool = True,
        sort_batch: bool = False,
        init_epoch: int = 0,
        seed: Optional[int] = None,
        **kwargs,
    ):
        for bad_kwarg in {
            "batch_size",
            "sampler",
            "batch_sampler",
            "collate_fn",
        }:
            if bad_kwarg in kwargs:
                raise TypeError(
                    f"keyword argument '{bad_kwarg}' invalid for {type(self)} types"
                )
        ds_kwargs, dl_kwargs = dict(), dict()
        for key, val in kwargs.items():
            if key in {
                "file_prefix",
                "file_suffix",
                "warn_on_missing",
                "subset_ids",
                "sos",
                "eos",
                "feat_subdir",
                "ali_subdir",
                "ref_subdir",
                "feat_mean",
                "feat_std",
                "suppress_alis",
                "suppress_uttids",
                "tokens_only",
            }:
                ds_kwargs[key] = val
            else:
                dl_kwargs[key] = val
        if data_params is None:
            data_params = params
        else:
            if hasattr(params, "subset_ids"):
                subset_ids = params.subset_ids
                if subset_ids:
                    warnings.warn(
                        "setting subset_ids in data loader parameters is deprecated. "
                        "Use data_params.subset_ids instead.",
                        DeprecationWarning,
                        2,
                    )
                    data_params.subset_ids = subset_ids
        self.batch_first, self.sort_batch = batch_first, sort_batch
        if isinstance(data, SpectDataSet):
            dataset = data
        else:
            dataset = SpectDataSet(data, params=data_params, **ds_kwargs)
        if shuffle:
            utt_sampler = EpochRandomSampler(
                dataset, init_epoch=init_epoch, base_seed=seed
            )
        else:
            utt_sampler = DistributableSequentialSampler(dataset)
        if not isinstance(
            params, (DynamicLengthDataLoaderParams, SpectDataLoaderParams)
        ) and isinstance(params, DataLoaderParams):
            warnings.warn(
                "Passing a DataLoaderParams instance as params is deprecated. "
                "Switch to DynamicLengthDataLoaderParams.",
                DeprecationWarning,
            )
            num_length_buckets = 1
        else:
            num_length_buckets = params.num_length_buckets
        if num_length_buckets > 1:
            idx2bucket, bucket2size = _get_bucket_batch_sampler_params(
                dataset,
                num_length_buckets,
                params.batch_size,
                params.size_batch_by_length,
            )
            batch_sampler = BucketBatchSampler(
                utt_sampler, idx2bucket, bucket2size, params.drop_last
            )
        else:
            batch_sampler = torch.utils.data.BatchSampler(
                utt_sampler, params.batch_size, drop_last=params.drop_last
            )
        super().__init__(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            **dl_kwargs,
        )

    def collate_fn(self, seq):
        return spect_seq_to_batch(
            seq,
            self.batch_first,
            self.sort_batch,
            not self.dataset.suppress_alis,
            not self.dataset.suppress_uttids,
        )

    @property
    def epoch(self) -> int:
        """int : the current epoch"""
        return self.batch_sampler.sampler.epoch

    @epoch.setter
    def epoch(self, val: int):
        self.batch_sampler.sampler.epoch = val


class SpectTrainingDataLoader(SpectDataLoader):
    """Serves batches of spectral data over random orders of utterances

    Deprecated. Use :class:`SpectDataLoader`.
    """

    def __init__(
        self,
        data: Union[str, SpectDataSet],
        params: Union[SpectDataLoaderParams, DynamicLengthDataLoaderParams],
        file_prefix: str = "",
        file_suffix: str = ".pt",
        warn_on_missing: bool = True,
        feat_subdir: str = "feat",
        ali_subdir: str = "ali",
        ref_subdir: str = "ref",
        init_epoch: int = 0,
        batch_first: bool = True,
        data_params: Optional[SpectDataParams] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        warnings.warn(
            "SpectTrainingDataLoader is deprecated. Use SpectDataLoader instead",
            DeprecationWarning,
        )
        shuffle = kwargs.pop("shuffle", True)
        suppress_alis = kwargs.pop("suppress_alis", False)
        suppress_uttids = kwargs.pop("suppress_uttids", True)
        tokens_only = kwargs.pop("tokens_only", False)
        sort_batch = kwargs.pop("sort_batch", True)
        super().__init__(
            data,
            params,
            data_params,
            shuffle,
            batch_first,
            sort_batch,
            init_epoch,
            seed,
            file_prefix=file_prefix,
            file_suffix=file_suffix,
            warn_on_missing=warn_on_missing,
            feat_subdir=feat_subdir,
            ali_subdir=ali_subdir,
            ref_subdir=ref_subdir,
            suppress_alis=suppress_alis,
            suppress_uttids=suppress_uttids,
            tokens_only=tokens_only,
            **kwargs,
        )


class SpectEvaluationDataLoader(SpectDataLoader):
    """Serves batches of spectral data over random orders of utterances

    Deprecated. Use :class:`SpectDataLoader`.
    """

    def __init__(
        self,
        data: Union[str, SpectDataSet],
        params: Union[SpectDataLoaderParams, DynamicLengthDataLoaderParams],
        file_prefix: str = "",
        file_suffix: str = ".pt",
        warn_on_missing: bool = True,
        feat_subdir: str = "feat",
        ali_subdir: str = "ali",
        ref_subdir: str = "ref",
        batch_first: bool = True,
        data_params: Optional[SpectDataParams] = None,
        **kwargs,
    ):
        warnings.warn(
            "SpectEvaluationDataLoader is deprecated. Use SpectDataLoader instead",
            DeprecationWarning,
        )
        shuffle = kwargs.pop("shuffle", False)
        suppress_alis = kwargs.pop("suppress_alis", False)
        suppress_uttids = kwargs.pop("suppress_uttids", False)
        tokens_only = kwargs.pop("tokens_only", False)
        init_epoch = kwargs.pop("init_epoch", 0)
        seed = kwargs.pop("seed", None)
        sort_batch = kwargs.pop("sort_batch", True)
        super().__init__(
            data,
            params,
            data_params,
            shuffle,
            batch_first,
            sort_batch,
            init_epoch,
            seed,
            file_prefix=file_prefix,
            file_suffix=file_suffix,
            warn_on_missing=warn_on_missing,
            feat_subdir=feat_subdir,
            ali_subdir=ali_subdir,
            ref_subdir=ref_subdir,
            suppress_alis=suppress_alis,
            suppress_uttids=suppress_uttids,
            tokens_only=tokens_only,
            **kwargs,
        )


def context_window_seq_to_batch(
    seq: Sequence[Tuple[Union[torch.Tensor, str, None], ...]], has_uttids: bool = False
) -> Tuple[Union[torch.Tensor, Sequence[str], None], ...]:
    r"""Convert a sequence of context window elements to a batch

    This function is used to collate sequences of elements from a
    :class:`ContextWindowDataSet` into batches.

    Assume `seq` is a finite length sequence of pairs of ``window, ali``, where
    ``window`` is of size ``(T, C, F)``, where ``T`` is some number of windows (which
    can vary across elements in the sequence), ``C`` is the window size, and ``F`` is
    some number filters, and ``ali`` is of size ``(T,)``. This method batches all the
    elements of the sequence into a pair of ``windows, alis``, where `windows` and
    `alis` will have shapes ``(N, C, F)`` and ``(N,)`` resp., where :math:`N = \sum T`
    is the total number of context windows over the utterances.

    If ``ali`` is :obj:`None` in any element, `alis` will also be :obj:`None`

    Parameters
    ----------
    seq
        A finite-length (``N``) sequence of tuples, each tuple corresponding to an
        utterance and containing, in order:

        1. `window_n`, a tensor of size ``(T_n, C, F)`` representing windowed spectral
           features.
        2. `ali_n`, either :obj:`None` or a tensor of shape ``(T_n,)`` representing
           per-window alignment ids.
        3. `uttid_n` (if :obj:`has_refs` is :obj:`True`), the utterance id.
    
    has_uttids
        Whether `utt_n` is part of the input values and both `window_sizes` and `uttids`
        are part of the output values.

    Returns
    -------
    batch
        A tuple containing the following elements:

        1. `windows`, a tensor of shape ``(sum_n T_n, C, F)`` containing the
           concatenated set of windows ``[window_1, window_2, ..., window_N]``
        2. `alis`, either :obj:`None` or a tensor of shape ``(sum_n T_n,)`` containing
           the concatenated alignment ids ``[ali_1, ali_2, ..., ali_N]``.
        3. `window_sizes` (if `has_uttids` is :obj:`True`), a tensor of shape ``(N,)``
           containing the sequence ``[T_1, T_2, ..., T_N]``.
        4. `uttids` (if `has_uttids` is :obj:`True`), an ``N``-tuple of utterance ids.
    """
    seq = list(zip(*seq))
    if has_uttids:
        windows, alis, uttids = seq
        window_sizes = torch.tensor([w.size(0) for w in windows])
    else:
        windows, alis = seq
    windows = torch.cat(windows)
    if all(a is not None for a in alis):
        alis = torch.cat(alis)
    else:
        alis = None
    if has_uttids:
        return windows, alis, window_sizes, tuple(uttids)
    else:
        return windows, alis


class ContextWindowDataLoaderParams(ContextWindowDataParams, DataLoaderParams):
    """Parameters for a :class:`ContextWindowDataLoader`

    This implements the :class:`pydrobert.param.optuna.TunableParameterized` interface.
    """

    @classmethod
    def get_tunable(cls):
        """Returns a set of tunable parameters"""
        return DataLoaderParams.get_tunable() | ContextWindowDataParams.get_tunable()

    @classmethod
    def suggest_params(cls, trial, base=None, only=None, prefix=""):
        """Populate a parameterized instance with values from trial"""
        params = DataLoaderParams.suggest_params(
            trial, base=base, only=only, prefix=prefix
        )
        params = ContextWindowDataParams.suggest_params(
            trial, base=params, only=only, prefix=prefix
        )
        return params

    @classmethod
    def get_tunable(cls) -> Set[str]:
        return ContextWindowDataParams.get_tunable() | DataLoaderParams.get_tunable()

    @classmethod
    def suggest_params(
        cls, trial, base=None, only: Container[str] = None, prefix: str = ""
    ):
        params = cls() if base is None else base
        ContextWindowDataParams.suggest_params(trial, params, only, prefix)
        DataLoaderParams.suggest_params(trial, params, only, prefix)
        return params


class ContextWindowDataLoader(torch.utils.data.DataLoader):
    """DataLoader for :class:`ContextWindowDataSet`

    Parameters
    ----------
    data
        Either a :class:`ContextWindowDataSet` or a path to the data directory.
    params
        Contains at least the parameters specific to the loader. May also contain
        data set params --- see `data_params`.
    data_params
        Data set parameters. Relevant only when `data` is a path. Used to initialize
        the underlying :class:`ContextWindowDataset`. If :obj:`None`, `params` is
        assumed to also contain the data set parameters.
    shuffle
        Whether utterances are shuffled at every epoch or presented sequentially.
    sort_batch
        Whether utterances in a batch are sorted by feature length.
    init_epoch
        The epoch to resume from. When combined with a fixed `seed`, ensures the same
        batches are always delivered for a given epoch.
    seed
        The initial seed used for shuffling data. If unset, a random one will be
        generated.
    **kwargs
        Additional keyword arguments to initialize :class:`ContextWindowDataSet` and
        :class:`torch.utils.data.DataLoader`. The former is only relevant when `data` is
        a path.
    
    Yields
    ------
    batch
        A tuple ``windows, alis[, window_sizes, uttids]``, with `window_sizes` and
        `uttids` included if `suppress_uttids` is :obj:`False`. See
        :func:`context_window_seq_to_batch` for more information on the elements.
    """

    dataset: ContextWindowDataSet
    batch_sampler: torch.utils.data.BatchSampler
    batch_first: bool

    def collate_fn(self, seq):
        return context_window_seq_to_batch(seq, not self.dataset.suppress_uttids)

    def __init__(
        self,
        data: Union[str, ContextWindowDataSet],
        params: Union[ContextWindowDataLoaderParams, DataLoaderParams],
        data_params: Optional[ContextWindowDataParams] = None,
        shuffle: bool = True,
        init_epoch: int = 0,
        seed: Optional[int] = None,
        **kwargs,
    ):
        for bad_kwarg in (
            "batch_size",
            "sampler",
            "batch_sampler",
            "collate_fn",
        ):
            if bad_kwarg in kwargs:
                raise TypeError(
                    'keyword argument "{}" invalid for {} types'.format(
                        bad_kwarg, type(self)
                    )
                )
        ds_kwargs, dl_kwargs = dict(), dict()
        for key, val in kwargs.items():
            if key in {
                "left",
                "right",
                "file_prefix",
                "file_suffix",
                "warn_on_missing",
                "subset_ids",
                "feat_subdir",
                "ali_subdir",
                "reverse",
                "feat_mean",
                "feat_std",
                "suppress_uttids",
            }:
                ds_kwargs[key] = val
            else:
                dl_kwargs[key] = val
        if seed is None and hasattr(params, "seed"):
            seed = params.seed
        if data_params is None:
            data_params = params
        else:
            if hasattr(params, "subset_ids"):
                subset_ids = params.subset_ids
                if subset_ids:
                    warnings.warn(
                        "setting subset_ids in data loader parameters is deprecated. "
                        "Use data_params.subset_ids instead.",
                        DeprecationWarning,
                        2,
                    )
                    data_params.subset_ids = subset_ids
        if isinstance(data, ContextWindowDataSet):
            dataset = data
            data_dir = data.data_dir
        else:
            data_dir = data
            dataset = ContextWindowDataSet(data_dir, params=data_params, **ds_kwargs)
        if shuffle:
            utt_sampler = EpochRandomSampler(dataset, init_epoch, seed)
        else:
            utt_sampler = DistributableSequentialSampler(dataset, init_epoch)
        batch_sampler = torch.utils.data.BatchSampler(
            utt_sampler, params.batch_size, drop_last=params.drop_last
        )
        super().__init__(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            **dl_kwargs,
        )

    @property
    def epoch(self) -> int:
        """int : the current epoch"""
        return self.batch_sampler.sampler.epoch

    @epoch.setter
    def epoch(self, val: int):
        self.batch_sampler.sampler.epoch = val


class ContextWindowTrainingDataLoader(ContextWindowDataLoader):
    """Serve batches of context windows over a random order of utterances

    Deprecated. Use :class:`ContextWindowDataLoader`.
    """

    def __init__(
        self,
        data: Union[str, ContextWindowDataSet],
        params: Union[ContextWindowDataLoaderParams, DataLoaderParams],
        file_prefix: str = "",
        file_suffix: str = ".pt",
        warn_on_missing: bool = True,
        feat_subdir: str = "feat",
        ali_subdir: str = "ali",
        init_epoch: int = 0,
        data_params: Optional[ContextWindowDataParams] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        warnings.warn(
            "ContextWindowTrainingDataLoader is deprecated. Use "
            "ContextWindowDataLoader instead",
            DeprecationWarning,
        )
        shuffle = kwargs.pop("shuffle", True)
        suppress_uttids = kwargs.pop("suppress_uttids", True)
        super().__init__(
            data,
            params,
            data_params,
            shuffle,
            init_epoch,
            seed,
            file_prefix=file_prefix,
            file_suffix=file_suffix,
            warn_on_missing=warn_on_missing,
            feat_subdir=feat_subdir,
            ali_subdir=ali_subdir,
            suppress_uttids=suppress_uttids,
            **kwargs,
        )


class ContextWindowEvaluationDataLoader(ContextWindowDataLoader):
    """Serves batches of context windows over sequential utterances

    Deprecated. Use :class:`ContextWindowDataLoader`.
    """

    def __init__(
        self,
        data: Union[str, ContextWindowDataSet],
        params: Union[ContextWindowDataLoaderParams, DataLoaderParams],
        file_prefix: str = "",
        file_suffix: str = ".pt",
        warn_on_missing: bool = True,
        feat_subdir: str = "feat",
        ali_subdir: str = "ali",
        data_params: Optional[ContextWindowDataParams] = None,
        **kwargs,
    ):
        warnings.warn(
            "ContextWindowEvaluationDataLoader is deprecated. Use "
            "ContextWindowDataLoader instead",
            DeprecationWarning,
        )
        shuffle = kwargs.pop("shuffle", False)
        suppress_uttids = kwargs.pop("suppress_uttids", False)
        init_epoch = kwargs.pop("init_epoch", 0)
        seed = kwargs.pop("seed", None)
        super().__init__(
            data,
            params,
            data_params,
            shuffle,
            init_epoch,
            seed,
            file_prefix=file_prefix,
            file_suffix=file_suffix,
            warn_on_missing=warn_on_missing,
            feat_subdir=feat_subdir,
            ali_subdir=ali_subdir,
            suppress_uttids=suppress_uttids,
            **kwargs,
        )

