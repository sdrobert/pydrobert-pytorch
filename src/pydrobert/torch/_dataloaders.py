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

from typing import Optional, Iterator, Container, Set, Sequence, Tuple

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


class EpochRandomSampler(torch.utils.data.Sampler):
    """Return random samples that are the same for a fixed epoch

    Parameters
    ----------
    data_source : torch.utils.data.Dataset
        The total number of samples
    init_epoch : int, optional
        The initial epoch
    base_seed : int, optional
        Determines the starting seed of the sampler. Sampling is seeded with
        ``base_seed + epoch``. If unset, a seed is randomly generated from
        the default generator

    Attributes
    ----------
    base_seed : int
    epoch : int
        The current epoch. Responsible for seeding the upcoming samples
    data_source : torch.utils.data.Dataset

    Examples
    --------

    >>> sampler = EpochRandomSampler(
    ...     torch.data.utils.TensorDataset(torch.arange(100)))
    >>> samples_ep0 = tuple(sampler)  # random
    >>> samples_ep1 = tuple(sampler)  # random, probably not same as first
    >>> assert tuple(sampler.get_samples_for_epoch(0)) == samples_ep0
    >>> assert tuple(sampler.get_samples_for_epoch(1)) == samples_ep1
    """

    def __init__(
        self,
        data_source: torch.utils.data.Dataset,
        init_epoch: int = 0,
        base_seed: Optional[int] = None,
    ):
        super(EpochRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.epoch = init_epoch
        if base_seed is None:
            # we use numpy RandomState so that we can run in parallel with
            # torch's RandomState, but we acquire the initial random seed from
            # torch so that we'll be deterministic with a prior call to
            # torch.manual_seed(...)
            base_seed = torch.randint(np.iinfo(np.int32).max, (1,)).long().item()
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.data_source)

    def get_samples_for_epoch(self, epoch: int) -> np.ndarray:
        """np.ndarray : samples for a specific epoch"""
        rs = np.random.RandomState(self.base_seed + epoch)
        return rs.permutation(list(range(len(self.data_source))))

    def __iter__(self) -> Iterator[np.ndarray]:
        ret = iter(self.get_samples_for_epoch(self.epoch))
        self.epoch += 1
        return ret


class DataLoaderParams(param.Parameterized):
    """General parameters for a DataSet from pydrobert.torch.data

    This implements the :class:`pydrobert.param.optuna.TunableParameterized`
    interface.
    """

    batch_size = param.Integer(
        10,
        bounds=(1, None),
        softbounds=(5, 10),
        doc="Number of elements in a batch, which equals the number of "
        "utterances in the batch",
    )
    drop_last = param.Boolean(
        False, doc="Whether to drop the last batch if it does reach batch_size"
    )
    subset_ids = param.List(
        [],
        class_=str,
        bounds=None,
        doc="A list of utterance ids. If non-empty, the data set will be "
        "restricted to these utterances",
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


class SpectDataLoaderParams(SpectDataParams, DataLoaderParams):
    """Parameters for a Spect*DataLoader

    This implements the :class:`pydrobert.param.optuna.TunableParameterized`
    interface.

    See Also
    --------
    pydrobert.torch.data.SpectTrainingDataLoader
    pydrobert.torch.data.SpectEvaluationDataLoader
        Where to use these parameters.
    """


def spect_seq_to_batch(
    seq: Sequence[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]],
    batch_first: bool = True,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
    Optional[torch.Tensor],
]:
    """Convert a sequence of spectral data to a batch

    Assume `seq` is a finite length sequence of tuples ``feat, ali, ref``, where
    ``feat`` is of size ``(T, F)``, where ``T`` is some number of frames (which can vary
    across elements in the sequence), ``F`` is some number of filters, ``ali`` is of
    size ``(T,)``, and ``ref`` is of size ``(R[, 3])``, where ``R`` is some number of
    reference tokens (which can vary across elements in the sequence) and the ``3`` is a
    triple of id, start frame and end frame (optional). This method batches all the
    elements of the sequence into a tuple of ``feats, alis, refs, feat_sizes,
    ref_sizes``. `feats` and `alis` will have dimensions ``(N, T*, F)``, and ``(N,
    T*)``, resp., where ``N`` is the batch size, and ``T*`` is the maximum number of
    frames in `seq` (or ``(T*, N, F)``, ``(T*, N)`` if `batch_first` is :obj:`False`).
    Similarly, `refs` will have dimensions ``(N, R*[, 3])`` (or ``(R*, N[, 3])``).
    `feat_sizes` and `ref_sizes` are long tensor  of shape ``(N,)`` containing the
    original ``T`` and ``R`` values. The batch will be sorted by decreasing numbers of
    frames. `feats` is zero-padded while `alis` and `refs` are padded with module
    constant :const:`pydrobert.torch.config.INDEX_PAD_VALUE`

    If ``ali`` or ``ref`` is :obj:`None` in any element, `alis` or `refs` and
    `ref_sizes` will also be :obj:`None`

    Parameters
    ----------
    seq : sequence

    Returns
    -------
    feats : torch.Tensor
    alis : torch.Tensor or None
    refs : torch.Tensor or None
    feat_sizes : torch.Tensor
    ref_sizes : torch.Tensor or None
    """
    seq = sorted(seq, key=lambda x: -x[0].shape[0])
    feats, alis, refs = list(zip(*seq))
    has_ali = all(x is not None for x in alis)
    has_ref = all(x is not None for x in refs)
    feat_sizes = torch.tensor([len(x) for x in feats])
    feats = torch.nn.utils.rnn.pad_sequence(
        feats, padding_value=0, batch_first=batch_first
    )
    if has_ali:
        alis = torch.nn.utils.rnn.pad_sequence(
            alis, padding_value=config.INDEX_PAD_VALUE, batch_first=batch_first
        )
    if has_ref:
        ref_sizes = torch.tensor([len(x) for x in refs])
        refs = torch.nn.utils.rnn.pad_sequence(
            refs, padding_value=config.INDEX_PAD_VALUE, batch_first=batch_first
        )
    return (
        feats,
        alis if has_ali else None,
        refs if has_ref else None,
        feat_sizes,
        ref_sizes if has_ref else None,
    )


class SpectTrainingDataLoader(torch.utils.data.DataLoader):
    """Serves batches of spectral data over random orders of utterances

    Parameters
    ----------
    data_dir : str
    params : SpectDataLoaderParams or DataLoaderParams
        Either provides all the parameters necessary to instantiate this loader (a
        :class:`SpectDataLoaderParams`) or just those related to or just those related
        to the loader (a :class:`DataLoaderParams`). If the latter, `data_params` must
        be specified.
    file_prefix : str, optional
    file_suffix : str, optional
    warn_on_missing : bool, optional
    feat_subdir : str, optional
    ali_subdir : str, optional
    ref_subdir : str, optional
    init_epoch : int, optional
        Where training should resume from.
    batch_first : bool, optional
    data_params : SpectDataParams or :obj:`None`, optional
        If specified, provides the parameters necessary to instantiate the underlying
        :class:`SpectDataSet`. Parameters in `data_params` will pre-empt any found in
        `params`.
    seed : int or :obj:`None`, optional
        The seed used to shuffle data. If unset, will be set randomly.
    kwargs : keyword arguments, optional
        Additional :class:`torch.utils.data.DataLoader` arguments

    Yields
    ------
    feats : torch.Tensor
        A tensor of shape ``(N, T*, F)`` (or ``(T*, N, F)`` if `batch_first` is
        :obj:`False`), where ``N`` is ``params.batch_size``, ``T*`` is the
        maximum number of frames in an utterance in the batch, and ``F`` is the
        number of filters per frame
    alis : torch.Tensor or None
        A long tensor size ``(N, T*)`` (or ``(T*, N)`` if `batch_first` is :obj:`False`)
        if an ``ali/`` dir exists, otherwise :obj:`None`
    refs : torch.Tensor or None
        A long tensor size ``(N, R*[, 3])`` (or ``(R*, N[, 3])`` if `batch_first` is
        :obj:`False`), where ``R*`` is the maximum number of reference tokens
        in the batch. The 3rd dimension will only exist if data were saved with
        frame start/end indices. If the ``refs/`` directory does not exist,
        `refs` and `ref_sizes` are :obj:`None`.
    feat_sizes : torch.Tensor
        A long tensor of shape ``(N,)`` specifying the lengths of utterances in the
        batch
    ref_sizes : torch.Tensor or None
        A long tensor of shape ``(N,)`` specifying the number reference tokens per
        utterance in the batch. If the ``refs/`` directory does not exist, `refs` and
        `ref_sizes` are :obj:`None`.

    Attributes
    ----------
    epoch : int
        The current epoch.

    Notes
    -----

    The first axis of each of `feats`, `alis`, `refs`, `feat_sizes`, and
    `ref_sizes` is ordered by utterances of descending frame lengths. Shorter
    utterances in `feats` are zero-padded to the right, `alis` is padded with
    the module constant :const:`pydrobert.torch.config.INDEX_PAD_VALUE`

    `batch_first` is separated from `params` because it is usually a matter of
    model architecture whether it is :obj:`True` - something the user doesn't
    configure. Further, the attribute `batch_first` can be modified after
    initialization of this loader (outside of the for loop) to suit a model's
    needs.

    Examples
    --------
    Training on alignments for one epoch

    >>> # see 'SpectDataSet' to initialize data set
    >>> num_filts, num_ali_classes = 40, 100
    >>> model = torch.nn.LSTM(num_filts, num_ali_classes)
    >>> optim = torch.optim.Adam(model.parameters())
    >>> loss = torch.nn.CrossEntropyLoss()
    >>> params = SpectDataLoaderParams()
    >>> loader = SpectTrainingDataLoader('data', params, batch_first=False)
    >>> for feats, alis, _, feat_sizes, _ in loader:
    >>>     optim.zero_grad()
    >>>     packed_feats = torch.nn.utils.rnn.pack_padded_sequence(
    ...         feats, feat_sizes)
    >>>     packed_alis = torch.nn.utils.rnn.pack_padded_sequence(
    ...         alis, feat_sizes)
    >>>     packed_logits, _ = model(packed_feats)
    >>>     # no need to unpack: loss is the same as if we ignored padded vals
    >>>     loss(packed_logits.data, packed_alis.data).backward()
    >>>     optim.step()

    Training on reference tokens with CTC for one epoch

    >>> num_filts, num_ref_classes, kern = 40, 2000, 3
    >>> # we use padding to ensure gradients are unaffected by batch padding
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Conv2d(1, 1, kern, padding=(kern - 1) // 2),
    ...     torch.nn.ReLU(),
    ...     torch.nn.Linear(num_filts, num_ref_classes),
    ...     torch.nn.LogSoftmax(-1))
    >>> optim = torch.optim.Adam(model.parameters(), 1e-4)
    >>> loss = torch.nn.CTCLoss()
    >>> params = SpectDataLoaderParams()
    >>> loader = SpectTrainingDataLoader('data', params)
    >>> for feats, _, refs, feat_sizes, ref_sizes in loader:
    >>>     optim.zero_grad()
    >>>     feats = feats.unsqueeze(1)  # channels dim
    >>>     log_prob = model(feats).squeeze(1)
    >>>     loss(
    ...         log_prob.transpose(0, 1),
    ...         refs[..., 0], feat_sizes, ref_sizes).backward()
    >>>     optim.step()
    """

    def __init__(
        self,
        data_dir: str,
        params: DataLoaderParams,
        init_epoch: int = 0,
        file_prefix: str = "",
        file_suffix: str = ".pt",
        warn_on_missing: bool = True,
        feat_subdir: str = "feat",
        ali_subdir: str = "ali",
        ref_subdir: str = "ref",
        batch_first: bool = True,
        data_params: Optional[SpectDataParams] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        for bad_kwarg in (
            "batch_size",
            "sampler",
            "batch_sampler",
            "shuffle",
            "collate_fn",
        ):
            if bad_kwarg in kwargs:
                raise TypeError(
                    'keyword argument "{}" invalid for {} types'.format(
                        bad_kwarg, type(self)
                    )
                )
        self.data_dir = data_dir
        self.params = params
        if data_params is None:
            self.data_params = params
        else:
            self.data_params = data_params
        self.batch_first = batch_first
        self.data_source = SpectDataSet(
            data_dir,
            file_prefix=file_prefix,
            file_suffix=file_suffix,
            warn_on_missing=warn_on_missing,
            subset_ids=set(params.subset_ids) if params.subset_ids else None,
            feat_subdir=feat_subdir,
            ali_subdir=ali_subdir,
            ref_subdir=ref_subdir,
            params=self.data_params,
        )
        if not self.data_source.has_ali and not self.data_source.has_ref:
            raise ValueError(
                "'{}' must have either alignments or reference tokens for "
                "training".format(data_dir)
            )
        epoch_sampler = EpochRandomSampler(
            self.data_source, init_epoch=init_epoch, base_seed=seed
        )
        batch_sampler = torch.utils.data.BatchSampler(
            epoch_sampler, params.batch_size, drop_last=params.drop_last
        )
        super(SpectTrainingDataLoader, self).__init__(
            self.data_source,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            **kwargs
        )

    def collate_fn(
        self,
        seq: Sequence[
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
        ],
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        return spect_seq_to_batch(seq, batch_first=self.batch_first)

    @property
    def epoch(self) -> int:
        return self.batch_sampler.sampler.epoch

    @epoch.setter
    def epoch(self, val: int):
        self.batch_sampler.sampler.epoch = val


class SpectEvaluationDataLoader(torch.utils.data.DataLoader):
    """Serves batches of spectral data over a fixed order of utterances

    Parameters
    ----------
    data_dir : str
    params : DataLoaderParams
        Either provides all the parameters necessary to instantiate this loader (a
        :class:`SpectDataLoaderParams`) or just those related to or just those related
        to the loader (a :class:`DataLoaderParams`). If the latter, `data_params` must
        be specified.
    file_prefix : str, optional
    file_suffix : str, optional
    warn_on_missing : bool, optional
    feat_subdir, ali_subdir, ref_subdir : str, optional
    batch_first : bool, optional
    data_params : SpectDataParams or :obj:`None`, optional
    data_params : SpectDataParams or :obj:`None`, optional
        If specified, provides the parameters necessary to instantiate the underlying
        :class:`SpectDataSet`. Parameters in `data_params` will pre-empt any found in
        `params`.
    kwargs : keyword arguments, optional
        Additional :class:`torch.utils.data.DataLoader` arguments

    Yields
    ------
    feats : torch.Tensor
        A tensor of shape ``(N, T*, F)`` (or ``(T*, N, F)`` if `batch_first` is
        :obj:`False`), where ``N`` is ``params.batch_size``, ``T*`` is the
        maximum number of frames in an utterance in the batch, and ``F`` is the
        number of filters per frame
    alis : torch.Tensor or None
        A long tensor of size ``(N, T*)`` (or ``(T*, N)`` if `batch_first` is
        :obj:`False`) if an ``ali/`` dir exists, otherwise :obj:`None`
    refs : torch.Tensor or None
        A long tensor of size ``(N, R*[, 3])`` (or ``(R*, N[, 3])`` if `batch_first` is
        :obj:`False`), where ``R*`` is the maximum number of reference tokens
        in the batch. The 3rd dimension will only exist if data were saved with
        frame start/end indices. If the ``refs/`` directory does not exist,
        `refs` and `ref_sizes` are :obj:`None`.
    feat_sizes : torch.Tensor
        A long tensor of shape ``(N,)`` specifying the lengths of utterances in the
        batch
    ref_sizes : torch.Tensor or None
        A long tensor of shape ``(N,)`` specifying the number reference tokens per
        utterance in the batch. If the ``refs/`` directory does not exist, `refs` and
        `ref_sizes` are :obj:`None`.
    utt_ids : tuple
        An ``N``-tuple specifying the names of utterances in the batch

    Notes
    -----
    Shorter utterances in `feats` are zero-padded to the right, `alis` and `refs` are
    padded with :const:`pydrobert.torch.config.INDEX_PAD_VALUE`

    Examples
    --------
    Computing class likelihoods and writing them to disk

    >>> # see 'SpectDataSet' to initialize data set
    >>> num_filts, num_ali_classes = 40, 100
    >>> model = torch.nn.LSTM(num_filts, num_ali_classes)
    >>> params = SpectDataLoaderParams()
    >>> loader = SpectEvaluationDataLoader('data', params, batch_first=False)
    >>> for feats, _, _, feat_sizes, _, utt_ids in loader:
    >>>     packed_feats = torch.nn.utils.rnn.pack_padded_sequence(
    ...         feats, feat_sizes)
    >>>     packed_logits, _ = model(packed_feats)
    >>>     logits, _ = torch.nn.utils.rnn.pad_packed_sequence(
    >>>         packed_logits, batch_first=True)
    >>>     log_probs = torch.nn.functional.log_softmax(logits, -1)
    >>>     for pdf, feat_size, utt_id in zip(log_probs, feat_sizes, utt_ids):
    >>>         loader.data_source.write_pdf(utt_id, pdf[:feat_size])

    Transcribing utterances with CTC

    >>> num_filts, num_ref_classes, kern = 40, 2000, 3
    >>> # we use padding to ensure gradients are unaffected by batch padding
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Conv2d(1, 1, kern, padding=(kern - 1) // 2),
    ...     torch.nn.ReLU(),
    ...     torch.nn.Linear(num_filts, num_ref_classes),
    ...     torch.nn.LogSoftmax(-1)).eval()
    >>> params = SpectDataLoaderParams()
    >>> loader = SpectEvaluationDataLoader('data', params)
    >>> for feats, _, _, feat_sizes, _, utt_ids in loader:
    >>>     feats = feats.unsqueeze(1)  # channels dim
    >>>     log_prob = model(feats).squeeze(1)
    >>>     paths = log_prob.argmax(-1)  # best path decoding
    >>>     for path, feat_size, utt_id in zip(paths, feat_sizes, utt_ids):
    >>>         path = path[:feat_size]
    >>>         pathpend = torch.cat([torch.tensor([-1]), path])
    >>>         path = path.masked_select(
    ...             (path != 0) & (path - pathpend[:-1] != 0))
    >>>         hyp = torch.stack(
    ...             [path] + [torch.full_like(path, INDEX_PAD_VALUE)] * 2)
    >>>         loader.data_source.write_hyp(utt_id, hyp)
    """

    class SEvalDataSet(SpectDataSet):
        """Append utt_id to each sample's tuple"""

        def __getitem__(
            self, idx: int
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], str]:
            feat, ali, ref = super(
                SpectEvaluationDataLoader.SEvalDataSet, self
            ).__getitem__(idx)
            utt_id = self.utt_ids[idx]
            return feat, ali, ref, utt_id

    def eval_collate_fn(
        self,
        seq: Sequence[
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], str]
        ],
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        Sequence[str],
    ]:
        """Update context_window_seq_to_batch to handle feat_sizes, utt_ids"""
        feats, alis, refs, utt_ids = list(zip(*seq))
        # spect_seq_to_batch sorts by descending number of frames, so we
        # sort utt_ids here
        utt_ids = tuple(
            x[1] for x in sorted(zip(feats, utt_ids), key=lambda x: -x[0].shape[0])
        )
        feats, alis, refs, feat_sizes, ref_sizes = spect_seq_to_batch(
            list(zip(feats, alis, refs)), batch_first=self.batch_first
        )
        return feats, alis, refs, feat_sizes, ref_sizes, utt_ids

    def __init__(
        self,
        data_dir: str,
        params: DataLoaderParams,
        file_prefix: str = "",
        file_suffix: str = ".pt",
        warn_on_missing: bool = True,
        feat_subdir: str = "feat",
        ali_subdir: str = "ali",
        ref_subdir: str = "ref",
        batch_first: bool = True,
        data_params: Optional[SpectDataParams] = None,
        **kwargs
    ):
        for bad_kwarg in (
            "batch_size",
            "sampler",
            "batch_sampler",
            "shuffle",
            "collate_fn",
        ):
            if bad_kwarg in kwargs:
                raise TypeError(
                    'keyword argument "{}" invalid for {} types'.format(
                        bad_kwarg, type(self)
                    )
                )
        self.data_dir = data_dir
        self.params = params
        if data_params is None:
            self.data_params = params
        else:
            self.data_params = data_params
        self.batch_first = batch_first
        self.data_source = self.SEvalDataSet(
            data_dir,
            file_prefix=file_prefix,
            file_suffix=file_suffix,
            warn_on_missing=warn_on_missing,
            subset_ids=set(params.subset_ids) if params.subset_ids else None,
            feat_subdir=feat_subdir,
            ali_subdir=ali_subdir,
            ref_subdir=ref_subdir,
            params=self.data_params,
        )
        super(SpectEvaluationDataLoader, self).__init__(
            self.data_source,
            batch_size=params.batch_size,
            shuffle=False,
            collate_fn=self.eval_collate_fn,
            **kwargs
        )


def context_window_seq_to_batch(
    seq: Sequence[Tuple[torch.Tensor, Optional[torch.Tensor]]]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Convert a sequence of context window elements to a batch

    Assume `seq` is a finite length sequence of pairs of ``window, ali``, where
    ``window`` is of size ``(T, C, F)``, where ``T`` is some number of windows
    (which can vary across elements in the sequence), ``C`` is the window size,
    and ``F`` is some number filters, and ``ali`` is of size ``(T,)``. This
    method batches all the elements of the sequence into a pair of ``windows,
    alis``, where `windows` and `alis` will have shapes ``(N, C, F)`` and
    ``(N,)`` resp., where :math:`N = \sum T` is the total number of context
    windows over the utterances.

    If ``ali`` is :obj:`None` in any element, `alis` will also be :obj:`None`

    Parameters
    ----------
    seq : sequence

    Returns
    -------
    windows : list
    alis : list
    """
    windows = []
    batch_ali = []
    for window, ali in seq:
        windows.append(window)
        if ali is None:
            # assume every remaining ali will be none
            batch_ali = None
        else:
            batch_ali.append(ali)
    windows = torch.cat(windows)
    if batch_ali is not None:
        batch_ali = torch.cat(batch_ali)
    return windows, batch_ali


class ContextWindowDataLoaderParams(ContextWindowDataParams, DataLoaderParams):
    """Parameters for a ContextWindow*DataLoader

    This implements the :class:`pydrobert.param.optuna.TunableParameterized`
    interface

    See Also
    --------
    pydrobert.torch.data.ContextWindowTrainingDataLoader
    pydrobert.torch.data.ContextWindowEvaluationDataLoader
        Where to use these parameters.
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


class ContextWindowTrainingDataLoader(torch.utils.data.DataLoader):
    """Serve batches of context windows over a random order of utterances

    Parameters
    ----------
    data_dir : str
    params : ContextWindowDataLoaderParams or DataLoaderParams
        Either provides all the parameters necessary to instantiate this loader (a
        :class:`ContextWindowDataLoaderParams`) or just those related to the loader
        (a :class:`DataLoaderParams`). If the latter, `data_params` must be specified.
    file_prefix : str, optional
    file_suffix : str, optional
    warn_on_missing : bool, optional
    feat_subdir, ali_subdir : str, optional
    init_epoch : int, optional
        Where training should resume from
    data_params : ContextWindowDataParams or :obj:`None`, optional
        If specified, provides the parameters necessary to instantiate the underlying
        :class:`ContextWindowDataSet`. Parameters in `data_params` will pre-empt any
        found in `params`.
    seed : int or :obj:`None`, optional
        The seed used to shuffle data. If :obj:`None`, `params` is checked for
        a `seed` parameter or, if none is found, one will be generated randomly
    kwargs : keyword arguments, optional
        Additional :class:`torch.utils.data.DataLoader` arguments

    Attributes
    ----------
    epoch : int
        The current epoch.

    Yields
    ------
    windows : torch.Tensor
        A tensor of size ``(N, C, F)``, where ``N`` is the total number of context
        windows over all utterances in the batch, ``C`` is the context window
        size, and ``F`` is the number of filters per frame
    alis : torch.Tensor or None
        A long tensor of size ``(N,)`` (or :obj:`None` if the ``ali`` dir was not
        specified)

    Examples
    --------
    Training on alignments for one epoch

    >>> # see 'SpectDataSet' to initialize data set
    >>> num_filts, num_ali_classes, left, right = 40, 100, 4, 4
    >>> window_width = left + right + 1
    >>> model = torch.torch.nn.Linear(
    ...     num_filts * window_width, num_ali_classes)
    >>> optim = torch.optim.Adam(model.parameters())
    >>> loss = torch.nn.CrossEntropyLoss()
    >>> params = ContextWindowDataLoaderParams(
    ...     context_left=left, context_right=right)
    >>> loader = ContextWindowTrainingDataLoader('data', params)
    >>> for windows, alis in loader:
    >>>     optim.zero_grad()
    >>>     windows = windows.view(-1, num_filts * window_width)  # flatten win
    >>>     logits = model(windows)
    >>>     loss(logits, alis).backward()
    >>>     optim.step()
    """

    def __init__(
        self,
        data_dir: str,
        params: DataLoaderParams,
        init_epoch: int = 0,
        file_prefix: str = "",
        file_suffix: str = ".pt",
        warn_on_missing: bool = True,
        feat_subdir: str = "feat",
        ali_subdir: str = "ali",
        data_params: Optional[ContextWindowDataParams] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        for bad_kwarg in (
            "batch_size",
            "sampler",
            "batch_sampler",
            "shuffle",
            "collate_fn",
        ):
            if bad_kwarg in kwargs:
                raise TypeError(
                    'keyword argument "{}" invalid for {} types'.format(
                        bad_kwarg, type(self)
                    )
                )
        self.data_dir = data_dir
        self.params = params
        if seed is None and hasattr(params, "seed"):
            seed = params.seed
        if data_params is None:
            self.data_params = params
        else:
            self.data_params = data_params
        self.data_source = ContextWindowDataSet(
            data_dir,
            file_prefix=file_prefix,
            file_suffix=file_suffix,
            warn_on_missing=warn_on_missing,
            subset_ids=set(params.subset_ids) if params.subset_ids else None,
            feat_subdir=feat_subdir,
            ali_subdir=ali_subdir,
            params=self.data_params,
        )
        if not self.data_source.has_ali:
            raise ValueError(
                "'{}' must have alignment info for training".format(data_dir)
            )
        epoch_sampler = EpochRandomSampler(
            self.data_source, init_epoch=init_epoch, base_seed=seed
        )
        batch_sampler = torch.utils.data.BatchSampler(
            epoch_sampler, params.batch_size, drop_last=params.drop_last
        )
        super(ContextWindowTrainingDataLoader, self).__init__(
            self.data_source,
            batch_sampler=batch_sampler,
            collate_fn=context_window_seq_to_batch,
            **kwargs
        )

    @property
    def epoch(self) -> int:
        return self.batch_sampler.sampler.epoch

    @epoch.setter
    def epoch(self, val: int):
        self.batch_sampler.sampler.epoch = val


class ContextWindowEvaluationDataLoader(torch.utils.data.DataLoader):
    """Serves batches of context windows over sequential utterances

    Parameters
    ----------
    data_dir : str
    params : ContextWindowDataLoaderParams or DataLoaderParams
        Either provides all the parameters necessary to instantiate this loader (a
        :class:`ContextWindowDataLoaderParams`) or just those related to the loader
        (a :class:`DataLoaderParams`). If the latter, `data_params` must be specified.
    file_prefix : str, optional
    file_suffix : str, optional
    warn_on_missing : bool, optional
    feat_subdir, ali_subdir : str, optional
    data_params : ContextWindowDataParams or :obj:`None`, optional
        If specified, provides the parameters necessary to instantiate the underlying
        :class:`ContextWindowDataSet`. Parameters in `data_params` will pre-empt any
        found in `params`.
    kwargs : keyword arguments, optional
        Additional :class:`torch.utils.data.DataLoader` arguments

    Yields
    ------
    windows : torch.Tensor
        A tensor of size ``(N, C, F)``, where ``N`` is the number of context
        windows, ``C`` is the context window size, and ``F`` is the number of filters
        per frame
    alis : torch.Tensor or None
        A long tensor of size ``(N,)`` (or :obj:`None` if the ``ali`` dir was not
        specified)
    win_sizes : torch.Tensor
        A long tensor of shape ``(params.batch_size,)`` specifying the number of context
        windows per utterance in the batch.
        ``windows[sum(win_sizes[:i]):sum(win_sizes[:i+1])]`` are the context windows for
        the ``i``-th utterance in the batch (``sum(win_sizes) == N``)
    utt_ids : tuple
        `utt_ids` is a tuple of size ``params.batch_size`` naming the
        utterances in the batch

    Examples
    --------
    Computing class likelihoods and writing them to disk

    >>> # see 'SpectDataSet' to initialize data set
    >>> num_filts, num_ali_classes, left, right = 40, 100, 4, 4
    >>> window_width = left + right + 1
    >>> model = torch.torch.nn.Linear(
    ...     num_filts * window_width, num_ali_classes).eval()
    >>> params = ContextWindowDataLoaderParams(
    ...     context_left=left, context_right=right)
    >>> loader = ContextWindowEvaluationDataLoader('data', params)
    >>> for windows, _, win_sizes, utt_ids in loader:
    >>>     windows = windows.view(-1, num_filts * window_width)  # flatten win
    >>>     logits = model(windows)
    >>>     log_probs = torch.nn.functional.log_softmax(logits, -1)
    >>>     for win_size, utt_id in zip(win_sizes, utt_ids):
    >>>         assert log_probs[:win_size].shape[0] == win_size
    >>>         loader.data_source.write_pdf(utt_id, log_probs[:win_size])
    >>>         log_probs = log_probs[win_size:]
    """

    class CWEvalDataSet(ContextWindowDataSet):
        """Append feat_size and utt_id to each sample's tuple"""

        def __getitem__(
            self, idx: int
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, str]:
            window, ali = super(
                ContextWindowEvaluationDataLoader.CWEvalDataSet, self
            ).__getitem__(idx)
            win_size = window.size()[0]
            utt_id = self.utt_ids[idx]
            return window, ali, win_size, utt_id

    @staticmethod
    def eval_collate_fn(
        seq: Tuple[
            Sequence[torch.Tensor],
            Sequence[Optional[torch.Tensor]],
            Sequence[int],
            Sequence[str],
        ]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Sequence[str]]:
        """Update context_window_seq_to_batch to handle feat_sizes, utt_ids"""
        windows, alis, feat_sizes, utt_ids = list(zip(*seq))
        windows, alis = context_window_seq_to_batch(list(zip(windows, alis)))
        return (windows, alis, torch.tensor(feat_sizes), tuple(utt_ids))

    def __init__(
        self,
        data_dir: str,
        params: DataLoaderParams,
        file_prefix: str = "",
        file_suffix: str = ".pt",
        warn_on_missing: bool = True,
        feat_subdir: str = "feat",
        ali_subdir: str = "ali",
        data_params: Optional[ContextWindowDataParams] = None,
        **kwargs
    ):
        for bad_kwarg in (
            "batch_size",
            "sampler",
            "batch_sampler",
            "shuffle",
            "collate_fn",
        ):
            if bad_kwarg in kwargs:
                raise TypeError(
                    'keyword argument "{}" invalid for {} types'.format(
                        bad_kwarg, type(self)
                    )
                )
        self.data_dir = data_dir
        self.params = params
        if data_params is None:
            self.data_params = params
        else:
            self.data_params = data_params
        self.data_source = self.CWEvalDataSet(
            data_dir,
            file_prefix=file_prefix,
            file_suffix=file_suffix,
            warn_on_missing=warn_on_missing,
            subset_ids=set(params.subset_ids) if params.subset_ids else None,
            feat_subdir=feat_subdir,
            ali_subdir=ali_subdir,
            params=self.data_params,
        )
        super(ContextWindowEvaluationDataLoader, self).__init__(
            self.data_source,
            batch_size=params.batch_size,
            shuffle=False,
            collate_fn=self.eval_collate_fn,
            **kwargs
        )
