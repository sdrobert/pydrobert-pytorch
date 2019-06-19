'''Classes and functions related to storing/retrieving data'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

import numpy as np
import torch
import torch.utils.data
import param

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"
__all__ = [
    'extract_window',
    'SpectDataSet',
    'validate_spect_data_set',
    'ContextWindowDataSet',
    'EpochRandomSampler',
    'context_window_seq_to_batch',
    'DataSetParams',
    'SpectDataParams',
    'ContextWindowDataParams',
    'SpectDataSetParams',
    'ContextWindowDataSetParams',
    'ContextWindowTrainingDataLoader',
    'ContextWindowEvaluationDataLoader',
]


def extract_window(signal, frame_idx, left, right, reverse=False):
    '''Slice the signal to extract a context window

    Parameters
    ----------
    signal : torch.Tensor
        Of shape ``(T, F)``, where ``T`` is the time/frame axis, and ``F``
        is the frequency axis
    frame_idx : int
        The "center frame" ``0 <= frame_idx < T``
    left : int
        The number of frames in the window to the left (before) the center
        frame. Any frames below zero are edge-padded
    right : int
        The number of frames in the window to the right (after) the center
        frame. Any frames above ``T`` are edge-padded
    reverse : bool, optional
        If ``True``, flip the window along the time/frame axis

    Returns
    -------
    window : torch.Tensor
        Of shape ``(1 + left + right, F)``
    '''
    T, F = signal.shape
    if frame_idx - left < 0 or frame_idx + right + 1 > T:
        win_size = 1 + left + right
        window = signal.new(win_size, F)
        left_pad = max(left - frame_idx, 0)
        right_pad = max(frame_idx + right + 1 - T, 0)
        window[left_pad:win_size - right_pad] = signal[
            max(0, frame_idx - left):frame_idx + right + 1]
        if left_pad:
            window[:left_pad] = signal[0]
        if right_pad:
            window[-right_pad:] = signal[-1]
    else:
        window = signal[frame_idx - left:frame_idx + right + 1]
    if reverse:
        window = torch.flip(window, [0])
    return window


class SpectDataSet(torch.utils.data.Dataset):
    '''Accesses spectrographic filter data stored in a data directory

    ``SpectDataSet`` assumes that `data_dir` is structured as

    ::
        data_dir/
            feats/
                <file_prefix><utt_ids[0]><file_suffix>
                <file_prefix><utt_ids[1]><file_suffix>
                ...
            [
            ali/
                <file_prefix><utt_ids[0]><file_suffix>
                <file_prefix><utt_ids[1]><file_suffix>
                ...
            ]
            [
            ref/
                <file_prefix><utt_ids[0]><file_suffix>
                <file_prefix><utt_ids[1]><file_suffix>
                ...
            ]

    The ``feats`` dir stores filter bank data in the form of
    ``torch.FloatTensor``s of size ``(T, F)``, where ``T`` is the time
    dimension and ``F`` is the filter/log-frequency dimension. ``feats`` is
    the only required directory.

    ``ali`` stores ``torch.LongTensor``s of size ``(T,)``, indicating the
    pdf-id of the most likely target. ``ali`` is suitable for discriminative
    training of DNNs in hybrid DNN-HMM recognition, or any frame-wise loss.
    ``ali/`` is optional.

    ``ref`` stores ``torch.LongTensor``s of size ``(R,3)``, indicating
    reference transcriptions. Letting ``r`` be such a tensor, ``r[..., 0]``
    is the sequence of token ids for the utterance and ``r[..., 1:]`` are
    the 0-indexed frames they start (inclusive) and end (exclusive) at,
    respectively. Negative values can be used when the start and end frames are
    unknown. ``ref`` is suitable for end-to-end training. ``ref/`` is optional.

    Parameters
    ----------
    data_dir : str
        A path to feature directory
    file_prefix : str, optional
        The prefix that indicates that the file counts toward the data set
    file_suffix : str, optional
        The suffix that indicates that the file counts toward the data set
    warn_on_missing : bool, optional
        If ``ali/`` or ``ref/`` exist, there's a mismatch between the
        utterances in the directories, and `warn_on_missing` is ``True``, a
        warning will be issued (via ``warnings``) regarding each such mismatch
    subset_ids : set, optional
        If set, only utterances with ids listed in this set will count towards
        the data set. The rest will be ignored
    feats_subdir, ali_subdir, ref_subdir : str, optional
        Change the names of the subdirectories under which feats, alignments,
        and references are stored. If `ali_subdir` or `ref_subdir` is ``None``,
        they will not be searched for

    Attributes
    ----------
    data_dir : str
    feats_subdir : str
    ali_subdir : str
    ref_subdir : str
    file_suffix : str
    has_ali : bool
        Whether alignment data exist
    has_ref : bool
        Whether reference data exist
    utt_ids : tuple
        A tuple of all utterance ids extracted from the data directory. They
        are stored in the same order as features and alignments via
        ``__getitem__``. If the ``ali/`` or ``ref/`` directories exist,
        `utt_ids` contains only the utterances in the intersection of each
        directory (and `subset_ids`, if it was specified)

    Yields
    ------
    feat, ali, ref : tuple
    For the i-th yielded item, `feat` corresponds to the features at
    ``utt_ids[i]``, `ali` the alignments, and `ref` the reference sequence.
    If ``ali/`` or ``ref/`` did not exist on initialization, `ali` or `ref`
    will be ``None``
    '''

    def __init__(
            self, data_dir, file_prefix='', file_suffix='.pt',
            warn_on_missing=True, subset_ids=None,
            feats_subdir='feats', ali_subdir='ali', ref_subdir='ref'):
        super(SpectDataSet, self).__init__()
        self.data_dir = data_dir
        self.feats_subdir = feats_subdir
        self.ali_subdir = ali_subdir
        self.ref_subdir = ref_subdir
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        if ali_subdir:
            self.has_ali = os.path.isdir(os.path.join(data_dir, ali_subdir))
        else:
            self.has_ali = False
        if ref_subdir:
            self.has_ref = os.path.isdir(os.path.join(data_dir, ref_subdir))
        else:
            self.has_ref = False
        if self.has_ali:
            self.has_ali = any(
                x.startswith(file_prefix) and x.endswith(file_suffix)
                for x in os.listdir(os.path.join(data_dir, ali_subdir))
            )
        if self.has_ref:
            self.has_ref = any(
                x.startswith(file_prefix) and x.endswith(file_suffix)
                for x in os.listdir(os.path.join(data_dir, ref_subdir))
            )
        self.utt_ids = tuple(sorted(
            self.find_utt_ids(warn_on_missing, subset_ids=subset_ids)))

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        return self.get_utterance_tuple(idx)

    def find_utt_ids(self, warn_on_missing, subset_ids=None):
        '''Returns a set of all utterance ids from data_dir'''
        neg_fsl = -len(self.file_suffix)
        if not neg_fsl:
            neg_fsl = None
        fpl = len(self.file_prefix)
        utt_ids = set(
            os.path.basename(x)[fpl:neg_fsl]
            for x in os.listdir(os.path.join(self.data_dir, self.feats_subdir))
            if x.startswith(self.file_prefix) and x.endswith(self.file_suffix)
        )
        if subset_ids is not None:
            utt_ids &= subset_ids
        if self.has_ali:
            ali_utt_ids = set(
                os.path.basename(x)[fpl:neg_fsl]
                for x in os.listdir(
                    os.path.join(self.data_dir, self.ali_subdir))
                if x.startswith(self.file_prefix) and
                x.endswith(self.file_suffix)
            )
            if subset_ids is not None:
                ali_utt_ids &= subset_ids
            if warn_on_missing:
                for utt_id in utt_ids.difference(ali_utt_ids):
                    warnings.warn("Missing ali for uttid: '{}'".format(utt_id))
                for utt_id in ali_utt_ids.difference(utt_ids):
                    warnings.warn(
                        "Missing feats for uttid: '{}'".format(utt_id))
        if self.has_ref:
            ref_utt_ids = set(
                os.path.basename(x)[fpl:neg_fsl]
                for x in os.listdir(
                    os.path.join(self.data_dir, self.ref_subdir))
                if x.startswith(self.file_prefix) and
                x.endswith(self.file_suffix)
            )
            if subset_ids is not None:
                ref_utt_ids &= subset_ids
            if warn_on_missing:
                for utt_id in utt_ids.difference(ref_utt_ids):
                    warnings.warn("Missing ref for uttid: '{}'".format(utt_id))
                for utt_id in ref_utt_ids.difference(utt_ids):
                    warnings.warn(
                        "Missing feats for uttid: '{}'".format(utt_id))
        if self.has_ali:
            utt_ids &= ali_utt_ids
        if self.has_ref:
            utt_ids &= ref_utt_ids
        return utt_ids

    def get_utterance_tuple(self, idx):
        '''Get a tuple of features, alignments, and references'''
        utt_id = self.utt_ids[idx]
        feats = torch.load(
            os.path.join(
                self.data_dir,
                self.feats_subdir,
                self.file_prefix + utt_id + self.file_suffix))
        if self.has_ali:
            ali = torch.load(
                os.path.join(
                    self.data_dir,
                    self.ali_subdir,
                    self.file_prefix + utt_id + self.file_suffix))
        else:
            ali = None
        if self.has_ref:
            ref = torch.load(
                os.path.join(
                    self.data_dir,
                    self.ref_subdir,
                    self.file_prefix + utt_id + self.file_suffix))
        else:
            ref = None
        return feats, ali, ref

    def write_pdf(self, utt, pdf, pdfs_dir=None):
        '''Write a pdf FloatTensor to the data directory

        This method writes a pdf matrix to the directory `pdfs_dir`
        with the name ``<file_prefix><utt><file_suffix>``

        Parameters
        ----------
        utt : str or int
            The name of the utterance to write. If an integer is specified,
            `utt` is assumed to index an utterance id specified in
            ``self.utt_ids``
        pdf : torch.Tensor
            The tensor to write. It will be converted to a ``FloatTensor``
            using the command ``pdf.cpu().float()``
        pdfs_dir : str or None, optional
            The directory pdfs are written to. If ``None``, it will be set to
            ``self.data_dir + '/pdfs'``
        '''
        if isinstance(utt, int):
            utt = self.utt_ids[utt]
        if pdfs_dir is None:
            pdfs_dir = os.path.join(self.data_dir, 'pdfs')
        if not os.path.isdir(pdfs_dir):
            os.makedirs(pdfs_dir)
        torch.save(
            pdf.cpu().float(),
            os.path.join(pdfs_dir, self.file_prefix + utt + self.file_suffix)
        )

    def write_hyp(self, utt, hyp, hyp_dir=None):
        '''Write hypothesis LongTensor to the data directory

        This method writes a sequence of hypothesis tokens to the directory
        `hyp_dir` with the name ``<file_prefix><utt><file_suffix>``

        Parameters
        ----------
        utt : str or int
            The name of the utterance to write. If an integer is specified,
            `utt` is assumed to index an utterance id specified in
            ``self.utt_ids``
        hyp : torch.Tensor
            The tensor to write. It will be converted to a ``LongTensor``
            using the command ``pdf.cpu().long()``
        hyp_dir : str or None, optional
            The directory pdfs are written to. If ``None``, it will be set to
            ``self.data_dir + '/hyp'``
        '''
        if isinstance(utt, int):
            utt = self.utt_ids[utt]
        if hyp_dir is None:
            hyp_dir = os.path.join(self.data_dir, 'hyp')
        if not os.path.isdir(hyp_dir):
            os.makedirs(hyp_dir)
        torch.save(
            hyp.cpu().long(),
            os.path.join(hyp_dir, self.file_prefix + utt + self.file_suffix)
        )


def validate_spect_data_set(data_set):
    '''Validate SpectDataSet data directory

    The data directory is valid if the following conditions are observed

     1. All features are ``FloatTensor`` instances
     2. All features have two axes
     3. All features have the same size second axis
     4. If alignments are present
        1. All alignments are ``LongTensor`` instances
        2. All alignments have one axis
        3. Features and alignments have the same size first axes for a given
           utterance id
     5. If reference sequences are present
        1. All references are ``LongTensor`` instances
        2. All alignments have two axes, the second of size 3
        3. For the start and end points of a reference token, ``r[i, 1:]``,
           either both of them are negative (indicating no alignment), or
           ``0 <= r[i, 1] < r[i, 2] <= T``, where ``T`` is the number of
           frames in the utterance. We do not enforce tokens be
           non-overlapping

    Raises a ``ValueError`` if a condition is violated
    '''
    num_filts = None
    for idx in range(len(data_set.utt_ids)):
        feats, ali, ref = data_set.get_utterance_tuple(idx)
        if not isinstance(feats, torch.FloatTensor):
            raise ValueError(
                "'{}' (index {}) in '{}' is not a FloatTensor".format(
                    data_set.utt_ids[idx] + data_set.file_suffix, idx,
                    os.path.join(data_set.data_dir, data_set.feats_subdir)))
        if len(feats.size()) != 2:
            raise ValueError(
                "'{}' (index {}) in '{}' does not have two axes".format(
                    data_set.utt_ids[idx] + data_set.file_suffix, idx,
                    os.path.join(data_set.data_dir, data_set.feats_subdir)
                ))
        if num_filts is None:
            num_filts = feats.shape[1]
        elif feats.shape[1] != num_filts:
            raise ValueError(
                "'{}' (index {}) in '{}' has second axis size {}, which "
                "does not match prior utterance ('{}') size of {}".format(
                    data_set.utt_ids[idx] + data_set.file_suffix, idx,
                    os.path.join(data_set.data_dir, data_set.feats_subdir),
                    feats.shape[1],
                    data_set.utt_ids[idx - 1] + data_set.file_suffix,
                    num_filts))
        if ali is not None:
            if not isinstance(ali, torch.LongTensor):
                raise ValueError(
                    "'{}' (index {}) in '{}' is not a LongTensor".format(
                        data_set.utt_ids[idx] + data_set.file_suffix, idx,
                        os.path.join(data_set.data_dir, data_set.ali_subdir)))
            if len(ali.shape) != 1:
                raise ValueError(
                    "'{}' (index {}) in '{}' does not have one axis".format(
                        data_set.utt_ids[idx] + data_set.file_suffix, idx,
                        os.path.join(data_set.data_dir, data_set.ali_subdir)))
            if ali.shape[0] != feats.shape[0]:
                raise ValueError(
                    "'{}' (index {}) in '{}' does not have the same first axis"
                    " size ({}) as it's companion in '{}' ({})".format(
                        data_set.utt_ids[idx] + data_set.file_suffix, idx,
                        os.path.join(data_set.data_dir, data_set.feats_subdir),
                        feats.shape[0],
                        os.path.join(data_set.data_dir, data_set.ali_subdir),
                        ali.shape[0]))
        if ref is not None:
            if not isinstance(ref, torch.LongTensor):
                raise ValueError(
                    "'{}' (index {}) in '{}' is not a LongTensor".format(
                        data_set.utt_ids[idx] + data_set.file_suffix, idx,
                        os.path.join(data_set.data_dir, data_set.ref_subdir)))
            if len(ref.shape) != 2 or ref.shape[1] != 3:
                raise ValueError(
                    "'{}' (index {}) in '{}' does not have shape (D, 3)"
                    "".format(
                        data_set.utt_ids[idx] + data_set.file_suffix, idx,
                        os.path.join(data_set.data_dir, data_set.ref_subdir)))
            for idx2, r in enum(ref):
                if not (r[1] < 0 and r[2] < 0) or not (
                        0 <= r[1] < r[2] < feats.shape[0]):
                    raise ValueError(
                        "'{}' (index {}) in '{}', has a reference token "
                        "(index {}) with bounds outside the utterance"
                        "".format(
                            data_set.utt_ids[idx] + data_set.file_suffix, idx,
                            os.path.join(
                                data_set.data_dir, data_set.ref_subdir),
                            idx2))


class ContextWindowDataSet(SpectDataSet):
    '''SpectDataSet, extracting fixed-width windows over the utterance

    Like a ``SpectDataSet``, ``ContextWindowDataSet`` indexes tuples of
    features and alignments. Instead of returning features of shape ``(T, F)``,
    instances return features of shape ``(T, 1 + left + right, F)``, where the
    ``T`` axis indexes the so-called center frame and the ``1 + left + right``
    axis contains frame vectors (size ``F``) including the center frame,
    ``left`` frames in time before the center frame, and ``right`` frames
    after.

    ``ContextWindowDataSet`` does not have support for reference token
    subdirectories as it is unclear how to always pair tokens with context
    windows. If tokens are one-to-one with frames, it is suggested that
    ``ali/`` be re-used for this purpose.

    Parameters
    ----------
    data_dir : str
    left : int
    right : int
    file_prefix : str, optional
    file_suffix : str, optional
    warn_on_missing : bool, optional
    feats_subdir, ali_subdir : str, optional
    reverse : bool, optional
        If ``True``, context windows will be reversed along the time
        dimension

    Attributes
    ----------
    data_dir : str
    left : int
    right : int
    has_ali : bool
    has_ref : bool
        Always ``False``
    utt_ids : tuple
    reverse : bool

    Yields
    ------
    windowed, ali : tuple
    '''

    def __init__(
            self, data_dir, left, right, file_prefix='',
            file_suffix='.pt', warn_on_missing=True, subset_ids=None,
            feats_subdir='feats', ali_subdir='ali', reverse=False):
        super(ContextWindowDataSet, self).__init__(
            data_dir, file_prefix=file_prefix, file_suffix=file_suffix,
            warn_on_missing=warn_on_missing, subset_ids=subset_ids,
            feats_subdir=feats_subdir, ali_subdir=ali_subdir,
            ref_subdir=None)
        self.left = left
        self.right = right
        self.reverse = reverse

    def get_utterance_tuple(self, idx):
        '''Get a tuple of features and alignments'''
        return super(ContextWindowDataSet, self).get_utterance_tuple(idx)[:2]

    def get_windowed_utterance(self, idx):
        '''Get pair of features (w/ context window) and alignments'''
        feats, ali = self.get_utterance_tuple(idx)
        num_frames, num_filts = feats.shape
        windowed = torch.empty(
            num_frames, 1 + self.left + self.right, num_filts)
        for center_frame in range(num_frames):
            windowed[center_frame] = extract_window(
                feats, center_frame, self.left, self.right,
                reverse=self.reverse)
        return windowed, ali

    def __getitem__(self, idx):
        return self.get_windowed_utterance(idx)


class EpochRandomSampler(torch.utils.data.Sampler):
    '''Return random samples that are the same for a fixed epoch

    Parameters
    ----------
    data_source : torch.data.utils.Dataset
        The total number of samples
    init_epoch : int, optional
        The initial epoch
    base_seed : int, optional
        Determines the starting seed of the sampler. Sampling is seeded with
        ``base_seed + epoch``. If unset, a seed is randomly generated from
        the default generator

    Attributes
    ----------
    epoch : int
        The current epoch. Responsible for seeding the upcoming samples
    data_source : torch.data.utils.Dataset
    base_seed : int

    Examples
    --------
    >>> sampler = EpochRandomSampler(
    ...     torch.data.utils.TensorDataset(torch.arange(100)))
    >>> samples_ep0 = tuple(sampler)  # random
    >>> samples_ep1 = tuple(sampler)  # random, probably not same as first
    >>> assert tuple(sampler.get_samples_for_epoch(0)) == samples_ep0
    >>> assert tuple(sampler.get_samples_for_epoch(1)) == samples_ep1
    '''

    def __init__(self, data_source, init_epoch=0, base_seed=None):
        super(EpochRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.epoch = init_epoch
        if base_seed is None:
            base_seed = np.random.randint(np.iinfo(np.uint32).max)
        self.base_seed = base_seed

    def __len__(self):
        return len(self.data_source)

    def get_samples_for_epoch(self, epoch):
        '''tuple : samples for a specific epoch'''
        rs = np.random.RandomState(self.base_seed + epoch)
        return rs.permutation(range(len(self.data_source)))

    def __iter__(self):
        ret = iter(self.get_samples_for_epoch(self.epoch))
        self.epoch += 1
        return ret


def context_window_seq_to_batch(seq):
    r'''Convert a sequence of context window elements to a batch

    Assume `seq` is a fixed length sequence of pairs of ``feats, ali``,
    where ``feats`` is of size ``(T, C, F)``, where ``T`` is some number of
    windows (which can vary across elements in the sequence), ``C`` is the
    window size, and ``F`` is some number filters, and ``ali`` is of
    size ``(T,)``, and ``ref`` is of size ``(R, 3)``. This method batches
    all the elements of the sequence into a pair of ``batch_feats,
    batch_ali``, where `batch_feats` and `batch_ali` will have shapes
    ``(N, C, F)`` and ``(N,)`` resp., where :math:`N = \sum T` is the total
    number of context windows over the utterances.

    If ``ali`` is ``None`` in any element, `batch_ali` will also be ``None``

    Parameters
    ----------
    seq : sequence

    Returns
    -------
    batch_feats, batch_ali : tuple
    '''
    batch_feats = []
    batch_ali = []
    for feats, ali in seq:
        batch_feats.append(feats)
        if ali is None:
            # assume every remaining ali will be none
            batch_ali = None
        else:
            batch_ali.append(ali)
    batch_feats = torch.cat(batch_feats)
    if batch_ali is not None:
        batch_ali = torch.cat(batch_ali)
    return batch_feats, batch_ali


class DataSetParams(param.Parameterized):
    '''General parameters for a single partition of data'''
    batch_size = param.Integer(
        10, bounds=(1, None),
        doc='Number of elements in a batch, which equals the number of '
        'utterances in the batch'
    )
    seed = param.Integer(
        None,
        doc='The seed used to shuffle data. The seed is incremented at every '
        'epoch'
    )
    drop_last = param.Boolean(
        False,
        doc='Whether to drop the last batch if it does reach batch_size'
    )
    subset_ids = param.List(
        [], class_=str, bounds=None,
        doc='A list of utterance ids. If non-empty, the data set will be '
        'restricted to these utterances'
    )


class SpectDataParams(param.Parameterized):
    '''Parameters for spectral data'''
    pass


class ContextWindowDataParams(SpectDataParams):
    # context windows are more model parameters than data parameters, but
    # we're going to extract them as part of the data loading process, which
    # is easily parallelized by the DataLoader
    context_left = param.Integer(
        4, bounds=(0, None), softbounds=(3, 8),
        doc='How many frames to the left of (before) the current frame are '
        'included when determining the class of the current frame'
    )
    context_right = param.Integer(
        4, bounds=(0, None), softbounds=(3, 8),
        doc='How many frames to the right of (after) the current frame are '
        'included when determining the class of the current frame'
    )
    reverse = param.Boolean(
        False,
        doc='Whether to reverse each context window along the time/frame '
        'dimension'
    )


class SpectDataSetParams(SpectDataParams, DataSetParams):
    '''Data set parameters for a specific partition of spectral data'''
    pass


class ContextWindowDataSetParams(ContextWindowDataParams, SpectDataSetParams):
    '''Data set parameters for specific partition of windowed spectral data'''
    pass


class ContextWindowTrainingDataLoader(torch.utils.data.DataLoader):
    '''Serve batches of context windows over a random order of utterances

    Parameters
    ----------
    data_dir : str
    params : ContextWindowDataSetParams
    file_prefix : str, optional
    file_suffix : str, optional
    warn_on_missing : bool, optional
    feats_subdir, ali_subdir : str, optional
    init_epoch : int, optional
        Where training should resume from
    kwargs : keyword arguments, optional
        Additional ``DataLoader`` arguments

    Attributes
    ----------
    data_dir : str
    params : cnn_model.params.SpectDataParams

    Yields
    ------
    feats, alis
        `feats` is a ``FloatTensor`` of size ``(N, C, F)``, where ``N`` is the
        number of context windows, ``C`` is the context window size, and ``F``
        is the number of filters per frame. `ali` is a ``LongTensor`` of size
        ``(N,)``, where ``N`` corresponds to the sum of the number of context
        windows from ``params.batch_size`` utterances.
    '''

    def __init__(
            self, data_dir, params, init_epoch=0, file_prefix='',
            file_suffix='.pt', warn_on_missing=True,
            feats_subdir='feats', ali_subdir='ali', **kwargs):
        for bad_kwarg in (
                'batch_size', 'sampler', 'batch_sampler', 'shuffle',
                'collate_fn'):
            if bad_kwarg in kwargs:
                raise TypeError(
                    'keyword argument "{}" invalid for {} types'.format(
                        bad_kwarg, type(self)))
        self.data_dir = data_dir
        self.params = params
        self.data_source = ContextWindowDataSet(
            data_dir, params.context_left, params.context_right,
            reverse=params.reverse,
            file_prefix=file_prefix, file_suffix=file_suffix,
            warn_on_missing=warn_on_missing,
            subset_ids=set(params.subset_ids) if params.subset_ids else None,
            feats_subdir=feats_subdir, ali_subdir=ali_subdir,
        )
        if not self.data_source.has_ali:
            raise ValueError(
                "'{}' must have alignment info for training".format(
                    data_dir))
        self.__sampler = EpochRandomSampler(
            self.data_source, init_epoch=init_epoch, base_seed=params.seed)
        batch_sampler = torch.utils.data.BatchSampler(
            self.__sampler, params.batch_size, drop_last=params.drop_last)
        super(ContextWindowTrainingDataLoader, self).__init__(
            self.data_source,
            batch_sampler=batch_sampler,
            collate_fn=context_window_seq_to_batch,
            **kwargs
        )

    @property
    def epoch(self):
        '''int : the current epoch'''
        return self.__sampler.epoch

    @epoch.setter
    def epoch(self, val):
        self.__sampler.epoch = val


class ContextWindowEvaluationDataLoader(torch.utils.data.DataLoader):
    '''Serves batches of context windows over sequential utterances

    Parameters
    ----------
    data_dir : str
    params : ContextWindowDataSetParams
    file_prefix : str, optional
    file_suffix : str, optional
    warn_on_missing : bool, optional
    feats_subdir, ali_subdir : str, optional
    kwargs : keyword arguments, optional
        Additional ``DataLoader`` arguments

    Attributes
    ----------
    data_dir : str
    params : cnn_model.params.SpectDataParams

    Yields
    ------
    feats, alis, feat_sizes, utt_ids
        `feats` is a ``FloatTensor`` of size ``(N, C, F)``, where ``N`` is
        the number of context windows, ``C`` is the context window size, and
        ``F`` is the number of filters per frame. `ali` is a ``LongTensor``
        of size ``(N,)`` (or ``None`` if the ``ali`` dir was not specified).
        `feat_sizes` is a tuple of size ``params.batch_size`` specifying the
        number of context windows per utterance in the batch.
        ``feats[sum(feat_sizes[:i]):sum(feat_sizes[:i+1])]`` are the context
        windows for the ``i``-th utterance in the batch
        (``sum(feat_sizes) == N``). utt_ids` is a tuple of size
        ``params.batch_size`` naming the utterances in the batch
    '''
    class CWEvalDataSet(ContextWindowDataSet):
        '''Append feat_size and utt_id to each sample's tuple'''

        def __getitem__(self, idx):
            feats, alis = super(
                ContextWindowEvaluationDataLoader.CWEvalDataSet, self
            ).__getitem__(idx)
            feat_size = feats.size()[0]
            utt_id = self.utt_ids[idx]
            return feats, alis, feat_size, utt_id

    @staticmethod
    def eval_collate_fn(seq):
        '''Update context_window_seq_to_batch to handle feat_sizes, utt_ids'''
        feats, alis, feat_sizes, utt_ids = zip(*seq)
        feats, alis = context_window_seq_to_batch(zip(feats, alis))
        return (feats, alis, tuple(feat_sizes), tuple(utt_ids))

    def __init__(
            self, data_dir, params, file_prefix='', file_suffix='.pt',
            warn_on_missing=True, feats_subdir='feats', ali_subdir='ali',
            **kwargs):
        for bad_kwarg in (
                'batch_size', 'sampler', 'batch_sampler', 'shuffle',
                'collate_fn'):
            if bad_kwarg in kwargs:
                raise TypeError(
                    'keyword argument "{}" invalid for {} types'.format(
                        bad_kwarg, type(self)))
        self.data_dir = data_dir
        self.params = params
        self.data_source = self.CWEvalDataSet(
            data_dir, params.context_left, params.context_right,
            reverse=params.reverse,
            file_prefix=file_prefix, file_suffix=file_suffix,
            warn_on_missing=warn_on_missing,
            subset_ids=set(params.subset_ids) if params.subset_ids else None,
            feats_subdir=feats_subdir, ali_subdir=ali_subdir,
        )
        super(ContextWindowEvaluationDataLoader, self).__init__(
            self.data_source,
            batch_size=params.batch_size,
            shuffle=False,
            collate_fn=self.eval_collate_fn,
            **kwargs
        )
