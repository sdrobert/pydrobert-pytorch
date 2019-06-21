'''Classes and functions related to storing/retrieving speech data'''

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
    'ALI_PAD_VALUE',
    'context_window_seq_to_batch',
    'ContextWindowDataParams',
    'ContextWindowDataSet',
    'ContextWindowDataSetParams',
    'ContextWindowEvaluationDataLoader',
    'ContextWindowTrainingDataLoader',
    'DataSetParams',
    'EpochRandomSampler',
    'extract_window',
    'REF_PAD_VALUE',
    'spect_seq_to_batch',
    'SpectDataParams',
    'SpectDataSet',
    'SpectDataSetParams',
    'SpectEvaluationDataLoader',
    'SpectTrainingDataLoader',
    'validate_spect_data_set',
]

'''The value to right-pad alignments with when batching

The default value (-100) was chosen to coincide with the PyTorch 1.0 default
for ``ignore_index`` in the likelihood losses
'''
ALI_PAD_VALUE = -100

'''The value to right-pad token sequences with when batching

The default value (-100) was chosen to coincide with the PyTorch 1.0 default
for ``ignore_index`` in the likelihood losses
'''
REF_PAD_VALUE = -100


class SpectDataSet(torch.utils.data.Dataset):
    '''Accesses spectrographic filter data stored in a data directory

    ``SpectDataSet`` assumes that `data_dir` is structured as

    ::
        data_dir/
            feat/
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

    The ``feat`` dir stores filter bank data in the form of
    ``torch.FloatTensor``s of size ``(T, F)``, where ``T`` is the time
    dimension and ``F`` is the filter/log-frequency dimension. ``feat`` is
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
    feat_subdir, ali_subdir, ref_subdir : str, optional
        Change the names of the subdirectories under which feats, alignments,
        and references are stored. If `ali_subdir` or `ref_subdir` is ``None``,
        they will not be searched for

    Attributes
    ----------
    data_dir : str
    feat_subdir : str
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
    feat, ali, ref
    For the i-th yielded item, `feat` corresponds to the features at
    ``utt_ids[i]``, `ali` the alignments, and `ref` the reference sequence.
    If ``ali/`` or ``ref/`` did not exist on initialization, `ali` or `ref`
    will be ``None``

    Examples
    --------
    Creating a spectral data directory with random data
    >>> data_dir = 'data'
    >>> os.makedirs(data_dir + '/feat', exist_ok=True)
    >>> os.makedirs(data_dir + '/ali', exist_ok=True)
    >>> os.makedirs(data_dir + '/ref', exist_ok=True)
    >>> num_filts, min_frames, max_frames, min_ref, max_ref = 40, 10, 20, 3, 10
    >>> num_ali_classes, num_ref_classes = 100, 2000
    >>> for utt_idx in range(30):
    >>>     num_frames = torch.randint(
    ...         min_frames, max_frames + 1, (1,)).long().item()
    >>>     num_tokens = torch.randint(
    ...         min_ref, max_ref + 1, (1,)).long().item()
    >>>     feats = torch.randn(num_frames, num_filts)
    >>>     torch.save(feats, data_dir + '/feats/{:02d}.pt'.format(utt_idx))
    >>>     ali = torch.randint(num_ali_classes, (num_frames,)).long()
    >>>     torch.save(ali, data_dir + '/ali/{:02d}.pt'.format(utt_idx))
    >>>     # usually these would be sorted by order in utterance. Negative
    >>>     # values represent "unknown" for start end end frames
    >>>     ref_tokens = torch.randint(num_tokens, (num_tokens,))
    >>>     ref_starts = torch.randint(1, num_frames // 2, (num_tokens,))
    >>>     ref_ends = 2 * ref_starts
    >>>     ref = torch.stack([ref_tokens, ref_starts, ref_ends], -1).long()
    >>>     torch.save(ref, data_dir + '/ref/{:02d}.pt'.format(utt_idx))

    Accessing individual elements in a spectral data directory
    >>> data = SpectDataSet('data')
    >>> data[0]  # random access feats, ali
    >>> for feat, ali, ref in data:  # iterator
    >>>     pass

    Writing evaluation data back to the directory
    >>> data = SpectDataSet('data')
    >>> num_ali_classes, num_ref_classes, min_ref, max_ref = 100, 2000, 3, 10
    >>> num_frames = data[3][0].shape[0]
    >>> # pdfs (or more accurately, pms) are likelihoods of classes over data
    >>> # per frame, used in hybrid models. Usually logits
    >>> pdf = torch.randn(num_frames, num_ali_classes)
    >>> data.write_pdf(3, pdf)  # will share name with data.utt_ids[3]
    >>> # both refs and hyps are sequences of tokens, such as words or phones,
    >>> # with optional frame alignments
    >>> num_tokens = torch.randint(min_ref, max_ref, (1,)).long().item()
    >>> hyp = torch.full((num_tokens, 3), REF_PAD_VALUE).long()
    >>> hyp[..., 0] = torch.randint(num_ref_classes, (num_tokens,))
    >>> data.write_hyp('special', hyp)  # custom name
    '''

    def __init__(
            self, data_dir, file_prefix='', file_suffix='.pt',
            warn_on_missing=True, subset_ids=None,
            feat_subdir='feat', ali_subdir='ali', ref_subdir='ref'):
        super(SpectDataSet, self).__init__()
        self.data_dir = data_dir
        self.feat_subdir = feat_subdir
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
            for x in os.listdir(os.path.join(self.data_dir, self.feat_subdir))
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
                        "Missing feat for uttid: '{}'".format(utt_id))
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
                        "Missing feat for uttid: '{}'".format(utt_id))
        if self.has_ali:
            utt_ids &= ali_utt_ids
        if self.has_ref:
            utt_ids &= ref_utt_ids
        return utt_ids

    def get_utterance_tuple(self, idx):
        '''Get a tuple of features, alignments, and references'''
        utt_id = self.utt_ids[idx]
        feat = torch.load(
            os.path.join(
                self.data_dir,
                self.feat_subdir,
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
        return feat, ali, ref

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
        feat, ali, ref = data_set.get_utterance_tuple(idx)
        if not isinstance(feat, torch.FloatTensor):
            raise ValueError(
                "'{}' (index {}) in '{}' is not a FloatTensor".format(
                    data_set.utt_ids[idx] + data_set.file_suffix, idx,
                    os.path.join(data_set.data_dir, data_set.feat_subdir)))
        if len(feat.size()) != 2:
            raise ValueError(
                "'{}' (index {}) in '{}' does not have two axes".format(
                    data_set.utt_ids[idx] + data_set.file_suffix, idx,
                    os.path.join(data_set.data_dir, data_set.feat_subdir)
                ))
        if num_filts is None:
            num_filts = feat.shape[1]
        elif feat.shape[1] != num_filts:
            raise ValueError(
                "'{}' (index {}) in '{}' has second axis size {}, which "
                "does not match prior utterance ('{}') size of {}".format(
                    data_set.utt_ids[idx] + data_set.file_suffix, idx,
                    os.path.join(data_set.data_dir, data_set.feat_subdir),
                    feat.shape[1],
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
            if ali.shape[0] != feat.shape[0]:
                raise ValueError(
                    "'{}' (index {}) in '{}' does not have the same first axis"
                    " size ({}) as it's companion in '{}' ({})".format(
                        data_set.utt_ids[idx] + data_set.file_suffix, idx,
                        os.path.join(data_set.data_dir, data_set.feat_subdir),
                        feat.shape[0],
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
            for idx2, r in enumerate(ref):
                if not (r[1] < 0 and r[2] < 0) and not (
                        0 <= r[1] < r[2] <= feat.shape[0]):
                    raise ValueError(
                        "'{}' (index {}) in '{}', has a reference token "
                        "(index {}) with bounds outside the utterance"
                        "".format(
                            data_set.utt_ids[idx] + data_set.file_suffix, idx,
                            os.path.join(
                                data_set.data_dir, data_set.ref_subdir),
                            idx2))


def extract_window(feat, frame_idx, left, right, reverse=False):
    '''Slice the feature matrix to extract a context window

    Parameters
    ----------
    feat : torch.Tensor
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
    T, F = feat.shape
    if frame_idx - left < 0 or frame_idx + right + 1 > T:
        win_size = 1 + left + right
        window = feat.new(win_size, F)
        left_pad = max(left - frame_idx, 0)
        right_pad = max(frame_idx + right + 1 - T, 0)
        window[left_pad:win_size - right_pad] = feat[
            max(0, frame_idx - left):frame_idx + right + 1]
        if left_pad:
            window[:left_pad] = feat[0]
        if right_pad:
            window[-right_pad:] = feat[-1]
    else:
        window = feat[frame_idx - left:frame_idx + right + 1]
    if reverse:
        window = torch.flip(window, [0])
    return window


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
    feat_subdir, ali_subdir : str, optional
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
    window, ali

    Examples
    --------
    >>> # see 'SpectDataSet' to set up data directory
    >>> data = ContextWindowDataSet('data', 3, 3)
    >>> data[0]  # random access returns (window, ali) pairs
    >>> for window, ali in data:
    >>>     pass  # so does the iterator
    >>> data.get_utterance_tuple(3)  # gets the original (feat, ali) pair
    '''

    def __init__(
            self, data_dir, left, right, file_prefix='',
            file_suffix='.pt', warn_on_missing=True, subset_ids=None,
            feat_subdir='feat', ali_subdir='ali', reverse=False):
        super(ContextWindowDataSet, self).__init__(
            data_dir, file_prefix=file_prefix, file_suffix=file_suffix,
            warn_on_missing=warn_on_missing, subset_ids=subset_ids,
            feat_subdir=feat_subdir, ali_subdir=ali_subdir,
            ref_subdir=None)
        self.left = left
        self.right = right
        self.reverse = reverse

    def get_utterance_tuple(self, idx):
        '''Get a tuple of features and alignments'''
        return super(ContextWindowDataSet, self).get_utterance_tuple(idx)[:2]

    def get_windowed_utterance(self, idx):
        '''Get pair of features (w/ context windows) and alignments'''
        feat, ali = self.get_utterance_tuple(idx)
        num_frames, num_filts = feat.shape
        window = torch.empty(
            num_frames, 1 + self.left + self.right, num_filts)
        for center_frame in range(num_frames):
            window[center_frame] = extract_window(
                feat, center_frame, self.left, self.right,
                reverse=self.reverse)
        return window, ali

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
            # we use numpy RandomState so that we can run in parallel with
            # torch's RandomState, but we acquire the initial random seed from
            # torch so that we'll be deterministic with a prior call to
            # torch.manual_seed(...)
            base_seed = torch.randint(
                np.iinfo(np.int32).max, (1,)).long().item()
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


class SpectDataSetParams(SpectDataParams, DataSetParams):
    '''Data set parameters for a specific partition of spectral data'''
    pass


def spect_seq_to_batch(seq):
    r'''Convert a sequence of spectral data to a batch

    Assume `seq` is a finite length sequence of tuples ``feat, ali, ref``,
    where ``feat`` is of size ``(T, F)``, where ``T`` is some number of frames
    (which can vary across elements in the sequence), ``F`` is some number of
    filters, ``ali`` is of size ``(T,)``, and ``ref`` is of size ``(R, 3)``,
    where ``R`` is some number of reference tokens (which can vary across
    elements in the sequence). This method batches all the elements of the
    sequence into a tuple of ``feats, alis, refs, feat_sizes, ref_sizes``.
    `feats` and `alis` will have dimensions ``(N, T*, F)``, and ``(N, T*)``,
    resp., where ``N`` is the batch size, and ``T*`` is the maximum number of
    frames in `seq`. Similarly, `refs` will have dimensions ``(N, R*, 3)``.
    `feat_sizes` and `ref_sizes` are tuples of ints containing the original
    ``T`` and ``R`` values. The batch will be sorted by decreasing numbers of
    frames. `feats` is zero-padded while `alis` and `refs` are padded with
    module constants ``ALI_PAD_VALUE`` and ``REF_PAD_VALUE``, respectively.

    If ``ali`` or ``ref`` is ``None`` in any element, `alis` or `refs` and
    `ref_sizes` will also be ``None``

    Parameters
    ----------
    seq : sequence

    Returns
    -------
    feats, alis, refs, feat_sizes, ref_sizes
    '''
    seq = sorted(seq, key=lambda x: -x[0].shape[0])
    T_star, F = seq[0][0].shape
    feats = torch.zeros(len(seq), T_star, F)
    feat_sizes = []
    has_ali = all(x[1] is not None for x in seq)
    R_star = max(float('inf') if x[2] is None else x[2].shape[0] for x in seq)
    has_ref = R_star < float('inf')
    if has_ali:
        alis = torch.full(
            (len(seq), T_star), ALI_PAD_VALUE, dtype=torch.long)
    else:
        alis = None
    if has_ref:
        refs = torch.full(
            (len(seq), R_star, 3), REF_PAD_VALUE, dtype=torch.long)
        ref_sizes = []
    else:
        refs = None
        ref_sizes = None
    for n, (feat, ali, ref) in enumerate(seq):
        feat_size = feat.shape[0]
        feats[n, :feat_size] = feat
        feat_sizes.append(feat_size)
        if has_ali:
            alis[n, :feat_size] = ali
        if has_ref:
            ref_size = ref.shape[0]
            ref_sizes.append(ref_size)
            refs[n, :ref_size] = ref

    return (
        feats, alis, refs,
        feat_sizes if feat_sizes is None else tuple(feat_sizes),
        ref_sizes if ref_sizes is None else tuple(ref_sizes),
    )


class SpectTrainingDataLoader(torch.utils.data.DataLoader):
    '''Serves batches of spectral data over random orders of utterances

    Parameters
    ----------
    data_dir : str
    params : SpectDataSetParams
    file_prefix : str, optional
    file_suffix : str, optional
    warn_on_missing : bool, optional
    feat_subdir, ali_subdir, ref_subdir : str, optional
    init_epoch : int, optional
        Where training should resume from
    kwargs : keyword arguments, optional
        Additional ``DataLoader`` arguments

    Attributes
    ----------
    data_dir : str
    params : SpectDataSetParams

    Yields
    ------
    feats, alis, refs, feat_sizes, ref_sizes
        `feats` is a ``FloatTensor`` of size ``(N, T*, F)``, where ``N`` is
        ``params.batch_size``, ``T*`` is the maximum number of frames in an
        utterance in the batch, and ``F`` is the number of filters per frame.
        `ali` is a ``LongTensor`` of size ``(N, T*)`` if an ``ali/`` dir
        exists, otherwise ``None``. ``feat_sizes`` is an ``N``-tuple of
        integers specifying the lengths of utterances in the batch. `refs` is
        a ``LongTensor`` of size ``(N, R*, 3)``, where ``R*`` is the maximum
        number of reference tokens in the batch. `ref_sizes` is an ``N``-tuple
        of integers specifying the number of reference tokens per utterance in
        the batch. If the ``refs/`` directory does not exist, `refs` and
        `ref_sizes` are ``None``. The first axis of each of `feats`, `alis`,
        `refs`, `feat_sizes`, and `ref_sizes` is ordered by utterances of
        descending frame lengths. Shorter utterances in `feats` are zero-padded
        to the right, `alis` is padded with the module constant
        ``ALI_PAD_VALUE``, and `refs` is padded with ``REF_PAD_VALUE``

    Examples
    --------
    Training on alignments for one epoch
    >>> # see 'SpectDataSet' to initialize data set
    >>> num_filts, num_ali_classes = 40, 100
    >>> model = torch.nn.LSTM(num_filts, num_ali_classes)
    >>> optim = torch.optim.Adam(model.parameters())
    >>> loss = torch.nn.CrossEntropyLoss()
    >>> params = SpectDataSetParams()
    >>> loader = SpectTrainingDataLoader('data', params)
    >>> for feats, alis, _, feat_sizes, _ in loader:
    >>>     optim.zero_grad()
    >>>     feat_sizes = torch.tensor(feat_sizes)
    >>>     packed_feats = torch.nn.utils.rnn.pack_padded_sequence(
    ...         feats, feat_sizes, batch_first=True)
    >>>     packed_alis = torch.nn.utils.rnn.pack_padded_sequence(
    ...         ali, feat_sizes, batch_first=True)
    >>>     packed_logits, _ = model(packed_feats)
    >>>     # no need to unpack: loss is the same as if we ignored padded vals
    >>>     loss(packed_logits.data, packed_ali.data).backward()
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
    >>> params = SpectDataSetParams()
    >>> loader = SpectTrainingDataLoader('data', params)
    >>> for feats, _, refs, feat_sizes, ref_sizes in loader:
    >>>     optim.zero_grad()
    >>>     feat_sizes = torch.tensor(feat_sizes)
    >>>     ref_sizes = torch.tensor(ref_sizes)
    >>>     feats = feats.unsqueeze(1)  # channels dim
    >>>     log_prob = model(feats).squeeze(1)
    >>>     loss(
    ...         log_prob.transpose(0, 1),
    ...         refs[..., 0], feat_sizes, ref_sizes).backward()
    >>>     optim.step()
    '''

    def __init__(
            self, data_dir, params, init_epoch=0, file_prefix='',
            file_suffix='.pt', warn_on_missing=True,
            feat_subdir='feat', ali_subdir='ali',
            ref_subdir='ref', **kwargs):
        for bad_kwarg in (
                'batch_size', 'sampler', 'batch_sampler', 'shuffle',
                'collate_fn'):
            if bad_kwarg in kwargs:
                raise TypeError(
                    'keyword argument "{}" invalid for {} types'.format(
                        bad_kwarg, type(self)))
        self.data_dir = data_dir
        self.params = params
        self.data_source = SpectDataSet(
            data_dir,
            file_prefix=file_prefix, file_suffix=file_suffix,
            warn_on_missing=warn_on_missing,
            subset_ids=set(params.subset_ids) if params.subset_ids else None,
            feat_subdir=feat_subdir, ali_subdir=ali_subdir,
            ref_subdir=ref_subdir,
        )
        if not self.data_source.has_ali and not self.data_source.has_ref:
            raise ValueError(
                "'{}' must have either alignments or reference tokens for "
                "training".format(data_dir))
        epoch_sampler = EpochRandomSampler(
            self.data_source, init_epoch=init_epoch, base_seed=params.seed)
        batch_sampler = torch.utils.data.BatchSampler(
            epoch_sampler, params.batch_size, drop_last=params.drop_last)
        super(SpectTrainingDataLoader, self).__init__(
            self.data_source,
            batch_sampler=batch_sampler,
            collate_fn=spect_seq_to_batch,
            **kwargs
        )

    @property
    def epoch(self):
        '''int : the current epoch'''
        return self.batch_sampler.sampler.epoch

    @epoch.setter
    def epoch(self, val):
        self.batch_sampler.sampler.epoch = val


class SpectEvaluationDataLoader(torch.utils.data.DataLoader):
    '''Serves batches of spectral data over a fixed order of utterances

    Parameters
    ----------
    data_dir : str
    params : SpectDataSetParams
    file_prefix : str, optional
    file_suffix : str, optional
    warn_on_missing : bool, optional
    feat_subdir, ali_subdir, ref_subdir : str, optional
    kwargs : keyword arguments, optional
        Additional ``DataLoader`` arguments

    Attributes
    ----------
    data_dir : str
    params : SpectDataSetParams

    Yields
    ------
    feats, alis, refs, feat_sizes, ref_sizes, utt_ids
        `feats` is a ``FloatTensor`` of size ``(N, T*, F)``, where ``N`` is
        ``params.batch_size``, ``T*`` is the maximum number of frames in an
        utterance in the batch, and ``F`` is the number of filters per frame.
        `ali` is a ``LongTensor`` of size ``(N, T*)`` if an ``ali/`` dir
        exists, otherwise ``None``. ``feat_sizes`` is an ``N``-tuple of
        integers specifying the lengths of utterances in the batch. `refs` is
        a ``LongTensor`` of size ``(N, R*, 3)``, where ``R*`` is the maximum
        number of reference tokens in the batch. `ref_sizes` is an ``N``-tuple
        of integers specifying the number of reference tokens per utterance in
        the batch. If the ``refs/`` directory does not exist, `refs` and
        `ref_sizes` are ``None``. ``utt_ids`` is an ``N``-tuple specifying
        the names of utterances in the batch. The first axis of each of
        `feats`, `alis`, `refs`, `feat_sizes`, `ref_sizes`, and `utt_ids`
        is ordered by utterances of descending frame lengths. Shorter
        utterances in `feats` are zero-padded to the right, `alis` is padded
        with the module constant ``ALI_PAD_VALUE``, and `refs` is padded with
        ``REF_PAD_VALUE``

    Examples
    --------
    Computing class likelihoods and writing them to disk
    >>> # see 'SpectDataSet' to initialize data set
    >>> num_filts, num_ali_classes = 40, 100
    >>> model = torch.nn.LSTM(num_filts, num_ali_classes)
    >>> params = SpectDataSetParams()
    >>> loader = SpectEvaluationDataLoader('data', params)
    >>> for feats, _, _, feat_sizes, _, utt_ids in loader:
    >>>     feat_sizes = torch.tensor(feat_sizes)
    >>>     packed_feats = torch.nn.utils.rnn.pack_padded_sequence(
    ...         feats, feat_sizes, batch_first=True)
    >>>     packed_logits, _ = model(packed_feats)
    >>>     logits, _ = torch.nn.utils.rnn.pad_packed_sequence(
    >>>         packed_logits, batch_first=True)
    >>>     log_probs = torch.nn.functional.log_softmax(logits, -1)
    >>>     for pdf, feat_size, utt_id in zip(log_probs, feat_sizes, utt_ids):
    >>>         loader.data_source.write_pdf(utt_id, pdf)

    Transcribing utterances with CTC
    >>> num_filts, num_ref_classes, kern = 40, 2000, 3
    >>> # we use padding to ensure gradients are unaffected by batch padding
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Conv2d(1, 1, kern, padding=(kern - 1) // 2),
    ...     torch.nn.ReLU(),
    ...     torch.nn.Linear(num_filts, num_ref_classes),
    ...     torch.nn.LogSoftmax(-1)).eval()
    >>> params = SpectDataSetParams()
    >>> loader = SpectEvaluationDataLoader('data', params)
    >>> for feats, _, _, feat_sizes, _, utt_ids in loader:
    >>>     feat_sizes = torch.tensor(feat_sizes)
    >>>     ref_sizes = torch.tensor(ref_sizes)
    >>>     feats = feats.unsqueeze(1)  # channels dim
    >>>     log_prob = model(feats).squeeze(1)
    >>>     paths = log_prob.argmax(-1)  # best path decoding
    >>>     for path, feat_size, utt_id in zip(paths, feat_sizes, utt_ids):
    >>>         path = path[:feat_size]
    >>>         pathpend = torch.cat([torch.tensor([-1]), path])
    >>>         path = path.masked_select(
    ...             (path != 0) & (path - pathpend[:-1] != 0))
    >>>         hyp = torch.stack(
    ...             [path] + [torch.full_like(path, REF_PAD_VALUE)] * 2)
    >>>         loader.data_source.write_hyp(utt_id, hyp)
    '''
    class SEvalDataSet(SpectDataSet):
        '''Append utt_id to each sample's tuple'''

        def __getitem__(self, idx):
            feat, ali, ref = super(
                SpectEvaluationDataLoader.SEvalDataSet, self).__getitem__(idx)
            utt_id = self.utt_ids[idx]
            return feat, ali, ref, utt_id

    @staticmethod
    def eval_collate_fn(seq):
        '''Update context_window_seq_to_batch to handle feat_sizes, utt_ids'''
        feats, alis, refs, utt_ids = zip(*seq)
        # spect_seq_to_batch sorts by descending number of frames, so we
        # sort utt_ids here
        utt_ids = tuple(
            x[1]
            for x in sorted(zip(feats, utt_ids), key=lambda x: -x[0].shape[0])
        )
        feats, alis, refs, feat_sizes, ref_sizes = spect_seq_to_batch(
            zip(feats, alis, refs))
        return feats, alis, refs, feat_sizes, ref_sizes, utt_ids

    def __init__(
            self, data_dir, params, file_prefix='', file_suffix='.pt',
            warn_on_missing=True, feat_subdir='feat', ali_subdir='ali',
            ref_subdir='ref', **kwargs):
        for bad_kwarg in (
                'batch_size', 'sampler', 'batch_sampler', 'shuffle',
                'collate_fn'):
            if bad_kwarg in kwargs:
                raise TypeError(
                    'keyword argument "{}" invalid for {} types'.format(
                        bad_kwarg, type(self)))
        self.data_dir = data_dir
        self.params = params
        self.data_source = self.SEvalDataSet(
            data_dir,
            file_prefix=file_prefix, file_suffix=file_suffix,
            warn_on_missing=warn_on_missing,
            subset_ids=set(params.subset_ids) if params.subset_ids else None,
            feat_subdir=feat_subdir, ali_subdir=ali_subdir,
            ref_subdir=ref_subdir,
        )
        super(SpectEvaluationDataLoader, self).__init__(
            self.data_source,
            batch_size=params.batch_size,
            shuffle=False,
            collate_fn=self.eval_collate_fn,
            **kwargs
        )


def context_window_seq_to_batch(seq):
    r'''Convert a sequence of context window elements to a batch

    Assume `seq` is a finite length sequence of pairs of ``window, ali``, where
    ``window`` is of size ``(T, C, F)``, where ``T`` is some number of windows
    (which can vary across elements in the sequence), ``C`` is the window size,
    and ``F`` is some number filters, and ``ali`` is of size ``(T,)``. This
    method batches all the elements of the sequence into a pair of ``windows,
    alis``, where `windows` and `alis` will have shapes ``(N, C, F)`` and
    ``(N,)`` resp., where :math:`N = \sum T` is the total number of context
    windows over the utterances.

    If ``ali`` is ``None`` in any element, `alis` will also be ``None``

    Parameters
    ----------
    seq : sequence

    Returns
    -------
    windows, alis : tuple
    '''
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
    feat_subdir, ali_subdir : str, optional
    init_epoch : int, optional
        Where training should resume from
    kwargs : keyword arguments, optional
        Additional ``DataLoader`` arguments

    Attributes
    ----------
    data_dir : str
    params : ContextWindowDataSetParams
    data_source : ContextWindowDataSet

    Yields
    ------
    windows, alis
        `windows` is a ``FloatTensor`` of size ``(N, C, F)``, where ``N`` is
        the total number of context windows over all utterances in the batch,
        ``C`` is the context window size, and ``F`` is the number of filters
        per frame. `alis` is a ``LongTensor`` of size ``(N,)`` (or ``None``
        if the ``ali`` dir was not specified).


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
    >>> params = ContextWindowDataSetParams(
    ...     context_left=left, context_right=right)
    >>> loader = ContextWindowTrainingDataLoader('data', params)
    >>> for windows, alis in loader:
    >>>     optim.zero_grad()
    >>>     windows = windows.view(-1, num_filts * window_width)  # flatten win
    >>>     logits = model(windows)
    >>>     loss(logits, alis).backward()
    >>>     optim.step()
    '''

    def __init__(
            self, data_dir, params, init_epoch=0, file_prefix='',
            file_suffix='.pt', warn_on_missing=True,
            feat_subdir='feat', ali_subdir='ali', **kwargs):
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
            feat_subdir=feat_subdir, ali_subdir=ali_subdir,
        )
        if not self.data_source.has_ali:
            raise ValueError(
                "'{}' must have alignment info for training".format(
                    data_dir))
        epoch_sampler = EpochRandomSampler(
            self.data_source, init_epoch=init_epoch, base_seed=params.seed)
        batch_sampler = torch.utils.data.BatchSampler(
            epoch_sampler, params.batch_size, drop_last=params.drop_last)
        super(ContextWindowTrainingDataLoader, self).__init__(
            self.data_source,
            batch_sampler=batch_sampler,
            collate_fn=context_window_seq_to_batch,
            **kwargs
        )

    @property
    def epoch(self):
        '''int : the current epoch'''
        return self.batch_sampler.sampler.epoch

    @epoch.setter
    def epoch(self, val):
        self.batch_sampler.sampler.epoch = val


class ContextWindowEvaluationDataLoader(torch.utils.data.DataLoader):
    '''Serves batches of context windows over sequential utterances

    Parameters
    ----------
    data_dir : str
    params : ContextWindowDataSetParams
    file_prefix : str, optional
    file_suffix : str, optional
    warn_on_missing : bool, optional
    feat_subdir, ali_subdir : str, optional
    kwargs : keyword arguments, optional
        Additional ``DataLoader`` arguments

    Attributes
    ----------
    data_dir : str
    params : SpectDataParams

    Yields
    ------
    windows, alis, win_sizes, utt_ids
        `windows` is a ``FloatTensor`` of size ``(N, C, F)``, where ``N`` is
        the number of context windows, ``C`` is the context window size, and
        ``F`` is the number of filters per frame. `alis` is a ``LongTensor``
        of size ``(N,)`` (or ``None`` if the ``ali`` dir was not specified).
        `win_sizes` is a tuple of size ``params.batch_size`` specifying the
        number of context windows per utterance in the batch.
        ``windows[sum(win_sizes[:i]):sum(win_sizes[:i+1])]`` are the context
        windows for the ``i``-th utterance in the batch
        (``sum(win_sizes) == N``). utt_ids` is a tuple of size
        ``params.batch_size`` naming the utterances in the batch

    Examples
    --------
    Computing class likelihoods and writing them to disk
    >>> # see 'SpectDataSet' to initialize data set
    >>> num_filts, num_ali_classes, left, right = 40, 100, 4, 4
    >>> window_width = left + right + 1
    >>> model = torch.torch.nn.Linear(
    ...     num_filts * window_width, num_ali_classes).eval()
    >>> params = ContextWindowDataSetParams(
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
    '''
    class CWEvalDataSet(ContextWindowDataSet):
        '''Append feat_size and utt_id to each sample's tuple'''

        def __getitem__(self, idx):
            window, ali = super(
                ContextWindowEvaluationDataLoader.CWEvalDataSet, self
            ).__getitem__(idx)
            win_size = window.size()[0]
            utt_id = self.utt_ids[idx]
            return window, ali, win_size, utt_id

    @staticmethod
    def eval_collate_fn(seq):
        '''Update context_window_seq_to_batch to handle feat_sizes, utt_ids'''
        windows, alis, feat_sizes, utt_ids = zip(*seq)
        windows, alis = context_window_seq_to_batch(zip(windows, alis))
        return (windows, alis, tuple(feat_sizes), tuple(utt_ids))

    def __init__(
            self, data_dir, params, file_prefix='', file_suffix='.pt',
            warn_on_missing=True, feat_subdir='feat', ali_subdir='ali',
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
            feat_subdir=feat_subdir, ali_subdir=ali_subdir,
        )
        super(ContextWindowEvaluationDataLoader, self).__init__(
            self.data_source,
            batch_size=params.batch_size,
            shuffle=False,
            collate_fn=self.eval_collate_fn,
            **kwargs
        )
