# Copyright 2019 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Classes and functions related to storing/retrieving speech data'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
from collections import OrderedDict


import numpy as np
import torch
import torch.utils.data
import param
import pydrobert.torch

try:
    basestring
except NameError:
    basestring = str

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"
__all__ = [
    'context_window_seq_to_batch',
    'ContextWindowDataParams',
    'ContextWindowDataSet',
    'ContextWindowDataSetParams',
    'ContextWindowEvaluationDataLoader',
    'ContextWindowTrainingDataLoader',
    'DataSetParams',
    'EpochRandomSampler',
    'extract_window',
    'read_ctm',
    'read_trn',
    'spect_seq_to_batch',
    'SpectDataParams',
    'SpectDataSet',
    'SpectDataSetParams',
    'SpectEvaluationDataLoader',
    'SpectTrainingDataLoader',
    'token_to_transcript',
    'transcript_to_token',
    'validate_spect_data_set',
    'write_ctm',
    'write_trn',
]


class SpectDataSet(torch.utils.data.Dataset):
    '''Accesses spectrographic filter data stored in a data directory

    :class:`SpectDataSet` assumes that `data_dir` is structured as

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
    :class:`torch.FloatTensor` of size ``(T, F)``, where ``T`` is the time
    dimension and ``F`` is the filter/log-frequency dimension. ``feat`` is
    the only required directory.

    ``ali`` stores :class:`torch.LongTensor` of size ``(T,)``, indicating the
    pdf-id of the most likely target. ``ali`` is suitable for discriminative
    training of DNNs in hybrid DNN-HMM recognition, or any frame-wise loss.
    ``ali/`` is optional.

    ``ref`` stores :class:`torch.LongTensor` of size ``(R,3)``, indicating
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
        utterances in the directories, and `warn_on_missing` is :obj:`True`, a
        warning will be issued (via ``warnings``) regarding each such mismatch
    subset_ids : set, optional
        If set, only utterances with ids listed in this set will count towards
        the data set. The rest will be ignored
    eos : int, optional
        `eos` is a special token used to delimit the end of a reference
        or hypothesis sequence. If specified, an extra `eos` token without
        positional information will be appended to the end of each reference
        tanscript. It will also have ramifications for the method
        ``write_hyp()``
    feat_subdir : str, optional
    ali_subdir : str, optional
    ref_subdir : str, optional
        Change the names of the subdirectories under which feats, alignments,
        and references are stored. If `ali_subdir` or `ref_subdir` is
        :obj:`None`, they will not be searched for

    Attributes
    ----------
    data_dir, feat_subdir, ali_subdir, ref_subdir, file_suffix : str
    has_ali : bool
        Whether alignment data exist
    has_ref : bool
        Whether reference data exist
    utt_ids : tuple
        A tuple of all utterance ids extracted from the data directory. They
        are stored in the same order as features and alignments via
        :func:`__getitem__`. If the ``ali/`` or ``ref/`` directories exist,
        `utt_ids` contains only the utterances in the intersection of each
        directory (and `subset_ids`, if it was specified)
    eos : int or None

    Yields
    ------
    feat : torch.FloatTensor
    ali : torch.LongTensor or None
    ref : torch.LongTensor or None

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
    >>> data[0]  # random access feat, ali, ref
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
    >>> hyp = torch.full((num_tokens, 3), INDEX_PAD_VALUE).long()
    >>> hyp[..., 0] = torch.randint(num_ref_classes, (num_tokens,))
    >>> data.write_hyp('special', hyp)  # custom name
    '''

    def __init__(
            self, data_dir, file_prefix='', file_suffix='.pt',
            warn_on_missing=True, subset_ids=None, eos=None,
            feat_subdir='feat', ali_subdir='ali', ref_subdir='ref'):
        super(SpectDataSet, self).__init__()
        self.data_dir = data_dir
        self.feat_subdir = feat_subdir
        self.ali_subdir = ali_subdir
        self.ref_subdir = ref_subdir
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.eos = eos
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
            x[fpl:neg_fsl]
            for x in os.listdir(os.path.join(self.data_dir, self.feat_subdir))
            if x.startswith(self.file_prefix) and x.endswith(self.file_suffix)
        )
        if subset_ids is not None:
            utt_ids &= subset_ids
        if self.has_ali:
            ali_utt_ids = set(
                x[fpl:neg_fsl]
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
                x[fpl:neg_fsl]
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
            if self.eos is not None:
                eos_sym = torch.full_like(ref[0], -1)
                eos_sym[0] = self.eos
                ref = torch.cat([ref, eos_sym.unsqueeze(0)])
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
            The tensor to write. It will be converted to a CPU
            :class:`torch.FloatTensor` using the command ``pdf.cpu().float()``
        pdfs_dir : str or None, optional
            The directory pdfs are written to. If :obj:`None`, it will be set
            to ``self.data_dir + '/pdfs'``
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

        If the ``eos`` attribute of this instance is not :obj:`None`, any
        tokens in `hyp` matching it will be considered the end of the sequence,
        so every symbol including and after the first instance will be removed
        from the utterance before saving

        Parameters
        ----------
        utt : str or int
            The name of the utterance to write. If an integer is specified,
            `utt` is assumed to index an utterance id specified in
            ``self.utt_ids``
        hyp : torch.Tensor
            The tensor to write. It will be converted to a
            :class:`torch.LongTensor` using the command ``pdf.cpu().long()``
        hyp_dir : str or None, optional
            The directory pdfs are written to. If :obj:`None`, it will be set
            to ``self.data_dir + '/hyp'``
        '''
        if isinstance(utt, int):
            utt = self.utt_ids[utt]
        if hyp_dir is None:
            hyp_dir = os.path.join(self.data_dir, 'hyp')
        if not os.path.isdir(hyp_dir):
            os.makedirs(hyp_dir)
        hyp = hyp.cpu().long()
        if self.eos is not None:
            eos_idxs = hyp[:, 0].eq(self.eos).nonzero()
            if eos_idxs.numel():
                eos_idx = eos_idxs[0].item()
                hyp = hyp[:eos_idx]
        torch.save(
            hyp.cpu().long(),
            os.path.join(hyp_dir, self.file_prefix + utt + self.file_suffix)
        )


def validate_spect_data_set(data_set):
    '''Validate SpectDataSet data directory

    The data directory is valid if the following conditions are observed

    1. All features are :class:`torch.FloatTensor` instances
    2. All features have two axes
    3. All features have the same size second axis
    4. If alignments are present

       1. All alignments are :class:`torch.LongTensor` instances
       2. All alignments have one axis
       3. Features and alignments have the same size first axes for a given
          utterance id

    5. If reference sequences are present

       1. All references are :class:`torch.LongTensor` instances
       2. All alignments have two axes, the second of size 3
       3. For the start and end points of a reference token, ``r[i, 1:]``,
          either both of them are negative (indicating no alignment), or
          ``0 <= r[i, 1] < r[i, 2] <= T``, where ``T`` is the number of
          frames in the utterance. We do not enforce tokens be
          non-overlapping

    Raises a :class:`ValueError` if a condition is violated
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
                        "(index {}) with invalid boundaries"
                        "".format(
                            data_set.utt_ids[idx] + data_set.file_suffix, idx,
                            os.path.join(
                                data_set.data_dir, data_set.ref_subdir),
                            idx2))


def read_trn(trn, warn=True):
    '''Read a NIST sclite transcript file into a list of transcripts

    `sclite <http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm>`__
    is a commonly used scoring tool for ASR.

    This function converts a transcript input file ("trn" format) into a
    list of `transcripts`, where each element is a tuple of
    ``utt_id, transcript``. ``transcript`` is a list split by spaces.

    Parameters
    ----------
    trn : file or str
        The transcript input file. Will open if `trn` is a path
    warn : bool, optional
        The "trn" format uses curly braces and forward slashes to indicate
        transcript alterations. This is largely for scoring purposes, such as
        swapping between filled pauses, not for training. If `warn` is
        :obj:`True`, a warning will be issued via the ``warnings`` module every
        time an alteration appears in the "trn" file. Alterations appear in
        `transcripts` as elements of ``([[alt_1_word_1, alt_1_word_2, ...],
        [alt_2_word_1, alt_2_word_2, ...], ...], -1, -1)`` so that
        ``transcript_to_token`` will not attempt to process alterations as
        token start and end times

    Returns
    -------
    transcripts : list

    Notes
    -----
    Any null words (``@``) in the "trn" file are encoded verbatim.
    '''
    # implementation note: there's a lot of weirdness here. I'm trying to
    # match sclite's behaviour. A few things
    # - the last parentheses are always the utterance. Everything else is
    #   the word
    # - An unmatched '}' is treated as a word
    # - A '/' not within curly braces is a word
    # - If the utterance ends without closing its alternate, the alternate is
    #   discarded
    # - Comments from other formats are not comments here...
    # - ...but everything passed the last pair of parentheses is ignored...
    # - ...and internal parentheses are treated as words
    # - Spaces are treated as part of the utterance id
    # - Seg faults on empty alternates
    class AltTree(object):
        def __init__(self, parent=None):
            self.parent = parent
            self.tokens = []
            if parent is not None:
                parent.tokens.append([self.tokens])

        def new_branch(self):
            assert self.parent
            self.tokens = []
            self.parent.tokens[-1].append(self.tokens)
    if isinstance(trn, basestring):
        with open(trn, 'r') as trn:
            return read_trn(trn)
    transcripts = []
    for line in trn:
        line = line.strip()
        if not line:
            continue
        try:
            last_open = line.rindex('(')
            last_close = line.rindex(')')
            if last_open > last_close:
                raise ValueError()
        except ValueError:
            raise IOError('Line does not end in utterance id')
        utt_id = line[last_open + 1:last_close]
        line = line[:last_open].strip()
        transcript = []
        token = ''
        alt_tree = AltTree()
        found_alt = False
        while len(line):
            c = line[0]
            line = line[1:]
            if c == '{':
                found_alt = True
                if token:
                    if alt_tree.parent is None:
                        transcript.append(token)
                    else:
                        alt_tree.tokens.append(token)
                    token = ''
                alt_tree = AltTree(alt_tree)
            elif c == '/' and alt_tree.parent is not None:
                if token:
                    alt_tree.tokens.append(token)
                    token = ''
                alt_tree.new_branch()
            elif c == '}' and alt_tree.parent is not None:
                if token:
                    alt_tree.tokens.append(token)
                    token = ''
                if not alt_tree.tokens:
                    raise IOError('Empty alternate found ("{ }")')
                alt_tree = alt_tree.parent
                if alt_tree.parent is None:
                    assert len(alt_tree.tokens) == 1
                    transcript.append((alt_tree.tokens[0], -1, -1))
                    alt_tree.tokens = []
            elif c == ' ':
                if token:
                    if alt_tree.parent is None:
                        transcript.append(token)
                    else:
                        alt_tree.tokens.append(token)
                    token = ''
            else:
                token += c
        if token and alt_tree.parent is None:
            transcript.append(token)
        if found_alt and warn:
            warnings.warn(
                'Found an alternate in transcription for utt="{}". '
                'Transcript will contain an array of alternates at that '
                'point, and will not be compatible with transcript_to_token '
                'until resolved. To suppress this warning, set warn=False'
                ''.format(utt_id))
        transcripts.append((utt_id, transcript))
    return transcripts


def write_trn(transcripts, trn):
    '''From a list of transcripts, write to a NIST "trn" file

    This is largely the inverse operation of :func:`read_trn`. In general,
    elements of a transcript (`transcripts` contains pairs of ``utt_id,
    transcript``) could be tokens or tuples of ``x, start, end`` (providing the
    start and end times of tokens, respectively). However, ``start`` and
    ``end`` are ignored when writing "trn" files. ``x`` could be the token or a
    list of alternates, as described in :func:`read_trn`

    Parameters
    ----------
    transcripts : sequence
    trn : file or str
    '''
    if isinstance(trn, basestring):
        with open(trn, 'w') as trn:
            return write_trn(transcripts, trn)

    def _handle_x(x):
        if isinstance(x, basestring):
            return x + ' '  # x was a token
        # x is a list of alternates
        ret = []
        for alts in x:
            elem = ''
            for xx in alts:
                elem += _handle_x(xx)
            ret.append(elem)
        ret = '{ ' + '/ '.join(ret) + '} '
        return ret
    for utt_id, transcript in transcripts:
        line = ''
        for x in transcript:
            # first get rid of starts and ends, if possible. This is not
            # ambiguous with numerical alternates, since alternates should
            # always be strings and, more importantly, always have placeholder
            # start and end values
            try:
                if (
                        len(x) == 3 and np.isreal(x[1]) and
                        np.isreal(x[2])):
                    x = x[0]
            except TypeError:
                pass
            line += _handle_x(x)
        trn.write(line)
        trn.write('(')
        trn.write(utt_id)
        trn.write(')\n')


def read_ctm(ctm, wc2utt=None):
    '''Read a NIST sclite "ctm" file into a list of transcriptions

    `sclite <http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm>`__
    is a commonly used scoring tool for ASR.

    This function converts a time-marked conversation file ("ctm" format) into
    a list of `transcripts`. Each element is a tuple of ``utt_id, transcript``,
    where ``transcript`` is itself a list of triples ``token, start, end``,
    ``token`` being a string, ``start`` being the start time of the token
    (in seconds), and ``end`` being the end time of the token (in seconds)

    Parameters
    ----------
    ctm : file or str
        The time-marked conversation file pointer. Will open if `ctm` is a
        path
    wc2utt : dict, optional
        "ctm" files identify utterances by waveform file name and channel. If
        specified, `wc2utt` consists of keys ``wfn, chan`` (e.g.
        ``'940328', 'A'``) to unique utterance IDs. If `wc2utt` is
        unspecified, the waveform file names are treated as the utterance IDs,
        and the channel is ignored

    Returns
    -------
    transcripts : list

    Notes
    -----
    "ctm", like "trn", has "support" for alternate transcriptions. It is, as of
    sclite version 2.10, very buggy. For example, it cannot handle multiple
    alternates in the same utterance. Plus, tools like `Kaldi
    <http://kaldi-asr.org/>`__ use the Unix command that the sclite
    documentation recommends to sort a ctm, ``sort +0 -1 +1 -2 +2nb -3``, which
    does not maintain proper ordering for alternate delimiters. Thus,
    :func:`read_ctm` will error if it comes across those delimiters
    '''
    if isinstance(ctm, str):
        with open(ctm, 'r') as ctm:
            return read_ctm(ctm, wc2utt)
    transcripts = OrderedDict()
    for line_no, line in enumerate(ctm):
        line = line.split(';;')[0].strip()
        if not line:
            continue
        line = line.split()
        try:
            if len(line) not in {5, 6}:
                raise ValueError()
            wfn, chan, start, dur, token = line[:5]
            if wc2utt is None:
                utt_id = wfn
            else:
                utt_id = wc2utt[(wfn, chan)]
            start = float(start)
            end = start + float(dur)
            if start < 0. or start > end:
                raise ValueError()
            transcripts.setdefault(utt_id, []).append((token, start, end))
        except ValueError:
            raise ValueError(
                'Could not parse line {} of ctm'.format(line_no + 1))
        except KeyError:
            raise KeyError(
                'ctm line {}: ({}, {}) was not found in wc2utt'.format(
                    line_no, wfn, chan))
    return [
        (utt_id, sorted(transcript, key=lambda x: x[1]))
        for utt_id, transcript in transcripts.items()
    ]


def write_ctm(transcripts, ctm, utt2wc='A'):
    '''From a list of transcripts, write to a NIST "ctm" file

    This is the inverse operation of :func:`read_ctm`. For each element of
    ``transcript`` within the ``utt_id, transcript`` pairs of elements in
    `transcripts`, ``token, start, end``, ``start`` and ``end`` must be
    non-negative

    Parameters
    ----------
    transcripts : sequence
    ctm : file or str
    utt2wc : dict or str, optional
        "ctm" files identify utterances by waveform file name and channel. If
        specified as a dict, `utt2wc` consists of utterance IDs as keys, and
        wavefile name and channels as values ``wfn, chan`` (e.g.
        ``'940328', 'A'``). If `utt2wc` is a string, each utterance IDs will
        be mapped to ``wfn`` and `utt2wc` as the channel
    '''
    if isinstance(ctm, str):
        with open(ctm, 'w') as ctm:
            return write_ctm(transcripts, ctm, utt2wc)
    is_dict = not isinstance(utt2wc, basestring)
    segments = []
    for utt_id, transcript in transcripts:
        try:
            wfn, chan = utt2wc[utt_id] if is_dict else (utt_id, utt2wc)
        except KeyError:
            raise KeyError('Utt "{}" has no value in utt2wc'.format(utt_id))
        for tup in transcript:
            if (
                    isinstance(tup, basestring) or
                    len(tup) != 3 or
                    tup[1] < 0. or
                    tup[2] < 0.):
                raise ValueError(
                    'Utt "{}" contains token "{}" with no timing info'
                    ''.format(utt_id, tup))
            token, start, end = tup
            duration = end - start
            if duration < 0.:
                raise ValueError(
                    'Utt "{}" contains token with negative duration'
                    ''.format(utt_id))
            segments.append((wfn, chan, start, duration, token))
    segments = sorted(segments)
    for segment in segments:
        ctm.write('{} {} {} {} {}\n'.format(*segment))


def transcript_to_token(transcript, token2id=None, frame_shift_ms=None):
    '''Convert a transcript to a SpectDataSet token sequence

    This method converts `transcript` of length ``R`` to a
    :class:`torch.LongTensor` `tok` of shape ``(R, 3)``, the latter suitable as
    a reference or hypothesis token sequence for an utterance of
    :class:`SpectDataSet`. An element of `transcript` can either be a ``token``
    or a 3-tuple of ``(token, start, end)``. ``id = token2id.get(token, token)
    if token2id is not None else token`` dictates the conversion from ``token``
    to identifier. If `frame_shift_ms` is specified, ``start`` and ``end`` are
    taken as the start and end times, in seconds, of the token, and will be
    converted to frames for `tok`. If `frame_shift_ms` is unspecified,
    ``start`` and ``end`` are assumed to already be frame times. If ``start``
    and ``end`` were unspecified, values of ``-1``, representing unknown, will
    be inserted into ``r[i, 1:]``

    Parameters
    ----------
    transcript : sequence
    token2id : dict, optional
    frame_shift_ms : int, optional

    Returns
    -------
    tok : torch.LongTensor
    '''
    tok = torch.empty((len(transcript), 3), dtype=torch.long)
    for i, token in enumerate(transcript):
        start = end = -1
        try:
            if len(token) == 3 and np.isreal(token[1]) and np.isreal(token[2]):
                token, start, end = token
                if frame_shift_ms:
                    start = (1000 * start) / frame_shift_ms
                    end = (1000 * end) / frame_shift_ms
                start, end = int(start), int(end)
        except TypeError:
            pass
        id = token2id.get(token, token) if token2id is not None else token
        tok[i, 0] = id
        tok[i, 1] = start
        tok[i, 2] = end
    return tok


def token_to_transcript(tok, id2token=None, frame_shift_ms=None):
    '''Convert a SpectDataSet token sequence to a transcript

    The inverse operation of :func:`transcript_to_token`

    Parameters
    ----------
    tok : torch.LongTensor
    id2token : dict, optional
    frame_shift_ms : int, optional

    Returns
    -------
    token : list
    '''
    transcript = []
    for tup in tok:
        id, start, end = tup.tolist()
        token = id2token.get(id, id) if id2token is not None else id
        if start == -1 or end == -1:
            transcript.append(token)
        else:
            if frame_shift_ms:
                start = start * frame_shift_ms / 1000
                end = end * frame_shift_ms / 1000
            transcript.append((token, start, end))
    return transcript


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
        If :obj:`True`, flip the window along the time/frame axis

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

    Like a :class:`SpectDataSet`, :class:`ContextWindowDataSet` indexes tuples
    of features and alignments. Instead of returning features of shape ``(T,
    F)``, instances return features of shape ``(T, 1 + left + right, F)``,
    where the ``T`` axis indexes the so-called center frame and the ``1 + left
    + right`` axis contains frame vectors (size ``F``) including the center
    frame, ``left`` frames in time before the center frame, and ``right``
    frames after.

    :class:`ContextWindowDataSet` does not have support for reference token
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
        If :obj:`True`, context windows will be reversed along the time
        dimension

    Attributes
    ----------
    data_dir, feat_subdir, ali_subdir, file_suffix : str
    left, right : int
    has_ali, reverse : bool
    utt_ids : tuple
    ref_subdir, eos : :obj:`None`
    has_ref : :obj:`False`

    Yields
    ------
    window : torch.FloatTensor
    ali : torch.LongTensor

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
    base_seed : int
    epoch : int
        The current epoch. Responsible for seeding the upcoming samples
    data_source : torch.data.utils.Dataset

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
    eos = param.Integer(
        None, doc='A special symbol used to indicate the end of a sequence in '
        'reference and hypothesis transcriptions. If set, `eos` will be '
        'appended to every reference transcription on read'
    )


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
    `feat_sizes` and `ref_sizes` are :class:`torch.LongTensor`s of shape
    ``(N,)`` containing the original ``T`` and ``R`` values. The batch will be
    sorted by decreasing numbers of frames. `feats` is zero-padded while `alis`
    and `refs` are padded with module constant
    :const:`pydrobert.torch.INDEX_PAD_VALUE`

    If ``ali`` or ``ref`` is :obj:`None` in any element, `alis` or `refs` and
    `ref_sizes` will also be :obj:`None`

    Parameters
    ----------
    seq : sequence

    Returns
    -------
    feats : torch.FloatTensor
    alis : torch.LongTensor or None
    refs : torch.LongTensor or None
    feat_sizes : torch.LongTensor
    ref_sizes : torch.LongTensor or None
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
            (len(seq), T_star), pydrobert.torch.INDEX_PAD_VALUE,
            dtype=torch.long)
    else:
        alis = None
    if has_ref:
        refs = torch.full(
            (len(seq), R_star, 3), pydrobert.torch.INDEX_PAD_VALUE,
            dtype=torch.long)
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
        feat_sizes if feat_sizes is None else torch.tensor(feat_sizes),
        ref_sizes if ref_sizes is None else torch.tensor(ref_sizes),
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
    feat_subdir : str, optional
    ali_subdir : str, optional
    ref_subdir : str, optional
    init_epoch : int, optional
        Where training should resume from
    kwargs : keyword arguments, optional
        Additional :class:`torch.utils.data.DataLoader` arguments

    Yields
    ------
    feats : torch.FloatTensor
        Of shape ``(N, T*, F)``, where ``N`` is ``params.batch_size``, ``T*``
        is the maximum number of frames in an utterance in the batch, and ``F``
        is the number of filters per frame
    alis : torch.LongTensor or None
        Of size ``(N, T*)`` if an ``ali/`` dir exists, otherwise :obj:`None`
    refs : torch.LongTensor or None
        Of size ``(N, R*, 3)``, where ``R*`` is the maximum number of reference
        tokens in the batch. If the ``refs/`` directory does not exist, `refs`
        and `ref_sizes` are :obj:`None`.
    feat_sizes : torch.LongTensor
        `feat_sizes` is a :class:`torch.LongTensor` of shape ``(N,)``
        specifying the lengths of utterances in the batch
    ref_sizes : torch.LongTensor or None
        `ref_sizes` is of shape ``(N,)`` specifying the number reference tokens
        per utterance in the batch. If the ``refs/`` directory does not exist,
        `refs` and `ref_sizes` are :obj:`None`.

    Attributes
    ----------
    params : SpectDataSetParams
    data_dir : str
    data_source : SpectDataSet
        The instance that will serve unbatched utterances
    epoch : int

    Notes
    -----
    The first axis of each of `feats`, `alis`, `refs`, `feat_sizes`, and
    `ref_sizes` is ordered by utterances of descending frame lengths. Shorter
    utterances in `feats` are zero-padded to the right, `alis` is padded with
    the module constant :const:`pydrobert.torch.INDEX_PAD_VALUE`

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
            eos=params.eos,
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
        Additional :class:`torch.utils.data.DataLoader` arguments

    Attributes
    ----------
    data_dir : str
    params : SpectDataSetParams
    data_source : SpectEvaluationDataLoader.SEvalDataSet
        Serves the utterances. A :class:`SpectDataSet`, but adds utterance IDs

    Yields
    ------
    feats : torch.FloatTensor
        Of shape ``(N, T*, F)``, where ``N`` is ``params.batch_size``, ``T*``
        is the maximum number of frames in an utterance in the batch, and ``F``
        is the number of filters per frame
    alis : torch.LongTensor or None
        Of size ``(N, T*)`` if an ``ali/`` dir exists, otherwise :obj:`None`
    refs : torch.LongTensor or None
        Of size ``(N, R*, 3)``, where ``R*`` is the maximum number of reference
        tokens in the batch. If the ``refs/`` directory does not exist, `refs`
        and `ref_sizes` are :obj:`None`.
    feat_sizes : torch.LongTensor
        `feat_sizes` is a :class:`torch.LongTensor` of shape ``(N,)``
        specifying the lengths of utterances in the batch
    ref_sizes : torch.LongTensor or None
        `ref_sizes` is of shape ``(N,)`` specifying the number reference tokens
        per utterance in the batch. If the ``refs/`` directory does not exist,
        `refs` and `ref_sizes` are :obj:`None`.
    utt_ids : tuple
        An ``N``-tuple specifying the names of utterances in the batch

    Notes
    -----

    Shorter utterances in `feats` are zero-padded to the right, `alis` and
    `refs` are padded with :const:`pydrobert.torch.INDEX_PAD_VALUE`

    Examples
    --------
    Computing class likelihoods and writing them to disk

    >>> # see 'SpectDataSet' to initialize data set
    >>> num_filts, num_ali_classes = 40, 100
    >>> model = torch.nn.LSTM(num_filts, num_ali_classes)
    >>> params = SpectDataSetParams()
    >>> loader = SpectEvaluationDataLoader('data', params)
    >>> for feats, _, _, feat_sizes, _, utt_ids in loader:
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
            eos=params.eos,
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

    If ``ali`` is :obj:`None` in any element, `alis` will also be :obj:`None`

    Parameters
    ----------
    seq : sequence

    Returns
    -------
    windows : list
    alis : list
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
        Additional :class:`torch.utils.data.DataLoader` arguments

    Attributes
    ----------
    epoch : int
    data_dir : str
    params : ContextWindowDataSetParams
    data_source : ContextWindowDataSet
        The instance that serves an utterance-worth of context windows

    Yields
    ------
    windows : torch.FloatTensor
        Of size ``(N, C, F)``, where ``N`` is the total number of context
        windows over all utterances in the batch, ``C`` is the context window
        size, and ``F`` is the number of filters per frame
    alis : torch.LongTensor or None
        Of size ``(N,)`` (or :obj:`None` if the ``ali`` dir was not specified)

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
        Additional :class:`torch.utils.data.DataLoader` arguments

    Attributes
    ----------
    data_dir : str
    params : SpectDataParams
    data_source : ContextWindowEvaluationDataLoader.CWEvalDataSet
        Serves utterances. A :class:`ContextWindowDataSet`, but adds feature
        sizes and utterance ids to each yielded tuple

    Yields
    ------
    windows : torch.FloatTensor
        Of size ``(N, C, F)``, where ``N`` is the number of context windows,
        ``C`` is the context window size, and ``F`` is the number of filters
        per frame
    alis : torch.LongTensor or None
        Of size ``(N,)`` (or :obj:`None` if the ``ali`` dir was not specified)
    win_sizes : torch.LongTensor
        Of shape ``(params.batch_size,)`` specifying the number of context
        windows per utterance in the batch.
        ``windows[sum(win_sizes[:i]):sum(win_sizes[:i+1])]`` are the context
        windows for the ``i``-th utterance in the batch (``sum(win_sizes) ==
        N``)
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
        return (windows, alis, torch.tensor(feat_sizes), tuple(utt_ids))

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
