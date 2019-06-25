# Copyright 2018 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import math
import warnings

import torch
import pydrobert.torch.data as data

from past.builtins import basestring


__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"
__all__ = [
    'ctm_to_torch_token_data_dir',
    'get_torch_spect_data_dir_info',
    'torch_token_data_dir_to_trn',
    'trn_to_torch_token_data_dir',
]


def _get_torch_spect_data_dir_info_parse_args(args):
    parser = argparse.ArgumentParser(
        description=get_torch_spect_data_dir_info.__doc__,
    )
    parser.add_argument('dir', type=str, help='The torch data directory')
    parser.add_argument(
        'out_file', nargs='?', type=argparse.FileType('w'),
        default=sys.stdout,
        help='The file to write to. If unspecified, stdout',
    )
    parser.add_argument(
        '--strict', action='store_true', default=False,
        help='If set, validate the data directory before collecting info. The '
        'process is described in pydrobert.torch.data.validate_spect_data_set'
    )
    parser.add_argument(
        '--file-prefix', default='',
        help='The file prefix indicating a torch data file'
    )
    parser.add_argument(
        '--file-suffix', default='.pt',
        help='The file suffix indicating a torch data file'
    )
    parser.add_argument(
        '--feat-subdir', default='feat',
        help='Subdirectory where features are stored'
    )
    parser.add_argument(
        '--ali-subdir', default='ali',
        help='Subdirectory where alignments are stored'
    )
    parser.add_argument(
        '--ref-subdir', default='ref',
        help='Subdirectory where reference token sequences are stored'
    )
    return parser.parse_args(args)


def get_torch_spect_data_dir_info(args=None):
    '''Write info about the specified SpectDataSet data dir

    A torch ``SpectDataSet`` data dir is of the form::

        dir/
            feat/
                <file_prefix><utt1><file_suffix>
                <file_prefix><utt2><file_suffix>
                ...
            [ali/
                <file_prefix><utt1><file_suffix>
                <file_prefix><utt1><file_suffix>
                ...
            ]
            [ref/
                <file_prefix><utt1><file_suffix>
                <file_prefix><utt1><file_suffix>
                ...
            ]

    Where ``feat`` contains ``FloatTensor``s of shape ``(N, F)``, where
    ``N`` is the number of frames (variable) and ``F`` is the number of
    filters (fixed), ``ali``, if there, contains ``LongTensor``s of shape
    ``(N,)`` indicating the appropriate class labels (likely pdf-ids for
    discriminative training in an DNN-HMM), and ``ref``, if there,
    contains ``LongTensor``s of shape ``(R, 3)`` indicating a sequence of
    reference tokens where element indexed by ``[i, 0]`` is a token id,
    ``[i, 1]`` is the inclusive start frame of the token (or a negative value
    if unknown), and ``[i, 2]`` is the exclusive end frame of the token.

    This command writes the following space-delimited key-value pairs to an
    output file in sorted order:

    1. "max_ali_class", the maximum inclusive class id found over ``ali/``
        (if available, ``-1`` if not)
    2. "max_ref_class", the maximum inclussive class id found over ``ref/``
        (if available, ``-1`` if not)
    3. "num_utterances", the total number of listed utterances
    4. "num_filts", ``F``
    5. "total_frames", ``sum(N)`` over the data dir
    6. "count_<i>", the number of instances of the class "<i>" that appear
        in ``ali`` (if available). If "count_<i>" is a valid key, then so
        are "count_<0 to i>". "count_<i>" is left-padded with zeros to ensure
        that the keys remain in the same order in the table as the class
        indices.  The maximum ``i`` will be equal to ``maximum_ali_class``

    Note that the output can be parsed as a Kaldi text table of integers.
    '''
    try:
        options = _get_torch_spect_data_dir_info_parse_args(args)
    except SystemExit as ex:
        return ex.code
    if not os.path.isdir(options.dir):
        print("'{}' is not a directory".format(options.dir), file=sys.stderr)
        return 1
    data_set = data.SpectDataSet(
        options.dir,
        file_prefix=options.file_prefix,
        file_suffix=options.file_suffix,
        feat_subdir=options.feat_subdir,
        ali_subdir=options.ali_subdir,
        ref_subdir=options.ref_subdir,
    )
    if options.strict:
        data.validate_spect_data_set(data_set)
    info_dict = {
        'num_utterances': len(data_set),
        'total_frames': 0,
        'max_ali_class': -1,
        'max_ref_class': -1,
    }
    counts = dict()
    for feat, ali, ref in data_set:
        info_dict['num_filts'] = feat.size()[1]
        info_dict['total_frames'] += feat.size()[0]
        if ali is not None:
            for class_idx in ali:
                class_idx = class_idx.item()
                if class_idx < 0:
                    raise ValueError('Got a negative ali class idx')
                info_dict['max_ali_class'] = max(
                    class_idx, info_dict['max_ali_class'])
                counts[class_idx] = counts.get(class_idx, 0) + 1
        if ref is not None:
            if ref.min().item() < 0:
                raise ValueError('Got a negative ref class idx')
            info_dict['max_ref_class'] = max(
                info_dict['max_ref_class'], ref.max().item())
    if info_dict['max_ali_class'] == 0:
        info_dict['count_0'] = counts[0]
    elif info_dict['max_ali_class'] > 0:
        count_fmt_str = 'count_{{:0{}d}}'.format(
            int(math.log10(info_dict['max_ali_class'])) + 1)
        for class_idx in range(info_dict['max_ali_class'] + 1):
            info_dict[count_fmt_str.format(class_idx)] = counts.get(
                class_idx, 0)
    info_list = sorted(info_dict.items())
    for key, value in info_list:
        options.out_file.write("{} {}\n".format(key, value))
    if options.out_file != sys.stdout:
        options.out_file.close()
    return 0


def _trn_to_torch_token_data_dir_parse_args(args):
    parser = argparse.ArgumentParser(
        description=trn_to_torch_token_data_dir.__doc__,
    )
    parser.add_argument(
        'trn', type=argparse.FileType('r'),
        help='The input trn file'
    )
    parser.add_argument(
        'token2id', type=argparse.FileType('r'),
        help='A file containing mappings from tokens (e.g. words or phones) '
        'to unique IDs. Each line has the format ``<token> <id>``. The flag '
        '``--swap`` can be used to swap the expected ordering (i.e. '
        '``<id> <token>``)'
    )
    parser.add_argument(
        'dir',
        help='The directory to store token sequences to. If the directory '
        'does not exist, it will be created'
    )
    parser.add_argument(
        '--alt-handler', default='error', choices=('error', 'first'),
        help='How to handle transcription alternates. If "error", error if '
        'the "trn" file contains alternates. If "first", always treat the '
        'alternate as canon'
    )
    parser.add_argument(
        '--file-prefix', default='',
        help='The file prefix indicating a torch data file'
    )
    parser.add_argument(
        '--file-suffix', default='.pt',
        help='The file suffix indicating a torch data file'
    )
    parser.add_argument(
        '--swap', action='store_true', default=False,
        help='If set, swaps the order of key and value in `token2id`'
    )
    return parser.parse_args(args)


def _parse_token2id(file, swap, return_swap):
    ret = dict()
    ret_swapped = dict()
    for line_no, line in enumerate(file):
        line = line.strip()
        if not line:
            continue
        ls = line.split()
        if len(ls) != 2 or not ls[1 - int(swap)].isdigit():
            print(
                'Cannot parse line {} of {}'.format(line_no + 1, file.name),
                file=sys.stderr,
            )
            return 1
        key, value = ls
        key, value = (int(key), value) if swap else (key, int(value))
        if key in ret:
            warnings.warn(
                '{} line {}: "{}" already exists. Mapping will be ambiguous'
                ''.format(file.name, line_no + 1, key))
        if value in ret_swapped:
            warnings.warn(
                '{} line {}: "{}" already exists. Mapping will be ambiguous'
                ''.format(file.name, line_no + 1, value))
        ret[key] = value
        ret_swapped[value] = key
    return ret_swapped if return_swap else ret


def _save_transcripts_to_dir(
        transcripts, token2id, file_prefix, file_suffix, dir_,
        frame_shift_ms=None):
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    for utt_id, transcript in transcripts:
        tok = data.transcript_to_token(transcript, token2id, frame_shift_ms)
        path = os.path.join(dir_, file_prefix + utt_id + file_suffix)
        torch.save(tok, path)


def trn_to_torch_token_data_dir(args=None):
    '''Convert a NIST "trn" file to the specified SpectDataSet data dir

    A "trn" file is the standard transcription file without alignment
    information used in the `sclite
    <http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm>`_
    toolkit. It has the format::

        here is a transcription (utterance_a)
        here is another (utterance_b)

    This command reads in a "trn" file and writes its contents as token
    sequences compatible with the ``ref/`` directory of a ``SpectDataSet``.
    See the command ``get_torch_spect_data_dir_info`` (command line
    "get-torch-spect-data-dir-info") for more information on a
    ``SpectDataSet``
    '''
    try:
        options = _trn_to_torch_token_data_dir_parse_args(args)
    except SystemExit as ex:
        return ex.code
    token2id = _parse_token2id(
        options.token2id, options.swap, options.swap)
    transcripts = data.read_trn(options.trn)
    # we manually search for alternates in a first pass, as we don't know what
    # filters users have on warnings
    for utt_id, transcript in transcripts:
        old_transcript = transcript[:]
        transcript[:] = []
        while len(old_transcript):
            x = old_transcript.pop(0)
            if len(x) == 3 and x[1] == -1:
                x = x[0]
            if isinstance(x, basestring):
                transcript.append(x)
            elif options.alt_handler == 'error':
                print(
                    'Cannot handle alternate in "{}"'.format(utt_id),
                    file=sys.stderr)
                return 1
            else:  # first
                x[0].extend(old_transcript)
                old_transcript = x[0]
    _save_transcripts_to_dir(
        transcripts, token2id, options.file_prefix, options.file_suffix,
        options.dir)
    return 0


def _torch_token_data_dir_to_trn_parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=torch_token_data_dir_to_trn.__doc__)
    parser.add_argument(
        'dir', help='The directory to read token sequences from')
    parser.add_argument(
        'id2token', type=argparse.FileType('r'),
        help='A file containing the mappings from unique IDs to tokens (e.g. '
        'words or phones). Each line has the format ``<id> <token>``. The '
        'flag ``--swap`` can be used to swap the expected ordering (i.e. '
        '``<token> <id>``)'
    )
    parser.add_argument(
        "trn", type=argparse.FileType('w'),
        help='The "trn" file to write transcriptions to'
    )
    parser.add_argument(
        '--file-prefix', default='',
        help='The file prefix indicating a torch data file'
    )
    parser.add_argument(
        '--file-suffix', default='.pt',
        help='The file suffix indicating a torch data file'
    )
    parser.add_argument(
        '--swap', action='store_true', default=False,
        help='If set, swaps the order of key and value in `id2token`'
    )
    return parser.parse_args(args)


def _load_transcripts_from_data_dir(
        dir_, id2token, file_prefix, file_suffix, frame_shift_ms=None):
    fpl = len(file_prefix)
    neg_fsl = -len(file_suffix)
    utt_ids = sorted(
        x[fpl:neg_fsl]
        for x in os.listdir(dir_)
        if x.startswith(file_prefix) and
        x.endswith(file_suffix)
    )
    transcripts = []
    for utt_id in utt_ids:
        tok = torch.load(os.path.join(
            dir_, file_prefix + utt_id + file_suffix))
        transcript = data.token_to_transcript(tok, id2token, frame_shift_ms)
        for token in transcript:
            if len(token) == 3 and isinstance(token[1], int):
                token = token[0]
            if isinstance(token, int):
                assert token not in id2token
                print(
                    'Utterance "{}": ID "{}" could not be found in id2token'
                    ''.format(utt_id, token))
        transcripts.append((utt_id, transcript))
    return transcripts


def torch_token_data_dir_to_trn(args=None):
    '''Convert a SpectDataSet token data dir to a NIST trn file

    A "trn" file is the standard transcription file without alignment
    information used in the `sclite
    <http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm>`_
    toolkit. It has the format::

        here is a transcription (utterance_a)
        here is another (utterance_b)

    This command scans the contents of a directory like ``ref/`` in a
    ``SpectDataSet`` and converts each such file into a transcription. Each
    such transcription is then written to a "trn" file. See the command
    ``get_torch_spect_data_dir_info`` (command line
    "get-torch-spect-data-dir-info") for more information on a ``SpectDataSet``
    '''
    try:
        options = _torch_token_data_dir_to_trn_parse_args(args)
    except SystemExit as ex:
        return ex.code
    if not os.path.isdir(options.dir):
        print('"{}" is not a directory'.format(options.dir), file=sys.stderr)
        return 1
    id2token = _parse_token2id(
        options.id2token, not options.swap, options.swap)
    transcripts = _load_transcripts_from_data_dir(
        options.dir, id2token, options.file_prefix, options.file_suffix)
    data.write_trn(transcripts, options.trn)
    return 0


def _ctm_to_torch_token_data_dir_parse_args(args):
    parser = argparse.ArgumentParser(
        description=ctm_to_torch_token_data_dir.__doc__)
    parser.add_argument(
        'ctm', type=argparse.FileType('r'),
        help='The "ctm" file to read token segments from')
    parser.add_argument(
        'token2id', type=argparse.FileType('r'),
        help='A file containing mappings from tokens (e.g. words or phones) '
        'to unique IDs. Each line has the format ``<token> <id>``. The flag '
        '``--swap`` can be used to swap the expected ordering (i.e. '
        '``<id> <token>``)'
    )
    parser.add_argument(
        'dir', help='The directory to store token sequences to. If the '
        'directory does not exist, it will be created'
    )
    parser.add_argument(
        '--file-prefix', default='',
        help='The file prefix indicating a torch data file'
    )
    parser.add_argument(
        '--file-suffix', default='.pt',
        help='The file suffix indicating a torch data file'
    )
    parser.add_argument(
        '--swap', action='store_true', default=False,
        help='If set, swaps the order of key and value in `token2id`'
    )
    parser.add_argument(
        '--frame-shift-ms', type=int, default=10,
        help='The number of milliseconds that have passed between consecutive '
        'frames. Used to convert between time in seconds and frame index'
    )
    utt_group = parser.add_mutually_exclusive_group()
    utt_group.add_argument(
        '--wc2utt', type=argparse.FileType('r'), default=None,
        help='A file mapping wavefile name and channel combinations (e.g. '
        '``utt_1 A``) to utterance IDs. Each line of the file has the format '
        '``<wavefile_name> <channel> <utt_id>``. If neither "--wc2utt" nor '
        '"--utt2wc" has been specied, the wavefile name will be treated as '
        'the utterance ID')
    utt_group.add_argument(
        '--utt2wc', type=argparse.FileType('r'), default=None,
        help='A file mapping utterance IDs to wavefile name and channel '
        'combinations (e.g. ``utt_1 A``). Each line of the file has the '
        'format ``<utt_id> <wavefile_name> <channel>``. If neither "--wc2utt" '
        'nor "--utt2wc" has been specied, the wavefile name will be treated '
        'as the utterance ID')
    return parser.parse_args(args)


def _parse_wc2utt(file, swap, return_swap):
    ret = dict()
    ret_swapped = dict()
    for line_no, line in enumerate(file):
        line = line.strip()
        if not line:
            continue
        ls = line.split()
        if len(ls) != 3:
            print(
                'Cannot parse line {} of {}'.format(line_no + 1, file.name),
                file=sys.stderr,
            )
            return 1
        first, mid, last = ls
        key, value = ((mid, last), first) if swap else ((first, mid), last)
        if key in ret:
            warnings.warn(
                '{} line {}: "{}" already exists. Mapping will be '
                ''.format(file.name, line_no + 1, key)
            )
        if value in ret_swapped:
            warnings.warn(
                '{} line {}: "{}" already exists. Mapping will be '
                ''.format(file.name, line_no + 1, value)
            )
        ret[key] = value
        ret_swapped[value] = key
    return ret_swapped if return_swap else ret


def ctm_to_torch_token_data_dir(args=None):
    '''Convert a NIST "ctm" file to a SpectDataSet token data dir

    A "ctm" file is a transcription file with token alignments (a.k.a. a
    time-marked conversation file) used in the `sclite
    <http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm>`_ toolkit.
    Here is the format::

        utt_1 A 0.2 0.1 hi
        utt_1 A 0.3 1.0 there  ;; comment
        utt_2 A 0.0 1.0 next
        utt_3 A 0.1 0.4 utterance

    Where the first number specifies the token start time (in seconds) and the
    second the duration.

    This command reads in a "ctm" file and writes its contents as token
    sequences compatible with the ``ref/`` directory of a ``SpectDataSet``.
    See the command ``get_torch_spect_data_dir_info`` (command line
    "get-torch-spect-data-dir-info") for more information on a
    ``SpectDataSet``
    '''
    try:
        options = _ctm_to_torch_token_data_dir_parse_args(args)
    except SystemExit as ex:
        return ex.code
    token2id = _parse_token2id(
        options.token2id, options.swap, options.swap)
    if options.wc2utt:
        wc2utt = _parse_wc2utt(options.wc2utt, False, False)
    elif options.utt2wc:
        wc2utt = _parse_wc2utt(options.utt2wc, True, False)
    else:
        wc2utt = None
    transcripts = data.read_ctm(options.ctm, wc2utt)
    _save_transcripts_to_dir(
        transcripts, token2id, options.file_prefix, options.file_suffix,
        options.dir, options.frame_shift_ms)
    return 0


def _torch_token_data_dir_to_ctm_parse_args(args):
    parser = argparse.ArgumentParser(
        description=torch_token_data_dir_to_ctm.__doc__)
    parser.add_argument(
        'dir', help='The directory to read token sequences from')
    parser.add_argument(
        'id2token', type=argparse.FileType('r'),
        help='A file containing mappings from unique IDs to tokens (e.g. '
        'words or phones). Each line has the format ``<id> <token>``. The '
        '``--swap`` can be used to swap the expected ordering (i.e. '
        '``<token> <id>``)'
    )
    parser.add_argument(
        'ctm', type=argparse.FileType('w'),
        help='The "ctm" file to write token segments to')
    parser.add_argument(
        '--file-prefix', default='',
        help='The file prefix indicating a torch data file'
    )
    parser.add_argument(
        '--file-suffix', default='.pt',
        help='The file suffix indicating a torch data file'
    )
    parser.add_argument(
        '--swap', action='store_true', default=False,
        help='If set, swaps the order of key and value in `id2token`'
    )
    parser.add_argument(
        '--frame-shift-ms', type=int, default=10,
        help='The number of milliseconds that have passed between consecutive '
        'frames. Used to convert between time in seconds and frame index'
    )
    utt_group = parser.add_mutually_exclusive_group()
    utt_group.add_argument(
        '--wc2utt', type=argparse.FileType('r'), default=None,
        help='A file mapping wavefile name and channel combinations (e.g. '
        '``utt_1 A``) to utterance IDs. Each line of the file has the format '
        '``<wavefile_name> <channel> <utt_id>``')
    utt_group.add_argument(
        '--utt2wc', type=argparse.FileType('r'), default=None,
        help='A file mapping utterance IDs to wavefile name and channel '
        'combinations (e.g. ``utt_1 A``). Each line of the file has the '
        'format ``<utt_id> <wavefile_name> <channel>``')
    utt_group.add_argument(
        '--channel', default='A',
        help='If neither "--wc2utt" nor "--utt2wc" is specified, utterance '
        'IDs are treated as wavefile names and are given the value of this '
        'flag as a channel'
    )
    return parser.parse_args(args)


def torch_token_data_dir_to_ctm(args=None):
    '''Convert a ``SpectDataSet`` token data directory to a NIST "ctm" file

    A "ctm" file is a transcription file with token alignments (a.k.a. a
    time-marked conversation file) used in the `sclite
    <http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm>`_ toolkit.
    Here is the format::

        utt_1 A 0.2 0.1 hi
        utt_1 A 0.3 1.0 there  ;; comment
        utt_2 A 0.0 1.0 next
        utt_3 A 0.1 0.4 utterance

    Where the first number specifies the token start time (in seconds) and the
    second the duration.

    This command scans the contents of a directory like ``ref/`` in a
    ``SpectDataSet`` and converts each such file into a transcription. Every
    token in a given transcription must have information about its duration.
    Each such transcription is then written to the "ctm" file. See the command
    ``get_torch_spect_data_dir_info`` (command line
    "get-torch-spect-data-dir-info") for more information on a ``SpectDataSet``
    '''
    try:
        options = _torch_token_data_dir_to_ctm_parse_args(args)
    except SystemExit as ex:
        return ex.code
    id2token = _parse_token2id(
        options.id2token, not options.swap, options.swap)
    if options.wc2utt:
        utt2wc = _parse_wc2utt(options.wc2utt, False, True)
    elif options.utt2wc:
        utt2wc = _parse_wc2utt(options.utt2wc, True, True)
    else:
        utt2wc = options.channel
    transcripts = _load_transcripts_from_data_dir(
        options.dir, id2token, options.file_prefix, options.file_suffix,
        options.frame_shift_ms)
    data.write_ctm(transcripts, options.ctm, utt2wc)
    return 0
