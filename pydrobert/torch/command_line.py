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
    'get_torch_spect_data_dir_info',
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
        'to unique IDs. Each line has the format ``<token> <id>``'
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
    return parser.parse_args(args)


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
    if not os.path.isdir(options.dir):
        os.makedirs(options.dir)
    token2id = dict()
    for line_no, line in enumerate(options.token2id):
        line = line.strip()
        if not line:
            continue
        ls = line.split()
        if len(ls) != 2 or not ls[1].isnumeric():
            print(
                'Cannot parse line {} of {}'.format(
                    line_no + 1, options.token2id.name),
                file=sys.stderr,
            )
            return 1
        key, value = ls
        if key in token2id:
            warnings.warn(
                '{} line {}: "{}" already exists. Mapping will be '
                'ambiguous'.format(options.token2id.name, line_no + 1, key)
            )
        token2id[key] = int(value)
    transcripts = data.read_trn(options.trn)
    # we manually search for alternates in a first pass, as we don't know what
    # filters users have on warnings
    for utt_id, transcript in transcripts:
        old_transcript = transcript.copy()
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
    for utt_id, transcript in transcripts:
        tok = data.transcript_to_token(transcript, token2id)
        path = os.path.join(
            options.dir, options.file_prefix + utt_id + options.file_suffix)
        torch.save(tok, path)
    return 0
