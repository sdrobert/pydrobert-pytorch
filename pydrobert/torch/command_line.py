# Copyright 2019 Sean Robertson

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
import itertools

from collections import defaultdict, OrderedDict

import torch
import pydrobert.torch.data as data
import pydrobert.torch.util as util

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"
__all__ = [
    "compute_torch_token_data_dir_error_rates",
    "ctm_to_torch_token_data_dir",
    "get_torch_spect_data_dir_info",
    "torch_token_data_dir_to_ctm",
    "torch_token_data_dir_to_trn",
    "trn_to_torch_token_data_dir",
]


def _get_torch_spect_data_dir_info_parse_args(args):
    parser = argparse.ArgumentParser(description=get_torch_spect_data_dir_info.__doc__,)
    parser.add_argument("dir", type=str, help="The torch data directory")
    parser.add_argument(
        "out_file",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="The file to write to. If unspecified, stdout",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="If set, validate the data directory before collecting info. The "
        "process is described in pydrobert.torch.data.validate_spect_data_set",
    )
    parser.add_argument(
        "--file-prefix", default="", help="The file prefix indicating a torch data file"
    )
    parser.add_argument(
        "--file-suffix",
        default=".pt",
        help="The file suffix indicating a torch data file",
    )
    parser.add_argument(
        "--feat-subdir", default="feat", help="Subdirectory where features are stored"
    )
    parser.add_argument(
        "--ali-subdir", default="ali", help="Subdirectory where alignments are stored"
    )
    parser.add_argument(
        "--ref-subdir",
        default="ref",
        help="Subdirectory where reference token sequences are stored",
    )
    return parser.parse_args(args)


def get_torch_spect_data_dir_info(args=None):
    """Write info about the specified SpectDataSet data dir

    A torch :class:`pydrobert.torch.data.SpectDataSet` data dir is of the
    form::

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

    Where ``feat`` contains :class:`torch.FloatTensor` of shape ``(N, F)``,
    where ``N`` is the number of frames (variable) and ``F`` is the number of
    filters (fixed), ``ali``, if there, contains :class:`torch.LongTensor` of
    shape ``(N,)`` indicating the appropriate class labels (likely pdf-ids for
    discriminative training in an DNN-HMM), and ``ref``, if there, contains
    :class:`torch.LongTensor` of shape ``(R, 3)`` indicating a sequence of
    reference tokens where element indexed by ``[i, 0]`` is a token id, ``[i,
    1]`` is the inclusive start frame of the token (or a negative value if
    unknown), and ``[i, 2]`` is the exclusive end frame of the token.

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

    Note that the output can be parsed as a `Kaldi <http://kaldi-asr.org/>`__
    text table of integers.
    """
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
        "num_utterances": len(data_set),
        "total_frames": 0,
        "max_ali_class": -1,
        "max_ref_class": -1,
    }
    counts = dict()
    for feat, ali, ref in data_set:
        info_dict["num_filts"] = feat.size()[1]
        info_dict["total_frames"] += feat.size()[0]
        if ali is not None:
            for class_idx in ali:
                class_idx = class_idx.item()
                if class_idx < 0:
                    raise ValueError("Got a negative ali class idx")
                info_dict["max_ali_class"] = max(class_idx, info_dict["max_ali_class"])
                counts[class_idx] = counts.get(class_idx, 0) + 1
        if ref is not None:
            ref = ref[..., 0]
            if ref.min().item() < 0:
                raise ValueError("Got a negative ref class idx")
            info_dict["max_ref_class"] = max(
                info_dict["max_ref_class"], ref.max().item()
            )
    if info_dict["max_ali_class"] == 0:
        info_dict["count_0"] = counts[0]
    elif info_dict["max_ali_class"] > 0:
        count_fmt_str = "count_{{:0{}d}}".format(
            int(math.log10(info_dict["max_ali_class"])) + 1
        )
        for class_idx in range(info_dict["max_ali_class"] + 1):
            info_dict[count_fmt_str.format(class_idx)] = counts.get(class_idx, 0)
    info_list = sorted(info_dict.items())
    for key, value in info_list:
        options.out_file.write("{} {}\n".format(key, value))
    if options.out_file != sys.stdout:
        options.out_file.close()
    return 0


def _trn_to_torch_token_data_dir_parse_args(args):
    parser = argparse.ArgumentParser(description=trn_to_torch_token_data_dir.__doc__,)
    parser.add_argument("trn", type=argparse.FileType("r"), help="The input trn file")
    parser.add_argument(
        "token2id",
        type=argparse.FileType("r"),
        help="A file containing mappings from tokens (e.g. words or phones) "
        "to unique IDs. Each line has the format ``<token> <id>``. The flag "
        "``--swap`` can be used to swap the expected ordering (i.e. "
        "``<id> <token>``)",
    )
    parser.add_argument(
        "dir",
        help="The directory to store token sequences to. If the directory "
        "does not exist, it will be created",
    )
    parser.add_argument(
        "--alt-handler",
        default="error",
        choices=("error", "first"),
        help='How to handle transcription alternates. If "error", error if '
        'the "trn" file contains alternates. If "first", always treat the '
        "alternate as canon",
    )
    parser.add_argument(
        "--file-prefix", default="", help="The file prefix indicating a torch data file"
    )
    parser.add_argument(
        "--file-suffix",
        default=".pt",
        help="The file suffix indicating a torch data file",
    )
    parser.add_argument(
        "--swap",
        action="store_true",
        default=False,
        help="If set, swaps the order of key and value in `token2id`",
    )
    parser.add_argument(
        "--unk-symbol",
        default=None,
        help="If set, will map out-of-vocabulary tokens to this symbol",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=torch.multiprocessing.cpu_count(),
        help="The number of workers to spawn to process the data. 0 is serial."
        " Defaults to the cpu count",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="The number of lines that a worker will process at once. Impacts "
        "speed and memory consumption.",
    )
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--skip-frame-times",
        action="store_true",
        default=False,
        help="If true, will store token tensors of shape (R,) instead of "
        "(R, 3), foregoing segment start and end times (which trn does not "
        "have).",
    )
    size_group.add_argument(
        "--feat-sizing",
        action="store_true",
        default=False,
        help="If true, will store token tensors of shape (R, 1) instead of "
        "(R, 3), foregoing segment start and end times (which trn does not "
        "have). The extra dimension will allow data in this directory to be "
        "loaded as features in a SpectDataSet.",
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
        if len(ls) != 2 or not ls[1 - int(swap)].lstrip("-").isdigit():
            raise ValueError(
                "Cannot parse line {} of {}".format(line_no + 1, file.name)
            )
        key, value = ls
        key, value = (int(key), value) if swap else (key, int(value))
        if key in ret:
            warnings.warn(
                '{} line {}: "{}" already exists. Mapping will be ambiguous'
                "".format(file.name, line_no + 1, key)
            )
        if value in ret_swapped:
            warnings.warn(
                '{} line {}: "{}" already exists. Mapping will be ambiguous'
                "".format(file.name, line_no + 1, value)
            )
        ret[key] = value
        ret_swapped[value] = key
    return ret_swapped if return_swap else ret


def _save_transcripts_to_dir_worker(
    token2id,
    file_prefix,
    file_suffix,
    dir_,
    frame_shift_ms,
    unk,
    skip_frame_times,
    feat_sizing,
    queue,
    first_timeout=30,
    rest_timeout=10,
):
    transcripts = queue.get(True, first_timeout)
    while transcripts is not None:
        for utt_id, transcript in transcripts:
            tok = data.transcript_to_token(
                transcript,
                token2id,
                frame_shift_ms,
                unk,
                skip_frame_times or feat_sizing,
            )
            if feat_sizing:
                tok = tok.unsqueeze(-1)
            path = os.path.join(dir_, file_prefix + utt_id + file_suffix)
            torch.save(tok, path)
        transcripts = queue.get(True, rest_timeout)


def _save_transcripts_to_dir(
    transcripts,
    token2id,
    file_prefix,
    file_suffix,
    dir_,
    frame_shift_ms=None,
    unk=None,
    skip_frame_times=False,
    feat_sizing=False,
    num_workers=0,
    chunk_size=1000,
):
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    if num_workers:
        queue = torch.multiprocessing.Queue(num_workers)
        try:
            with torch.multiprocessing.Pool(
                num_workers,
                _save_transcripts_to_dir_worker,
                (
                    token2id,
                    file_prefix,
                    file_suffix,
                    dir_,
                    frame_shift_ms,
                    unk,
                    skip_frame_times,
                    feat_sizing,
                    queue,
                ),
            ) as pool:
                chunk = tuple(itertools.islice(transcripts, chunk_size))
                while len(chunk):
                    queue.put(chunk)
                    chunk = tuple(itertools.islice(transcripts, chunk_size))
                for _ in range(num_workers):
                    queue.put(None)
                pool.close()
                pool.join()
        except AttributeError:  # 2.7
            pool = torch.multiprocessing.Pool(
                num_workers,
                _save_transcripts_to_dir_worker,
                (
                    token2id,
                    file_prefix,
                    file_suffix,
                    dir_,
                    frame_shift_ms,
                    unk,
                    skip_frame_times,
                    queue,
                ),
            )
            try:
                chunk = tuple(itertools.islice(transcripts, chunk_size))
                while len(chunk):
                    queue.put(chunk)
                    chunk = tuple(itertools.islice(transcripts, chunk_size))
                for _ in range(num_workers):
                    queue.put(None)
                pool.close()
                pool.join()
            finally:
                pool.terminate()
    else:
        for utt_id, transcript in transcripts:
            tok = data.transcript_to_token(
                transcript,
                token2id,
                frame_shift_ms,
                unk,
                skip_frame_times or feat_sizing,
            )
            if feat_sizing:
                tok = tok.unsqueeze(-1)
            path = os.path.join(dir_, file_prefix + utt_id + file_suffix)
            torch.save(tok, path)


def trn_to_torch_token_data_dir(args=None):
    """Convert a NIST "trn" file to the specified SpectDataSet data dir

    A "trn" file is the standard transcription file without alignment
    information used in the `sclite
    <http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm>`__ toolkit.
    It has the format::

        here is a transcription (utterance_a)
        here is another (utterance_b)

    This command reads in a "trn" file and writes its contents as token
    sequences compatible with the ``ref/`` directory of a
    :class:`pydrobert.torch.data.SpectDataSet`. See the command
    :func:`get_torch_spect_data_dir_info` (command line
    "get-torch-spect-data-dir-info") for more information on a
    :class:`pydrobert.torch.data.SpectDataSet`
    """
    try:
        options = _trn_to_torch_token_data_dir_parse_args(args)
    except SystemExit as ex:
        return ex.code
    token2id = _parse_token2id(options.token2id, options.swap, options.swap)
    if options.unk_symbol is not None and options.unk_symbol not in token2id:
        print(
            'Unk symbol "{}" is not in token2id'.format(options.unk_symbol),
            file=sys.stderr,
        )
        return 1
    # we're going to do all the threading on the tensor creation part of
    # things
    transcripts = data.read_trn_iter(options.trn)

    def error_handling_iter():
        for utt_id, transcript in transcripts:
            old_transcript = transcript[:]
            transcript[:] = []
            while len(old_transcript):
                x = old_transcript.pop(0)
                if len(x) == 3 and x[1] == -1:
                    x = x[0]
                if isinstance(x, str):
                    transcript.append(x)
                elif options.alt_handler == "error":
                    raise ValueError('Cannot handle alternate in "{}"'.format(utt_id))
                else:  # first
                    x[0].extend(old_transcript)
                    old_transcript = x[0]
            yield utt_id, transcript

    _save_transcripts_to_dir(
        error_handling_iter(),
        token2id,
        options.file_prefix,
        options.file_suffix,
        options.dir,
        unk=options.unk_symbol,
        skip_frame_times=options.skip_frame_times,
        feat_sizing=options.feat_sizing,
        num_workers=options.num_workers,
        chunk_size=options.chunk_size,
    )
    return 0


def _torch_token_data_dir_to_trn_parse_args(args=None):
    parser = argparse.ArgumentParser(description=torch_token_data_dir_to_trn.__doc__)
    parser.add_argument("dir", help="The directory to read token sequences from")
    parser.add_argument(
        "id2token",
        type=argparse.FileType("r"),
        help="A file containing the mappings from unique IDs to tokens (e.g. "
        "words or phones). Each line has the format ``<id> <token>``. The "
        "flag ``--swap`` can be used to swap the expected ordering (i.e. "
        "``<token> <id>``)",
    )
    parser.add_argument(
        "trn",
        type=argparse.FileType("w"),
        help='The "trn" file to write transcriptions to',
    )
    parser.add_argument(
        "--file-prefix", default="", help="The file prefix indicating a torch data file"
    )
    parser.add_argument(
        "--file-suffix",
        default=".pt",
        help="The file suffix indicating a torch data file",
    )
    parser.add_argument(
        "--swap",
        action="store_true",
        default=False,
        help="If set, swaps the order of key and value in `id2token`",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=torch.multiprocessing.cpu_count(),
        help="The number of workers to spawn to process the data. 0 is serial."
        " Defaults to the cpu count",
    )
    return parser.parse_args(args)


class _TranscriptDataSet(torch.utils.data.Dataset):
    def __init__(
        self, dir_, id2token, file_prefix, file_suffix, frame_shift_ms, strip_timing
    ):
        super(_TranscriptDataSet, self).__init__()
        fpl = len(file_prefix)
        neg_fsl = -len(file_suffix)
        self.utt_ids = sorted(
            x[fpl:neg_fsl]
            for x in os.listdir(dir_)
            if x.startswith(file_prefix) and x.endswith(file_suffix)
        )
        self.dir_ = dir_
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.id2token = id2token
        self.frame_shift_ms = frame_shift_ms
        self.strip_timing = strip_timing

    def __getitem__(self, index):
        utt_id = self.utt_ids[index]
        tok = torch.load(
            os.path.join(self.dir_, self.file_prefix + utt_id + self.file_suffix)
        )
        transcript = data.token_to_transcript(tok, self.id2token, self.frame_shift_ms)
        for idx in range(len(transcript)):
            token = transcript[idx]
            if isinstance(token, tuple):
                token = token[0]
                if self.strip_timing:
                    transcript[idx] = token
            if isinstance(token, int) and self.id2token is not None:
                assert token not in self.id2token
                raise ValueError(
                    'Utterance "{}": ID "{}" could not be found in id2token'
                    "".format(utt_id, token)
                )
        return utt_id, transcript

    def __len__(self):
        return len(self.utt_ids)


def _noop_collate(x):
    return x[0]


def _load_transcripts_from_data_dir(
    dir_,
    id2token,
    file_prefix,
    file_suffix,
    frame_shift_ms=None,
    strip_timing=False,
    num_workers=0,
):
    ds = _TranscriptDataSet(
        dir_, id2token, file_prefix, file_suffix, frame_shift_ms, strip_timing
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=num_workers, collate_fn=_noop_collate
    )
    for utt_ids, transcripts in dl:
        yield utt_ids, transcripts
    del dl, ds


def torch_token_data_dir_to_trn(args=None):
    """Convert a SpectDataSet token data dir to a NIST trn file

    A "trn" file is the standard transcription file without alignment
    information used in the `sclite
    <http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm>`_
    toolkit. It has the format::

        here is a transcription (utterance_a)
        here is another (utterance_b)

    This command scans the contents of a directory like ``ref/`` in a
    :class:`pydrobert.torch.data.SpectDataSet` and converts each such file
    into a transcription. Each such transcription is then written to a "trn"
    file. See the command :func:`get_torch_spect_data_dir_info` (command line
    "get-torch-spect-data-dir-info") for more information on a
    :class:`pydrobert.torch.data.SpectDataSet`.
    """
    try:
        options = _torch_token_data_dir_to_trn_parse_args(args)
    except SystemExit as ex:
        return ex.code
    if not os.path.isdir(options.dir):
        print('"{}" is not a directory'.format(options.dir), file=sys.stderr)
        return 1
    id2token = _parse_token2id(options.id2token, not options.swap, options.swap)
    transcripts = _load_transcripts_from_data_dir(
        options.dir,
        id2token,
        options.file_prefix,
        options.file_suffix,
        strip_timing=True,
        num_workers=options.num_workers,
    )
    data.write_trn(transcripts, options.trn)
    return 0


def _ctm_to_torch_token_data_dir_parse_args(args):
    parser = argparse.ArgumentParser(description=ctm_to_torch_token_data_dir.__doc__)
    parser.add_argument(
        "ctm",
        type=argparse.FileType("r"),
        help='The "ctm" file to read token segments from',
    )
    parser.add_argument(
        "token2id",
        type=argparse.FileType("r"),
        help="A file containing mappings from tokens (e.g. words or phones) "
        "to unique IDs. Each line has the format ``<token> <id>``. The flag "
        "``--swap`` can be used to swap the expected ordering (i.e. "
        "``<id> <token>``)",
    )
    parser.add_argument(
        "dir",
        help="The directory to store token sequences to. If the "
        "directory does not exist, it will be created",
    )
    parser.add_argument(
        "--file-prefix", default="", help="The file prefix indicating a torch data file"
    )
    parser.add_argument(
        "--file-suffix",
        default=".pt",
        help="The file suffix indicating a torch data file",
    )
    parser.add_argument(
        "--swap",
        action="store_true",
        default=False,
        help="If set, swaps the order of key and value in `token2id`",
    )
    parser.add_argument(
        "--frame-shift-ms",
        type=float,
        default=10.0,
        help="The number of milliseconds that have passed between consecutive "
        "frames. Used to convert between time in seconds and frame index. If your "
        "features are the raw sample, set this to 1000 / sample_rate_hz",
    )
    utt_group = parser.add_mutually_exclusive_group()
    utt_group.add_argument(
        "--wc2utt",
        type=argparse.FileType("r"),
        default=None,
        help="A file mapping wavefile name and channel combinations (e.g. "
        "``utt_1 A``) to utterance IDs. Each line of the file has the format "
        '``<wavefile_name> <channel> <utt_id>``. If neither "--wc2utt" nor '
        '"--utt2wc" has been specied, the wavefile name will be treated as '
        "the utterance ID",
    )
    utt_group.add_argument(
        "--utt2wc",
        type=argparse.FileType("r"),
        default=None,
        help="A file mapping utterance IDs to wavefile name and channel "
        "combinations (e.g. ``utt_1 A``). Each line of the file has the "
        'format ``<utt_id> <wavefile_name> <channel>``. If neither "--wc2utt" '
        'nor "--utt2wc" has been specied, the wavefile name will be treated '
        "as the utterance ID",
    )
    parser.add_argument(
        "--unk-symbol",
        default=None,
        help="If set, will map out-of-vocabulary tokens to this symbol",
    )
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
            raise ValueError(
                "Cannot parse line {} of {}".format(line_no + 1, file.name),
            )
        first, mid, last = ls
        key, value = ((mid, last), first) if swap else ((first, mid), last)
        if key in ret:
            warnings.warn(
                '{} line {}: "{}" already exists. Mapping will be '
                "".format(file.name, line_no + 1, key)
            )
        if value in ret_swapped:
            warnings.warn(
                '{} line {}: "{}" already exists. Mapping will be '
                "".format(file.name, line_no + 1, value)
            )
        ret[key] = value
        ret_swapped[value] = key
    return ret_swapped if return_swap else ret


def ctm_to_torch_token_data_dir(args=None):
    """Convert a NIST "ctm" file to a SpectDataSet token data dir

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
    sequences compatible with the ``ref/`` directory of a
    :class:`pydrobert.torch.data.SpectDataSet`. See the command
    :func:`get_torch_spect_data_dir_info` (command line
    "get-torch-spect-data-dir-info") for more information on a
    :class:`pydrobert.torch.data.SpectDataSet`
    """
    try:
        options = _ctm_to_torch_token_data_dir_parse_args(args)
    except SystemExit as ex:
        return ex.code
    token2id = _parse_token2id(options.token2id, options.swap, options.swap)
    if options.unk_symbol is not None and options.unk_symbol not in token2id:
        print(
            'Unk symbol "{}" is not in token2id'.format(options.unk_symbol),
            file=sys.stderr,
        )
        return 1
    if options.wc2utt:
        wc2utt = _parse_wc2utt(options.wc2utt, False, False)
    elif options.utt2wc:
        wc2utt = _parse_wc2utt(options.utt2wc, True, False)
    else:
        wc2utt = None
    transcripts = data.read_ctm(options.ctm, wc2utt)
    _save_transcripts_to_dir(
        transcripts,
        token2id,
        options.file_prefix,
        options.file_suffix,
        options.dir,
        options.frame_shift_ms,
        options.unk_symbol,
    )
    return 0


def _torch_token_data_dir_to_ctm_parse_args(args):
    parser = argparse.ArgumentParser(description=torch_token_data_dir_to_ctm.__doc__)
    parser.add_argument("dir", help="The directory to read token sequences from")
    parser.add_argument(
        "id2token",
        type=argparse.FileType("r"),
        help="A file containing mappings from unique IDs to tokens (e.g. "
        "words or phones). Each line has the format ``<id> <token>``. The "
        "``--swap`` can be used to swap the expected ordering (i.e. "
        "``<token> <id>``)",
    )
    parser.add_argument(
        "ctm",
        type=argparse.FileType("w"),
        help='The "ctm" file to write token segments to',
    )
    parser.add_argument(
        "--file-prefix", default="", help="The file prefix indicating a torch data file"
    )
    parser.add_argument(
        "--file-suffix",
        default=".pt",
        help="The file suffix indicating a torch data file",
    )
    parser.add_argument(
        "--swap",
        action="store_true",
        default=False,
        help="If set, swaps the order of key and value in `id2token`",
    )
    parser.add_argument(
        "--frame-shift-ms",
        type=float,
        default=10.0,
        help="The number of milliseconds that have passed between consecutive "
        "frames. Used to convert between time in seconds and frame index. If your "
        "features are the raw samples, set this to 1000 / sample_rate_hz",
    )
    utt_group = parser.add_mutually_exclusive_group()
    utt_group.add_argument(
        "--wc2utt",
        type=argparse.FileType("r"),
        default=None,
        help="A file mapping wavefile name and channel combinations (e.g. "
        "``utt_1 A``) to utterance IDs. Each line of the file has the format "
        "``<wavefile_name> <channel> <utt_id>``",
    )
    utt_group.add_argument(
        "--utt2wc",
        type=argparse.FileType("r"),
        default=None,
        help="A file mapping utterance IDs to wavefile name and channel "
        "combinations (e.g. ``utt_1 A``). Each line of the file has the "
        "format ``<utt_id> <wavefile_name> <channel>``",
    )
    utt_group.add_argument(
        "--channel",
        default="A",
        help='If neither "--wc2utt" nor "--utt2wc" is specified, utterance '
        "IDs are treated as wavefile names and are given the value of this "
        "flag as a channel",
    )
    return parser.parse_args(args)


def torch_token_data_dir_to_ctm(args=None):
    """Convert a SpectDataSet token data directory to a NIST "ctm" file

    A "ctm" file is a transcription file with token alignments (a.k.a. a
    time-marked conversation file) used in the `sclite
    <http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm>`__ toolkit.
    Here is the format::

        utt_1 A 0.2 0.1 hi
        utt_1 A 0.3 1.0 there  ;; comment
        utt_2 A 0.0 1.0 next
        utt_3 A 0.1 0.4 utterance

    Where the first number specifies the token start time (in seconds) and the
    second the duration.

    This command scans the contents of a directory like ``ref/`` in a
    :class:`pydrobert.torch.data.SpectDataSet` and converts each such file into
    a transcription. Every token in a given transcription must have information
    about its duration. Each such transcription is then written to the "ctm"
    file. See the command :func:`get_torch_spect_data_dir_info` (command line
    "get-torch-spect-data-dir-info") for more information on a
    :class:`pydrobert.torch.data.SpectDataSet`
    """
    try:
        options = _torch_token_data_dir_to_ctm_parse_args(args)
    except SystemExit as ex:
        return ex.code
    id2token = _parse_token2id(options.id2token, not options.swap, options.swap)
    if options.wc2utt:
        utt2wc = _parse_wc2utt(options.wc2utt, False, True)
    elif options.utt2wc:
        utt2wc = _parse_wc2utt(options.utt2wc, True, True)
    else:
        utt2wc = options.channel
    transcripts = _load_transcripts_from_data_dir(
        options.dir,
        id2token,
        options.file_prefix,
        options.file_suffix,
        options.frame_shift_ms,
    )
    data.write_ctm(transcripts, options.ctm, utt2wc)
    return 0


def _compute_torch_token_data_dir_parse_args(args):
    parser = argparse.ArgumentParser(
        description=compute_torch_token_data_dir_error_rates.__doc__
    )
    parser.add_argument(
        "dir",
        help="If the `hyp` argument is not specified, this is the "
        "parent directory of two subdirectories, ``ref/`` and ``hyp/``, which "
        "contain the reference and hypothesis transcripts, respectively. If "
        "the ``--hyp`` argument is specified, this is the reference "
        "transcript directory",
    )
    parser.add_argument(
        "hyp", nargs="?", default=None, help="The hypothesis transcript directory",
    )
    parser.add_argument(
        "out",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Where to print the error rate to. Defaults to stdout",
    )
    parser.add_argument(
        "--id2token",
        type=argparse.FileType("r"),
        default=None,
        help="A file containing mappings from unique IDs to tokens (e.g. "
        "words or phones). Each line has the format ``<id> <token>``. The "
        "``--swap`` flag can be used to swap the expected ordering (i.e. "
        "``<token> <id>``). ``--id2token`` can be used to collapse unique IDs "
        "together. Also, ``--ignore`` will contain a list of strings instead "
        "of IDs",
    )
    parser.add_argument(
        "--replace",
        type=argparse.FileType("r"),
        default=None,
        help="A file containing pairs of elements per line. The first is the "
        "element to replace, the second what to replace it with. If "
        "``--id2token`` is specified, the file should contain tokens. If "
        "``--id2token`` is not specified, the file should contain IDs "
        "(integers). This is processed before ``--ignore``",
    )
    parser.add_argument(
        "--ignore",
        type=argparse.FileType("r"),
        default=None,
        help="A file containing a whitespace-delimited list of elements to "
        "ignore in both the reference and hypothesis transcripts. If "
        "``--id2token`` is specified, the file should contain tokens. If "
        "``--id2token`` is not specified, the file should contain IDs "
        "(integers). This is processed after ``--replace``",
    )
    parser.add_argument(
        "--file-prefix", default="", help="The file prefix indicating a torch data file"
    )
    parser.add_argument(
        "--file-suffix",
        default=".pt",
        help="The file suffix indicating a torch data file",
    )
    parser.add_argument(
        "--swap",
        action="store_true",
        default=False,
        help="If set, swaps the order of key and value in `id2token`",
    )
    parser.add_argument(
        "--warn-missing",
        action="store_true",
        default=False,
        help="If set, warn and exclude any utterances that are missing either "
        "a reference or hypothesis transcript. The default is to error",
    )
    parser.add_argument(
        "--distances",
        action="store_true",
        default=False,
        help="If set, do not normalize by reference lengths",
    )
    parser.add_argument(
        "--per-utt",
        action="store_true",
        default=False,
        help="If set, return lines of ``<utt_id> <error_rate>`` denoting the "
        "per-utterance error rates instead of the average",
    )
    parser.add_argument(
        "--ins-cost",
        type=float,
        default=1.0,
        help="The cost of an adding a superfluous token to a hypothesis " "transcript",
    )
    parser.add_argument(
        "--del-cost",
        type=float,
        default=1.0,
        help="The cost of missing a token from a reference transcript",
    )
    parser.add_argument(
        "--sub-cost",
        type=float,
        default=1.0,
        help="The cost of swapping a reference token with a hypothesis token",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="The number of error rates to compute at once. Reduce if you "
        "run into memory errors",
    )
    return parser.parse_args(args)


def compute_torch_token_data_dir_error_rates(args=None):
    """Compute error rates between reference and hypothesis token data dirs

    This is a very simple script that computes and prints the error rates
    between the ``ref/`` (reference/gold standard) token sequences and ``hyp/``
    (hypothesis/generated) token sequences in a
    :class:`pydrobert.torch.data.SpectDataSet` directory. An error rate is
    merely a `Levenshtein Distance
    <https://en.wikipedia.org/wiki/Levenshtein_distance>`__ normalized to the
    reference sequence length.

    While convenient and accurate, this script has very few features. Consider
    pairing the command ``torch-token-data-dir-to-trn`` with `sclite
    <http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm>`__ instead.

    Many tasks will ignore some tokens (e.g. silences) or collapse others (e.g.
    phones). Please consult a standard recipe (such as those in `Kaldi
    <http://kaldi-asr.org/>`__) before performing these computations
    """
    try:
        options = _compute_torch_token_data_dir_parse_args(args)
    except SystemExit as ex:
        return ex.code
    if options.hyp:
        ref_dir, hyp_dir = options.dir, options.hyp
    else:
        ref_dir = os.path.join(options.dir, "ref")
        hyp_dir = os.path.join(options.dir, "hyp")
    if not os.path.isdir(ref_dir):
        print('"{}" is not a directory'.format(ref_dir), file=sys.stderr)
        return 1
    if not os.path.isdir(hyp_dir):
        print('"{}" is not a directory'.format(hyp_dir), file=sys.stderr)
        return 1
    if options.id2token:
        id2token = _parse_token2id(options.id2token, not options.swap, options.swap)
    else:
        id2token = None
    replace = dict()
    if options.replace:
        for line in options.replace:
            replaced, replacement = line.strip().split()
            if id2token is None:
                try:
                    replaced, replacement = int(replaced), int(replacement)
                except ValueError:
                    raise ValueError(
                        'If --id2token is not set, all elements in "{}" must '
                        "be integers".format(options.replace.name)
                    )
            replace[replaced] = replacement
    if options.ignore:
        ignore = set(options.ignore.read().strip().split())
        if id2token is None:
            try:
                ignore = {int(x) for x in ignore}
            except ValueError:
                raise ValueError(
                    'If --id2token is not set, all elements in "{}" must be '
                    "integers".format(options.ignore.name)
                )
    else:
        ignore = set()
    ref_transcripts = list(
        _load_transcripts_from_data_dir(
            ref_dir,
            id2token,
            options.file_prefix,
            options.file_suffix,
            strip_timing=True,
        )
    )
    hyp_transcripts = list(
        _load_transcripts_from_data_dir(
            hyp_dir,
            id2token,
            options.file_prefix,
            options.file_suffix,
            strip_timing=True,
        )
    )
    idx = 0
    while idx < max(len(ref_transcripts), len(hyp_transcripts)):
        missing_ref = missing_hyp = False
        if idx == len(ref_transcripts):
            missing_hyp = True
        elif idx == len(hyp_transcripts):
            missing_ref = True
        elif ref_transcripts[idx][0] < hyp_transcripts[idx][0]:
            missing_ref = True
        elif hyp_transcripts[idx][0] < ref_transcripts[idx][0]:
            missing_hyp = True
        if missing_hyp or missing_ref:
            if missing_hyp:
                fmt_tup = hyp_dir, hyp_transcripts[idx][0], ref_dir
                del hyp_transcripts[idx]
            else:
                fmt_tup = ref_dir, ref_transcripts[idx][0], hyp_dir
                del ref_transcripts[idx]
            msg = (
                'Directory "{}" contains utterance "{}" which directory "{}" '
                "does not contain"
            ).format(*fmt_tup)
            if options.warn_missing:
                warnings.warn(msg + ". Skipping")
            else:
                raise ValueError(msg)
        else:
            idx += 1
    assert len(ref_transcripts) == len(hyp_transcripts)
    idee_, eos, padding = [0], -1, -2

    def get_idee():
        v = idee_[0]
        idee_[0] += 1
        return v

    token2id = defaultdict(get_idee)
    error_rates = OrderedDict()
    while len(ref_transcripts):
        batch_ref_transcripts = ref_transcripts[: options.batch_size]
        batch_hyp_transcripts = hyp_transcripts[: options.batch_size]
        ref_transcripts = ref_transcripts[options.batch_size :]
        hyp_transcripts = hyp_transcripts[options.batch_size :]
        ref = torch.nn.utils.rnn.pad_sequence(
            [
                torch.tensor(
                    [
                        token2id[replace.get(token, token)]
                        for token in transcript
                        if replace.get(token, token) not in ignore
                    ]
                    + [eos]
                )
                for _, transcript in batch_ref_transcripts
            ],
            padding_value=padding,
        )
        hyp = torch.nn.utils.rnn.pad_sequence(
            [
                torch.tensor(
                    [
                        token2id[replace.get(token, token)]
                        for token in transcript
                        if replace.get(token, token) not in ignore
                    ]
                    + [eos]
                )
                for _, transcript in batch_hyp_transcripts
            ],
            padding_value=padding,
        )
        ers = util.error_rate(
            ref,
            hyp,
            eos=eos,
            include_eos=False,
            ins_cost=options.ins_cost,
            del_cost=options.del_cost,
            sub_cost=options.sub_cost,
            norm=not options.distances,
        )
        for (utt_id, _), er in zip(batch_ref_transcripts, ers):
            error_rates[utt_id] = er.item()
    if options.per_utt:
        for utt_id, er in error_rates.items():
            options.out.write("{} {}\n".format(utt_id, er))
    else:
        options.out.write("{}\n".format(sum(error_rates.values()) / len(error_rates)))
    return 0
