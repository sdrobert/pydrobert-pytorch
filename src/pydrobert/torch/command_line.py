# Copyright 2021 Sean Robertson

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
import sys
import argparse
import math
import warnings
import itertools
import tarfile
import io

from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Optional, Sequence

import torch

import pydrobert.torch.data as data
import pydrobert.torch.modules as modules
from ._string import error_rate


_COMMON_ARGS = {
    "--file-prefix": {
        "default": "",
        "help": "The file prefix indicating a torch data file",
    },
    "--file-suffix": {
        "default": ".pt",
        "help": "The file suffix indicating a torch data file",
    },
    "token2id": {
        "type": argparse.FileType("r"),
        "help": "A file containing mappings from tokens (e.g. words or phones) to "
        'unique IDs. Each line has the format "<token> <id>". The flag "--swap" can be '
        'used to swap the expected ordering (i.e. to "<id> <token>")',
    },
    "id2token": {
        "type": argparse.FileType("r"),
        "help": "A file containing mappings from unique IDs to tokens (e.g. words or "
        'phones). Each line has the format "<id> <token>". The flag "--swap" can be '
        'used to swap the expected ordering (i.e. to "<token> <id>")',
    },
    "--num-workers": {
        "type": int,
        "default": torch.multiprocessing.cpu_count(),
        "help": "The number of workers to spawn to process the data. 0 is serial. "
        "Defaults to the CPU count",
    },
    "--swap": {
        "action": "store_true",
        "default": False,
        "help": "If set, swaps the order of the key and value in token/id mapping",
    },
    "--unk-symbol": {
        "default": None,
        "help": "If set, will map out-of-vocabulary tokens to this symbol",
    },
    "--frame-shift-ms": {
        "type": float,
        "default": 10.0,
        "help": "The number of milliseconds that have passed between consecutive "
        "frames. Used to convert between time in seconds and frame index. If your "
        "features are the raw samples, set this to 1000 / sample_rate_hz",
    },
    "--skip-frame-times": {
        "action": "store_true",
        "default": False,
        "help": "If true, will store token tensors of shape (R,) instead of "
        "(R, 3), foregoing segment start and end times.",
    },
    "--feat-sizing": {
        "action": "store_true",
        "default": False,
        "help": "If true, will store token tensors of shape (R, 1) instead of "
        "(R, 3), foregoing segment start and end times (which trn does not "
        "have). The extra dimension will allow data in this directory to be "
        "loaded as features in a SpectDataSet.",
    },
    "--chunk-size": {
        "type": int,
        "default": 1000,
        "help": "The number of utterances that a worker will process at once. Impacts "
        "speed and memory consumption.",
    },
    "--timeout": {
        "type": int,
        "default": None,
        "help": "When using multiple workers, how long (in seconds) without new data "
        "before terminating. The default is to wait indefinitely.",
    },
}


def _add_common_arg(parser: argparse.ArgumentParser, flag: str):
    assert flag in _COMMON_ARGS
    kwargs = _COMMON_ARGS[flag]
    parser.add_argument(flag, **kwargs)


def _get_torch_spect_data_dir_info_parse_args(args):
    parser = argparse.ArgumentParser(
        description=get_torch_spect_data_dir_info.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dir", type=str, help="The torch data directory")
    parser.add_argument(
        "out_file",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="The file to write to. If unspecified, stdout",
    )
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
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

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="If set, validate the data directory before collecting info. The "
        "process is described in pydrobert.torch.data.validate_spect_data_set",
    )
    group.add_argument(
        "--fix",
        action="store_true",
        default=False,
        help="If set, validate the data directory before collecting info, potentially "
        "fixing small errors in the directory. The process is described in "
        "pydrobert.torch.validate_spect_data_set",
    )

    return parser.parse_args(args)


def get_torch_spect_data_dir_info(args: Optional[Sequence[str]] = None):
    """Write info about the specified SpectDataSet data dir

A torch SpectDataSet data dir is of the form

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

Where "feat/" contains float tensors of shape (T, F), where T is the number of frames
(variable) and F is the number of filters (fixed). "ali/" if there, contains long
tensors of shape (T,) indicating the appropriate per-frame class labels (likely pdf-ids
for discriminative training in an DNN-HMM). "ref/", if there, contains long tensors of
shape (R, 3) indicating a sequence of reference tokens where element indexed by "[i, 0]"
is a token id, "[i, 1]" is the inclusive start frame of the token (or a negative value
if unknown), and "[i, 2]" is the exclusive end frame of the token. Token sequences may
instead be of shape (R,) if no segment times are available in the corpus.

This command writes the following space-delimited key-value pairs to an output file in
sorted order:

1. "max_ali_class", the maximum inclusive class id found over "ali/"
    (if available, -1 if not)
2. "max_ref_class", the maximum inclussive class id found over "ref/"
    (if available, -1 if not)
3. "num_utterances", the total number of listed utterances
4. "num_filts", F
5. "total_frames", the sum of N over the data dir
6. "count_<i>", the number of instances of the class "<i>" that appear in "ali/"
   (if available). If "count_<i>" is a valid key, then so are "count_<0 to i>".
   "count_<i>" is left-padded with zeros to ensure that the keys remain in the same
   order in the table as the class indices.  The maximum i will be equal to the value
   of "max_ali_class"

Note that the output can be parsed as a Kaldi (http://kaldi-asr.org/) text table of
integers.
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
        suppress_alis=False,
        tokens_only=False,
    )
    if options.strict or options.fix:
        data.validate_spect_data_set(data_set, options.fix)
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
    parser = argparse.ArgumentParser(
        description=trn_to_torch_token_data_dir.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("trn", type=argparse.FileType("r"), help="The input trn file")
    _add_common_arg(parser, "token2id")
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
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--swap")
    _add_common_arg(parser, "--unk-symbol")
    _add_common_arg(parser, "--num-workers")
    _add_common_arg(parser, "--chunk-size")
    _add_common_arg(parser, "--timeout")
    size_group = parser.add_mutually_exclusive_group()
    _add_common_arg(size_group, "--skip-frame-times")
    _add_common_arg(size_group, "--feat-sizing")
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


def _save_transcripts_to_dir_do_work(
    transcripts, token2id, dir_, frame_shift_ms, unk, skip_frame_times, feat_sizing,
):
    for basename, transcript in transcripts:
        tok = data.transcript_to_token(
            transcript, token2id, frame_shift_ms, unk, skip_frame_times or feat_sizing,
        )
        if feat_sizing:
            tok = tok.unsqueeze(-1)
        path = os.path.join(dir_, basename)
        torch.save(tok, path)


def _save_transcripts_to_dir_worker(
    queue, token2id, dir_, frame_shift_ms, unk, skip_frame_times, feat_sizing, timeout,
):
    transcripts = queue.get(True, timeout)
    while transcripts is not None:
        _save_transcripts_to_dir_do_work(
            transcripts,
            token2id,
            dir_,
            frame_shift_ms,
            unk,
            skip_frame_times,
            feat_sizing,
        )
        del transcripts
        transcripts = queue.get(True, timeout)


def _save_transcripts_to_dir(
    transcripts,
    token2id,
    dir_,
    frame_shift_ms=None,
    unk=None,
    skip_frame_times=False,
    feat_sizing=False,
    num_workers=0,
    chunk_size=1000,
    timeout=None,
):
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    if num_workers:
        queue = torch.multiprocessing.Queue(num_workers)
        with torch.multiprocessing.Pool(
            num_workers,
            _save_transcripts_to_dir_worker,
            (
                queue,
                token2id,
                dir_,
                frame_shift_ms,
                unk,
                skip_frame_times,
                feat_sizing,
                timeout,
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

    else:
        _save_transcripts_to_dir_do_work(
            transcripts,
            token2id,
            dir_,
            frame_shift_ms,
            unk,
            skip_frame_times,
            feat_sizing,
        )


def trn_to_torch_token_data_dir(args: Optional[Sequence[str]] = None):
    """Convert a NIST "trn" file to the specified SpectDataSet data dir

A "trn" file is the standard transcription file without alignment information used in
the sclite (http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm) toolkit. It
has the format

    here is a transcription (utterance_a)
    here is another (utterance_b)

This command reads in a "trn" file and writes its contents as token sequences compatible
with the "ref/" directory of a SpectDataSet. See the command
"get-torch-spect-data-dir-info" for more info about a SpectDataSet directory."""
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
            yield options.file_prefix + utt_id + options.file_suffix, transcript

    _save_transcripts_to_dir(
        error_handling_iter(),
        token2id,
        options.dir,
        unk=options.unk_symbol,
        skip_frame_times=options.skip_frame_times,
        feat_sizing=options.feat_sizing,
        num_workers=options.num_workers,
        chunk_size=options.chunk_size,
        timeout=options.timeout,
    )
    return 0


def _torch_token_data_dir_to_trn_parse_args(args):
    parser = argparse.ArgumentParser(
        description=torch_token_data_dir_to_trn.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dir", help="The directory to read token sequences from")
    _add_common_arg(parser, "id2token")
    parser.add_argument(
        "trn",
        type=argparse.FileType("w"),
        help='The "trn" file to write transcriptions to',
    )
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--swap")
    _add_common_arg(parser, "--num-workers")
    return parser.parse_args(args)


class _DirectoryDataset(torch.utils.data.Dataset):
    def __init__(self, dir_, file_prefix, file_suffix):
        super().__init__()
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

    def __getitem__(self, index):
        utt_id = self.utt_ids[index]
        value = torch.load(
            os.path.join(self.dir_, self.file_prefix + utt_id + self.file_suffix)
        )
        return utt_id, value

    def __len__(self):
        return len(self.utt_ids)


class _TranscriptDataSet(_DirectoryDataset):
    def __init__(
        self, dir_, id2token, file_prefix, file_suffix, frame_shift_ms, strip_timing
    ):
        super().__init__(dir_, file_prefix, file_suffix)
        self.id2token = id2token
        self.frame_shift_ms = frame_shift_ms
        self.strip_timing = strip_timing

    def __getitem__(self, index):
        utt_id, tok = super().__getitem__(index)
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


def torch_token_data_dir_to_trn(args: Optional[Sequence[str]] = None):
    """Convert a SpectDataSet token data dir to a NIST trn file

A "trn" file is the standard transcription file without alignment information used
in the sclite (http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm)
toolkit. It has the format

    here is a transcription (utterance_a)
    here is another (utterance_b)

This command scans the contents of a directory like "ref/" in a SpectDataSeet and
converts each such file into a transcription. Each such transcription is then
written to a "trn" file. See the command "get-torch-spect-data-dir-info" for more
info about a SpectDataSet directory."""
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
    parser = argparse.ArgumentParser(
        description=ctm_to_torch_token_data_dir.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ctm",
        type=argparse.FileType("r"),
        help='The "ctm" file to read token segments from',
    )
    _add_common_arg(parser, "token2id")
    parser.add_argument(
        "dir",
        help="The directory to store token sequences to. If the "
        "directory does not exist, it will be created",
    )
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--swap")
    _add_common_arg(parser, "--unk-symbol")
    _add_common_arg(parser, "--num-workers")
    _add_common_arg(parser, "--chunk-size")
    _add_common_arg(parser, "--timeout")
    size_group = parser.add_mutually_exclusive_group()
    _add_common_arg(size_group, "--skip-frame-times")
    _add_common_arg(size_group, "--feat-sizing")
    _add_common_arg(size_group, "--frame-shift-ms")
    utt_group = parser.add_mutually_exclusive_group()
    utt_group.add_argument(
        "--wc2utt",
        type=argparse.FileType("r"),
        default=None,
        help="A file mapping wavefile name and channel combinations (e.g. "
        "'utt_1 A') to utterance IDs. Each line of the file has the format "
        "'<wavefile_name> <channel> <utt_id>'. If neither '--wc2utt' nor "
        "'--utt2wc' has been specied, the wavefile name will be treated as "
        "the utterance ID",
    )
    utt_group.add_argument(
        "--utt2wc",
        type=argparse.FileType("r"),
        default=None,
        help="A file mapping utterance IDs to wavefile name and channel "
        "combinations (e.g. 'utt_1 A'). Each line of the file has the "
        "format '<utt_id> <wavefile_name> <channel>'. If neither '--wc2utt' "
        "nor '--utt2wc' has been specied, the wavefile name will be treated "
        "as the utterance ID",
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


def ctm_to_torch_token_data_dir(args: Optional[Sequence[str]] = None):
    """Convert a NIST "ctm" file to a SpectDataSet token data dir

A "ctm" file is a transcription file with token alignments (a.k.a. a time-marked
conversation file) used in the sclite
(http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm>) toolkit. Here is the
format

    utt_1 A 0.2 0.1 hi
    utt_1 A 0.3 1.0 there  ;; comment
    utt_2 A 0.0 1.0 next
    utt_3 A 0.1 0.4 utterance

Where the first number specifies the token start time (in seconds) and the second the
duration.

This command reads in a "ctm" file and writes its contents as token sequences compatible
with the "ref/" directory of a SpectDataSet. See the command
"get-torch-spect-data-dir-info" for more info about a SpectDataSet directory."""
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
        ((options.file_prefix + x[0] + options.file_suffix, x[1]) for x in transcripts),
        token2id,
        options.dir,
        options.frame_shift_ms,
        options.unk_symbol,
        options.skip_frame_times,
        options.feat_sizing,
        options.num_workers,
        options.chunk_size,
        options.timeout,
    )
    return 0


def _textgrids_to_torch_token_data_dir_parse_args(args):
    parser = argparse.ArgumentParser(
        description=textgrids_to_torch_token_data_dir.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("tg_dir", help="The directory containing the TextGrid files")
    _add_common_arg(parser, "token2id")
    parser.add_argument(
        "dir",
        help="The directory to store token sequences to. If the "
        "directory does not exist, it will be created",
    )
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--swap")
    _add_common_arg(parser, "--unk-symbol")
    _add_common_arg(parser, "--num-workers")
    _add_common_arg(parser, "--chunk-size")
    _add_common_arg(parser, "--timeout")
    size_group = parser.add_mutually_exclusive_group()
    _add_common_arg(size_group, "--skip-frame-times")
    _add_common_arg(size_group, "--feat-sizing")
    _add_common_arg(size_group, "--frame-shift-ms")
    parser.add_argument(
        "--textgrid-suffix",
        default=".TextGrid",
        help="The file suffix in tg_dir indicating a TextGrid file",
    )
    parser.add_argument(
        "--fill-symbol",
        default=None,
        help="If set, unlabelled intervals in the TextGrid files will be "
        "assigned this symbol. Relevant only if a point grid.",
    )
    tier_grp = parser.add_mutually_exclusive_group()
    tier_grp.add_argument(
        "--tier-name", dest="tier_id", help="The name of the tier to extract."
    )
    tier_grp.add_argument(
        "--tier-idx",
        dest="tier_id",
        type=int,
        help="The index of the tier to extract.",
    )
    return parser.parse_args(args)


def textgrids_to_torch_token_data_dir(args: Optional[Sequence[str]] = None):
    """Convert a directory of TextGrid files into a SpectDataSet token data dir

A "TextGrid" file is a transcription file for a single utterance used by the Praat
software (https://www.fon.hum.uva.nl/praat/).

This command accepts a directory "tg_dir" of TextGrid files

    tg_dir/
        <file-prefix>utt_1.TextGrid
        <file-prefix>utt_2.TextGrid
        ...

and writes each file as a separate token sequence compatible with the "ref/" directory
of a SpectDataSet. If the extracted tier is an IntervalTier, the start and end points
will be saved with each token. Otherwise, only the token sequence will be saved. See the
command "get-torch-spect-data-dir-info" for more info about a SpectDataSet directory."""
    try:
        options = _textgrids_to_torch_token_data_dir_parse_args(args)
    except SystemExit as ex:
        return ex.code
    if not os.path.isdir(options.tg_dir):
        raise ValueError(f"'{options.tg_dir}' is not a directory")
    token2id = _parse_token2id(options.token2id, options.swap, options.swap)
    if options.unk_symbol is not None and options.unk_symbol not in token2id:
        print(f"Unk symbol '{options.unk_symbol}' is not in token2id", file=sys.stderr)
        return 1
    if options.fill_symbol is not None and options.fill_symbol not in token2id:
        print(
            f"Fill symbol '{options.fill_symbol}' is not in token2id", file=sys.stderr
        )
        return 1
    if options.tier_id is None:
        options.tier_id = 0

    def textgrid_iter():
        for file_name in os.listdir(options.tg_dir):
            if not file_name.endswith(
                options.textgrid_suffix
            ) or not file_name.startswith(options.file_prefix):
                continue
            utt_id = (
                file_name[: len(file_name) - len(options.textgrid_suffix)]
                + options.file_suffix
            )
            yield utt_id, data.read_textgrid(
                os.path.join(options.tg_dir, file_name),
                options.tier_id,
                options.fill_symbol,
            )[0]

    _save_transcripts_to_dir(
        textgrid_iter(),
        token2id,
        options.dir,
        options.frame_shift_ms,
        options.unk_symbol,
        options.skip_frame_times,
        options.feat_sizing,
        options.num_workers,
        options.chunk_size,
        options.timeout,
    )
    return 0


def _torch_token_data_dir_to_ctm_parse_args(args):
    parser = argparse.ArgumentParser(
        description=torch_token_data_dir_to_ctm.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dir", help="The directory to read token sequences from")
    _add_common_arg(parser, "id2token")
    parser.add_argument(
        "ctm",
        type=argparse.FileType("w"),
        help='The "ctm" file to write token segments to',
    )
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--swap")
    _add_common_arg(parser, "--frame-shift-ms")
    utt_group = parser.add_mutually_exclusive_group()
    utt_group.add_argument(
        "--wc2utt",
        type=argparse.FileType("r"),
        default=None,
        help="A file mapping wavefile name and channel combinations (e.g. "
        "'utt_1 A') to utterance IDs. Each line of the file has the format "
        "'<wavefile_name> <channel> <utt_id>'.",
    )
    utt_group.add_argument(
        "--utt2wc",
        type=argparse.FileType("r"),
        default=None,
        help="A file mapping utterance IDs to wavefile name and channel "
        "combinations (e.g. 'utt_1 A'). Each line of the file has the "
        "format '<utt_id> <wavefile_name> <channel>'.",
    )
    utt_group.add_argument(
        "--channel",
        default="A",
        help='If neither "--wc2utt" nor "--utt2wc" is specified, utterance '
        "IDs are treated as wavefile names and are given the value of this "
        "flag as a channel",
    )
    return parser.parse_args(args)


def torch_token_data_dir_to_ctm(args: Optional[Sequence[str]] = None):
    """Convert a SpectDataSet token data directory to a NIST "ctm" file

A "ctm" file is a transcription file with token alignments (a.k.a. a time-marked
conversation file) used in the sclite
(http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm) toolkit. Here is the
format::

    utt_1 A 0.2 0.1 hi
    utt_1 A 0.3 1.0 there  ;; comment
    utt_2 A 0.0 1.0 next
    utt_3 A 0.1 0.4 utterance

Where the first number specifies the token start time (in seconds) and the second the
duration.

This command scans the contents of a directory like "ref/" in a SpectDataSet and
converts each such file into a transcription. Every token in a given transcription must
have information about its duration. Each such transcription is then written to the
"ctm" file. See the command "get-torch-spect-data-dir-info" for more info about a
SpectDataSet directory."""
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
        description=compute_torch_token_data_dir_error_rates.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dir",
        help="If the 'hyp' argument is not specified, this is the "
        "parent directory of two subdirectories, 'ref/' and 'hyp/', which "
        "contain the reference and hypothesis transcripts, respectively. If "
        "the '--hyp' argument is specified, this is the reference "
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
        help=_COMMON_ARGS["id2token"]["help"],
    )
    parser.add_argument(
        "--replace",
        type=argparse.FileType("r"),
        default=None,
        help="A file containing pairs of elements per line. The first is the "
        "element to replace, the second what to replace it with. If "
        "'--id2token' is specified, the file should contain tokens. If "
        "'--id2token' is not specified, the file should contain IDs "
        "(integers). This is processed before '--ignore'",
    )
    parser.add_argument(
        "--ignore",
        type=argparse.FileType("r"),
        default=None,
        help="A file containing a whitespace-delimited list of elements to "
        "ignore in both the reference and hypothesis transcripts. If "
        "'--id2token' is specified, the file should contain tokens. If "
        "'--id2token' is not specified, the file should contain IDs "
        "(integers). This is processed after '--replace'",
    )
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--swap")
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
        help="If set, return the average distance per utterance instead of the total "
        "errors over the number of reference tokens",
    )
    parser.add_argument(
        "--per-utt",
        action="store_true",
        default=False,
        help="If set, return lines of ``<utt_id> <error_rate>`` denoting the "
        "per-utterance error rates instead of the average",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="The number of error rates to compute at once. Reduce if you "
        "run into memory errors",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress warnings which arise from edit distance computations",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--costs",
        nargs=3,
        type=float,
        metavar=("INS", "DEL", "SUB"),
        default=(1.0, 1.0, 1.0),
        help="The costs of an insertion, deletion, and substitution, respectively",
    )
    group.add_argument(
        "--nist-costs",
        action="store_true",
        default=False,
        help="Use NIST (sclite, score) default costs for insertions, deletions, and "
        "substitutions (3/3/4)",
    )
    return parser.parse_args(args)


def compute_torch_token_data_dir_error_rates(args: Optional[Sequence[str]] = None):
    """Compute error rates between reference and hypothesis token data dirs

WARNING!!!!
The error rates reported by this command have changed since version v0.3.0 of
pydrobert-pytorch when the insertion, deletion, and substitution costs do not all equal
1. Consult the documentation of "pydrobert.torch.functional.error_rate" for more
information.

This is a very simple script that computes and prints the error rates between the "ref/"
(reference/gold standard) token sequences and "hyp/" (hypothesis/generated) token
sequences in a SpectDataSet directory. Consult the Wikipedia article on the Levenshtein
distance (https://en.wikipedia.org/wiki/Levenshtein_distance>) for more info on error
rates. The error rate for the entire partition will be calculated as the total number of
insertions, deletions, and substitutions made in all transcriptions divided by the sum
of lengths of reference transcriptions.

Error rates are printed as ratios, not by "percentage."

While convenient and accurate, this script has very few features. Consider pairing the
command "torch-token-data-dir-to-trn" with sclite
(http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm) instead.

Many tasks will ignore some tokens (e.g. silences) or collapse others (e.g. phones).
Please consult a standard recipe (such as those in Kaldi http://kaldi-asr.org/) before
performing these computations."""
    try:
        options = _compute_torch_token_data_dir_parse_args(args)
    except SystemExit as ex:
        return ex.code
    if options.nist_costs:
        options.costs = (3.0, 3.0, 4.0)
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
    tot_errs = 0
    total_ref_tokens = 0.0
    while len(ref_transcripts):
        batch_ref_transcripts = [
            (
                utt,
                [
                    token2id[replace.get(t, t)]
                    for t in transcript
                    if replace.get(t, t) not in ignore
                ],
            )
            for (utt, transcript) in ref_transcripts[: options.batch_size]
        ]
        batch_hyp_transcripts = [
            (
                utt,
                [
                    token2id[replace.get(t, t)]
                    for t in transcript
                    if replace.get(t, t) not in ignore
                ],
            )
            for (utt, transcript) in hyp_transcripts[: options.batch_size]
        ]
        ref_transcripts = ref_transcripts[options.batch_size :]
        hyp_transcripts = hyp_transcripts[options.batch_size :]
        ref = torch.nn.utils.rnn.pad_sequence(
            [
                torch.tensor(transcript + [eos])
                for _, transcript in batch_ref_transcripts
            ],
            padding_value=padding,
        )
        hyp = torch.nn.utils.rnn.pad_sequence(
            [
                torch.tensor(transcript + [eos])
                for _, transcript in batch_hyp_transcripts
            ],
            padding_value=padding,
        )
        ers = error_rate(
            ref,
            hyp,
            eos=eos,
            include_eos=False,
            ins_cost=options.costs[0],
            del_cost=options.costs[1],
            sub_cost=options.costs[2],
            norm=False,
            warn=not options.quiet,
        )
        for (utt_id, transcript), er in zip(batch_ref_transcripts, ers):
            error_rates[utt_id] = er.item() / (
                1 if options.distances else len(transcript)
            )
            tot_errs += er.item()
            total_ref_tokens += len(transcript)
    if options.per_utt:
        for utt_id, er in list(error_rates.items()):
            options.out.write("{} {}\n".format(utt_id, er))
    else:
        options.out.write(
            "{}\n".format(
                tot_errs / (len(error_rates) if options.distances else total_ref_tokens)
            )
        )


def _torch_spect_data_dir_to_wds_parse_args(args):
    parser = argparse.ArgumentParser(
        description=torch_spect_data_dir_to_wds.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dir", help="The torch data directory")
    parser.add_argument("tar_path", help="The path to store files to")
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    parser.add_argument(
        "--feat-subdir", default="feat", help="Subdirectory where features are stored."
    )
    parser.add_argument(
        "--ali-subdir", default="ali", help="Subdirectory where alignments are stored."
    )
    parser.add_argument(
        "--ref-subdir",
        default="ref",
        help="Subdirectory where reference token sequences are stored.",
    )
    parser.add_argument(
        "--is-uri",
        action="store_true",
        default=False,
        help="If set, tar_pattern will be treated as a URI rather than a path/",
    )
    parser.add_argument(
        "--shard",
        action="store_true",
        default=False,
        help="Split samples among multiple tar files. 'tar_path' will be extended with "
        "a suffix '.x', where x is the shard number.",
    )
    parser.add_argument(
        "--max-samples-per-shard",
        type=int,
        default=1e5,
        help="If sharding ('--shard' is specified), dictates the number of samples in "
        "each file.",
    )
    parser.add_argument(
        "--max-size-per-shard",
        type=int,
        default=3e9,
        help="If sharding ('--shard' is specified), dictates the maximum size in bytes "
        "of each file.",
    )

    return parser.parse_args(args)


def torch_spect_data_dir_to_wds(args: Optional[Sequence[str]] = None):
    """Convert a SpectDataSet to a WebDataset
    
A torch SpectDataSet data dir is of the form

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

Where "feat/" contains float tensors of shape (N, F), where N is the number of
frames (variable) and F is the number of filters (fixed). "ali/" if there, contains
long tensors of shape (N,) indicating the appropriate class labels (likely pdf-ids
for discriminative training in an DNN-HMM). "ref/", if there, contains long tensors
of shape (R, 3) indicating a sequence of reference tokens where element indexed by
"[i, 0]" is a token id, "[i, 1]" is the inclusive start frame of the token (or a
negative value if unknown), and "[i, 2]" is the exclusive end frame of the token.

This command converts the data directory into a tar file to be used as a
WebDataset (https://github.com/webdataset/webdataset), whose contents are files

    <utt1>.feat.pth
    [<utt1>.ali.pth]
    [<utt1>.ref.pth]
    <utt2>.feat.pth
    [<utt2>.ali.pth]
    [<utt2>.ref.pth]
    ...

holding tensors with the same interpretation as above.

This command does not require WebDataset to be installed."""
    try:
        options = _torch_spect_data_dir_to_wds_parse_args(args)
    except SystemExit as ex:
        return ex.code
    if not os.path.isdir(options.dir):
        print(f"'{options.dir}' is not a directory", file=sys.stderr)
        return 1
    data_set = data.SpectDataSet(
        options.dir,
        file_prefix=options.file_prefix,
        file_suffix=options.file_suffix,
        feat_subdir=options.feat_subdir,
        ali_subdir=options.ali_subdir,
        ref_subdir=options.ref_subdir,
        suppress_alis=False,
        tokens_only=False,
    )
    pattern = Path(options.tar_path)
    os.makedirs(pattern.parent, exist_ok=True)
    if pattern.suffix in {".tgz", ".gz"}:
        compression = "gz"
    elif pattern.suffix == ".bz2":
        compression = "bz2"
    elif pattern.suffix == ".xz":
        compression = "xz"
    else:
        compression = ""
    pattern = str(pattern.resolve())
    NN = len(data_set)
    if options.shard:
        max_bytes = options.max_size_per_shard
        max_count = options.max_samples_per_shard
        if max_bytes <= 0:
            raise argparse.ArgumentTypeError("--max-size-per-shard must be positive")
        if max_count <= 0:
            raise argparse.ArgumentTypeError("--max-samples-per-shard must be positive")
        max_num_shards = (NN - 1) // max_count + 1
        max_shard = max(max_num_shards - 1, 1)
        pattern += f".{{shard:0{int(math.ceil(math.log(max_shard)))}d}}"
    else:
        max_bytes = float("inf")
        max_count = NN
    cur_count = cur_bytes = shard = 0
    cur_tar = tarfile.open(pattern.format(shard=shard), f"w|{compression}")
    for idx in range(NN):
        feat, ali, ref = data_set[idx]
        utt_id = data_set.utt_ids[idx]
        if cur_count >= max_count or cur_bytes >= max_bytes:
            cur_tar.close()
            shard += 1
            cur_count = cur_bytes = 0
            cur_tar = tarfile.open(pattern.format(shard=shard), f"w|{compression}")
        for name, tensor in (("ali", ali), ("feat", feat), ("ref", ref)):
            if tensor is None:
                continue
            buf = io.BytesIO()
            torch.save(tensor, buf)
            buf.read()
            member = tarfile.TarInfo(f"{utt_id}.{name}.pth")
            member.size = buf.tell()
            cur_bytes += member.size
            member.mode = 0o0444
            buf.seek(0)
            cur_tar.addfile(member, buf)
            del buf
        cur_count += 1
    cur_tar.close()
    return


def _compute_mvn_stats_for_torch_feat_data_dir_parse_args(args):
    parser = argparse.ArgumentParser(
        description=compute_mvn_stats_for_torch_feat_data_dir.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
If --id2gid is specified, it points to a file which maps file ids to groups. Each group
gets its own statistics which are estimated using only the feature vectors from the
files assigned to them. With <id_1>, <id_2>, etc. part of the file names in the feature
directory as above and <gid_1>, <gid_2>, etc. strings without spaces representing group
ids, then the argument passed to --id2gid is a file with lines

    <id_x> <gid_y>

defining a surjective mapping from file ids to group ids. 'out' will then store a
pickled, nested dictionary

    {
        <gid_1>: {'mean': ..., 'var': ...},
        <gid_2>: {'mean': ..., 'var': ...},
        ...
    }

of the statistics of all groups.
""",
    )
    parser.add_argument("dir", help="The feature directory")
    parser.add_argument("out", help="Output path")
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--num-workers")
    parser.add_argument(
        "--dim", type=int, default=-1, help="The dimension of the feature vector"
    )
    parser.add_argument(
        "--id2gid",
        type=argparse.FileType("r"),
        default=None,
        help="Path to a file mapping feature tensors to groups. See below for more info",
    )
    parser.add_argument(
        "--bessel",
        action="store_true",
        default=False,
        help="Apply Bessel's correction "
        "(https://en.wikipedia.org/wiki/Bessel's_correction) to estimates.",
    )
    return parser.parse_args(args)


def compute_mvn_stats_for_torch_feat_data_dir(args: Optional[Sequence[str]] = None):
    """Compute mean and standard deviation over a torch feature directory

A feature directory is of the form

dir/
    <file_prefix><id_1><file_suffix>
    <file_prefix><id_2><file_suffix>
    ...

where each file contains a dynamically-sized tensor whose last dimension (by default) is
a feature vector. Letting F be a feature vector, this command computes the mean and
standard deviation of the features in the directory, storing them as a pickled
dictionary of tensors (with keys 'mean' and 'std') to the file 'out'. Those statistics
may be used with a pydrobert.torch.modules.MeanVarianceNormalization layer."""
    try:
        options = _compute_mvn_stats_for_torch_feat_data_dir_parse_args(args)
    except SystemExit as ex:
        return ex.code

    if options.id2gid is not None:
        id2gid = dict()
        for ln, line in enumerate(options.id2gid):
            line = line.strip().split()
            if not len(line):
                continue
            if len(line) != 2:
                print(
                    f"{options.id2gid.name} line {ln + 1}: expected two ids, got "
                    f"{len(line)}",
                    file=sys.stderr,
                )
                return 1
            id_, gid = line
            if id_ in id2gid:
                print(
                    f"{options.id2gid.name} line {ln + 1}: duplicate entry for id '{id_}'",
                    file=sys.stderr,
                )
                return 1
            id2gid[id_] = gid
        gid2mvn = dict((x, None) for x in id2gid.values())
    else:
        id2gid = defaultdict(lambda: None)
        gid2mvn = {None: None}

    ds = _DirectoryDataset(options.dir, options.file_prefix, options.file_suffix)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=options.num_workers, collate_fn=_noop_collate
    )

    for id_, x in dl:
        try:
            gid = id2gid[id_]
        except KeyError:
            print(
                f"'{options.dir}' id '{id_}' was not listed in '{options.id2gid.name}'",
                file=sys.stderr,
            )
            return 1
        mvn = gid2mvn[gid]
        if mvn is None:
            gid2mvn[gid] = mvn = modules.MeanVarianceNormalization(options.dim)
        mvn.accumulate(x)

    gid2stats = dict()
    for gid, mvn in gid2mvn.items():
        if mvn is None:
            if gid is None:
                print("No features - no stats!", file=sys.stderr)
                return 1
            print(f"Gid '{gid}' had no accumulated stats - not saving", file=sys.stderr)
            continue
        mvn.store(bessel=options.bessel)
        gid2stats[gid] = {"mean": mvn.mean, "std": mvn.std}

    if set(gid2stats) == {None}:
        gid2stats = gid2stats[None]

    torch.save(gid2stats, options.out)


def _torch_token_data_dir_to_torch_ali_data_dir_parse_args(args):
    parser = argparse.ArgumentParser(
        description=torch_token_data_dir_to_torch_ali_data_dir.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ref_dir", help="The token sequence data directory (input)",
    )
    parser.add_argument(
        "ali_dir", help="The frame alignment data directory (output)",
    )
    parser.add_argument(
        "--feat-dir",
        default=None,
        help="The feature data directory. While not necessary for the conversion, "
        "specifying this directory will allow the total number of frames in each "
        "utterance to be checked by loading the associated feature matrix.",
    )
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--num-workers")
    _add_common_arg(parser, "--chunk-size")
    return parser.parse_args(args)


def _torch_token_data_dir_to_torch_ali_dir_do_work(
    basenames: Sequence[str],
    ref_dir: str,
    ali_dir: str,
    feat_dir: Optional[str] = None,
):
    for basename in basenames:
        ref_path = os.path.join(ref_dir, basename)
        ref = torch.load(ref_path)
        err_msg = f"Error converting '{ref_path}' to ali:"
        if ref.ndim != 2 or ref.size(0) == 0 or ref.size(1) != 3:
            raise ValueError(f"{err_msg} invalid size '{ref.shape}'")
        if (ref[:, 1:] < 0).any():
            raise ValueError(f"{err_msg} some token boundaries missing")
        if ref[0, 1] != 0:
            raise ValueError(f"{err_msg} starts at frame {ref[0, 1].item()}, not 0")
        if (ref[:-1, 2] != ref[1:, 1]).any():
            raise ValueError(f"{err_msg} not all boundaries are contiguous")
        if feat_dir is not None:
            feat_path = os.path.join(feat_dir, basename)
            T = torch.load(feat_path).size(0)
            if ref[-1, 2] != T:
                raise ValueError(
                    f"{err_msg} feats at '{feat_path}' report {T} frames. ref "
                    f"ends with {ref[-1, 2].item()}"
                )
        print(basename)
        ali = torch.repeat_interleave(ref[:, 0], ref[:, 2] - ref[:, 1])
        torch.save(ali, os.path.join(ali_dir, basename))


def _torch_token_data_dir_to_torch_ali_dir_worker(
    queue: torch.multiprocessing.Queue,
    ref_dir: str,
    ali_dir: str,
    feat_dir: Optional[str] = None,
    first_timeout=None,
    rest_timeout=None,
):
    basenames = queue.get(True, first_timeout)
    while basenames is not None:
        _torch_token_data_dir_to_torch_ali_dir_do_work(
            basenames, ref_dir, ali_dir, feat_dir
        )
        basenames = queue.get(True, rest_timeout)


def torch_token_data_dir_to_torch_ali_data_dir(args: Optional[Sequence[str]] = None):
    """Convert a ref/ dir to an ali/ dir

See the command "get-torch-spect-data-dir-info" for more info on token sequences and
alignments.

A reference token sequence "tok" partitions a frame sequence of length T if

1. tok is of shape (R, 3), with R > 1 and all tok[r, 1:] >= 0 (it contains segment
   boundaries).
2. tok[0, 1] = 0 (it starts at frame 0).
3. for all 0 <= r < R - 1, tok[r, 2] = tok[r + 1, 1] (boundaries contiguous).
4. tok[R - 1, 2] = T (it ends after T frames).

When tok partitions the frame sequence, it can be converted into a per-frame alignment
tensor "ali" of shape (T,), where tok[r, 1] <= t < tok[r, 2] implies ali[t] = tok[r, 0].

This command applies the above reasoning to convert all tensors in a token sequence
directory into a frame alignment directory.

WARNING! This operation is potentially destructive: a per-frame alignment cannot
distinguish between two of the same token next to one another and one larger token.
"""
    try:
        options = _torch_token_data_dir_to_torch_ali_data_dir_parse_args(args)
    except SystemExit as ex:
        return ex.code
    if not os.path.isdir(options.ref_dir):
        print(f"'{options.ref_dir}' is not a directory", file=sys.stderr)
        return 1
    basenames = (
        x
        for x in os.listdir(options.ref_dir)
        if x.startswith(options.file_prefix) and x.endswith(options.file_prefix)
    )
    os.makedirs(options.ali_dir, exist_ok=True)
    if options.num_workers:
        queue = torch.multiprocessing.Queue(options.num_workers)
        with torch.multiprocessing.Pool(
            options.num_workers,
            _torch_token_data_dir_to_torch_ali_dir_worker,
            (queue, options.ref_dir, options.ali_dir, options.feat_dir,),
        ) as pool:
            chunk = tuple(itertools.islice(basenames, options.chunk_size))
            while len(chunk):
                queue.put(chunk)
                chunk = tuple(itertools.islice(basenames, options.chunk_size))
            for _ in range(options.num_workers):
                queue.put(None)
            pool.close()
            pool.join()
    else:
        _torch_token_data_dir_to_torch_ali_dir_do_work(
            basenames, options.ref_dir, options.ali_dir, options.feat_dir,
        )
    return 0
