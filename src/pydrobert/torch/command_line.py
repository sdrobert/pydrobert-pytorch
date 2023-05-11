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

import collections
import os
import sys
import argparse
import math
import warnings
import itertools
import tarfile
import io
import random
import shutil

from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Any, Callable, Dict, Optional, Sequence, Type, TypeVar
from typing_extensions import Literal

import torch

from . import data, modules, config
from ._feats import SliceSpectData, ChunkTokenSequencesBySlices
from ._pad import ChunkBySlices
from ._string import error_rate


def rdir(x: str) -> str:
    if not os.path.isdir(x):
        raise TypeError(f"'{x}' is not a directory")
    return x


R = TypeVar("R")


def bounded(
    x: str,
    type: Type[R],
    lower: Optional[R] = None,
    upper: Optional[R] = None,
    inclusive_lower: bool = True,
    inclusive_upper: bool = True,
) -> R:
    x: R = type(x)
    if lower is not None:
        if inclusive_lower and x < lower:
            raise TypeError(f"Expected {x} >= {lower}")
        elif not inclusive_lower and x <= lower:
            raise TypeError(f"Expected {x} > {lower}")
    if upper is not None:
        if inclusive_upper and x > upper:
            raise TypeError(f"Expected {x} <= {upper}")
        elif not inclusive_upper and x >= upper:
            raise TypeError(f"Expected {x} < {upper}")
    return x


nat = lambda x: bounded(x, int, 1)
nat0 = lambda x: bounded(x, int, 0)
pos = lambda x: bounded(x, float, 0, inclusive_lower=False)
open01 = lambda x: bounded(x, float, 0.0, 1.0, False, False)
closed01 = lambda x: bounded(x, float, 0.0, 1.0)


_COMMON_ARGS = {
    "--file-prefix": {
        "default": config.DEFT_FILE_PREFIX,
        "help": "The file prefix indicating a torch data file",
    },
    "--file-suffix": {
        "default": config.DEFT_FILE_SUFFIX,
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
        "type": pos,
        "default": config.DEFT_FRAME_SHIFT_MS,
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
    "--mp-chunk-size": {
        "type": nat,
        "default": config.DEFT_CHUNK_SIZE,
        "help": "The number of utterances that a multiprocessing worker will process "
        "at once. Impacts speed and memory consumption.",
    },
    "--textgrid-suffix": {
        "default": config.DEFT_TEXTGRID_SUFFIX,
        "help": "The file suffix in tg_dir indicating a TextGrid file.",
    },
    "--feat-subdir": {
        "default": config.DEFT_FEAT_SUBDIR,
        "help": "Subdirectory where features are stored.",
    },
    "--ali-subdir": {
        "default": config.DEFT_ALI_SUBDIR,
        "help": "Subdirectory where per-frame alignments are stored.",
    },
    "--ref-subdir": {
        "default": config.DEFT_REF_SUBDIR,
        "help": "Subdirectory where reference token sequences are stored.",
    },
}


def _add_common_arg(parser: argparse.ArgumentParser, flag: str):
    assert flag in _COMMON_ARGS
    kwargs = _COMMON_ARGS[flag]
    parser.add_argument(flag, **kwargs)


def get_torch_spect_data_dir_info(args: Optional[Sequence[str]] = None):
    """Write info about the specified SpectDataSet data dir

NOTE: additional keys (6, 8-10) have been added since pydrobert-pytorch v0.3.0. In
addition, validation now allows for empty reference segments.

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

1.  "max_ali_class", the maximum inclusive class id found over "ali/"
     (if available, -1 if not).
2.  "max_ref_class", the maximum inclussive class id found over "ref/"
     (if available, -1 if not).
3.  "num_utterances", the total number of listed utterances.
4.  "num_filts", F.
5.  "total_frames", the sum of T over the data dir.
6.  "total_tokens", the sum of R over the data dir (if available, -1 if not).
7.  "count_<i>", the number of instances of the class "<i>" that appear in "ali/"
    (if available).
8.  "segs_<i>". The number of segments of the class "<i>" that appear in "ali/"
    (if available). A segment of "<i>" is a maximal run of instances of "<i>" which
    appear sequentially in an alignment. For example, the alignment "0 1 0 1 1 1" would
    have "count_0 = 2" and "count_1 = 4", but "segs_0 = segs_1 = 2".
9.  "rcount_<i>", the total number of frames reference tokens with type index "<i>"
    occupy according to the segment boundaries listed in the sequences in "ref/" (if
    available). If any token sequence containing index "<i>" does not provide segment
    boundaries (or "<i>" never occurs), "rcount_<i>" is set to "-1".
10. "rsegs_<i>", the total number of segments (i.e. tokens) with type index "<i>"
    that appear in "ref/" (if available).

If "max_ali_class" was found (>= 0), all key/value pairs for "count_0-<max_ali_class>"
and "segs_0-<max_ali_class>" will be specified in the file, even if they aren't found
in the directory. Indices "<i>" will be left-padded with zeros so that keys are sorted
in increasing index. The same holds for "max_ref_class", "rcount_<i>", and "rsegs_<i>".

In an invalid data directory, the stored key/value pairs are not guaranteed to be
correct. Passing the "--strict" flag will validate the directory first. Passing "--fix"
instead will validate the directory and fix any small issues. See the function
"validate_spect_data_set" in the pydrobert.torch.data Python module for more
information on the validation process.

Note that the output can be parsed as a Kaldi (http://kaldi-asr.org/) text table of
integers."""
    parser = argparse.ArgumentParser(
        description=get_torch_spect_data_dir_info.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dir", type=rdir, help="The torch data directory")
    parser.add_argument(
        "out_file",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="The file to write to. If unspecified, stdout",
    )
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--feat-subdir")
    _add_common_arg(parser, "--ali-subdir")
    _add_common_arg(parser, "--ref-subdir")
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
        nargs="?",
        metavar="N",
        type=nat0,
        const=1,
        default=None,
        help="If set, validate the data directory before collecting info, potentially "
        "fixing small errors in the directory. An optional integer argument controls "
        "the cropping threshold for ali/ and ref/ (defaults to 1). The process is "
        "described in pydrobert.torch.validate_spect_data_set.",
    )
    try:
        options = parser.parse_args(args)
    except SystemExit as ex:
        return ex.code

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
    counts, segs, rcounts, rsegs = dict(), dict(), dict(), dict()
    for feat, ali, ref in data_set:
        info_dict["num_filts"] = feat.size()[1]
        info_dict["total_frames"] += feat.size()[0]
        if ali is not None:
            class_idxs, counts_ = ali.unique_consecutive(return_counts=True)
            for class_idx, count in zip(class_idxs.tolist(), counts_.tolist()):
                if class_idx < 0:
                    raise ValueError("Got a negative ali class idx")
                info_dict["max_ali_class"] = max(class_idx, info_dict["max_ali_class"])
                counts[class_idx] = counts.get(class_idx, 0) + count
                segs[class_idx] = segs.get(class_idx, 0) + 1
        if ref is not None:
            if ref.ndim == 1:
                ref = ref.unsqueeze(1)
                ref = torch.cat(
                    [ref, torch.full((ref.size(0), 2), -1, dtype=torch.long)], 1
                )
            for tok, start, end in ref.tolist():
                if tok < 0:
                    raise ValueError(f"Got a negative reference token index '{tok}'")
                info_dict["total_tokens"] = info_dict.get("total_tokens", 0) + 1
                info_dict["max_ref_class"] = max(info_dict["max_ref_class"], tok)
                rcount = rcounts.get(tok, 0)
                if rcount >= 0 and end > start >= 0:
                    rcounts[tok] = rcount + end - start
                else:
                    rcounts[tok] = -1
                rsegs[tok] = rsegs.get(tok, 0) + 1

    info_dict.setdefault("total_tokens", -1)

    max_ali_class = info_dict["max_ali_class"]
    if max_ali_class >= 0:
        digits = int(math.log10(max(max_ali_class, 1))) + 1
        count_fmt_str = f"count_{{:0{digits}d}}"
        seg_fmt_str = f"segs_{{:0{digits}d}}"
        for class_idx in range(max_ali_class + 1):
            info_dict[count_fmt_str.format(class_idx)] = counts.get(class_idx, 0)
            info_dict[seg_fmt_str.format(class_idx)] = segs.get(class_idx, 0)

    max_ref_class = info_dict["max_ref_class"]
    if max_ref_class >= 0:
        digits = int(math.log10(max(max_ref_class, 1))) + 1
        count_fmt_str = f"rcount_{{:0{digits}d}}"
        seg_fmt_str = f"rsegs_{{:0{digits}d}}"
        for class_idx in range(max_ref_class + 1):
            info_dict[count_fmt_str.format(class_idx)] = rcounts.get(class_idx, -1)
            info_dict[seg_fmt_str.format(class_idx)] = rsegs.get(class_idx, 0)

    info_list = sorted(info_dict.items())
    for key, value in info_list:
        options.out_file.write(f"{key} {value}\n")
    if options.out_file != sys.stdout:
        options.out_file.close()
    return 0


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
    bt, token2id, dir_, frame_shift_ms, unk, skip_frame_times, feat_sizing,
):
    basename, transcript = bt
    tok = data.transcript_to_token(
        transcript, token2id, frame_shift_ms, unk, skip_frame_times or feat_sizing,
    )
    if feat_sizing:
        tok = tok.unsqueeze(-1)
    path = os.path.join(dir_, basename)
    torch.save(tok, path)


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
    _add_common_arg(parser, "--mp-chunk-size")
    size_group = parser.add_mutually_exclusive_group()
    _add_common_arg(size_group, "--skip-frame-times")
    _add_common_arg(size_group, "--feat-sizing")
    try:
        options = parser.parse_args(args)
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

    os.makedirs(options.dir, exist_ok=True)
    _multiprocessor_pattern(
        error_handling_iter(),
        options,
        _save_transcripts_to_dir_do_work,
        token2id,
        options.dir,
        None,  # frame_shift_ms
        options.unk_symbol,
        options.skip_frame_times,
        options.feat_sizing,
    )
    return 0


class _DirectoryDataset(torch.utils.data.Dataset):
    def __init__(self, dir_, file_prefix, file_suffix):
        super().__init__()
        fpl = len(file_prefix)
        fsl = len(file_suffix)
        self.utt_ids = sorted(
            x[fpl : len(x) - fsl]
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
    parser = argparse.ArgumentParser(
        description=torch_token_data_dir_to_trn.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dir", type=rdir, help="The directory to read token sequences from"
    )
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

    try:
        options = parser.parse_args(args)
    except SystemExit as ex:
        return ex.code
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
    _add_common_arg(parser, "--mp-chunk-size")
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

    try:
        options = parser.parse_args(args)
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

    transcripts = (
        (options.file_prefix + x[0] + options.file_suffix, x[1])
        for x in data.read_ctm(options.ctm, wc2utt)
    )

    os.makedirs(options.dir, exist_ok=True)
    _multiprocessor_pattern(
        transcripts,
        options,
        _save_transcripts_to_dir_do_work,
        token2id,
        options.dir,
        options.frame_shift_ms,
        options.unk_symbol,
        options.skip_frame_times,
        options.feat_sizing,
    )
    return 0


def textgrids_to_torch_token_data_dir(args: Optional[Sequence[str]] = None):
    """Convert a directory of TextGrid files into a SpectDataSet ref/ dir

A "TextGrid" file is a transcription file for a single utterance used by the Praat
software (https://www.fon.hum.uva.nl/praat/).

This command accepts a directory of TextGrid files

    tg_dir/
        <file-prefix>utt_1.<textgrid_suffix>
        <file-prefix>utt_2.<textgrid_suffix>
        ...

and writes each file as a separate token sequence compatible with the "ref/" directory
of a SpectDataSet. If the extracted tier is an IntervalTier, the start and end points
will be saved with each token. If a TextTier (PointTier), the start and end points of
each segment will be identified with the point.

See the command "get-torch-spect-data-dir-info" for more info about a SpectDataSet
directory."""
    parser = argparse.ArgumentParser(
        description=textgrids_to_torch_token_data_dir.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "tg_dir", type=rdir, help="The directory containing the TextGrid files"
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
    _add_common_arg(parser, "--mp-chunk-size")
    _add_common_arg(parser, "--textgrid-suffix")
    parser.add_argument(
        "--fill-symbol",
        default=None,
        help="If set, unlabelled intervals in the TextGrid files will be "
        "assigned this symbol. Relevant only if a point grid.",
    )
    size_group = parser.add_mutually_exclusive_group()
    _add_common_arg(size_group, "--skip-frame-times")
    _add_common_arg(size_group, "--feat-sizing")
    _add_common_arg(size_group, "--frame-shift-ms")
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

    try:
        options = parser.parse_args(args)
    except SystemExit as ex:
        return ex.code
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
        options.tier_id = config.DEFT_TEXTGRID_TIER_ID

    def textgrid_iter():
        for file_name in os.listdir(options.tg_dir):
            if not file_name.endswith(
                options.textgrid_suffix
            ) or not file_name.startswith(options.file_prefix):
                continue
            basename = (
                file_name[: len(file_name) - len(options.textgrid_suffix)]
                + options.file_suffix
            )
            yield basename, data.read_textgrid(
                os.path.join(options.tg_dir, file_name),
                options.tier_id,
                options.fill_symbol,
            )[0]

    os.makedirs(options.dir, exist_ok=True)
    _multiprocessor_pattern(
        textgrid_iter(),
        options,
        _save_transcripts_to_dir_do_work,
        token2id,
        options.dir,
        options.frame_shift_ms,
        options.unk_symbol,
        options.skip_frame_times,
        options.feat_sizing,
    )
    return 0


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
    parser = argparse.ArgumentParser(
        description=torch_token_data_dir_to_ctm.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dir", type=rdir, help="The directory to read token sequences from"
    )
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
        default=config.DEFT_CTM_CHANNEL,
        help='If neither "--wc2utt" nor "--utt2wc" is specified, utterance '
        "IDs are treated as wavefile names and are given the value of this "
        "flag as a channel",
    )

    try:
        options = parser.parse_args(args)
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
    parser = argparse.ArgumentParser(
        description=compute_torch_token_data_dir_error_rates.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dir",
        type=rdir,
        help="If the 'hyp' argument is not specified, this is the "
        "parent directory of two subdirectories, 'ref/' and 'hyp/', which "
        "contain the reference and hypothesis transcripts, respectively. If "
        "the '--hyp' argument is specified, this is the reference "
        "transcript directory",
    )
    parser.add_argument(
        "hyp",
        nargs="?",
        type=rdir,
        default=None,
        help="The hypothesis transcript directory",
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
        type=nat,
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
        default=(config.DEFT_INS_COST, config.DEFT_DEL_COST, config.DEFT_SUB_COST),
        help="The costs of an insertion, deletion, and substitution, respectively",
    )
    group.add_argument(
        "--nist-costs",
        action="store_true",
        default=False,
        help="Use NIST (sclite, score) default costs for insertions, deletions, and "
        "substitutions (3/3/4)",
    )

    try:
        options = parser.parse_args(args)
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
    parser = argparse.ArgumentParser(
        description=torch_spect_data_dir_to_wds.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dir", type=rdir, help="The torch data directory")
    parser.add_argument("tar_path", help="The path to store files to")
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--feat-subdir")
    _add_common_arg(parser, "--ali-subdir")
    _add_common_arg(parser, "--ref-subdir")
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
        type=nat,
        default=1e5,
        help="If sharding ('--shard' is specified), dictates the number of samples in "
        "each file.",
    )
    parser.add_argument(
        "--max-size-per-shard",
        type=nat,
        default=3e9,
        help="If sharding ('--shard' is specified), dictates the maximum size in bytes "
        "of each file.",
    )
    try:
        options = parser.parse_args(args)
    except SystemExit as ex:
        return ex.code

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
    parser = argparse.ArgumentParser(
        description=compute_mvn_stats_for_torch_feat_data_dir.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
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
    parser.add_argument("dir", type=rdir, help="The feature directory")
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
    try:
        options = parser.parse_args(args)
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


def _torch_token_data_dir_to_torch_ali_dir_do_work(
    basename: str, ref_dir: str, ali_dir: str, feat_dir: Optional[str] = None,
):
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
    ali = torch.repeat_interleave(ref[:, 0], ref[:, 2] - ref[:, 1])
    torch.save(ali, os.path.join(ali_dir, basename))


def torch_token_data_dir_to_torch_ali_data_dir(args: Optional[Sequence[str]] = None):
    """Convert a ref/ dir to an ali/ dir

This command converts a "ref/" directory from a SpectDataSet to an "ali/" directory. The
former contains sequences of tokens; the latter contains frame-wise alignments. The
token ids are set to the frame-wise labels.

A reference token sequence "ref" partitions a frame sequence of length T if

1. ref is of shape (R, 3), with R > 1 and all ref[r, 1:] >= 0 (it contains segment
   boundaries).
2. ref[0, 1] = 0 (it starts at frame 0).
3. for all 0 <= r < R - 1, ref[r, 2] = ref[r + 1, 1] (boundaries contiguous).
4. ref[R - 1, 2] = T (it ends after T frames).

When ref partitions the frame sequence, it can be converted into a per-frame alignment
tensor "ali" of shape (T,), where ref[r, 1] <= t < ref[r, 2] implies ali[t] = ref[r, 0].

WARNING! This operation is potentially destructive: a per-frame alignment cannot
distinguish between two of the same token next to one another and one larger token.

See the command "get-torch-spect-data-dir-info" for more info SpectDataSet directories.
"""
    parser = argparse.ArgumentParser(
        description=torch_token_data_dir_to_torch_ali_data_dir.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ref_dir", type=rdir, help="The token sequence data directory (input)",
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
    _add_common_arg(parser, "--mp-chunk-size")
    try:
        options = parser.parse_args(args)
    except SystemExit as ex:
        return ex.code

    basenames = (
        x
        for x in os.listdir(options.ref_dir)
        if x.startswith(options.file_prefix) and x.endswith(options.file_prefix)
    )
    os.makedirs(options.ali_dir, exist_ok=True)
    _multiprocessor_pattern(
        basenames,
        options,
        _torch_token_data_dir_to_torch_ali_dir_do_work,
        options.ref_dir,
        options.ali_dir,
        options.feat_dir,
    )
    return 0


def _torch_ali_dir_to_torch_token_dir_do_work(
    basename: str, ali_dir: str, ref_dir: str,
):
    zeros_ = torch.zeros(1, dtype=torch.long)
    ali_path = os.path.join(ali_dir, basename)
    ali = torch.load(ali_path)
    tok, c = ali.unique_consecutive(return_counts=True)
    c = torch.cat([zeros_, c]).cumsum(0)
    start, end = c[:-1], c[1:]
    ref = torch.stack([tok, start, end], -1)
    torch.save(ref, os.path.join(ref_dir, basename))


def torch_ali_data_dir_to_torch_token_data_dir(args: Optional[Sequence[str]] = None):
    """Convert an ali/ dir to a ref/ dir

This command converts a "ali/" directory from a SpectDataSet to an "ref/" directory.
The former contains frame-wise alignments; the latter contains token sequences. The
frame-wise labels are set to the token ids.

To construct the token sequence, the alignment sequence is partitioned into segments,
each segment corresponding to the longest contiguous span of the same frame-wise label.

See the command "get-torch-spect-data-dir-info" for more info SpectDataSet directories.
"""
    parser = argparse.ArgumentParser(
        description=torch_ali_data_dir_to_torch_token_data_dir.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ali_dir", type=rdir, help="The frame alignment data directory (input)",
    )
    parser.add_argument(
        "ref_dir", help="The token sequence data directory (output)",
    )
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--num-workers")
    _add_common_arg(parser, "--mp-chunk-size")
    try:
        options = parser.parse_args(args)
    except SystemExit as ex:
        return ex.code

    basenames = (
        x
        for x in os.listdir(options.ali_dir)
        if x.startswith(options.file_prefix) and x.endswith(options.file_prefix)
    )
    os.makedirs(options.ref_dir, exist_ok=True)
    _multiprocessor_pattern(
        basenames,
        options,
        _torch_ali_dir_to_torch_token_dir_do_work,
        options.ali_dir,
        options.ref_dir,
    )
    return 0


def _torch_token_data_dir_to_textgrids_do_work(
    utt_id: str,
    ref_dir: str,
    id2token: Dict[int, str],
    feat_dir: Optional[str],
    tg_dir: str,
    in_suffix: str,
    out_suffix: str,
    frame_shift_ms: float,
    tier_name: str,
    precision: int,
    quiet: bool,
    force_method: Optional[Literal[1, 2, 3]],
):
    in_name = utt_id + in_suffix
    ref_name = os.path.join(ref_dir, in_name)
    ref = torch.load(ref_name)
    err_msg = f"Failure converting '{ref_name}' to TextGrid:"
    T = -1
    has_segment_index = ref.ndim == 2 and ref.size(1) == 3
    if not has_segment_index and ref.ndim != 1:
        raise ValueError(f"{err_msg} tensor is an invalid size")
    if feat_dir is not None:
        feat_name = os.path.join(feat_dir, in_name)
        if not os.path.isfile(feat_name):
            raise ValueError(
                f"{err_msg} corresponding feature file '{feat_name}' does not exist"
            )
        feat = torch.load(feat_name)
        if feat.ndim != 2:
            raise ValueError(f"{err_msg} feature tensor is an invalid size")
        T = feat.size(0)
    elif has_segment_index:
        T = ref[..., 1:].max()
    else:
        if not quiet:
            warnings.warn(f"Could not determine length of '{ref_name}'. Setting to 0")
        T = 0
    T = (T * frame_shift_ms) / 1000
    try_method = force_method if force_method else 1
    if try_method == 1:
        if (
            has_segment_index
            and ((ref[..., 2] > ref[..., 1]) & (ref[..., 1] >= 0)).all()
        ):
            point_tier = False
        elif force_method:
            raise ValueError(f"{err_msg} does not have enough info for method 1")
        else:
            try_method += 1
    if try_method == 2:
        if has_segment_index:
            maxes = ref[..., 1:].max(1)[0]
        else:
            maxes = torch.tensor(-1)
        if (maxes >= 0).all():
            ref[..., 1:] = maxes.unsqueeze(1)
            point_tier = True
        elif force_method:
            raise ValueError(f"{err_msg} does not have enough info for method 2")
        else:
            try_method += 1
    if try_method == 3:
        if ref.ndim != 1:
            ref = ref[..., 0]
        transcript = [(" ".join(id2token.get(x.item(), x.item()) for x in ref), 0.0, T)]
        point_tier = False
    else:
        transcript = data.token_to_transcript(ref, id2token, frame_shift_ms)
    if any(isinstance(x[0], int) for x in transcript):
        raise ValueError(f"{err_msg} not all ids exist in '{id2token.name}'")
    try:
        data.write_textgrid(
            transcript,
            os.path.join(tg_dir, utt_id + out_suffix),
            0.0,
            T,
            tier_name,
            point_tier,
            precision,
        )
    except Exception as e:
        raise ValueError(f"{err_msg} could not write textgrid") from e


def torch_token_data_dir_to_textgrids(args: Optional[Sequence[str]] = None):
    """Convert a SpectDataSet ref/ dir into a directory of TextGrid files

A "TextGrid" file is a transcription file for a single utterance used by the Praat
software (https://www.fon.hum.uva.nl/praat/).

This command accepts a directory of token sequences compatible with the "ref/"
directory of a SpectDataSet and outputs a directory of TextGrid files

    tg_dir/
        <file-prefix>utt_1.<textgrid_suffix>
        <file-prefix>utt_2.<textgrid_suffix>
        ...

A token sequence ref is a tensor of shape either (R, 3) or just (R,). The latter has no
segment information and is just the tokens. The former contains triples "tok, start,
end", where "tok" is the token id, "start" is the starting frame inclusive, and "end" is
the ending frame exclusive. A negative value for either boundary means the information
is not available.

By default, this command tries to save the sequence as a tier preserving as much
information in the token sequence as possible in a consistent way. The following methods
are attempted in order:

1. If ref is of shape (R, 3), all segments boundaries are available, and all segments
   are of nonzero length, the sequence will be saved as an IntervalTier containing
   segment boundaries.
2. If ref is of shape (R, 3) and either the start or end boundary is available for every
   token, the sequence will be saved as a TextTier (PointTier) with points set to the
   available boundary (with precedence going to the greater).
3. Otherwise, the token sequence is written as an interval tier with a single segment
   spanning the recording and containing all tokens.

In addition, the total length of the features in frames must be determined. Either the
flag "--feat-dir" must be specified in order to get the length directly from the feature
sequences, or "--infer" must be specified. The latter guesses the length to be the
maximum end boundary of the token sequence available, or 0 (with a warning if "--quiet"
unset) if none are.

Note that Praat usually works either with point data or with intervals which
collectively partition the audio. It can parse TextGrid files with non-contiguous
intervals, but they are rendered strangely.

See the command "get-torch-spect-data-dir-info" for more info about a SpectDataSet
directory."""
    parser = argparse.ArgumentParser(
        description=torch_token_data_dir_to_textgrids.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ref_dir", type=rdir, help="The token sequence data directory (input)"
    )
    _add_common_arg(parser, "id2token")
    parser.add_argument("tg_dir", help="The TextGrid directory (output)")
    len_opt = parser.add_mutually_exclusive_group(required=True)
    len_opt.add_argument("--feat-dir", default=None, help="Path to features")
    len_opt.add_argument(
        "--infer",
        action="store_true",
        default=False,
        help="Infer lengths based on maximum segment boundaries",
    )
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--swap")
    _add_common_arg(parser, "--frame-shift-ms")
    _add_common_arg(parser, "--num-workers")
    _add_common_arg(parser, "--mp-chunk-size")
    _add_common_arg(parser, "--textgrid-suffix")
    parser.add_argument(
        "--tier-name", default="transcript", help="The name to save the tier with"
    )
    parser.add_argument(
        "--precision",
        type=nat0,
        default=config.DEFT_FLOAT_PRINT_PRECISION,
        help="Precision with which to save floating point values in TextGrid files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="If set, suppresses warnings when lengths cannot be determined",
    )
    parser.add_argument(
        "--force-method",
        default=None,
        type=int,
        choices=[1, 2, 3],
        help="Force a specific method of writing to TextGrid (1-3 above). Not enough "
        "information will lead to an error.",
    )

    try:
        options = parser.parse_args(args)
    except SystemExit as ex:
        return ex.code
    if not os.path.isdir(options.ref_dir):
        raise ValueError(f"'{options.ref_dir}' is not a directory")
    if options.feat_dir is not None and not os.path.isdir(options.feat_dir):
        raise ValueError(f"'{options.feat_dir}' is not a directory")

    id2token = _parse_token2id(options.id2token, not options.swap, options.swap)
    utt_ids = (
        x[: len(x) - len(options.file_suffix)]
        for x in os.listdir(options.ref_dir)
        if x.startswith(options.file_prefix) and x.endswith(options.file_suffix)
    )
    os.makedirs(options.tg_dir, exist_ok=True)
    _multiprocessor_pattern(
        utt_ids,
        options,
        _torch_token_data_dir_to_textgrids_do_work,
        options.ref_dir,
        id2token,
        options.feat_dir,
        options.tg_dir,
        options.file_suffix,
        options.textgrid_suffix,
        options.frame_shift_ms,
        options.tier_name,
        options.precision,
        options.quiet,
        options.force_method,
    )


def _chunk_torch_spect_data_dir_do_work(
    utt_id: str,
    in_feat_dir: str,
    in_ali_dir: Optional[str],
    in_ref_dir: Optional[str],
    file_prefix: str,
    file_suffix: str,
    policy: str,
    lobe_size: int,
    window_type: str,
    pad_mode: Optional[str],
    pad_constant: float,
    partial_tokens: bool,
    retain_token_boundaries: bool,
    quiet: bool,
    format_utt: str,
    out_feat_dir: str,
    out_ali_dir: str,
    out_ref_dir: str,
):
    slicer = SliceSpectData(policy, window_type, pad_mode is None, lobe_size)
    chunker = ChunkBySlices("constant" if pad_mode is None else pad_mode, pad_constant)
    ref_chunker = ChunkTokenSequencesBySlices(partial_tokens, retain_token_boundaries)
    in_basename = file_prefix + utt_id + file_suffix
    feats = torch.load(os.path.join(in_feat_dir, in_basename)).unsqueeze(0)
    if in_ali_dir is None:
        alis = None
    else:
        alis = torch.load(os.path.join(in_ali_dir, in_basename)).unsqueeze(0)
    if in_ref_dir is None:
        refs = None
    else:
        refs = torch.load(os.path.join(in_ref_dir, in_basename)).unsqueeze(0)
    if policy == "fixed":
        slices, _ = slicer(feats)
    elif policy == "ali":
        slices, _ = slicer(alis)
    else:
        slices, _ = slicer(refs)
    M = slices.size(0)
    new_utt_ids = [
        format_utt.format(utt_id=utt_id, idx=n, start=x[0], end=x[1])
        for (n, x) in enumerate(slices.tolist())
    ]
    if not quiet and len(set(new_utt_ids)) != M:
        warnings.warn(f"new utterance names for '{utt_id}' are not unique")
    feats, lens = chunker(feats.expand(M, -1, -1), slices)
    if alis is not None:
        alis, lens_ = chunker(alis.expand(M, -1), slices)
        assert (lens == lens_).all()
    if refs is not None:
        refs, ref_lens = ref_chunker(refs.expand(M, *refs.shape[1:]), slices)
    for n, new_utt_id in enumerate(new_utt_ids):
        out_basename = file_prefix + new_utt_id + file_suffix
        torch.save(feats[n, : lens[n]], os.path.join(out_feat_dir, out_basename))
        if alis is not None:
            torch.save(alis[n, : lens[n]], os.path.join(out_ali_dir, out_basename))
        if refs is not None:
            torch.save(refs[n, : ref_lens[n]], os.path.join(out_ref_dir, out_basename))


def chunk_torch_spect_data_dir(args: Optional[Sequence[str]] = None):
    """Create a new SpectDataSet directory by chunking another

This command breaks SpectDataSet sequences into sub-sequences (chunks), storing the
results in a new directory. New utterances are named according to "--format-utt".

Sequences are sliced according to one of three policies set by the "--policy" flag
(default "fixed"). They are:

- fixed: extract a fixed-sized window at fixed-length intervals along the feature
         sequence.
- ali: use per-frame alignments to segment the feature sequence into intervals with
       matching labels. Requires per-frame alignments (data in the "ali/" subdirectory).
- ref: use reference token sequence segments as slices. Requires reference sequences
       (data in the "ali/" subdirectory) and for them to contain segment boundary
       information.

Overlapping chunks may be created by specifying "--lobe-size" (default "0") and
"--window-type" (default "symmetric"). More details on the policies and windowing can
be found in the Python module pydrobert.torch.modules.SliceSpectData.

By default, only valid slices (i.e. those entirely within the boundaries of the input
sequences) are counted. Specifying "--pad-mode" will include slices partially within
boundaries as well as how to pad features and per-frame alignments to fill the
remainder.

See the command "get-torch-spect-data-dir-info" for more info SpectDataSet directories.
"""
    parser = argparse.ArgumentParser(
        description=chunk_torch_spect_data_dir.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "in_dir", type=rdir, help="The torch data directory to chunk (input)"
    )
    parser.add_argument(
        "out_dir", help="The torch data directory to store chunks (output)"
    )
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--feat-subdir")
    _add_common_arg(parser, "--ali-subdir")
    _add_common_arg(parser, "--ref-subdir")
    _add_common_arg(parser, "--num-workers")
    _add_common_arg(parser, "--mp-chunk-size")
    parser.add_argument(
        "--policy",
        default="fixed",
        choices=["fixed", "ali", "ref"],
        help="The policy for determining slices from the data. See SliceSpectData.",
    )
    parser.add_argument(
        "--lobe-size",
        type=nat0,
        default=0,
        help="Size of a side lobe of a slice. See SliceSpectData.",
    )
    parser.add_argument(
        "--window-type",
        default="symmetric",
        choices=["symmetric", "causal", "future"],
        help="Type of window used in slicing. See SliceSpectData.",
    )
    parser.add_argument(
        "--pad-mode",
        default=None,
        choices=["constant", "reflect", "replicate"],
        help="If specified, determines how to chunks of features and alignments "
        " exceeding the original sequence boundaries. constant: pad with the value of "
        "'--pad-constant'. reflect: padded values are the reflection around sequence "
        "boundaries. replicate: padded values match the first and final sequence "
        "values.",
    )
    parser.add_argument(
        "--pad-constant",
        type=float,
        default=0.0,
        help="Constant used when padding with '--pad-mode=constant'",
    )
    parser.add_argument(
        "--partial-tokens",
        action="store_true",
        default=False,
        help="If set, reference token sequences which only partly overlap with a chunk "
        "will still be included with the chunk.",
    )
    parser.add_argument(
        "--retain-token-boundaries",
        action="store_true",
        default=False,
        help="If set, segment boundaries of reference token sequences will keep their "
        "original values rather than being made relative to the chunk.",
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False, help="Suppress any warnings.",
    )
    parser.add_argument(
        "--format-utt",
        default=r"{utt_id}.{start:05d}.{end:05d}",
        help="Format string with which to format utterance ids of chunks. Available "
        "keys are 'utt_id': the old utterance id, 'start': the start frame of the "
        "chunk (inclusive), 'end': the end frame of the chunk (exclusive), and 'idx': "
        "the 0-index of the chunk within the utterance",
    )
    try:
        options = parser.parse_args(args)
    except SystemExit as ex:
        return ex.code

    if any(
        os.path.isabs(x)
        for x in (options.feat_subdir, options.ref_subdir, options.ali_subdir)
    ):
        print(
            "--feat-subdir, --ali-subdir, and --ref-subdir must all be relative paths",
            file=sys.stderr,
        )
        return 1
    in_feat_dir = os.path.join(options.in_dir, options.feat_subdir)
    in_ali_dir = os.path.join(options.in_dir, options.ali_subdir)
    in_ref_dir = os.path.join(options.in_dir, options.ref_subdir)
    if not os.path.isdir(in_feat_dir):
        print(f"'{in_feat_dir}' does not exist", file=sys.stderr)
        return 1
    out_feat_dir = os.path.join(options.out_dir, options.feat_subdir)
    out_ali_dir = os.path.join(options.out_dir, options.ali_subdir)
    out_ref_dir = os.path.join(options.out_dir, options.ref_subdir)
    os.makedirs(out_feat_dir, exist_ok=True)
    if os.path.isdir(in_ali_dir):
        os.makedirs(out_ali_dir, exist_ok=True)
    else:
        in_ali_dir = None
    if os.path.isdir(in_ref_dir):
        os.makedirs(out_ref_dir, exist_ok=True)
    else:
        in_ref_dir = None

    utt_ids = iter(
        data.SpectDataSet(
            options.in_dir,
            options.file_prefix,
            options.file_suffix,
            not options.quiet,
            feat_subdir=options.feat_subdir,
            ali_subdir=options.ali_subdir,
            ref_subdir=options.ref_subdir,
            suppress_alis=False,
            tokens_only=False,
        ).utt_ids
    )

    _multiprocessor_pattern(
        utt_ids,
        options,
        _chunk_torch_spect_data_dir_do_work,
        in_feat_dir,
        in_ali_dir,
        in_ref_dir,
        options.file_prefix,
        options.file_suffix,
        options.policy,
        options.lobe_size,
        options.window_type,
        options.pad_mode,
        options.pad_constant,
        options.partial_tokens,
        options.retain_token_boundaries,
        options.quiet,
        options.format_utt,
        out_feat_dir,
        out_ali_dir,
        out_ref_dir,
    )


def _copy_spect_data_dir_do_work(
    basename: str,
    src_dir: str,
    dest_dir: str,
    cp: Callable[[str, str], Any],
    feat_subdir: Optional[str],
    ali_subdir: Optional[str],
    ref_subdir: Optional[str],
):
    if feat_subdir is None:
        cp(os.path.join(src_dir, basename), os.path.join(dest_dir, basename))
    else:
        for x in (feat_subdir, ali_subdir, ref_subdir):
            if x is not None:
                src = os.path.join(src_dir, x, basename)
                if os.path.exists(src):
                    cp(src, os.path.join(dest_dir, x, basename))


def subset_torch_spect_data_dir(args: Optional[Sequence[str]] = None):
    """Make a new SpectDataDir from a subset of utterances of another

This command determines a set of utterances via a flag, then hard links all files in the
"feat/", "ali/" and "ref/" subdirectories matching the utterance id to in the "src"
directory to the "dest" directory.

See the command "get-torch-spect-data-dir-info" for more info about a SpectDataSet
directory.
"""
    parser = argparse.ArgumentParser(
        description=subset_torch_spect_data_dir.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Available utterances to extract are determined by the contents of the "feat/"
subdirectory, unless "--only" was specified. Any extra or missing utterances in "ali/"
and "ref/" will be ignored.

If "--utt-list" or "--utt-list-file" is chosen, this command ignores any missing
utterances.

When a criterion involves extracting some number of utterances which exceeds the total
number of utterances, that total is extracted instead.

Ratios are rounded down to the nearest utterance.

Sorting by id is performed according to python's sort method, i.e. by locale.

When "--only" is paired with "--shortest-*" or "--longest-*", "src" is assumed to also
be the directory to extract lengths from. Otherwise it's "feat/".

This command has a similar functionality to Kaldi's (https://github.com/kaldi-asr)
subset_data_dir.sh script, but defaults to hard links for cross-compatibility.
""",
    )
    parser.add_argument("src", type=rdir, help="The directory to extract from")
    parser.add_argument("dest", help="The directory to extract to")
    style = parser.add_mutually_exclusive_group()
    style.add_argument(
        "--copy",
        action="store_true",
        default=False,
        help="Copy extracted files (instead of hard link)",
    )
    style.add_argument(
        "--symlink",
        action="store_true",
        default=False,
        help="Symlink extracted files (instead of hard link). Symlinks will be "
        "relative to the destination.",
    )
    criteria = parser.add_mutually_exclusive_group(required=True)
    criteria.add_argument(
        "--utt-list",
        nargs="+",
        metavar="UTTID",
        default=None,
        help="Extract the utterances listed directly after this flag",
    )
    criteria.add_argument(
        "--utt-list-file",
        type=argparse.FileType("r"),
        metavar="PATH",
        default=None,
        help="Extract the utterances listed in the passed file, one-per-line",
    )
    criteria.add_argument(
        "--first-n",
        type=nat0,
        metavar="N",
        default=None,
        help="Extract this number of utterances listed first by id",
    )
    criteria.add_argument(
        "--first-ratio",
        type=closed01,
        metavar="R",
        default=None,
        help="Extract this ratio of utterances (rounding down) listed first by id",
    )
    criteria.add_argument(
        "--last-n",
        type=nat0,
        metavar="N",
        default=None,
        help="Extract this number of utterances listed last by id",
    )
    criteria.add_argument(
        "--last-ratio",
        type=closed01,
        metavar="R",
        default=None,
        help="Extract this ratio of utterances (rounding down) listed last by id",
    )
    criteria.add_argument(
        "--shortest-n",
        type=nat0,
        metavar="N",
        default=None,
        help="Extract this number of utterances listed first by increasing length, "
        "then by id",
    )
    criteria.add_argument(
        "--shortest-ratio",
        type=closed01,
        metavar="R",
        default=None,
        help="Extract this ratio of utterances listed first by increasing length, "
        "then by id",
    )
    criteria.add_argument(
        "--longest-n",
        type=nat0,
        metavar="N",
        default=None,
        help="Extract this number of utterances listed first by decreasing length, "
        "then by id",
    )
    criteria.add_argument(
        "--longest-ratio",
        type=closed01,
        metavar="R",
        default=None,
        help="Extract this ratio of utterances listed first by decreasing length, "
        "then by id",
    )
    criteria.add_argument(
        "--rand-n",
        type=nat0,
        metavar="N",
        default=None,
        help="Extract this number of utterances listed randomly",
    )
    criteria.add_argument(
        "--rand-ratio",
        type=closed01,
        metavar="R",
        default=None,
        help="Extract this ratio of utterances listed randomly",
    )
    parser.add_argument(
        "--only",
        action="store_true",
        default=False,
        help="If set, extract only the data directly stored in 'src'",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="Seed used in --rand-* flags for determinism. If unspecified, "
        "non-deterministic",
    )
    _add_common_arg(parser, "--feat-subdir")
    _add_common_arg(parser, "--ali-subdir")
    _add_common_arg(parser, "--ref-subdir")
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--num-workers")
    _add_common_arg(parser, "--mp-chunk-size")
    try:
        options = parser.parse_args(args)
    except SystemExit as ex:
        return ex.code

    if options.only:
        options.feat_subdir = options.ali_subdir = options.ref_subdir = None
        feat_dir = options.src
    else:
        feat_dir = os.path.join(options.src, options.feat_subdir)
        if not os.path.isdir(feat_dir):
            print(f"'{feat_dir}' does not exist", file=sys.stderr)
            return 1
        if not os.path.isdir(os.path.join(options.src, options.ali_subdir)):
            options.ali_subdir = None
        if not os.path.isdir(os.path.join(options.src, options.ref_subdir)):
            options.ref_subdir = None

    ds = _DirectoryDataset(feat_dir, options.file_prefix, options.file_suffix)
    if any(
        x is not None
        for x in (
            options.shortest_n,
            options.shortest_ratio,
            options.longest_n,
            options.longest_ratio,
        )
    ):
        dl = torch.utils.data.DataLoader(
            ds, batch_size=1, num_workers=options.num_workers, collate_fn=_noop_collate
        )
        pairs = [(x[1].size(0), x[0]) for x in dl]
        if options.shortest_n is not None or options.shortest_ratio is not None:
            pairs.sort()
        else:
            pairs.sort(key=lambda x: (-x[0], x[1]))
        all_utt_ids = [x[1] for x in pairs]
        del ds, dl, pairs
    else:
        all_utt_ids = ds.utt_ids
        del ds
        if options.last_n is not None or options.last_ratio is not None:
            all_utt_ids.sort(reverse=True)
        elif options.rand_n is not None or options.rand_ratio is not None:
            random.seed(options.seed)
            random.shuffle(all_utt_ids)

    if options.utt_list is not None or options.utt_list_file is not None:
        all_utt_ids = set(all_utt_ids)
        if options.utt_list_file is not None:
            utt_ids = (x.strip() for x in options.utt_list_file)
        else:
            utt_ids = options.utt_list
        utt_ids = (x for x in utt_ids if x in all_utt_ids)
    else:
        a = (
            0 if x is None else x
            for x in (
                options.shortest_n,
                options.longest_n,
                options.first_n,
                options.last_n,
                options.rand_n,
            )
        )
        b = (
            0 if x is None else int(len(all_utt_ids) * x)
            for x in (
                options.shortest_ratio,
                options.longest_ratio,
                options.first_ratio,
                options.last_ratio,
                options.rand_ratio,
            )
        )
        n = max(itertools.chain(a, b))
        utt_ids = iter(all_utt_ids[:n])
    basenames = (options.file_prefix + x + options.file_suffix for x in utt_ids)

    if options.copy:
        cp = shutil.copy
    elif options.symlink:
        def cp(src, dst):
            src = os.path.relpath(src, os.path.dirname(dst))
            return os.symlink(src, dst)
    else:
        cp = os.link

    os.makedirs(options.dest, exist_ok=True)
    for x in (options.feat_subdir, options.ali_subdir, options.ref_subdir):
        if x is not None:
            os.makedirs(os.path.join(options.dest, x), exist_ok=True)

    _multiprocessor_pattern(
        basenames,
        options,
        _copy_spect_data_dir_do_work,
        options.src,
        options.dest,
        cp,
        options.feat_subdir,
        options.ali_subdir,
        options.ref_subdir,
    )


def _print_torch_ali_data_dir_length_moments(file_name, exclude_ids):
    x = torch.load(file_name)
    counts, lens = x.unique_consecutive(return_counts=True)
    if exclude_ids is not None:
        not_excluded = (counts.unsqueeze(1) != exclude_ids).all(1)
        lens = lens[not_excluded]
    s, ss, c = lens.sum().item(), lens.square().sum().item(), lens.numel()
    return s, ss, c


def _do_mv_printing(s, ss, c, options):
    if c > 0:
        float_fmt_str = f"{{:0.0{options.precision}f}}"
        mean = s / c
        var = ss / c - mean ** 2
        mean = float_fmt_str.format(mean)
        if options.bessel and c == 1:
            var = "n/a"
        else:
            if options.bessel:
                var *= c / (c - 1)
            if options.std:
                var = math.sqrt(var)
            var = float_fmt_str.format(var)
        out_str = f"{mean} ({var})\n"
    else:
        out_str = "n/a (n/a)\n"
    options.out.write(out_str)


def print_torch_ali_data_dir_length_moments(args: Optional[Sequence[str]] = None):
    """Compute the mean and variance of segment lengths from an ali data dir

A segment in an "ali/" directory tensor is a maximal sequence of frames with the same
id. This command computes the mean and variance of segment lengths, printing them on one
line as

    <mean> (<var>)

The input to this command is the "ali/" subdirectory of the SpectDataSet, not its root.

See the command "get-torch-spect-data-dir-info" for more info about a SpectDataSet
directory."""
    parser = argparse.ArgumentParser(
        description=print_torch_ali_data_dir_length_moments.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dir", type=rdir, help="The ali/ dir (input)")
    parser.add_argument(
        "out",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Where to print statistics. Defaults to stdout",
    )
    parser.add_argument(
        "--precision",
        type=nat0,
        default=config.DEFT_FLOAT_PRINT_PRECISION,
        help="Precision with which to print stats",
    )
    parser.add_argument(
        "--bessel",
        action="store_true",
        default=False,
        help="Perform Bessel correction on the variance estimate",
    )
    parser.add_argument(
        "--std",
        action="store_true",
        default=False,
        help="Print standard deviation instead of variance",
    )
    parser.add_argument(
        "--exclude-ids",
        type=int,
        nargs="+",
        default=None,
        help="If specified, segments with ali ids in this list will be excluded from"
        "counts",
    )
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--num-workers")
    _add_common_arg(parser, "--mp-chunk-size")
    try:
        options = parser.parse_args(args)
    except SystemExit as ex:
        return ex.code

    filenames = (
        os.path.join(options.dir, x)
        for x in os.listdir(options.dir)
        if x.startswith(options.file_prefix) and x.endswith(options.file_suffix)
    )

    exclude_ids = options.exclude_ids
    if exclude_ids is not None:
        exclude_ids = torch.tensor(list(set(exclude_ids)))

    s = 0
    ss = 0
    c = 0
    for s_, ss_, c_ in _multiprocessor_pattern_generator(
        filenames, options, _print_torch_ali_data_dir_length_moments, exclude_ids
    ):
        s += s_
        ss += ss_
        c += c_

    _do_mv_printing(s, ss, c, options)


def _print_torch_ref_data_dir_length_moments(utt_id, dir_, prefix, suffix, exclude_ids):
    ref = torch.load(os.path.join(dir_, prefix + utt_id + suffix))
    eprefix = f"Utterance '{utt_id}':"
    if ref.ndim != 2 or ref.size(1) != 3:
        err_msg = f"{eprefix} expected tensor of shape '(R, 3)'; got '{ref.shape}'"
        return 0, 0, 0, err_msg
    lens = ref[:, 2] - ref[:, 1]
    valid = (0 <= ref[:, 1]) & (ref[:, 1] <= ref[:, 2])
    if exclude_ids is not None:
        not_excluded = (ref[:, :1] != exclude_ids).all(1)
    else:
        not_excluded = torch.ones_like(valid)
    invalid_and_not_excluded = ~valid & not_excluded
    if invalid_and_not_excluded.long().sum() != 0:
        idxs = invalid_and_not_excluded.nonzero().flatten().tolist()
        err_msg = f"{eprefix} segments {idxs} are invalid or missing"
    else:
        err_msg = None
    lens = lens[valid & not_excluded]
    s, ss, c = lens.sum().item(), lens.square().sum().item(), lens.numel()
    return s, ss, c, err_msg


def print_torch_ref_data_dir_length_moments(args: Optional[Sequence[str]] = None):
    """Compute the mean and variance of segment lengths from a ref data dir

Segment lengths from "ref/" directory tensors are computed from token segment boundaries
(end - start). For this to be computed, a token must include those boundaries. Those
utterances or segments without boundaries are discarded with a warning. This command
computes the mean and variance of segment lengths, printing them on one line as

    <mean> (<var>)

The input to this command is the "ref/" subdirectory of the SpectDataSet, not its root.

See the command "get-torch-spect-data-dir-info" for more info about a SpectDataSet
directory."""
    parser = argparse.ArgumentParser(
        description=print_torch_ali_data_dir_length_moments.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dir", type=rdir, help="The ref/ dir (input)")
    parser.add_argument(
        "out",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Where to print statistics. Defaults to stdout",
    )
    err_grp = parser.add_mutually_exclusive_group()
    err_grp.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Error when boundary info is not available",
    )
    err_grp.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress warnings about missing boundary info",
    )
    parser.add_argument(
        "--precision",
        type=nat0,
        default=config.DEFT_FLOAT_PRINT_PRECISION,
        help="Precision with which to print stats",
    )
    parser.add_argument(
        "--bessel",
        action="store_true",
        default=False,
        help="Perform Bessel correction on the variance estimate",
    )
    parser.add_argument(
        "--std",
        action="store_true",
        default=False,
        help="Print standard deviation instead of variance",
    )
    parser.add_argument(
        "--exclude-ids",
        type=int,
        nargs="+",
        default=None,
        help="If specified, segments with token ids in this list will be excluded from"
        "counts",
    )
    _add_common_arg(parser, "--file-prefix")
    _add_common_arg(parser, "--file-suffix")
    _add_common_arg(parser, "--num-workers")
    _add_common_arg(parser, "--mp-chunk-size")
    try:
        options = parser.parse_args(args)
    except SystemExit as ex:
        return ex.code

    utt_ids = (
        x[len(options.file_prefix) : len(x) - len(options.file_suffix)]
        for x in os.listdir(options.dir)
        if x.startswith(options.file_prefix) and x.endswith(options.file_suffix)
    )

    exclude_ids = options.exclude_ids
    if exclude_ids is not None:
        exclude_ids = torch.tensor(list(set(exclude_ids)))

    s = 0
    ss = 0
    c = 0
    for s_, ss_, c_, err_msg in _multiprocessor_pattern_generator(
        utt_ids,
        options,
        _print_torch_ref_data_dir_length_moments,
        options.dir,
        options.file_prefix,
        options.file_suffix,
        exclude_ids,
    ):
        if err_msg is not None:
            if options.strict:
                raise ValueError(err_msg)
            elif not options.quiet:
                warnings.warn(err_msg)
        s += s_
        ss += ss_
        c += c_

    _do_mv_printing(s, ss, c, options)


global _mp_args
global _mp_func


def _worker_init(func, *args):
    global _mp_args
    global _mp_func
    _mp_args = args
    _mp_func = func


def _worker_func(x_n):
    global _mp_func
    global _mp_args
    return _mp_func(x_n, *_mp_args)


def _multiprocessor_pattern(x, options, do_work_func, *args):
    collections.deque(
        iter(_multiprocessor_pattern_generator(x, options, do_work_func, *args)),
        maxlen=1,
    )


def _multiprocessor_pattern_generator(x, options, do_work_func, *args):
    if options.num_workers:
        with torch.multiprocessing.Pool(
            options.num_workers, _worker_init, (do_work_func, *args),
        ) as pool:
            yield from pool.imap_unordered(_worker_func, x, options.mp_chunk_size)
    else:
        yield from (do_work_func(x_n, *args) for x_n in x)
