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

import re
import warnings

from typing import (
    Dict,
    Tuple,
    Optional,
    List,
    TextIO,
    Union,
    Iterable,
    Sequence,
    Mapping,
)
from collections import OrderedDict

import torch
import numpy as np

import pydrobert.torch.config as config

from ._textgrid import TextGrid, TEXTTIER


def parse_arpa_lm(file_: Union[TextIO, str], token2id: Optional[dict] = None) -> list:
    r"""Parse an ARPA statistical language model

    An `ARPA language model <https://cmusphinx.github.io/wiki/arpaformat/>`__
    is an n-gram model with back-off probabilities. It is formatted as::

        \data\
        ngram 1=<count>
        ngram 2=<count>
        ...
        ngram <N>=<count>

        \1-grams:
        <logp> <token[t]> <logb>
        <logp> <token[t]> <logb>
        ...

        \2-grams:
        <logp> <token[t-1]> <token[t]> <logb>
        ...

        \<N>-grams:
        <logp> <token[t-<N>+1]> ... <token[t]>
        ...

        \end\

    Parameters
    ----------
    file_
        Either the path or a file pointer to the file.
    token2id
        A dictionary whose keys are token strings and values are ids. If set, tokens
        will be replaced with ids on read

    Returns
    -------
    prob_list : list
        A list of the same length as there are orders of n-grams in the
        file (e.g. if the file contains up to tri-gram probabilities then
        `prob_list` will be of length 3). Each element is a dictionary whose
        key is the word sequence (earliest word first). For 1-grams, this is
        just the word. For n > 1, this is a tuple of words. Values are either
        a tuple of ``logp, logb`` of the log-probability and backoff
        log-probability, or, in the case of the highest-order n-grams that
        don't need a backoff, just the log probability.
    
    Warnings
    --------
    This function is not safe for JIT scripting or tracing.
    """
    if isinstance(file_, str):
        with open(file_) as f:
            return parse_arpa_lm(f, token2id=token2id)
    line = ""
    for line in file_:
        if line.strip() == "\\data\\":
            break
    if line.strip() != "\\data\\":
        raise IOError("Could not find \\data\\ line. Is this an ARPA file?")
    ngram_counts = []
    count_pattern = re.compile(r"^ngram\s+(\d+)\s*=\s*(\d+)$")
    for line in file_:
        line = line.strip()
        if not line:
            continue
        match = count_pattern.match(line)
        if match is None:
            break
        n, count = (int(x) for x in match.groups())
        if len(ngram_counts) < n:
            ngram_counts.extend(0 for _ in range(n - len(ngram_counts)))
        ngram_counts[n - 1] = count
    prob_list = [dict() for _ in ngram_counts]
    ngram_header_pattern = re.compile(r"^\\(\d+)-grams:$")
    ngram_entry_pattern = re.compile(r"^(-?\d+(?:\.\d+)?)\s+(.*)$")
    while line != "\\end\\":
        match = ngram_header_pattern.match(line)
        if match is None:
            raise IOError('line "{}" is not valid'.format(line))
        ngram = int(match.group(1))
        if ngram > len(ngram_counts):
            raise IOError(
                "{}-grams count was not listed, but found entry" "".format(ngram)
            )
        dict_ = prob_list[ngram - 1]
        for line in file_:
            line = line.strip()
            if not line:
                continue
            match = ngram_entry_pattern.match(line)
            if match is None:
                break
            logp, rest = match.groups()
            tokens = tuple(rest.strip().split())
            # IRSTLM and SRILM allow for implicit backoffs on non-final
            # n-grams, but final n-grams must not have backoffs
            logb = 0.0
            if len(tokens) == ngram + 1 and ngram < len(prob_list):
                try:
                    logb = float(tokens[-1])
                    tokens = tokens[:-1]
                except ValueError:
                    pass
            if len(tokens) != ngram:
                raise IOError(
                    'expected line "{}" to be a(n) {}-gram' "".format(line, ngram)
                )
            if token2id is not None:
                tokens = tuple(token2id[tok] for tok in tokens)
            if ngram == 1:
                tokens = tokens[0]
            if ngram != len(ngram_counts):
                dict_[tokens] = (float(logp), logb)
            else:
                dict_[tokens] = float(logp)
    if line != "\\end\\":
        raise IOError("Could not find \\end\\ line")
    for ngram_m1, (ngram_count, dict_) in enumerate(zip(ngram_counts, prob_list)):
        if len(dict_) != ngram_count:
            raise IOError(
                "Expected {} {}-grams, got {}".format(ngram_count, ngram_m1, len(dict_))
            )
    return prob_list


class _AltTree(object):
    def __init__(self, parent=None):
        self.parent = parent
        self.tokens = []
        if parent is not None:
            parent.tokens.append([self.tokens])

    def new_branch(self):
        assert self.parent
        self.tokens = []
        self.parent.tokens[-1].append(self.tokens)


def _trn_line_to_transcript(x: Tuple[str, bool]) -> Optional[Tuple[str, List[str]]]:
    line, warn = x
    line = line.strip()
    if not line:
        return None
    try:
        last_open = line.rindex("(")
        last_close = line.rindex(")")
        if last_open > last_close:
            raise ValueError()
    except ValueError:
        raise IOError("Line does not end in utterance id")
    utt_id = line[last_open + 1 : last_close]
    line = line[:last_open].strip()
    transcript = []
    token = ""
    alt_tree = _AltTree()
    found_alt = False
    while len(line):
        c = line[0]
        line = line[1:]
        if c == "{":
            found_alt = True
            if token:
                if alt_tree.parent is None:
                    transcript.append(token)
                else:
                    alt_tree.tokens.append(token)
                token = ""
            alt_tree = _AltTree(alt_tree)
        elif c == "/" and alt_tree.parent is not None:
            if token:
                alt_tree.tokens.append(token)
                token = ""
            alt_tree.new_branch()
        elif c == "}" and alt_tree.parent is not None:
            if token:
                alt_tree.tokens.append(token)
                token = ""
            if not alt_tree.tokens:
                raise IOError('Empty alternate found ("{ }")')
            alt_tree = alt_tree.parent
            if alt_tree.parent is None:
                assert len(alt_tree.tokens) == 1
                transcript.append((alt_tree.tokens[0], -1, -1))
                alt_tree.tokens = []
        elif c == " ":
            if token:
                if alt_tree.parent is None:
                    transcript.append(token)
                else:
                    alt_tree.tokens.append(token)
                token = ""
        else:
            token += c
    if token and alt_tree.parent is None:
        transcript.append(token)
    if found_alt and warn:
        warnings.warn(
            'Found an alternate in transcription for utt="{}". '
            "Transcript will contain an array of alternates at that "
            "point, and will not be compatible with transcript_to_token "
            "until resolved. To suppress this warning, set warn=False"
            "".format(utt_id)
        )
    return utt_id, transcript


def read_trn_iter(
    trn: Union[TextIO, str],
    warn: bool = True,
    processes: int = 0,
    chunk_size: int = config.DEFT_CHUNK_SIZE,
) -> Tuple[str, List[str]]:
    """Read a NIST sclite transcript file, yielding individual transcripts

    Identical to :func:`read_trn`, but yields individual transcript entries rather than
    a full list. Ideal for large transcript files.

    Parameters
    ----------
    trn
    warn
    processes
    chunk_size

    Yields
    ------
    utt_id : str
    transcript : list of str
    """
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
    if isinstance(trn, str):
        with open(trn) as trn:
            yield from read_trn_iter(trn, warn, processes)
    elif processes == 0:
        for line in trn:
            x = _trn_line_to_transcript((line, warn))
            if x is not None:
                yield x
    else:
        with torch.multiprocessing.Pool(processes) as pool:
            transcripts = pool.imap(
                _trn_line_to_transcript, ((line, warn) for line in trn), chunk_size
            )
            for x in transcripts:
                if x is not None:
                    yield x
            pool.close()
            pool.join()


def read_trn(
    trn: Union[TextIO, str],
    warn: bool = True,
    processes: int = 0,
    chunk_size: int = config.DEFT_CHUNK_SIZE,
) -> List[Tuple[str, List[str]]]:
    """Read a NIST sclite transcript file into a list of transcripts

    `sclite <http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm>`__
    is a commonly used scoring tool for ASR.

    This function converts a transcript input file ("trn" format) into a
    list of `transcripts`, where each element is a tuple of
    ``utt_id, transcript``. ``transcript`` is a list split by spaces.

    Parameters
    ----------
    trn
        The transcript input file. Will open if `trn` is a path.
    warn
        The "trn" format uses curly braces and forward slashes to indicate
        transcript alterations. This is largely for scoring purposes, such as
        swapping between filled pauses, not for training. If `warn` is
        :obj:`True`, a warning will be issued via the ``warnings`` module every
        time an alteration appears in the "trn" file. Alterations appear in
        `transcripts` as elements of ``([[alt_1_word_1, alt_1_word_2, ...],
        [alt_2_word_1, alt_2_word_2, ...], ...], -1, -1)`` so that
        :func:`transcript_to_token` will not attempt to process alterations as
        token start and end times.
    processes
        The number of processes used to parse the lines of the trn file. If
        ``0``, will be performed on the main thread. Otherwise, the file will
        be read on the main thread and parsed using `processes` many processes.
    chunk_size
        The number of lines to be processed by a worker process at a time.
        Applicable when ``processes > 0``

    Returns
    -------
    transcripts : list
        A list of pairs ``utt_id, transcript`` where `utt_id` is a string identifying
        the utterance and `transcript` is a list of tokens in the utterance's
        transcript.

    Notes
    -----
    Any null words (``@``) in the "trn" file are encoded verbatim.
    """
    return list(read_trn_iter(trn, warn, processes, chunk_size))


def write_trn(
    transcripts: Iterable[Tuple[str, List[str]]], trn: Union[str, TextIO]
) -> None:
    """From an iterable of transcripts, write to a NIST "trn" file

    This is largely the inverse operation of :func:`read_trn`. In general,
    elements of a transcript (`transcripts` contains pairs of ``utt_id,
    transcript``) could be tokens or tuples of ``x, start, end`` (providing the
    start and end times of tokens, respectively). However, ``start`` and
    ``end`` are ignored when writing "trn" files. ``x`` could be the token or a
    list of alternates, as described in :func:`read_trn`.

    Parameters
    ----------
    transcripts
    trn
    """
    if isinstance(trn, str):
        with open(trn, "w") as trn:
            return write_trn(transcripts, trn)

    def _handle_x(x):
        if isinstance(x, str):
            return x + " "  # x was a token
        # x is a list of alternates
        ret = []
        for alts in x:
            elem = ""
            for xx in alts:
                elem += _handle_x(xx)
            ret.append(elem)
        ret = "{ " + "/ ".join(ret) + "} "
        return ret

    for utt_id, transcript in transcripts:
        line = ""
        for x in transcript:
            # first get rid of starts and ends, if possible. This is not
            # ambiguous with numerical alternates, since alternates should
            # always be strings and, more importantly, always have placeholder
            # start and end values
            try:
                if len(x) == 3 and np.isreal(x[1]) and np.isreal(x[2]):
                    x = x[0]
            except TypeError:
                pass
            line += _handle_x(x)
        trn.write(line)
        trn.write("(")
        trn.write(utt_id)
        trn.write(")\n")


def read_ctm(
    ctm: Union[TextIO, str], wc2utt: Optional[dict] = None
) -> List[Tuple[str, List[Tuple[str, float, float]]]]:
    """Read a NIST sclite "ctm" file into a list of transcriptions

    `sclite <http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm>`__ is a
    commonly used scoring tool for ASR.

    This function converts a time-marked conversation file ("ctm" format) into a list of
    `transcripts`. Each element is a tuple of ``utt_id, transcript``, where
    ``transcript`` is itself a list of triples ``token, start, end``, ``token`` being a
    string, ``start`` being the start time of the token (in seconds), and ``end`` being
    the end time of the token (in seconds)

    Parameters
    ----------
    ctm
        The time-marked conversation file pointer. Will open if `ctm` is a
        path
    wc2utt
        "ctm" files identify utterances by waveform file name and channel. If
        specified, `wc2utt` consists of keys ``wfn, chan`` (e.g.
        ``'940328', 'A'``) to unique utterance IDs. If `wc2utt` is
        unspecified, the waveform file names are treated as the utterance IDs,
        and the channel is ignored

    Returns
    -------
    transcripts : list
        Each element is a tuple of ``utt_id, transcript``. `utt_id` is a string
        identifying the utterance. `transcript` is a list of triples ``token, start,
        end``, `token` being the token (a string), `start` being a float of the start
        time of the token (in seconds), and `end` being the end time of the token.

    Notes
    -----
    "ctm", like "trn", has "support" for alternate transcriptions. It is, as of sclite
    version 2.10, very buggy. For example, it cannot handle multiple alternates in the
    same utterance. Plus, tools like `Kaldi <http://kaldi-asr.org/>`__ use the Unix
    command that the sclite documentation recommends to sort a ctm, ``sort +0 -1 +1 -2
    +2nb -3``, which does not maintain proper ordering for alternate delimiters. Thus,
    :func:`read_ctm` will error if it comes across those delimiters
    """
    if isinstance(ctm, str):
        with open(ctm, "r") as ctm:
            return read_ctm(ctm, wc2utt)
    transcripts = OrderedDict()
    for line_no, line in enumerate(ctm):
        line = line.split(";;")[0].strip()
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
            if start < 0.0 or start > end:
                raise ValueError()
            transcripts.setdefault(utt_id, []).append((token, start, end))
        except ValueError:
            raise ValueError("Could not parse line {} of ctm".format(line_no + 1))
        except KeyError:
            raise KeyError(
                "ctm line {}: ({}, {}) was not found in wc2utt".format(
                    line_no, wfn, chan
                )
            )
    return [
        (utt_id, sorted(transcript, key=lambda x: x[1]))
        for utt_id, transcript in list(transcripts.items())
    ]


def write_ctm(
    transcripts: Sequence[Tuple[str, Sequence[Tuple[str, float, float]]]],
    ctm: Union[TextIO, str],
    utt2wc: Union[Mapping[str, Tuple[str, str]], str] = config.DEFT_CTM_CHANNEL,
) -> None:
    f"""From a list of transcripts, write to a NIST "ctm" file

    This is the inverse operation of :func:`read_ctm`. For each element of
    ``transcript`` within the ``utt_id, transcript`` pairs of elements in `transcripts`,
    ``token, start, end``, ``start`` and ``end`` must be non-negative

    Parameters
    ----------
    transcripts
    ctm
    utt2wc
        "ctm" files identify utterances by waveform file name and channel. If specified
        as a dict, `utt2wc` consists of utterance IDs as keys, and wavefile name and
        channels as values ``wfn, chan`` (e.g. ``'940328',
        '{config.DEFT_CTM_CHANNEL}'``). If `utt2wc` is a string, each utterance IDs will
        be mapped to ``wfn`` and `utt2wc` as the channel.
    """
    if isinstance(ctm, str):
        with open(ctm, "w") as ctm:
            return write_ctm(transcripts, ctm, utt2wc)
    is_dict = not isinstance(utt2wc, str)
    segments = []
    for utt_id, transcript in transcripts:
        try:
            wfn, chan = utt2wc[utt_id] if is_dict else (utt_id, utt2wc)
        except KeyError:
            raise KeyError('Utt "{}" has no value in utt2wc'.format(utt_id))
        for tup in transcript:
            if isinstance(tup, str) or len(tup) != 3 or tup[1] < 0.0 or tup[2] < 0.0:
                raise ValueError(
                    'Utt "{}" contains token "{}" with no timing info'
                    "".format(utt_id, tup)
                )
            token, start, end = tup
            duration = end - start
            if duration < 0.0:
                raise ValueError(
                    'Utt "{}" contains token with negative duration' "".format(utt_id)
                )
            segments.append((wfn, chan, start, duration, token))
    segments = sorted(segments)
    for segment in segments:
        ctm.write("{} {} {} {} {}\n".format(*segment))


def read_textgrid(
    tg: Union[TextIO, str],
    tier_id: Union[str, int] = config.DEFT_TEXTGRID_TIER_ID,
    fill_token: Optional[str] = None,
) -> Tuple[List[Tuple[str, float, float]], float, float]:
    """Read TextGrid file as a transcription
    
    TextGrid is the transcription format of `Praat
    <https://www.fon.hum.uva.nl/praat/>`__.

    Parameters
    ----------
    tg 
        The TextGrid file. Will open if `tg` is a path.
    tier_id 
        Either the name of the tier (first occurence) or the index of the tier to
        extract.
    fill_token
        If :obj:`True`, any intervals missing from the tier will be filled with an
        interval of this token before being returned.

    Returns
    -------
    transcript : list
        A list of triples of ``token, start, end``, token` being the token (a string),
        `start` being a float of the start time of the token (in seconds), and `end`
        being the end time of the token. If the tier is a PointTier, `the start and
        end times will be the same.
    start_time : float
        The start time of the tier (in seconds)
    end_time : float
        The end time of the tier (in seconds)

    Notes
    -----
    This function does not check for whitespace in or around token labels. This may
    cause issues if writing as another file type, like :func:`write_trn`.

    Start and end times (including any filled intervals) are determined from the tier's
    values, not necessarily those of the top-level container. This is most likely a
    technicality, however: they should not differ normally.
    """

    if isinstance(tg, str):
        with open(tg) as f:
            return read_textgrid(f, tier_id, fill_token)

    tg_ = TextGrid(tg.read())
    if isinstance(tier_id, str):
        tier = None
        for tier_ in tg_.tiers:
            if tier_.nameid == tier_id:
                tier = tier_
                break
        if tier is None:
            raise ValueError(f"Could not find tier '{tier_id}'")
    else:
        tier = tg_.tiers[tier_id]

    if tier.classid == TEXTTIER:
        transcript = [
            (x[1], float(x[0]), float(x[0])) for x in sorted(tier.simple_transcript)
        ]
    else:
        transcript = [
            (x[2], float(x[0]), float(x[1])) for x in sorted(tier.simple_transcript)
        ]
    i = 0
    start_time = tier.xmin
    while i < len(transcript):
        _, next_start, end_time = transcript[i]
        if fill_token is not None and start_time < next_start:
            transcript.insert(i, (fill_token, start_time, next_start))
            i += 1
        i += 1
        start_time = end_time
    if fill_token is not None and tier.xmax is not None and start_time < tier.xmax:
        transcript.append((fill_token, start_time, tier.xmax))
    return transcript, tier.xmin, tier.xmax


def write_textgrid(
    transcript: Sequence[Tuple[str, float, float]],
    tg: Union[TextIO, str],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    tier_name: str = config.DEFT_TEXTGRID_TIER_NAME,
    point_tier: Optional[bool] = None,
    precision: int = config.DEFT_FLOAT_PRINT_PRECISION,
) -> None:
    """Write a transcription as a TextGrid file
    
    TextGrid is the transcription format of `Praat
    <https://www.fon.hum.uva.nl/praat/>`__.

    This function saves `transcript` as a tier within a TextGrid file.

    Parameters
    ----------
    transcript
        The transcription to write. Contains triples ``tok, start, end``, where `tok` is
        the token, `start` is its start time, and `end` is its end time. `transcript`
        must be non-empty.
    tg
        The file to write. Will open if `tg` is a path.
    start_time
        The start time of the recording (in seconds). If not specified, it will be
        inferred from the minimum start time of the intervals in `transcript`.
    end_time
        The end time of the recording (in seconds). If not specified, it will be
        inferred from the maximum end time of the intervals in `transcript`.
    tier_name
        What name to save the tier with.
    point_tier
        Whether to save as a point tier (:obj:`True`) or an interval tier. If unset, the
        value is inferred to be a point tier if all segments are length 0 (within
        precision `precision`); an interval tier otherwise.
    precision
        The precision of floating-point values to save times with.
    """
    if isinstance(tg, str):
        with open(tg, "w") as tg:
            return write_textgrid(transcript, tg, start_time, end_time, tier_name)
    transcript = list(transcript)
    if not len(transcript):
        raise ValueError(f"Will not write an empty transcript")
    tier_start_time = min(x[1] for x in transcript)
    tier_end_time = max(x[2] for x in transcript)
    if start_time is None:
        start_time = tier_start_time
    elif start_time > tier_start_time:
        raise ValueError(
            f"gave start_time={start_time} but an interval starts at "
            f"{tier_start_time}"
        )
    if end_time is None:
        end_time = tier_end_time
    elif end_time < tier_end_time:
        raise ValueError(
            f"gave end_time={end_time} but an interval ends at {tier_end_time}"
        )
    if point_tier is None:
        point_tier = all(
            f"{x[1]:0.{precision}f}" == f"{x[2]:0.{precision}f}" for x in transcript
        )
    tier_type = "TextTier" if point_tier else "IntervalTier"
    # fmt: off
    tg.write(
        'File type = "ooTextFile"\n'
        'Object class = "TextGrid"\n'
        f"{start_time:0.{precision}f}\n"
        f"{end_time:0.{precision}f}\n"
        "<exists>\n"
        "1\n"
        f'"{tier_type}"\n'
        f'"{tier_name}"\n'
        f"{tier_start_time:0.{precision}f}\n"
        f"{tier_end_time:0.{precision}f}\n"
        f"{len(transcript)}\n"
    )
    # fmt: on
    for tok, start, end in transcript:
        if point_tier:
            tg.write(f'{start:0.{precision}f}\n"{tok}"\n')
        else:
            tg.write(f'{start:0.{precision}f}\n{end:0.{precision}f}\n"{tok}"\n')


def transcript_to_token(
    transcript: Sequence[Union[str, Tuple[str, float, float]]],
    token2id: Optional[dict] = None,
    frame_shift_ms: Optional[float] = None,
    unk: Optional[Union[str, int]] = None,
    skip_frame_times: bool = False,
) -> torch.Tensor:
    r"""Convert a transcript to a token sequence

    This function converts `transcript` of length ``R`` to a long tensor `tok` of shape
    ``(R, 3)``, the latter suitable as a reference or hypothesis token sequence for an
    utterance of :class:`SpectDataSet`. An element of `transcript` can either be a
    ``token`` or a 3-tuple of ``(token, start, end)``. If `token2id` is not :obj:`None`,
    the token id is determined by checking ``token2id[token]``. If the token does not
    exist in `token2id` and `unk` is not :obj:`None`, the token will be replaced with
    `unk`. If `unk` is :obj:`None`, `token` will be used directly as the id. If
    `token2id` is not specified, `token` will be used directly as the identifier. If
    `frame_shift_ms` is specified, ``start`` and ``end`` are taken as the start and end
    times, in seconds, of the token, and will be converted to frames for `tok`. If
    `frame_shift_ms` is unspecified, ``start`` and ``end`` are assumed to already be
    frame times. If ``start`` and ``end`` were unspecified, values of ``-1``,
    representing unknown, will be inserted into ``tok[r, 1:]``

    Parameters
    ----------
    transcript
    token2id
    frame_shift_ms
    unk
        The out-of-vocabulary token, if specified. If `unk` exists in `token2id`, the
        ``token2id[unk]`` will be used as the out-of-vocabulary identifier. If
        ``token2id[unk]`` does not exist, `unk` will be assumed to be the identifier
        already. If `token2id` is :obj:`None`, `unk` has no effect.
    skip_frame_times
        If :obj:`True`, `tok` will be of shape ``(R,)`` and contain only the token ids.
        Suitable for :class:`BitextDataSet`.

    Returns
    -------
    tok : torch.Tensor

    Warnings
    --------
    The frame index bounds inferred using `frame_shift_ms` should not be used directly
    in evaluation. See the below note.

    Notes
    -----
    If you are dealing with raw audio, each "frame" is just a sample. The appropriate
    value for `frame_shift_ms` is ``1000 / sample_rate_hz`` (since there are
    ``sample_rate_hz / 1000`` samples per millisecond).

    Converting to frame indices from start and end times follows an overly-simplistic
    equation. Letting :math:`(s_s, e_s)` be the start and end times in seconds,
    :math:`(s_f, e_f)` be the corresponding start and end frames, :math:`\Delta` be the
    frame shift in milliseconds, and :math:`I[\cdot]` be the indicator function. Then

    .. math::

        s_f = floor\left(\frac{1000s_s}{\Delta}\right) \\
        e_f = \max\left(s_s + I[s_s = e_s],
                        round\left(\frac{1000e_s}{\Delta}\right)\right)

    For a given token index, ``tok[r, 1] = s_f`` and ``tok[r, 2] = e_f``. ``tok[r, 1]``
    is supposed to be the inclusive start frame of the segment and ``tok[r, 2]`` the
    exclusive end frame. :math:`(s_f, e_f)` fail to be these on two accounts. First,
    they do not consider the frame length. First, while frames may be spaced
    :math:`\Delta` milliseconds apart, they will usually be overlapping. Because of this
    overlap, the coefficients of frames :math:`s_f - 1` and :math:`e_f` may be in part
    dependent on the audio samples within the segment. Second, ignoring frame length,
    :math:`e_f = ceil(1000e_s/\Delta)` would be more appropriate for an exclusive upper
    bound. However, :mod:`pydrobert.speech.compute` (and other, mainstream feature
    computation packages), the total number of frames in the utterance is calculated as
    :math:`T_f = ceil(1000T_s/\Delta)`, where :math:`T_s` is the length of the utterance
    in seconds. The above equation ensures :math:`\max(e_f) \leq T_f`, which is a
    neccessary criterion for a valid :class:`SpectDataSet` (see
    :func:`validate_spec_data_set`).

    Accounting for both of these assumptions would involve computing the support of each
    existing frame in seconds and intersecting that with the provided interval in
    seconds. As such, the derived frame bounds should not be used for an official
    evaluation. This function should suffice for most training objectives, however.
    """
    if token2id is not None and unk in token2id:
        unk = token2id[unk]
    tok_size = (len(transcript),)
    if not skip_frame_times:
        tok_size = tok_size + (3,)
    tok = torch.empty(tok_size, dtype=torch.long)
    for i, token in enumerate(transcript):
        start = end = -1
        try:
            if len(token) == 3 and np.isreal(token[1]) and np.isreal(token[2]):
                token, start, end = token
                if frame_shift_ms:
                    if start == end:
                        start = end = (1000 * start) // frame_shift_ms
                    else:
                        start = (1000 * start) // frame_shift_ms
                        end = (1000 * end + 0.5 * frame_shift_ms) // frame_shift_ms
                        end = max(end, start + 1)
                else:
                    start, end = int(start), int(end)
        except TypeError:
            pass
        if token2id is None:
            id_ = token
        else:
            id_ = token2id.get(token, token if unk is None else unk)
        if skip_frame_times:
            tok[i] = id_
        else:
            tok[i, 0] = id_
            tok[i, 1] = start
            tok[i, 2] = end
    return tok


def token_to_transcript(
    ref: torch.Tensor,
    id2token: Optional[Dict[int, str]] = None,
    frame_shift_ms: Optional[float] = None,
) -> List[Union[str, int, Tuple[Union[str, int], float, float]]]:
    """Convert a token sequence to a transcript

    The inverse operation of :func:`transcript_to_token`.

    Parameters
    ----------
    ref
        A long tensor either of shape ``(R, 3)`` with segmentation info or ``(R, 1)`` or
        ``(R,)`` without
    id2token
    frame_shift_ms

    Returns
    -------
    transcript

    Warnings
    --------
    The time interval inferred using `frame_shift_ms` is unlikely to be perfectly
    correct. See the note in :func:`transcript_to_token` for more details about the
    ambiguity in converting between seconds and frames.
    """
    transcript = []
    for tup in ref:
        start = end = -1
        if tup.ndim:
            id_ = tup[0].item()
            if tup.numel() == 3:
                start = tup[1].item()
                end = tup[2].item()
        else:
            id_ = tup.item()
        token = id2token.get(id_, id_) if id2token is not None else id_
        if start == -1 or end == -1:
            transcript.append(token)
        else:
            if frame_shift_ms:
                start = start * frame_shift_ms / 1000
                end = end * frame_shift_ms / 1000
            transcript.append((token, start, end))
    return transcript
