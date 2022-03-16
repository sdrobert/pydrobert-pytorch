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

from io import StringIO
from tempfile import SpooledTemporaryFile

import pytest
import torch

import pydrobert.torch.data as data


@pytest.mark.cpu
def test_parse_arpa_lm():
    file_ = SpooledTemporaryFile(mode="w+")
    file_.write(
        r"""\
This is from https://cmusphinx.github.io/wiki/arpaformat/
I've removed the backoff for </s> b/c IRSTLM likes to do things like that

\data\
ngram 1=7
ngram 2=7

\1-grams:
-1.0000 <unk>	-0.2553
-98.9366 <s>	 -0.3064
-1.0000 </s>
-0.6990 wood	 -0.2553
-0.6990 cindy	-0.2553
-0.6990 pittsburgh		-0.2553
-0.6990 jean	 -0.1973

\2-grams:
-0.2553 <unk> wood
-0.2553 <s> <unk>
-0.2553 wood pittsburgh
-0.2553 cindy jean
-0.2553 pittsburgh cindy
-0.5563 jean </s>
-0.5563 jean wood

\end\
"""
    )
    file_.seek(0)
    ngram_list = data.parse_arpa_lm(file_)
    assert len(ngram_list) == 2
    assert set(ngram_list[0]) == {
        "<unk>",
        "<s>",
        "</s>",
        "wood",
        "cindy",
        "pittsburgh",
        "jean",
    }
    assert set(ngram_list[1]) == {
        ("<unk>", "wood"),
        ("<s>", "<unk>"),
        ("wood", "pittsburgh"),
        ("cindy", "jean"),
        ("pittsburgh", "cindy"),
        ("jean", "</s>"),
        ("jean", "wood"),
    }
    assert abs(ngram_list[0]["cindy"][0] + 0.6990) < 1e-4
    assert abs(ngram_list[0]["jean"][1] + 0.1973) < 1e-4
    assert abs(ngram_list[1][("cindy", "jean")] + 0.2553) < 1e-4
    file_.seek(0)
    token2id = dict((c, hash(c)) for c in ngram_list[0])
    ngram_list = data.parse_arpa_lm(file_, token2id=token2id)
    assert set(ngram_list[0]) == set(token2id.values())
    file_.seek(0)
    file_.write(
        r"""\
Here's one where we skip right to 10-grams

\data\
ngram 10 = 1

\10-grams:
0.0 1 2 3 4 5 6 7 8 9 10

\end\
"""
    )
    file_.seek(0)
    ngram_list = data.parse_arpa_lm(file_)
    assert all(x == dict() for x in ngram_list[:-1])
    assert not ngram_list[9][tuple(str(x) for x in range(1, 11))]
    file_.seek(0)
    file_.write(
        r"""\
Here's one where we erroneously include backoffs

\data\
ngram 1 = 1

\1-grams:
0.0 a 0.0

\end\
"""
    )
    file_.seek(0)
    with pytest.raises(IOError):
        data.parse_arpa_lm(file_)
    file_.seek(0)
    file_.write(
        r"""\
Here's an empty one

\data\
\end\
"""
    )
    file_.seek(0)
    assert data.parse_arpa_lm(file_) == []


@pytest.mark.cpu
@pytest.mark.parametrize("processes", [0, 2])
def test_read_trn(processes):
    trn = StringIO()
    trn.write(
        """\
here is a simple example (a)
nothing should go wrong (b)
"""
    )
    trn.seek(0)
    act = data.read_trn(trn, processes=processes, chunk_size=1)
    assert act == [
        ("a", ["here", "is", "a", "simple", "example"]),
        ("b", ["nothing", "should", "go", "wrong"]),
    ]
    trn.seek(0)
    trn.write(
        """\
here is an { example /with} some alternates (a)
} and /here/ is {something really / {really}} (stupid) { ignore this (b)
(c)
a11 (d)
"""
    )
    trn.seek(0)
    act = data.read_trn(trn, warn=False, processes=processes)
    assert act == [
        (
            "a",
            [
                "here",
                "is",
                "an",
                ([["example"], ["with"]], -1, -1),
                "some",
                "alternates",
            ],
        ),
        (
            "b",
            [
                "}",
                "and",
                "/here/",
                "is",
                ([["something", "really"], [[["really"]]]], -1, -1),
                "(stupid)",
            ],
        ),
        ("c", []),
        ("d", ["a11"]),
    ]


@pytest.mark.cpu
def test_read_ctm():
    ctm = StringIO()
    ctm.write(
        """\
utt1 A 0.0 0.1 a
utt1 A 0.5 0.1 c  ;; ctm files should always be ordered, but we tolerate
                  ;; different orders
utt2 B 0.1 1.0 d
utt1 B 0.4 0.3 b
;; utt2 A 0.2 1.0 f
"""
    )
    ctm.seek(0)
    act = data.read_ctm(ctm)
    assert act == [
        ("utt1", [("a", 0.0, 0.1), ("b", 0.4, 0.7), ("c", 0.5, 0.6)]),
        ("utt2", [("d", 0.1, 1.1)]),
    ]
    ctm.seek(0)
    act = data.read_ctm(
        ctm, {("utt1", "A"): "foo", ("utt1", "B"): "bar", ("utt2", "B"): "baz"}
    )
    assert act == [
        ("foo", [("a", 0.0, 0.1), ("c", 0.5, 0.6)]),
        ("baz", [("d", 0.1, 1.1)]),
        ("bar", [("b", 0.4, 0.7)]),
    ]
    with pytest.raises(ValueError):
        ctm.write("utt3 -0.1 1.0 woop\n")
        ctm.seek(0)
        data.read_ctm(ctm)


@pytest.mark.cpu
def test_write_trn():
    trn = StringIO()
    transcripts = [
        ("a", ["again", "a", "simple", "example"]),
        ("b", ["should", "get", "right", "no", "prob"]),
    ]
    data.write_trn(transcripts, trn)
    trn.seek(0)
    assert (
        """\
again a simple example (a)
should get right no prob (b)
"""
        == trn.read()
    )
    trn.seek(0)
    trn.truncate()
    transcripts = [
        (
            " c ",
            [
                ("unnecessary", -1, -1),
                ([["complexity", [["can"]]], ["also", "be"]], 10, 4),
                "handled",
            ],
        ),
        ("d", []),
        ("e", ["a11"]),
    ]
    data.write_trn(transcripts, trn)
    trn.seek(0)
    assert (
        """\
unnecessary { complexity { can } / also be } handled ( c )
(d)
a11 (e)
"""
        == trn.read()
    )


@pytest.mark.cpu
def test_write_ctm():
    ctm = StringIO()
    transcripts = [
        (
            "c",
            [
                ("here", 0.1, 0.2),
                ("are", 0.3, 0.5),
                ("some", 0.2, 0.4),
                ("unordered", 0.5, 0.5),
                ("tokens", 10.0, 1000),
            ],
        ),
        ("b", []),
        ("a", [("hullo", 0.0, 10.0111)]),
    ]
    data.write_ctm(transcripts, ctm)
    ctm.seek(0)
    assert (
        """\
a A 0.0 10.0111 hullo
c A 0.1 0.1 here
c A 0.2 0.2 some
c A 0.3 0.2 are
c A 0.5 0.0 unordered
c A 10.0 990.0 tokens
"""
        == ctm.read()
    )
    ctm.seek(0)
    ctm.truncate()
    data.write_ctm(
        transcripts,
        ctm,
        {"a": ("last", "A"), "b": ("middle", "B"), "c": ("first", "C")},
    )
    ctm.seek(0)
    assert (
        """\
first C 0.1 0.1 here
first C 0.2 0.2 some
first C 0.3 0.2 are
first C 0.5 0.0 unordered
first C 10.0 990.0 tokens
last A 0.0 10.0111 hullo
"""
        == ctm.read()
    )
    transcripts.append(("foo", [("a", 0.1, 0.2), ("b", 0.2, 0.1)]))
    with pytest.raises(ValueError):
        data.write_ctm(transcripts, ctm)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "transcript,token2id,unk,skip_frame_times,exp",
    [
        ([], None, None, False, torch.LongTensor(0, 3)),
        ([1, 2, 3, 4], None, None, True, torch.LongTensor([1, 2, 3, 4]),),
        (
            [1, ("a", 4, 10), "a", 3],
            {"a": 2},
            None,
            False,
            torch.LongTensor([[1, -1, -1], [2, 4, 10], [2, -1, -1], [3, -1, -1]]),
        ),
        (
            ["foo", 1, "bar"],
            {"foo": 0, "baz": 3},
            "baz",
            False,
            torch.LongTensor([[0, -1, -1], [3, -1, -1], [3, -1, -1]]),
        ),
    ],
)
def test_transcript_to_token(transcript, token2id, unk, skip_frame_times, exp):
    act = data.transcript_to_token(
        transcript, token2id, unk=unk, skip_frame_times=skip_frame_times
    )
    assert torch.all(exp == act)
    transcript = ["foo"] + transcript
    with pytest.raises(Exception):
        data.transcript_to_token(transcript, token2id)


@pytest.mark.cpu
def test_transcript_to_token_frame_shift():
    trans = [(12, 0.5, 0.81), 420, (1, 2.1, 2.2), (3, 2.8, 2.815), (12, 2.9, 3.0025)]
    # normal case: frame shift 10ms. Frame happens every hundredth of a second,
    # so multiply by 100. Half-frames should round up; quarter-frames down
    tok = data.transcript_to_token(trans, frame_shift_ms=10)
    assert torch.allclose(
        tok,
        torch.LongTensor(
            [[12, 50, 81], [420, -1, -1], [1, 210, 220], [3, 280, 282], [12, 290, 300]]
        ),
    )
    # raw case @ 8000Hz sample rate. "Frame" is every sample. frames/msec =
    # 1000 / sample_rate_hz = 1 / 8.
    tok = data.transcript_to_token(trans, frame_shift_ms=1 / 8)
    assert torch.allclose(
        tok,
        torch.LongTensor(
            [
                [12, 4000, 6480],
                [420, -1, -1],
                [1, 16800, 17600],
                [3, 22400, 22520],
                [12, 23200, 24020],
            ]
        ),
    )


@pytest.mark.cpu
@pytest.mark.parametrize(
    "tok,id2token,exp",
    [
        (torch.LongTensor(0, 3), None, []),
        (
            torch.LongTensor([[1, -1, -1], [2, -1, -1], [3, -1, -1], [4, -1, -1]]),
            None,
            [1, 2, 3, 4],
        ),
        (
            torch.LongTensor([[1, 3, 4], [3, 4, 5], [2, -1, -1]]),
            {1: "a", 2: "b"},
            [("a", 3, 4), (3, 4, 5), "b"],
        ),
        (torch.tensor(range(10)), None, list(range(10))),
        (torch.tensor(range(5)).unsqueeze(-1), None, list(range(5))),
    ],
)
def test_token_to_transcript(tok, id2token, exp):
    act = data.token_to_transcript(tok, id2token)
    assert exp == act


@pytest.mark.cpu
def test_token_to_transcript_frame_shift():
    tok = torch.LongTensor([[1, -1, 10], [2, 1000, 2000], [3, 12345, 678910]])
    # standard case: 10ms frame shift
    # 10ms per frame means divide frame number by 100
    trans = data.token_to_transcript(tok, frame_shift_ms=10)
    assert trans == [1, (2, 10.0, 20.0), (3, 123.45, 6789.10)]
    # raw case: 8000 samples / sec = 8 samples / msec so frame shift is 1 / 8
    trans = data.token_to_transcript(tok, frame_shift_ms=1 / 8)
    assert trans == [
        1,
        (2, 1000 / 8000, 2000 / 8000),
        (3, 12345 / 8000, 678910 / 8000),
    ]
