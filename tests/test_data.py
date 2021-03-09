# Copyright 2021 Sean Robertson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from itertools import repeat
from io import StringIO


import pytest
import torch
import torch.utils.data
import pydrobert.torch.data as data

from pydrobert.torch import INDEX_PAD_VALUE


@pytest.mark.cpu
@pytest.mark.parametrize("left", [0, 1, 100])
@pytest.mark.parametrize("right", [0, 1, 100])
@pytest.mark.parametrize("T", [1, 5, 10])
def test_extract_window(left, right, T):
    signal = torch.arange(T).view(-1, 1).expand(-1, 10)
    for frame_idx in range(T):
        window = data.extract_window(signal, frame_idx, left, right)
        left_pad = max(left - frame_idx, 0)
        right_pad = max(frame_idx + right + 1 - T, 0)
        assert tuple(window.shape) == (1 + left + right, 10)
        if left_pad:
            assert torch.all(window[:left_pad] == torch.tensor([0]))
        if right_pad:
            assert torch.all(window[-right_pad:] == torch.tensor([T - 1]))
        assert torch.all(
            window[left_pad : 1 + left + right - right_pad]
            == torch.arange(
                frame_idx - left + left_pad, frame_idx + right - right_pad + 1
            )
            .view(-1, 1)
            .expand(-1, 10)
        )


@pytest.mark.cpu
@pytest.mark.parametrize("num_utts", [1, 2, 10])
@pytest.mark.parametrize("file_prefix", ["prefix_", ""])
@pytest.mark.parametrize("eos", [1000, None])
@pytest.mark.parametrize("sos", [2000, None])
@pytest.mark.parametrize("feat_dtype", [torch.float, torch.int])
def test_valid_spect_data_set(
    temp_dir, num_utts, file_prefix, populate_torch_dir, sos, eos, feat_dtype
):
    feats, _, _, _, _, utt_ids = populate_torch_dir(
        temp_dir,
        num_utts,
        file_prefix=file_prefix,
        include_ali=False,
        include_ref=False,
        feat_dtype=feat_dtype,
    )
    # note that this'll just resave the same features if there's no file
    # prefix. If there is, these ought to be ignored by the data set
    populate_torch_dir(
        temp_dir, num_utts, include_ali=False, include_ref=False, feat_dtype=feat_dtype
    )
    if not os.path.isdir(os.path.join(temp_dir, "feat", "fake")):
        os.makedirs(os.path.join(temp_dir, "feat", "fake"))
    torch.save(
        torch.randint(100, (10, 5), dtype=feat_dtype),
        os.path.join(temp_dir, "feat", "fake", file_prefix + "fake.pt"),
    )
    data_set = data.SpectDataSet(temp_dir, file_prefix=file_prefix, eos=eos)
    assert not data_set.has_ali and not data_set.has_ref
    assert len(utt_ids) == len(data_set.utt_ids)
    assert all(utt_a == utt_b for (utt_a, utt_b) in zip(utt_ids, data_set.utt_ids))
    assert all(
        ali_b is None and ref_b is None and torch.allclose(feat_a, feat_b)
        for (feat_a, (feat_b, ali_b, ref_b)) in zip(feats, data_set)
    )
    feats, alis, refs, _, _, utt_ids = populate_torch_dir(
        temp_dir, num_utts, file_prefix=file_prefix, feat_dtype=feat_dtype
    )
    if sos is not None:
        sos_sym = torch.full((3,), -1, dtype=torch.long)
        sos_sym[0] = sos
        sos_sym = sos_sym.unsqueeze(0)
        refs = [torch.cat([sos_sym, x]) for x in refs]
    if eos is not None:
        eos_sym = torch.full((3,), -1, dtype=torch.long)
        eos_sym[0] = eos
        eos_sym = eos_sym.unsqueeze(0)
        refs = [torch.cat([x, eos_sym]) for x in refs]
    data_set = data.SpectDataSet(temp_dir, file_prefix=file_prefix, sos=sos, eos=eos)
    assert data_set.has_ali and data_set.has_ref
    assert len(utt_ids) == len(data_set.utt_ids)
    assert all(utt_a == utt_b for (utt_a, utt_b) in zip(utt_ids, data_set.utt_ids))
    assert all(
        torch.all(ali_a == ali_b)
        and torch.all(ref_a == ref_b)
        and feat_a.dtype == feat_b.dtype
        and torch.allclose(feat_a, feat_b)
        for ((feat_a, ali_a, ref_a), (feat_b, ali_b, ref_b)) in zip(
            zip(feats, alis, refs), data_set
        )
    )
    subset_ids = data_set.utt_ids[: num_utts // 2]
    data_set = data.SpectDataSet(
        temp_dir, file_prefix=file_prefix, subset_ids=set(subset_ids), sos=sos, eos=eos
    )
    assert all(utt_a == utt_b for (utt_a, utt_b) in zip(subset_ids, data_set.utt_ids))
    assert all(
        torch.all(ali_a == ali_b)
        and torch.all(ref_a == ref_b)
        and torch.allclose(feat_a, feat_b)
        for ((feat_a, ali_a, ref_a), (feat_b, ali_b, ref_b)) in zip(
            zip(feats[: num_utts // 2], alis[: num_utts // 2], refs[: num_utts // 2]),
            data_set,
        )
    )


@pytest.mark.cpu
def test_spect_data_set_warnings(temp_dir):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, "feat")
    ali_dir = os.path.join(temp_dir, "ali")
    os.makedirs(feat_dir)
    os.makedirs(ali_dir)
    torch.save(torch.rand(3, 3), os.path.join(feat_dir, "a.pt"))
    torch.save(torch.rand(4, 3), os.path.join(feat_dir, "b.pt"))
    torch.save(torch.randint(10, (4,), dtype=torch.long), os.path.join(ali_dir, "b.pt"))
    torch.save(torch.randint(10, (5,), dtype=torch.long), os.path.join(ali_dir, "c.pt"))
    data_set = data.SpectDataSet(temp_dir, warn_on_missing=False)
    assert data_set.has_ali
    assert data_set.utt_ids == ("b",)
    with pytest.warns(UserWarning) as warnings:
        data_set = data.SpectDataSet(temp_dir)
    assert len(warnings) == 2
    assert any(str(x.message) == "Missing ali for uttid: 'a'" for x in warnings)
    assert any(str(x.message) == "Missing feat for uttid: 'c'" for x in warnings)


def test_spect_data_write_pdf(temp_dir, device):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, "feat")
    os.makedirs(feat_dir)
    torch.save(torch.rand(3, 3), os.path.join(feat_dir, "a.pt"))
    data_set = data.SpectDataSet(temp_dir)
    z = torch.randint(10, (4, 5), dtype=torch.long)
    if device == "cuda":
        data_set.write_pdf("b", z.cuda())
    else:
        data_set.write_pdf("b", z)
    zp = torch.load(os.path.join(temp_dir, "pdfs", "b.pt"))
    assert isinstance(zp, torch.FloatTensor)
    assert torch.allclose(zp, z.float())
    data_set.write_pdf(0, torch.rand(10, 4))
    assert os.path.exists(os.path.join(temp_dir, "pdfs", "a.pt"))
    data_set.write_pdf("c", z, pdfs_dir=os.path.join(temp_dir, "foop"))
    assert os.path.exists(os.path.join(temp_dir, "foop", "c.pt"))


@pytest.mark.parametrize("eos", [None, -1])
@pytest.mark.parametrize("sos", [None, -2])
def test_spect_data_write_hyp(temp_dir, device, sos, eos):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, "feat")
    os.makedirs(feat_dir)
    torch.save(torch.rand(3, 3), os.path.join(feat_dir, "a.pt"))
    data_set = data.SpectDataSet(temp_dir, sos=sos, eos=eos)
    z = torch.randint(10, (4, 3), dtype=torch.float)
    zz = z
    if sos:
        zz = torch.cat([torch.full_like(zz, sos), zz])
    if eos:
        zz = torch.cat([zz, torch.full_like(z, eos)])
    if device == "cuda":
        data_set.write_hyp("b", zz.cuda())
    else:
        data_set.write_hyp("b", zz)
    zp = torch.load(os.path.join(temp_dir, "hyp", "b.pt"))
    assert isinstance(zp, torch.LongTensor)
    assert torch.all(zp == z.long())
    data_set.write_hyp(0, torch.randint(10, (11, 3)))
    assert os.path.exists(os.path.join(temp_dir, "hyp", "a.pt"))
    data_set.write_hyp("c", z, hyp_dir=os.path.join(temp_dir, "foop"))
    assert os.path.exists(os.path.join(temp_dir, "foop", "c.pt"))


@pytest.mark.cpu
@pytest.mark.parametrize("eos", [None, 10000])
def test_spect_data_set_validity(temp_dir, eos):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, "feat")
    ali_dir = os.path.join(temp_dir, "ali")
    ref_dir = os.path.join(temp_dir, "ref")
    feats_a_pt = os.path.join(feat_dir, "a.pt")
    feats_b_pt = os.path.join(feat_dir, "b.pt")
    ali_a_pt = os.path.join(ali_dir, "a.pt")
    ali_b_pt = os.path.join(ali_dir, "b.pt")
    ref_a_pt = os.path.join(ref_dir, "a.pt")
    ref_b_pt = os.path.join(ref_dir, "b.pt")
    os.makedirs(feat_dir)
    os.makedirs(ali_dir)
    os.makedirs(ref_dir)
    torch.save(torch.rand(10, 4), feats_a_pt)
    torch.save(torch.rand(4, 4), feats_b_pt)
    torch.save(torch.randint(10, (10,), dtype=torch.long), ali_a_pt)
    torch.save(torch.randint(10, (4,), dtype=torch.long), ali_b_pt)
    torch.save(
        torch.cat(
            [
                torch.randint(10, (11, 1), dtype=torch.long),
                torch.full((11, 2), -1, dtype=torch.long),
            ],
            -1,
        ),
        ref_a_pt,
    )
    torch.save(torch.tensor([[0, 3, 4], [1, 1, 2]]), ref_b_pt)
    data_set = data.SpectDataSet(temp_dir, eos=eos)
    data.validate_spect_data_set(data_set)
    torch.save(torch.rand(4, 4).long(), feats_b_pt)
    with pytest.raises(ValueError, match="not the same tensor type"):
        data.validate_spect_data_set(data_set)
    torch.save(torch.rand(4,), feats_b_pt)
    with pytest.raises(ValueError, match="does not have two axes"):
        data.validate_spect_data_set(data_set)
    torch.save(torch.rand(4, 3), feats_b_pt)
    with pytest.raises(ValueError, match="has second axis size 3.*"):
        data.validate_spect_data_set(data_set)
    torch.save(torch.rand(4, 4), feats_b_pt)
    data.validate_spect_data_set(data_set)
    torch.save(torch.randint(10, (4,)).int(), ali_b_pt)
    with pytest.raises(ValueError, match="is not a long tensor"):
        data.validate_spect_data_set(data_set)
    torch.save(torch.randint(10, (4, 1), dtype=torch.long), ali_b_pt)
    with pytest.raises(ValueError, match="does not have one axis"):
        data.validate_spect_data_set(data_set)
    torch.save(torch.randint(10, (3,), dtype=torch.long), ali_b_pt)
    with pytest.raises(ValueError, match="does not have the same first"):
        data.validate_spect_data_set(data_set)
    torch.save(torch.randint(10, (4,), dtype=torch.long), ali_b_pt)
    data.validate_spect_data_set(data_set)
    torch.save(torch.tensor([[0, -1, 2], [1, 1, 2]]), ref_b_pt)
    with pytest.raises(ValueError, match="invalid boundaries"):
        data.validate_spect_data_set(data_set)
    torch.save(torch.tensor([[0, 0, 1], [1, 5, 30]]), ref_b_pt)
    with pytest.raises(ValueError, match="invalid boundaries"):
        data.validate_spect_data_set(data_set)
    torch.save(torch.tensor([1, 2, 3]), ref_b_pt)
    with pytest.raises(ValueError, match="were 2D"):
        data.validate_spect_data_set(data_set)
    torch.save(torch.tensor([10, 4, 2, 5]), ref_a_pt)
    data.validate_spect_data_set(data_set)


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
    trans = [(12, 0.5, 0.81), 420, (1, 2.1, 2.2)]
    # normal case: frame shift 10ms. Frame happens every hundredth of a second,
    # so multiply by 100
    tok = data.transcript_to_token(trans, frame_shift_ms=10)
    assert torch.allclose(
        tok, torch.LongTensor([[12, 50, 81], [420, -1, -1], [1, 210, 220]])
    )
    # raw case @ 8000Hz sample rate. "Frame" is every sample. frames/msec =
    # 1000 / sample_rate_hz = 1 / 8.
    tok = data.transcript_to_token(trans, frame_shift_ms=1 / 8)
    assert torch.allclose(
        tok, torch.LongTensor([[12, 4000, 6480], [420, -1, -1], [1, 16800, 17600]])
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


@pytest.mark.cpu
@pytest.mark.parametrize("reverse", [True, False])
def test_context_window_data_set(temp_dir, reverse):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, "feat")
    os.makedirs(feat_dir)
    a = torch.rand(2, 10)
    torch.save(a, os.path.join(feat_dir, "a.pt"))
    data_set = data.ContextWindowDataSet(temp_dir, 1, 1, reverse=reverse)
    windowed, _ = data_set[0]
    assert tuple(windowed.shape) == (2, 3, 10)
    if reverse:
        # [[a1, a0, a0], [a1, a1, a0]]
        assert torch.allclose(a[0], windowed[0, 1:])
        assert torch.allclose(a[1], windowed[0, 0])
        assert torch.allclose(a[0], windowed[1, 2])
        assert torch.allclose(a[1], windowed[1, :2])
    else:
        # [[a0, a0, a1], [a0, a1, a1]]
        assert torch.allclose(a[0], windowed[0, :2])
        assert torch.allclose(a[1], windowed[0, 2])
        assert torch.allclose(a[0], windowed[1, 0])
        assert torch.allclose(a[1], windowed[1, 1:])


@pytest.mark.cpu
def test_epoch_random_sampler(temp_dir):
    data_source = torch.utils.data.TensorDataset(torch.arange(100))
    sampler = data.EpochRandomSampler(data_source, base_seed=1)
    samples_ep0 = tuple(sampler)
    samples_ep1 = tuple(sampler)
    assert samples_ep0 != samples_ep1
    assert sorted(samples_ep0) == list(range(100))
    assert sorted(samples_ep1) == list(range(100))
    assert samples_ep0 == tuple(sampler.get_samples_for_epoch(0))
    assert samples_ep1 == tuple(sampler.get_samples_for_epoch(1))
    sampler = data.EpochRandomSampler(data_source, init_epoch=10, base_seed=1)
    assert samples_ep0 == tuple(sampler.get_samples_for_epoch(0))
    assert samples_ep1 == tuple(sampler.get_samples_for_epoch(1))
    # should be reproducible if we set torch manual seed
    torch.manual_seed(5)
    sampler = data.EpochRandomSampler(data_source)
    samples_ep0 = tuple(sampler)
    torch.manual_seed(5)
    sampler = data.EpochRandomSampler(data_source)
    assert samples_ep0 == tuple(sampler)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "feat_sizes",
    [((3, 5, 4), (4, 5, 4), (1, 5, 4)), ((2, 10, 5),) * 10],
    ids=["short", "long"],
)
@pytest.mark.parametrize("include_ali", [True, False])
def test_context_window_seq_to_batch(feat_sizes, include_ali):
    torch.manual_seed(1)
    feats = tuple(torch.rand(*x) for x in feat_sizes)
    if include_ali:
        alis = tuple(torch.randint(10, (x[0],), dtype=torch.long) for x in feat_sizes)
    else:
        alis = repeat(None)
    seq = zip(feats, alis)
    batch_feats, batch_ali = data.context_window_seq_to_batch(seq)
    assert torch.allclose(torch.cat(feats), batch_feats)
    if include_ali:
        assert torch.all(torch.cat(alis) == batch_ali)
    else:
        assert batch_ali is None


@pytest.mark.cpu
@pytest.mark.parametrize("include_ali", [True, False])
@pytest.mark.parametrize(
    "include_ref,include_frame_shift", [(True, True), (True, False), (False, None)]
)
@pytest.mark.parametrize("batch_first", [True, False])
def test_spect_seq_to_batch(include_ali, include_ref, batch_first, include_frame_shift):
    torch.manual_seed(1)
    feat_sizes = tuple(
        torch.randint(1, 30, (1,)).item()
        for _ in range(torch.randint(3, 10, (1,)).item())
    )
    feats = tuple(torch.randn(x, 5) for x in feat_sizes)
    if include_ali:
        alis = tuple(torch.randint(100, (x,), dtype=torch.long) for x in feat_sizes)
    else:
        alis = repeat(None)
    if include_ref:
        ref_sizes = tuple(
            torch.randint(1, 30, (1,)).item() for _ in range(len(feat_sizes))
        )
        extra_dim = (3,) if include_frame_shift else tuple()
        refs = tuple(
            torch.randint(100, (x,) + extra_dim, dtype=torch.long) for x in ref_sizes
        )
    else:
        ref_sizes = repeat(None)
        refs = repeat(None)
    (
        batch_feats,
        batch_ali,
        batch_ref,
        batch_feat_sizes,
        batch_ref_sizes,
    ) = data.spect_seq_to_batch(zip(feats, alis, refs), batch_first=batch_first)
    feat_sizes, feats, alis, refs, ref_sizes = zip(
        *sorted(zip(feat_sizes, feats, alis, refs, ref_sizes), key=lambda x: -x[0])
    )
    assert torch.all(torch.tensor(feat_sizes) == batch_feat_sizes)
    if not batch_first:
        batch_feats = batch_feats.transpose(0, 1)
        if include_ali:
            batch_ali = batch_ali.transpose(0, 1)
        if include_ref:
            batch_ref = batch_ref.transpose(0, 1)
    assert all(
        torch.allclose(a[: b.shape[0]], b)
        and torch.allclose(a[b.shape[0] :], torch.tensor([0.0]))
        for (a, b) in zip(batch_feats, feats)
    )
    if include_ali:
        assert all(
            torch.all(a[: b.shape[0]] == b)
            and torch.all(a[b.shape[0] :] == torch.tensor([INDEX_PAD_VALUE]))
            for (a, b) in zip(batch_ali, alis)
        )
    else:
        assert batch_ali is None
    if include_ref:
        assert torch.all(torch.tensor(ref_sizes) == batch_ref_sizes)
        assert all(
            torch.all(a[: b.shape[0]] == b)
            and torch.all(a[b.shape[0] :] == torch.tensor([INDEX_PAD_VALUE]))
            for (a, b) in zip(batch_ref, refs)
        )
    else:
        assert batch_ref is None
        assert batch_ref_sizes is None


@pytest.mark.cpu
@pytest.mark.parametrize("eos", [None, -1])
@pytest.mark.parametrize("sos", [None, -2])
@pytest.mark.parametrize("split_params", [True, False])
@pytest.mark.parametrize("include_frame_shift", [True, False])
@pytest.mark.parametrize("feat_dtype", [torch.float, torch.int])
def test_spect_training_data_loader(
    temp_dir,
    populate_torch_dir,
    sos,
    eos,
    split_params,
    include_frame_shift,
    feat_dtype,
):
    torch.manual_seed(40)
    num_utts, batch_size, num_filts = 20, 5, 11
    populate_torch_dir(
        temp_dir,
        num_utts,
        num_filts=num_filts,
        include_frame_shift=include_frame_shift,
        feat_dtype=feat_dtype,
    )
    if split_params:
        params = data.DataSetParams(batch_size=batch_size)
        data_params = data.SpectDataParams(sos=sos, eos=eos)
    else:
        params = data.SpectDataSetParams(batch_size=batch_size, sos=sos, eos=eos)
        data_params = None
    # check missing either ali or ref gives None in batches
    data_loader = data.SpectTrainingDataLoader(
        temp_dir, params, data_params=data_params, ali_subdir=None, seed=2
    )
    assert next(iter(data_loader))[1] is None
    data_loader = data.SpectTrainingDataLoader(
        temp_dir, params, data_params=data_params, ref_subdir=None, seed=2
    )
    assert next(iter(data_loader))[2] is None
    assert next(iter(data_loader))[4] is None
    data_loader = data.SpectTrainingDataLoader(
        temp_dir, params, data_params=data_params, seed=2
    )

    def _get_epoch(sort):
        ep_feats, ep_ali, ep_ref = [], [], []
        ep_feat_sizes, ep_ref_sizes = [], []
        max_T = 0
        max_R = 0
        batch_first = data_loader.batch_first
        for b_feats, b_ali, b_ref, b_feat_sizes, b_ref_sizes in data_loader:
            if not batch_first:
                b_feats = b_feats.transpose(0, 1)
                b_ali = b_ali.transpose(0, 1)
                b_ref = b_ref.transpose(0, 1)
            max_T = max(max_T, b_feat_sizes[0])
            R_star = max(b_ref_sizes)
            max_R = max(max_R, R_star)
            assert b_feats.shape[0] == batch_size
            assert b_ali.shape[0] == batch_size
            assert b_ref.shape[0] == batch_size
            assert b_feats.shape[-1] == num_filts
            assert b_feats.shape[1] == b_feat_sizes[0]
            assert b_ali.shape[1] == b_feat_sizes[0]
            assert b_ref.shape[1] == R_star
            assert b_ref.dim() == (3 if include_frame_shift else 2)
            ep_feats += tuple(b_feats)
            ep_ali += tuple(b_ali)
            ep_ref += tuple(b_ref)
            ep_feat_sizes += tuple(b_feat_sizes)
            ep_ref_sizes += tuple(b_ref_sizes)
        assert len(ep_feats) == num_utts
        assert len(ep_ali) == num_utts
        for i in range(num_utts):
            ep_feats[i] = torch.nn.functional.pad(
                ep_feats[i], (0, 0, 0, max_T - ep_ali[i].shape[0])
            )
            ep_ali[i] = torch.nn.functional.pad(
                ep_ali[i], (0, max_T - ep_ali[i].shape[0]), value=INDEX_PAD_VALUE
            )
            if include_frame_shift:
                ep_ref[i] = torch.nn.functional.pad(
                    ep_ref[i],
                    (0, 0, 0, max_R - ep_ref[i].shape[0]),
                    value=INDEX_PAD_VALUE,
                )
            else:
                ep_ref[i] = torch.nn.functional.pad(
                    ep_ref[i], (0, max_R - ep_ref[i].shape[0]), value=INDEX_PAD_VALUE
                )
        if sort:
            ep_feats, ep_ali, ep_ref, ep_feat_sizes, ep_ref_sizes = zip(
                *sorted(
                    zip(ep_feats, ep_ali, ep_ref, ep_feat_sizes, ep_ref_sizes),
                    key=lambda x: (-x[3], -x[4], x[0][0, 0]),
                )
            )
        return ep_feats, ep_ali, ep_ref, ep_feat_sizes, ep_ref_sizes

    def _compare_epochs(ep_a, ep_b, same):
        a_feats, a_ali, a_ref, a_feat_sizes, a_ref_sizes = ep_a
        b_feats, b_ali, b_ref, b_feat_sizes, b_ref_sizes = ep_b
        a_feats, b_feats = torch.stack(a_feats), torch.stack(b_feats)
        a_ali, b_ali = torch.stack(a_ali), torch.stack(b_ali)
        a_ref, b_ref = torch.stack(a_ref), torch.stack(b_ref)
        if same:
            assert a_feat_sizes == b_feat_sizes
            assert a_ref_sizes == b_ref_sizes
            assert torch.allclose(a_feats, b_feats)
            assert torch.all(a_ali == b_ali)
            assert torch.all(a_ref == b_ref)
        else:
            assert a_feat_sizes != b_feat_sizes
            assert a_ref_sizes != b_ref_sizes
            assert not torch.allclose(a_feats, b_feats)
            assert torch.any(a_ali != b_ali)
            assert torch.any(a_ref != b_ref)

    ep0 = _get_epoch(False)
    ep1 = _get_epoch(False)
    _compare_epochs(ep0, ep1, False)  # could be same by fluke
    _compare_epochs(_get_epoch(True), _get_epoch(True), True)
    data_loader.epoch = 1
    _compare_epochs(ep1, _get_epoch(False), True)
    data_loader = data.SpectTrainingDataLoader(
        temp_dir, params, data_params=data_params, num_workers=4, seed=2
    )
    _compare_epochs(ep0, _get_epoch(False), True)
    _compare_epochs(ep1, _get_epoch(False), True)
    data_loader.batch_first = False
    data_loader.epoch = 0
    _compare_epochs(ep0, _get_epoch(False), True)
    _compare_epochs(ep1, _get_epoch(False), True)


@pytest.mark.cpu
@pytest.mark.parametrize("eos", [None, -1])
@pytest.mark.parametrize("sos", [None, -2])
@pytest.mark.parametrize("split_params", [True, False])
@pytest.mark.parametrize("include_frame_shift", [True, False])
@pytest.mark.parametrize("feat_dtype", [torch.float, torch.int])
def test_spect_evaluation_data_loader(
    temp_dir,
    populate_torch_dir,
    sos,
    eos,
    split_params,
    include_frame_shift,
    feat_dtype,
):
    torch.manual_seed(41)
    feat_dir = os.path.join(temp_dir, "feat")
    ali_dir = os.path.join(temp_dir, "ali")
    os.makedirs(feat_dir)
    os.makedirs(ali_dir)
    batch_size = 5
    if split_params:
        params = data.DataSetParams(batch_size=batch_size)
        data_params = data.SpectDataParams(sos=sos, eos=eos)
    else:
        params = data.SpectDataSetParams(batch_size=batch_size, sos=sos, eos=eos)
        data_params = None
    feats, ali, ref, feat_sizes, ref_sizes, utt_ids = populate_torch_dir(
        temp_dir, 20, include_frame_shift=include_frame_shift, feat_dtype=feat_dtype
    )
    if sos is not None:
        if include_frame_shift:
            sos_sym = torch.full((3,), -1, dtype=torch.long)
            sos_sym[0] = sos
            sos_sym = sos_sym.unsqueeze(0)
        else:
            sos_sym = torch.full((1,), sos, dtype=torch.long)
        ref = [torch.cat([sos_sym, x], 0) for x in ref]
        ref_sizes = [x + 1 for x in ref_sizes]
    if eos is not None:
        if include_frame_shift:
            eos_sym = torch.full((3,), eos, dtype=torch.long)
            eos_sym[0] = eos
            eos_sym = eos_sym.unsqueeze(0)
        else:
            eos_sym = torch.full((1,), eos, dtype=torch.long)
        ref = [torch.cat([x, eos_sym], 0) for x in ref]
        ref_sizes = [x + 1 for x in ref_sizes]
    # check that ali and ref can be missing
    data_loader = data.SpectEvaluationDataLoader(
        temp_dir, params, data_params=data_params, ali_subdir=None, ref_subdir=None
    )
    assert next(iter(data_loader))[1:3] == (None, None)
    assert next(iter(data_loader))[4] is None
    data_loader = data.SpectEvaluationDataLoader(
        temp_dir, params, data_params=data_params
    )

    def _compare_data_loader():
        batch_first = data_loader.batch_first
        assert len(data_loader) == 4
        cur_idx = 0
        for (
            b_feats,
            b_ali,
            b_ref,
            b_feat_sizes,
            b_ref_sizes,
            b_utt_ids,
        ) in data_loader:
            if not batch_first:
                b_feats = b_feats.transpose(0, 1)
                b_ali = b_ali.transpose(0, 1)
                b_ref = b_ref.transpose(0, 1)
            R_star = max(b_ref_sizes)
            assert tuple(b_feats.shape) == (5, b_feat_sizes[0], 5)
            assert tuple(b_ali.shape) == (5, b_feat_sizes[0])
            if include_frame_shift:
                assert tuple(b_ref.shape) == (5, R_star, 3)
            else:
                assert tuple(b_ref.shape) == (5, R_star)
            # sort the sub-section of the master list by feature size
            s_feats, s_ali, s_ref, s_feat_sizes, s_ref_sizes, s_utt_ids = zip(
                *sorted(
                    zip(
                        feats[cur_idx : cur_idx + 5],
                        ali[cur_idx : cur_idx + 5],
                        ref[cur_idx : cur_idx + 5],
                        feat_sizes[cur_idx : cur_idx + 5],
                        ref_sizes[cur_idx : cur_idx + 5],
                        utt_ids[cur_idx : cur_idx + 5],
                    ),
                    key=lambda x: -x[3],
                )
            )
            assert b_utt_ids == s_utt_ids
            assert tuple(b_feat_sizes) == s_feat_sizes
            assert tuple(b_ref_sizes) == s_ref_sizes
            for a, b in zip(b_feats, s_feats):
                assert torch.allclose(a[: b.shape[0]], b)
                assert torch.allclose(
                    a[b.shape[0] :], torch.tensor([0], dtype=feat_dtype)
                )
            for a, b in zip(b_ali, s_ali):
                assert torch.all(a[: b.shape[0]] == b)
                assert torch.all(a[b.shape[0] :] == torch.tensor([INDEX_PAD_VALUE]))
            for a, b in zip(b_ref, s_ref):
                assert torch.all(a[: b.shape[0]] == b)
                assert torch.all(a[b.shape[0] :] == torch.tensor([INDEX_PAD_VALUE]))
            cur_idx += 5

    _compare_data_loader()
    _compare_data_loader()  # order should not change
    data_loader = data.SpectEvaluationDataLoader(
        temp_dir, params, data_params=data_params, num_workers=4
    )
    _compare_data_loader()  # order should still not change
    data_loader.batch_first = False
    _compare_data_loader()


@pytest.mark.cpu
@pytest.mark.parametrize("split_params", [True, False])
def test_window_training_data_loader(temp_dir, populate_torch_dir, split_params):
    populate_torch_dir(temp_dir, 5, num_filts=2)
    seed, batch_size, context_left, context_right = 2, 5, 1, 1
    if split_params:
        params = data.DataSetParams(batch_size=batch_size, drop_last=True)
        data_params = data.ContextWindowDataParams(
            context_left=context_left, context_right=context_right
        )
    else:
        params = data.ContextWindowDataSetParams(
            context_left=context_left,
            context_right=context_right,
            batch_size=batch_size,
            drop_last=True,
        )
        data_params = None
    data_loader = data.ContextWindowTrainingDataLoader(
        temp_dir, params, data_params=data_params, seed=seed
    )
    total_windows_ep0 = 0
    for feat, ali in data_loader:
        windows = feat.shape[0]
        assert tuple(feat.shape) == (windows, 3, 2)
        assert tuple(ali.shape) == (windows,)
        total_windows_ep0 += windows
    assert total_windows_ep0 >= batch_size
    feats_ep1_a, alis_ep1_a = [], []
    total_windows_ep1 = 0
    for feats, alis in data_loader:
        windows = feat.shape[0]
        assert tuple(feat.shape) == (windows, 3, 2)
        assert tuple(ali.shape) == (windows,)
        feats_ep1_a.append(feats)
        alis_ep1_a.append(alis)
        total_windows_ep1 += windows
    assert total_windows_ep0 == total_windows_ep1
    data_loader = data.ContextWindowTrainingDataLoader(
        temp_dir,
        params,
        init_epoch=1,
        data_params=data_params,
        num_workers=4,
        seed=seed,
    )
    feats_ep1_b, alis_ep1_b = [], []
    for feats, alis in data_loader:
        feats_ep1_b.append(feats)
        alis_ep1_b.append(alis)
    assert all(
        torch.allclose(feats_a, feats_b)
        for (feats_a, feats_b) in zip(feats_ep1_a, feats_ep1_b)
    )
    assert all(
        torch.all(alis_a == alis_b) for (alis_a, alis_b) in zip(alis_ep1_a, alis_ep1_b)
    )
    data_loader.epoch = 1
    feats_ep1_c, alis_ep1_c = [], []
    for feats, alis in data_loader:
        feats_ep1_c.append(feats)
        alis_ep1_c.append(alis)
    assert all(
        torch.allclose(feats_a, feats_c)
        for (feats_a, feats_c) in zip(feats_ep1_a, feats_ep1_c)
    )
    assert all(
        torch.all(alis_a == alis_c) for (alis_a, alis_c) in zip(alis_ep1_a, alis_ep1_c)
    )


@pytest.mark.cpu
@pytest.mark.parametrize("split_params", [True, False])
def test_window_evaluation_data_loader(temp_dir, populate_torch_dir, split_params):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, "feat")
    ali_dir = os.path.join(temp_dir, "ali")
    os.makedirs(feat_dir)
    os.makedirs(ali_dir)
    if split_params:
        params = data.DataSetParams(batch_size=5)
        data_params = data.ContextWindowDataParams(context_left=1, context_right=1)
    else:
        params = data.ContextWindowDataSetParams(
            context_left=1, context_right=1, batch_size=5
        )
        data_params = None
    feats, alis, _, feat_sizes, _, utt_ids = populate_torch_dir(
        temp_dir, 20, include_ref=False
    )

    def _compare_data_loader(data_loader):
        assert len(data_loader) == 4
        cur_idx = 0
        for b_feats, b_alis, b_feat_sizes, b_utt_ids in data_loader:
            assert tuple(b_feats.shape[1:]) == (3, 5)
            assert b_feats.shape[0] == sum(b_feat_sizes)
            assert tuple(b_utt_ids) == tuple(utt_ids[cur_idx : cur_idx + 5])
            assert torch.allclose(
                b_feats[:, 1], torch.cat(feats[cur_idx : cur_idx + 5])
            )
            assert torch.all(b_alis == torch.cat(alis[cur_idx : cur_idx + 5]))
            cur_idx += 5

    data_loader = data.ContextWindowEvaluationDataLoader(
        temp_dir, params, data_params=data_params, ali_subdir=None
    )
    # check batching works when alignments are empty
    assert next(iter(data_loader))[1] is None
    data_loader = data.ContextWindowEvaluationDataLoader(
        temp_dir, params, data_params=data_params
    )
    _compare_data_loader(data_loader)
    _compare_data_loader(data_loader)  # order should not change
    data_loader = data.ContextWindowEvaluationDataLoader(
        temp_dir, params, data_params=data_params, num_workers=4
    )
    _compare_data_loader(data_loader)  # order should still not change


@pytest.mark.cpu
def test_pydrobert_param_optuna_hooks():
    poptuna = pytest.importorskip("pydrobert.param.optuna")
    optuna = pytest.importorskip("optuna")
    for class_ in (
        data.DataSetParams,
        data.SpectDataSetParams,
        data.ContextWindowDataParams,
        data.ContextWindowDataSetParams,
    ):
        assert issubclass(class_, poptuna.TunableParameterized)
    global_dict = {
        "data_set": data.DataSetParams(),
        "spect_data": data.SpectDataParams(),
        "spect_data_set": data.SpectDataSetParams(),
        "context_window_data": data.ContextWindowDataParams(),
        "context_window_data_set": data.ContextWindowDataSetParams(),
    }
    assert {
        "data_set.batch_size",
        "spect_data.eos",
        "spect_data_set.batch_size",
        "context_window_data.reverse",
        "context_window_data_set.batch_size",
    } - poptuna.get_param_dict_tunable(global_dict) == {"spect_data.eos"}

    def objective(trial):
        param_dict = poptuna.suggest_param_dict(trial, global_dict)
        return param_dict["data_set"].batch_size

    sampler = optuna.samplers.RandomSampler(seed=5)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10)
    assert not {
        "data_set.batch_size",
        "spect_data_set.batch_size",
        "context_window_data.reverse",
        "context_window_data_set.batch_size",
    } - set(study.best_params)
    assert study.best_params["data_set.batch_size"] < 7
