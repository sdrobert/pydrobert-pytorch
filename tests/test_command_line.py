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
import warnings
import glob

import pytest
import torch
import pydrobert.torch.command_line as command_line


@pytest.mark.cpu
@pytest.mark.parametrize("include_frame_shift", [True, False])
def test_get_torch_spect_data_dir_info(
    temp_dir, populate_torch_dir, include_frame_shift
):
    _, alis, _, feat_sizes, _, _ = populate_torch_dir(
        temp_dir,
        19,
        num_filts=5,
        max_ali_class=10,
        include_frame_shift=include_frame_shift,
    )
    # add one with class idx 10 to ensure all classes are accounted for
    torch.save(torch.rand(1, 5), os.path.join(temp_dir, "feat", "utt19.pt"))
    torch.save(torch.tensor([10]), os.path.join(temp_dir, "ali", "utt19.pt"))
    if include_frame_shift:
        torch.save(
            torch.tensor([[100, 0, 1]]), os.path.join(temp_dir, "ref", "utt19.pt")
        )
    else:
        torch.save(torch.tensor([100]), os.path.join(temp_dir, "ref", "utt19.pt"))
    feat_sizes += (1,)
    alis = torch.cat(alis + [torch.tensor([10])])
    alis = [class_idx.item() for class_idx in alis]
    table_path = os.path.join(temp_dir, "info")
    assert not command_line.get_torch_spect_data_dir_info(
        [temp_dir, table_path, "--strict"]
    )

    def check():
        table = dict()
        with open(table_path) as table_file:
            for line in table_file:
                line = line.split()
                table[line[0]] = int(line[1])
        assert table["num_utterances"] == 20
        assert table["total_frames"] == sum(feat_sizes)
        assert table["num_filts"] == 5
        assert table["max_ali_class"] == 10
        assert table["max_ref_class"] == 100
        for class_idx in range(11):
            key = "count_{:02d}".format(class_idx)
            assert table[key] == alis.count(class_idx)

    check()
    if include_frame_shift:
        # ensure we're only looking at the ids in the recorded refs
        torch.save(
            torch.tensor([[100, 0, 101]]), os.path.join(temp_dir, "ref", "utt19.pt")
        )
        assert not command_line.get_torch_spect_data_dir_info([temp_dir, table_path])
        table = dict()
        with open(table_path) as table_file:
            for line in table_file:
                line = line.split()
                table[line[0]] = int(line[1])
        assert table["max_ref_class"] == 100
        # invalidate the data set and try again
        torch.save(
            torch.tensor([[100, 0, 1]]).int(), os.path.join(temp_dir, "ref", "utt19.pt")
        )
        with pytest.raises(ValueError, match="long tensor"):
            command_line.get_torch_spect_data_dir_info(
                [temp_dir, table_path, "--strict"]
            )
        # ...but the problem is fixable. So if we set the flag...
        with pytest.warns(UserWarning, match="long tensor"):
            command_line.get_torch_spect_data_dir_info([temp_dir, table_path, "--fix"])
        check()
        # ...it shouldn't happen again
        command_line.get_torch_spect_data_dir_info([temp_dir, table_path, "--strict"])
        check()


def _write_token2id(path, swap, collapse_vowels=False):
    vowels = {ord(x) for x in "aeiou"}
    with open(path, "w") as f:
        for v in range(ord("a"), ord("z") + 1):
            if swap:
                if collapse_vowels and v in vowels:
                    f.write("{} a\n".format(v - ord("a")))
                else:
                    f.write("{} {}\n".format(v - ord("a"), chr(v)))
            else:
                assert not collapse_vowels
                f.write("{} {}\n".format(chr(v), v - ord("a")))


@pytest.mark.cpu
@pytest.mark.parametrize("tokens", ["token2id", "id2token"])
@pytest.mark.parametrize(
    "skip_frame_times,feat_sizing", [(True, False), (False, True), (False, False)]
)
def test_trn_to_torch_token_data_dir(temp_dir, tokens, skip_frame_times, feat_sizing):
    trn_path = os.path.join(temp_dir, "ref.trn")
    tokens_path = os.path.join(temp_dir, "token2id")
    ref_dir = os.path.join(temp_dir, "ref")
    _write_token2id(tokens_path, tokens == "id2token")
    with open(trn_path, "w") as trn:
        trn.write(
            """\
a b b c (utt1)
(utt2)

d { e / f } g (utt3)
{{{h / i} / j} / k} (utt4)
A a (utt5)
"""
        )
    with warnings.catch_warnings(record=True):
        assert not command_line.trn_to_torch_token_data_dir(
            [
                trn_path,
                tokens_path,
                ref_dir,
                "--alt-handler=first",
                "--unk-symbol=c",
                "--chunk-size=1",
            ]
            + (["--swap"] if tokens == "id2token" else [])
            + (["--skip-frame-times"] if skip_frame_times else [])
            + (["--feat-sizing"] if feat_sizing else [])
        )
    exp_utt1 = torch.tensor([0, 1, 1, 2])
    exp_utt3 = torch.tensor([3, 4, 6])
    exp_utt4 = torch.tensor([7])
    exp_utt5 = torch.tensor([2, 0])
    if feat_sizing:
        exp_utt1 = exp_utt1.unsqueeze(-1)
        exp_utt3 = exp_utt3.unsqueeze(-1)
        exp_utt4 = exp_utt4.unsqueeze(-1)
        exp_utt5 = exp_utt5.unsqueeze(-1)
    elif not skip_frame_times:
        neg1_tensor = torch.tensor([[-1, -1]] * 10)
        exp_utt1 = torch.cat([exp_utt1.unsqueeze(-1), neg1_tensor[:4]], -1)
        exp_utt3 = torch.cat([exp_utt3.unsqueeze(-1), neg1_tensor[:3]], -1)
        exp_utt4 = torch.cat([exp_utt4.unsqueeze(-1), neg1_tensor[:1]], -1)
        exp_utt5 = torch.cat([exp_utt5.unsqueeze(-1), neg1_tensor[:2]], -1)
    act_utt1 = torch.load(os.path.join(ref_dir, "utt1.pt"))
    assert exp_utt1.shape == act_utt1.shape
    assert torch.all(act_utt1 == exp_utt1)
    act_utt2 = torch.load(os.path.join(ref_dir, "utt2.pt"))
    assert not act_utt2.numel()
    act_utt3 = torch.load(os.path.join(ref_dir, "utt3.pt"))
    assert exp_utt3.shape == act_utt3.shape
    assert torch.all(act_utt3 == exp_utt3)
    act_utt4 = torch.load(os.path.join(ref_dir, "utt4.pt"))
    assert exp_utt4.shape == act_utt4.shape
    assert torch.all(act_utt4 == exp_utt4)
    act_utt5 = torch.load(os.path.join(ref_dir, "utt5.pt"))
    assert exp_utt5.shape == act_utt5.shape
    assert torch.all(act_utt5 == exp_utt5)


@pytest.mark.cpu
@pytest.mark.parametrize("tokens", ["token2id", "id2token"])
@pytest.mark.parametrize("include_frame_shift", [True, False])
def test_torch_token_data_dir_to_trn(temp_dir, tokens, include_frame_shift):
    torch.manual_seed(1000)
    num_utts = 100
    max_tokens = 10
    num_digits = torch.log10(torch.tensor(float(num_utts))).long().item() + 1
    utt_fmt = "utt{{:0{}d}}".format(num_digits)
    trn_path = os.path.join(temp_dir, "ref.trn")
    tokens_path = os.path.join(temp_dir, "id2token")
    ref_dir = os.path.join(temp_dir, "ref")
    _write_token2id(tokens_path, tokens == "id2token")
    if not os.path.isdir(ref_dir):
        os.makedirs(ref_dir)
    exps = []
    for utt_idx in range(num_utts):
        utt_id = utt_fmt.format(utt_idx)
        num_tokens = torch.randint(max_tokens + 1, (1,)).long().item()
        ids = torch.randint(26, (num_tokens,)).long()
        if include_frame_shift:
            tok = torch.stack([ids] + ([torch.full_like(ids, -1)] * 2), -1)
        else:
            tok = ids
        torch.save(tok, os.path.join(ref_dir, utt_id + ".pt"))
        transcript = " ".join([chr(x + ord("a")) for x in ids.tolist()])
        transcript += " ({})".format(utt_id)
        exps.append(transcript)
    assert not command_line.torch_token_data_dir_to_trn(
        [ref_dir, tokens_path, trn_path] + (["--swap"] if tokens == "token2id" else [])
    )
    with open(trn_path, "r") as trn:
        acts = trn.readlines()
    assert len(exps) == len(acts)
    for exp, act in zip(exps, acts):
        assert exp.strip() == act.strip()


def _write_wc2utt(path, swap, chan, num_utts):
    num_digits = torch.log10(torch.tensor(float(num_utts))).long().item() + 1
    idx_fmt = "{{0:0{}d}}".format(num_digits)
    if swap:
        fmt = "u_{0} w_{0} {{1}}\n".format(idx_fmt)
    else:
        fmt = "w_{0} {{1}} u_{0}\n".format(idx_fmt)
    with open(path, "w") as f:
        for utt_idx in range(num_utts):
            f.write(fmt.format(utt_idx, chan))


@pytest.mark.cpu
@pytest.mark.parametrize("tokens", ["token2id", "id2token"])
@pytest.mark.parametrize("channels", ["wc2utt", "utt2wc", None])
def test_ctm_to_torch_token_data_dir(temp_dir, tokens, channels):
    ctm_path = os.path.join(temp_dir, "ref.ctm")
    tokens_path = os.path.join(temp_dir, tokens)
    channels_path = os.path.join(temp_dir, channels) if channels else None
    ref_dir = os.path.join(temp_dir, "ref")
    _write_token2id(tokens_path, tokens == "id2token")
    if channels:
        _write_wc2utt(channels_path, channels == "utt2wc", "A", 5)
    with open(ctm_path, "w") as ctm:
        ctm.write(
            """\
;; some text
w_1 A 0.1 1.0 a

w_1 A 0.2 1.0 b
w_1 A 0.3 1.0 c   ;; ignore this comment
w_2 A 0.0 0.0 b
w_3 A 0.0 1000.0 d
w_3 A 1.0 0.1 d
w_4 A 0.0 2.0 Z
w_4 A 0.1 1.1 a
"""
        )
    args = [ctm_path, tokens_path, ref_dir, "--unk-symbol=a"]
    if tokens == "id2token":
        args.append("--swap")
    if channels == "utt2wc":
        args.append("--utt2wc={}".format(channels_path))
    elif channels == "wc2utt":
        args.append("--wc2utt={}".format(channels_path))
    assert not command_line.ctm_to_torch_token_data_dir(args)
    act_utt1 = torch.load(os.path.join(ref_dir, "u_1.pt" if channels else "w_1.pt"))
    assert torch.all(
        act_utt1 == torch.tensor([[0, 10, 110], [1, 20, 120], [2, 30, 130]])
    )
    act_utt2 = torch.load(os.path.join(ref_dir, "u_2.pt" if channels else "w_2.pt"))
    assert torch.all(act_utt2 == torch.tensor([[1, 0, 0]]))
    act_utt3 = torch.load(os.path.join(ref_dir, "u_3.pt" if channels else "w_3.pt"))
    assert torch.all(act_utt3 == torch.tensor([[3, 0, 100000], [3, 100, 110]]))
    act_utt4 = torch.load(os.path.join(ref_dir, "u_4.pt" if channels else "w_4.pt"))
    assert torch.all(act_utt4 == torch.tensor([[0, 0, 200], [0, 10, 120]]))


@pytest.mark.cpu
@pytest.mark.parametrize("tokens", ["token2id", "id2token"])
@pytest.mark.parametrize("channels", ["wc2utt", "utt2wc", None])
@pytest.mark.parametrize("frame_shift_ms", [20.0, 0.1])
def test_torch_token_data_dir_to_ctm(temp_dir, tokens, channels, frame_shift_ms):
    torch.manual_seed(420)
    ctm_path = os.path.join(temp_dir, "ref.ctm")
    tokens_path = os.path.join(temp_dir, tokens)
    channels_path = os.path.join(temp_dir, channels) if channels else None
    ref_dir = os.path.join(temp_dir, "ref")
    num_utts, max_tokens, max_start, max_dur = 100, 10, 1000, 100
    max_tokens = 10
    num_digits = torch.log10(torch.tensor(float(num_utts))).long().item() + 1
    utt_fmt = "u_{{:0{}d}}".format(num_digits)
    wfn_fmt = "{}_{{:0{}d}}".format("w" if channels else "u", num_digits)
    _write_token2id(tokens_path, tokens == "id2token")
    if channels:
        _write_wc2utt(channels_path, channels == "utt2wc", "A", num_utts)
    if not os.path.isdir(ref_dir):
        os.makedirs(ref_dir)
    exps = []
    for utt_idx in range(num_utts):
        utt_id = utt_fmt.format(utt_idx)
        wfn_id = wfn_fmt.format(utt_idx)
        num_tokens = torch.randint(max_tokens + 1, (1,)).long().item()
        ids = torch.randint(26, (num_tokens,)).long()
        starts = torch.randint(max_start, (num_tokens,)).long()
        durs = torch.randint(max_dur, (num_tokens,)).long()
        ends = starts + durs
        tok = torch.stack([ids, starts, ends], -1)
        torch.save(tok, os.path.join(ref_dir, utt_id + ".pt"))
        for token, start, end in sorted(tok.tolist(), key=lambda x: x[1:]):
            start = start * frame_shift_ms / 1000
            end = end * frame_shift_ms / 1000
            exps.append(
                "{} A {} {} {}".format(
                    wfn_id, start, end - start, chr(token + ord("a"))
                )
            )
    args = [
        ref_dir,
        tokens_path,
        ctm_path,
        "--frame-shift-ms={}".format(frame_shift_ms),
    ]
    if tokens == "token2id":
        args.append("--swap")
    if channels == "utt2wc":
        args.append("--utt2wc={}".format(channels_path))
    elif channels == "wc2utt":
        args.append("--wc2utt={}".format(channels_path))
    assert not command_line.torch_token_data_dir_to_ctm(args)
    with open(ctm_path, "r") as ctm:
        acts = ctm.readlines()
    assert len(exps) == len(acts)
    for exp, act in zip(exps, acts):
        assert exp.strip() == act.strip()


@pytest.mark.cpu
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("per_utt", [True, False])
@pytest.mark.parametrize(
    "tokens,collapse_vowels",
    [("token2id", False), ("id2token", True), ("id2token", False), (None, False)],
)
@pytest.mark.parametrize("norm", [True, False])
@pytest.mark.parametrize("with_timing", [True, False])
def test_compute_torch_token_data_dir_error_rates(
    temp_dir, per_utt, tokens, collapse_vowels, norm, with_timing
):
    torch.manual_seed(3820)
    tokens_path = os.path.join(temp_dir, "map")
    ignore_path = os.path.join(temp_dir, "ignore")
    replace_path = os.path.join(temp_dir, "replace")
    out_path = os.path.join(temp_dir, "out")
    ref_dir = os.path.join(temp_dir, "ref")
    hyp_dir = os.path.join(temp_dir, "hyp")
    if not os.path.isdir(ref_dir):
        os.makedirs(ref_dir)
    if not os.path.isdir(hyp_dir):
        os.makedirs(hyp_dir)
    num_elem = 40
    missing_prob = 0.1
    max_fillers = 5
    ignore_chars = "_#"
    replace_chars = "*/"
    with open(ignore_path, "w") as f:
        if tokens is None:
            f.write(" ".join([str(ord(c) - ord("a")) for c in ignore_chars]))
        else:
            f.write(" ".join(ignore_chars))
        f.flush()
    with open(replace_path, "w") as f:
        for c in replace_chars:
            if tokens is None:
                f.write(
                    "{} {}\n".format(ord(c) - ord("a"), ord(ignore_chars[0]) - ord("a"))
                )
            else:
                f.write("{} {}\n".format(c, ignore_chars[0]))
    ignore_chars += replace_chars
    tuples = (
        ("cat", "bat", 1, 1),
        ("transubstantiation", "transwhatnow", 10, 10),
        ("cool", "coal", 1, 0),
        ("zap", "zippy", 3, 2),
    )
    if tokens is not None:
        _write_token2id(
            tokens_path, tokens == "id2token", collapse_vowels=collapse_vowels
        )
        with open(tokens_path, "a") as f:
            for c in ignore_chars:
                if tokens == "id2token":
                    f.write("{} {}\n".format(ord(c) - ord("a"), c))
                else:
                    f.write("{} {}\n".format(c, ord(c) - ord("a")))
    exp = dict()
    tot_err = 0
    tot_len = 0
    num_utt_ids = 0
    while len(exp) < num_elem:
        tuple_ = tuples[torch.randint(len(tuples), (1,)).item()]
        if collapse_vowels:
            ref, hyp, _, er = tuple_
        else:
            ref, hyp, er, _ = tuple_
        len_ = len(ref)
        num_ref_fillers = torch.randint(max_fillers, (1,)).item()
        num_hyp_fillers = torch.randint(max_fillers, (1,)).item()
        for _ in range(num_ref_fillers):
            fill_idx = torch.randint(len(ref) + 1, (1,)).item()
            filler = ignore_chars[torch.randint(len(ignore_chars), (1,)).item()]
            ref = ref[:fill_idx] + filler + ref[fill_idx:]
        for _ in range(num_hyp_fillers):
            fill_idx = torch.randint(len(hyp) + 1, (1,)).item()
            filler = ignore_chars[torch.randint(len(ignore_chars), (1,)).item()]
            hyp = hyp[:fill_idx] + filler + hyp[fill_idx:]
        if with_timing:
            ref = torch.tensor([(ord(c) - ord("a"), -1, -1) for c in ref])
            hyp = torch.tensor([(ord(c) - ord("a"), -1, -1) for c in hyp])
        else:
            ref = torch.tensor([ord(c) - ord("a") for c in ref])
            hyp = torch.tensor([ord(c) - ord("a") for c in hyp])
        utt_id = num_utt_ids
        num_utt_ids += 1
        ref_path = os.path.join(ref_dir, "{}.pt".format(utt_id))
        hyp_path = os.path.join(hyp_dir, "{}.pt".format(utt_id))
        if torch.rand(1).item() < missing_prob:
            if torch.randint(2, (1,)).item():
                torch.save(ref, ref_path)
            else:
                torch.save(hyp, hyp_path)
        torch.save(ref, ref_path)
        torch.save(hyp, hyp_path)
        tot_err += er
        tot_len += len_
        exp[str(utt_id)] = er / (len_ if norm else 1)
    args = [
        ref_dir,
        hyp_dir,
        out_path,
        "--ignore",
        ignore_path,
        "--replace",
        replace_path,
        "--warn-missing",
    ]
    if not norm:
        args.append("--distances")
    if per_utt:
        args.append("--per-utt")
    if tokens is not None:
        args += ["--id2token", tokens_path]
        if tokens == "token2id":
            args.append("--swap")
    assert not command_line.compute_torch_token_data_dir_error_rates(args)
    if per_utt:
        with open(out_path) as f:
            while True:
                ls = f.readline().strip().split()
                if len(ls) != 2:
                    break
                assert abs(exp[ls[0]] - float(ls[1])) < 1e-5
                del exp[ls[0]]
        assert not len(exp)
    else:
        exp = tot_err / (tot_len if norm else num_utt_ids)
        with open(out_path) as f:
            act = float(f.read().strip())
            assert abs(exp - act) < 1e-4


@pytest.mark.cpu
def test_error_rates_match_sclite_with_flag(temp_dir):
    dir_ = os.path.join(os.path.dirname(__file__), "sclite")
    token2id = os.path.join(dir_, "token2id.txt")
    per_utt_act_file = os.path.join(temp_dir, "per_utt.txt")
    total_act_file = os.path.join(temp_dir, "total.txt")
    per_utt_exp_file = os.path.join(dir_, "per_utt.txt")
    total_exp_file = os.path.join(dir_, "total.txt")
    ref_dir = os.path.join(temp_dir, "ref")
    hyp_dir = os.path.join(temp_dir, "hyp")
    assert not command_line.trn_to_torch_token_data_dir(
        [os.path.join(dir_, "ref.trn"), token2id, ref_dir]
    )
    assert not command_line.trn_to_torch_token_data_dir(
        [os.path.join(dir_, "hyp.trn"), token2id, hyp_dir]
    )
    assert not command_line.compute_torch_token_data_dir_error_rates(
        [ref_dir, hyp_dir, total_act_file, "--nist-costs", "--quiet"]
    )
    assert not command_line.compute_torch_token_data_dir_error_rates(
        [ref_dir, hyp_dir, per_utt_act_file, "--nist-costs", "--per-utt", "--quiet"]
    )
    per_utt_exp = dict()
    per_utt_act = dict()
    for fn, dict_ in ((per_utt_exp_file, per_utt_exp), (per_utt_act_file, per_utt_act)):
        with open(fn) as file_:
            for line in file_:
                utt, v = line.strip().split()
                v = "{:.03f}".format(float(v))
                dict_[utt] = v
    for utt in per_utt_exp:
        assert per_utt_exp[utt] == per_utt_act[utt], utt
    with open(total_exp_file) as file_:
        total_exp = "{:.03f}".format(float(file_.read().strip()))
    with open(total_act_file) as file_:
        total_act = "{:.03f}".format(float(file_.read().strip()))
    assert total_exp == total_act


@pytest.mark.cpu
def test_torch_spect_data_dir_to_wds(temp_dir, populate_torch_dir):
    wds = pytest.importorskip("webdataset")

    NN, N = 100, 10
    torch_dir = os.path.join(temp_dir, "foo")
    tar = os.path.join(temp_dir, "foo.tar")

    feats, alis, refs, _, _, utt_ids = populate_torch_dir(torch_dir, NN)
    assert not command_line.torch_spect_data_dir_to_wds([torch_dir, tar])

    ds = (
        wds.WebDataset("file:" + tar.replace("\\", "/"))
        .decode(wds.handle_extension(".pth", torch.load))
        .to_tuple("feat.pth", "ali.pth", "ref.pth", "__key__")
    )

    for idx, (feat, ali, ref, utt_id) in enumerate(ds):
        assert (feat == feats[idx]).all()
        assert (ali == alis[idx]).all()
        assert (ref == refs[idx]).all()
        assert utt_id == utt_ids[idx]
    assert idx == NN - 1

    assert not command_line.torch_spect_data_dir_to_wds(
        [torch_dir, tar, "--shard", "--max-samples-per-shard", str(N)]
    )

    shards = glob.glob(f"{glob.escape(temp_dir)}/foo.tar.*")
    assert len(shards) == (NN - 1) // N + 1
    ds = wds.DataPipeline(
        wds.SimpleShardList(sorted("file:" + x for x in shards)),
        wds.tarfile_to_samples(),
        wds.decode(wds.handle_extension(".pth", torch.load)),
        wds.to_tuple("feat.pth", "ali.pth", "ref.pth", "__key__"),
    )

    for idx, (feat, ali, ref, utt_id) in enumerate(ds):
        assert (feat == feats[idx]).all()
        assert (ali == alis[idx]).all()
        assert (ref == refs[idx]).all()
        assert utt_id == utt_ids[idx]
    assert idx == NN - 1


@pytest.mark.cpu
@pytest.mark.parametrize("groups", [True, False])
def test_compute_mvn_stats_for_torch_feat_data_dir(
    temp_dir, populate_torch_dir, groups
):
    N, G = 100, 4
    feats, _, _, _, _, utt_ids = populate_torch_dir(temp_dir, N)
    feat_dir = os.path.join(temp_dir, "feat")
    assert os.path.exists(feat_dir)
    out_file = os.path.join(temp_dir, "out.pt")
    id2gid_path = os.path.join(temp_dir, "id2gid.map")
    args = [feat_dir, out_file]
    if groups:
        args += ["--id2gid", id2gid_path]
        gids = tuple(chr(g + ord("a")) for g in range(G))
        feats_ = dict((g, []) for g in gids)
        with open(id2gid_path, "w") as id2gid:
            for i, (feat, utt_id) in enumerate(zip(feats, utt_ids)):
                gid = gids[i % G]
                id2gid.write(f"{utt_id} {gid}\n")
                feats_[gid].append(feat)
        exp = dict()
        for gid, feat in feats_.items():
            feat = torch.cat(feat, 0).double()
            exp[gid] = {"mean": feat.mean(0), "std": feat.std(0, False)}
    else:
        feats = torch.cat(feats, 0).double()
        exp = {None: {"mean": feats.mean(0), "std": feats.std(0, False)}}

    assert not command_line.compute_mvn_stats_for_torch_feat_data_dir(args)

    act = torch.load(out_file)
    if not groups:
        act = {None: act}

    assert set(act) == set(exp)

    for gid in act:
        stats_exp, stats_act = exp[gid], act[gid]
        assert set(stats_exp) == set(stats_act) == {"mean", "std"}, gid
        for stat in stats_act:
            assert torch.allclose(stats_exp[stat], stats_act[stat]), (gid, stat)
