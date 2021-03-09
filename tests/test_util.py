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
from tempfile import SpooledTemporaryFile

import torch
import pytest
import pydrobert.torch.util as util
import numpy as np


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
    ngram_list = util.parse_arpa_lm(file_)
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
    ngram_list = util.parse_arpa_lm(file_, token2id=token2id)
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
    ngram_list = util.parse_arpa_lm(file_)
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
        util.parse_arpa_lm(file_)
    file_.seek(0)
    file_.write(
        r"""\
Here's an empty one

\data\
\end\
"""
    )
    file_.seek(0)
    assert util.parse_arpa_lm(file_) == []


@pytest.mark.cpu
@pytest.mark.parametrize("distribution", [True, False])
def test_beam_search_advance_greedy(distribution):
    torch.manual_seed(50)
    N, C, T = 30, 100, 25
    logits = torch.randn(T, N, C)
    if distribution:
        greedy_logits, greedy_paths = torch.nn.functional.log_softmax(logits, -1).max(2)
    else:
        greedy_logits, greedy_paths = logits.max(2)
    greedy_scores = greedy_logits.sum(0) + torch.log(torch.tensor(1 / C))
    y = None
    score = None
    for logit in logits:
        score, y, _ = util.beam_search_advance(
            logit, 1, score, y, distribution=distribution
        )
    score, y = score.squeeze(1), y.squeeze(2)
    assert torch.allclose(score, greedy_scores)
    assert torch.all(y == greedy_paths)


@pytest.mark.parametrize("prevent_eos", [True, False])
def test_beam_search_advance(device, prevent_eos):
    logits_1 = torch.tensor(
        [[-10, -2, -4, -10], [-2, 0, 1, 1.1]], device=device
    )  # ~[[x, 0, -4, x], [x, x, -.8, -.6]]
    score_1, y_1, s_1 = util.beam_search_advance(logits_1, 2)
    assert torch.all(y_1 == torch.tensor([[[1, 2], [3, 2]]], device=device))
    assert torch.all(s_1 == 0)
    logits_2 = torch.tensor(
        [
            [[-5.0, -6, -7, -8.0], [-400, -300, -200, -1]],  # beam for batch 1
            [[2, 1, 1, 1], [-1, -2, -3, -4]],  # beam for batch 2
        ],
        device=device,
    )  # ~ [[[-.4 -1.4 x x] [x x x 0]],
    #                      [[-.7 -1.7 x x] [-.4 -1.4 x x]]
    # batch 0: 0->0 0->1 win b/c 1->3 can't make up for score_1
    # batch 1: 1->0 0->0 win b/c score_1 about even
    score_2, y_2, s_2 = util.beam_search_advance(logits_2, 2, score_1)
    assert torch.all(y_2 == torch.tensor([[[0, 1], [0, 0]]], device=device))
    assert torch.all(s_2 == torch.tensor([[0, 0], [1, 0]], device=device))
    logits_3 = torch.tensor(
        [[[1000.0, 0, 0, 0], [0, -100, 0, 0]], [[0, 0, 0, 100], [2, 2, 1000, 10]]],
        device=device,
    )  # ~ [[[0 x x x] [-1 -101 -1 -1]],
    #                      [[x x x 0] [x x 0 x]]
    # batch 0: 0->0 1->1 batch 1 done, but no priority b/c 0->0 very small
    # batch 1: 0->3 1->2
    score_3, y_3, s_3 = util.beam_search_advance(logits_3, 2, score_2, y_2, 1)
    assert torch.all(
        y_3 == torch.tensor([[[0, 1], [0, 0]], [[0, 1], [3, 2]]], device=device)  # y_2
    )
    assert torch.all(s_3 == torch.tensor([[0, 1], [0, 1]], device=device))
    logits_4 = torch.tensor(
        [[[1.0, 2, 3, 4], [5, 6, 7, 8]], [[2, 2, 3, 2], [5, 6, 7, 8]]],
        device=device,
        requires_grad=True,
    )
    # (note no eos condition)
    score_4, y_4, s_4 = util.beam_search_advance(logits_4, 1, score_3)
    assert torch.all(y_4.flatten() == torch.tensor([3, 3], device=device))
    assert torch.all(s_4.flatten() == torch.tensor([0, 1], device=device))
    g = torch.autograd.grad(
        [score_4], [logits_4], grad_outputs=torch.ones_like(score_4)
    )[0]
    # we should only have a gradient for the chosen beam per batch
    assert torch.allclose(g[0, 1, :], torch.zeros_like(g[0, 1, :]))
    assert torch.allclose(g[1, 0, :], torch.zeros_like(g[1, 0, :]))
    # all elements of a non-peaked softmax in the chosen beam should have a
    # non-negligible gradient
    assert abs(g[0, 0, 1].item()) > 1e-4
    assert abs(g[1, 1, 1].item()) > 1e-4
    # ensure that all paths are always unequal
    torch.manual_seed(30)
    y = score = None
    N, W, C, T = 5, 10, 20, 100
    eos = 0 if prevent_eos else -1
    logits_t = torch.randn(N, C).to(device)
    lens = torch.randint(1, T, (N,)).to(device)
    while y is None or not torch.all(y[-1].eq(eos)):
        score, y, _ = util.beam_search_advance(
            logits_t, W, score, y, eos=eos, lens=lens, prevent_eos=prevent_eos
        )
        logits_t = torch.randn(N, W, C).to(device)
        for i in range(W - 1):
            beam_i = y[..., i]
            for j in range(i + 1, W - 1):
                beam_j = y[..., j]
                for k in range(N):
                    assert not torch.all(beam_i[:, k] == beam_j[:, k])
    for bt, l in enumerate(lens):
        for bm in range(W):
            assert torch.all(y[l.item() :, bt, bm] == eos)
            assert not torch.any(y[: l.item(), bt, bm] == eos)


@pytest.mark.parametrize("norm", [True, False])
@pytest.mark.parametrize("include_eos", [0, 1])
@pytest.mark.parametrize("batch_first", [True, False])
def test_error_rate_against_known(device, norm, include_eos, batch_first):
    eos = 0
    pairs = (
        ((1, 2, 3), (1, 2, 3), 0,),
        ((2, 3), (1, 2, 3), 1,),
        ((1, 3), (1, 2, 3), 1,),
        ((3,), (1, 2, 3), 2,),
        ((1, 2, 3), (1, 3), 1,),
        ((1, 2, 3), (1, 2,), 1,),
        ((1, 2, 3), (1,), 2,),
        ((1, 3, 1, 2, 3), (1, 2, 3), 2,),
        ((1, 2, 3), (4, 5, 6), 3,),
        ((2, 2, 2), (2,), 2,),
        (tuple(), (1,), 1,),
        (tuple(), tuple(), 0,),
    )
    ref_lens = torch.tensor([len(x[0]) + include_eos for x in pairs], device=device)
    hyp_lens = torch.tensor([len(x[1]) + include_eos for x in pairs], device=device)
    ref = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x[0] + (eos,) * include_eos) for x in pairs],
        padding_value=eos,
        batch_first=batch_first,
    ).to(device)
    hyp = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x[1] + (eos,) * include_eos) for x in pairs],
        padding_value=eos,
        batch_first=batch_first,
    ).to(device)
    exp = torch.tensor([float(x[2]) for x in pairs], device=device)
    if norm:
        exp = torch.where(ref_lens == 0, hyp_lens.ne(0).float(), exp / ref_lens.float())
    act = util.error_rate(
        ref,
        hyp,
        eos=eos,
        warn=False,
        norm=norm,
        include_eos=include_eos,
        batch_first=batch_first,
    )
    assert torch.allclose(exp, act)


@pytest.mark.parametrize("ins_cost", [-0.1, 0.0, 1.0])
@pytest.mark.parametrize("del_cost", [-0.1, 0.0, 1.0])
@pytest.mark.parametrize("sub_cost", [-0.1, 0.0, 1.0])
@pytest.mark.parametrize("ref_bigger", [True, False])
def test_error_rate_against_simple_impl(
    device, ins_cost, del_cost, sub_cost, ref_bigger
):
    torch.manual_seed(2502)
    hyp_steps, ref_steps, batch_size, num_classes = 10, 9, 50, 10
    if ref_bigger:
        ref_steps, hyp_steps = hyp_steps, ref_steps
    ref = torch.randint(num_classes, (ref_steps, batch_size), device=device)
    hyp = torch.randint(num_classes, (hyp_steps, batch_size), device=device)
    # here's a standard, non-vectorized (except for batch) implementation that
    # is hard to screw up
    cost_matrix = torch.empty(hyp_steps + 1, ref_steps + 1, batch_size, device=device)
    cost_matrix[0] = (
        torch.arange(float(ref_steps + 1), device=device).unsqueeze(-1) * del_cost
    )
    cost_matrix[:, 0] = (
        torch.arange(float(hyp_steps + 1), device=device).unsqueeze(-1) * ins_cost
    )
    for hyp_idx in range(1, hyp_steps + 1):
        for ref_idx in range(1, ref_steps + 1):
            sub = torch.where(
                ref[ref_idx - 1] == hyp[hyp_idx - 1],
                torch.tensor(0.0, device=device),
                torch.tensor(sub_cost, device=device),
            )
            cost_matrix[hyp_idx, ref_idx] = torch.min(
                torch.min(
                    cost_matrix[hyp_idx - 1, ref_idx - 1] + sub,
                    cost_matrix[hyp_idx - 1, ref_idx] + ins_cost,
                ),
                cost_matrix[hyp_idx, ref_idx - 1] + del_cost,
            )
    exp = cost_matrix[-1, -1]
    act = util.error_rate(
        ref, hyp, norm=False, ins_cost=ins_cost, del_cost=del_cost, sub_cost=sub_cost
    )
    assert torch.allclose(exp, act)


@pytest.mark.parametrize("include_eos", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("exclude_last", [True, False])
def test_optimal_completion(device, include_eos, batch_first, exclude_last):
    eos, padding = ord("#"), -1
    triplets = (
        (
            "sunday#",
            "saturday#",
            ["s", "u", "un", "und", "n", "nd", "a", "y", "#", ""],
        ),
        ("sunday#", "satrapy#", ["s", "u", "un", "und", "unda", "y", "y#", "#", ""],),
        ("abc#", "abc#", ["a", "b", "c", "#", ""]),
        ("foot#", "bot#", ["f", "fo", "o", "ot#", ""]),
        ("abc#", "def#", ["a", "ab", "abc", "abc#", ""]),
    )
    ref = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor([ord(c) for c in word]) for (word, _, _) in triplets],
        batch_first=batch_first,
        padding_value=padding,
    ).to(device)
    hyp = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor([ord(c) for c in word]) for (_, word, _) in triplets],
        batch_first=batch_first,
        padding_value=eos,
    ).to(device)
    act = util.optimal_completion(
        ref,
        hyp,
        eos=eos,
        padding=padding,
        batch_first=batch_first,
        exclude_last=exclude_last,
        include_eos=include_eos,
    )
    if not batch_first:
        act = act.transpose(0, 1)  # (batch, hyp, ref)
    assert act.shape[0] == len(triplets)
    for act_bt, (_, _, exp_bt) in zip(act, triplets):
        if not include_eos:
            exp_bt = [nexts.replace("#", "") for nexts in exp_bt[:-1]]
        if exclude_last:
            exp_bt = exp_bt[:-1]
        assert act_bt.shape[0] >= len(exp_bt)
        assert torch.all(act_bt[len(exp_bt) :].eq(padding))
        for act_bt_hyp, exp_bt_hyp in zip(act_bt, exp_bt):
            act_bt_hyp = act_bt_hyp.masked_select(act_bt_hyp.ne(padding))
            act_bt_hyp = sorted(chr(i) for i in act_bt_hyp.tolist())
            assert sorted(exp_bt_hyp) == act_bt_hyp


def test_random_walk_advance(device):
    torch.manual_seed(3487209)
    N, T, S, C = 5, 1000, 4, 10
    logits = torch.randn(C, N, C, device=device)
    transitions = torch.nn.functional.softmax(logits.transpose(0, 1), -1)
    last = transitions
    stationary = torch.bmm(last, transitions)
    while not torch.allclose(stationary, last):
        last, stationary = stationary, torch.bmm(stationary, stationary)
    assert torch.allclose(stationary.sum(2), torch.tensor(1.0, device=device))
    exp = (stationary[:, 0] * torch.arange(float(C)).to(device)).sum(1)
    y = None
    for _ in range(T):
        if y is None:
            logits_t = logits[0]
        else:
            logits_t = torch.gather(
                logits.unsqueeze(2).expand(C, N, S, C),
                0,
                y[-1].unsqueeze(0).unsqueeze(-1).expand(1, N, S, C),
            ).squeeze(0)
        y = util.random_walk_advance(logits_t, S, y)
    act = y.float().mean(0).mean(1)
    assert torch.allclose(exp, act, atol=0.1)


def test_random_walk_advance_relaxation(device):
    torch.manual_seed(652916)
    N, S, C = 23, 32, 12
    logits_t = torch.randn(N, C, device=device, requires_grad=True)
    y = util.random_walk_advance(logits_t, S)
    y, z = util.random_walk_advance(logits_t, S, include_relaxation=True)
    assert torch.all(y[-1] == z.argmax(dim=-1))
    (g,) = torch.autograd.grad([z], [logits_t], grad_outputs=torch.ones_like(z))
    assert g.ne(0.0).any()
    y[..., : S // 2] = -1
    logits_t = logits_t.unsqueeze(1).expand_as(z)
    y, z = util.random_walk_advance(logits_t, S, y, eos=-1, include_relaxation=True)
    assert y[..., : S // 2].eq(-1).all()
    assert y[..., S // 2 :].ne(-1).all()
    assert torch.isinf(-z[:, : S // 2]).all()
    assert not torch.isinf(-z[:, S // 2 :]).any()
    (g,) = torch.autograd.grad([z], [logits_t], grad_outputs=torch.ones_like(z))
    assert g.ne(0.0).any()
    assert g[:, : S // 2].eq(0.0).all()
    y[-1] = -1
    y, z = util.random_walk_advance(logits_t, S, y, eos=-1, include_relaxation=True)
    # it should be defined, but zero
    (g,) = torch.autograd.grad([z], [logits_t], grad_outputs=torch.ones_like(z))
    assert g.eq(0.0).all()


@pytest.mark.parametrize("prevent_eos", [True, False])
@pytest.mark.parametrize("lens", [True, False])
def test_random_walk_advance_config(device, prevent_eos, lens):
    torch.manual_seed(332)
    N, T, S, C = 20, 100, 5, 4
    eos = 0 if prevent_eos else -1
    lens = torch.randint(1, T, (N,), device=device) if lens else None
    y = None
    for _ in range(T):
        logits_t = torch.randn(N, S, C, device=device)
        y = util.random_walk_advance(logits_t, S, y, eos, lens, prevent_eos)
    if lens is None:
        lens = torch.tensor(T, device=device).expand(N)
    for bt, l in enumerate(lens):
        for smp in range(S):
            assert torch.all(y[l.item() :, bt, smp] == eos)
            assert not torch.any(y[: l.item(), bt, smp] == eos)


@pytest.mark.parametrize("exclude_last", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("norm", [True, False])
@pytest.mark.parametrize("ins_cost", [1.0, 0.5])
def test_prefix_error_rates(device, exclude_last, batch_first, norm, ins_cost):
    torch.manual_seed(1937540)
    N, max_ref_steps, max_hyp_steps, C, eos = 30, 11, 12, 10, -1
    padding = -2
    hyp_lens = torch.randint(1, max_hyp_steps + 1, (N,), device=device)
    ref_lens = torch.randint(1, max_ref_steps + 1, (N,), device=device)
    hyp = torch.randint(C, (max_hyp_steps, N), device=device)
    ref = torch.randint(C, (max_ref_steps, N), device=device)
    hyp[hyp_lens - 1, range(N)] = eos
    ref[ref_lens - 1, range(N)] = eos
    ref_lens -= 1  # exclude the eos
    hyp_lens -= 1
    act = util.prefix_error_rates(
        ref.t().contiguous() if batch_first else ref,
        hyp.t().contiguous() if batch_first else hyp,
        eos=eos,
        include_eos=False,
        norm=norm,
        ins_cost=ins_cost,
        exclude_last=exclude_last,
        padding=padding,
        batch_first=batch_first,
        warn=False,
    )
    if batch_first:
        act = act.t().contiguous()
    exp = torch.empty(max_hyp_steps + (0 if exclude_last else 1), N, device=device)
    # if include_eos were true, `hyp` would get a bonus for the final `eos`
    # which isn't in its prefix
    for pref_len in range(max_hyp_steps - (1 if exclude_last else 0), -1, -1):
        hyp[pref_len:] = eos
        exp[pref_len] = util.error_rate(
            ref,
            hyp,
            eos=eos,
            include_eos=False,
            norm=norm,
            ins_cost=ins_cost,
            warn=False,
        )
    exp = exp.masked_fill(
        (
            torch.arange(exp.shape[0], device=device).unsqueeze(1)
            >= hyp_lens + (0 if exclude_last else 1)
        ),
        padding,
    )
    assert torch.allclose(exp, act)


@pytest.mark.parametrize("dim", [0, 2, -1, None])
def test_sequence_log_probs(device, dim):
    torch.manual_seed(24519)
    max_steps, num_classes, eos = 30, 10, 0
    dim1, dim2, dim3, dim4 = 5, 2, 1, 3
    logits = torch.full(
        (max_steps, dim1, dim2, dim3, dim4, num_classes), -float("inf"), device=device
    )
    hyp = torch.randint(
        1, num_classes, (max_steps, dim1, dim2, dim3, dim4), device=device
    )
    hyp_lens = torch.randint(2, max_steps, (dim1, dim2, dim3, dim4), device=device)
    len_mask = torch.arange(max_steps, device=device).unsqueeze(-1)
    if dim is None:
        # hyp_lens must be 1d for packed sequences, so we get rid of dim1..dim4
        hyp_lens = hyp_lens.view(-1)
        hyp = hyp.view(max_steps, -1)
        logits = logits.view(max_steps, -1, num_classes)
        hyp_lens, _ = hyp_lens.sort(descending=True)
    else:
        len_mask = len_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    hyp = hyp.masked_fill(len_mask == hyp_lens, eos)
    logits = logits.scatter(-1, hyp.unsqueeze(-1), 0.0)
    rand_mask = torch.randint_like(hyp, 2).eq(1)
    # > 0 to ensure that at least one valid value exists in the path
    rand_mask = rand_mask & (len_mask < hyp_lens) & (len_mask > 0)
    hyp = hyp.masked_fill(rand_mask, -1)
    padding_mask = (len_mask > hyp_lens) | rand_mask
    logits = logits.masked_fill(padding_mask.unsqueeze(-1), -float("inf"))
    if dim is None:
        hyp = torch.nn.utils.rnn.pack_padded_sequence(hyp, hyp_lens.cpu())
        logits = torch.nn.utils.rnn.pack_padded_sequence(logits, hyp_lens.cpu())
    elif dim:
        hyp_dim = (dim + 5) % 5
        hyp = hyp.transpose(0, hyp_dim).contiguous()
        logits = logits.transpose(0, hyp_dim).contiguous()
    log_probs = util.sequence_log_probs(logits, hyp, dim=dim, eos=eos)
    assert log_probs.eq(0.0).all()
    if dim is None:
        logits = torch.nn.utils.rnn.PackedSequence(
            torch.randn_like(logits[0]), logits[1]
        )
    else:
        logits = torch.randn_like(logits)
    log_probs = util.sequence_log_probs(logits, hyp, dim=dim, eos=eos)
    assert log_probs.ne(0.0).any()


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("gamma", [0.0, 0.95])
def test_time_distributed_return(device, batch_first, gamma):
    torch.manual_seed(290129)
    steps, batch_size = 1000, 30
    r = torch.randn(steps, batch_size, device=device)
    exp = torch.empty_like(r)
    exp[-1] = r[-1]
    for step in range(steps - 2, -1, -1):
        exp[step] = r[step] + gamma * exp[step + 1]
    if batch_first:
        r = r.t().contiguous()
        exp = exp.t().contiguous()
    act = util.time_distributed_return(r, gamma, batch_first=batch_first)
    assert torch.allclose(exp, act, atol=1e-5)


def test_polyharmonic_interpolation_linear(device):
    # when the order is 1, this should simply be linear interpolation
    x = torch.arange(3, device=device).unsqueeze(0).unsqueeze(-1).float()
    y = torch.tensor([[[0.0], [1.0], [0.0]]], device=device)
    y = torch.cat([y, 1.0 - y], 2)  # (1, 3, 2)
    q = torch.tensor([[[0.0], [0.5], [1.0], [1.6], [2.0]]], device=device)
    exp = torch.tensor(
        [[[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.4, 0.6], [0.0, 1.0]]], device=device
    )
    act = util.polyharmonic_spline(x, y, q, 1)
    assert torch.allclose(exp, act)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_polyharmonic_interpolation_equal_on_knots(order, device):
    torch.manual_seed(3487210)
    N, T, in_, out = 10, 11, 12, 13
    x = torch.rand(N, T, in_, device=device) * 2
    y = torch.rand(N, T, out, device=device) * 10.0 + 10
    act = util.polyharmonic_spline(x, y, x, order)
    # the high tolerance seems a numerical stability issue caused by polynomials in
    # the RBF
    assert torch.allclose(y, act, atol=1e-3), (y - act).abs().max()


@pytest.mark.parametrize("order", [1, 2, 3])
def test_polyharmonic_interpolation_matches_tensorflow(order, device):
    dir_ = os.path.join(os.path.dirname(__file__), "polyharmonic_spline")
    x = torch.tensor(np.load(os.path.join(dir_, "x.npy")), device=device)
    y = torch.tensor(np.load(os.path.join(dir_, "y.npy")), device=device)
    q = torch.tensor(np.load(os.path.join(dir_, "q.npy")), device=device)
    exp = torch.tensor(
        np.load(os.path.join(dir_, "o{}.npy".format(order))), device=device
    )
    act = util.polyharmonic_spline(x, y, q, order, full_matrix=True)
    assert torch.allclose(exp, act, atol=1e-3), (exp - act).abs().max()


@pytest.mark.parametrize("flip_h", [True, False])
@pytest.mark.parametrize("flip_w", [True, False])
def test_dense_image_warp_flow_flips(device, flip_h, flip_w):
    H, W = 30, 40
    img = torch.arange(H * W, dtype=torch.float32, device=device).view(1, 1, H, W)
    exp = img
    if flip_h:
        h = 2 * torch.arange(H, dtype=torch.float32, device=device) - H + 1
        exp = exp.flip(2)
    else:
        h = torch.zeros((H,), dtype=torch.float32, device=device)
    if flip_w:
        w = 2 * torch.arange(W, dtype=torch.float32, device=device) - W + 1
        exp = exp.flip(3)
    else:
        w = torch.zeros((W,), dtype=torch.float32, device=device)
    exp = exp.flatten()
    flow = torch.stack(torch.meshgrid(h, w), 2)
    act = util.dense_image_warp(img, flow).flatten()
    assert torch.allclose(exp, act, atol=1e-4), (exp - act).abs().max()
    act = util.dense_image_warp(img, flow, mode="nearest").flatten()
    assert torch.allclose(exp, act), (exp - act).abs().max()


def test_dense_image_warp_shift_right(device):
    torch.manual_seed(40462)
    N, C, H, W = 11, 20, 50, 19
    img = torch.rand(N, C, H, W, device=device)
    flow = torch.ones(N, H, W, 2, device=device)
    exp = img[..., :-1, :-1]
    act = util.dense_image_warp(img, flow)[..., 1:, 1:]
    assert torch.allclose(exp, act, atol=1e-5), (exp - act).abs().max()
    act = util.dense_image_warp(img, flow, mode="nearest")[..., 1:, 1:]
    assert torch.allclose(exp, act), (exp - act).abs().max()


@pytest.mark.parametrize("indexing", ["hw", "wh"])
def test_dense_image_warp_matches_tensorflow(device, indexing):
    dir_ = os.path.join(os.path.dirname(__file__), "dense_image_warp")
    img = torch.tensor(np.load(os.path.join(dir_, "img.npy")), device=device)
    flow = torch.tensor(np.load(os.path.join(dir_, "flow.npy")), device=device)
    if indexing == "wh":
        flow = flow.flip(-1)
    exp = torch.tensor(np.load(os.path.join(dir_, "warped.npy")), device=device)
    act = util.dense_image_warp(img, flow, indexing=indexing)
    assert torch.allclose(exp, act), (exp - act).abs().max()


@pytest.mark.parametrize("pinned_boundary_points", [0, 1, 2])
def test_sparse_image_warp_identity(device, pinned_boundary_points):
    torch.manual_seed(34207)
    N, C, H, W = 50, 12, 8, 3
    img = exp = torch.rand(N, C, H, W, device=device) * 255
    # we add 3 random control pointrs under the identity mapping to ensure a
    # non-degenerate interpolate
    src = dst = torch.rand(N, 3, 2, device=device) * min(H, W)
    act, flow = util.sparse_image_warp(
        img,
        src,
        dst,
        pinned_boundary_points=pinned_boundary_points,
        dense_interpolation_mode="nearest",
    )
    assert torch.allclose(flow, torch.tensor(0.0, device=device))
    assert torch.allclose(exp, act), (exp - act).abs().max()


@pytest.mark.parametrize("include_flow", [True, False])
@pytest.mark.parametrize("pinned_boundary_points", [0, 2])
def test_sparse_image_warp_matches_tensorflow(
    device, include_flow, pinned_boundary_points
):
    dir_ = os.path.join(os.path.dirname(__file__), "sparse_image_warp")
    img = torch.tensor(np.load(os.path.join(dir_, "img.npy")), device=device)
    src = torch.tensor(np.load(os.path.join(dir_, "src.npy")), device=device)
    dst = torch.tensor(np.load(os.path.join(dir_, "dst.npy")), device=device)
    exp_warped = torch.tensor(
        np.load(os.path.join(dir_, "warped_{}.npy".format(pinned_boundary_points))),
        device=device,
    )
    if include_flow:
        exp_flow = torch.tensor(
            np.load(os.path.join(dir_, "flow_{}.npy".format(pinned_boundary_points))),
            device=device,
        )
        act_warped, act_flow = util.sparse_image_warp(
            img, src, dst, pinned_boundary_points=pinned_boundary_points
        )
        assert torch.allclose(exp_flow, act_flow, atol=1e-3), (
            (exp_flow - act_flow).abs().max()
        )
    else:
        act_warped = util.sparse_image_warp(
            img,
            src,
            dst,
            pinned_boundary_points=pinned_boundary_points,
            include_flow=False,
        )
    assert torch.allclose(exp_warped, act_warped, atol=1e-3), (
        (exp_warped - act_warped).abs().max()
    )
