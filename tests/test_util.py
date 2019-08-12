from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import chain

import torch
import pytest
import pydrobert.torch.util as util

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"


@pytest.mark.cpu
def test_beam_search_advance_greedy():
    torch.manual_seed(50)
    N, C, T = 30, 100, 25
    logits = torch.randn(T, N, C)
    greedy_logits, greedy_paths = torch.nn.functional.log_softmax(
        logits, -1).max(2)
    greedy_scores = greedy_logits.sum(0) + torch.log(torch.tensor(1 / C))
    y = None
    score = None
    for logit in logits:
        score, y, _ = util.beam_search_advance(logit, 1, score, y)
    score, y = score.squeeze(1), y.squeeze(2)
    assert torch.allclose(score, greedy_scores)
    assert torch.all(y == greedy_paths)


@pytest.mark.parametrize('prevent_eos', [True, False])
def test_beam_search_advance(device, prevent_eos):
    logits_1 = torch.tensor([
        [-10, -2, -4, -10],
        [-2, 0, 1, 1.1],
    ], device=device)  # ~[[x, 0, -4, x], [x, x, -.8, -.6]]
    score_1, y_1, s_1 = util.beam_search_advance(logits_1, 2)
    assert torch.all(y_1 == torch.tensor([[
        [1, 2],
        [3, 2],
    ]], device=device))
    assert torch.all(s_1 == 0)
    logits_2 = torch.tensor([
        [  # beam for batch 1
            [-5., -6, -7, -8.],
            [-400, -300, -200, -1],
        ],
        [  # beam for batch 2
            [2, 1, 1, 1],
            [-1, -2, -3, -4],
        ],
    ], device=device)  # ~ [[[-.4 -1.4 x x] [x x x 0]],
    #                      [[-.7 -1.7 x x] [-.4 -1.4 x x]]
    # batch 0: 0->0 0->1 win b/c 1->3 can't make up for score_1
    # batch 1: 1->0 0->0 win b/c score_1 about even
    score_2, y_2, s_2 = util.beam_search_advance(logits_2, 2, score_1)
    assert torch.all(y_2 == torch.tensor([[
        [0, 1],
        [0, 0],
    ]], device=device))
    assert torch.all(s_2 == torch.tensor([
        [0, 0],
        [1, 0],
    ], device=device))
    logits_3 = torch.tensor([
        [
            [1000., 0, 0, 0],
            [0, -100, 0, 0],
        ],
        [
            [0, 0, 0, 100],
            [2, 2, 1000, 10],
        ],
    ], device=device)  # ~ [[[0 x x x] [-1 -101 -1 -1]],
    #                      [[x x x 0] [x x 0 x]]
    # batch 0: 0->0 1->1 batch 1 done, but no priority b/c 0->0 very small
    # batch 1: 0->3 1->2
    score_3, y_3, s_3 = util.beam_search_advance(logits_3, 2, score_2, y_2, 1)
    assert torch.all(y_3 == torch.tensor([
        [  # y_2
            [0, 1],
            [0, 0],
        ],
        [
            [0, 1],
            [3, 2],
        ]
    ], device=device))
    assert torch.all(s_3 == torch.tensor([
        [0, 1],
        [0, 1],
    ], device=device))
    logits_4 = torch.tensor([
        [
            [1., 2, 3, 4],
            [5, 6, 7, 8],
        ],
        [
            [2, 2, 3, 2],
            [5, 6, 7, 8],
        ]
    ], device=device, requires_grad=True)
    # (note no eos condition)
    score_4, y_4, s_4 = util.beam_search_advance(logits_4, 1, score_3)
    assert torch.all(y_4.flatten() == torch.tensor([3, 3], device=device))
    assert torch.all(s_4.flatten() == torch.tensor([0, 1], device=device))
    g = torch.autograd.grad(
        [score_4], [logits_4], grad_outputs=torch.ones_like(score_4))[0]
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
            logits_t, W, score, y, eos=eos, lens=lens, prevent_eos=prevent_eos)
        logits_t = torch.randn(N, W, C).to(device)
        for i in range(W - 1):
            beam_i = y[..., i]
            for j in range(i + 1, W - 1):
                beam_j = y[..., j]
                for k in range(N):
                    assert not torch.all(beam_i[:, k] == beam_j[:, k])
    for bt, l in enumerate(lens):
        for bm in range(W):
            assert torch.all(y[l.item():, bt, bm] == eos)
            assert not torch.any(y[:l.item(), bt, bm] == eos)


@pytest.mark.parametrize('norm', [True, False])
@pytest.mark.parametrize('include_eos', [0, 1])
@pytest.mark.parametrize('batch_first', [True, False])
def test_error_rate(device, norm, include_eos, batch_first):
    eos = 0
    pairs = (
        (
            (1, 2, 3),
            (1, 2, 3),
            0,
        ), (
            (   2, 3),
            (1, 2, 3),
            1,
        ), (
            (1,    3),
            (1, 2, 3),
            1,
        ), (
            (      3,),
            (1, 2, 3),
            2,
        ), (
            (1, 2, 3),
            (1,    3),
            1,
        ), (
            (1, 2, 3),
            (1, 2,  ),
            1,
        ), (
            (1, 2, 3),
            (1,     ),
            2,
        ), (
            (1, 3, 1, 2, 3),
            (1, 2, 3),
            2,
        ), (
            (1, 2, 3),
            (4, 5, 6),
            3,
        ), (
            (2, 2, 2),
            (2,),
            2,
        ), (
            tuple(),
            (1,),
            1,
        ), (
            tuple(),
            tuple(),
            0,
        )
    )
    ref_lens = torch.tensor(
        [len(x[0]) + include_eos for x in pairs], device=device)
    hyp_lens = torch.tensor(
        [len(x[1]) + include_eos for x in pairs], device=device)
    ref = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x[0] + (eos,) * include_eos) for x in pairs],
        padding_value=eos, batch_first=batch_first,
    ).to(device)
    hyp = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x[1] + (eos,) * include_eos) for x in pairs],
        padding_value=eos, batch_first=batch_first,
    ).to(device)
    exp = torch.tensor([float(x[2]) for x in pairs], device=device)
    if norm:
        exp = torch.where(
            ref_lens == 0,
            hyp_lens.ne(0).float(),
            exp / ref_lens.float()
        )
    act = util.error_rate(
        ref, hyp, eos=eos, warn=False, norm=norm, include_eos=include_eos,
        batch_first=batch_first
    )
    assert torch.allclose(exp, act)


@pytest.mark.parametrize('include_eos', [True, False])
@pytest.mark.parametrize('batch_first', [True, False])
@pytest.mark.parametrize('exclude_last', [True, False])
def test_optimal_completion(device, include_eos, batch_first, exclude_last):
    eos, padding = ord('#'), -1
    triplets = (
        (
            'sunday#', 'saturday#',
            ['s', 'u', 'un', 'und', 'n', 'nd', 'a', 'y', '#', ''],
        ),
        (
            'sunday#', 'satrapy#',
            ['s', 'u', 'un', 'und', 'unda', 'y', 'y#', '#', ''],
        ),
        ('abc#', 'abc#', ['a', 'b', 'c', '#', '']),
        ('foot#', 'bot#', ['f', 'fo', 'o', 'ot#', '']),
        ('abc#', 'def#', ['a', 'ab', 'abc', 'abc#', '']),
    )
    ref = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor([ord(c) for c in word]) for (word, _, _) in triplets],
        batch_first=batch_first, padding_value=padding,
    ).to(device)
    hyp = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor([ord(c) for c in word]) for (_, word, _) in triplets],
        batch_first=batch_first, padding_value=eos,
    ).to(device)
    act = util.optimal_completion(
        ref, hyp, eos=eos, padding=padding, batch_first=batch_first,
        exclude_last=exclude_last, include_eos=include_eos,
    )
    if not batch_first:
        act = act.transpose(0, 1)  # (batch, hyp, ref)
    assert act.shape[0] == len(triplets)
    for act_bt, (_, _, exp_bt) in zip(act, triplets):
        if not include_eos:
            exp_bt = [nexts.replace('#', '') for nexts in exp_bt[:-1]]
        if exclude_last:
            exp_bt = exp_bt[:-1]
        assert act_bt.shape[0] >= len(exp_bt)
        assert torch.all(act_bt[len(exp_bt):].eq(padding))
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
    assert torch.allclose(stationary.sum(2), torch.tensor(1., device=device))
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
    g, = torch.autograd.grad([z], [logits_t], grad_outputs=torch.ones_like(z))
    assert g.ne(0.).any()
    y[..., :S // 2] = -1
    logits_t = logits_t.unsqueeze(1).expand_as(z)
    y, z = util.random_walk_advance(
        logits_t, S, y, eos=-1, include_relaxation=True)
    assert y[..., :S // 2].eq(-1).all()
    assert y[..., S // 2:].ne(-1).all()
    assert torch.isinf(-z[:, :S // 2]).all()
    assert not torch.isinf(-z[:, S // 2:]).any()
    g, = torch.autograd.grad([z], [logits_t], grad_outputs=torch.ones_like(z))
    assert g.ne(0.).any()
    assert g[:, :S // 2].eq(0.).all()
    y[-1] = -1
    y, z = util.random_walk_advance(
        logits_t, S, y, eos=-1, include_relaxation=True)
    # it should be defined, but zero
    g, = torch.autograd.grad([z], [logits_t], grad_outputs=torch.ones_like(z))
    assert g.eq(0.).all()


@pytest.mark.parametrize('prevent_eos', [True, False])
@pytest.mark.parametrize('lens', [True, False])
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
            assert torch.all(y[l.item():, bt, smp] == eos)
            assert not torch.any(y[:l.item(), bt, smp] == eos)


@pytest.mark.parametrize('exclude_last', [True, False])
@pytest.mark.parametrize('batch_first', [True, False])
@pytest.mark.parametrize('norm', [True, False])
@pytest.mark.parametrize('ins_cost', [1., .5])
def test_prefix_error_rates(
        device, exclude_last, batch_first, norm, ins_cost):
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
        eos=eos, include_eos=False, norm=norm, ins_cost=ins_cost,
        exclude_last=exclude_last, padding=padding, batch_first=batch_first,
        warn=False,
    )
    if batch_first:
        act = act.t().contiguous()
    exp = torch.empty(
        max_hyp_steps + (0 if exclude_last else 1), N, device=device)
    # if include_eos were true, `hyp` would get a bonus for the final `eos`
    # which isn't in its prefix
    for pref_len in range(max_hyp_steps - (1 if exclude_last else 0), -1, -1):
        hyp[pref_len:] = eos
        exp[pref_len] = util.error_rate(
            ref, hyp, eos=eos, include_eos=False, norm=norm,
            ins_cost=ins_cost, warn=False,
        )
    exp = exp.masked_fill(
        (
            torch.arange(exp.shape[0], device=device).unsqueeze(1) >=
            hyp_lens + (0 if exclude_last else 1)
        ),
        padding
    )
    assert torch.allclose(exp, act)


@pytest.mark.parametrize('dim', [0, 2, -1, None])
def test_sequence_log_probs(device, dim):
    torch.manual_seed(24519)
    max_steps, num_classes, eos = 30, 10, 0
    dim1, dim2, dim3, dim4 = 5, 2, 1, 3
    logits = torch.full(
        (max_steps, dim1, dim2, dim3, dim4, num_classes),
        -float('inf'), device=device)
    hyp = torch.randint(
        1, num_classes, (max_steps, dim1, dim2, dim3, dim4), device=device)
    hyp_lens = torch.randint(
        2, max_steps, (dim1, dim2, dim3, dim4), device=device)
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
    logits = logits.scatter(-1, hyp.unsqueeze(-1), 0.)
    rand_mask = torch.randint_like(hyp, 2).eq(1)
    # > 0 to ensure that at least one valid value exists in the path
    rand_mask = rand_mask & (len_mask < hyp_lens) & (len_mask > 0)
    hyp = hyp.masked_fill(rand_mask, -1)
    padding_mask = (len_mask > hyp_lens) | rand_mask
    logits = logits.masked_fill(padding_mask.unsqueeze(-1), -float('inf'))
    if dim is None:
        hyp = torch.nn.utils.rnn.pack_padded_sequence(hyp, hyp_lens)
        logits = torch.nn.utils.rnn.pack_padded_sequence(logits, hyp_lens)
    elif dim:
        hyp_dim = (dim + 5) % 5
        hyp = hyp.transpose(0, hyp_dim).contiguous()
        logits = logits.transpose(0, hyp_dim).contiguous()
    log_probs = util.sequence_log_probs(logits, hyp, dim=dim, eos=eos)
    assert log_probs.eq(0.).all()
    if dim is None:
        logits = torch.nn.utils.rnn.PackedSequence(
            torch.randn_like(logits[0]), logits[1])
    else:
        logits = torch.randn_like(logits)
    log_probs = util.sequence_log_probs(logits, hyp, dim=dim, eos=eos)
    assert log_probs.ne(0.).any()
