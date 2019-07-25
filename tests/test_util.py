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
def test_optimal_completion(device, include_eos, batch_first):
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
        ref, hyp, eos=eos, padding=padding, batch_first=batch_first)
    if not batch_first:
        act = act.transpose(0, 1)  # (batch, hyp, ref)
    assert act.shape[0] == len(triplets)
    for act_bt, (_, _, exp_bt) in zip(act, triplets):
        assert act_bt.shape[0] >= len(exp_bt)
        assert torch.all(act_bt[len(exp_bt):].eq(padding))
        for act_bt_hyp, exp_bt_hyp in zip(act_bt, exp_bt):
            act_bt_hyp = act_bt_hyp.masked_select(act_bt_hyp.ne(padding))
            act_bt_hyp = sorted(chr(i) for i in act_bt_hyp.tolist())
            assert sorted(exp_bt_hyp) == act_bt_hyp
