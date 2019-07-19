from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def test_beam_search_advance_steps(device):
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


@pytest.mark.parametrize('norm', [True, False])
def test_error_rate(device, norm):
    padding, eos = -1, 0
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
            (    1,     2,         3,     4,   ),
            (-1, 1, -1, 2, -1, -1, 3, -1, 5, -1),
            1,
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
    ref_lens = torch.tensor([len(x[0]) for x in pairs], device=device)
    hyp_lens = torch.tensor([len(x[1]) for x in pairs], device=device)
    ref = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x[0]) for x in pairs],
        padding_value=eos,
    ).to(device)
    hyp = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x[1]) for x in pairs],
        padding_value=eos,
    ).to(device)
    exp = torch.tensor([float(x[2]) for x in pairs], device=device)
    if norm:
        exp = torch.where(
            ref_lens == 0,
            hyp_lens.ne(0).float(),
            exp / ref_lens.float()
        )
    act = util.error_rate(
        ref, hyp, eos=eos, warn=False, norm=norm, padding=padding)
    assert torch.allclose(exp, act)
