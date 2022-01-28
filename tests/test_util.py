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
import itertools

import torch
import pytest
import pydrobert.torch.util as util
import numpy as np

from pydrobert.torch._compat import meshgrid


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


def test_beam_search_advance_greedy(device):
    N, C, T = 30, 100, 25
    logits = torch.randn((T, N, C), device=device)
    greedy_logits, greedy_paths = logits.max(2)
    greedy_scores = greedy_logits.sum(0)
    y = torch.empty((0, N, 1), dtype=torch.long, device=device)
    log_probs = torch.zeros((N, 1), device=device)
    for logits_t in logits:
        y, _, log_probs, _ = util.beam_search_advance(
            logits_t.unsqueeze(1), 1, log_probs, y
        )
    y, log_probs = y.squeeze(2), log_probs.squeeze(1)
    assert torch.allclose(log_probs, greedy_scores)
    assert torch.all(y == greedy_paths)


@pytest.mark.parametrize(
    "probs,in_lens,max_exp,paths_exp",
    [
        (
            [
                [[0.5, 0.3, 0.2], [0.1, 0.2, 0.7], [0.1, 0.6, 0.3]],
                [[0.25, 0.25, 0.5], [0.25, 0.25, 0.5], [0.25, 0.25, 0.5]],
            ],
            None,
            [0.21, 0.125],
            [[0, 1], []],
        ),
        (
            [[[0.6, 0.4], [0.6, 0.4]], [[0.6, 0.4], [0.6, 0.4]]],
            [1, 0],
            [0.6, 1.0],
            [[0], []],
        ),
        (
            [
                [
                    [0.6, 0.3, 0.1],
                    [0.6, 0.3, 0.1],
                    [0.3, 0.1, 0.6],
                    [0.6, 0.3, 0.1],
                    [0.3, 0.6, 0.1],
                ]
            ],
            None,
            [0.6 ** 5],
            [[0, 0, 1]],
        ),
    ],
    ids=("A", "B", "C"),
)
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("is_probs", [True, False])
def test_ctc_greedy_search(
    probs, in_lens, max_exp, paths_exp, batch_first, is_probs, device
):
    probs = torch.tensor(probs, device=device)
    if in_lens is not None:
        in_lens = torch.tensor(in_lens, device=device)
    max_exp = torch.tensor(max_exp, device=device)
    out_lens_exp = torch.tensor([len(x) for x in paths_exp], device=device)
    paths_exp = tuple(
        torch.tensor(x, device=device, dtype=torch.long) for x in paths_exp
    )
    if not batch_first:
        probs = probs.transpose(0, 1)
    if not is_probs:
        max_exp = max_exp.log()
        probs = probs.log()
    max_act, paths_act, out_lens_act = util.ctc_greedy_search(
        probs, in_lens, batch_first=batch_first, is_probs=is_probs
    )
    assert max_exp.shape == max_act.shape
    assert torch.allclose(max_exp, max_act)
    assert out_lens_exp.shape == out_lens_act.shape
    assert (out_lens_exp == out_lens_act).all()
    assert paths_act.dim() == 2
    if not batch_first:
        paths_act = paths_act.t()
    for paths_exp_n, paths_act_n in zip(paths_exp, paths_act):
        assert (paths_exp_n == paths_act_n[: len(paths_exp_n)]).all()


@pytest.mark.parametrize("batch_first", [True, False])
def test_ctc_greedy_search_ignores_padding(device, batch_first):
    Tmax, N, V = 30, 51, 10
    lens = torch.randint(1, Tmax + 1, size=(N,), device=device)
    logits = torch.rand(N, Tmax, V + 1, device=device)
    max_a, out_paths_a, out_lens_a = [], [], []
    for logits_n, lens_n in zip(logits, lens):
        max_a_n, out_paths_a_n, out_lens_a_n = util.ctc_greedy_search(
            logits_n[:lens_n].unsqueeze(0 if batch_first else 1),
            lens_n.view(1),
            batch_first=batch_first,
        )
        max_a.append(max_a_n)
        out_paths_a.append(out_paths_a_n.squeeze(0 if batch_first else 1))
        out_lens_a.append(out_lens_a_n)
    if not batch_first:
        logits = logits.transpose(0, 1)
    max_a = torch.cat(max_a, dim=0)
    out_paths_a = torch.nn.utils.rnn.pad_sequence(out_paths_a, batch_first=batch_first)
    out_lens_a = torch.cat(out_lens_a, dim=0)
    max_b, out_paths_b, out_lens_b = util.ctc_greedy_search(
        logits, lens, batch_first=batch_first
    )
    assert (out_lens_a == out_lens_b).all(), (out_lens_a, out_lens_b)
    assert torch.allclose(max_a, max_b)
    if not batch_first:
        out_paths_a = out_paths_a.t()
        out_paths_b = out_paths_b.t()
    for out_path_a, out_path_b, len_ in zip(out_paths_a, out_paths_b, out_lens_a):
        assert (out_path_a[:len_] == out_path_b[:len_]).all()


@pytest.mark.parametrize(
    "probs_t,probs_prev,y_prev,y_next_exp,probs_next_exp,next_src_exp,"
    "next_is_nonext_exp",
    [
        (
            ([0.1, 0.7], 0.2),
            ([0.1, 0.4], [0.3, 0.2]),
            [[0], [1]],
            [[1], [0, 1], [1, 1], [0], [1, 0], [0, 0]],
            ([0.28, 0.28, 0.14, 0.01, 0.06, 0.03], [0.12, 0.0, 0.0, 0.08, 0.0, 0.0]),
            [1, 0, 1, 0, 1, 0],
            [True, False, False, True, False, False],
        ),
        (
            ([0.1, 0.2, 0.3], 0.4),
            ([0.0], [1.0]),
            [[]],
            [[], [2], [1], [0]],
            ([0.0, 0.3, 0.2, 0.1], [0.4, 0.0, 0.0, 0.0]),
            [0, 0, 0, 0],
            [True, False, False, False],
        ),
        (
            ([0.2, 0.3, 0.1], 0.4),
            ([0.1, 0.3, 0.5], [0.07, 0.11, 0.0]),
            [[0], [0, 1], [0, 1, 2]],
            [
                [0, 1],
                [0, 1, 2],
                [0, 1, 2, 1],
                [0, 1, 2, 0],
                [0],
                [0, 1, 0],
                [0, 1, 1],
                [0, 2],
                [0, 0],
                [0, 1, 2, 2],
            ],
            (
                [0.141, 0.091, 0.15, 0.1, 0.02, 0.082, 0.033, 0.017, 0.014, 0.0],
                [0.164, 0.2, 0.0, 0.0, 0.068, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            [1, 2, 2, 2, 0, 1, 1, 0, 0, 2],
            [True, True, False, False, True, False, False, False, False, False],
        ),
        (
            ([[0.1, 0.3], [0.2, 0.3]], [0.4, 0.5], 0.1),
            ([0.1, 0.3], [0.2, 0.0]),
            [[0], [1]],
            [[1], [0, 1], [0], [1, 0], [0, 0], [1, 1]],
            [[0.15, 0.09, 0.04, 0.06, 0.02, 0.0], [0.03, 0.0, 0.03, 0.0, 0.0, 0.0]],
            [1, 0, 0, 1, 0, 1],
            [True, False, True, False, False, False],
        ),
    ],
    ids=("A", "B", "C", "D"),
)
@pytest.mark.parametrize("batch_size", [1, 2, 7])
def test_ctc_prefix_search_advance(
    probs_t,
    probs_prev,
    y_prev,
    y_next_exp,
    probs_next_exp,
    next_src_exp,
    next_is_nonext_exp,
    batch_size,
    device,
):
    Kp, K, N = len(y_prev), len(y_next_exp), batch_size

    y_prev_lens = torch.tensor([[len(x) for x in y_prev]] * N, device=device)
    assert y_prev_lens.shape == (N, Kp)

    y_prev_last = torch.tensor([[x[-1] if x else 0 for x in y_prev]] * N, device=device)
    assert y_prev_last.shape == (N, Kp)

    prev_is_prefix = []
    for k, kp in itertools.product(range(Kp), repeat=2):
        prev_is_prefix.append(y_prev[k] == y_prev[kp][: len(y_prev[k])])
    prev_is_prefix = (
        torch.tensor(prev_is_prefix, device=device).view(1, Kp, Kp).expand(N, Kp, Kp)
    )

    y_prev = (
        torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long, device=device) for x in y_prev],
            batch_first=False,
            padding_value=0,
        )
        .unsqueeze(1)
        .expand(-1, N, Kp)
    )
    assert y_prev.dtype == torch.long

    if len(probs_t) == 2:
        # the usual CTC setup w/o fusion: extension probabilities are the same as
        # non-extension probabilities
        probs_t = ([probs_t[0]] * Kp, probs_t[0], probs_t[1])
    probs_t = tuple(torch.stack([torch.tensor(x, device=device)] * N) for x in probs_t)
    V = probs_t[0].size(2)
    assert probs_t[0].shape == (N, Kp, V)
    assert probs_t[1].shape == (N, V)
    assert probs_t[2].shape == (N,)

    probs_prev = tuple(
        torch.tensor(x, device=device).unsqueeze(0).expand(N, Kp) for x in probs_prev
    )

    y_next_lens_exp = torch.tensor([[len(x) for x in y_next_exp]] * N, device=device)
    assert y_next_lens_exp.shape == (N, K)

    y_next_last_exp = torch.tensor(
        [[x[-1] if x else 0 for x in y_next_exp]] * N, device=device
    )
    assert y_next_last_exp.shape == (N, K)

    next_is_prefix_exp = []
    for k, kp in itertools.product(range(K), repeat=2):
        next_is_prefix_exp.append(y_next_exp[k] == y_next_exp[kp][: len(y_next_exp[k])])
    next_is_prefix_exp = (
        torch.tensor(next_is_prefix_exp, device=device).view(1, K, K).expand(N, K, K)
    )

    y_next_exp = tuple(
        torch.tensor(x, device=device).unsqueeze(1).expand(-1, N) for x in y_next_exp
    )

    probs_next_exp = tuple(
        torch.tensor(x, device=device).unsqueeze(0).expand(N, K) for x in probs_next_exp
    )

    next_src_exp = torch.tensor(next_src_exp, device=device).unsqueeze(0).expand(N, K)
    next_is_nonext_exp = (
        torch.tensor(next_is_nonext_exp, device=device).unsqueeze(0).expand(N, K)
    )

    (
        y_next_act,
        y_next_last_act,
        y_next_lens_act,
        probs_next_act,
        next_is_prefix_act,
        next_src_act,
        next_is_nonext_act,
    ) = util.ctc_prefix_search_advance(
        probs_t, K, probs_prev, y_prev, y_prev_last, y_prev_lens, prev_is_prefix
    )

    assert y_next_lens_exp.shape == y_next_lens_act.shape
    assert (y_next_lens_exp == y_next_lens_act).all(), (
        y_next_lens_exp,
        y_next_lens_act,
    )

    assert y_next_act.dim() == 3
    assert y_next_act.shape[1:] == (N, K)
    for k in range(K):
        y_next_exp_k = y_next_exp[k]
        y_next_act_k = y_next_act[: y_next_exp_k.size(0), :, k]
        assert (y_next_exp_k == y_next_act_k).all()

    assert y_next_last_act.shape == y_next_last_exp.shape
    assert (y_next_last_act == y_next_last_exp).all()

    for probs_next_act_i, probs_next_exp_i in zip(probs_next_act, probs_next_exp):
        assert probs_next_act_i.shape == probs_next_exp_i.shape
        assert torch.allclose(probs_next_act_i, probs_next_exp_i)

    assert next_is_prefix_act.shape == next_is_prefix_exp.shape
    assert (next_is_prefix_act == next_is_prefix_exp).all()

    assert next_src_exp.shape == next_src_act.shape
    assert (next_src_act == next_src_exp).all()

    assert next_is_nonext_act.shape == next_is_nonext_exp.shape
    assert (next_is_nonext_act == next_is_nonext_exp).all()


def test_ctc_prefix_search_advance_big_width(device):
    # ensure a large beam width produces zero-probability paths past the possible
    # number of true paths
    T, N, V, Kp = 100, 10, 20, 5
    K = (V + 1) * Kp * 2
    assert V >= Kp
    # all nonzero
    ext_probs_t = torch.rand((N, Kp, V), device=device) + 0.01
    nonext_probs_t = torch.rand((N, V), device=device) + 0.01
    blank_probs_t = torch.rand((N,), device=device) + 0.01
    nb_probs_prev = torch.rand((N, Kp), device=device) + 0.01
    b_probs_prev = torch.rand((N, Kp), device=device) + 0.01
    # the first path will be a strict prefix of every other path. The remaining paths
    # will extend that first path by one different token each (they are not prefixes of
    # one another)
    y_prev = torch.randint(1, V + 1, (T - 1, N, 1), device=device).expand(T - 1, N, Kp)
    y_prev = torch.cat(
        [y_prev, torch.arange(Kp, device=device).view(1, 1, Kp).expand(1, N, Kp)], 0
    )
    y_prev_lens = torch.tensor([[T - 1] + [T] * (Kp - 1)] * N, device=device)
    y_prev_last = y_prev.gather(0, (y_prev_lens - 1).unsqueeze(0)).squeeze(0)
    prev_is_prefix = torch.eye(Kp, device=device, dtype=torch.bool)
    prev_is_prefix[0] = True  # first element is a prefix of everyone else
    prev_is_prefix = prev_is_prefix.unsqueeze(0).expand(N, Kp, Kp)
    (
        y_next,
        y_next_last,
        y_next_lens,
        (nb_probs_next, b_probs_next),
        next_is_prefix,
        next_src,
        next_is_nonext,
    ) = util.ctc_prefix_search_advance(
        (ext_probs_t, nonext_probs_t, blank_probs_t),
        K,
        (b_probs_prev, nb_probs_prev),
        y_prev,
        y_prev_last,
        y_prev_lens,
        prev_is_prefix,
    )
    assert y_next.shape == (T + 1, N, K)
    assert y_next_last.shape == (N, K)
    assert y_next_lens.shape == (N, K)
    assert nb_probs_next.shape == (N, K)
    assert b_probs_next.shape == (N, K)
    assert next_is_prefix.shape == (N, K, K)
    assert next_src.shape == (N, K)
    assert next_is_nonext.shape == (N, K)
    # the first prefix can be extended in V ways and stay the same in 1 way. Kp - 1 of
    # the extensions already exist in the beam, so the first prefix leads to V - Kp + 2
    # new prefixes. The remaining Kp - 1 prefixes can be extended in V ways and stay the
    # same 1 way.
    exp_num_valid_prefixes = (V - Kp + 2) + (Kp - 1) * (V + 1)
    nb_probs_next_is_valid = nb_probs_next >= 0.0
    assert nb_probs_next_is_valid[:, :exp_num_valid_prefixes].all()
    assert not nb_probs_next_is_valid[:, exp_num_valid_prefixes:].any(), (
        nb_probs_next_is_valid.sum(1),
        exp_num_valid_prefixes,
    )
    assert (b_probs_next >= 0.0)[:, :exp_num_valid_prefixes].all()
    # the corresponding blank probabilities may not be invalidated (it's not worth the
    # extra computation when the end-user just sees the total probability), but they
    # should be invalid or nonzero
    assert (b_probs_next <= 0.0)[:, exp_num_valid_prefixes:].all()
    # Each of these paths has a non-blank probability mass because they are
    # either extensions or had some non-blank probability mass already.
    # the only prefixes that have nonzero blank probality are the ones that aren't
    # extensions. There are Kp such unique prefixes
    nb_probs_next = nb_probs_next[:, :exp_num_valid_prefixes]
    b_probs_next = b_probs_next[:, :exp_num_valid_prefixes]
    assert ((nb_probs_next != 0.0).sum(1) == exp_num_valid_prefixes).all()
    assert ((b_probs_next != 0.0).sum(1) == Kp).all()


def test_beam_search_advance(device):
    N, V, Kp, num_contours = 1, 256, 16, 5
    assert V > Kp
    N_ = torch.arange(N, device=device)
    Kp_ = torch.arange(Kp, device=device)
    V_ = torch.arange(V, device=device)
    log_probs_t = (N_.view(N, 1, 1) + Kp_.view(Kp, 1) + V_).float()
    # likewise for log_probs_prev
    log_probs_prev = (N_.view(N, 1) + Kp_).float()

    max_sum_lpt = (V - 1) + (Kp - 1)
    max_sum_lpn = max_sum_lpt + (Kp - 1)
    # the log_probs_prev count and log_probs_t count are the same along the Kp index.
    # there's a smarter way to do this, but I can't be arsed
    lpn_num_ties = [0 for _ in range(max_sum_lpn + 1)]
    for kp in range(Kp):
        for v in range(V):
            lpn_num_ties[2 * kp + v] += 1
    lpn_num_ties = lpn_num_ties[-num_contours:]
    K = sum(lpn_num_ties)
    log_probs_next_exp = list(
        itertools.chain(
            *(
                itertools.repeat(max_sum_lpn - i, num)
                for (i, num) in enumerate(reversed(lpn_num_ties))
            )
        )
    )
    assert len(log_probs_next_exp) == K
    log_probs_next_exp = (
        2 * torch.arange(N, device=device).view(N, 1)
        + torch.tensor(log_probs_next_exp, device=device)
    ).float()
    # set the length of each path to path idx. idx - 1 is a strict prefix of idx. The
    # chosen paths are also the longest
    y_prev = torch.arange(V, device=device).view(V, 1, 1).expand(V, N, Kp)
    y_prev_lens = torch.arange(Kp, device=device).unsqueeze(0).expand(N, Kp)
    # the highest probability path is unique; we can compute its expected values
    y_next_0_exp = torch.arange(Kp, device=device)
    y_next_0_exp[-1] = V - 1
    y_next_0_exp = y_next_0_exp.unsqueeze(1).expand(Kp, N)
    y_next_lens_0_exp = Kp
    next_src_0_exp = Kp - 1
    (
        y_next_act,
        y_next_lens_act,
        log_probs_next_act,
        next_src_act,
    ) = util.beam_search_advance(log_probs_t, K, log_probs_prev, y_prev, y_prev_lens)
    assert torch.allclose(log_probs_next_exp, log_probs_next_act)
    assert (y_next_lens_act[:, 0] == y_next_lens_0_exp).all()
    assert (next_src_act[:, 0] == next_src_0_exp).all()
    assert (y_next_act[:Kp, :, 0] == y_next_0_exp).all()


@pytest.mark.parametrize("ins_cost", [2.0, 0.5, 1.0], ids=("i2.0", "i0.5", "i1.0"))
@pytest.mark.parametrize("del_cost", [2.0, 0.5, 1.0], ids=("d2.0", "d0.5", "d1.0"))
@pytest.mark.parametrize("sub_cost", [2.0, 0.5, 1.0], ids=("s2.0", "s0.5", "s1.0"))
@pytest.mark.parametrize("distance", [True, False], ids=("edit", "rate"))
@pytest.mark.parametrize("ref_bigger", [True, False])
def test_error_rate_against_simple_impl(
    device, ins_cost, del_cost, sub_cost, ref_bigger, distance
):
    torch.manual_seed(2502)
    hyp_steps, ref_steps, batch_size, num_classes = 10, 9, 10, 10
    eps = 1e-4
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
    edit_matrix = torch.empty(hyp_steps + 1, ref_steps + 1, batch_size, device=device)
    edit_matrix[0] = torch.arange(float(ref_steps + 1), device=device).unsqueeze(-1)
    edit_matrix[:, 0] = torch.arange(float(hyp_steps + 1), device=device).unsqueeze(-1)
    for hyp_idx in range(1, hyp_steps + 1):
        for ref_idx in range(1, ref_steps + 1):
            neq_mask = (ref[ref_idx - 1] != hyp[hyp_idx - 1]).float()
            sub_align = cost_matrix[hyp_idx - 1, ref_idx - 1] + sub_cost * neq_mask
            ins_align = cost_matrix[hyp_idx - 1, ref_idx] + ins_cost + eps
            del_align = cost_matrix[hyp_idx, ref_idx - 1] + del_cost + eps
            cur_costs, argmin = torch.stack([sub_align, ins_align, del_align]).min(0)
            cur_costs -= argmin.gt(0) * eps
            cost_matrix[hyp_idx, ref_idx] = cur_costs
            sub_count = edit_matrix[hyp_idx - 1, ref_idx - 1] + neq_mask
            ins_count = edit_matrix[hyp_idx - 1, ref_idx] + 1
            del_count = edit_matrix[hyp_idx, ref_idx - 1] + 1
            cur_counts = (
                torch.stack([sub_count, ins_count, del_count])
                .gather(0, argmin.unsqueeze(0))
                .squeeze(0)
            )
            edit_matrix[hyp_idx, ref_idx] = cur_counts
    if ins_cost == del_cost == sub_cost == 1:
        assert torch.allclose(cost_matrix, edit_matrix)
    if distance:
        exp = cost_matrix[-1, -1]
        func = util.edit_distance
    else:
        exp = edit_matrix[-1, -1]
        func = util.error_rate
    act = func(
        ref,
        hyp,
        norm=False,
        ins_cost=ins_cost,
        del_cost=del_cost,
        sub_cost=sub_cost,
        warn=False,
    )
    assert torch.allclose(exp, act)


@pytest.mark.parametrize("ins_cost", [0.5, 1.0], ids=("i0.5", "i1.0"))
@pytest.mark.parametrize("del_cost", [0.5, 1.0], ids=("d0.5", "d1.0"))
@pytest.mark.parametrize("sub_cost", [0.5, 1.0], ids=("s0.5", "s1.0"))
@pytest.mark.parametrize("norm", [True, False], ids=("normed", "unnormed"))
@pytest.mark.parametrize("distance", [True, False], ids=("edit", "rate"))
def test_error_rate_ignores_padding(
    device, ins_cost, del_cost, sub_cost, norm, distance
):
    N, Tmax, V, eos = 11, 50, 5, -1
    ref_lens = torch.randint(Tmax, size=(N,), device=device)
    refs = [torch.randint(V, size=(len_.item(),), device=device) for len_ in ref_lens]
    hyp_lens = torch.randint(Tmax, size=(N,), device=device)
    hyps = [torch.randint(V, size=(len_.item(),), device=device) for len_ in hyp_lens]
    if distance:
        func = util.edit_distance
    else:
        func = util.error_rate
    out_a = []
    for ref, hyp in zip(refs, hyps):
        out_a.append(
            func(
                ref.unsqueeze(1),
                hyp.unsqueeze(1),
                norm=norm,
                ins_cost=ins_cost,
                del_cost=del_cost,
                sub_cost=sub_cost,
                warn=False,
            )
        )
    out_a = torch.cat(out_a, 0)
    assert out_a.dim() == 1 and out_a.size(0) == N
    refs = torch.nn.utils.rnn.pad_sequence(refs, padding_value=eos)
    hyps = torch.nn.utils.rnn.pad_sequence(hyps, padding_value=eos)
    out_b = func(
        refs,
        hyps,
        eos=eos,
        norm=norm,
        ins_cost=ins_cost,
        del_cost=del_cost,
        sub_cost=sub_cost,
        warn=False,
    )
    assert torch.allclose(out_a, out_b)


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
    flow = torch.stack(meshgrid(h, w), 2)
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

