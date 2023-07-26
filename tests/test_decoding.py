# Copyright 2022 Sean Robertson
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

import itertools

import torch
import pytest
import time

from typing import Dict, Tuple

from pydrobert.torch.modules import (
    BeamSearch,
    CTCGreedySearch,
    CTCPrefixSearch,
    MixableSequentialLanguageModel,
    MixableShallowFusionLanguageModel,
    RandomWalk,
    SequenceLogProbabilities,
    SequentialLanguageModel,
)
from pydrobert.torch.functional import (
    beam_search_advance,
    ctc_prefix_search_advance,
    fill_after_eos,
)
from pydrobert.torch.distributions import SequentialLanguageModelDistribution


class RNNLM(MixableSequentialLanguageModel):
    def __init__(self, vocab_size, embed_size=128, hidden_size=512):
        super().__init__(vocab_size)
        self.hidden_size = hidden_size
        self.embed = torch.nn.Embedding(
            vocab_size + 1, embed_size, padding_idx=vocab_size
        )
        self.cell = torch.nn.LSTMCell(embed_size, hidden_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size)
        self.lstm.weight_ih_l0 = self.cell.weight_ih
        self.lstm.weight_hh_l0 = self.cell.weight_hh
        self.lstm.bias_ih_l0 = self.cell.bias_ih
        self.lstm.bias_hh_l0 = self.cell.bias_hh
        self.ff = torch.nn.Linear(hidden_size, vocab_size)

    @torch.jit.export
    def extract_by_src(
        self, prev: Dict[str, torch.Tensor], src: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            "hidden": prev["hidden"].index_select(0, src),
            "cell": prev["cell"].index_select(0, src),
        }

    @torch.jit.export
    def mix_by_mask(
        self,
        prev_true: Dict[str, torch.Tensor],
        prev_false: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mask = mask.unsqueeze(1)
        return {
            "hidden": torch.where(mask, prev_true["hidden"], prev_false["hidden"]),
            "cell": torch.where(mask, prev_true["cell"], prev_false["cell"]),
        }

    @torch.jit.export
    def update_input(
        self, prev: Dict[str, torch.Tensor], hist: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if len(prev):
            return prev
        N = hist.size(1)
        zeros = self.ff.weight.new_zeros((N, self.hidden_size))
        return {"hidden": zeros, "cell": zeros}

    @torch.jit.export
    def calc_full_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        hist = torch.cat([hist.new_full((1, hist.size(1)), self.vocab_size), hist], 0)
        x = self.embed(hist)
        x = self.lstm(x)[0]
        logits = self.ff(x)
        return torch.nn.functional.log_softmax(logits, -1)

    @torch.jit.export
    def calc_idx_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor], idx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        idx_zero = idx == 0
        if idx_zero.all():
            x = torch.full(
                (hist.size(1),), self.vocab_size, dtype=hist.dtype, device=hist.device
            )
        else:
            x = hist.gather(
                0, (idx - 1).expand(hist.shape[1:]).clamp(min=0).unsqueeze(0)
            ).squeeze(
                0
            )  # (N,)
            x = x.masked_fill(idx_zero.expand(x.shape), self.vocab_size)
        x = self.embed(x)
        h_1, c_1 = self.cell(x, (prev["hidden"], prev["cell"]))
        logits = self.ff(h_1)
        return (
            torch.nn.functional.log_softmax(logits, -1),
            {"hidden": h_1, "cell": c_1},
        )

    @torch.jit.unused
    def train(self, iters: int = 1, len_: int = 100, batch: int = 10):
        # a convenience method for training a little bit. Avoids too-similar values
        optim = torch.optim.Adam(self.parameters())
        hist = torch.randint(
            0, self.vocab_size, (len_, batch), device=self.lstm.weight_ih_l0.device
        )
        for iter_ in range(iters):
            optim.zero_grad()
            logits = self(hist[:-1])
            loss = torch.nn.CrossEntropyLoss()(logits.flatten(0, 1), hist.flatten())
            loss.backward()
            optim.step()


class MyBigramLM(MixableSequentialLanguageModel):

    bigram_table: torch.Tensor

    def __init__(self):
        super().__init__(2)
        self.register_buffer(
            "bigram_table",
            torch.tensor(
                [
                    [1.0, 0.0],  # P(0|<s>), P(1|<s>)
                    [0.5, 0.5],  # P(0|0), P(1|0)
                    [0.0, 1.0],  # P(0|1), P(1|1)
                ]
            ).log(),
        )

    def extract_by_src(self, in_prev, src):
        return in_prev

    def mix_by_mask(self, in_prev_true, in_prev_false, mask):
        return in_prev_true

    def calc_idx_log_probs(self, hist, prev, idx):
        # note we shift + 1 to make room for <s>
        idx_zero = idx == 0
        if idx_zero.all():
            x = hist.new_full((hist.size(1),), 0)
        else:
            x = (
                hist.gather(0, (idx - 1).clamp(min=0).unsqueeze(0)).squeeze(0) + 1
            )  # (N,)
            x = x.masked_fill(idx_zero, 0)
        return self.bigram_table.gather(0, x.unsqueeze(1).expand(-1, 2)), None


def test_ctc_prefix_search(device):

    T, N, K, V = 3, 128, 2, 3
    logits = (
        torch.tensor(
            [[1 / 2, 1 / 3, 1 / 6], [1 / 3, 1 / 6, 1 / 2], [1 / 6, 1 / 2, 1 / 3],]
        )
        .log()
        .unsqueeze(1)
        .expand(T, N, V)
        .to(device)
    )
    exps = [
        (0.0, [[0, 1], [0]], [5 / 24, 1 / 6]),
        (1.0, [[0], [0, 1]], [5 / 24, 17 / 144]),
    ]
    lm = MyBigramLM().to(device)
    for beta, y_exp, probs_exp in exps:
        search = CTCPrefixSearch(K, beta, lm)
        y_act, y_lens_act, probs_act = search(logits)
        assert y_act.shape == (T, N, K)
        assert y_lens_act.shape == (N, K)
        assert probs_act.shape == (N, K)
        for y_k_exp, probs_k_exp, y_k_act, y_lens_k_act, probs_k_act in zip(
            y_exp, probs_exp, y_act.transpose(0, 2), y_lens_act.t(), probs_act.t()
        ):
            Tp = len(y_k_exp)
            assert (y_lens_k_act == Tp).all()
            y_k_exp = torch.tensor(y_k_exp, device=device).unsqueeze(0)  # (1, Tp)
            y_k_act = y_k_act[:, :Tp]  # (N, Tp)
            assert (y_k_act == y_k_exp).all()
            probs_k_exp = torch.tensor(probs_k_exp, device=device).unsqueeze(0)
            assert torch.allclose(probs_k_exp, probs_k_act)


@pytest.mark.parametrize("shallow_fusion", [True, False], ids=["fusion", "nofusion"])
def test_ctc_prefix_search_batch(device, jit_type, shallow_fusion):
    if jit_type == "trace":
        pytest.xfail("trace unsupported for CTCPrefixSearch")
    T, N, V, K = 31, 32, 33, 5
    assert K <= V
    lm = RNNLM(V)
    lm.train()
    if jit_type == "script":
        if shallow_fusion:
            pytest.xfail("script unsupported for shallow_fusion")
        lm = torch.jit.script(lm)
    if shallow_fusion:
        lm2 = RNNLM(V)
        lm2.train()
        lm = MixableShallowFusionLanguageModel(lm, lm2)
    search = CTCPrefixSearch(K, lm=lm).to(device)
    if jit_type == "script":
        search = torch.jit.script(search)
    logits = torch.randn((T, N, V + 1), device=device)
    lens = torch.randint(0, T, (N,), device=device)

    exps = []
    for logits_n, lens_n in zip(logits.transpose(0, 1), lens):
        logits_n = logits_n[:lens_n].unsqueeze(1)
        lens_n = lens_n.view(1)
        y_n_exp, y_lens_n_exp, probs_n_exp = search(logits_n, lens_n)
        y_n_exp = y_n_exp.squeeze(1)  # (T_n, K_n)
        y_lens_n_exp = y_lens_n_exp.squeeze(0)  # (K_n,)
        probs_n_exp = probs_n_exp.squeeze(0)  # (K_n,)
        valid_prefix_mask_n_exp = probs_n_exp >= 0.0
        if not valid_prefix_mask_n_exp.all():
            assert not lens_n
            assert y_lens_n_exp[0] == 0
        else:
            assert (y_lens_n_exp <= lens_n).all()
        exps.append((y_n_exp, y_lens_n_exp, probs_n_exp))

    y_act, y_lens_act, probs_act = search(logits, lens)
    for (y_n_exp, y_lens_n_exp, probs_n_exp), y_n_act, y_lens_n_act, probs_n_act in zip(
        exps, y_act.transpose(0, 1), y_lens_act, probs_act
    ):
        assert y_n_exp.shape[1:] == y_n_act.shape[1:]
        assert y_n_exp.size(0) <= y_n_act.size(0)
        assert y_lens_n_exp.shape == y_lens_n_act.shape
        assert probs_n_exp.shape == probs_n_act.shape
        valid_prefix_mask_n_exp = probs_n_exp >= 0.0
        valid_prefix_mask_n_act = probs_n_act >= 0.0
        assert (valid_prefix_mask_n_exp == valid_prefix_mask_n_act).all()
        if not valid_prefix_mask_n_exp.all():
            assert valid_prefix_mask_n_exp.sum() == 1  # only one valid path: empty one
            y_n_exp, y_n_act = y_n_exp[:, :1], y_n_act[:, :1]
            y_lens_n_exp, y_lens_n_act = y_lens_n_exp[:1], y_lens_n_act[:1]
            probs_n_exp, probs_n_act = probs_n_exp[:1], probs_n_act[:1]
        assert (y_lens_n_exp == y_lens_n_act).all()
        assert torch.allclose(probs_n_exp, probs_n_act)
        rem = y_n_act.size(0) - y_n_exp.size(0)
        if rem > 0:
            y_n_exp = torch.cat([y_n_exp, y_n_exp.new_empty((rem, y_n_exp.size(1)))], 0)
            assert y_n_exp.shape == y_n_act.shape
        len_mask = (
            torch.arange(y_n_exp.size(0), device=device).unsqueeze(1) >= y_lens_n_exp
        )
        y_n_exp = y_n_exp.masked_fill_(len_mask, -1)
        y_n_act = y_n_act.masked_fill_(len_mask, -1)
        assert (y_n_exp == y_n_act).all()

    del lm


def test_beam_search_advance_greedy(device):
    N, C, T = 30, 20, 25
    logits = torch.randn((T, N, C), device=device)
    greedy_logits, greedy_paths = logits.max(2)
    greedy_scores = greedy_logits.sum(0)
    y = torch.empty((0, N, 1), dtype=torch.long, device=device)
    log_probs = torch.zeros((N, 1), device=device)
    for logits_t in logits:
        y, _, log_probs, _ = beam_search_advance(logits_t.unsqueeze(1), 1, log_probs, y)
    y, log_probs = y.squeeze(2), log_probs.squeeze(1)
    assert torch.allclose(log_probs, greedy_scores)
    assert torch.all(y == greedy_paths)


# @pytest.mark.parametrize("finish_all_paths", ["all", "first"])
# def test_beam_search_batch(device, jit_type, finish_all_paths):
#     if jit_type == "trace":
#         pytest.xfail("trace unsupported for BeamSearch")
#     T, N, V, K = 12, 4, 32, 8
#     assert K <= V and N * K <= V

#     lm = RNNLM(V).to(device)
#     lm.train()

#     initial_state = {
#         "hidden": torch.randn((N, lm.hidden_size), device=device),
#         "cell": torch.randn((N, lm.hidden_size), device=device),
#     }
#     if jit_type == "script":
#         lm = torch.jit.script(lm)
#     search = BeamSearch(lm, K, eos=0, pad_value=-1, finish_all_paths=finish_all_paths)
#     if jit_type == "script":
#         search = torch.jit.script(search)

#     y_exp, y_lens_exp, log_probs_exp = search(initial_state, N, T)
#     y_exp = fill_after_eos(y_exp, 0, fill=-1)
#     assert y_exp.device == y_lens_exp.device == log_probs_exp.device == device
#     assert y_exp.shape[1:] == y_lens_exp.shape == log_probs_exp.shape == (N, K)
#     assert not torch.allclose(log_probs_exp, log_probs_exp[:1])

#     for n in range(N):
#         y_exp_n, y_lens_exp_n = y_exp[:, n], y_lens_exp[n]
#         log_probs_exp_n = log_probs_exp[n]
#         initial_state_n = dict((k, v[n : n + 1]) for (k, v) in initial_state.items())
#         y_act_n, y_lens_act_n, log_probs_act_n = search(initial_state_n, max_iters=T)
#         assert y_act_n.shape[1:] == y_lens_act_n.shape == log_probs_act_n.shape == (K,)
#         assert (y_lens_exp_n == y_lens_act_n).all(), n
#         assert torch.allclose(log_probs_exp_n, log_probs_act_n), n
#         y_act_n = fill_after_eos(y_act_n, 0, fill=-1)
#         y_exp_n, y_act_n = y_exp_n[y_exp_n != -1], y_act_n[y_act_n != -1]
#         assert y_exp_n.numel() == y_act_n.numel(), n
#         assert (y_exp_n == y_act_n).all(), n

#     del lm


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
    probs, in_lens, max_exp, paths_exp, batch_first, is_probs, device, jit_type
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
    ctc_greedy_search = CTCGreedySearch(batch_first=batch_first, is_probs=is_probs)
    if jit_type == "script":
        ctc_greedy_search = torch.jit.script(ctc_greedy_search)
    elif jit_type == "trace":
        ctc_greedy_search = torch.jit.trace(
            ctc_greedy_search,
            (torch.empty(1, 1, 1),)
            + (tuple() if in_lens is None else (torch.ones((1), dtype=torch.long),)),
        )
    if in_lens is None:
        max_act, paths_act, out_lens_act = ctc_greedy_search(probs)
    else:
        max_act, paths_act, out_lens_act = ctc_greedy_search(probs, in_lens)
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
    ctc_greedy_search = CTCGreedySearch(batch_first=batch_first)
    for logits_n, lens_n in zip(logits, lens):
        max_a_n, out_paths_a_n, out_lens_a_n = ctc_greedy_search(
            logits_n[:lens_n].unsqueeze(0 if batch_first else 1), lens_n.view(1),
        )
        max_a.append(max_a_n)
        out_paths_a.append(out_paths_a_n.squeeze(0 if batch_first else 1))
        out_lens_a.append(out_lens_a_n)
    if not batch_first:
        logits = logits.transpose(0, 1)
    max_a = torch.cat(max_a, dim=0)
    out_paths_a = torch.nn.utils.rnn.pad_sequence(out_paths_a, batch_first=batch_first)
    out_lens_a = torch.cat(out_lens_a, dim=0)
    max_b, out_paths_b, out_lens_b = ctc_greedy_search(logits, lens)
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
    ) = ctc_prefix_search_advance(
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
    ) = ctc_prefix_search_advance(
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
    ) = beam_search_advance(log_probs_t, K, log_probs_prev, y_prev, y_prev_lens)
    assert torch.allclose(log_probs_next_exp, log_probs_next_act)
    assert (y_next_lens_act[:, 0] == y_next_lens_0_exp).all()
    assert (next_src_act[:, 0] == next_src_0_exp).all()
    assert (y_next_act[:Kp, :, 0] == y_next_0_exp).all()


class StationaryLM(SequentialLanguageModel):
    def __init__(self, vocab_size: int):
        super().__init__(vocab_size)
        self.logits = torch.nn.Parameter(torch.randn(vocab_size, vocab_size))

    def calc_idx_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor], idx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if idx == 0:
            hist = torch.zeros(1, hist.size(1), device=hist.device, dtype=hist.dtype)
        return self.logits.index_select(0, hist[(idx - 1).clamp_min_(0)]), prev


def test_random_walk(device):

    T, V = 10000, 4
    lm = StationaryLM(V).to(device)
    stationary = lm.logits.detach().softmax(-1)
    while True:
        stationary_ = stationary @ stationary
        if torch.allclose(stationary, stationary_):
            break
        stationary = stationary_
    exp_expectation = (
        stationary[0] * torch.arange(V, device=device, dtype=stationary.dtype)
    ).sum()

    random_walk = RandomWalk(lm)
    sample, lens, exp_lprobs = random_walk(max_iters=T)
    assert sample.shape == (T,)
    assert lens == T
    lm_sample = lm(sample[:-1].unsqueeze(1)).log_softmax(-1)
    assert lm_sample.shape == (T, 1, V)
    act_lprobs = lm_sample.gather(2, sample.view(T, 1, 1)).sum()
    assert torch.isclose(exp_lprobs / T, act_lprobs / T, atol=1e-2)
    act_expectation = sample.float().mean()
    assert torch.allclose(exp_expectation, act_expectation, atol=1e-2)


# it's deterministic and always terminates
class SpinningLM(SequentialLanguageModel):
    def calc_idx_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor], idx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if idx == 0:
            hist = prev["init"]
        next_ = (hist[idx - 1] + 1) % self.vocab_size
        return (
            torch.nn.functional.one_hot(next_, self.vocab_size).float().log() - 1,
            prev,
        )


def test_random_walk_batch(device, jit_type):
    if jit_type == "trace":
        pytest.xfail("trace unsupported for RandomWalk")

    N, V = 128, 16
    assert V > 1
    lm = SpinningLM(V)
    if jit_type == "script":
        lm = torch.jit.script(lm)
    walk = RandomWalk(lm, eos=V - 1).to(device)
    if jit_type == "script":
        walk = torch.jit.script(walk)
    y_prev = torch.randint(0, V, (1, N), device=device)
    y_prev[0, 0] = V - 2  # cycles to V - 1 (eos) immediately

    y_exp, y_lens_exp, log_probs_exp = walk({"init": y_prev}, N)
    assert y_exp.shape[1:] == y_lens_exp.shape == log_probs_exp.shape == (N,)
    assert y_lens_exp[0] == 1
    assert (log_probs_exp == 0).all()

    for n in range(N):
        y_exp_n, y_lens_exp_n = y_exp[:, n], y_lens_exp[n]
        y_lens_exp_n = y_lens_exp_n.item()
        assert (y_exp_n[y_lens_exp_n - 1 :] == V - 1).all()
        y_exp_n = y_exp_n[:y_lens_exp_n]
        y_prev_n = y_prev[:, n : n + 1]
        y_act_n, y_lens_act_n, log_probs_act_n = walk({"init": y_prev_n})
        assert y_lens_act_n.shape == log_probs_act_n.shape == tuple()
        assert y_lens_act_n == y_lens_exp_n
        assert log_probs_act_n == 0
        assert y_act_n.shape == (y_lens_exp_n,)
        assert (y_act_n == y_exp_n).all()


@pytest.mark.parametrize("dim", [0, 2, -1, None])
def test_sequence_log_probs(device, dim, jit_type):
    max_steps, num_classes, eos = 30, 10, 0
    dim1, dim2, dim3, dim4 = 5, 2, 1, 4
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
        hyp_lens = hyp_lens.flatten()
        hyp = hyp.flatten(1)
        logits = logits.view(max_steps, -1, num_classes)
        trace_logits = torch.nn.utils.rnn.pack_sequence(
            [torch.empty(1, 1, device=device)], enforce_sorted=False
        )
        trace_hyp = torch.zeros(1, 1, device=device, dtype=torch.long)
    else:
        len_mask = len_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        trace_logits = torch.empty(1, 1, 1, 1, device=device)
        trace_hyp = torch.zeros(1, 1, 1, dtype=torch.long, device=device)
    hyp = hyp.masked_fill(len_mask == hyp_lens, eos)
    logits = logits.scatter(-1, hyp.unsqueeze(-1), 0.0)
    rand_mask = torch.randint_like(hyp, 2).eq(1)
    # > 0 to ensure that at least one valid value exists in the path
    rand_mask = rand_mask & (len_mask < hyp_lens) & (len_mask > 0)
    hyp = hyp.masked_fill(rand_mask, -1)
    padding_mask = (len_mask > hyp_lens) | rand_mask
    logits = logits.masked_fill(padding_mask.unsqueeze(-1), -float("inf"))
    if dim is None:
        logits = torch.nn.utils.rnn.pack_padded_sequence(
            logits, hyp_lens.cpu(), enforce_sorted=False
        )
    elif dim:
        hyp_dim = (dim + 5) % 5
        hyp = hyp.transpose(0, hyp_dim).contiguous()
        logits = logits.transpose(0, hyp_dim).contiguous()
    sequence_log_probs = SequenceLogProbabilities(0 if dim is None else dim, eos)
    if jit_type == "trace":
        sequence_log_probs = torch.jit.trace(
            sequence_log_probs, (trace_logits, trace_hyp)
        )
    elif jit_type == "script":
        sequence_log_probs = torch.jit.script(sequence_log_probs)
    log_probs = sequence_log_probs(logits, hyp)
    assert log_probs.eq(0.0).all()
    if dim is None:
        logits = torch.nn.utils.rnn.PackedSequence(
            torch.randn_like(logits.data),
            logits.batch_sizes,
            logits.sorted_indices,
            logits.unsorted_indices,
        )
    else:
        logits = torch.randn_like(logits)
    log_probs_1 = sequence_log_probs(logits, hyp)
    assert log_probs_1.ne(0.0).any()
    # this is mostly a test to ensure the packed sequences are being properly
    # sorted/unsorted
    log_probs_1 = log_probs_1[::2]
    if dim is None:
        logits, _ = torch.nn.utils.rnn.pad_packed_sequence(logits)
        logits = logits[:, ::2]
        hyp = hyp[:, ::2]
        hyp_lens = hyp_lens[::2]
        logits = torch.nn.utils.rnn.pack_padded_sequence(
            logits, hyp_lens.cpu(), enforce_sorted=False
        )
    elif dim == 0:
        logits = logits[:, ::2]
        hyp = hyp[:, ::2]
    else:
        logits = logits[::2]
        hyp = hyp[::2]
    log_probs_2 = sequence_log_probs(logits, hyp)
    assert log_probs_1.shape == log_probs_2.shape
    assert torch.allclose(log_probs_1, log_probs_2)


@pytest.mark.parametrize("eos", [None, 0], ids=["w/o eos", "eos=0"])
@pytest.mark.parametrize("batch_size", [None, 4], ids=["w/o batch", "batch=4"])
def test_sequential_language_model_distribution(device, eos, batch_size):
    T, V, M = 4, 5, 3
    N = 1 if batch_size is None else batch_size
    lm = RNNLM(V)
    random_walk = RandomWalk(lm, eos).to(device)

    dist = SequentialLanguageModelDistribution(
        random_walk, batch_size, max_iters=T, cache_samples=True, validate_args=True,
    )

    assert dist.has_enumerate_support
    support = dist.enumerate_support()
    if eos is None:
        total = V ** T
    else:
        total = 0
        for eos_pos in range(T + 1):
            total += (V - 1) ** eos_pos
    assert support.shape == (total,) + dist.batch_shape + (T,)
    support_lprobs = dist.log_prob(support)
    assert torch.allclose(
        support_lprobs.logsumexp(0), torch.tensor(0.0, device=device), atol=1e-5,
    )

    sample = dist.sample([M])
    assert sample.shape[:-1] == (M,) + dist.batch_shape
    if eos is None:
        assert sample.shape[-1] == T
    else:
        assert sample.shape[-1] <= T
        sample = torch.nn.functional.pad(sample, (0, T - sample.shape[-1]), value=eos)
    match = (sample.unsqueeze(1) == support.unsqueeze(0)).all(-1)
    assert match.any(1).all()
    exp_lprobs = (support_lprobs.unsqueeze(0).expand(match.shape) * match).sum(1)
    act_lprobs = dist.log_prob(sample)
    assert act_lprobs.shape == sample.shape[:-1]
    assert torch.allclose(exp_lprobs, act_lprobs)
    dist.clear_cache()
    act_lprobs = dist.log_prob(sample)
    assert torch.allclose(exp_lprobs, act_lprobs)

    sample = torch.randint(0, V, (M,) + dist.batch_shape + (T,), device=device)
    dist.log_prob(sample)  # in support
    if eos is not None:
        assert T >= 3
        sample[..., 1] = eos
        exp_lprobs = dist.log_prob(sample)
        sample[..., 2] = V + 1
        act_lprobs = dist.log_prob(sample)  # since invalid entry comes after eos
        assert torch.allclose(exp_lprobs, act_lprobs)
    sample[..., 0] = V + 1
    with pytest.raises(ValueError):
        dist.log_prob(sample)  # invalid entry comes before any eos

    sample = dist.sample()
    assert 0 < sample.numel() <= T * N
    del lm

