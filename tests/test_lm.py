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

import os
import itertools

from typing import Dict, Tuple

import torch
import pytest
import numpy as np

from pydrobert.torch.modules import (
    LookupLanguageModel,
    MixableSequentialLanguageModel,
    ShallowFusionLanguageModel,
)
from pydrobert.torch.data import parse_arpa_lm

INF = float("inf")
NAN = float("nan")


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


@pytest.mark.cpu
@pytest.mark.parametrize(
    "prob_dicts",
    [
        None,
        [{0: 0.0, 1: 1.0, 4: 4.0}],
        [
            {1: (1.0, -1.0), 2: (2.0, -2.0), 3: (3.0, -3)},
            {(1, 0): 1.1, (1, 1): 11.1, (3, 2): 23.1},
        ],
        [
            {1: (1.0, -1.0), 2: (2.0, -2.0), 3: (3.0, -3.0), 4: (4.0, -4.0)},
            {
                (1, 1): (11.1, -11.1),
                (2, 3): (32.1, -32.1),
                (2, 4): (42.1, -42.1),
                (4, 1): (14.1, -14.1),
            },
            {(0, 0, 1): 1.2, (0, 0, 2): 2.2, (4, 1, 4): 414.2, (3, 4, 1): 143.2},
        ],
    ],
    ids=["deft", "unigram", "bigram", "trigram"],
)
def test_lookup_language_model_builds_trie(prob_dicts):
    vocab_size, N = 5, 0 if prob_dicts is None else len(prob_dicts)
    lm = LookupLanguageModel(vocab_size, 0, prob_dicts=prob_dicts)

    class ReverseTrieNode(object):
        def __init__(self, lprobs=(-INF, 0)):
            try:
                lprobs = float(lprobs[0]), float(lprobs[1])
            except:
                lprobs = float(lprobs)
            self.children, self.lprobs = dict(), lprobs

        def __repr__(self) -> str:
            return self._str()

        def _str(self, ws="  ") -> str:
            if isinstance(self.lprobs, float):
                s = f"{self.lprobs:.1f}"
            else:
                s = f"{self.lprobs[0]:.1f}, {self.lprobs[1]:.1f}"
            if len(self.children):
                c = f"\n{ws}".join(
                    f"{k} -> {v._str(ws + '  ')}" for (k, v) in self.children.items()
                )
                s += f":\n{ws}{c}"
            return s

        def __getitem__(self, key):
            if isinstance(key, int):
                key = (key,)
            if len(key):
                return self.children[key[-1]][key[:-1]]
            else:
                return self.lprobs

    root = ReverseTrieNode()
    S = 0
    if N > 0:
        if N == 1:
            parents = [ReverseTrieNode(lprob) for lprob in lm.logps[:vocab_size]]
        else:
            parents = [
                ReverseTrieNode(lprob)
                for lprob in zip(lm.logps[:vocab_size], lm.logbs[:vocab_size])
            ]
            last_idxs = range(vocab_size)
        root.children.update(enumerate(parents))
        for n in range(2, N + 1):
            children, idxs = [], []
            for parent, last_idx in zip(parents, last_idxs):
                left_idx = int(lm.offsets[last_idx]) + last_idx
                right_idx = int(lm.offsets[last_idx + 1]) + last_idx + 1
                for idx in range(left_idx, right_idx):
                    id_ = int(lm.ids[idx - vocab_size - 1])
                    logp = lm.logps[idx]
                    if n == N:
                        lprobs = logp
                    else:
                        lprobs = (logp, lm.logbs[idx])
                    child = ReverseTrieNode(lprobs)
                    parent.children[id_] = child
                    children.append(child)
                    idxs.append(idx)
                S = max(S, right_idx - left_idx)
            parents, last_idxs = children, idxs

        for n, prob_dict in enumerate(prob_dicts):
            for key, exp in prob_dict.items():
                act = root[key]
                if n == N - 1:
                    assert np.isclose(exp, act), key
                else:
                    assert np.isclose(exp[0], act[0])
                    assert np.isclose(exp[1], act[1])
    assert lm.max_direct_descendants == S


@pytest.mark.parametrize("N", [1, 2, 5])
def test_lookup_language_model_log_probs(device, N, jit_type):
    vocab_size, sos = 10, -1
    prob_dicts = []
    for n in range(1, N + 1):
        max_ngrams = vocab_size**n
        has_ngram = torch.randint(2, (max_ngrams,), device=device).eq(1)
        dict_ = dict()
        last = n == N
        for idx, has in enumerate(has_ngram):
            if not has:
                continue
            key = []
            for _ in range(n):
                key.append(idx % vocab_size)
                idx //= vocab_size
            if n == 1:
                key = key[0]
            else:
                key = tuple(key)
            if last:
                dict_[key] = torch.randn((1,), device=device).item()
            else:
                dict_[key] = torch.randn((2,), device=device).tolist()
        prob_dicts.append(dict_)
    # we're not going to pad anything
    all_queries = [[(x,) for x in range(vocab_size)]]
    for _ in range(2, N + 1):
        all_queries.append(
            [
                x + (y,)
                for (x, y) in itertools.product(all_queries[-1], range(vocab_size))
            ]
        )

    def lookup(query, print_=False, backoff=False):
        n = len(query)
        prob_dict = prob_dicts[n - 1]
        if n == 1:
            query = query[0]
        val = prob_dict.get(query, None)
        if val is None:
            if backoff:
                if print_:
                    print(f"{query} backoff 0")
                return 0.0
            elif n == 1:
                if print_:
                    print(f"{query} logp -inf")
                return -INF
            logp = lookup(query[1:], print_)
            logb = lookup(query[:-1], print_, True)
            if print_:
                print(f"backing off {query}: {logp} + {logb} = {logp + logb}")
            return logp + logb
        if print_:
            print(f"query {query} matched {val}")
        if n == N:
            assert not backoff
            return val
        else:
            return val[1 if backoff else 0]

    exps = [
        torch.tensor([lookup(query) for query in ngram_queries], device=device).view(
            -1, vocab_size
        )
        for ngram_queries in all_queries
    ]
    hists = [torch.empty(0, 1, dtype=torch.long, device=device)] + [
        torch.tensor(ngram_queries, device=device).view(-1, nm1 + 1).t()
        for nm1, ngram_queries in enumerate(all_queries[:-1])
    ]
    del all_queries
    # the sos shouldn't matter -- it isn't in the lookup table. The lm will
    # back off to B(<sos>_) Pr(_rest), and B(<sos>_) will not exist and thus
    # be 0
    lm = LookupLanguageModel(vocab_size, sos, prob_dicts=prob_dicts).to(device)
    if jit_type == "script":
        lm = torch.jit.script(lm)
    elif jit_type == "trace":
        pytest.xfail("lookup_language_model trace unsupported")
    for exp, hist in zip(exps, hists):
        act = lm(hist, None, -1)[0]
        for n in range(hist.size(1)):
            assert torch.allclose(exp[n], act[n], atol=1e-5), n


def test_lookup_language_model_nonuniform_idx(device):
    S, N, B = 20, 5, 2
    prob_dicts = []
    vocab_size, sos = 10, -1
    prob_dicts = []
    for n in range(1, N + 1):
        max_ngrams = vocab_size**n
        has_ngram = torch.randint(2, (max_ngrams,), device=device).eq(1)
        dict_ = dict()
        last = n == N
        for idx, has in enumerate(has_ngram):
            if not has:
                continue
            key = []
            for _ in range(n):
                key.append(idx % vocab_size)
                idx //= vocab_size
            if n == 1:
                key = key[0]
            else:
                key = tuple(key)
            if last:
                dict_[key] = torch.randn((1,), device=device).item()
            else:
                dict_[key] = torch.randn((2,), device=device).tolist()
        prob_dicts.append(dict_)
    prob_dicts[0][sos] = (-99, 0)
    lm = LookupLanguageModel(
        vocab_size, sos, prob_dicts=prob_dicts, destructive=True
    ).to(device)
    hist = torch.randint(0, vocab_size, (S, B), device=device)
    exp = lm(hist)
    assert not exp.isnan().any()
    idx = torch.randint(0, S + 1, (B,), device=device)
    exp = exp.gather(0, idx.view(1, B, 1).expand(1, B, vocab_size)).squeeze(0)
    act, _ = lm(hist, idx=idx)
    assert torch.allclose(exp, act, atol=1e-5)


def test_lookup_language_model_sos_context(device):
    # 0 = sos
    prob_dicts = [
        {0: (-99, 0.0), 1: (0.1, -0.1), 2: (0.2, -0.2), 3: (0.3, -0.3)},
        {(0, 1): (0.01, -0.01), (0, 2): (0.02, -0.02)},
        {(0, 0, 1): 0.001},
    ]
    lm = LookupLanguageModel(4, sos=0, prob_dicts=prob_dicts, destructive=True)
    lm.to(device)
    # P(0|0) = P(0) = -99
    # P(1|0) = P(1|0) = 0.01
    # P(2|0) = P(2|0) = 0.02
    # P(3|0) = P(3) = 0.3
    exp = torch.tensor([[[-99.0, 0.01, 0.02, 0.3]]], device=device)
    act = lm(torch.empty((0, 1), device=device, dtype=torch.long))
    assert torch.allclose(exp, act, atol=1e-5)


@pytest.mark.gpu  # this is a really slow test on the cpu
def test_lookup_language_model_republic():
    device = torch.device("cuda:0")
    dir_ = os.path.join(os.path.dirname(__file__), "republic")
    arpa_file = os.path.join(dir_, "republic.arpa")
    assert os.path.exists(arpa_file)
    token2id_file = os.path.join(dir_, "token2id.map")
    assert os.path.exists(token2id_file)
    queries_file = os.path.join(dir_, "queries.txt")
    assert os.path.exists(queries_file)
    exp_file = os.path.join(dir_, "exp.txt")
    assert os.path.exists(exp_file)
    token2id = dict()
    with open(token2id_file) as f:
        for line in f:
            token, id_ = line.strip().split()
            assert token not in token2id
            token2id[token] = int(id_)
    sos, eos, oov = token2id["<s>"], token2id["</s>"], token2id["<unk>"]
    vocab_size = len(token2id)
    assert all(x in token2id.values() for x in range(vocab_size))
    queries = []
    with open(queries_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            query = [token2id.get(token, oov) for token in line.split()]
            queries.append(torch.tensor(query, device=device))
    queries = torch.nn.utils.rnn.pad_sequence(queries, padding_value=eos)
    exp = []
    with open(exp_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            exp.append(float(line))
    exp = torch.tensor(exp, device=device)
    assert exp.shape[0] == queries.shape[1]
    prob_dicts = parse_arpa_lm(arpa_file, token2id, np.float32)
    lm = LookupLanguageModel(
        vocab_size, sos=sos, prob_dicts=prob_dicts, destructive=True
    )
    lm = lm.to(device)
    log_probs = lm(queries)
    queries = torch.cat([queries, torch.full_like(queries[:1], eos)])
    assert log_probs.shape[:-1] == queries.shape
    log_probs = log_probs.gather(2, queries.unsqueeze(2)).squeeze(2)
    # determine the first location of eos and zero everything afterwards
    eos_mask = queries[:-1].eq(eos).cumsum(0).bool()
    eos_mask = torch.cat([eos_mask.new_full((1, queries.size(1)), False), eos_mask], 0)
    log_probs.masked_fill_(eos_mask, 0.0)
    assert torch.allclose(log_probs.sum(0), exp, atol=1e-5)


@pytest.mark.cpu
def test_lookup_language_model_state_dict():
    vocab_size, sos = 10, -1
    uni_list = [{0: 0.0, 1: 0.1, 2: 0.2}]
    lm_a = LookupLanguageModel(vocab_size, sos, prob_dicts=uni_list)
    lm_b = LookupLanguageModel(vocab_size, sos)

    def compare(assert_same):
        same_max_ngram = lm_a.max_ngram == lm_b.max_ngram
        same_max_ngram_nodes = lm_a.max_ngram_nodes == lm_b.max_ngram_nodes
        same_offsets = len(lm_a.offsets) == len(lm_b.offsets)
        if same_offsets:
            same_offsets = (lm_a.offsets == lm_b.offsets).all()
        same_ids = len(lm_a.ids) == len(lm_b.ids)
        if same_ids:
            same_ids = (lm_a.ids == lm_b.ids).all()
        same_logs = len(lm_a.logps) == len(lm_b.logps)
        same_logs &= len(lm_a.logbs) == len(lm_b.logbs)
        if same_logs:
            nan_mask = torch.isnan(lm_a.logps)
            same_logs = (
                torch.allclose(
                    lm_a.logps.masked_select(nan_mask.eq(0)),
                    lm_b.logps.masked_select(nan_mask.eq(0)),
                    atol=1e-5,
                )
                & (nan_mask == torch.isnan(lm_b.logps)).all().item()
            )
            nan_mask = torch.isnan(lm_a.logbs)
            same_logs &= (
                torch.allclose(
                    lm_a.logbs.masked_select(nan_mask.eq(0)),
                    lm_b.logbs.masked_select(nan_mask.eq(0)),
                    atol=1e-5,
                )
                & (nan_mask == torch.isnan(lm_b.logbs)).all().item()
            )
        if assert_same:
            assert same_max_ngram
            assert same_max_ngram_nodes
            assert same_offsets
            assert same_ids
            assert same_logs
        else:
            assert not (
                same_max_ngram
                and same_max_ngram_nodes
                and same_offsets
                and same_ids
                and same_logs
            )

    compare(False)
    lm_b.load_state_dict(lm_a.state_dict())
    compare(True)
    bi_list = [{2: (0.2, -0.2), 3: (0.3, -0.3)}, {(0, 3): 0.03, (2, 4): 0.24}]
    lm_a = LookupLanguageModel(vocab_size, sos=sos, prob_dicts=bi_list)
    compare(False)
    lm_b.load_state_dict(lm_a.state_dict())
    compare(True)
    tri_list = [
        {0: (0.0, 0.0)},
        dict(),
        {(0, 0, 0): 0.0, (4, 4, 4): 0.444, (2, 3, 2): 0.232},
    ]
    lm_a = LookupLanguageModel(vocab_size, sos=sos, prob_dicts=tri_list)
    compare(False)
    lm_b.load_state_dict(lm_a.state_dict())
    compare(True)


def test_sequential_language_model(device, jit_type):
    S, N, V = 30, 10, 50
    hist = torch.randint(0, V, (S, N), device=device)
    lm = RNNLM(V).to(device)
    if jit_type == "script":
        lm = torch.jit.script(lm)
    elif jit_type == "trace":
        pytest.xfail("trace unsupported for SequentialLanguageModel")
    log_probs = lm(hist)
    prev = dict()
    for idx in range(S):
        log_probs_idx, next_ = lm(hist[:idx], prev, idx=idx)
        assert torch.allclose(log_probs[idx], log_probs_idx)
        # this is more for the scripting to ensure we can handle both tensor and
        # integer indexes
        log_probs_idx_ = lm(hist[:idx], prev, idx=torch.as_tensor(idx).to(device))[0]
        assert torch.allclose(log_probs_idx, log_probs_idx_)
        prev = next_


def test_shallow_fusion_language_model(device):
    S, N, V, beta = 30, 10, 50, 0.2
    hist = torch.randint(0, V, (S, N), device=device)
    first = RNNLM(V).to(device)
    second = RNNLM(V).to(device)
    lm = ShallowFusionLanguageModel(first, second, beta)
    log_probs = lm(hist)
    prev = dict()
    for idx in range(S):
        log_probs_idx, next_ = lm(hist[:idx], prev, idx=idx)
        assert torch.allclose(log_probs[idx], log_probs_idx)
        # this is more for the scripting to ensure we can handle both tensor and
        # integer indexes
        log_probs_idx_ = lm(hist[:idx], prev, idx=torch.as_tensor(idx).to(device))[0]
        assert torch.allclose(log_probs_idx, log_probs_idx_)
        prev = next_
    log_probs_first = first(hist)
    log_probs_second = second(hist)
    log_probs_ = log_probs_first + beta * log_probs_second
    assert torch.allclose(log_probs, log_probs_)
