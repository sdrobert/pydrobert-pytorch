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
import itertools
import warnings

import pytest
import torch
import pydrobert.torch.layers as layers
import pydrobert.torch.util as util

INF = float("inf")
NAN = float("nan")


@pytest.mark.cpu
@pytest.mark.parametrize(
    "prob_list,pointers,ids,logs",
    [
        (
            None,
            torch.tensor([], dtype=torch.uint8),
            torch.tensor([], dtype=torch.uint8),
            -torch.tensor([5] * 5, dtype=torch.float).log(),
        ),
        (
            [{0: 0.0, 1: 0.1, 4: 0.4}],
            torch.tensor([], dtype=torch.uint8),
            torch.tensor([], dtype=torch.uint8),
            torch.tensor([0.0, 0.1, -INF, -INF, 0.4]),
        ),
        (
            [
                {1: (0.1, -0.1), 2: (0.2, -0.2), 3: (0.3, -0.3)},
                {(1, 0): 1.0, (1, 1): 1.1, (3, 2): 3.3},
            ],
            torch.tensor([6, 5, 6, 5, 5, 4], dtype=torch.uint8),
            torch.tensor([0, 1, 2], dtype=torch.uint8),
            torch.tensor(
                [
                    -INF,
                    0.1,
                    0.2,
                    0.3,
                    -INF,
                    NAN,  # logp 1-gram
                    1.0,
                    1.1,
                    3.3,  # logp 2-gram
                    0.0,
                    -0.1,
                    -0.2,
                    -0.3,
                    0.0,
                    NAN,  # logb 1-gram
                ]
            ),
        ),
        (
            [
                {1: (0.1, -0.1), 2: (0.2, -0.2), 3: (0.3, -0.3), 4: (0.4, -0.4)},
                {
                    (1, 1): (1.1, -1.1),
                    (2, 3): (2.3, -2.3),
                    (2, 4): (2.4, -2.4),
                    (4, 1): (4.1, -4.1),
                },
                {(0, 0, 1): 0.01, (0, 0, 2): 0.02, (4, 1, 4): 4.14},
            ],
            torch.tensor(
                [
                    6,
                    6,
                    6,
                    7,
                    6,
                    6,  # 1-gram -> 2-gram
                    6,
                    7,
                    6,
                    5,
                    4,
                    4,  # 2-gram -> 3-gram
                ],
                dtype=torch.uint8,
            ),
            torch.tensor(
                [0, 1, 3, 4, 1, 0, 1, 2, 4],  # 2-gram suffix  # 3-gram suffix
                dtype=torch.uint8,
            ),
            torch.tensor(
                [
                    -INF,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    NAN,  # logp 1-gram
                    -INF,
                    1.1,
                    2.3,
                    2.4,
                    4.1,
                    NAN,  # logp 2-gram
                    0.01,
                    0.02,
                    4.14,  # logp 3-gram
                    0.0,
                    -0.1,
                    -0.2,
                    -0.3,
                    -0.4,
                    NAN,  # logb 1-gram
                    0.0,
                    -1.1,
                    -2.3,
                    -2.4,
                    -4.1,
                    NAN,  # Logb 2-gram
                ]
            ),
        ),
    ],
    ids=["deft", "unigram", "bigram", "trigram"],
)
def test_lookup_language_model_builds_trie(prob_list, pointers, ids, logs):
    vocab_size = 5
    lm = layers.LookupLanguageModel(vocab_size, 0, prob_list=prob_list)
    assert lm.pointers.shape == pointers.shape
    assert lm.ids.shape == ids.shape
    assert lm.logs.shape == logs.shape
    assert (lm.ids == ids).all()
    assert (lm.pointers == pointers).all()
    nan_mask = torch.isnan(lm.logs)
    assert torch.isnan(lm.logs).eq(nan_mask).all()
    assert torch.allclose(
        logs.masked_select(~nan_mask), lm.logs.masked_select(~nan_mask)
    )


@pytest.mark.parametrize("N", [1, 2, 5])
def test_lookup_language_model_log_probs(device, N):
    torch.manual_seed(1900)
    vocab_size, sos = 10, -1
    prob_list = []
    for n in range(1, N + 1):
        max_ngrams = vocab_size ** n
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
        prob_list.append(dict_)
    # we're not going to pad anything
    all_queries = [[(x,) for x in range(vocab_size)]]
    for _ in range(2, N + 1):
        all_queries.append(
            [
                x + (y,)
                for (x, y) in itertools.product(all_queries[-1], range(vocab_size))
            ]
        )

    def lookup(list_, query):
        if len(list_) > len(query):
            return lookup(list_[:-1], query)
        if len(list_) == 1:
            if N == 1:
                return list_[0].get(query[0], -INF)
            else:
                return list_[0].get(query[0], (-INF, 0.0))[0]
        val = list_[-1].get(query, None)
        if val is None:
            if len(list_) == 2:
                backoff = list_[-2].get(query[0], (-INF, 0.0))[1]
            else:
                backoff = list_[-2].get(query[:-1], (-INF, 0.0))[1]
            return backoff + lookup(list_[:-1], query[1:])
        if len(list_) == N:
            return val
        else:
            return val[0]

    exps = [
        torch.tensor(
            [lookup(prob_list, query) for query in ngram_queries], device=device
        ).view(-1, vocab_size)
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
    lm = layers.LookupLanguageModel(vocab_size, sos, prob_list=prob_list)
    lm = lm.to(device)
    for exp, hist in zip(exps, hists):
        act = lm(hist, None, torch.tensor(-1, device=device))[0]
        assert torch.allclose(exp, act, atol=1e-5)


def test_lookup_language_model_nonuniform_idx(device):
    S, N, B = 100, 5, 30
    prob_list = []
    vocab_size, sos = 10, -1
    prob_list = []
    for n in range(1, N + 1):
        max_ngrams = vocab_size ** n
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
        prob_list.append(dict_)
    prob_list[0][sos] = (-99, 0)
    lm = layers.LookupLanguageModel(vocab_size, sos, prob_list=prob_list).to(device)
    hist = torch.randint(0, vocab_size, (S, B), device=device)
    exp = lm(hist)
    idx = torch.randint(0, S + 1, (B,), device=device)
    exp = exp.gather(0, idx.view(1, B, 1).expand(1, B, vocab_size)).squeeze(0)
    act, _ = lm(hist, idx=idx)
    assert torch.allclose(exp, act, atol=1e-5)


def test_lookup_language_model_sos_context(device):
    # 0 = sos
    prob_list = [
        {0: (-99, 0.0), 1: (0.1, -0.1), 2: (0.2, -0.2), 3: (0.3, -0.3)},
        {(0, 1): (0.01, -0.01), (0, 2): (0.02, -0.02)},
        {(0, 0, 1): 0.001},
    ]
    lm = layers.LookupLanguageModel(4, sos=0, prob_list=prob_list)
    lm.to(device)
    # XXX(sdrobert): pad_sos_to_n has been removed now - it's always true
    # P(0|0, 0) = P(0) = -99
    # P(1|0, 0) = 0.001
    # P(2|0, 0) = P(2|0) = 0.02
    # P(3|0, 0) = P(3) = 0.3
    exp = torch.tensor([[[-99.0, 0.001, 0.02, 0.3]]], device=device)
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
    import pydrobert.torch.util as util

    prob_list = util.parse_arpa_lm(arpa_file, token2id=token2id)
    lm = layers.LookupLanguageModel(vocab_size, sos=sos, prob_list=prob_list)
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
    lm_a = layers.LookupLanguageModel(vocab_size, sos, prob_list=uni_list)
    lm_b = layers.LookupLanguageModel(vocab_size, sos)

    def compare(assert_same):
        same_max_ngram = lm_a.max_ngram == lm_b.max_ngram
        same_max_ngram_nodes = lm_a.max_ngram_nodes == lm_b.max_ngram_nodes
        same_pointers = len(lm_a.pointers) == len(lm_b.pointers)
        if same_pointers:
            same_pointers = (lm_a.pointers == lm_b.pointers).all()
        same_ids = len(lm_a.ids) == len(lm_b.ids)
        if same_ids:
            same_ids = (lm_a.ids == lm_b.ids).all()
        same_logs = len(lm_a.logs) == len(lm_b.logs)
        if same_logs:
            nan_mask = torch.isnan(lm_a.logs)
            assert (nan_mask == torch.isnan(lm_b.logs)).all()
            same_logs = torch.allclose(
                lm_a.logs.masked_select(nan_mask.eq(0)),
                lm_b.logs.masked_select(nan_mask.eq(0)),
                atol=1e-5,
            )
        if assert_same:
            assert same_max_ngram
            assert same_max_ngram_nodes
            assert same_pointers
            assert same_ids
            assert same_logs
        else:
            assert not (
                same_max_ngram
                and same_max_ngram_nodes
                and same_pointers
                and same_ids
                and same_logs
            )

    compare(False)
    lm_b.load_state_dict(lm_a.state_dict())
    compare(True)
    bi_list = [{2: (0.2, -0.2), 3: (0.3, -0.3)}, {(0, 3): 0.03, (2, 4): 0.24}]
    lm_a = layers.LookupLanguageModel(vocab_size, sos=sos, prob_list=bi_list)
    compare(False)
    lm_b.load_state_dict(lm_a.state_dict())
    compare(True)
    tri_list = [
        {0: (0.0, 0.0)},
        dict(),
        {(0, 0, 0): 0.0, (4, 4, 4): 0.444, (2, 3, 2): 0.232},
    ]
    lm_a = layers.LookupLanguageModel(vocab_size, sos=sos, prob_list=tri_list)
    compare(False)
    lm_b.load_state_dict(lm_a.state_dict())
    compare(True)


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("sub_avg", [True, False])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_minimum_error_rate_loss(device, batch_first, sub_avg, reduction, trace):
    torch.manual_seed(100)
    num_batches, samples, num_classes = 5, 5, 30
    max_ref_steps, max_hyp_steps = 10, 5
    assert max_ref_steps > max_hyp_steps  # nonzero loss guaranteed
    if batch_first:
        hyp = torch.randint(
            num_classes, (num_batches, samples, max_hyp_steps), device=device
        )
        hyp[..., 0] = 0
        ref = torch.randint(num_classes, (num_batches, max_ref_steps), device=device)
        ref[..., 0] = 0
    else:
        hyp = torch.randint(
            num_classes, (max_hyp_steps, num_batches, samples), device=device
        )
        hyp[0] = 0
        ref = torch.randint(num_classes, (max_ref_steps, num_batches), device=device)
        ref[0] = 0
    log_probs = torch.randn(num_batches, samples, device=device)
    loss = layers.MinimumErrorRateLoss(
        eos=None, sub_avg=sub_avg, batch_first=batch_first, reduction=reduction,
    )
    if trace:
        loss = torch.jit.trace(loss, (log_probs, ref, hyp))
    l1 = loss(log_probs, ref, hyp)
    assert l1.ne(0.0).any()
    l2 = loss(log_probs, ref, hyp)
    assert torch.allclose(l1, l2)
    loss = layers.MinimumErrorRateLoss(
        eos=0, sub_avg=sub_avg, batch_first=batch_first, reduction=reduction,
    )
    if trace:
        loss = torch.jit.trace(loss, (log_probs, ref, hyp))
    l3 = loss(log_probs, ref, hyp)
    assert l3.eq(0.0).all()


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("eos", [None, 0])
@pytest.mark.parametrize("ref_steps_times", [1, 2])
@pytest.mark.parametrize("reduction", ["mean", "none"])
@pytest.mark.parametrize("include_eos", [True, False])
def test_hard_optimal_completion_distillation_loss(
    device, batch_first, eos, ref_steps_times, reduction, include_eos, trace
):
    torch.manual_seed(209384)
    num_batches, max_steps, num_classes = 20, 41, 10
    if eos is None:
        hyp_lens = torch.tensor(max_steps).expand(num_batches)
        ref_lens = torch.tensor(ref_steps_times * max_steps)
        ref_lens = ref_lens.expand(num_batches)
    else:
        hyp_lens = torch.randint(1, max_steps + 1, (num_batches,))
        ref_lens = torch.randint(2, ref_steps_times * max_steps + 1, (num_batches,))
    ref = torch.nn.utils.rnn.pad_sequence(
        [torch.randint(1, num_classes, (x,)) for x in ref_lens],
        padding_value=num_classes - 1,
        batch_first=batch_first,
    )
    hyp = torch.nn.utils.rnn.pad_sequence(
        [torch.randint(1, num_classes, (x,)) for x in hyp_lens],
        padding_value=-1,
        batch_first=batch_first,
    )
    if eos is not None:
        for bt in range(num_batches):
            if batch_first:
                ref[bt, ref_lens[bt] - 1] = eos
                hyp[bt, hyp_lens[bt] - 1] = eos
            else:
                ref[ref_lens[bt] - 1, bt] = eos
                hyp[hyp_lens[bt] - 1, bt] = eos
        if not include_eos:
            ref_lens = ref_lens - 1
            hyp_lens = hyp_lens - 1
    logits = torch.rand(tuple(hyp.shape) + (num_classes,))
    if batch_first:
        len_mask = torch.arange(hyp.shape[1]).unsqueeze(0) < hyp_lens.unsqueeze(1)
    else:
        len_mask = torch.arange(hyp.shape[0]).unsqueeze(1) < hyp_lens
    logits, ref, hyp = logits.to(device), ref.to(device), hyp.to(device)
    ref_lens, hyp_lens = ref_lens.to(device), hyp_lens.to(device)
    len_mask = len_mask.to(device)
    inv_len_mask = len_mask.eq(0)
    logits.requires_grad_(True)
    loss = layers.HardOptimalCompletionDistillationLoss(
        eos=eos, include_eos=include_eos, batch_first=batch_first, reduction=reduction,
    )
    if trace:
        loss = torch.jit.trace(loss, (logits, ref, hyp))
    l1 = loss(logits, ref, hyp)
    assert torch.all(l1 == l1)  # no nans
    if reduction == "none":
        assert torch.all(l1.masked_select(inv_len_mask).eq(0.0))
        # reference transcriptions are all positive length, so the first
        # optimal completion (assuming hyp length is nonzero) will always be
        # the first token in ref (and only the first token), given that there's
        # no ambiguity in the alignment of the prefix ""
        log_probs = torch.nn.functional.log_softmax(logits, 2)
        if batch_first:
            zero_length_mask = ref_lens.eq(0).unsqueeze(1)
            first_loss = torch.where(
                zero_length_mask,
                torch.zeros_like(log_probs[:, 0, 0]),
                -log_probs[:, 0].gather(1, ref[:, 0].unsqueeze(-1)).squeeze(-1),
            )
            assert torch.allclose(l1[:, 0], first_loss)
        else:
            zero_length_mask = ref_lens.eq(0).unsqueeze(0)
            first_loss = torch.where(
                zero_length_mask,
                torch.zeros_like(log_probs[0, :, 0]),
                -log_probs[0].gather(1, ref[0].unsqueeze(-1)).squeeze(-1),
            )
            assert torch.allclose(l1[0], first_loss)
        l1 = l1.mean()
    (g,) = torch.autograd.grad([l1], [logits])
    assert torch.all(g.masked_select(inv_len_mask.unsqueeze(-1)).eq(0.0))
    assert not torch.all(g.eq(0.0))


def test_sequential_language_model(device):
    class RNNLM(layers.SequentialLanguageModel):
        def __init__(self, vocab_size, embed_size=128, hidden_size=512):
            super(RNNLM, self).__init__(vocab_size)
            self.embed = torch.nn.Embedding(
                vocab_size + 1, embed_size, padding_idx=vocab_size
            )
            self.rnn = torch.nn.LSTMCell(embed_size, hidden_size)
            self.ff = torch.nn.Linear(hidden_size, vocab_size)

        def calc_idx_log_probs(self, hist, prev, idx):
            N = hist.size(1)
            if idx == 0:
                in_ = hist.new_full((N,), self.vocab_size)
                prev = [self.rnn.weight_hh.new_zeros((N, self.rnn.hidden_size))] * 2
            else:
                if not prev:
                    prev = self.calc_idx_log_probs(hist, None, idx - 1)[1]
                in_ = hist[idx - 1]
            embedding = self.embed(in_)
            h_1, c_1 = self.rnn(embedding, prev)
            logits = self.ff(h_1)
            return torch.nn.functional.log_softmax(logits, -1), (h_1, c_1)

    S, N, V = 100, 10, 50
    hist = torch.randint(0, V, (S, N), device=device)
    lm = RNNLM(V).to(device)
    log_probs = lm(hist)
    for idx in torch.arange(S, -1, -1, device=device):
        log_probs_idx = lm(hist[:idx], idx=idx)[0]
        assert torch.allclose(log_probs[idx], log_probs_idx)


def test_ctc_prefix_search(device):
    class MyLM(layers.MixableSequentialLanguageModel):
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
    lm = MyLM().to(device)
    for beta, y_exp, probs_exp in exps:
        search = layers.CTCPrefixSearch(K, beta, lm)
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


def test_ctc_prefix_search_batch(device):
    class RNNLM(layers.MixableSequentialLanguageModel):
        def __init__(self, vocab_size, embed_size=128, hidden_size=512):
            super().__init__(vocab_size)
            self.hidden_size = hidden_size
            self.embed = torch.nn.Embedding(
                vocab_size + 1, embed_size, padding_idx=vocab_size
            )
            self.cell = torch.nn.LSTMCell(embed_size, hidden_size)
            self.ff = torch.nn.Linear(hidden_size, vocab_size)

        def extract_by_src(self, prev, src):
            return {
                "hidden_state": prev["hidden_state"].index_select(0, src),
                "cell_state": prev["cell_state"].index_select(0, src),
            }

        def mix_by_mask(self, prev_true, prev_false, mask):
            return dict(
                (k, torch.where(mask.unsqueeze(1), prev_true[k], prev_false[k]))
                for k in prev_true
            )

        def update_input(self, prev, hist):
            if len(prev):
                return prev
            N = hist.size(1)
            zeros = self.ff.weight.new_zeros((N, self.hidden_size))
            return {"hidden_state": zeros, "cell_state": zeros}

        def calc_idx_log_probs(self, hist, prev, idx):
            idx_zero = idx == 0
            if idx_zero.all():
                x = hist.new_full((hist.size(1),), self.vocab_size)
            else:
                x = hist.gather(0, (idx - 1).clamp(min=0).unsqueeze(0)).squeeze(
                    0
                )  # (N,)
                x = x.masked_fill(idx_zero, self.vocab_size)
            x = self.embed(x)
            h_1, c_1 = self.cell(x, (prev["hidden_state"], prev["cell_state"]))
            logits = self.ff(h_1)
            return (
                torch.nn.functional.log_softmax(logits, -1),
                {"hidden_state": h_1, "cell_state": c_1},
            )

    T, N, V, K = 50, 128, 50, 5
    assert K <= V
    lm = RNNLM(V)
    search = layers.CTCPrefixSearch(K, lm=lm).to(device)
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


def test_beam_search(device):
    class MyLM(layers.ExtractableSequentialLanguageModel):
        def __init__(self, vocab_size):
            super().__init__(vocab_size)
            bigram_table = (
                torch.arange(1, vocab_size + 1, dtype=torch.float)
                .unsqueeze(0)
                .expand(vocab_size, vocab_size)
            )
            # dist over idx = [0, ..., 0, idx + 1, idx + 2, -(idx - 3), ..., -V]
            bigram_table = bigram_table - bigram_table.triu(2) - bigram_table.tril(-1)
            self.register_buffer("bigram_table", bigram_table)

        def update_input(self, in_prev, hist):
            return in_prev

        def extract_by_src(self, in_prev, src):
            return in_prev

        def calc_idx_log_probs(self, hist, in_prev, idx):
            hist = torch.cat(
                [torch.arange(hist.size(1), device=hist.device).unsqueeze(0), hist], 0
            )
            vocab = hist.gather(0, idx.unsqueeze(0)).squeeze(0)
            return self.bigram_table.index_select(vocab, 0), in_prev


def test_beam_search_batch(device):
    torch.manual_seed(1029)

    class RNNLM(layers.ExtractableSequentialLanguageModel):
        def __init__(self, vocab_size, embed_size=128, hidden_size=512):
            super().__init__(vocab_size)
            self.hidden_size = hidden_size
            self.embed = torch.nn.Embedding(
                vocab_size + 1, embed_size, padding_idx=vocab_size
            )
            self.cell = torch.nn.LSTMCell(embed_size, hidden_size)
            self.ff = torch.nn.Linear(hidden_size, vocab_size)

        def extract_by_src(self, prev, src):
            return {
                "hidden_state": prev["hidden_state"].index_select(0, src),
                "cell_state": prev["cell_state"].index_select(0, src),
            }

        def update_input(self, prev, hist):
            if len(prev):
                return prev
            N = hist.size(1)
            zeros = self.ff.weight.new_zeros((N, self.hidden_size))
            return {"hidden_state": zeros, "cell_state": zeros}

        def calc_idx_log_probs(self, hist, prev, idx):
            idx_zero = idx == 0
            if idx_zero.all():
                x = torch.arange(hist.size(0), device=hist.device)
            elif not idx.dim():
                x = hist[idx - 1]
            else:
                x = hist.gather(0, (idx - 1).clamp(min=0).unsqueeze(0)).squeeze(
                    0
                )  # (N,)
                x = x.masked_fill(idx_zero, self.vocab_size)
            x = self.embed(x)
            h_1, c_1 = self.cell(x, (prev["hidden_state"], prev["cell_state"]))
            logits = self.ff(h_1)
            return (
                torch.nn.functional.log_softmax(logits, -1),
                {"hidden_state": h_1, "cell_state": c_1},
            )

    T, N, V, K = 64, 16, 128, 8
    assert K <= V and N * K <= V
    lm = RNNLM(V)
    search = layers.BeamSearch(lm, K, eos=0, max_iters=T).to(device)
    y_prev = torch.arange(N, device=device)

    exps = []
    for y_prev_n in y_prev:
        y_prev_n = y_prev_n.view(1, 1)
        y_n_exp, y_lens_n_exp, log_probs_n_exp = search(y_prev_n)
        y_n_exp = y_n_exp.squeeze(1)  # (T_n, K_n)
        y_lens_n_exp = y_lens_n_exp.squeeze(0)  # (K_n,)
        log_probs_n_exp = log_probs_n_exp.squeeze(0)  # (K_n,)
        valid_prefix_mask_n_exp = log_probs_n_exp > -float("inf")
        if not valid_prefix_mask_n_exp.all():
            assert y_lens_n_exp[0] == 1
        exps.append((y_n_exp, y_lens_n_exp, log_probs_n_exp))

    y_act, y_lens_act, log_probs_act = search(y_prev.unsqueeze(0))
    for (
        (y_n_exp, y_lens_n_exp, log_probs_n_exp),
        y_n_act,
        y_lens_n_act,
        log_probs_n_act,
    ) in zip(exps, y_act.transpose(0, 1), y_lens_act, log_probs_act):
        assert y_n_exp.shape[1:] == y_n_act.shape[1:]
        assert y_n_exp.size(0) <= y_n_act.size(0)
        assert y_lens_n_exp.shape == y_lens_n_act.shape
        assert log_probs_n_exp.shape == log_probs_n_act.shape
        valid_prefix_mask_n_exp = log_probs_n_exp > -float("inf")
        valid_prefix_mask_n_act = log_probs_n_act > -float("inf")
        assert (valid_prefix_mask_n_exp == valid_prefix_mask_n_act).all()
        if not valid_prefix_mask_n_exp.all():
            assert valid_prefix_mask_n_exp.sum() == 1  # only one valid path: empty one
            y_n_exp, y_n_act = y_n_exp[:, :1], y_n_act[:, :1]
            y_lens_n_exp, y_lens_n_act = y_lens_n_exp[:1], y_lens_n_act[:1]
            log_probs_n_exp, log_probs_n_act = log_probs_n_exp[:1], log_probs_n_act[:1]
        assert (y_lens_n_exp == y_lens_n_act).all()
        assert torch.allclose(log_probs_n_exp, log_probs_n_act)
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


@pytest.mark.parametrize("dim", [0, 1])
def test_global_soft_attention(device, dim):
    class FirstIsBest(layers.GlobalSoftAttention):
        def score(self, query, key):
            e = torch.full_like(key[..., 0], -float("inf"))
            e.narrow(self.dim, 0, 1).fill_(0.0)
            return e

    class ILoveEveryoneEqually(layers.GlobalSoftAttention):
        def score(self, query, key):
            return torch.zeros_like(key[..., 0])

    torch.manual_seed(562992)
    T, max_dim, max_dim_size = 12, 10, 10
    num_dim = torch.randint(dim + 2, max_dim + 1, (1,), device=device).item()
    key_shape = torch.randint(
        1, max_dim_size + 1, (num_dim + 1,), device=device
    ).tolist()
    key_shape[dim] = T
    query_shape = key_shape[:dim] + key_shape[dim + 1 : -1]
    del key_shape[-2]
    key_lens = torch.randint(1, T + 1, query_shape[:-1], device=device)
    query = torch.randn(*query_shape, device=device)
    key = torch.randn(*key_shape, device=device)
    query_size = query_shape[-1]
    key_size = key_shape[-1]
    arange_shape = [1] * (num_dim - 1)
    arange_shape[dim] = T
    mask = torch.arange(T, device=device).view(*arange_shape)
    mask = mask < key_lens.unsqueeze(dim)
    key.requires_grad_(True)
    first_attention = FirstIsBest(query_size, key_size, dim).to(device)
    equal_attention = ILoveEveryoneEqually(query_size, key_size, dim).to(device)
    out1 = first_attention(query, key, key)
    assert torch.allclose(out1, key.narrow(dim, 0, 1).squeeze(dim))
    out2 = first_attention(query, key, key, mask)
    assert torch.allclose(out1, out2)
    (g,) = torch.autograd.grad([out1], [key], grad_outputs=torch.ones_like(out1))
    assert g.narrow(dim, 0, 1).eq(1).all()
    assert g.narrow(dim, 1, T - 1).eq(0).all()
    out1 = equal_attention(query, key, key)
    # the softmax introduces a slight numeric instability
    assert torch.allclose(out1, key.mean(dim), atol=1e-5)
    out2 = equal_attention(query, key, key, mask)
    assert not torch.allclose(out1, out2)
    exp = key.masked_fill(mask.unsqueeze(-1).eq(0), 0.0)
    exp = exp.sum(dim)
    exp = exp / key_lens.float().unsqueeze(-1)
    assert torch.allclose(out2, exp, atol=1e-5)
    (g,) = torch.autograd.grad([out2], [key], grad_outputs=torch.ones_like(out2))
    assert g.masked_select(mask.eq(0).unsqueeze(-1)).eq(0).all()
    assert torch.allclose(g.sum(dim), torch.tensor(1.0, device=device), atol=1e-5)


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_dot_product_soft_attention(device, dim, trace):
    torch.manual_seed(387420)
    dim1, dim2, dim3, dim4 = 50, 30, 12, 100
    key_shape = (dim1, dim2, dim3, dim4)
    key = torch.randn(*key_shape, device=device)
    query_shape = key_shape[:dim] + key_shape[dim + 1 :]
    query = torch.zeros(*query_shape, device=device)
    query[..., 0] = 2.0
    exp = torch.nn.functional.softmax(key[..., 0], dim).unsqueeze(-1) * key
    exp = exp.sum(dim)
    attention = layers.DotProductSoftAttention(dim4, dim, scale_factor=0.5)
    if trace:
        pytest.xfail("Trace doesn't work for attention")
        attention = torch.jit.trace(attention, (query, key, key))
    act = attention(query, key, key)
    assert torch.allclose(exp, act)


@pytest.mark.cpu
def test_dot_product_soft_attention_on_transformer_input():
    class MatrixVersion(torch.nn.Module):
        """Scaled dot product attention, specifically for transformers

        This was blatantly ripped from `speech transformers
        <https://github.com/kaituoxu/Speech-Transformer/blob/a0bbd58da193051bb0ea597e1c4120021a721c16/src/transformer/attention.py#L65>`__.

        This is a more straightforward implementation of the scaled dot product
        attention for transformer networks. We're showing that our
        implementation yields the same output and gradient as this.
        """

        def __init__(self, temperature):
            super(MatrixVersion, self).__init__()
            self.temperature = temperature
            self.softmax = torch.nn.Softmax(dim=2)

        def forward(self, q, k, v, mask=None):
            attn = torch.bmm(q, k.transpose(1, 2))
            attn = attn / self.temperature
            if mask is not None:
                attn = attn.masked_fill(mask, -float("inf"))
            attn = self.softmax(attn)
            output = torch.bmm(attn, v)
            return output

    torch.manual_seed(34229)
    num_batch, len_q, len_k, d_k, d_v = 30, 40, 20, 10, 50
    temp = 2.0
    query = torch.randn(num_batch, len_q, d_k, requires_grad=True)
    key = torch.randn(num_batch, len_k, d_k, requires_grad=True)
    value = torch.randn(num_batch, len_k, d_v, requires_grad=True)
    matrix_attention = MatrixVersion(temp)
    our_attention = layers.DotProductSoftAttention(d_k, 1, 1 / temp)
    out1 = matrix_attention(query, key, value)
    out2 = our_attention(query, key.unsqueeze(2), value.unsqueeze(2))
    assert torch.allclose(out1, out2, atol=1e-5)
    g_q1, g_k1, g_v1 = torch.autograd.grad(
        [out1], [query, key, value], grad_outputs=torch.ones_like(out1)
    )
    g_q2, g_k2, g_v2 = torch.autograd.grad(
        [out2], [query, key, value], grad_outputs=torch.ones_like(out2)
    )
    assert torch.allclose(g_q1, g_q2, atol=1e-5)
    assert torch.allclose(g_k1, g_k2, atol=1e-5)
    assert torch.allclose(g_v1, g_v2, atol=1e-5)
    mask = torch.randint(2, (num_batch, len_q, len_k)).eq(1)
    out1 = matrix_attention(query, key, value, mask)
    # the "contiguous" is necessary for 1.3.0
    # https://github.com/pytorch/pytorch/issues/28502
    out2 = our_attention(
        query,
        key.unsqueeze(2),
        value.unsqueeze(2),
        mask.transpose(1, 2).contiguous().eq(0),  # we use the inverse of mask
    )
    assert torch.allclose(out1, out2, atol=1e-5)


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "layer", ["general", "concat", "multihead_general", "multihead_concat"]
)
def test_learnable_soft_attention(device, dim, bias, layer, trace):
    torch.manual_seed(347201)
    max_dim, max_dim_size, max_num_heads = 5, 30, 10
    num_dim = torch.randint(dim + 2, max_dim + 1, (1,), device=device).item()
    # dim size must be at least 2. Otherwise, softmax will have only one
    # element and gradient will be zero through it
    key_shape = torch.randint(
        2, max_dim_size + 1, (num_dim + 1,), device=device
    ).tolist()
    query_shape = key_shape[:dim] + key_shape[dim + 1 : -1]
    del key_shape[-2]
    key = torch.randn(*key_shape, device=device)
    query = torch.randn(*query_shape, device=device)
    key_size = key_shape[-1]
    query_size = query_shape[-1]
    if layer == "general":
        attention = layers.GeneralizedDotProductSoftAttention(
            query_size, key_size, dim, bias
        )
    elif layer == "concat":
        attention = layers.ConcatSoftAttention(query_size, key_size, dim, bias)
    elif layer.startswith("multihead_"):
        num_heads = torch.randint(1, max_num_heads + 1, (1,), device=device).item()
        d_q = max(1, query_size // num_heads)
        d_k = max(1, key_size // num_heads)
        if layer.endswith("general"):
            single_head_attention = layers.GeneralizedDotProductSoftAttention(
                d_q, d_k, dim, bias
            )
        elif layer.endswith("concat"):
            single_head_attention = layers.ConcatSoftAttention(
                query_size, key_size, dim, bias
            )
        attention = layers.MultiHeadedAttention(
            query_size,
            key_size,
            key_size,
            num_heads,
            single_head_attention,
            bias_WQ=bias,
            bias_WK=bias,
            bias_WV=bias,
            bias_WC=bias,
        )
    attention = attention.to(device)
    torch.manual_seed(30)
    attention.reset_parameters()
    optim = torch.optim.Adam(attention.parameters(), lr=1.0)
    optim.zero_grad()
    if trace:
        pytest.xfail("Trace doesn't work for attention")
        attention_trace = torch.jit.trace(attention, (query, key, key))
    else:
        attention_trace = attention
    out1 = attention_trace(query, key, key)
    out1.mean().backward()
    optim.step()
    optim.zero_grad()
    out2 = attention_trace(query, key, key)
    assert not torch.allclose(out1, out2, atol=1e-5)
    torch.manual_seed(30)
    attention.reset_parameters()
    out3 = attention_trace(query, key, key)
    assert torch.allclose(out1, out3, atol=1e-5)


def test_spec_augment_compare_1d_warp_to_2d_warp(device):
    # it turns out that the 2D warp isn't fully agnostic to the control points in the
    # frequency dimension (as seen if you uncomment the final line in this test). It
    # appears that while the interpolated flow is zero along the frequency dimension,
    # there are some strange oscillations in the flow's time dimension in the 2D case.
    # I've verified with Daniel Park that this was not intended. In any event, it makes
    # this test flawed
    torch.manual_seed(21302)
    N, T, F, W = 12, 30, 20, 5
    feats = torch.rand(N, T, F, device=device)
    spec_augment = layers.SpecAugment(
        max_time_warp=W, max_freq_warp=0, max_time_mask=0, max_freq_mask=0
    )  # no masking
    params = spec_augment.draw_parameters(feats)
    w_0, w = params[:2]

    # the 2D transform outlined in the paper features 6 zero-flow boundary points -
    # one in each corner and two at the mid-points of the left and right edges -
    # and a single point w_0 along the midpoint line being shifted to w_0 + w.
    # the corner boundary points are handled by pinned_boundary_points=1
    # note: for coordinates, it's (freq, time), not (time, freq), despite the order in
    # feats
    midF = F // 2
    zeros = torch.zeros_like(w_0)  # (N,)
    shift_src = torch.stack([zeros + midF, w_0], 1)  # (N, 2)
    shift_dst = torch.stack([zeros + midF, w_0 + w], 1)  # (N, 2)
    left_src = left_dst = torch.stack([zeros + midF, zeros], 1)  # (N, 2)
    right_src = right_dst = torch.stack([zeros + midF, zeros + T - 1], 1)  # (N, 2)
    src = torch.stack([left_src, shift_src, right_src], 1)  # (N, 3, 2)
    dst = torch.stack([left_dst, shift_dst, right_dst], 1)  # (N, 3, 2)
    exp, flow = util.sparse_image_warp(
        feats.unsqueeze(1),
        src,
        dst,
        indexing="wh",
        pinned_boundary_points=1,
        include_flow=True,
    )
    exp = exp.squeeze(1)
    assert torch.allclose(flow[..., 0], torch.tensor(0.0, device=device))

    # act = spec_augment.apply_parameters(feats, params)
    # assert torch.allclose(exp, act), (exp - act).abs().max()  # will fail!


def test_spec_augment_batch(device):
    torch.manual_seed(83740212)
    N, T, F, W = 10, 30, 5, 5
    feats = torch.rand(N, T, F, device=device)
    lengths = torch.randint(1, T + 1, (N,), device=device)
    lengths[0] = T
    spec_augment = layers.SpecAugment(
        max_time_warp=W, max_freq_warp=0, max_time_mask=0, max_freq_mask=0
    )
    params = spec_augment.draw_parameters(feats, lengths)
    w_0, w = params[:2]
    feats.requires_grad = True
    act_feats = spec_augment.apply_parameters(feats, params, lengths)
    ones = (
        (lengths.unsqueeze(-1) > torch.arange(T, device=device))
        .float()
        .unsqueeze(-1)
        .expand(N, T, F)
    )
    (act_g,) = torch.autograd.grad([act_feats], [feats], ones)
    for n in range(N):
        feats_n = feats[n : n + 1, : lengths[n]]
        w_0_n, w_n = w_0[n : n + 1], w[n : n + 1]
        params_n = (w_0_n, w_n) + params[2:]
        exp_feats_n = spec_augment.apply_parameters(feats_n, params_n)
        act_feats_n = act_feats[n : n + 1, : lengths[n]]
        assert torch.allclose(exp_feats_n, act_feats_n, atol=1e-5), (
            (exp_feats_n - act_feats_n).abs().max()
        )
        (exp_g_n,) = torch.autograd.grad(
            [exp_feats_n], [feats_n], torch.ones_like(feats_n)
        )
        act_g_n = act_g[n : n + 1, : lengths[n]]
        assert torch.allclose(exp_g_n, act_g_n, atol=1e-5), (
            (exp_g_n - act_g_n).abs().max()
        )


def test_spec_augment_zero_params_is_identity(device):
    torch.manual_seed(6123810)
    N, T, F = 50, 200, 80
    feats = exp = torch.rand(N, T, F, device=device)
    spec_augment = layers.SpecAugment(
        max_time_warp=1000, max_freq_warp=1000, max_time_mask=1000, max_freq_mask=1000
    )
    params = spec_augment.draw_parameters(feats)
    # w_0 and v_0 must remain nonzero b/c otherwise interpolation would include the
    # border twice
    params[1].zero_()  # w
    params[3].zero_()  # v
    params[5].zero_()  # t
    params[7].zero_()  # f
    act = feats = spec_augment.apply_parameters(feats, params)
    assert torch.allclose(exp, act, atol=1e-4), (exp - act).abs().max()


def test_spec_augment_masking(device):
    torch.manual_seed(10927392)
    N, T, F = 500, 200, 80
    max_time_mask = max_freq_mask = 20
    nT = nF = 2
    feats = torch.ones(N, T, F, device=device)
    spec_augment = layers.SpecAugment(
        max_time_warp=0,
        max_freq_warp=0,
        max_time_mask=max_time_mask,
        max_freq_mask=max_freq_mask,
        max_time_mask_proportion=1.0,
        num_time_mask=nT,
        num_time_mask_proportion=1 / (100 * T),
        num_freq_mask=nF,
        interpolation_order=2,
    )

    assert nT == nF == 2  # logic below only works when 2

    # current setting shouldn't draw any time masks
    params = spec_augment.draw_parameters(feats)
    t = params[5]
    assert not (t > 0).any()

    spec_augment.num_time_mask_proportion = 1.0

    params = spec_augment.draw_parameters(feats)
    t_0, t, f_0, f = params[4:]
    assert (t > 0).any()  # some t could coincidentally land on zero
    t_1, f_1 = t_0 + t, f_0 + f  # (N, nT), (N, nF)

    max_t0s = torch.max(t_0.unsqueeze(1), t_0.unsqueeze(2))  # (N, nT, nT)
    min_t1s = torch.min(t_1.unsqueeze(1), t_1.unsqueeze(2))  # (N, nT, nT)
    diff_t = torch.clamp(min_t1s - max_t0s, min=0).tril()  # (N, nT, nT)
    diff_t = diff_t * torch.eye(nT, device=device, dtype=torch.long) - diff_t * (
        1 - torch.eye(nT, device=device, dtype=torch.long)
    )
    exp_masked_t = diff_t.sum(2).sum(1)  # (N,)
    assert torch.all(exp_masked_t <= t.sum(1))

    max_f0s = torch.max(f_0.unsqueeze(1), f_0.unsqueeze(2))  # (N, nF, nF)
    min_f1s = torch.min(f_1.unsqueeze(1), f_1.unsqueeze(2))  # (N, nF, nF)
    diff_f = torch.clamp(min_f1s - max_f0s, min=0).tril()  # (N, nF, nF)
    diff_f = diff_f * torch.eye(nF, device=device, dtype=torch.long) - diff_f * (
        1 - torch.eye(nF, device=device, dtype=torch.long)
    )
    exp_masked_f = diff_f.sum(2).sum(1)  # (N,)
    assert torch.all(exp_masked_f <= f.sum(1))

    act_feats = spec_augment.apply_parameters(feats, params)
    eq_0 = (act_feats == 0.0).long()
    act_masked_t = eq_0.prod(2).sum(1)
    act_masked_f = eq_0.prod(1).sum(1)

    assert torch.all(act_masked_t == exp_masked_t)
    assert torch.all(act_masked_f == exp_masked_f)


def test_spec_augment_call(device, trace):
    N, T, F = 30, 2048, 80
    max_time_warp, max_freq_warp = 15, 20
    max_time_mask, max_freq_mask = 30, 7
    num_time_mask, num_freq_mask = 2, 3
    max_time_mask_proportion = 0.2
    lengths = torch.randint(1, T + 1, (N,), device=device)
    feats = torch.rand(N, T, F, device=device)
    spec_augment = layers.SpecAugment(
        max_time_warp=max_time_warp,
        max_freq_warp=max_freq_warp,
        max_time_mask=max_time_mask,
        max_freq_mask=max_freq_mask,
        max_time_mask_proportion=max_time_mask_proportion,
        num_time_mask=num_time_mask,
        num_freq_mask=num_freq_mask,
    ).to(device)
    if trace:
        # spec_augment is nondeterministic, so we don't check repeat return values
        spec_augment = torch.jit.trace(
            spec_augment, (feats, lengths), check_trace=False
        )
    spec_augment(feats, lengths)


@pytest.mark.parametrize("mode", ["reflect", "constant", "replicate"])
def test_random_shift_call(device, mode, trace):
    torch.manual_seed(50)
    N, T, A, B = 50, 300, 13, 11
    in_ = torch.rand(N, T, A, B, device=device)
    in_lens = torch.randint(1, T + 1, (N,), device=device)
    rand_shift = layers.RandomShift(1.0, mode).to(device)
    if trace:
        # random_shift is nondeterministic, so we don't check repeat return values
        rand_shift = torch.jit.trace(rand_shift, (in_, in_lens), check_trace=False)
    out, out_lens = rand_shift(in_, in_lens)
    assert out.dim() == 4
    assert (out_lens >= in_lens).all()
    assert out.size(0) == N
    assert out.size(1) >= out_lens.max()
    assert out.size(2) == A
    assert out.size(3) == B


@pytest.mark.parametrize("dim", [0, 2, -1, None])
def test_sequence_log_probs(device, dim, trace):
    torch.manual_seed(24519)
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
        logits = torch.nn.utils.rnn.pack_padded_sequence(
            logits, hyp_lens.cpu(), enforce_sorted=False
        )
    elif dim:
        hyp_dim = (dim + 5) % 5
        hyp = hyp.transpose(0, hyp_dim).contiguous()
        logits = logits.transpose(0, hyp_dim).contiguous()
    sequence_log_probs = layers.SequentialLogProbabilities(
        0 if dim is None else dim, eos
    )
    if trace:
        with warnings.catch_warnings():
            # FIXME(sdrobert): there's an irritating warning caused by
            # torch.as_tensor being called in pack_padded_sequence. Since the input is
            # always a tensor already, it should be a harmless call.
            warnings.simplefilter("ignore")
            sequence_log_probs = torch.jit.trace(sequence_log_probs, (logits, hyp))
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
