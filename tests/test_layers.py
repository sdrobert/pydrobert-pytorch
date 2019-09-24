from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import itertools

import pytest
import torch
import pydrobert.torch.layers as layers

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"

INF = float('inf')
NAN = float('nan')


@pytest.mark.cpu
@pytest.mark.parametrize('ngram_list,pointers,ids,logs', [
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
        torch.tensor([
            -INF, 0.1,  0.2,  0.3,  -INF, NAN,  # logp 1-gram
            1.0,  1.1,  3.3,                    # logp 2-gram
            0.0,  -0.1, -0.2, -0.3, 0.0,  NAN,  # logb 1-gram
        ]),
    ),
    (
        [
            {1: (0.1, -0.1), 2: (0.2, -0.2), 3: (0.3, -0.3), 4: (0.4, -0.4)},
            {
                (1, 1): (1.1, -1.1), (2, 3): (2.3, -2.3), (2, 4): (2.4, -2.4),
                (4, 1): (4.1, -4.1)
            },
            {(0, 0, 1): 0.01, (0, 0, 2): 0.02, (4, 1, 4): 4.14},
        ],
        torch.tensor([
            6, 6, 6, 7, 6, 6,  # 1-gram -> 2-gram
            6, 7, 6, 5, 4, 4,  # 2-gram -> 3-gram
        ], dtype=torch.uint8),
        torch.tensor([
            0, 1, 3, 4, 1, 0,   # 2-gram suffix
            1, 2, 4,            # 3-gram suffix
        ], dtype=torch.uint8),
        torch.tensor([
            -INF, 0.1,  0.2,  0.3,  0.4,  NAN,  # logp 1-gram
            -INF, 1.1,  2.3,  2.4,  4.1,  NAN,  # logp 2-gram
            0.01, 0.02, 4.14,                   # logp 3-gram
            0.0,  -0.1, -0.2, -0.3, -0.4, NAN,  # logb 1-gram
            0.0,  -1.1, -2.3, -2.4, -4.1, NAN,  # Logb 2-gram
        ]),
    ),
], ids=['deft', 'unigram', 'bigram', 'trigram'])
def test_lookup_language_model_builds_trie(ngram_list, pointers, ids, logs):
    vocab_size = 5
    lm = layers.LookupLanguageModel(vocab_size, ngram_list=ngram_list)
    assert lm.pointers.shape == pointers.shape
    assert lm.ids.shape == ids.shape
    assert lm.logs.shape == logs.shape
    assert (lm.ids == ids).all()
    assert (lm.pointers == pointers).all()
    nan_mask = torch.isnan(lm.logs)
    assert torch.isnan(lm.logs).eq(nan_mask).all()
    assert torch.allclose(
        logs.masked_select(~nan_mask),
        lm.logs.masked_select(~nan_mask)
    )


@pytest.mark.parametrize('N', [1, 2, 5])
@pytest.mark.parametrize('sos', [-2, None])
def test_lookup_language_model_log_probs(device, N, sos):
    torch.manual_seed(1900)
    vocab_size, eos = 10, -1
    ngram_list = []
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
        ngram_list.append(dict_)
    # we're not going to pad anything
    all_queries = [[(x,) for x in range(vocab_size)]]
    for _ in range(2, N + 1):
        all_queries.append([
            x + (y,)
            for (x, y) in itertools.product(all_queries[-1], range(vocab_size))
        ])

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
            [lookup(ngram_list, query) for query in ngram_queries],
            device=device
        ).view(-1, vocab_size)
        for ngram_queries in all_queries
    ]
    hists = [torch.empty(0, 1, dtype=torch.long, device=device)] + [
        torch.tensor(
            ngram_queries, device=device).view(-1, nm1 + 1).t()
        for nm1, ngram_queries in enumerate(all_queries[:-1])
    ]
    del all_queries
    # the sos shouldn't matter -- it isn't in the lookup table. The lm will
    # back off to B(<sos>_) Pr(_rest), and B(<sos>_) will not exist and thus
    # be 0
    lm = layers.LookupLanguageModel(
        vocab_size, sos=sos, eos=eos, ngram_list=ngram_list)
    lm = lm.to(device)
    for exp, hist in zip(exps, hists):
        act = lm(hist)
        assert torch.allclose(exp, act, atol=1e-5)
        hist = torch.cat([
            hist,
            torch.full(
                (1,) + hist.shape[1:], eos, dtype=torch.long, device=device)
        ])
        act = lm(hist, full=True)
        assert torch.allclose(exp, act[-2], atol=1e-5)
        assert torch.allclose(torch.zeros_like(exp), act[-1])


@pytest.mark.gpu   # this is a really slow test on the cpu
def test_lookup_language_model_republic():
    device = torch.device('cuda:0')
    dir_ = os.path.join(os.path.dirname(__file__), 'republic')
    arpa_file = os.path.join(dir_, 'republic.arpa')
    assert os.path.exists(arpa_file)
    token2id_file = os.path.join(dir_, 'token2id.map')
    assert os.path.exists(token2id_file)
    queries_file = os.path.join(dir_, 'queries.txt')
    assert os.path.exists(queries_file)
    exp_file = os.path.join(dir_, 'exp.txt')
    assert os.path.exists(exp_file)
    token2id = dict()
    with open(token2id_file) as f:
        for line in f:
            token, id_ = line.strip().split()
            assert token not in token2id
            token2id[token] = int(id_)
    sos, eos, oov = token2id['<s>'], token2id['</s>'], token2id['<unk>']
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
    ngram_list = util.parse_arpa_lm(arpa_file, token2id=token2id)
    lm = layers.LookupLanguageModel(
        vocab_size, sos=sos, eos=eos, oov=oov, ngram_list=ngram_list)
    lm = lm.to(device)
    log_probs = lm(queries, full=True)
    queries = torch.cat([queries, torch.full_like(queries[:1], eos)])
    assert log_probs.shape[:-1] == queries.shape
    log_probs = log_probs.gather(2, queries.unsqueeze(2)).squeeze(2)
    assert torch.allclose(log_probs.sum(0), exp, atol=1e-5)


@pytest.mark.cpu
def test_lookup_language_model_state_dict():
    vocab_size, sos, eos = 10, -1, 0
    uni_list = [{0: 0.0, 1: 0.1, 2: 0.2}]
    lm_a = layers.LookupLanguageModel(
        vocab_size, sos=sos, eos=eos, ngram_list=uni_list)
    lm_b = layers.LookupLanguageModel(vocab_size, sos=sos, eos=eos)

    def compare(assert_same):
        same_max_ngram = (lm_a.max_ngram == lm_b.max_ngram)
        same_max_ngram_nodes = (lm_a.max_ngram_nodes == lm_b.max_ngram_nodes)
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
                lm_b.logs.masked_select(nan_mask.eq(0)), atol=1e-5)
        if assert_same:
            assert same_max_ngram
            assert same_max_ngram_nodes
            assert same_pointers
            assert same_ids
            assert same_logs
        else:
            assert not (
                same_max_ngram and same_max_ngram_nodes and same_pointers and
                same_ids and same_logs
            )

    compare(False)
    lm_b.load_state_dict(lm_a.state_dict())
    compare(True)
    bi_list = [
        {2: (0.2, -0.2), 3: (0.3, -0.3)},
        {(0, 3): 0.03, (2, 4): 0.24}
    ]
    lm_a = layers.LookupLanguageModel(
        vocab_size, sos=sos, eos=eos, ngram_list=bi_list)
    compare(False)
    lm_b.load_state_dict(lm_a.state_dict())
    compare(True)
    tri_list = [
        {0: (0.0, 0.0)},
        dict(),
        {(0, 0, 0): 0.0, (4, 4, 4): 0.444, (2, 3, 2): 0.232}
    ]
    lm_a = layers.LookupLanguageModel(
        vocab_size, sos=sos, eos=eos, ngram_list=tri_list)
    compare(False)
    lm_b.load_state_dict(lm_a.state_dict())
    compare(True)


@pytest.mark.parametrize('batch_first', [True, False])
@pytest.mark.parametrize('sub_avg', [True, False])
@pytest.mark.parametrize('reduction', ['mean', 'none'])
def test_minimum_error_rate_loss(device, batch_first, sub_avg, reduction):
    torch.manual_seed(100)
    num_batches, samples, num_classes = 5, 5, 30
    max_ref_steps, max_hyp_steps = 10, 5
    assert max_ref_steps > max_hyp_steps  # nonzero loss guaranteed
    if batch_first:
        hyp = torch.randint(
            num_classes, (num_batches, samples, max_hyp_steps), device=device)
        hyp[..., 0] = 0
        ref = torch.randint(
            num_classes, (num_batches, max_ref_steps), device=device)
        ref[..., 0] = 0
    else:
        hyp = torch.randint(
            num_classes, (max_hyp_steps, num_batches, samples), device=device)
        hyp[0] = 0
        ref = torch.randint(
            num_classes, (max_ref_steps, num_batches), device=device)
        ref[0] = 0
    log_probs = torch.randn(num_batches, samples, device=device)
    loss = layers.MinimumErrorRateLoss(
        eos=None, sub_avg=sub_avg, batch_first=batch_first,
        reduction=reduction,
    )
    l1 = loss(log_probs, ref, hyp)
    assert l1.ne(0.).any()
    l2 = loss(log_probs, ref, hyp)
    assert torch.allclose(l1, l2)
    loss.eos = 0
    l3 = loss(log_probs, ref, hyp)
    assert l3.eq(0.).all()


@pytest.mark.parametrize('batch_first', [True, False])
@pytest.mark.parametrize('eos', [None, 0])
@pytest.mark.parametrize('ref_steps_times', [1, 2])
@pytest.mark.parametrize('reduction', ['mean', 'none'])
@pytest.mark.parametrize('include_eos', [True, False])
def test_hard_optimal_completion_distillation_loss(
        device, batch_first, eos, ref_steps_times, reduction, include_eos):
    torch.manual_seed(209384)
    num_batches, max_steps, num_classes = 20, 41, 10
    if eos is None:
        hyp_lens = torch.tensor(max_steps).expand(num_batches)
        ref_lens = torch.tensor(ref_steps_times * max_steps)
        ref_lens = ref_lens.expand(num_batches)
    else:
        hyp_lens = torch.randint(1, max_steps + 1, (num_batches,))
        ref_lens = torch.randint(
            2, ref_steps_times * max_steps + 1, (num_batches,))
    ref = torch.nn.utils.rnn.pad_sequence(
        [
            torch.randint(1, num_classes, (x,))
            for x in ref_lens
        ],
        padding_value=num_classes - 1, batch_first=batch_first,
    )
    hyp = torch.nn.utils.rnn.pad_sequence(
        [
            torch.randint(1, num_classes, (x,))
            for x in hyp_lens
        ],
        padding_value=-1, batch_first=batch_first,
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
        len_mask = (
            torch.arange(hyp.shape[1]).unsqueeze(0) < hyp_lens.unsqueeze(1)
        )
    else:
        len_mask = torch.arange(hyp.shape[0]).unsqueeze(1) < hyp_lens
    logits, ref, hyp = logits.to(device), ref.to(device), hyp.to(device)
    ref_lens, hyp_lens = ref_lens.to(device), hyp_lens.to(device)
    len_mask = len_mask.to(device)
    inv_len_mask = len_mask.eq(0)
    logits.requires_grad_(True)
    loss = layers.HardOptimalCompletionDistillationLoss(
        eos=eos, include_eos=include_eos, batch_first=batch_first,
        reduction=reduction,
    )
    l1 = loss(logits, ref, hyp)
    assert torch.all(l1 == l1)  # no nans
    if reduction == 'none':
        assert torch.all(l1.masked_select(inv_len_mask).eq(0.))
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
                -log_probs[:, 0].gather(
                    1, ref[:, 0].unsqueeze(-1)
                ).squeeze(-1),
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
    g, = torch.autograd.grad([l1], [logits])
    assert torch.all(g.masked_select(inv_len_mask.unsqueeze(-1)).eq(0.))
    assert not torch.all(g.eq(0.))


@pytest.mark.parametrize('sos', [None, 0])
def test_sequential_language_model(device, sos):

    class LM(layers.SequentialLanguageModel):
        def __init__(self, vocab_size, sos=None, eos=None, oov=None):
            super(LM, self).__init__(vocab_size, eos=eos, oov=oov)
            self.embed = torch.nn.Embedding(vocab_size, vocab_size)

        def calc_last_log_probs(self, hist, eos_mask):
            if hist.shape[0]:
                out = torch.nn.functional.log_softmax(
                        self.embed(hist[-1]), dim=-1)
            else:
                assert self.sos is None
                out = -torch.full(
                    (hist.shape[1], self.vocab_size,),
                    self.vocab_size, device=device).log()
            return out

    torch.manual_seed(61094)
    S, vocab_size, max_dim, max_dim_size = 30, 20, 10, 5
    eos, oov = vocab_size - 2, vocab_size - 1
    num_dim = torch.randint(2, max_dim + 1, (1,), device=device).item()
    hist_shape = [S] + torch.randint(
        1, max_dim_size, (num_dim,), device=device).tolist()
    hist = torch.randint(
        -vocab_size // 5, (vocab_size * 6) // 5, hist_shape, device=device)
    hist = hist.masked_fill(hist.eq(eos), vocab_size + 1)
    N = torch.tensor(hist_shape[1:]).prod().item()
    first_eos_locs = torch.randint(1, hist_shape[0], (N,), device=device)
    hist.view(-1, N)[first_eos_locs, range(N)] = eos
    lm = LM(vocab_size, sos, eos, oov).to(device)
    log_probs = lm(hist, full=True)
    assert (
        list(log_probs.shape) ==
        [hist_shape[0] + 1] + hist_shape[1:] + [vocab_size]
    )
    for log_probs_n, first_eos_n in zip(
            log_probs.view(-1, N, vocab_size).transpose(0, 1),
            first_eos_locs.tolist()):
        # the index first_eos_n in log_probs refers to the probabilties of the
        # current token being whatever given the history first_eos_n - 1, so
        # we haven't yet stored the eos in history.
        assert not log_probs_n[:first_eos_n + 1].eq(0.).any()
        assert log_probs_n[first_eos_n + 1:].eq(0.).all()
    for s in range(log_probs.shape[0]):
        assert torch.allclose(log_probs[s], lm(hist[:s]))
    assert torch.allclose(lm(hist[:0], full=True).squeeze(0), log_probs[0])


@pytest.mark.parametrize('dim', [0, 1])
def test_global_soft_attention(device, dim):

    class FirstIsBest(layers.GlobalSoftAttention):

        def score(self, query, key):
            e = torch.full_like(key[..., 0], -float('inf'))
            e.narrow(self.dim, 0, 1).fill_(0.)
            return e

    class ILoveEveryoneEqually(layers.GlobalSoftAttention):

        def score(self, query, key):
            return torch.zeros_like(key[..., 0])

    torch.manual_seed(562992)
    T, max_dim, max_dim_size = 12, 10, 10
    num_dim = torch.randint(dim + 2, max_dim + 1, (1,), device=device).item()
    key_shape = torch.randint(
        1, max_dim_size + 1, (num_dim + 1,), device=device).tolist()
    key_shape[dim] = T
    query_shape = key_shape[:dim] + key_shape[dim + 1:-1]
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
    equal_attention = ILoveEveryoneEqually(
        query_size, key_size, dim).to(device)
    out1 = first_attention(query, key, key)
    assert torch.allclose(out1, key.narrow(dim, 0, 1).squeeze(dim))
    out2 = first_attention(query, key, key, mask)
    assert torch.allclose(out1, out2)
    g, = torch.autograd.grad([out1], [key], grad_outputs=torch.ones_like(out1))
    assert g.narrow(dim, 0, 1).eq(1).all()
    assert g.narrow(dim, 1, T - 1).eq(0).all()
    out1 = equal_attention(query, key, key)
    # the softmax introduces a slight numeric instability
    assert torch.allclose(out1, key.mean(dim), atol=1e-5)
    out2 = equal_attention(query, key, key, mask)
    assert not torch.allclose(out1, out2)
    exp = key.masked_fill(mask.unsqueeze(-1).eq(0), 0.)
    exp = exp.sum(dim)
    exp = exp / key_lens.float().unsqueeze(-1)
    assert torch.allclose(out2, exp, atol=1e-5)
    g, = torch.autograd.grad([out2], [key], grad_outputs=torch.ones_like(out2))
    assert g.masked_select(mask.eq(0).unsqueeze(-1)).eq(0).all()
    assert torch.allclose(
        g.sum(dim),
        torch.tensor(1., device=device),
        atol=1e-5
    )


@pytest.mark.parametrize('dim', [0, 1, 2])
def test_dot_product_soft_attention(device, dim):
    torch.manual_seed(387420)
    dim1, dim2, dim3, dim4 = 50, 30, 12, 100
    key_shape = (dim1, dim2, dim3, dim4)
    key = torch.randn(*key_shape, device=device)
    query_shape = key_shape[:dim] + key_shape[dim + 1:]
    query = torch.zeros(*query_shape, device=device)
    query[..., 0] = 2.
    exp = torch.nn.functional.softmax(key[..., 0], dim).unsqueeze(-1) * key
    exp = exp.sum(dim)
    attention = layers.DotProductSoftAttention(dim4, dim, scale_factor=.5)
    act = attention(query, key, key)
    assert torch.allclose(exp, act)


@pytest.mark.cpu
def test_dot_product_soft_attention_on_transformer_input():

    class MatrixVersion(torch.nn.Module):
        '''Scaled dot product attention, specifically for transformers

        This was blatantly ripped from `speech transformers
        <https://github.com/kaituoxu/Speech-Transformer/blob/a0bbd58da193051bb0ea597e1c4120021a721c16/src/transformer/attention.py#L65>`__.

        This is a more straightforward implementation of the scaled dot product
        attention for transformer networks. We're showing that our
        implementation yields the same output and gradient as this.
        '''

        def __init__(self, temperature):
            super(MatrixVersion, self).__init__()
            self.temperature = temperature
            self.softmax = torch.nn.Softmax(dim=2)

        def forward(self, q, k, v, mask=None):
            attn = torch.bmm(q, k.transpose(1, 2))
            attn = attn / self.temperature
            if mask is not None:
                attn = attn.masked_fill(mask, -float('inf'))
            attn = self.softmax(attn)
            output = torch.bmm(attn, v)
            return output

    torch.manual_seed(34229)
    num_batch, len_q, len_k, d_k, d_v = 30, 40, 20, 10, 50
    temp = 2.
    query = torch.randn(num_batch, len_q, d_k, requires_grad=True)
    key = torch.randn(num_batch, len_k, d_k, requires_grad=True)
    value = torch.randn(num_batch, len_k, d_v, requires_grad=True)
    matrix_attention = MatrixVersion(temp)
    our_attention = layers.DotProductSoftAttention(d_k, 1, 1 / temp)
    out1 = matrix_attention(query, key, value)
    out2 = our_attention(query, key.unsqueeze(2), value.unsqueeze(2))
    assert torch.allclose(out1, out2, atol=1e-5)
    g_q1, g_k1, g_v1 = torch.autograd.grad(
        [out1], [query, key, value], grad_outputs=torch.ones_like(out1))
    g_q2, g_k2, g_v2 = torch.autograd.grad(
        [out2], [query, key, value], grad_outputs=torch.ones_like(out2))
    assert torch.allclose(g_q1, g_q2, atol=1e-5)
    assert torch.allclose(g_k1, g_k2, atol=1e-5)
    assert torch.allclose(g_v1, g_v2, atol=1e-5)
    mask = torch.randint(2, (num_batch, len_q, len_k)).eq(1)
    out1 = matrix_attention(query, key, value, mask)
    out2 = our_attention(
        query, key.unsqueeze(2), value.unsqueeze(2),
        mask.transpose(1, 2).eq(0)  # we use the inverse of their mask
    )
    assert torch.allclose(out1, out2, atol=1e-5)


@pytest.mark.parametrize('dim', [0, 1, 2])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize(
    'layer', ['general', 'concat', 'multihead_general', 'multihead_concat'])
def test_learnable_soft_attention(device, dim, bias, layer):
    torch.manual_seed(347201)
    max_dim, max_dim_size, max_num_heads = 5, 30, 10
    num_dim = torch.randint(dim + 2, max_dim + 1, (1,), device=device).item()
    # dim size must be at least 2. Otherwise, softmax will have only one
    # element and gradient will be zero through it
    key_shape = torch.randint(
        2, max_dim_size + 1, (num_dim + 1,), device=device).tolist()
    query_shape = key_shape[:dim] + key_shape[dim + 1:-1]
    del key_shape[-2]
    key = torch.randn(*key_shape, device=device)
    query = torch.randn(*query_shape, device=device)
    key_size = key_shape[-1]
    query_size = query_shape[-1]
    if layer == 'general':
        attention = layers.GeneralizedDotProductSoftAttention(
            query_size, key_size, dim, bias)
    elif layer == 'concat':
        attention = layers.ConcatSoftAttention(
            query_size, key_size, dim, bias)
    elif layer.startswith('multihead_'):
        num_heads = torch.randint(
            1, max_num_heads + 1, (1,), device=device).item()
        d_q = max(1, query_size // num_heads)
        d_k = max(1, key_size // num_heads)
        if layer.endswith('general'):
            single_head_attention = layers.GeneralizedDotProductSoftAttention(
                d_q, d_k, dim, bias)
        elif layer.endswith('concat'):
            single_head_attention = layers.ConcatSoftAttention(
                query_size, key_size, dim, bias)
        attention = layers.MultiHeadedAttention(
            query_size, key_size, key_size, num_heads, single_head_attention,
            bias_WQ=bias, bias_WK=bias, bias_WV=bias, bias_WC=bias,
        )
    attention = attention.to(device)
    torch.manual_seed(30)
    attention.reset_parameters()
    optim = torch.optim.Adam(attention.parameters(), lr=1.)
    optim.zero_grad()
    out1 = attention(query, key, key)
    out1.mean().backward()
    optim.step()
    optim.zero_grad()
    out2 = attention(query, key, key)
    assert not torch.allclose(out1, out2, atol=1e-5)
    torch.manual_seed(30)
    attention.reset_parameters()
    out3 = attention(query, key, key)
    assert torch.allclose(out1, out3, atol=1e-5)
