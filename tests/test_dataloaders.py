# Copyright 2022 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from itertools import repeat

import torch
import pytest

import pydrobert.torch.data as data
from pydrobert.torch.config import INDEX_PAD_VALUE


@pytest.mark.cpu
def test_epoch_random_sampler():
    data_source = torch.utils.data.TensorDataset(torch.arange(100))
    sampler = data.EpochRandomSampler(data_source, base_seed=1)
    samples_ep0 = tuple(sampler)
    samples_ep1 = tuple(sampler)
    assert samples_ep0 != samples_ep1
    assert sorted(samples_ep0) == list(range(100))
    assert sorted(samples_ep1) == list(range(100))
    assert samples_ep0 == tuple(sampler.get_samples_for_epoch(0))
    assert samples_ep1 == tuple(sampler.get_samples_for_epoch(1))
    sampler = data.EpochRandomSampler(data_source, init_epoch=10, base_seed=1)
    assert samples_ep0 == tuple(sampler.get_samples_for_epoch(0))
    assert samples_ep1 == tuple(sampler.get_samples_for_epoch(1))
    # should be reproducible if we set torch manual seed
    torch.manual_seed(5)
    sampler = data.EpochRandomSampler(data_source)
    samples_ep0 = tuple(sampler)
    torch.manual_seed(5)
    sampler = data.EpochRandomSampler(data_source)
    assert samples_ep0 == tuple(sampler)


@pytest.mark.cpu
@pytest.mark.parametrize("drop_incomplete", [True, False])
def test_bucket_batch_sampler(drop_incomplete):
    N = 5
    sampler = torch.utils.data.SequentialSampler([1] * (6 * N + 3))
    idx2bucket = dict((n, int(n % 2 == 0) + 2 * int(n % 3 == 0)) for n in sampler)
    bucket2size = {0: 2, 1: 2, 2: 1, 3: 1}
    sampler = data.BucketBatchSampler(sampler, idx2bucket, bucket2size, drop_incomplete)
    sampler_ = iter(sampler)
    count = 0
    for n in range(0, 6 * N, 6):
        assert next(sampler_) == [n]
        assert next(sampler_) == [n + 3]
        assert next(sampler_) == [n + 2, n + 4]
        assert next(sampler_) == [n + 1, n + 5]
        count += 4
    assert next(sampler_) == [6 * N]
    count += 1
    if not drop_incomplete:
        assert next(sampler_) == [6 * N + 1]
        assert next(sampler_) == [6 * N + 2]
        count += 2
    with pytest.raises(StopIteration):
        next(sampler_)
    assert len(sampler) == count


@pytest.mark.cpu
@pytest.mark.parametrize(
    "feat_sizes",
    [((3, 5, 4), (4, 5, 4), (1, 5, 4)), ((2, 10, 5),) * 10],
    ids=["short", "long"],
)
@pytest.mark.parametrize("include_ali", [True, False])
def test_context_window_seq_to_batch(feat_sizes, include_ali):
    feats = tuple(torch.rand(*x) for x in feat_sizes)
    if include_ali:
        alis = tuple(torch.randint(10, (x[0],), dtype=torch.long) for x in feat_sizes)
    else:
        alis = repeat(None)
    seq = zip(feats, alis)
    batch_feats, batch_ali = data.context_window_seq_to_batch(seq)
    assert torch.allclose(torch.cat(feats), batch_feats)
    if include_ali:
        assert torch.all(torch.cat(alis) == batch_ali)
    else:
        assert batch_ali is None


@pytest.mark.cpu
@pytest.mark.parametrize("include_ali", [True, False])
@pytest.mark.parametrize(
    "include_ref,include_frame_shift", [(True, True), (True, False), (False, None)]
)
@pytest.mark.parametrize("batch_first", [True, False])
def test_spect_seq_to_batch(include_ali, include_ref, batch_first, include_frame_shift):
    feat_sizes = tuple(
        torch.randint(1, 30, (1,)).item()
        for _ in range(torch.randint(3, 10, (1,)).item())
    )
    feats = tuple(torch.randn(x, 5) for x in feat_sizes)
    if include_ali:
        alis = tuple(torch.randint(100, (x,), dtype=torch.long) for x in feat_sizes)
    else:
        alis = repeat(None)
    if include_ref:
        ref_sizes = tuple(
            torch.randint(1, 30, (1,)).item() for _ in range(len(feat_sizes))
        )
        extra_dim = (3,) if include_frame_shift else tuple()
        refs = tuple(
            torch.randint(100, (x,) + extra_dim, dtype=torch.long) for x in ref_sizes
        )
    else:
        ref_sizes = repeat(None)
        refs = repeat(None)
    (
        batch_feats,
        batch_ali,
        batch_ref,
        batch_feat_sizes,
        batch_ref_sizes,
    ) = data.spect_seq_to_batch(zip(feats, alis, refs), batch_first=batch_first)
    feat_sizes, feats, alis, refs, ref_sizes = zip(
        *sorted(zip(feat_sizes, feats, alis, refs, ref_sizes), key=lambda x: -x[0])
    )
    assert torch.all(torch.tensor(feat_sizes) == batch_feat_sizes)
    if not batch_first:
        batch_feats = batch_feats.transpose(0, 1)
        if include_ali:
            batch_ali = batch_ali.transpose(0, 1)
        if include_ref:
            batch_ref = batch_ref.transpose(0, 1)
    assert all(
        torch.allclose(a[: b.shape[0]], b)
        and torch.allclose(a[b.shape[0] :], torch.tensor([0.0]))
        for (a, b) in zip(batch_feats, feats)
    )
    if include_ali:
        assert all(
            torch.all(a[: b.shape[0]] == b)
            and torch.all(a[b.shape[0] :] == torch.tensor([INDEX_PAD_VALUE]))
            for (a, b) in zip(batch_ali, alis)
        )
    else:
        assert batch_ali is None
    if include_ref:
        assert torch.all(torch.tensor(ref_sizes) == batch_ref_sizes)
        assert all(
            torch.all(a[: b.shape[0]] == b)
            and torch.all(a[b.shape[0] :] == torch.tensor([INDEX_PAD_VALUE]))
            for (a, b) in zip(batch_ref, refs)
        )
    else:
        assert batch_ref is None
        assert batch_ref_sizes is None


@pytest.mark.cpu
@pytest.mark.parametrize("eos", [None, -1])
@pytest.mark.parametrize("sos", [None, -2])
@pytest.mark.parametrize("split_params", [True, False])
@pytest.mark.parametrize("include_frame_shift", [True, False])
@pytest.mark.parametrize("feat_dtype", [torch.float, torch.int])
def test_spect_training_data_loader(
    temp_dir,
    populate_torch_dir,
    sos,
    eos,
    split_params,
    include_frame_shift,
    feat_dtype,
):
    num_utts, batch_size, num_filts = 20, 5, 11
    populate_torch_dir(
        temp_dir,
        num_utts,
        num_filts=num_filts,
        include_frame_shift=include_frame_shift,
        feat_dtype=feat_dtype,
    )
    if split_params:
        params = data.DynamicLengthDataLoaderParams(batch_size=batch_size)
        data_params = data.SpectDataParams(sos=sos, eos=eos)
    else:
        params = data.SpectDataLoaderParams(batch_size=batch_size, sos=sos, eos=eos)
        data_params = None
    # check missing either ali or ref gives None in batches
    data_loader = data.SpectTrainingDataLoader(
        temp_dir, params, data_params=data_params, ali_subdir=None, seed=2
    )
    assert next(iter(data_loader))[1] is None
    data_loader = data.SpectTrainingDataLoader(
        temp_dir, params, data_params=data_params, ref_subdir=None, seed=2
    )
    assert next(iter(data_loader))[2] is None
    assert next(iter(data_loader))[4] is None
    data_loader = data.SpectTrainingDataLoader(
        temp_dir, params, data_params=data_params, seed=2
    )

    def _get_epoch(sort):
        ep_feats, ep_ali, ep_ref = [], [], []
        ep_feat_sizes, ep_ref_sizes = [], []
        max_T = 0
        max_R = 0
        batch_first = data_loader.batch_first
        for b_feats, b_ali, b_ref, b_feat_sizes, b_ref_sizes in data_loader:
            if not batch_first:
                b_feats = b_feats.transpose(0, 1)
                b_ali = b_ali.transpose(0, 1)
                b_ref = b_ref.transpose(0, 1)
            max_T = max(max_T, b_feat_sizes[0])
            R_star = max(b_ref_sizes)
            max_R = max(max_R, R_star)
            assert b_feats.shape[0] == batch_size
            assert b_ali.shape[0] == batch_size
            assert b_ref.shape[0] == batch_size
            assert b_feats.shape[-1] == num_filts
            assert b_feats.shape[1] == b_feat_sizes[0]
            assert b_ali.shape[1] == b_feat_sizes[0]
            assert b_ref.shape[1] == R_star
            assert b_ref.dim() == (3 if include_frame_shift else 2)
            ep_feats += tuple(b_feats)
            ep_ali += tuple(b_ali)
            ep_ref += tuple(b_ref)
            ep_feat_sizes += tuple(b_feat_sizes)
            ep_ref_sizes += tuple(b_ref_sizes)
        assert len(ep_feats) == num_utts
        assert len(ep_ali) == num_utts
        for i in range(num_utts):
            ep_feats[i] = torch.nn.functional.pad(
                ep_feats[i], (0, 0, 0, max_T - ep_ali[i].shape[0])
            )
            ep_ali[i] = torch.nn.functional.pad(
                ep_ali[i], (0, max_T - ep_ali[i].shape[0]), value=INDEX_PAD_VALUE
            )
            if include_frame_shift:
                ep_ref[i] = torch.nn.functional.pad(
                    ep_ref[i],
                    (0, 0, 0, max_R - ep_ref[i].shape[0]),
                    value=INDEX_PAD_VALUE,
                )
            else:
                ep_ref[i] = torch.nn.functional.pad(
                    ep_ref[i], (0, max_R - ep_ref[i].shape[0]), value=INDEX_PAD_VALUE
                )
        if sort:
            ep_feats, ep_ali, ep_ref, ep_feat_sizes, ep_ref_sizes = zip(
                *sorted(
                    zip(ep_feats, ep_ali, ep_ref, ep_feat_sizes, ep_ref_sizes),
                    key=lambda x: (-x[3], -x[4], x[0][0, 0]),
                )
            )
        return ep_feats, ep_ali, ep_ref, ep_feat_sizes, ep_ref_sizes

    def _compare_epochs(ep_a, ep_b, same):
        a_feats, a_ali, a_ref, a_feat_sizes, a_ref_sizes = ep_a
        b_feats, b_ali, b_ref, b_feat_sizes, b_ref_sizes = ep_b
        a_feats, b_feats = torch.stack(a_feats), torch.stack(b_feats)
        a_ali, b_ali = torch.stack(a_ali), torch.stack(b_ali)
        a_ref, b_ref = torch.stack(a_ref), torch.stack(b_ref)
        if same:
            assert a_feat_sizes == b_feat_sizes
            assert a_ref_sizes == b_ref_sizes
            assert torch.allclose(a_feats, b_feats)
            assert torch.all(a_ali == b_ali)
            assert torch.all(a_ref == b_ref)
        else:
            assert a_feat_sizes != b_feat_sizes
            assert a_ref_sizes != b_ref_sizes
            assert not torch.allclose(a_feats, b_feats)
            assert torch.any(a_ali != b_ali)
            assert torch.any(a_ref != b_ref)

    ep0 = _get_epoch(False)
    ep1 = _get_epoch(False)
    _compare_epochs(ep0, ep1, False)  # could be same by fluke
    _compare_epochs(_get_epoch(True), _get_epoch(True), True)
    data_loader.epoch = 1
    _compare_epochs(ep1, _get_epoch(False), True)
    # XXX(sdrobert): warning spit out on CI if num_workers > 2
    data_loader = data.SpectTrainingDataLoader(
        temp_dir, params, data_params=data_params, num_workers=2, seed=2
    )
    _compare_epochs(ep0, _get_epoch(False), True)
    _compare_epochs(ep1, _get_epoch(False), True)
    data_loader.batch_first = False
    data_loader.epoch = 0
    _compare_epochs(ep0, _get_epoch(False), True)
    _compare_epochs(ep1, _get_epoch(False), True)


@pytest.mark.cpu
@pytest.mark.parametrize("eos", [None, -1])
@pytest.mark.parametrize("sos", [None, -2])
@pytest.mark.parametrize("split_params", [True, False])
@pytest.mark.parametrize("include_frame_shift", [True, False])
@pytest.mark.parametrize("feat_dtype", [torch.float, torch.int])
def test_spect_evaluation_data_loader(
    temp_dir,
    populate_torch_dir,
    sos,
    eos,
    split_params,
    include_frame_shift,
    feat_dtype,
):
    feat_dir = os.path.join(temp_dir, "feat")
    ali_dir = os.path.join(temp_dir, "ali")
    os.makedirs(feat_dir)
    os.makedirs(ali_dir)
    batch_size = 5
    if split_params:
        params = data.DynamicLengthDataLoaderParams(batch_size=batch_size)
        data_params = data.SpectDataParams(sos=sos, eos=eos)
    else:
        params = data.SpectDataLoaderParams(batch_size=batch_size, sos=sos, eos=eos)
        data_params = None
    feats, ali, ref, feat_sizes, ref_sizes, utt_ids = populate_torch_dir(
        temp_dir, 20, include_frame_shift=include_frame_shift, feat_dtype=feat_dtype
    )
    if sos is not None:
        if include_frame_shift:
            sos_sym = torch.full((3,), -1, dtype=torch.long)
            sos_sym[0] = sos
            sos_sym = sos_sym.unsqueeze(0)
        else:
            sos_sym = torch.full((1,), sos, dtype=torch.long)
        ref = [torch.cat([sos_sym, x], 0) for x in ref]
        ref_sizes = [x + 1 for x in ref_sizes]
    if eos is not None:
        if include_frame_shift:
            eos_sym = torch.full((3,), eos, dtype=torch.long)
            eos_sym[0] = eos
            eos_sym = eos_sym.unsqueeze(0)
        else:
            eos_sym = torch.full((1,), eos, dtype=torch.long)
        ref = [torch.cat([x, eos_sym], 0) for x in ref]
        ref_sizes = [x + 1 for x in ref_sizes]
    # check that ali and ref can be missing
    data_loader = data.SpectEvaluationDataLoader(
        temp_dir, params, data_params=data_params, ali_subdir=None, ref_subdir=None
    )
    assert next(iter(data_loader))[1:3] == (None, None)
    assert next(iter(data_loader))[4] is None
    data_loader = data.SpectEvaluationDataLoader(
        temp_dir, params, data_params=data_params
    )

    def _compare_data_loader():
        batch_first = data_loader.batch_first
        assert len(data_loader) == 4
        cur_idx = 0
        for (
            b_feats,
            b_ali,
            b_ref,
            b_feat_sizes,
            b_ref_sizes,
            b_utt_ids,
        ) in data_loader:
            if not batch_first:
                b_feats = b_feats.transpose(0, 1)
                b_ali = b_ali.transpose(0, 1)
                b_ref = b_ref.transpose(0, 1)
            R_star = max(b_ref_sizes)
            assert tuple(b_feats.shape) == (5, b_feat_sizes[0], 5)
            assert tuple(b_ali.shape) == (5, b_feat_sizes[0])
            if include_frame_shift:
                assert tuple(b_ref.shape) == (5, R_star, 3)
            else:
                assert tuple(b_ref.shape) == (5, R_star)
            # sort the sub-section of the master list by feature size
            s_feats, s_ali, s_ref, s_feat_sizes, s_ref_sizes, s_utt_ids = zip(
                *sorted(
                    zip(
                        feats[cur_idx : cur_idx + 5],
                        ali[cur_idx : cur_idx + 5],
                        ref[cur_idx : cur_idx + 5],
                        feat_sizes[cur_idx : cur_idx + 5],
                        ref_sizes[cur_idx : cur_idx + 5],
                        utt_ids[cur_idx : cur_idx + 5],
                    ),
                    key=lambda x: -x[3],
                )
            )
            assert b_utt_ids == s_utt_ids
            assert tuple(b_feat_sizes) == s_feat_sizes
            assert tuple(b_ref_sizes) == s_ref_sizes
            for a, b in zip(b_feats, s_feats):
                assert torch.allclose(a[: b.shape[0]], b)
                assert torch.allclose(
                    a[b.shape[0] :], torch.tensor([0], dtype=feat_dtype)
                )
            for a, b in zip(b_ali, s_ali):
                assert torch.all(a[: b.shape[0]] == b)
                assert torch.all(a[b.shape[0] :] == torch.tensor([INDEX_PAD_VALUE]))
            for a, b in zip(b_ref, s_ref):
                assert torch.all(a[: b.shape[0]] == b)
                assert torch.all(a[b.shape[0] :] == torch.tensor([INDEX_PAD_VALUE]))
            cur_idx += 5

    _compare_data_loader()
    _compare_data_loader()  # order should not change
    data_loader = data.SpectEvaluationDataLoader(
        temp_dir, params, data_params=data_params, num_workers=2
    )
    _compare_data_loader()  # order should still not change
    data_loader.batch_first = False
    _compare_data_loader()


@pytest.mark.cpu
@pytest.mark.parametrize("split_params", [True, False])
def test_window_training_data_loader(temp_dir, populate_torch_dir, split_params):
    populate_torch_dir(temp_dir, 5, num_filts=2)
    seed, batch_size, context_left, context_right = 2, 5, 1, 1
    if split_params:
        params = data.DataLoaderParams(batch_size=batch_size, drop_last=True)
        data_params = data.ContextWindowDataParams(
            context_left=context_left, context_right=context_right
        )
    else:
        params = data.ContextWindowDataLoaderParams(
            context_left=context_left,
            context_right=context_right,
            batch_size=batch_size,
            drop_last=True,
        )
        data_params = None
    data_loader = data.ContextWindowTrainingDataLoader(
        temp_dir, params, data_params=data_params, seed=seed
    )
    total_windows_ep0 = 0
    for feat, ali in data_loader:
        windows = feat.shape[0]
        assert tuple(feat.shape) == (windows, 3, 2)
        assert tuple(ali.shape) == (windows,)
        total_windows_ep0 += windows
    assert total_windows_ep0 >= batch_size
    feats_ep1_a, alis_ep1_a = [], []
    total_windows_ep1 = 0
    for feats, alis in data_loader:
        windows = feat.shape[0]
        assert tuple(feat.shape) == (windows, 3, 2)
        assert tuple(ali.shape) == (windows,)
        feats_ep1_a.append(feats)
        alis_ep1_a.append(alis)
        total_windows_ep1 += windows
    assert total_windows_ep0 == total_windows_ep1
    data_loader = data.ContextWindowTrainingDataLoader(
        temp_dir,
        params,
        init_epoch=1,
        data_params=data_params,
        num_workers=2,
        seed=seed,
    )
    feats_ep1_b, alis_ep1_b = [], []
    for feats, alis in data_loader:
        feats_ep1_b.append(feats)
        alis_ep1_b.append(alis)
    assert all(
        torch.allclose(feats_a, feats_b)
        for (feats_a, feats_b) in zip(feats_ep1_a, feats_ep1_b)
    )
    assert all(
        torch.all(alis_a == alis_b) for (alis_a, alis_b) in zip(alis_ep1_a, alis_ep1_b)
    )
    data_loader.epoch = 1
    feats_ep1_c, alis_ep1_c = [], []
    for feats, alis in data_loader:
        feats_ep1_c.append(feats)
        alis_ep1_c.append(alis)
    assert all(
        torch.allclose(feats_a, feats_c)
        for (feats_a, feats_c) in zip(feats_ep1_a, feats_ep1_c)
    )
    assert all(
        torch.all(alis_a == alis_c) for (alis_a, alis_c) in zip(alis_ep1_a, alis_ep1_c)
    )


@pytest.mark.cpu
@pytest.mark.parametrize("split_params", [True, False])
def test_window_evaluation_data_loader(temp_dir, populate_torch_dir, split_params):
    feat_dir = os.path.join(temp_dir, "feat")
    ali_dir = os.path.join(temp_dir, "ali")
    os.makedirs(feat_dir)
    os.makedirs(ali_dir)
    if split_params:
        params = data.DataLoaderParams(batch_size=5)
        data_params = data.ContextWindowDataParams(context_left=1, context_right=1)
    else:
        params = data.ContextWindowDataLoaderParams(
            context_left=1, context_right=1, batch_size=5
        )
        data_params = None
    feats, alis, _, feat_sizes, _, utt_ids = populate_torch_dir(
        temp_dir, 20, include_ref=False
    )

    def _compare_data_loader(data_loader):
        assert len(data_loader) == 4
        cur_idx = 0
        for b_feats, b_alis, b_feat_sizes, b_utt_ids in data_loader:
            assert tuple(b_feats.shape[1:]) == (3, 5)
            assert b_feats.shape[0] == sum(b_feat_sizes)
            assert tuple(b_utt_ids) == tuple(utt_ids[cur_idx : cur_idx + 5])
            assert torch.allclose(
                b_feats[:, 1], torch.cat(feats[cur_idx : cur_idx + 5])
            )
            assert torch.all(b_alis == torch.cat(alis[cur_idx : cur_idx + 5]))
            cur_idx += 5

    data_loader = data.ContextWindowEvaluationDataLoader(
        temp_dir, params, data_params=data_params, ali_subdir=None
    )
    # check batching works when alignments are empty
    assert next(iter(data_loader))[1] is None
    data_loader = data.ContextWindowEvaluationDataLoader(
        temp_dir, params, data_params=data_params
    )
    _compare_data_loader(data_loader)
    _compare_data_loader(data_loader)  # order should not change
    data_loader = data.ContextWindowEvaluationDataLoader(
        temp_dir, params, data_params=data_params, num_workers=2
    )
    _compare_data_loader(data_loader)  # order should still not change


@pytest.mark.cpu
@pytest.mark.parametrize(
    "loader_cls",
    [data.SpectTrainingDataLoader, data.SpectEvaluationDataLoader],
    ids=["train", "eval"],
)
def test_data_loader_length_buckets(temp_dir, populate_torch_dir, loader_cls):
    NN, N, B = 31, 3, 5
    exp_feat_sizes = populate_torch_dir(temp_dir, NN, max_width=NN, include_ref=False)[
        3
    ]
    assert len(exp_feat_sizes) == NN
    exp_feat_sizes = sorted(exp_feat_sizes)
    params = data.SpectDataLoaderParams(
        batch_size=N, num_length_buckets=B, drop_last=False
    )
    loader = loader_cls(temp_dir, params)
    act_feat_sizes = [x[3] for x in loader]
    assert len(act_feat_sizes) == len(loader)
    for i, x in enumerate(act_feat_sizes):
        assert x.numel() == N or (i >= (NN // N - B) and x.numel() < N)
        i = exp_feat_sizes.index(x[0].item())
        b = (B * i) // NN
        assert b < B
        ui = NN - 1 if b == (B - 1) else (b + 1) * (NN // B) - 1
        li = b * (NN // B) - 1
        upper = exp_feat_sizes[ui]
        lower = exp_feat_sizes[li] if li > -1 else -1
        assert ((lower < x) & (x <= upper)).all()
    act_feat_sizes = torch.cat(act_feat_sizes)
    assert sorted(act_feat_sizes.tolist()) == exp_feat_sizes
    params.drop_last = True
    params.size_batch_by_length = True
    m = N * exp_feat_sizes[-1]
    loader = loader_cls(temp_dir, params)
    act_feat_sizes = [x[3] for x in loader]
    assert len(act_feat_sizes) == len(loader)
    for x in act_feat_sizes:
        i = exp_feat_sizes.index(x[0].item())
        b = (B * i) // NN
        assert b < B
        ui = NN - 1 if b == (B - 1) else (b + 1) * (NN // B) - 1
        upper = exp_feat_sizes[ui]
        assert upper * x.numel() <= m
        assert upper * (x.numel() + 1) > m


@pytest.mark.cpu
def test_pydrobert_param_optuna_hooks():
    poptuna = pytest.importorskip("pydrobert.param.optuna")
    optuna = pytest.importorskip("optuna")
    for class_ in (
        data.DataLoaderParams,
        data.SpectDataLoaderParams,
        data.ContextWindowDataParams,
        data.ContextWindowDataLoaderParams,
    ):
        assert issubclass(class_, poptuna.TunableParameterized)
    global_dict = {
        "data_set": data.DataLoaderParams(),
        "spect_data": data.SpectDataParams(),
        "spect_data_set": data.SpectDataLoaderParams(),
        "context_window_data": data.ContextWindowDataParams(),
        "context_window_data_set": data.ContextWindowDataLoaderParams(),
    }
    assert {
        "data_set.batch_size",
        "spect_data.eos",
        "spect_data_set.batch_size",
        "context_window_data.reverse",
        "context_window_data_set.batch_size",
    } - poptuna.get_param_dict_tunable(global_dict) == {"spect_data.eos"}

    def objective(trial):
        param_dict = poptuna.suggest_param_dict(trial, global_dict)
        return param_dict["data_set"].batch_size

    sampler = optuna.samplers.RandomSampler(seed=5)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10)
    assert not {
        "data_set.batch_size",
        "spect_data_set.batch_size",
        "context_window_data.reverse",
        "context_window_data_set.batch_size",
    } - set(study.best_params)
    assert study.best_params["data_set.batch_size"] < 7
