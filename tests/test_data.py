from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from itertools import repeat, chain

import pytest
import torch
import torch.utils.data
import pydrobert.torch.data as data

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


@pytest.mark.cpu
@pytest.mark.parametrize("left", [0, 1, 100])
@pytest.mark.parametrize("right", [0, 1, 100])
@pytest.mark.parametrize("T", [1, 5, 10])
def test_extract_window(left, right, T):
    # FIXME(sdrobert): the float conversion is due to a bug in torch.allclose.
    # Fix when fixed
    signal = torch.arange(T).float().view(-1, 1).expand(-1, 10)
    for frame_idx in range(T):
        window = data.extract_window(signal, frame_idx, left, right)
        left_pad = max(left - frame_idx, 0)
        right_pad = max(frame_idx + right + 1 - T, 0)
        assert tuple(window.shape) == (1 + left + right, 10)
        if left_pad:
            assert torch.allclose(window[:left_pad], torch.tensor([0]).float())
        if right_pad:
            assert torch.allclose(
                window[-right_pad:], torch.tensor([T - 1]).float())
        assert torch.allclose(
            window[left_pad:1 + left + right - right_pad],
            torch.arange(
                frame_idx - left + left_pad,
                frame_idx + right - right_pad + 1
            ).float().view(-1, 1).expand(-1, 10)
        )


@pytest.mark.cpu
@pytest.mark.parametrize('num_utts', [1, 2, 10])
@pytest.mark.parametrize('file_prefix', ['prefix_', ''])
def test_valid_spect_data_set(
        temp_dir, num_utts, file_prefix, populate_torch_dir):
    feats, _, _, _, _, utt_ids = populate_torch_dir(
        temp_dir, num_utts, file_prefix=file_prefix,
        include_ali=False, include_ref=False)
    # note that this'll just resave the same features if there's no file
    # prefix. If there is, these ought to be ignored by the data set
    populate_torch_dir(
        temp_dir, num_utts, include_ali=False, include_ref=False)
    data_set = data.SpectDataSet(temp_dir, file_prefix=file_prefix)
    assert not data_set.has_ali and not data_set.has_ref
    assert len(utt_ids) == len(data_set.utt_ids)
    assert all(
        utt_a == utt_b for (utt_a, utt_b) in zip(utt_ids, data_set.utt_ids))
    assert all(
        ali_b is None and ref_b is None and torch.allclose(feat_a, feat_b)
        for (feat_a, (feat_b, ali_b, ref_b)) in zip(feats, data_set)
    )
    feats, alis, refs, _, _, utt_ids = populate_torch_dir(
        temp_dir, num_utts, file_prefix=file_prefix)
    data_set = data.SpectDataSet(temp_dir, file_prefix=file_prefix)
    assert data_set.has_ali and data_set.has_ref
    assert len(utt_ids) == len(data_set.utt_ids)
    assert all(
        utt_a == utt_b for (utt_a, utt_b) in zip(utt_ids, data_set.utt_ids))
    assert all(
        torch.all(ali_a == ali_b) and
        torch.all(ref_a == ref_b) and
        torch.allclose(feat_a, feat_b)
        for ((feat_a, ali_a, ref_a), (feat_b, ali_b, ref_b))
        in zip(zip(feats, alis, refs), data_set)
    )
    subset_ids = data_set.utt_ids[:num_utts // 2]
    data_set = data.SpectDataSet(
        temp_dir, file_prefix=file_prefix, subset_ids=set(subset_ids))
    assert all(
        utt_a == utt_b for (utt_a, utt_b) in zip(subset_ids, data_set.utt_ids))
    assert all(
        torch.all(ali_a == ali_b) and
        torch.all(ref_a == ref_b) and
        torch.allclose(feat_a, feat_b)
        for ((feat_a, ali_a, ref_a), (feat_b, ali_b, ref_b))
        in zip(
            zip(
                feats[:num_utts // 2],
                alis[:num_utts // 2],
                refs[:num_utts // 2]),
            data_set)
    )


@pytest.mark.cpu
def test_spect_data_set_warnings(temp_dir):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, 'feat')
    ali_dir = os.path.join(temp_dir, 'ali')
    os.makedirs(feat_dir)
    os.makedirs(ali_dir)
    torch.save(torch.rand(3, 3), os.path.join(feat_dir, 'a.pt'))
    torch.save(torch.rand(4, 3), os.path.join(feat_dir, 'b.pt'))
    torch.save(torch.randint(10, (4,)).long(), os.path.join(ali_dir, 'b.pt'))
    torch.save(torch.randint(10, (5,)).long(), os.path.join(ali_dir, 'c.pt'))
    data_set = data.SpectDataSet(temp_dir, warn_on_missing=False)
    assert data_set.has_ali
    assert data_set.utt_ids == ('b',)
    with pytest.warns(UserWarning) as warnings:
        data_set = data.SpectDataSet(temp_dir)
    assert len(warnings) == 2
    assert any(
        str(x.message) == "Missing ali for uttid: 'a'" for x in warnings)
    assert any(
        str(x.message) == "Missing feat for uttid: 'c'" for x in warnings)


def test_spect_data_write_pdf(temp_dir, device):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, 'feat')
    os.makedirs(feat_dir)
    torch.save(torch.rand(3, 3), os.path.join(feat_dir, 'a.pt'))
    data_set = data.SpectDataSet(temp_dir)
    z = torch.randint(10, (4, 5)).long()
    if device == 'cuda':
        data_set.write_pdf('b', z.cuda())
    else:
        data_set.write_pdf('b', z)
    zp = torch.load(os.path.join(temp_dir, 'pdfs', 'b.pt'))
    assert isinstance(zp, torch.FloatTensor)
    assert torch.allclose(zp, z.float())
    data_set.write_pdf(0, torch.rand(10, 4))
    assert os.path.exists(os.path.join(temp_dir, 'pdfs', 'a.pt'))
    data_set.write_pdf('c', z, pdfs_dir=os.path.join(temp_dir, 'foop'))
    assert os.path.exists(os.path.join(temp_dir, 'foop', 'c.pt'))


def test_spect_data_write_hyp(temp_dir, device):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, 'feat')
    os.makedirs(feat_dir)
    torch.save(torch.rand(3, 3), os.path.join(feat_dir, 'a.pt'))
    data_set = data.SpectDataSet(temp_dir)
    z = torch.randint(10, (4, 3))
    if device == 'cuda':
        data_set.write_hyp('b', z.cuda())
    else:
        data_set.write_hyp('b', z)
    zp = torch.load(os.path.join(temp_dir, 'hyp', 'b.pt'))
    assert isinstance(zp, torch.LongTensor)
    assert torch.all(zp == z.long())
    data_set.write_hyp(0, torch.randint(10, (11, 3)))
    assert os.path.exists(os.path.join(temp_dir, 'hyp', 'a.pt'))
    data_set.write_hyp('c', z, hyp_dir=os.path.join(temp_dir, 'foop'))
    assert os.path.exists(os.path.join(temp_dir, 'foop', 'c.pt'))


@pytest.mark.cpu
def test_spect_data_set_validity(temp_dir):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, 'feat')
    ali_dir = os.path.join(temp_dir, 'ali')
    feats_a_pt = os.path.join(feat_dir, 'a.pt')
    feats_b_pt = os.path.join(feat_dir, 'b.pt')
    ali_a_pt = os.path.join(ali_dir, 'a.pt')
    ali_b_pt = os.path.join(ali_dir, 'b.pt')
    os.makedirs(feat_dir)
    os.makedirs(ali_dir)
    torch.save(torch.rand(10, 4), feats_a_pt)
    torch.save(torch.rand(4, 4), feats_b_pt)
    torch.save(torch.randint(10, (10,)).long(), ali_a_pt)
    torch.save(torch.randint(10, (4,)).long(), ali_b_pt)
    data_set = data.SpectDataSet(temp_dir)
    data.validate_spect_data_set(data_set)
    torch.save(torch.rand(4, 4).long(), feats_b_pt)
    with pytest.raises(ValueError, match='is not a FloatTensor'):
        data.validate_spect_data_set(data_set)
    torch.save(torch.rand(4,), feats_b_pt)
    with pytest.raises(ValueError, match='does not have two axes'):
        data.validate_spect_data_set(data_set)
    torch.save(torch.rand(4, 3), feats_b_pt)
    with pytest.raises(ValueError, match='has second axis size 3.*'):
        data.validate_spect_data_set(data_set)
    torch.save(torch.rand(4, 4), feats_b_pt)
    data.validate_spect_data_set(data_set)
    torch.save(torch.randint(10, (4,)).int(), ali_b_pt)
    with pytest.raises(ValueError, match='is not a LongTensor'):
        data.validate_spect_data_set(data_set)
    torch.save(torch.randint(10, (4, 1)).long(), ali_b_pt)
    with pytest.raises(ValueError, match='does not have one axis'):
        data.validate_spect_data_set(data_set)
    torch.save(torch.randint(10, (3,)).long(), ali_b_pt)
    with pytest.raises(ValueError, match='does not have the same first'):
        data.validate_spect_data_set(data_set)
    torch.save(torch.randint(10, (4,)).long(), ali_b_pt)
    data.validate_spect_data_set(data_set)


@pytest.mark.cpu
@pytest.mark.parametrize('reverse', [True, False])
def test_context_window_data_set(temp_dir, reverse):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, 'feat')
    os.makedirs(feat_dir)
    a = torch.rand(2, 10)
    torch.save(a, os.path.join(feat_dir, 'a.pt'))
    data_set = data.ContextWindowDataSet(temp_dir, 1, 1, reverse=reverse)
    windowed, _ = data_set[0]
    assert tuple(windowed.shape) == (2, 3, 10)
    if reverse:
        # [[a1, a0, a0], [a1, a1, a0]]
        assert torch.allclose(a[0], windowed[0, 1:])
        assert torch.allclose(a[1], windowed[0, 0])
        assert torch.allclose(a[0], windowed[1, 2])
        assert torch.allclose(a[1], windowed[1, :2])
    else:
        # [[a0, a0, a1], [a0, a1, a1]]
        assert torch.allclose(a[0], windowed[0, :2])
        assert torch.allclose(a[1], windowed[0, 2])
        assert torch.allclose(a[0], windowed[1, 0])
        assert torch.allclose(a[1], windowed[1, 1:])


@pytest.mark.cpu
def test_epoch_random_sampler(temp_dir):
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
@pytest.mark.parametrize('feat_sizes', [
    ((3, 5, 4), (4, 5, 4), (1, 5, 4)),
    ((2, 10, 5),) * 10,
], ids=['short', 'long'])
@pytest.mark.parametrize('include_ali', [True, False])
def test_context_window_seq_to_batch(feat_sizes, include_ali):
    torch.manual_seed(1)
    feats = tuple(torch.rand(*x) for x in feat_sizes)
    num_frames = sum(x[0] for x in feat_sizes)
    if include_ali:
        alis = tuple(torch.randint(10, (x[0],)).long() for x in feat_sizes)
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
@pytest.mark.parametrize('include_ali', [True, False])
@pytest.mark.parametrize('include_ref', [True, False])
def test_spect_seq_to_batch(include_ali, include_ref):
    torch.manual_seed(1)
    feat_sizes = tuple(
        torch.randint(1, 30, (1,)).long().item()
        for _ in range(torch.randint(3, 10, (1,)).long().item())
    )
    feats = tuple(torch.randn(x, 5) for x in feat_sizes)
    if include_ali:
        alis = tuple(torch.randint(100, (x,)).long() for x in feat_sizes)
    else:
        alis = repeat(None)
    if include_ref:
        ref_sizes = tuple(
            torch.randint(1, 30, (1,)).long().item()
            for _ in range(len(feat_sizes))
        )
        refs = tuple(torch.randint(100, (x, 3)).long() for x in ref_sizes)
    else:
        ref_sizes = repeat(None)
        refs = repeat(None)
    batch_feats, batch_ali, batch_ref, batch_feat_sizes, batch_ref_sizes = (
        data.spect_seq_to_batch(zip(feats, alis, refs)))
    feat_sizes, feats, alis, refs, ref_sizes = zip(*sorted(
        zip(feat_sizes, feats, alis, refs, ref_sizes), key=lambda x: -x[0]))
    assert torch.all(torch.tensor(feat_sizes) == batch_feat_sizes)
    assert all(
        torch.allclose(a[:b.shape[0]], b) and
        torch.allclose(a[b.shape[0]:], torch.tensor([0.]))
        for (a, b) in zip(batch_feats, feats)
    )
    if include_ali:
        assert all(
            torch.all(a[:b.shape[0]] == b) and
            torch.all(a[b.shape[0]:] == torch.tensor([data.ALI_PAD_VALUE]))
            for (a, b) in zip(batch_ali, alis)
        )
    else:
        assert batch_ali is None
    if include_ref:
        assert torch.all(torch.tensor(ref_sizes) == batch_ref_sizes)
        assert all(
            torch.all(a[:b.shape[0]] == b) and
            torch.all(a[b.shape[0]:] == torch.tensor([data.REF_PAD_VALUE]))
            for (a, b) in zip(batch_ref, refs)
        )
    else:
        assert batch_ref is None
        assert batch_ref_sizes is None


@pytest.mark.cpu
def test_spect_training_data_loader(temp_dir, populate_torch_dir):
    torch.manual_seed(40)
    num_utts, batch_size, num_filts = 20, 5, 11
    populate_torch_dir(temp_dir, num_utts, num_filts=num_filts)
    p = data.SpectDataSetParams(
        batch_size=batch_size,
        seed=2,
    )
    # check missing either ali or ref gives None in batches
    data_loader = data.SpectTrainingDataLoader(temp_dir, p, ali_subdir=None)
    assert next(iter(data_loader))[1] is None
    data_loader = data.SpectTrainingDataLoader(temp_dir, p, ref_subdir=None)
    assert next(iter(data_loader))[2] is None
    assert next(iter(data_loader))[4] is None
    data_loader = data.SpectTrainingDataLoader(temp_dir, p)

    def _get_epoch(sort):
        ep_feats, ep_ali, ep_ref = [], [], []
        ep_feat_sizes, ep_ref_sizes = [], []
        max_T = 0
        max_R = 0
        for b_feats, b_ali, b_ref, b_feat_sizes, b_ref_sizes in data_loader:
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
            ep_feats += tuple(b_feats)
            ep_ali += tuple(b_ali)
            ep_ref += tuple(b_ref)
            ep_feat_sizes += tuple(b_feat_sizes)
            ep_ref_sizes += tuple(b_ref_sizes)
        assert len(ep_feats) == num_utts
        assert len(ep_ali) == num_utts
        for i in range(num_utts):
            ep_feats[i] = torch.nn.functional.pad(
                ep_feats[i], (0, 0, 0, max_T - ep_ali[i].shape[0]))
            ep_ali[i] = torch.nn.functional.pad(
                ep_ali[i], (0, max_T - ep_ali[i].shape[0]),
                value=data.ALI_PAD_VALUE)
            ep_ref[i] = torch.nn.functional.pad(
                ep_ref[i], (0, 0, 0, max_R - ep_ref[i].shape[0]),
                value=data.REF_PAD_VALUE)
        if sort:
            ep_feats, ep_ali, ep_ref, ep_feat_sizes, ep_ref_sizes = zip(
                *sorted(
                    zip(ep_feats, ep_ali, ep_ref, ep_feat_sizes, ep_ref_sizes),
                    key=lambda x: (-x[3], -x[4], x[0][0, 0])))
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
    data_loader = data.SpectTrainingDataLoader(temp_dir, p, num_workers=4)
    _compare_epochs(ep0, _get_epoch(False), True)
    _compare_epochs(ep1, _get_epoch(False), True)


@pytest.mark.cpu
def test_spect_evaluation_data_loader(temp_dir, populate_torch_dir):
    torch.manual_seed(41)
    feat_dir = os.path.join(temp_dir, 'feat')
    ali_dir = os.path.join(temp_dir, 'ali')
    os.makedirs(feat_dir)
    os.makedirs(ali_dir)
    p = data.SpectDataSetParams(batch_size=5)
    feats, ali, ref, feat_sizes, ref_sizes, utt_ids = populate_torch_dir(
        temp_dir, 20)
    # check that ali and ref can be missing
    data_loader = data.SpectEvaluationDataLoader(
        temp_dir, p, ali_subdir=None, ref_subdir=None)
    assert next(iter(data_loader))[1:3] == (None, None)
    assert next(iter(data_loader))[4] is None
    data_loader = data.SpectEvaluationDataLoader(temp_dir, p)

    def _compare_data_loader():
        assert len(data_loader) == 4
        cur_idx = 0
        for (
                b_feats, b_ali, b_ref, b_feat_sizes, b_ref_sizes, b_utt_ids
                ) in data_loader:
            R_star = max(b_ref_sizes)
            assert tuple(b_feats.shape) == (5, b_feat_sizes[0], 5)
            assert tuple(b_ali.shape) == (5, b_feat_sizes[0])
            assert tuple(b_ref.shape) == (5, R_star, 3)
            # sort the sub-section of the master list by feature size
            s_feats, s_ali, s_ref, s_feat_sizes, s_ref_sizes, s_utt_ids = (
                zip(*sorted(
                    zip(
                        feats[cur_idx:cur_idx + 5],
                        ali[cur_idx:cur_idx + 5],
                        ref[cur_idx:cur_idx + 5],
                        feat_sizes[cur_idx:cur_idx + 5],
                        ref_sizes[cur_idx:cur_idx + 5],
                        utt_ids[cur_idx:cur_idx + 5]),
                    key=lambda x: -x[3])))
            assert b_utt_ids == s_utt_ids
            assert tuple(b_feat_sizes) == s_feat_sizes
            assert tuple(b_ref_sizes) == s_ref_sizes
            for a, b in zip(b_feats, s_feats):
                assert torch.allclose(a[:b.shape[0]], b)
                assert torch.allclose(a[b.shape[0]:], torch.tensor([0.]))
            for a, b in zip(b_ali, s_ali):
                assert torch.all(a[:b.shape[0]] == b)
                assert torch.all(a[b.shape[0]:] == torch.tensor(
                    [data.ALI_PAD_VALUE]))
            for a, b in zip(b_ref, s_ref):
                assert torch.all(a[:b.shape[0]] == b)
                assert torch.all(a[b.shape[0]:] == torch.tensor(
                    [data.REF_PAD_VALUE]))
            cur_idx += 5

    _compare_data_loader()
    _compare_data_loader()  # order should not change
    data_loader = data.SpectEvaluationDataLoader(temp_dir, p, num_workers=4)
    _compare_data_loader()  # order should still not change


@pytest.mark.cpu
def test_window_training_data_loader(temp_dir, populate_torch_dir):
    populate_torch_dir(temp_dir, 5, num_filts=2)
    p = data.ContextWindowDataSetParams(
        context_left=1,
        context_right=1,
        batch_size=5,
        seed=2,
        drop_last=True,
    )
    data_loader = data.ContextWindowTrainingDataLoader(temp_dir, p)
    total_windows_ep0 = 0
    for feat, ali in data_loader:
        windows = feat.shape[0]
        assert tuple(feat.shape) == (windows, 3, 2)
        assert tuple(ali.shape) == (windows,)
        total_windows_ep0 += windows
    assert total_windows_ep0 >= 5
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
        temp_dir, p,
        init_epoch=1,
        num_workers=4,
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
        torch.allclose(alis_a.float(), alis_b.float())
        for (alis_a, alis_b) in zip(alis_ep1_a, alis_ep1_b)
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
        torch.allclose(alis_a.float(), alis_c.float())
        for (alis_a, alis_c) in zip(alis_ep1_a, alis_ep1_c)
    )


@pytest.mark.cpu
def test_window_evaluation_data_loader(temp_dir, populate_torch_dir):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, 'feat')
    ali_dir = os.path.join(temp_dir, 'ali')
    os.makedirs(feat_dir)
    os.makedirs(ali_dir)
    p = data.ContextWindowDataSetParams(
        context_left=1,
        context_right=1,
        batch_size=5,
    )
    feats, alis, _, feat_sizes, _, utt_ids = populate_torch_dir(
        temp_dir, 20, include_ref=False)

    def _compare_data_loader(data_loader):
        assert len(data_loader) == 4
        cur_idx = 0
        for b_feats, b_alis, b_feat_sizes, b_utt_ids in data_loader:
            assert tuple(b_feats.shape[1:]) == (3, 5)
            assert b_feats.shape[0] == sum(b_feat_sizes)
            assert tuple(b_utt_ids) == tuple(utt_ids[cur_idx:cur_idx + 5])
            assert torch.allclose(
                b_feats[:, 1], torch.cat(feats[cur_idx:cur_idx + 5]))
            assert torch.all(
                b_alis == torch.cat(alis[cur_idx:cur_idx + 5]))
            cur_idx += 5
    data_loader = data.ContextWindowEvaluationDataLoader(
        temp_dir, p, ali_subdir=None)
    # check batching works when alignments are empty
    assert next(iter(data_loader))[1] is None
    data_loader = data.ContextWindowEvaluationDataLoader(temp_dir, p)
    _compare_data_loader(data_loader)
    _compare_data_loader(data_loader)  # order should not change
    data_loader = data.ContextWindowEvaluationDataLoader(
        temp_dir, p, num_workers=4)
    _compare_data_loader(data_loader)  # order should still not change
