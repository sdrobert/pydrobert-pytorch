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

import torch
import pytest

import pydrobert.torch.data as data


@pytest.mark.cpu
@pytest.mark.parametrize("left", [0, 1, 100])
@pytest.mark.parametrize("right", [0, 1, 100])
@pytest.mark.parametrize("T", [1, 5, 10])
def test_extract_window(left, right, T):
    signal = torch.arange(T).view(-1, 1).expand(-1, 10)
    for frame_idx in range(T):
        window = data.extract_window(signal, frame_idx, left, right)
        left_pad = max(left - frame_idx, 0)
        right_pad = max(frame_idx + right + 1 - T, 0)
        assert tuple(window.shape) == (1 + left + right, 10)
        if left_pad:
            assert torch.all(window[:left_pad] == torch.tensor([0]))
        if right_pad:
            assert torch.all(window[-right_pad:] == torch.tensor([T - 1]))
        assert torch.all(
            window[left_pad : 1 + left + right - right_pad]
            == torch.arange(
                frame_idx - left + left_pad, frame_idx + right - right_pad + 1
            )
            .view(-1, 1)
            .expand(-1, 10)
        )


@pytest.mark.cpu
@pytest.mark.parametrize("num_utts", [1, 2, 10])
@pytest.mark.parametrize("file_prefix", ["prefix_", ""])
@pytest.mark.parametrize("eos", [1000, None])
@pytest.mark.parametrize("sos", [2000, None])
@pytest.mark.parametrize("feat_dtype", [torch.float, torch.int])
def test_valid_spect_data_set(
    temp_dir, num_utts, file_prefix, populate_torch_dir, sos, eos, feat_dtype
):
    s = torch.get_rng_state()
    feats, _, _, _, _, utt_ids = populate_torch_dir(
        temp_dir,
        num_utts,
        file_prefix=file_prefix,
        include_ali=False,
        include_ref=False,
        feat_dtype=feat_dtype,
    )
    torch.set_rng_state(s)
    # note that this'll just resave the same features if there's no file
    # prefix. If there is, these ought to be ignored by the data set
    populate_torch_dir(
        temp_dir, num_utts, include_ali=False, include_ref=False, feat_dtype=feat_dtype
    )
    if not os.path.isdir(os.path.join(temp_dir, "feat", "fake")):
        os.makedirs(os.path.join(temp_dir, "feat", "fake"))
    torch.save(
        torch.randint(100, (10, 5), dtype=feat_dtype),
        os.path.join(temp_dir, "feat", "fake", file_prefix + "fake.pt"),
    )
    params = data.SpectDataParams(eos=eos)
    data_set = data.SpectDataSet(
        temp_dir,
        file_prefix=file_prefix,
        params=params,
        suppress_alis=False,
        tokens_only=False,
    )
    assert not data_set.has_ali and not data_set.has_ref
    assert len(utt_ids) == len(data_set.utt_ids)
    assert all(utt_a == utt_b for (utt_a, utt_b) in zip(utt_ids, data_set.utt_ids))
    for feat_a, (feat_b, ali_b, ref_b) in zip(feats, data_set):
        assert ali_b is None
        assert ref_b is None
        assert feat_a.shape == feat_b.shape
        assert (feat_a == feat_b).all()
    feats, alis, refs, _, _, utt_ids = populate_torch_dir(
        temp_dir, num_utts, file_prefix=file_prefix, feat_dtype=feat_dtype
    )
    if sos is not None:
        sos_sym = torch.full((3,), -1, dtype=torch.long)
        sos_sym[0] = sos
        sos_sym = sos_sym.unsqueeze(0)
        refs = [torch.cat([sos_sym, x]) for x in refs]
    if eos is not None:
        eos_sym = torch.full((3,), -1, dtype=torch.long)
        eos_sym[0] = eos
        eos_sym = eos_sym.unsqueeze(0)
        refs = [torch.cat([x, eos_sym]) for x in refs]
    params.sos = sos
    data_set = data.SpectDataSet(
        temp_dir,
        file_prefix=file_prefix,
        params=params,
        suppress_alis=False,
        tokens_only=False,
    )
    assert data_set.has_ali and data_set.has_ref
    assert len(utt_ids) == len(data_set.utt_ids)
    assert all(utt_a == utt_b for (utt_a, utt_b) in zip(utt_ids, data_set.utt_ids))
    assert all(
        torch.all(ali_a == ali_b)
        and torch.all(ref_a == ref_b)
        and feat_a.dtype == feat_b.dtype
        and torch.allclose(feat_a, feat_b)
        for ((feat_a, ali_a, ref_a), (feat_b, ali_b, ref_b)) in zip(
            zip(feats, alis, refs), data_set
        )
    )
    subset_ids = list(data_set.utt_ids[: num_utts // 2])
    params.subset_ids = subset_ids
    data_set = data.SpectDataSet(
        temp_dir,
        file_prefix=file_prefix,
        params=params,
        suppress_alis=False,
        tokens_only=False,
    )
    assert all(utt_a == utt_b for (utt_a, utt_b) in zip(subset_ids, data_set.utt_ids))
    assert all(
        torch.all(ali_a == ali_b)
        and torch.all(ref_a == ref_b)
        and torch.allclose(feat_a, feat_b)
        for ((feat_a, ali_a, ref_a), (feat_b, ali_b, ref_b)) in zip(
            zip(feats[: num_utts // 2], alis[: num_utts // 2], refs[: num_utts // 2]),
            data_set,
        )
    )


@pytest.mark.cpu
def test_spect_data_set_warnings(temp_dir):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, "feat")
    ali_dir = os.path.join(temp_dir, "ali")
    os.makedirs(feat_dir)
    os.makedirs(ali_dir)
    torch.save(torch.rand(3, 3), os.path.join(feat_dir, "a.pt"))
    torch.save(torch.rand(4, 3), os.path.join(feat_dir, "b.pt"))
    torch.save(torch.randint(10, (4,), dtype=torch.long), os.path.join(ali_dir, "b.pt"))
    torch.save(torch.randint(10, (5,), dtype=torch.long), os.path.join(ali_dir, "c.pt"))
    data_set = data.SpectDataSet(
        temp_dir, warn_on_missing=False, suppress_alis=False, tokens_only=False
    )
    assert data_set.has_ali
    assert data_set.utt_ids == ("b",)
    with pytest.warns(UserWarning) as warnings:
        data_set = data.SpectDataSet(temp_dir, suppress_alis=False, tokens_only=False)
    assert len(warnings) == 2
    assert any(str(x.message) == "Missing ali for uttid: 'a'" for x in warnings)
    assert any(str(x.message) == "Missing feat for uttid: 'c'" for x in warnings)


def test_spect_data_write_pdf(temp_dir, device):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, "feat")
    os.makedirs(feat_dir)
    torch.save(torch.rand(3, 3), os.path.join(feat_dir, "a.pt"))
    data_set = data.SpectDataSet(temp_dir, suppress_alis=False, tokens_only=False)
    z = torch.randint(10, (4, 5), dtype=torch.long)
    if device == "cuda":
        data_set.write_pdf("b", z.cuda())
    else:
        data_set.write_pdf("b", z)
    zp = torch.load(os.path.join(temp_dir, "pdfs", "b.pt"))
    assert isinstance(zp, torch.FloatTensor)
    assert torch.allclose(zp, z.float())
    data_set.write_pdf(0, torch.rand(10, 4))
    assert os.path.exists(os.path.join(temp_dir, "pdfs", "a.pt"))
    data_set.write_pdf("c", z, pdfs_dir=os.path.join(temp_dir, "foop"))
    assert os.path.exists(os.path.join(temp_dir, "foop", "c.pt"))


@pytest.mark.parametrize("eos", [None, -1])
@pytest.mark.parametrize("sos", [None, -2])
def test_spect_data_write_hyp(temp_dir, device, sos, eos):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, "feat")
    os.makedirs(feat_dir)
    torch.save(torch.rand(3, 3), os.path.join(feat_dir, "a.pt"))
    params = data.SpectDataParams(sos=sos, eos=eos)
    data_set = data.SpectDataSet(
        temp_dir, params=params, suppress_alis=False, tokens_only=False
    )
    z = torch.randint(10, (4, 3), dtype=torch.float)
    zz = z
    if sos:
        zz = torch.cat([torch.full_like(zz, sos), zz])
    if eos:
        zz = torch.cat([zz, torch.full_like(z, eos)])
    if device == "cuda":
        data_set.write_hyp("b", zz.cuda())
    else:
        data_set.write_hyp("b", zz)
    zp = torch.load(os.path.join(temp_dir, "hyp", "b.pt"))
    assert isinstance(zp, torch.LongTensor)
    assert torch.all(zp == z.long())
    data_set.write_hyp(0, torch.randint(10, (11, 3)))
    assert os.path.exists(os.path.join(temp_dir, "hyp", "a.pt"))
    data_set.write_hyp("c", z, hyp_dir=os.path.join(temp_dir, "foop"))
    assert os.path.exists(os.path.join(temp_dir, "foop", "c.pt"))


@pytest.mark.cpu
@pytest.mark.parametrize("eos", [None, 10000])
def test_spect_data_set_validity(temp_dir, eos):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, "feat")
    ali_dir = os.path.join(temp_dir, "ali")
    ref_dir = os.path.join(temp_dir, "ref")
    feats_a_pt = os.path.join(feat_dir, "a.pt")
    feats_b_pt = os.path.join(feat_dir, "b.pt")
    ali_a_pt = os.path.join(ali_dir, "a.pt")
    ali_b_pt = os.path.join(ali_dir, "b.pt")
    ref_a_pt = os.path.join(ref_dir, "a.pt")
    ref_b_pt = os.path.join(ref_dir, "b.pt")
    os.makedirs(feat_dir)
    os.makedirs(ali_dir)
    os.makedirs(ref_dir)
    torch.save(torch.rand(10, 4), feats_a_pt)
    torch.save(torch.rand(4, 4), feats_b_pt)
    torch.save(torch.randint(10, (10,), dtype=torch.long), ali_a_pt)
    torch.save(torch.randint(10, (4,), dtype=torch.long), ali_b_pt)
    torch.save(
        torch.cat(
            [
                torch.randint(10, (11, 1), dtype=torch.long),
                torch.full((11, 2), -1, dtype=torch.long),
            ],
            -1,
        ),
        ref_a_pt,
    )
    torch.save(torch.tensor([[0, 3, 4], [1, 1, 2]]), ref_b_pt)
    params = data.SpectDataParams(eos=eos)
    data_set = data.SpectDataSet(
        temp_dir, params=params, suppress_alis=False, tokens_only=False
    )
    data.validate_spect_data_set(data_set)
    torch.save(torch.rand(4, 4).long(), feats_b_pt)
    with pytest.raises(ValueError, match="not the same tensor type"):
        data.validate_spect_data_set(data_set)
    torch.save(
        torch.rand(4,), feats_b_pt,
    )
    with pytest.raises(ValueError, match="does not have two dimensions"):
        data.validate_spect_data_set(data_set)
    torch.save(torch.rand(4, 3), feats_b_pt)
    with pytest.raises(ValueError, match="has second dimension of size 3.*"):
        data.validate_spect_data_set(data_set)
    torch.save(torch.rand(4, 4), feats_b_pt)
    data.validate_spect_data_set(data_set)
    torch.save(torch.randint(10, (4,)).int(), ali_b_pt)
    with pytest.raises(ValueError, match="is not a long tensor"):
        data.validate_spect_data_set(data_set)
    with pytest.warns(UserWarning):
        data.validate_spect_data_set(data_set, True)  # will fix bad type
    data.validate_spect_data_set(data_set)  # fine after correction
    torch.save(torch.randint(10, (4, 1), dtype=torch.long), ali_b_pt)
    with pytest.raises(ValueError, match="does not have one dimension"):
        data.validate_spect_data_set(data_set)
    torch.save(torch.randint(10, (3,), dtype=torch.long), ali_b_pt)
    with pytest.raises(ValueError, match="does not have the same first"):
        data.validate_spect_data_set(data_set)
    torch.save(torch.randint(10, (4,), dtype=torch.long), ali_b_pt)
    data.validate_spect_data_set(data_set)
    torch.save(torch.Tensor([[0, 1, 2]]).int(), ref_b_pt)
    with pytest.raises(ValueError, match="is not a long tensor"):
        data.validate_spect_data_set(data_set)
    with pytest.warns(UserWarning):
        data.validate_spect_data_set(data_set, True)  # convert to long
    data.validate_spect_data_set(data_set)
    torch.save(torch.tensor([[0, -1, 2], [1, 1, 2]]), ref_b_pt)
    with pytest.raises(ValueError, match="invalid boundaries"):
        data.validate_spect_data_set(data_set)
    with pytest.warns(UserWarning):
        data.validate_spect_data_set(data_set, True)  # will remove end bound
    data.validate_spect_data_set(data_set)
    torch.save(torch.tensor([[0, 0, 1], [1, 3, 5]]), ref_b_pt)
    with pytest.raises(ValueError, match="invalid boundaries"):
        data.validate_spect_data_set(data_set)
    with pytest.warns(UserWarning):
        data.validate_spect_data_set(data_set, True)  # will trim 5 to 4
    data.validate_spect_data_set(data_set)
    torch.save(torch.tensor([[0, 0, 1], [1, 4, 5]]), ref_b_pt)
    with pytest.raises(ValueError, match="invalid boundaries"):
        data.validate_spect_data_set(data_set, True)  # will not trim b/c causes s == e
    torch.save(torch.tensor([1, 2, 3]), ref_b_pt)
    with pytest.raises(ValueError, match="were 2D"):
        data.validate_spect_data_set(data_set)
    torch.save(torch.tensor([10, 4, 2, 5]), ref_a_pt)
    data.validate_spect_data_set(data_set)


@pytest.mark.gpu
def test_validate_spect_data_set_cuda(temp_dir):
    torch.manual_seed(29)
    feat_dir = os.path.join(temp_dir, "feat")
    ali_dir = os.path.join(temp_dir, "ali")
    ref_dir = os.path.join(temp_dir, "ref")
    feats_pt = os.path.join(feat_dir, "a.pt")
    ali_pt = os.path.join(ali_dir, "a.pt")
    ref_pt = os.path.join(ref_dir, "a.pt")
    os.makedirs(feat_dir)
    os.makedirs(ali_dir)
    os.makedirs(ref_dir)
    torch.save(torch.rand(10, 5), feats_pt)
    torch.save(torch.randint(10, (10,), dtype=torch.long), ali_pt)
    torch.save(torch.tensor([1, 2, 3]), ref_pt)
    data_set = data.SpectDataSet(temp_dir, suppress_alis=False, tokens_only=False)
    data.validate_spect_data_set(data_set)
    torch.save(torch.rand(10, 5).cuda(), feats_pt)
    with pytest.raises(ValueError, match="cuda"):
        data.validate_spect_data_set(data_set)
    with pytest.warns(UserWarning):
        data.validate_spect_data_set(data_set, True)  # to CPU
    data.validate_spect_data_set(data_set)
    torch.save(torch.rand(10, 5).cuda(), feats_pt)
    torch.save(torch.randint(10, (10,), dtype=torch.long).cuda(), ali_pt)
    torch.save(torch.tensor([1, 2, 3]).cuda(), ref_pt)
    with pytest.raises(ValueError, match="cuda"):
        data.validate_spect_data_set(data_set)
    with pytest.warns(UserWarning):
        data.validate_spect_data_set(data_set, True)  # to CPU
    data.validate_spect_data_set(data_set)


@pytest.mark.cpu
@pytest.mark.parametrize("reverse", [True, False])
def test_context_window_data_set(temp_dir, reverse):
    torch.manual_seed(1)
    feat_dir = os.path.join(temp_dir, "feat")
    os.makedirs(feat_dir)
    a = torch.rand(2, 10)
    torch.save(a, os.path.join(feat_dir, "a.pt"))
    params = data.ContextWindowDataParams(
        context_left=1, context_right=1, reverse=reverse
    )
    data_set = data.ContextWindowDataSet(temp_dir, params=params)
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
