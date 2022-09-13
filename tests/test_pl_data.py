# Copyright 2022 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
import pytest

import pydrobert.torch.command_line as cmd


@pytest.fixture(scope="session")
def populate_lit_dir(request, populate_torch_dir):
    def _populate_lit_dir(
        root_dir,
        num_filts=5,
        max_ali_class=9,
        max_ref_class=99,
        train_utts=100,
        dev_utts=10,
        test_utts=20,
        predict_utts=None,
        include_ali=False,
        include_ref=True,
        with_mvn=True,
        **kwargs,
    ):
        x = [("train", train_utts), ("dev", dev_utts), ("test", test_utts)]
        if predict_utts is not None:
            x.append(("predict", predict_utts))
        if not include_ali:
            max_ali_class = -1
        if not include_ref:
            max_ref_class = -1
        for part, num_utts in x:
            dir_ = os.path.join(root_dir, part)
            os.makedirs(dir_, exist_ok=True)
            populate_torch_dir(
                dir_,
                num_utts,
                num_filts=num_filts,
                max_ali_class=max_ali_class,
                max_ref_class=max_ref_class,
                include_ali=include_ali,
                include_ref=include_ref,
                **kwargs,
            )
        with open(os.path.join(root_dir, "info.ark"), "w") as f:
            f.write(f"num_filts {num_filts}\n")
            f.write(f"max_ali_class {max_ali_class}\n")
            f.write(f"max_ref_class {max_ref_class}\n")
        params = dict(
            train_dir=f"{root_dir}/train",
            val_dir=f"{root_dir}/dev",
            test_dir=f"{root_dir}/test",
            info_path=f"{root_dir}/info.ark",
        )
        if with_mvn:
            assert not cmd.compute_mvn_stats_for_torch_feat_data_dir(
                [f"{root_dir}/train/feat", f"{root_dir}/mvn.pt"]
            )
            params["mvn_path"] = f"{root_dir}/mvn.pt"
        return params

    return _populate_lit_dir


def test_lit_spect_data_module_basic(temp_dir, populate_lit_dir):
    pl = pytest.importorskip("pytorch_lightning")
    import pydrobert.torch.lightning as plightning

    tN, VN, TN, N, F, A, V = 101, 11, 21, 10, 5, 9, 10
    params = plightning.LitSpectDataModuleParams(
        **populate_lit_dir(f"{temp_dir}/data", F, A, V - 1, tN, VN, TN)
    )
    params.prefer_split = False
    params.train_params.batch_size = N
    params.train_params.drop_last = True
    data = plightning.LitSpectDataModule(params)
    assert data.vocab_size is None
    data.prepare_data()
    assert data.vocab_size == V
    assert data.feat_size == F
    assert data.test_set is None
    assert data.val_set is None
    assert data.train_set is None
    assert data.predict_set is None
    data.setup()
    assert len(data.train_set) == tN
    assert len(data.val_set) == VN
    assert len(data.test_set) == TN
    assert len(data.predict_set) == TN
    pl.seed_everything(0)
    feat_lens, ref_lens = [], []
    for feat, ref, feat_len, ref_len in data.train_dataloader():
        assert feat.shape[1:] == (N, F)
        assert ref.shape[1:] == (N,)
        feat_lens.append(feat_len)
        ref_lens.append(ref_len)
        assert feat_len.shape == ref_len.shape == (N,)
    feat_lens_0, feat_lens = torch.cat(feat_lens), []
    ref_lens_0, ref_lens = torch.cat(ref_lens), []
    pl.seed_everything(0)
    for _, _, feat_len, ref_len in data.train_dataloader():
        feat_lens.append(feat_len)
        ref_lens.append(ref_len)
    feat_lens_1 = torch.cat(feat_lens)
    ref_lens_1 = torch.cat(ref_lens)
    assert (feat_lens_0 == feat_lens_1).all()
    assert (ref_lens_0 == ref_lens_1).all()
