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

import pytest
import os
import math

from tempfile import mkdtemp
from shutil import rmtree

import torch


@pytest.fixture
def temp_dir():
    dir_name = mkdtemp()
    yield dir_name
    rmtree(dir_name)


@pytest.fixture(
    params=[
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=pytest.mark.gpu),
    ],
    scope="session",
)
def device(request):
    if request.param == "cuda":
        return torch.device(torch.cuda.current_device())
    else:
        return torch.device(request.param)


CUDA_AVAIL = torch.cuda.is_available()


def pytest_runtest_setup(item):
    if any(mark.name == "gpu" for mark in item.iter_markers()):
        if not CUDA_AVAIL:
            pytest.skip("cuda is not available")


@pytest.fixture(scope="session")
def populate_torch_dir():
    def _populate_torch_dir(
        dr,
        num_utts,
        min_width=1,
        max_width=10,
        num_filts=5,
        max_class=10,
        include_ali=True,
        include_ref=True,
        file_prefix="",
        file_suffix=".pt",
        seed=1,
        include_frame_shift=True,
        feat_dtype=torch.float,
    ):
        torch.manual_seed(seed)
        feat_dir = os.path.join(dr, "feat")
        ali_dir = os.path.join(dr, "ali")
        ref_dir = os.path.join(dr, "ref")
        if not os.path.isdir(feat_dir):
            os.makedirs(feat_dir)
        if include_ali and not os.path.isdir(ali_dir):
            os.makedirs(ali_dir)
        if include_ref and not os.path.isdir(ref_dir):
            os.makedirs(ref_dir)
        feats, feat_sizes, utt_ids = [], [], []
        alis = [] if include_ali else None
        refs, ref_sizes = ([], []) if include_ref else (None, None)
        utt_id_fmt_str = "{{:0{}d}}".format(int(math.log10(num_utts)) + 1)
        for utt_idx in range(num_utts):
            utt_id = utt_id_fmt_str.format(utt_idx)
            feat_size = torch.randint(min_width, max_width + 1, (1,)).long()
            feat_size = feat_size.item()
            feat = (torch.rand(feat_size, num_filts) * 1000).to(dtype=feat_dtype)
            torch.save(feat, os.path.join(feat_dir, file_prefix + utt_id + file_suffix))
            feats.append(feat)
            feat_sizes.append(feat_size)
            utt_ids.append(utt_id)
            if include_ali:
                ali = torch.randint(max_class + 1, (feat_size,)).long()
                torch.save(
                    ali, os.path.join(ali_dir, file_prefix + utt_id + file_suffix)
                )
                alis.append(ali)
            if include_ref:
                ref_size = torch.randint(1, feat_size + 1, (1,)).long().item()
                max_ref_length = torch.randint(1, feat_size + 1, (1,)).long()
                max_ref_length = max_ref_length.item()
                ref = torch.randint(100, (ref_size,)).long()
                if include_frame_shift:
                    ref_starts = torch.randint(
                        feat_size - max_ref_length + 1, (ref_size,)
                    ).long()
                    ref_lengths = torch.randint(
                        1, max_ref_length + 1, (ref_size,)
                    ).long()
                    ref = torch.stack(
                        [ref, ref_starts, ref_starts + ref_lengths], dim=-1
                    )
                torch.save(
                    ref, os.path.join(ref_dir, file_prefix + utt_id + file_suffix)
                )
                ref_sizes.append(ref_size)
                refs.append(ref)
        return feats, alis, refs, feat_sizes, ref_sizes, utt_ids

    return _populate_torch_dir
