from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import os
import math

from tempfile import mkdtemp
from shutil import rmtree

import torch

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


@pytest.fixture
def temp_dir():
    dir_name = mkdtemp()
    yield dir_name
    rmtree(dir_name)


@pytest.fixture(params=[
    pytest.param('cpu', marks=pytest.mark.cpu),
    pytest.param('cuda', marks=pytest.mark.gpu),
], scope='session')
def device(request):
    return request.param


@pytest.fixture(scope='session')
def populate_torch_dir():

    def _populate_torch_dir(
            dr, num_utts, min_width=1, max_width=10, num_filts=5,
            max_class=10,
            include_ali=True, file_prefix='', file_suffix='.pt', seed=1):
        torch.manual_seed(seed)
        feats_dir = os.path.join(dr, 'feats')
        ali_dir = os.path.join(dr, 'ali')
        os.makedirs(feats_dir, exist_ok=True)
        if include_ali:
            os.makedirs(ali_dir, exist_ok=True)
        feats, feat_sizes, utt_ids = [], [], []
        alis = [] if include_ali else None
        utt_id_fmt_str = '{{:0{}d}}'.format(int(math.log10(num_utts)) + 1)
        for utt_idx in range(num_utts):
            utt_id = utt_id_fmt_str.format(utt_idx)
            feat_size = torch.randint(min_width, max_width + 1, (1,)).long()
            feat_size = feat_size.item()
            feat = torch.rand(feat_size, num_filts)
            torch.save(feat, os.path.join(
                feats_dir, file_prefix + utt_id + file_suffix))
            feats.append(feat)
            feat_sizes.append(feat_size)
            utt_ids.append(utt_id)
            if include_ali:
                ali = torch.randint(max_class + 1, (feat_size,)).long()
                torch.save(ali, os.path.join(
                    ali_dir, file_prefix + utt_id + file_suffix))
                alis.append(ali)
        return feats, alis, feat_sizes, utt_ids
    return _populate_torch_dir
