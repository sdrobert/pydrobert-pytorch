from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pytest
import torch
import pydrobert.torch.command_line as command_line

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


@pytest.mark.cpu
def test_get_torch_spect_data_dir_info(temp_dir, populate_torch_dir):
    _, alis, _, feat_sizes, _, _ = populate_torch_dir(
        temp_dir, 19, num_filts=5, max_class=10)
    # add one with class idx 10 to ensure all classes are accounted for
    torch.save(torch.rand(1, 5), os.path.join(temp_dir, 'feats', 'utt19.pt'))
    torch.save(torch.tensor([10]), os.path.join(temp_dir, 'ali', 'utt19.pt'))
    torch.save(
        torch.tensor([[0, 0, 1]]), os.path.join(temp_dir, 'ref', 'utt19.pt'))
    feat_sizes += (1,)
    alis = torch.cat(alis + [torch.tensor([10])])
    alis = [class_idx.item() for class_idx in alis]
    table_path = os.path.join(temp_dir, 'info')
    assert not command_line.get_torch_spect_data_dir_info(
        [temp_dir, table_path, '--strict'])
    table = dict()
    with open(table_path) as table_file:
        for line in table_file:
            line = line.split()
            table[line[0]] = int(line[1])
    assert table['num_utterances'] == 20
    assert table['total_frames'] == sum(feat_sizes)
    assert table['num_filts'] == 5
    for class_idx in range(11):
        key = 'count_{:02d}'.format(class_idx)
        assert table[key] == alis.count(class_idx)
    # invalidate the data set and try again
    torch.save(torch.rand(1, 4), os.path.join(temp_dir, 'feats', 'utt19.pt'))
    with pytest.raises(ValueError):
        command_line.get_torch_spect_data_dir_info(
            [temp_dir, table_path, '--strict'])
