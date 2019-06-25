from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

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
    torch.save(torch.rand(1, 5), os.path.join(temp_dir, 'feat', 'utt19.pt'))
    torch.save(torch.tensor([10]), os.path.join(temp_dir, 'ali', 'utt19.pt'))
    torch.save(
        torch.tensor([[100, 0, 1]]), os.path.join(temp_dir, 'ref', 'utt19.pt'))
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
    assert table['max_ali_class'] == 10
    assert table['max_ref_class'] == 100
    for class_idx in range(11):
        key = 'count_{:02d}'.format(class_idx)
        assert table[key] == alis.count(class_idx)
    # invalidate the data set and try again
    torch.save(torch.rand(1, 4), os.path.join(temp_dir, 'feat', 'utt19.pt'))
    with pytest.raises(ValueError):
        command_line.get_torch_spect_data_dir_info(
            [temp_dir, table_path, '--strict'])


def _write_token2id(path, swap):
    with open(path, 'w') as f:
        for v in range(ord('a'), ord('z') + 1):
            if swap:
                f.write('{} {}\n'.format(v - ord('a'), chr(v)))
            else:
                f.write('{} {}\n'.format(chr(v), v - ord('a')))


@pytest.mark.cpu
@pytest.mark.parametrize('tokens', ['token2id', 'id2token'])
def test_trn_to_torch_token_data_dir(temp_dir, tokens):
    trn_path = os.path.join(temp_dir, 'ref.trn')
    tokens_path = os.path.join(temp_dir, 'token2id')
    ref_dir = os.path.join(temp_dir, 'ref')
    _write_token2id(tokens_path, tokens == 'id2token')
    with open(trn_path, 'w') as trn:
        trn.write('''\
a b b c (utt1)
(utt2)

d { e / f } g (utt3)
{{{h / i} / j} / k} (utt4)
''')
    with warnings.catch_warnings(record=True):
        assert not command_line.trn_to_torch_token_data_dir(
            [trn_path, tokens_path, ref_dir, '--alt-handler=first'] +
            (['--swap'] if tokens == 'id2token' else [])
        )
    act_utt1 = torch.load(os.path.join(ref_dir, 'utt1.pt'))
    assert torch.all(act_utt1 == torch.tensor([
        [0, -1, -1], [1, -1, -1], [1, -1, -1], [2, -1, -1]]))
    act_utt2 = torch.load(os.path.join(ref_dir, 'utt2.pt'))
    assert not act_utt2.numel()
    act_utt3 = torch.load(os.path.join(ref_dir, 'utt3.pt'))
    assert torch.all(act_utt3 == torch.tensor([
        [3, -1, -1], [4, -1, -1], [6, -1, -1]]))
    act_utt4 = torch.load(os.path.join(ref_dir, 'utt4.pt'))
    assert torch.all(act_utt4 == torch.tensor([[7, -1, -1]]))


@pytest.mark.cpu
@pytest.mark.parametrize('tokens', ['token2id', 'id2token'])
def test_torch_token_data_dir_to_trn(temp_dir, tokens):
    torch.manual_seed(1000)
    num_utts = 100
    max_tokens = 10
    num_digits = torch.log10(torch.tensor(float(num_utts))).long().item() + 1
    utt_fmt = 'utt{{:0{}d}}'.format(num_digits)
    trn_path = os.path.join(temp_dir, 'ref.trn')
    tokens_path = os.path.join(temp_dir, 'id2token')
    ref_dir = os.path.join(temp_dir, 'ref')
    _write_token2id(tokens_path, tokens == 'id2token')
    if not os.path.isdir(ref_dir):
        os.makedirs(ref_dir)
    exps = []
    for utt_idx in range(num_utts):
        utt_id = utt_fmt.format(utt_idx)
        num_tokens = torch.randint(max_tokens + 1, (1,)).long().item()
        ids = torch.randint(26, (num_tokens,)).long()
        tok = torch.stack([ids] + ([torch.full_like(ids, -1)] * 2), -1)
        torch.save(tok, os.path.join(ref_dir, utt_id + '.pt'))
        transcript = ' '.join([chr(x + ord('a')) for x in ids.tolist()])
        transcript += ' ({})'.format(utt_id)
        exps.append(transcript)
    assert not command_line.torch_token_data_dir_to_trn(
        [ref_dir, tokens_path, trn_path] +
        (['--swap'] if tokens == 'token2id' else [])
    )
    with open(trn_path, 'r') as trn:
        acts = trn.readlines()
    assert len(exps) == len(acts)
    for exp, act in zip(exps, acts):
        assert exp.strip() == act.strip()


def _write_wc2utt(path, swap, chan):
    fmt = 'u_{0} w_{0} {1}\n' if swap else 'w_{0} {1} u_{0}\n'
    with open(path, 'w') as f:
        for utt_idx in range(1000):
            f.write(fmt.format(utt_idx, chan))


@pytest.mark.cpu
@pytest.mark.parametrize('tokens', ['token2id', 'id2token'])
@pytest.mark.parametrize('channels', ['wc2utt', 'utt2wc', None])
def test_ctm_to_torch_token_data_dir(temp_dir, tokens, channels):
    ctm_path = os.path.join(temp_dir, 'ref.ctm')
    tokens_path = os.path.join(temp_dir, tokens)
    channels_path = os.path.join(temp_dir, channels) if channels else None
    ref_dir = os.path.join(temp_dir, 'ref')
    _write_token2id(tokens_path, tokens == 'id2token')
    if channels:
        _write_wc2utt(channels_path, channels == 'utt2wc', 'A')
    with open(ctm_path, 'w') as ctm:
        ctm.write('''\
;; some text
w_1 A 0.1 1.0 a

w_1 A 0.2 1.0 b
w_1 A 0.3 1.0 c   ;; ignore this comment
w_2 A 0.0 0.0 b
w_3 A 0.0 1000.0 d
w_3 A 1.0 0.1 d
''')
    args = [ctm_path, tokens_path, ref_dir]
    if tokens == 'id2token':
        args.append('--swap')
    if channels == 'utt2wc':
        args.append('--utt2wc={}'.format(channels_path))
    elif channels == 'wc2utt':
        args.append('--wc2utt={}'.format(channels_path))
    assert not command_line.ctm_to_torch_token_data_dir(args)
    act_utt1 = torch.load(
        os.path.join(ref_dir, 'u_1.pt' if channels else 'w_1.pt'))
    assert torch.all(act_utt1 == torch.tensor([
        [0, 10, 110], [1, 20, 120], [2, 30, 130]]))
    act_utt2 = torch.load(
        os.path.join(ref_dir, 'u_2.pt' if channels else 'w_2.pt'))
    assert torch.all(act_utt2 == torch.tensor([[1, 0, 0]]))
    act_utt3 = torch.load(
        os.path.join(ref_dir, 'u_3.pt' if channels else 'w_3.pt'))
    assert torch.all(act_utt3 == torch.tensor([
        [3, 0, 100000], [3, 100, 110]]))
