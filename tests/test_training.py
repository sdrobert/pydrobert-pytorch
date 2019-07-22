from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import pytest
import pydrobert.torch.training as training

from pydrobert.torch.util import optimizer_to

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


class DummyModel(torch.nn.Module):
    def __init__(self, num_filts, num_classes, seed=1):
        super(DummyModel, self).__init__()
        self.seed = seed
        torch.manual_seed(seed)
        self.fc = torch.nn.Linear(num_filts, num_classes)
        self.drop = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc(x)
        return x.sum(1)  # sum out the context window

    def reset_parameters(self):
        torch.manual_seed(self.seed)
        self.fc.reset_parameters()

    @property
    def dropout(self):
        return self.drop.p

    @dropout.setter
    def dropout(self, p):
        self.drop.p = p


def test_controller_stores_and_retrieves(temp_dir, device):
    torch.manual_seed(3)
    model = DummyModel(2, 2, seed=1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    p = training.TrainingStateParams()
    state_csv_path = os.path.join(temp_dir, 'a.csv')
    state_dir = os.path.join(temp_dir, 'states')
    controller = training.TrainingStateController(
        p,
        state_csv_path=state_csv_path,
        state_dir=state_dir,
    )
    epoch_info = {
        'epoch': 10,
        'es_resume_cd': 3,
        'es_patience_cd': 4,
        'rlr_resume_cd': 10,
        'rlr_patience_cd': 5,
        'lr': 1e-7,
        'train_met': 10,
        'val_met': 4,
    }
    controller.save_model_and_optimizer_with_info(
        model, optimizer, epoch_info)
    controller.save_info_to_hist(epoch_info)
    assert controller[10] == epoch_info
    torch.manual_seed(4)
    model_2 = DummyModel(2, 2, seed=2)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=20)
    controller.load_model_and_optimizer_for_epoch(model_2, optimizer_2, 10)
    model_2.to(device)
    for parameter_1, parameter_2 in zip(
            model.parameters(), model_2.parameters()):
        assert torch.allclose(parameter_1, parameter_2)
    assert optimizer.param_groups[0]['lr'] == optimizer_2.param_groups[0]['lr']
    controller = training.TrainingStateController(
        p,
        state_csv_path=state_csv_path,
        state_dir=state_dir,
    )
    assert controller[10] == epoch_info
    torch.manual_seed(4)
    model_2 = DummyModel(2, 2, seed=2)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=20)
    controller.load_model_and_optimizer_for_epoch(model_2, optimizer_2, 10)
    model_2.to(device)
    for parameter_1, parameter_2 in zip(
            model.parameters(), model_2.parameters()):
        assert torch.allclose(parameter_1, parameter_2)
    assert optimizer.param_groups[0]['lr'] == optimizer_2.param_groups[0]['lr']


@pytest.mark.cpu
def test_controller_scheduling():

    def is_close(a, b):
        return abs(a - b) < 1e-10
    model = DummyModel(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    p = training.TrainingStateParams(
        early_stopping_threshold=0.1,
        early_stopping_patience=10,
        early_stopping_burnin=1,
        reduce_lr_threshold=0.2,
        reduce_lr_factor=.5,
        reduce_lr_patience=5,
        reduce_lr_cooldown=2,
        reduce_lr_burnin=4,
    )
    controller = training.TrainingStateController(p)
    controller.load_model_and_optimizer_for_epoch(model, optimizer)
    init_lr = optimizer.param_groups[0]['lr']
    for _ in range(8):
        assert controller.update_for_epoch(model, optimizer, 1, 1)
    assert is_close(optimizer.param_groups[0]['lr'], init_lr)
    assert controller.update_for_epoch(model, optimizer, 1, 1)
    assert is_close(optimizer.param_groups[0]['lr'], init_lr / 2)
    for _ in range(6):
        assert controller.update_for_epoch(model, optimizer, .89, .89)
    assert is_close(optimizer.param_groups[0]['lr'], init_lr / 2)
    assert controller.update_for_epoch(model, optimizer, .68, .68)
    assert is_close(optimizer.param_groups[0]['lr'], init_lr / 2)
    for _ in range(9):
        assert controller.update_for_epoch(model, optimizer, .68, .68)
    assert not controller.update_for_epoch(model, optimizer, .68, .68)


@pytest.mark.cpu
def test_controller_best(temp_dir):
    model_1 = DummyModel(100, 100, seed=1)
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=1)
    model_2 = DummyModel(100, 100, seed=2)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=2)
    model_3 = DummyModel(100, 100, seed=1)
    optimizer_3 = torch.optim.Adam(model_1.parameters(), lr=3)
    controller = training.TrainingStateController(
        training.TrainingStateParams(), state_dir=temp_dir)
    assert controller.get_best_epoch() == 0
    controller.update_for_epoch(model_1, optimizer_1, .5, .5)
    assert controller.get_best_epoch() == 1
    controller.update_for_epoch(model_2, optimizer_2, 1, 1)
    assert controller.get_best_epoch() == 1
    controller.update_for_epoch(model_2, optimizer_2, 1, 1)
    with pytest.raises(IOError):
        # neither best nor last
        controller.load_model_and_optimizer_for_epoch(model_3, optimizer_3, 2)
    controller.load_model_and_optimizer_for_epoch(model_3, optimizer_3, 1)
    for parameter_1, parameter_3 in zip(
            model_1.parameters(), model_3.parameters()):
        assert torch.allclose(parameter_1, parameter_3)
    assert optimizer_3.param_groups[0]['lr'] == 1
    controller.load_model_and_optimizer_for_epoch(model_3, optimizer_3, 3)
    for parameter_3, parameter_2 in zip(
            model_3.parameters(), model_2.parameters()):
        assert torch.allclose(parameter_3, parameter_2)
    assert optimizer_3.param_groups[0]['lr'] == 2
    controller.update_for_epoch(model_1, optimizer_1, .6, .6)
    assert controller.get_best_epoch() == 1
    controller.update_for_epoch(model_1, optimizer_1, .4, .4)
    assert controller.get_best_epoch() == 5
    controller.load_model_and_optimizer_for_epoch(model_3, optimizer_3, 5)
    for parameter_1, parameter_3 in zip(
            model_1.parameters(), model_3.parameters()):
        assert torch.allclose(parameter_1, parameter_3)
    with pytest.raises(IOError):
        # no longer the best
        controller.load_model_and_optimizer_for_epoch(model_3, optimizer_3, 1)
    # ensure we're keeping track of the last when the model name is not
    # unique
    controller = training.TrainingStateController(
        training.TrainingStateParams(
            saved_model_fmt='model.pt',
        ),
        state_dir=temp_dir
    )
    model_1.reset_parameters()
    model_2.reset_parameters()
    controller.update_for_epoch(model_1, optimizer_1, .6, .6)
    controller.update_for_epoch(model_2, optimizer_2, .4, .4)
    controller.load_model_and_optimizer_for_epoch(model_3, optimizer_3, 2)
    for parameter_2, parameter_3 in zip(
            model_2.parameters(), model_3.parameters()):
        assert torch.allclose(parameter_2, parameter_3)
    controller.update_for_epoch(model_1, optimizer_1, .5, .5)
    controller.load_model_and_optimizer_for_epoch(model_3, optimizer_3, 3)
    for parameter_1, parameter_3 in zip(
            model_1.parameters(), model_3.parameters()):
        assert torch.allclose(parameter_1, parameter_3)


@pytest.mark.parametrize('batch_first', [True, False])
@pytest.mark.parametrize('eos', [None, 0])
@pytest.mark.parametrize('ref_steps_times', [1, 2])
@pytest.mark.parametrize('sub_avg', [True, False])
@pytest.mark.parametrize('reduction', ['mean', 'none'])
def test_minimum_error_rate_loss(
        device, batch_first, eos, ref_steps_times, sub_avg, reduction):
    torch.manual_seed(100)
    num_batches, num_paths, max_steps, num_classes = 5, 5, 30, 20
    hyp_lens = torch.randint(1, max_steps + 1, (num_batches, num_paths))
    ref_lens = torch.randint(
        1, ref_steps_times * max_steps + 1,
        hyp_lens.shape
    )
    ref = torch.nn.utils.rnn.pad_sequence(
        [
            torch.randint(1, num_classes, (x,))
            for x in ref_lens.flatten()
        ],
        padding_value=-1
    ).view(-1, num_batches, num_paths)
    hyp = torch.nn.utils.rnn.pad_sequence(
        [
            torch.randint(1, num_classes, (x,))
            for x in hyp_lens.flatten()
        ],
        padding_value=num_classes - 1,
    ).view(-1, num_batches, num_paths)
    assert hyp_lens.max() == hyp.shape[0]
    if eos is not None:
        for i in range(num_batches):
            for j in range(num_paths):
                ref[ref_lens[i, j] - 1, i, j] = eos
                hyp[hyp_lens[i, j] - 1, i, j] = eos
    if batch_first:
        ref = ref.view(-1, num_batches * num_paths)
        ref = ref.t().view(num_batches, num_paths, -1).contiguous()
        hyp = hyp.view(-1, num_batches * num_paths)
        hyp = hyp.t().view(num_batches, num_paths, -1).contiguous()
        logits = torch.rand(
            num_batches, num_paths, hyp_lens.max(), num_classes)
    else:
        logits = torch.rand(
            hyp_lens.max(), num_batches, num_paths, num_classes)
    ref, hyp, logits = ref.to(device), hyp.to(device), logits.to(device)
    dist = torch.nn.functional.log_softmax(logits, 3)
    logits_on_paths = dist.gather(3, hyp.unsqueeze(3)).squeeze(3)
    log_probs = logits_on_paths.sum(2 if batch_first else 0)
    loss = training.MinimumErrorRateLoss(
        eos=eos, sub_avg=sub_avg, batch_first=batch_first, ignore_index=-1,
        reduction=reduction, lmb=0.
    )
    l1 = loss(logits, ref, hyp)
    l2 = loss(logits, ref, hyp)
    assert torch.allclose(l1, l2)
    loss.lmb = 1.
    log_probs.requires_grad_(True)
    logits.requires_grad_(True)
    l3 = loss(log_probs, ref, hyp)
    assert torch.allclose(l2, l3)
    d_log_probs_1, = torch.autograd.grad(
        [l3], [log_probs], grad_outputs=torch.ones_like(l3))
    l4 = loss(logits, ref, hyp, log_probs)
    assert not torch.allclose(l3, l4)
    d_logits_1, d_log_probs_2 = torch.autograd.grad(
        [l4], [logits, log_probs], grad_outputs=torch.ones_like(l4))
    assert torch.allclose(d_log_probs_1, d_log_probs_2)
    l5 = loss(logits, ref, hyp)
    assert torch.allclose(l4, l5)
    d_logits_2, = torch.autograd.grad(
        [l5], [logits], grad_outputs=torch.ones_like(l5))
    assert not torch.allclose(d_logits_1, d_logits_2)
