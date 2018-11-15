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
