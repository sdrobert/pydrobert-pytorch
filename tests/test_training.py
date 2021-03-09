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

import os

import torch
import pytest
import pydrobert.torch.training as training


@pytest.mark.parametrize('opt_class', [
    torch.optim.Adam,
    torch.optim.Adagrad,
    torch.optim.LBFGS,
    torch.optim.SGD,
])
def test_controller_stores_and_retrieves(temp_dir, device, opt_class):
    torch.manual_seed(50)
    model = torch.nn.Linear(2, 2).to(device)
    optimizer = opt_class(model.parameters(), lr=20)
    p = training.TrainingStateParams(seed=5, log10_learning_rate=-1)
    state_csv_path = os.path.join(temp_dir, 'a.csv')
    state_dir = os.path.join(temp_dir, 'states')
    controller = training.TrainingStateController(
        p,
        state_csv_path=state_csv_path,
        state_dir=state_dir,
    )
    controller.add_entry('cool_guy_entry', int)
    controller.load_model_and_optimizer_for_epoch(model, optimizer, 0)
    assert optimizer.param_groups[0]['lr'] == 10 ** p.log10_learning_rate
    inp = torch.randn(5, 2, device=device)

    def closure():
        optimizer.zero_grad()
        loss = model(inp).sum()
        loss.backward()
        return loss

    model_2 = torch.nn.Linear(2, 2).to(device)
    optimizer_2 = opt_class(model_2.parameters(), lr=20)
    controller.load_model_and_optimizer_for_epoch(model_2, optimizer_2, 0)
    assert optimizer_2.param_groups[0]['lr'] == 10 ** p.log10_learning_rate
    for parameter_1, parameter_2 in zip(
            model.parameters(), model_2.parameters()):
        assert parameter_1.device == device
        assert parameter_2.device == device
        assert torch.allclose(parameter_1, parameter_2)
    optimizer.step(closure)
    for parameter_1, parameter_2 in zip(
            model.parameters(), model_2.parameters()):
        assert not torch.allclose(parameter_1, parameter_2)

    def closure():
        optimizer_2.zero_grad()
        loss = model_2(inp).sum()
        loss.backward()
        return loss

    optimizer_2.step(closure)
    for parameter_1, parameter_2 in zip(
            model.parameters(), model_2.parameters()):
        assert torch.allclose(parameter_1, parameter_2)
    epoch_info = {
        'epoch': 10,
        'es_resume_cd': 3,
        'es_patience_cd': 4,
        'rlr_resume_cd': 10,
        'rlr_patience_cd': 5,
        'lr': 1e-7,
        'train_met': 10,
        'val_met': 4,
        'cool_guy_entry': 30,
    }
    controller.save_model_and_optimizer_with_info(model, optimizer, epoch_info)
    controller.save_info_to_hist(epoch_info)
    assert controller[10] == epoch_info
    torch.manual_seed(4)
    model_2 = torch.nn.Linear(2, 2).to(device)
    optimizer_2 = opt_class(model_2.parameters(), lr=20)
    controller.load_model_and_optimizer_for_epoch(model_2, optimizer_2, 10)
    for parameter_1, parameter_2 in zip(
            model.parameters(), model_2.parameters()):
        assert parameter_1.device == device
        assert parameter_2.device == device
        assert torch.allclose(parameter_1, parameter_2)
    optimizer_2.step(closure)
    for parameter_1, parameter_2 in zip(
            model.parameters(), model_2.parameters()):
        assert not torch.allclose(parameter_1, parameter_2)
    controller = training.TrainingStateController(
        p,
        state_csv_path=state_csv_path,
        state_dir=state_dir,
    )
    assert 'cool_guy_entry' not in controller[10]
    assert controller[10]['es_resume_cd'] == epoch_info['es_resume_cd']
    controller.add_entry('cool_guy_entry', int)
    assert controller[10] == epoch_info
    model_3 = torch.nn.Linear(2, 2).to(device)
    optimizer_3 = opt_class(model_3.parameters(), lr=20)
    controller.load_model_and_optimizer_for_epoch(model_3, optimizer_3, 10)
    model_3.to(device)
    for parameter_1, parameter_3 in zip(
            model.parameters(), model_3.parameters()):
        assert parameter_3.device == device
        assert torch.allclose(parameter_1, parameter_3)

    def closure():
        optimizer_3.zero_grad()
        loss = model_3(inp).sum()
        loss.backward()
        return loss

    optimizer_3.step(closure)
    for parameter_1, parameter_2, parameter_3 in zip(
            model.parameters(), model_2.parameters(), model_3.parameters()):
        assert not torch.allclose(parameter_1, parameter_2)
        assert torch.allclose(parameter_2, parameter_3)
    torch.manual_seed(300)
    model_2 = torch.nn.Linear(2, 2).to(device)
    optimizer_2 = opt_class(model_2.parameters(), lr=20)
    epoch_info['epoch'] = 3
    epoch_info['val_met'] = 2
    controller.save_model_and_optimizer_with_info(
        model_2, optimizer_2, epoch_info)
    controller.save_info_to_hist(epoch_info)
    # by default, load_model_and_optimizer_for_epoch loads last
    controller.load_model_and_optimizer_for_epoch(model_3, optimizer_3)
    for parameter_1, parameter_2, parameter_3 in zip(
            model.parameters(), model_2.parameters(), model_3.parameters()):
        assert torch.allclose(parameter_1, parameter_3)
        assert not torch.allclose(parameter_2, parameter_3)
    # by default, load_model_for_epoch loads best
    controller.load_model_for_epoch(model_3)
    for parameter_1, parameter_2, parameter_3 in zip(
            model.parameters(), model_2.parameters(), model_3.parameters()):
        assert not torch.allclose(parameter_1, parameter_3)
        assert torch.allclose(parameter_2, parameter_3)


@pytest.mark.cpu
def test_controller_stops_at_num_epochs():
    num_epochs = 10
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    params = training.TrainingStateParams(
        num_epochs=num_epochs, early_stopping_threshold=0.0)
    controller = training.TrainingStateController(params)
    for _ in range(9):
        assert controller.update_for_epoch(model, optimizer, 0.1, 0.1)
        assert controller.continue_training()
    assert not controller.update_for_epoch(model, optimizer, 0.1, 0.1)
    assert not controller.continue_training()


@pytest.mark.cpu
def test_controller_scheduling():

    def is_close(a, b):
        return abs(a - b) < 1e-10
    model = torch.nn.Linear(2, 2)
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
        assert controller.continue_training()
    assert is_close(optimizer.param_groups[0]['lr'], init_lr)
    assert controller.update_for_epoch(model, optimizer, 1, 1)
    assert is_close(optimizer.param_groups[0]['lr'], init_lr / 2)
    for _ in range(6):
        assert controller.update_for_epoch(model, optimizer, .89, .89)
        assert controller.continue_training()
    assert is_close(optimizer.param_groups[0]['lr'], init_lr / 2)
    assert controller.update_for_epoch(model, optimizer, .68, .68)
    assert controller.continue_training()
    assert is_close(optimizer.param_groups[0]['lr'], init_lr / 2)
    for _ in range(9):
        assert controller.update_for_epoch(model, optimizer, .68, .68)
        assert controller.continue_training()
    assert not controller.update_for_epoch(model, optimizer, .68, .68)
    assert not controller.continue_training()
    p.early_stopping_threshold = 0.0
    p.reduce_lr_threshold = 0.0
    controller = training.TrainingStateController(p)
    controller.load_model_and_optimizer_for_epoch(model, optimizer)
    init_lr = optimizer.param_groups[0]['lr']
    for _ in range(20):
        assert controller.update_for_epoch(model, optimizer, 0., 0.)
        assert controller.continue_training()
    assert is_close(optimizer.param_groups[0]['lr'], init_lr)


@pytest.mark.cpu
def test_controller_best(temp_dir):
    torch.manual_seed(10)
    model_1 = torch.nn.Linear(100, 100)
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=1)
    model_2 = torch.nn.Linear(100, 100)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=2)
    model_3 = torch.nn.Linear(100, 100)
    optimizer_3 = torch.optim.Adam(model_1.parameters(), lr=3)
    training.TrainingStateController.SCIENTIFIC_PRECISION = 5
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
    # round-on-even dictates .400005 will round to .40000
    controller.update_for_epoch(model_1, optimizer_1, .400005, .400005)
    assert controller.get_best_epoch() == 5
    controller.load_model_and_optimizer_for_epoch(model_3, optimizer_3, 5)
    for parameter_1, parameter_3 in zip(
            model_1.parameters(), model_3.parameters()):
        assert torch.allclose(parameter_1, parameter_3)
    with pytest.raises(IOError):
        # no longer the best
        controller.load_model_and_optimizer_for_epoch(model_3, optimizer_3, 1)
    # this block ensures that negligible differences in the loss aren't being
    # considered "better." This is necessary to remain consistent
    # with the truncated floats saved to history
    controller.update_for_epoch(model_1, optimizer_1, .4, .4)
    # last
    controller.load_model_and_optimizer_for_epoch(model_3, optimizer_3, 6)
    # best (because ~ equal and older)
    controller.load_model_and_optimizer_for_epoch(model_3, optimizer_3, 5)
    # ensure we're keeping track of the last when the model name is not
    # unique
    controller = training.TrainingStateController(
        training.TrainingStateParams(
            saved_model_fmt='model.pt',
        ),
        state_dir=temp_dir,
        warn=False
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


@pytest.mark.cpu
def test_pydrobert_param_optuna_hooks():
    poptuna = pytest.importorskip('pydrobert.param.optuna')
    optuna = pytest.importorskip('optuna')
    assert issubclass(
        training.TrainingStateParams, poptuna.TunableParameterized)
    global_dict = {'training': training.TrainingStateParams()}
    assert 'training.log10_learning_rate' in poptuna.get_param_dict_tunable(
        global_dict)

    def objective(trial):
        param_dict = poptuna.suggest_param_dict(trial, global_dict)
        return param_dict['training'].log10_learning_rate

    sampler = optuna.samplers.RandomSampler(seed=5)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=50)
    assert study.best_params['training.log10_learning_rate'] < -5
