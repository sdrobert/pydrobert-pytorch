from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import pytest
import pydrobert.torch.training as training

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


def test_controller_stores_and_retrieves(temp_dir, device):
    torch.manual_seed(50)
    model = torch.nn.Linear(2, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters())
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
    optimizer.zero_grad()
    inp = torch.randn(5, 2, device=device)
    model(inp).sum().backward()
    model_2 = torch.nn.Linear(2, 2).to(device)
    optimizer_2 = torch.optim.Adam(model_2.parameters())
    controller.load_model_and_optimizer_for_epoch(model_2, optimizer_2, 0)
    assert optimizer_2.param_groups[0]['lr'] == 10 ** p.log10_learning_rate
    for parameter_1, parameter_2 in zip(
            model.parameters(), model_2.parameters()):
        assert parameter_1.device == device
        assert parameter_2.device == device
        assert torch.allclose(parameter_1, parameter_2)
    optimizer.step()
    for parameter_1, parameter_2 in zip(
            model.parameters(), model_2.parameters()):
        assert not torch.allclose(parameter_1, parameter_2)
    optimizer_2.zero_grad()
    model_2(inp).sum().backward()
    optimizer_2.step()
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
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=20)
    controller.load_model_and_optimizer_for_epoch(model_2, optimizer_2, 10)
    for parameter_1, parameter_2 in zip(
            model.parameters(), model_2.parameters()):
        assert parameter_1.device == device
        assert parameter_2.device == device
        assert torch.allclose(parameter_1, parameter_2)
    optimizer_2.zero_grad()
    model_2(inp).sum().backward()
    optimizer_2.step()
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
    optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=20)
    controller.load_model_and_optimizer_for_epoch(model_3, optimizer_3, 10)
    model_3.to(device)
    for parameter_1, parameter_3 in zip(
            model.parameters(), model_3.parameters()):
        assert parameter_3.device == device
        assert torch.allclose(parameter_1, parameter_3)
    optimizer_3.zero_grad()
    model_3(inp).sum().backward()
    optimizer_3.step()
    for parameter_1, parameter_2, parameter_3 in zip(
            model.parameters(), model_2.parameters(), model_3.parameters()):
        assert not torch.allclose(parameter_1, parameter_2)
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


@pytest.mark.parametrize('batch_first', [True, False])
@pytest.mark.parametrize('sub_avg', [True, False])
@pytest.mark.parametrize('reduction', ['mean', 'none'])
def test_minimum_error_rate_loss(device, batch_first, sub_avg, reduction):
    torch.manual_seed(100)
    num_batches, samples, num_classes = 5, 5, 30
    max_ref_steps, max_hyp_steps = 10, 5
    assert max_ref_steps > max_hyp_steps  # nonzero loss guaranteed
    if batch_first:
        hyp = torch.randint(
            num_classes, (num_batches, samples, max_hyp_steps), device=device)
        hyp[..., 0] = 0
        ref = torch.randint(
            num_classes, (num_batches, max_ref_steps), device=device)
        ref[..., 0] = 0
    else:
        hyp = torch.randint(
            num_classes, (max_hyp_steps, num_batches, samples), device=device)
        hyp[0] = 0
        ref = torch.randint(
            num_classes, (max_ref_steps, num_batches), device=device)
        ref[0] = 0
    log_probs = torch.randn(num_batches, samples, device=device)
    loss = training.MinimumErrorRateLoss(
        eos=None, sub_avg=sub_avg, batch_first=batch_first,
        reduction=reduction,
    )
    l1 = loss(log_probs, ref, hyp)
    assert l1.ne(0.).any()
    l2 = loss(log_probs, ref, hyp)
    assert torch.allclose(l1, l2)
    loss.eos = 0
    l3 = loss(log_probs, ref, hyp)
    assert l3.eq(0.).all()


@pytest.mark.parametrize('batch_first', [True, False])
@pytest.mark.parametrize('eos', [None, 0])
@pytest.mark.parametrize('ref_steps_times', [1, 2])
@pytest.mark.parametrize('reduction', ['mean', 'none'])
@pytest.mark.parametrize('include_eos', [True, False])
def test_hard_optimal_completion_distillation_loss(
        device, batch_first, eos, ref_steps_times, reduction, include_eos):
    torch.manual_seed(209384)
    num_batches, max_steps, num_classes = 20, 41, 10
    if eos is None:
        hyp_lens = torch.tensor(max_steps).expand(num_batches)
        ref_lens = torch.tensor(ref_steps_times * max_steps)
        ref_lens = ref_lens.expand(num_batches)
    else:
        hyp_lens = torch.randint(1, max_steps + 1, (num_batches,))
        ref_lens = torch.randint(
            2, ref_steps_times * max_steps + 1, (num_batches,))
    ref = torch.nn.utils.rnn.pad_sequence(
        [
            torch.randint(1, num_classes, (x,))
            for x in ref_lens
        ],
        padding_value=num_classes - 1, batch_first=batch_first,
    )
    hyp = torch.nn.utils.rnn.pad_sequence(
        [
            torch.randint(1, num_classes, (x,))
            for x in hyp_lens
        ],
        padding_value=-1, batch_first=batch_first,
    )
    if eos is not None:
        for bt in range(num_batches):
            if batch_first:
                ref[bt, ref_lens[bt] - 1] = eos
                hyp[bt, hyp_lens[bt] - 1] = eos
            else:
                ref[ref_lens[bt] - 1, bt] = eos
                hyp[hyp_lens[bt] - 1, bt] = eos
        if not include_eos:
            ref_lens = ref_lens - 1
            hyp_lens = hyp_lens - 1
    logits = torch.rand(tuple(hyp.shape) + (num_classes,))
    if batch_first:
        len_mask = (
            torch.arange(hyp.shape[1]).unsqueeze(0) < hyp_lens.unsqueeze(1)
        )
    else:
        len_mask = torch.arange(hyp.shape[0]).unsqueeze(1) < hyp_lens
    logits, ref, hyp = logits.to(device), ref.to(device), hyp.to(device)
    ref_lens, hyp_lens = ref_lens.to(device), hyp_lens.to(device)
    len_mask = len_mask.to(device)
    inv_len_mask = len_mask.eq(0)
    logits.requires_grad_(True)
    loss = training.HardOptimalCompletionDistillationLoss(
        eos=eos, include_eos=include_eos, batch_first=batch_first,
        reduction=reduction,
    )
    l1 = loss(logits, ref, hyp)
    assert torch.all(l1 == l1)  # no nans
    if reduction == 'none':
        assert torch.all(l1.masked_select(inv_len_mask).eq(0.))
        # reference transcriptions are all positive length, so the first
        # optimal completion (assuming hyp length is nonzero) will always be
        # the first token in ref (and only the first token), given that there's
        # no ambiguity in the alignment of the prefix ""
        log_probs = torch.nn.functional.log_softmax(logits, 2)
        if batch_first:
            zero_length_mask = ref_lens.eq(0).unsqueeze(1)
            first_loss = torch.where(
                zero_length_mask,
                torch.zeros_like(log_probs[:, 0, 0]),
                -log_probs[:, 0].gather(
                    1, ref[:, 0].unsqueeze(-1)
                ).squeeze(-1),
            )
            assert torch.allclose(l1[:, 0], first_loss)
        else:
            zero_length_mask = ref_lens.eq(0).unsqueeze(0)
            first_loss = torch.where(
                zero_length_mask,
                torch.zeros_like(log_probs[0, :, 0]),
                -log_probs[0].gather(1, ref[0].unsqueeze(-1)).squeeze(-1),
            )
            assert torch.allclose(l1[0], first_loss)
        l1 = l1.mean()
    g, = torch.autograd.grad([l1], [logits])
    assert torch.all(g.masked_select(inv_len_mask.unsqueeze(-1)).eq(0.))
    assert not torch.all(g.eq(0.))


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
