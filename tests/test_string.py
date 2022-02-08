# Copyright 2022 Sean Robertson
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

import torch
import pytest

from pydrobert.torch.modules import (
    EditDistance,
    ErrorRate,
    HardOptimalCompletionDistillationLoss,
    MinimumErrorRateLoss,
    OptimalCompletion,
    PrefixEditDistances,
    PrefixErrorRates,
)
from pydrobert.torch.functional import edit_distance, error_rate


@pytest.mark.parametrize("exclude_last", [True, False])
@pytest.mark.parametrize("norm", [True, False], ids=("normed", "unnormed"))
@pytest.mark.parametrize("distance", [True, False], ids=("edit", "rate"))
def test_prefix_error_rates(
    device, exclude_last, norm, distance, jit_type,
):
    N, max_ref_steps, max_hyp_steps, C, eos = 30, 11, 12, 10, -1
    ins_cost, del_cost, sub_cost = (float(x) for x in range(1, 4))
    padding = -2
    hyp_lens = torch.randint(1, max_hyp_steps + 1, (N,), device=device)
    ref_lens = torch.randint(1, max_ref_steps + 1, (N,), device=device)
    hyp = torch.randint(C, (max_hyp_steps, N), device=device)
    ref = torch.randint(C, (max_ref_steps, N), device=device)
    hyp[hyp_lens - 1, range(N)] = eos
    ref[ref_lens - 1, range(N)] = eos
    ref_lens -= 1  # exclude the eos
    hyp_lens -= 1
    func = edit_distance if distance else error_rate
    rates = (PrefixEditDistances if distance else PrefixErrorRates)(
        eos=eos,
        include_eos=False,
        norm=norm,
        ins_cost=ins_cost,
        del_cost=del_cost,
        sub_cost=sub_cost,
        exclude_last=exclude_last,
        padding=padding,
        warn=False,
    )
    if jit_type == "script":
        rates = torch.jit.script(rates)
    elif jit_type == "trace":
        rates = torch.jit.trace(rates, (torch.full((1, 1), eos, dtype=torch.long),) * 2)
    act = rates(ref, hyp)
    exp = torch.empty(max_hyp_steps + (0 if exclude_last else 1), N, device=device)
    # if include_eos were true, `hyp` would get a bonus for the final `eos`
    # which isn't in its prefix
    for pref_len in range(max_hyp_steps - (1 if exclude_last else 0), -1, -1):
        hyp[pref_len:] = eos
        exp[pref_len] = func(
            ref,
            hyp,
            eos=eos,
            include_eos=False,
            norm=norm,
            ins_cost=ins_cost,
            del_cost=del_cost,
            sub_cost=sub_cost,
            warn=False,
        )
    exp = exp.masked_fill(
        (
            torch.arange(exp.shape[0], device=device).unsqueeze(1)
            >= hyp_lens + (0 if exclude_last else 1)
        ),
        padding,
    )
    assert torch.allclose(exp, act)


@pytest.mark.parametrize("include_eos", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("exclude_last", [True, False])
def test_optimal_completion(device, include_eos, batch_first, exclude_last, jit_type):
    eos, padding = ord("#"), -1
    triplets = (
        (
            "sunday#",
            "saturday#",
            ["s", "u", "un", "und", "n", "nd", "a", "y", "#", ""],
        ),
        ("sunday#", "satrapy#", ["s", "u", "un", "und", "unda", "y", "y#", "#", ""],),
        ("abc#", "abc#", ["a", "b", "c", "#", ""]),
        ("foot#", "bot#", ["f", "fo", "o", "ot#", ""]),
        ("abc#", "def#", ["a", "ab", "abc", "abc#", ""]),
    )
    ref = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor([ord(c) for c in word]) for (word, _, _) in triplets],
        batch_first=batch_first,
        padding_value=padding,
    ).to(device)
    hyp = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor([ord(c) for c in word]) for (_, word, _) in triplets],
        batch_first=batch_first,
        padding_value=eos,
    ).to(device)
    optimal_completion = OptimalCompletion(
        eos=eos,
        padding=padding,
        batch_first=batch_first,
        exclude_last=exclude_last,
        include_eos=include_eos,
    )
    if jit_type == "script":
        optimal_completion = torch.jit.script(optimal_completion)
    elif jit_type == "trace":
        optimal_completion = torch.jit.trace(
            optimal_completion, (torch.full((1, 1), eos, dtype=torch.long),) * 2
        )
    act = optimal_completion(ref, hyp)
    if not batch_first:
        act = act.transpose(0, 1)  # (batch, hyp, ref)
    assert act.shape[0] == len(triplets)
    for act_bt, (_, _, exp_bt) in zip(act, triplets):
        if not include_eos:
            exp_bt = [nexts.replace("#", "") for nexts in exp_bt[:-1]]
        if exclude_last:
            exp_bt = exp_bt[:-1]
        assert act_bt.shape[0] >= len(exp_bt)
        assert torch.all(act_bt[len(exp_bt) :].eq(padding))
        for act_bt_hyp, exp_bt_hyp in zip(act_bt, exp_bt):
            act_bt_hyp = act_bt_hyp.masked_select(act_bt_hyp.ne(padding))
            act_bt_hyp = sorted(chr(i) for i in act_bt_hyp.tolist())
            assert sorted(exp_bt_hyp) == act_bt_hyp


@pytest.mark.parametrize("include_eos", [0, 1])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("norm", [True, False], ids=("normed", "unnormed"))
@pytest.mark.parametrize("distance", [True, False], ids=("edit", "rate"))
def test_error_rate_against_known(
    device, norm, include_eos, batch_first, distance, jit_type
):
    eos = 0
    pairs = (
        ((1, 2, 3), (1, 2, 3), 0),
        ((2, 3), (1, 2, 3), 1),
        ((1, 3), (1, 2, 3), 1),
        ((3,), (1, 2, 3), 2),
        ((1, 2, 3), (1, 3), 1),
        ((1, 2, 3), (1, 2,), 1),
        ((1, 2, 3), (1,), 2),
        ((1, 3, 1, 2, 3), (1, 2, 3), 2),
        ((1, 2, 3), (4, 5, 6), 3),
        ((2, 2, 2), (2,), 2),
        (tuple(), (1,), 1),
        (tuple(), tuple(), 0),
    )
    ref_lens = torch.tensor([len(x[0]) + include_eos for x in pairs], device=device)
    hyp_lens = torch.tensor([len(x[1]) + include_eos for x in pairs], device=device)
    ref = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x[0] + (eos,) * include_eos) for x in pairs],
        padding_value=eos,
        batch_first=batch_first,
    ).to(device)
    hyp = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x[1] + (eos,) * include_eos) for x in pairs],
        padding_value=eos,
        batch_first=batch_first,
    ).to(device)
    exp = torch.tensor([float(x[2]) for x in pairs], device=device)
    if norm:
        exp = torch.where(ref_lens == 0, hyp_lens.ne(0).float(), exp / ref_lens.float())
    # when all the costs are one, the edit distance should be the same as the error rate
    rate = (EditDistance if distance else ErrorRate)(
        eos=eos,
        warn=False,
        norm=norm,
        include_eos=bool(include_eos),
        batch_first=batch_first,
    )
    if jit_type == "script":
        rate = torch.jit.script(rate)
    elif jit_type == "trace":
        rate = torch.jit.trace(rate, (torch.zeros(1, 1), torch.zeros(1, 1)))
    act = rate(ref, hyp)
    assert torch.allclose(exp, act)


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("sub_avg", [True, False])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_minimum_error_rate_loss(device, batch_first, sub_avg, reduction, jit_type):
    num_batches, samples, num_classes = 5, 5, 30
    max_ref_steps, max_hyp_steps = 10, 5
    assert max_ref_steps > max_hyp_steps  # nonzero loss guaranteed
    if batch_first:
        hyp = torch.randint(
            num_classes, (num_batches, samples, max_hyp_steps), device=device
        )
        hyp[..., 0] = 0
        ref = torch.randint(num_classes, (num_batches, max_ref_steps), device=device)
        ref[..., 0] = 0
    else:
        hyp = torch.randint(
            num_classes, (max_hyp_steps, num_batches, samples), device=device
        )
        hyp[0] = 0
        ref = torch.randint(num_classes, (max_ref_steps, num_batches), device=device)
        ref[0] = 0
    log_probs = torch.randn(num_batches, samples, device=device)
    loss = MinimumErrorRateLoss(
        eos=None, sub_avg=sub_avg, batch_first=batch_first, reduction=reduction,
    )
    if jit_type == "trace":
        loss = torch.jit.trace(loss, (log_probs, ref, hyp))
    elif jit_type == "script":
        loss = torch.jit.script(loss)
    l1 = loss(log_probs, ref, hyp)
    assert l1.ne(0.0).any()
    l2 = loss(log_probs, ref, hyp)
    assert torch.allclose(l1, l2)
    loss = MinimumErrorRateLoss(
        eos=0, sub_avg=sub_avg, batch_first=batch_first, reduction=reduction,
    )
    if jit_type == "trace":
        loss = torch.jit.trace(loss, (log_probs, ref, hyp))
    elif jit_type == "script":
        loss = torch.jit.script(loss)
    l3 = loss(log_probs, ref, hyp)
    assert l3.eq(0.0).all()


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("eos", [None, 0])
@pytest.mark.parametrize("ref_steps_times", [1, 2])
@pytest.mark.parametrize("reduction", ["mean", "none"])
@pytest.mark.parametrize("include_eos", [True, False])
def test_hard_optimal_completion_distillation_loss(
    device, batch_first, eos, ref_steps_times, reduction, include_eos, jit_type
):
    torch.manual_seed(209384)
    num_batches, max_steps, num_classes = 20, 41, 10
    if eos is None:
        hyp_lens = torch.tensor(max_steps).expand(num_batches)
        ref_lens = torch.tensor(ref_steps_times * max_steps)
        ref_lens = ref_lens.expand(num_batches)
    else:
        hyp_lens = torch.randint(1, max_steps + 1, (num_batches,))
        ref_lens = torch.randint(2, ref_steps_times * max_steps + 1, (num_batches,))
    ref = torch.nn.utils.rnn.pad_sequence(
        [torch.randint(1, num_classes, (x,)) for x in ref_lens],
        padding_value=num_classes - 1,
        batch_first=batch_first,
    )
    hyp = torch.nn.utils.rnn.pad_sequence(
        [torch.randint(1, num_classes, (x,)) for x in hyp_lens],
        padding_value=-1,
        batch_first=batch_first,
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
        len_mask = torch.arange(hyp.shape[1]).unsqueeze(0) < hyp_lens.unsqueeze(1)
    else:
        len_mask = torch.arange(hyp.shape[0]).unsqueeze(1) < hyp_lens
    logits, ref, hyp = logits.to(device), ref.to(device), hyp.to(device)
    ref_lens, hyp_lens = ref_lens.to(device), hyp_lens.to(device)
    len_mask = len_mask.to(device)
    inv_len_mask = ~len_mask
    logits.requires_grad_(True)
    loss = HardOptimalCompletionDistillationLoss(
        eos=eos, include_eos=include_eos, batch_first=batch_first, reduction=reduction,
    )
    if jit_type == "script":
        loss = torch.jit.script(loss)
    elif jit_type == "trace":
        loss = torch.jit.trace(loss, (logits, ref, hyp))
    l1 = loss(logits, ref, hyp)
    assert torch.all(l1 == l1)  # no nans
    if reduction == "none":
        assert torch.all(l1.masked_select(inv_len_mask).eq(0.0))
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
                -log_probs[:, 0].gather(1, ref[:, 0].unsqueeze(-1)).squeeze(-1),
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
    (g,) = torch.autograd.grad([l1], [logits])
    assert torch.all(g.masked_select(inv_len_mask.unsqueeze(-1)).eq(0.0))
    assert not torch.all(g.eq(0.0))


@pytest.mark.parametrize("ins_cost", [2.0, 0.5, 1.0], ids=("i2.0", "i0.5", "i1.0"))
@pytest.mark.parametrize("del_cost", [2.0, 0.5, 1.0], ids=("d2.0", "d0.5", "d1.0"))
@pytest.mark.parametrize("sub_cost", [2.0, 0.5, 1.0], ids=("s2.0", "s0.5", "s1.0"))
@pytest.mark.parametrize("distance", [True, False], ids=("edit", "rate"))
@pytest.mark.parametrize("ref_bigger", [True, False])
def test_error_rate_against_simple_impl(
    device, ins_cost, del_cost, sub_cost, ref_bigger, distance
):
    hyp_steps, ref_steps, batch_size, num_classes = 10, 9, 10, 10
    eps = 1e-4
    if ref_bigger:
        ref_steps, hyp_steps = hyp_steps, ref_steps
    ref = torch.randint(num_classes, (ref_steps, batch_size), device=device)
    hyp = torch.randint(num_classes, (hyp_steps, batch_size), device=device)
    # here's a standard, non-vectorized (except for batch) implementation that
    # is hard to screw up
    cost_matrix = torch.empty(hyp_steps + 1, ref_steps + 1, batch_size, device=device)
    cost_matrix[0] = (
        torch.arange(float(ref_steps + 1), device=device).unsqueeze(-1) * del_cost
    )
    cost_matrix[:, 0] = (
        torch.arange(float(hyp_steps + 1), device=device).unsqueeze(-1) * ins_cost
    )
    edit_matrix = torch.empty(hyp_steps + 1, ref_steps + 1, batch_size, device=device)
    edit_matrix[0] = torch.arange(float(ref_steps + 1), device=device).unsqueeze(-1)
    edit_matrix[:, 0] = torch.arange(float(hyp_steps + 1), device=device).unsqueeze(-1)
    for hyp_idx in range(1, hyp_steps + 1):
        for ref_idx in range(1, ref_steps + 1):
            neq_mask = (ref[ref_idx - 1] != hyp[hyp_idx - 1]).float()
            sub_align = cost_matrix[hyp_idx - 1, ref_idx - 1] + sub_cost * neq_mask
            ins_align = cost_matrix[hyp_idx - 1, ref_idx] + ins_cost + eps
            del_align = cost_matrix[hyp_idx, ref_idx - 1] + del_cost + eps
            cur_costs, argmin = torch.stack([sub_align, ins_align, del_align]).min(0)
            cur_costs -= argmin.gt(0) * eps
            cost_matrix[hyp_idx, ref_idx] = cur_costs
            sub_count = edit_matrix[hyp_idx - 1, ref_idx - 1] + neq_mask
            ins_count = edit_matrix[hyp_idx - 1, ref_idx] + 1
            del_count = edit_matrix[hyp_idx, ref_idx - 1] + 1
            cur_counts = (
                torch.stack([sub_count, ins_count, del_count])
                .gather(0, argmin.unsqueeze(0))
                .squeeze(0)
            )
            edit_matrix[hyp_idx, ref_idx] = cur_counts
    if ins_cost == del_cost == sub_cost == 1:
        assert torch.allclose(cost_matrix, edit_matrix)
    if distance:
        exp = cost_matrix[-1, -1]
        func = edit_distance
    else:
        exp = edit_matrix[-1, -1]
        func = error_rate
    act = func(
        ref,
        hyp,
        norm=False,
        ins_cost=ins_cost,
        del_cost=del_cost,
        sub_cost=sub_cost,
        warn=False,
    )
    assert torch.allclose(exp, act)


@pytest.mark.parametrize("ins_cost", [0.5, 1.0], ids=("i0.5", "i1.0"))
@pytest.mark.parametrize("del_cost", [0.5, 1.0], ids=("d0.5", "d1.0"))
@pytest.mark.parametrize("sub_cost", [0.5, 1.0], ids=("s0.5", "s1.0"))
@pytest.mark.parametrize("norm", [True, False], ids=("normed", "unnormed"))
@pytest.mark.parametrize("distance", [True, False], ids=("edit", "rate"))
def test_error_rate_ignores_padding(
    device, ins_cost, del_cost, sub_cost, norm, distance
):
    N, Tmax, V, eos = 11, 50, 5, -1
    ref_lens = torch.randint(Tmax, size=(N,), device=device)
    refs = [torch.randint(V, size=(len_.item(),), device=device) for len_ in ref_lens]
    hyp_lens = torch.randint(Tmax, size=(N,), device=device)
    hyps = [torch.randint(V, size=(len_.item(),), device=device) for len_ in hyp_lens]
    func = edit_distance if distance else error_rate
    out_a = []
    for ref, hyp in zip(refs, hyps):
        out_a.append(
            func(
                ref.unsqueeze(1),
                hyp.unsqueeze(1),
                norm=norm,
                ins_cost=ins_cost,
                del_cost=del_cost,
                sub_cost=sub_cost,
                warn=False,
            )
        )
    out_a = torch.cat(out_a, 0)
    assert out_a.dim() == 1 and out_a.size(0) == N
    refs = torch.nn.utils.rnn.pad_sequence(refs, padding_value=eos)
    hyps = torch.nn.utils.rnn.pad_sequence(hyps, padding_value=eos)
    out_b = func(
        refs,
        hyps,
        eos=eos,
        norm=norm,
        ins_cost=ins_cost,
        del_cost=del_cost,
        sub_cost=sub_cost,
        warn=False,
    )
    assert torch.allclose(out_a, out_b)
