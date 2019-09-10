#! /usr/bin/env python

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch

# remove this line if you've installed pydrobert-torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.realpath(__file__)))))

from pydrobert.torch.estimators import *
from pydrobert.torch.util import *

eos, padding = 0, -1
batch_size, inp_size, num_classes, num_samps = 10, 20, 5, 100
T, S, sos = 30, 10, -1
hidden_size = 40
num_iters, num_seeds = 1000, 10
dict_ = dict()
c_ff = c_rnn = None


def f(hyp, ref, gamma=.95):
    dists = prefix_error_rates(
        ref, hyp.long(), eos=eos, norm=False, padding=-1)
    r = -(dists[1:] - dists[:-1])
    r = r.masked_fill(dists[1:].eq(padding), 0.)
    R = time_distributed_return(r, gamma)
    return R


def c(z, lens):
    # you can avoid this sorting business by using enforce_sorted=False in
    # later versions of pytorch
    lens, idxs = lens.sort(descending=True)
    _, rev_idxs = idxs.sort()
    z = z.index_select(1, idxs)
    zp = torch.nn.utils.rnn.pack_padded_sequence(z, lens)
    zp, _ = c_rnn(zp)
    zp = torch.nn.utils.rnn.PackedSequence(c_ff(zp[0]), zp[1])
    z, _ = torch.nn.utils.rnn.pad_packed_sequence(zp, total_length=z.shape[0])
    z = z.squeeze(-1)
    z = z.index_select(1, rev_idxs)
    return z


# estimator format: '<estimator_name>[-{d,g}-<hs>]'
# where <estimator_name> is 'reinforce' or 'relax'
# <hs> is the hidden size of the control variate
# "d" means use difference as control variate loss (i.e. MSE)
# "g" means use variance of entire gradient estimate as control variate loss
estimators = (
    'reinforce',
    'reinforce-d-5',
    'reinforce-d-10',
    'reinforce-g-10',
    'reinforce-d-20',
    'relax-g-5',
    'relax-d-10',
    'relax-g-10',
    'relax-g-20',
)

for estimator in estimators:
    if len(sys.argv) > 1:
        continue
    if '-' in estimator:
        cv_hidden_size = int(estimator.split('-')[-1])
    else:
        cv_hidden_size = 1
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        inp = torch.randn(T, batch_size, inp_size)
        ref_lens = torch.randint(1, S + 1, (batch_size,))
        ref = torch.nn.utils.rnn.pad_sequence(
            [torch.randint(1, num_classes, (x + 1,)) for x in ref_lens],
            padding_value=padding,
        )
        ref[ref_lens, range(batch_size)] = eos
        # repeat the same reference transcription for each sample
        ref_rep = ref.unsqueeze(-1).repeat(1, 1, num_samps)
        cell = torch.nn.RNNCell(inp_size + 1, hidden_size)
        ff = torch.nn.Linear(hidden_size, num_classes)
        c_rnn = torch.nn.RNN(num_classes, cv_hidden_size, bidirectional=True)
        c_ff = torch.nn.Linear(2 * cv_hidden_size, 1)
        optim = torch.optim.Adam(
            tuple(cell.parameters()) + tuple(ff.parameters()) +
            tuple(c_rnn.parameters()) + tuple(c_ff.parameters())
        )
        ers = []
        c_losses = []
        c_loss = float('nan')
        for t in range(num_iters):
            h_t = torch.zeros(batch_size, 1, hidden_size)
            hyp = torch.full((1, batch_size, 1), sos, dtype=torch.long)
            optim.zero_grad()
            logits = z = None
            for inp_t in inp:
                hyp_tm1 = hyp[-1]
                old_samp = hyp_tm1.shape[-1]
                inp_t = inp_t.unsqueeze(1).expand(
                    batch_size, old_samp, inp_size)
                x_t = torch.cat([inp_t, hyp_tm1.unsqueeze(2).float()], -1)
                h_t = cell(
                    x_t.view(batch_size * old_samp, inp_size + 1),
                    h_t.view(batch_size * old_samp, hidden_size),
                ).view(batch_size, old_samp, hidden_size)
                logits_t = ff(h_t)  # (batch_size, old_samp, num_classes)
                hyp, z_t = random_walk_advance(
                    logits_t, num_samps, hyp, eos, include_relaxation=True)
                if old_samp == 1:
                    h_t = h_t.repeat(1, num_samps, 1).contiguous()
                    logits = logits_t.unsqueeze(0).expand(
                        -1, -1, num_samps, -1)
                    z = z_t.unsqueeze(0)
                else:
                    logits = torch.cat([logits, logits_t.unsqueeze(0)], dim=0)
                    z = torch.cat([z, z_t.unsqueeze(0)], dim=0)
            ref_rep = ref_rep.view(-1, batch_size * num_samps)
            hyp = hyp[1:].view(-1, batch_size * num_samps)  # get rid of sos
            logits = logits.view(-1, batch_size * num_samps, num_classes)
            z = z.view(-1, batch_size * num_samps, num_classes)
            mask = torch.isinf(z).any(-1, keepdim=True)
            lens = mask.eq(0).long().sum(0).squeeze(-1)
            fb = f(hyp, ref_rep)
            if estimator.startswith('reinforce'):
                if '-' in estimator:
                    fb = diff = fb - c(logits.detach(), lens)
                g = reinforce(fb, hyp, logits, 'cat')
            else:
                (
                    diff, dlog_pb, dc_z, dc_z_tilde,
                ) = relax(
                    fb, hyp, logits, z, c, 'cat', lens=lens, components=True)
                g = diff * dlog_pb + dc_z - dc_z_tilde
            g = g.masked_fill(mask, 0.)
            logits.backward(-g)
            if g.grad_fn is not None:
                if '-g-' in estimator:
                    c_loss = (g ** 2).sum(0)
                else:
                    c_loss = (diff ** 2).sum(0)
                c_loss = (c_loss / lens.float().unsqueeze(-1)).mean()
                c_loss.backward()
                c_loss = c_loss.item()
            c_losses.append(c_loss)
            optim.step()
            ers.append(error_rate(ref_rep, hyp, eos=eos).mean().item())
            print(t, ers[-1], c_losses[-1])
        dict_[(estimator, 'ers', seed)] = pd.Series(ers)
        dict_[(estimator, 'c_losses', seed)] = pd.Series(c_losses)

if dict_:
    df = pd.DataFrame.from_dict(dict_, orient='index')
    df.to_csv(
        'estimator_convergence.csv', index_label=['estimator', 'val', 'seed'])

df = pd.read_csv(
    'estimator_convergence.csv', index_col=['estimator', 'val', 'seed'])

colours = (
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
)
fig, er_ax = plt.subplots()
c_loss_ax = er_ax.twinx()
for color, est in zip(colours, estimators):
    df_est = df.xs(est)
    df_est_ers = df_est.xs('ers')
    mu_ers = df_est_ers.mean()
    std_ers = df_est_ers.std()
    er_ax.plot(mu_ers.index.values, mu_ers, color=color, label=est)
    er_ax.fill_between(
        mu_ers.index.values, mu_ers - std_ers, mu_ers + std_ers, color=color,
        alpha=.5 / len(estimators))
    df_est_c_losses = df_est.xs('c_losses')
    mu_c_losses = df_est_c_losses.mean()
    if not mu_c_losses.isna().any():
        c_loss_ax.plot(
            mu_c_losses.index.values, mu_c_losses, color=color,
            linestyle='--')

er_ax.legend()
ax = fig.get_axes()[0]
er_ax.set_xticks(range(0, len(mu_ers), len(mu_ers) // 10))
er_ax.set_xlabel('iterations')
er_ax.set_ylabel('error rate')
c_loss_ax.set_ylabel('control variate loss')
fig.savefig('estimator_convergence.png', dpi=1080)
