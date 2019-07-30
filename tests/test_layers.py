from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import torch
import pydrobert.torch.layers as layers

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"


@pytest.mark.parametrize('batch_first', [True, False])
def test_global_soft_attention(device, batch_first):

    class FirstIsBest(layers.GlobalSoftAttention):

        def score(self, h_t, x):
            e_t = torch.full_like(x[..., 0], -float('inf'))
            if self.batch_first:
                e_t[:, 0] = 0.
            else:
                e_t[0] = 0.
            return e_t

    class ILoveEveryoneEqually(layers.GlobalSoftAttention):

        def score(self, h_t, x):
            return torch.zeros_like(x[..., 0])

    torch.manual_seed(562992)
    num_batch, T, input_size, hidden_size = 20, 100, 13, 12
    x_lens = torch.randint(1, T + 1, (num_batch,), device=device)
    if batch_first:
        x = torch.randn(num_batch, T, input_size, device=device)
        mask = torch.where(
            torch.arange(T, device=device).unsqueeze(0) < x_lens.unsqueeze(1),
            torch.tensor(1, device=device, dtype=torch.uint8),
            torch.tensor(0, device=device, dtype=torch.uint8),
        )
    else:
        x = torch.randn(T, num_batch, input_size, device=device)
        mask = torch.where(
            torch.arange(T, device=device).unsqueeze(1) < x_lens,
            torch.tensor(1, device=device, dtype=torch.uint8),
            torch.tensor(0, device=device, dtype=torch.uint8),
        )
    x.requires_grad_(True)
    h_t = torch.randn(num_batch, hidden_size, device=device)
    first_attention = FirstIsBest(
        input_size, hidden_size, batch_first=batch_first).to(device)
    equal_attention = ILoveEveryoneEqually(
        input_size, hidden_size, batch_first=batch_first).to(device)
    c_t1 = first_attention(h_t, x)
    if batch_first:
        assert torch.allclose(c_t1, x[:, 0])
    else:
        assert torch.allclose(c_t1, x[0])
    c_t2 = first_attention(h_t, x, mask)
    assert torch.allclose(c_t1, c_t2)
    g, = torch.autograd.grad([c_t1], [x], grad_outputs=torch.ones_like(c_t1))
    if batch_first:
        assert g[:, 0].eq(1).all()
        assert g[:, 1:].eq(0).all()
    else:
        assert g[0].eq(1).all()
        assert g[1:].eq(0).all()
    c_t1 = equal_attention(h_t, x)
    # the softmax introduces a slight numeric instability
    assert torch.allclose(c_t1, x.mean(1 if batch_first else 0), atol=1e-5)
    c_t2 = equal_attention(h_t, x, mask)
    assert not torch.allclose(c_t1, c_t2)
    exp = x.masked_fill(mask.unsqueeze(-1).eq(0), 0.)
    exp = exp.sum(1 if batch_first else 0)
    exp = exp / x_lens.float().unsqueeze(1)
    assert torch.allclose(c_t2, exp, atol=1e-5)
    g, = torch.autograd.grad([c_t2], [x], grad_outputs=torch.ones_like(c_t2))
    assert g.masked_select(mask.eq(0).unsqueeze(-1)).eq(0).all()
    assert torch.allclose(
        g.sum(1 if batch_first else 0),
        torch.tensor(1., device=device),
        atol=1e-5
    )


@pytest.mark.parametrize('batch_first', [True, False])
def test_dot_product_soft_attention(device, batch_first):
    torch.manual_seed(387420)
    num_batch, T, size = 50, 30, 12
    x = torch.randn(T, num_batch, size, device=device)
    h_t = torch.zeros(num_batch, size, device=device)
    h_t[..., 0] = 2.
    exp = torch.nn.functional.softmax(x[..., 0], 0).unsqueeze(-1) * x
    exp = exp.sum(0)
    if batch_first:
        x = x.transpose(0, 1).contiguous()
    attention = layers.DotProductSoftAttention(
        size, batch_first, scale_factor=.5)
    act = attention(h_t, x)
    assert torch.allclose(exp, act)


@pytest.mark.parametrize('batch_first', [True, False])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('layer', ['general', 'concat'])
def test_learnable_soft_attention(device, batch_first, bias, layer):
    torch.manual_seed(347201)
    num_batch, T, input_size, hidden_size = 50, 30, 12, 124
    if batch_first:
        x = torch.randn(num_batch, T, input_size, device=device)
    else:
        x = torch.randn(T, num_batch, input_size, device=device)
    h_t = torch.randn(num_batch, hidden_size, device=device)
    if layer == 'general':
        class_ = layers.GeneralizedDotProductSoftAttention
    elif layer == 'concat':
        class_ = layers.ConcatSoftAttention
    attention = class_(input_size, hidden_size, batch_first, bias).to(device)
    torch.manual_seed(30)
    attention.reset_parameters()
    optim = torch.optim.Adam(attention.parameters())
    optim.zero_grad()
    c_t1 = attention(h_t, x)
    c_t1.mean().backward()
    optim.step()
    optim.zero_grad()
    c_t2 = attention(h_t, x)
    assert not torch.allclose(c_t1, c_t2, atol=1e-5)
    torch.manual_seed(30)
    attention.reset_parameters()
    c_t3 = attention(h_t, x)
    assert torch.allclose(c_t1, c_t3)
