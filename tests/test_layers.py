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


@pytest.mark.parametrize('dim', [0, 1])
def test_global_soft_attention(device, dim):

    class FirstIsBest(layers.GlobalSoftAttention):

        def score(self, query, key):
            e = torch.full_like(key[..., 0], -float('inf'))
            e.narrow(self.dim, 0, 1).fill_(0.)
            return e

    class ILoveEveryoneEqually(layers.GlobalSoftAttention):

        def score(self, query, key):
            return torch.zeros_like(key[..., 0])

    torch.manual_seed(562992)
    T, max_dim, max_dim_size = 12, 10, 10
    num_dim = torch.randint(dim + 2, max_dim + 1, (1,), device=device).item()
    key_shape = torch.randint(
        1, max_dim_size + 1, (num_dim + 1,), device=device).tolist()
    key_shape[dim] = T
    query_shape = key_shape[:dim] + key_shape[dim + 1:-1]
    del key_shape[-2]
    key_lens = torch.randint(1, T + 1, query_shape[:-1], device=device)
    query = torch.randn(*query_shape, device=device)
    key = torch.randn(*key_shape, device=device)
    query_size = query_shape[-1]
    key_size = key_shape[-1]
    arange_shape = [1] * (num_dim - 1)
    arange_shape[dim] = T
    mask = torch.arange(T, device=device).view(*arange_shape)
    mask = mask < key_lens.unsqueeze(dim)
    key.requires_grad_(True)
    first_attention = FirstIsBest(query_size, key_size, dim).to(device)
    equal_attention = ILoveEveryoneEqually(
        query_size, key_size, dim).to(device)
    out1 = first_attention(query, key, key)
    assert torch.allclose(out1, key.narrow(dim, 0, 1).squeeze(dim))
    out2 = first_attention(query, key, key, mask)
    assert torch.allclose(out1, out2)
    g, = torch.autograd.grad([out1], [key], grad_outputs=torch.ones_like(out1))
    assert g.narrow(dim, 0, 1).eq(1).all()
    assert g.narrow(dim, 1, T - 1).eq(0).all()
    out1 = equal_attention(query, key, key)
    # the softmax introduces a slight numeric instability
    assert torch.allclose(out1, key.mean(dim), atol=1e-5)
    out2 = equal_attention(query, key, key, mask)
    assert not torch.allclose(out1, out2)
    exp = key.masked_fill(mask.unsqueeze(-1).eq(0), 0.)
    exp = exp.sum(dim)
    exp = exp / key_lens.float().unsqueeze(-1)
    assert torch.allclose(out2, exp, atol=1e-5)
    g, = torch.autograd.grad([out2], [key], grad_outputs=torch.ones_like(out2))
    assert g.masked_select(mask.eq(0).unsqueeze(-1)).eq(0).all()
    assert torch.allclose(
        g.sum(dim),
        torch.tensor(1., device=device),
        atol=1e-5
    )


@pytest.mark.parametrize('dim', [0, 1, 2])
def test_dot_product_soft_attention(device, dim):
    torch.manual_seed(387420)
    dim1, dim2, dim3, dim4 = 50, 30, 12, 100
    key_shape = (dim1, dim2, dim3, dim4)
    key = torch.randn(*key_shape, device=device)
    query_shape = key_shape[:dim] + key_shape[dim + 1:]
    query = torch.zeros(*query_shape, device=device)
    query[..., 0] = 2.
    exp = torch.nn.functional.softmax(key[..., 0], dim).unsqueeze(-1) * key
    exp = exp.sum(dim)
    attention = layers.DotProductSoftAttention(dim4, dim, scale_factor=.5)
    act = attention(query, key, key)
    assert torch.allclose(exp, act)


@pytest.mark.cpu
def test_dot_product_soft_attention_on_transformer_input():

    class MatrixVersion(torch.nn.Module):
        '''Scaled dot product attention, specifically for transformers

        This was blatantly ripped from `speech transformers
        <https://github.com/kaituoxu/Speech-Transformer/blob/a0bbd58da193051bb0ea597e1c4120021a721c16/src/transformer/attention.py#L65>`__.

        This is a more straightforward implementation of the scaled dot product
        attention for transformer networks. We're showing that our
        implementation yields the same output and gradient as this.
        '''

        def __init__(self, temperature):
            super(MatrixVersion, self).__init__()
            self.temperature = temperature
            self.softmax = torch.nn.Softmax(dim=2)

        def forward(self, q, k, v, mask=None):
            attn = torch.bmm(q, k.transpose(1, 2))
            attn = attn / self.temperature
            if mask is not None:
                attn = attn.masked_fill(mask, -float('inf'))
            attn = self.softmax(attn)
            output = torch.bmm(attn, v)
            return output

    torch.manual_seed(34229)
    num_batch, len_q, len_k, d_k, d_v = 30, 40, 20, 10, 50
    temp = 2.
    query = torch.randn(num_batch, len_q, d_k, requires_grad=True)
    key = torch.randn(num_batch, len_k, d_k, requires_grad=True)
    value = torch.randn(num_batch, len_k, d_v, requires_grad=True)
    matrix_attention = MatrixVersion(temp)
    our_attention = layers.DotProductSoftAttention(d_k, 1, 1 / temp)
    out1 = matrix_attention(query, key, value)
    out2 = our_attention(query, key.unsqueeze(2), value.unsqueeze(2))
    assert torch.allclose(out1, out2, atol=1e-5)
    g_q1, g_k1, g_v1 = torch.autograd.grad(
        [out1], [query, key, value], grad_outputs=torch.ones_like(out1))
    g_q2, g_k2, g_v2 = torch.autograd.grad(
        [out2], [query, key, value], grad_outputs=torch.ones_like(out2))
    assert torch.allclose(g_q1, g_q2, atol=1e-5)
    assert torch.allclose(g_k1, g_k2, atol=1e-5)
    assert torch.allclose(g_v1, g_v2, atol=1e-5)
    mask = torch.randint(2, (num_batch, len_q, len_k)).eq(1)
    out1 = matrix_attention(query, key, value, mask)
    out2 = our_attention(
        query, key.unsqueeze(2), value.unsqueeze(2),
        mask.transpose(1, 2).eq(0)  # we use the inverse of their mask
    )
    assert torch.allclose(out1, out2, atol=1e-5)


@pytest.mark.parametrize('dim', [0, 1, 2])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize(
    'layer', ['general', 'concat', 'multihead_general', 'multihead_concat'])
def test_learnable_soft_attention(device, dim, bias, layer):
    torch.manual_seed(347201)
    max_dim, max_dim_size, max_num_heads = 5, 30, 10
    num_dim = torch.randint(dim + 2, max_dim + 1, (1,), device=device).item()
    # dim size must be at least 2. Otherwise, softmax will have only one
    # element and gradient will be zero through it
    key_shape = torch.randint(
        2, max_dim_size + 1, (num_dim + 1,), device=device).tolist()
    query_shape = key_shape[:dim] + key_shape[dim + 1:-1]
    del key_shape[-2]
    key = torch.randn(*key_shape, device=device)
    query = torch.randn(*query_shape, device=device)
    key_size = key_shape[-1]
    query_size = query_shape[-1]
    if layer == 'general':
        attention = layers.GeneralizedDotProductSoftAttention(
            query_size, key_size, dim, bias)
    elif layer == 'concat':
        attention = layers.ConcatSoftAttention(
            query_size, key_size, dim, bias)
    elif layer.startswith('multihead_'):
        num_heads = torch.randint(
            1, max_num_heads + 1, (1,), device=device).item()
        d_q = max(1, query_size // num_heads)
        d_k = max(1, key_size // num_heads)
        if layer.endswith('general'):
            single_head_attention = layers.GeneralizedDotProductSoftAttention(
                d_q, d_k, dim, bias)
        elif layer.endswith('concat'):
            single_head_attention = layers.ConcatSoftAttention(
                query_size, key_size, dim, bias)
        attention = layers.MultiHeadedAttention(
            query_size, key_size, key_size, num_heads, single_head_attention,
            bias_WQ=bias, bias_WK=bias, bias_WV=bias, bias_WC=bias,
        )
    attention = attention.to(device)
    torch.manual_seed(30)
    attention.reset_parameters()
    optim = torch.optim.Adam(attention.parameters(), lr=1.)
    optim.zero_grad()
    out1 = attention(query, key, key)
    out1.mean().backward()
    optim.step()
    optim.zero_grad()
    out2 = attention(query, key, key)
    assert not torch.allclose(out1, out2, atol=1e-5)
    torch.manual_seed(30)
    attention.reset_parameters()
    out3 = attention(query, key, key)
    assert torch.allclose(out1, out3, atol=1e-5)
