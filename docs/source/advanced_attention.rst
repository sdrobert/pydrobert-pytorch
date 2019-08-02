Advanced Attention and Transformer Networks
===========================================

This document is a supplement for advanced usage of
:class:`pydrobert.torch.layers.GlobalSoftAttention`, such as for Transformer
Networks [vaswani2017]_. It picks up where the class' summary left off.

`query` is an (n - 1)-dimensional tensor for ``n > 1``. `key` is an
n-dimensional tensor, and `value` is some n-dimensional tensor. Letting
:math:`t` index the `dim`-th dimension of `key`, :math:`q` index the last
dimension of `query`, and :math:`k` index the last index of `key`. Let
:math:`query_{t=0}` indicate the "unsqueezed" version of `query` where
:math:`t` is inserted as the `dim`-th dimension. Then :math:`query_{t=0,q}`
must `broadcast
<https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics>`__
with :math:`key_k`. If specified, `mask` should broadcast with :math:`e`, that
is, broadcast with a tensor of the same shape as :math:`key_k` after it has
been broadcast to :math:`query_{t=0,q}`. Finally, `value` must broadcast with
:math:`a_{k=0}`, that is, :math:`a` with an unsqueezed final dimension. Care
should be taken to ensure that any added dimensions to `query`, `key`, and
`value` ensure that the dimension that is to be attended to (reduced)
broadcasts to the correct location.

We'll illustrate with an example. Here, we've designed a barebones version of a
transformer network. There are lots of extra bits in a full transformer network
-- check [vaswani2017]_. Here we focus on the single-headed attention mechanism
(though a multi-headed version would be trivial to implement with
:class:`pydrobert.torch.layers.MultiHeadedAttention`). You can probably skip
the explanation in the middle if all you want to make is a transformer network
-- these settings should work.

First the requisite imports:

>>> import torch
>>> from pydrobert.torch.layers import *

The encoder is going to take in transcripts `inp` of shape ``(T, num_batch)``,
which have been right-padded along dimension 0. It will output both its
encoding in the shape ``(T, num_batch, model_size)`` and a mask of shape
``(T, 1, num_batch)`` that will be used by the decoder to only consider the
region of the encoding that was unpadded. By not specifying `dim` when
initializing :class:`pydrobert.torch.layers.DotProductSoftAttention`, the
attention dimension is implicitly set to 0, which turns out to be our
sequence dimension.

>>> class Encoder(torch.nn.Module):
>>>     def __init__(self, model_size, num_classes, padding_idx=-1):
>>>         super(Encoder, self).__init__()
>>>         self.model_size = model_size
>>>         self.num_classes = num_classes
>>>         self.embedder = torch.nn.Embedding(
>>>             num_classes, model_size, padding_idx=padding_idx)
>>>         self.attention = DotProductSoftAttention(
>>>             model_size, scale_factor=model_size ** -.5)
>>>
>>>     def forward(self, inp):
>>>         embedding = self.embedder(inp)
>>>         query = embedding  # (T, num_batch, model_size)
>>>         kv = embedding.unsqueeze(1)  # (T, 1, num_batch, model_size)
>>>         mask = inp.ne(self.embedder.padding_idx)
>>>         enc_mask = mask.unsqueeze(1)
>>>         out = self.attention(query, kv, kv, enc_mask)
>>>         return out, mask.unsqueeze(1)

The ``unsqueeze()`` calls are intended to ensure broadcasting occurs properly.
We're going to reduce the 0-th dimension (of size ``T``) of `kv`, but the 0-th
dimension of :math:`query_{t=0,q}` has to be accounted for when creating
:math:`e`. Then, through broadcasting, we expect :math:`e` to be shaped as

::

    query_{t=0,q}   1   T   num_batch
    key_k           T   1   num_batch
    ---------------------------------
    e               T   T   num_batch

(The attention mechanism gets rid of the last dimension of `query` and `key`,
in this case by taking the inner product). In :math:`e`, the 0-th dimension is
going to refer to each index of the sequence in `key`, whereas the 1-st
dimension refers to each index in the sequence of `value`. Effectively, a
Cartesian Product has been produced between the sequence dimensions of both
`query` and `key`.

We've unsqueezed `mask` to have shape ``(T, 1, num_batch)``. `mask` is
responsible for ensuring only non-padded values of `key` are considered.
It broadcasts with :math:`e` as:

::

    mask            T   1   num_batch
    e               T   T   num_batch
    ---------------------------------
    e & mask        T   T   num_batch

Which means that the mask is being applied to the 0-th (`key` sequence)
dimension and copied for every 1-st (`query` sequence) dimension. Had we
instead unsqueezed the mask into shape ``(1, T, num_batch)``, the mask would
have been applied to the 1-st dimension and copied to the 0-th instead. This
mask would've introduced ``NaN`` into ``a[:, i]`` for some ``i``.

Finally, `value` must broadcast with :math:`a_{k=0}`:

::

    a_{k=0}         T   T   num_batch
    value           T   1   num_batch
    ---------------------------------
    a_{k=0} * value T   T   num_batch

The 0-th dimension of `value` corresponds to its sequence dimension, which is
lined up with the `key` sequence dimension, which is the one to be attended to.
Had `value` been shaped as ``(1, T, num_batch)``, its sequence value would line
up with that of `query`, :math:`a_{k=0} * value` would be constant along the
attention dimension, and the weighted combination of terms would just
yield the original `value` tensor.

Now on to the decoder

>>> class Decoder(torch.nn.Module):
>>>     def __init__(self, model_size, num_classes, padding_idx=-2):
>>>         super(Decoder, self).__init__()
>>>         self.model_size = model_size
>>>         self.num_classes = num_classes
>>>         self.embedder = torch.nn.Embedding(
>>>             num_classes, model_size, padding_idx=padding_idx)
>>>         self.attention = DotProductSoftAttention(
>>>             model_size, scale_factor=model_size ** -.5)
>>>         self.ff = torch.nn.Linear(model_size, num_classes)
>>>
>>>     def forward(self, enc_out, dec_in, enc_mask=None):
>>>         embedding = self.embedder(dec_in)
>>>         query = embedding  # (S, num_batch, model_size)
>>>         kv = embedding.unsqueeze(1)  # (S, 1, num_batch, model_size)
>>>         pad_mask = dec_in.ne(self.embedder.padding_idx)
>>>         pad_mask = pad_mask.unsqueeze(1)  # (S, 1, num_batch)
>>>         auto_mask = torch.ones(
>>>             query.shape[0], query.shape[0], dtype=torch.uint8)
>>>         auto_mask = torch.triu(auto_mask)
>>>         auto_mask = auto_mask.unsqueeze(-1)  # (S, S, 1)
>>>         dec_mask = pad_mask & auto_mask  # (S, S, num_batch)
>>>         dec_out = self.attention(query, kv, kv, dec_mask)
>>>         query = dec_out  # (S, num_batch, model_size)
>>>         kv = enc_out.unsqueeze(1)  # (T, 1, num_batch, model_size)
>>>         out = self.attention(query, kv, kv, enc_mask)
>>>         out = self.ff(out)
>>>         return out, pad_mask

You can follow a similar logic as from the encoder to figure out most of the
sizes here. The only not-so-clear part is the self-attention mask for the
decoder. `pad_mask` does the same job as the encoder's mask: it ensures only
non-padded values are considered in the attention vector. `auto_mask` ensures
the auto-regressive property of key-value computations. That is, letting
:math:`s` index the sequence dimension of `dec_in`, we want :math:`out_s` not
to depend on any :math:`dec\_in_{>s}`. Recall `query`, `key`, and `value` are
all `dec_in`. Letting :math:`s` be the sequence dimension for `key` (dim=0,
attended to), and :math:`s'` be the sequence dimension for `query` (dim=1,
kept), we find the upper-triangular `auto_mask` satisfies

.. math::

    auto\_mask_{s,s'} = \begin{cases}
      1 & \mbox{if } s \leq s' \\
      0 & \mbox{if } s > s'
    \end{cases}

Since `auto_mask` should be applied indiscriminately to all batches, we
unsqueeze a final dimension so that it broadcasts to the batch dimension of
`pad_mask`.

The rest is straightforward. Here is some prep for a random data set:

>>> T, num_batch, model_size = 100, 5, 1000
>>> num_classes, start, eos = 20, 0, 1
>>> padding = num_classes - 1
>>> inp_lens = torch.randint(1, T + 1, (num_batch,))
>>> inp = torch.nn.utils.rnn.pad_sequence(
>>>     [
>>>         torch.randint(2, num_classes - 1, (x + 1,))
>>>         for x in inp_lens
>>>     ],
>>>     padding_value=padding,
>>> )
>>> inp[inp_lens, range(num_batch)] = eos
>>> target_lens = torch.randint(1, T + 1, (num_batch,))
>>> y = torch.nn.utils.rnn.pad_sequence(
>>>     [
>>>         torch.randint(2, num_classes - 1, (x + 2,))
>>>         for x in target_lens
>>>     ],
>>>     padding_value=padding,
>>> )
>>> y[0] = start
>>> y[target_lens + 1, range(num_batch)] = eos
>>> dec_inp, targets = y[:-1], y[1:]
>>> encoder = Encoder(model_size, num_classes, padding_idx=padding)
>>> decoder = Decoder(model_size, num_classes, padding_idx=padding)
>>> loss = torch.nn.CrossEntropyLoss(ignore_index=padding)
>>> optim = torch.optim.Adam(
>>>     list(encoder.parameters()) + list(decoder.parameters()))

Here's training a batch (you'lll have to do this a whole lot of times to get
it to converge)

>>> optim.zero_grad()
>>> enc_out, enc_mask = encoder(inp)
>>> logits, _ = decoder(enc_out, dec_inp, enc_mask)
>>> logits = logits[..., :-1]  # get rid of padding logit
>>> l = loss(logits.view(-1, num_classes - 1), targets.flatten())
>>> l.backward()
>>> optim.step()

And finally, decoding a batch (test time) using greedy search

>>> enc_out, enc_mask = encoder(inp)
>>> dec_hyp = torch.full((1, num_batch), start, dtype=torch.long)
>>> enc_out, enc_mask = encoder(inp)
>>> done_mask = torch.zeros(num_batch, dtype=torch.uint8)
>>> while not done_mask.all():
>>>     logits, _ = decoder(enc_out, dec_hyp, enc_mask)
>>>     logits = logits[..., :-1]  # get rid of padding logit
>>>     pred = logits[-1].argmax(1)
>>>     pred.masked_fill_(done_mask, eos)
>>>     done_mask = pred.eq(eos)
>>>     dec_hyp = torch.cat([dec_hyp, pred.unsqueeze(0)], 0)
>>> dec_hyp = dec_hyp[1:]
