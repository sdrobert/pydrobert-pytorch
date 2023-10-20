.. _lm:

Language Modelling and Decoding
===============================

A Simple-ish Example
--------------------

*pydrobert-pytorch* features interfaces for sequential Language Models (LMs).
Sequential LMs are more easily integrated into the transcription/decoding
process for Automatic Speech Recognition (ASR) than non-sequential ones like
BERT [bert2019]_. We will start with a basic implementation of the interface
:class:`pydrobert.torch.modules.SequentialLanguageModel` and extend it until we
can perform a `Beam Search
<https://medium.com/@dhartidhami/beam-search-in-seq2seq-model-7606d55b21a5>`__
or a Connectionist Temporal Classification (CTC) `Prefix Search
<https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7>`__.
We will perform computations on the CPU, though the code can be trivially
adapted to the GPU by sending models and tensors to the appropriate device.

.. code-block:: python

  import torch
  from pydrobert.torch.modules import SequentialLanguageModel

  class RNNLM(SequentialLanguageModel):
      def __init__(self, vocab_size, embed_size=128, hidden_size=512):
          super(RNNLM, self).__init__(vocab_size)
          self.embed = torch.nn.Embedding(
              vocab_size + 1, embed_size, padding_idx=vocab_size
          )
          self.rnn = torch.nn.LSTMCell(embed_size, hidden_size)
          self.ff = torch.nn.Linear(hidden_size, vocab_size)

      def calc_idx_log_probs(self, hist, prev, idx):
          N = hist.size(1)
          if idx == 0:
              in_ = hist.new_full((N,), self.vocab_size)
              prev = [self.rnn.weight_hh.new_zeros((N, self.rnn.hidden_size))] * 2
          else:
              if not prev:
                  prev = self.calc_idx_log_probs(hist, None, idx - 1)[1]
              in_ = hist[idx - 1]
          embedding = self.embed(in_)
          h_1, c_1 = self.rnn(embedding, prev)
          logits = self.ff(h_1)
          return torch.nn.functional.log_softmax(logits, -1), (h_1, c_1)

This is a simple, auto-regressive, sequential language model. It has one
embedding layer, one LSTM cell layer, and one feed-forward layer to produce
logits over the next output.

It is easy enough to train and sample the above LM:

.. code-block:: python

    torch.manual_seed(0)
    vocab_size, batch_size, sequence_length, epochs = 520, 5, 15, 10
    lm = RNNLM(vocab_size)
    text = torch.randint(vocab_size, (sequence_length, batch_size))

    # training
    optim = torch.optim.Adam(lm.parameters())
    ce_loss = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        optim.zero_grad()
        hist = text[:-1]  # exclude the last token - don't need to predict next
        logits = lm(text[:-1])  # (sequence_length, batch_size, vocab_size)
        loss = ce_loss(logits.flatten(0, 1), text.flatten())
        loss.backward()
        optim.step()
    
    # random walk
    hist = torch.empty((0, batch_size), dtype=torch.long)
    prev = None
    for idx in torch.arange(sequence_length):
        log_p, prev = lm(hist, prev, idx)
        # log_p is of shape (batch_size, vocab_size)
        cur_tokens = torch.distributions.Categorical(logits=log_p).sample()
        hist = torch.cat([hist, cur_tokens.unsqueeze(0)])

:func:`calc_idx_log_probs` represents the work of a single step in the
sequential language model. It receives a prefix of tokens, `hist`, a previous
hidden state, `prev`, and an index, `idx`, telling it what index in `hist` to
compute the distribution over the next token for. `idx` is usually going to
increment by one with each subsequent call, i.e. ``0``, then ``1``, then ``2``,
and so on. `hist` is of shape ``(S, batch_size)``, where ``S`` represents the
prefix length. It will be no shorter than `idx`. The return value is a pair
``log_p, cur``. `log_p` contains the log probabilities of the distribution over
the next token (i.e. the token at `idx`). `cur` is the hidden state after
absorbing `prev` and the tokens at `idx - 1` in `hist`. `prev` and `cur` are
not necessary to implement (they can remain :obj:`None`), but they avoid
redundant computation in sequential loops.

The last few lines of :func:`calc_idx_log_probs` are straightforward enough:
take the last token of the prefix (``hist[idx - 1]``) and extract an embedding
from it using a the embedding layer ``self.embed``; pass that `embedding` and
the previous LSTM states `prev` into the LSTM layer to get back hidden and cell
states `cur`; pass the hidden states through the feedforward layer to get
`logits`; and return the normalized `logits` and `cur`. Normalizing the
`logits` into log probabilities is not strictly necessary for this example,
though it is when pairing with a search algorithm. A random walk with a few
more bells and whistles can be accomplished by the module
:class:`pydrobert.torch.modules.RandomWalk`.

Note at the beginning of the method that we check if `idx == 0`. This is for
when we're generating the first token. Since we can't extract a previous token
from the history to feed into our LSTM, we produce a special, start-of-sequence
token. We add the start-of-sequence type to end of the vocabulary (note the
size of the :class:`torch.Embedding` layer) and replace ``hist[idx - 1]`` with
a tensor of start-of-sequence tokens whenever ``idx == 0``.

To perform some form of search for the purposes of decoding, like a beam search
or a CTC prefix search, the module needs to get more complicated. This is
because the search needs to know how to manipulate the language model state
(`prev` or `cur`). For :class:`pydrobert.torch.modules.BeamSearch`, the LM must
implement :class:`pydrobert.torch.modules.ExtractableSequentialLanguageModel`,
which extends :class:`SequentialLanguageModel`. We reimplement our LM below:

.. code-block:: python

    import torch
    from pydrobert.torch.modules import ExtractableSequentialLanguageModel

    class RNNLM(ExtractableSequentialLanguageModel):
        def __init__(self, vocab_size, embed_size=128, hidden_size=512):
            super().__init__(vocab_size)
            self.hidden_size = hidden_size
            self.embed = torch.nn.Embedding(
                vocab_size + 1, embed_size, padding_idx=vocab_size
            )
            self.cell = torch.nn.LSTMCell(embed_size, hidden_size)
            self.ff = torch.nn.Linear(hidden_size, vocab_size)

        def extract_by_src(self, prev, src):
            return {
                "hidden_state": prev["hidden_state"].index_select(0, src),
                "cell_state": prev["cell_state"].index_select(0, src),
            }

        def update_input(self, prev, hist):
            if len(prev):
                return prev  # not first call
            N = hist.size(1)
            zeros = self.ff.weight.new_zeros((N, self.hidden_size))
            return {"hidden_state": zeros, "cell_state": zeros}

        def calc_idx_log_probs(self, hist, prev, idx):
            idx_zero = idx == 0
            if idx_zero.all():
                x = idx.new_full((hist.size(1),), self.vocab_size)
            elif not idx.dim():
                x = hist[idx - 1]
            else:
                x = hist.gather(0, (idx - 1).clamp(min=0).unsqueeze(0)).squeeze(0)
                x = x.masked_fill(idx_zero, self.vocab_size)
            x = self.embed(x)
            h_1, c_1 = self.cell(x, (prev["hidden_state"], prev["cell_state"]))
            logits = self.ff(h_1)
            return (
                torch.nn.functional.log_softmax(logits, -1),
                {"hidden_state": h_1, "cell_state": c_1},
            )

First, note that the code in :func:`calc_idx_log_probs` has been updated
slightly. Instead of `prev` being a pair ``(hidden_state, cell_state)``, it
is now a dictionary ``{'hidden_state': hidden_state, 'cell_state':
cell_state}``. This has nothing to do with
:class:`ExtractableSequentialLanguageModel` - none of the interfaces
particulary care about the contents of `prev` or `cur` (though dictionaries of
tensors are compatible with `TorchScript
<https://pytorch.org/docs/stable/jit.html?highlight=torchscript>`__). The only
other addition is a condition when `idx` is not just a single integer but a
vector of integers of size ``(N,)``. For now, think of ``N`` as the batch size.
The batch elements may no longer refer to the same index, so we gather the
appropriate indices using :func:`torch.Tensor.gather`. Because some batch
elements may not have started yet while others have, we use a mask to replace
the entries where ``idx == 0`` with the start-of-sequence token.

There is a new function called :func:`update_input` as well. This is called in
the forward pass of the LM before any calls to :func:`calc_idx_log_probs` and
is used to initialize the value of `prev`. The function takes the role of the
``prev = [...]`` statement in the previous implementation by initializing the
hidden and cell states with all zeros. The argument `hist` is some prefix of
the token sequence being passed to the language model. Usually and here as
well, the sole purpose of passing `hist` is to determine the batch dimension
``N``. If `prev` already has contents, we assume :func:`update_input` has
already been called once and the states initilialized. This satisfies the
requirement of :func:`update_input` that it be robust to repeated calls, i.e.
``update_input(prev, hist) == update_input(update_input(prev, y), hist)``.
:func:`update_input` was also available in :class:`SequentialLanguageModel`
interface, we just didn't use it.

The only addition unique to the :class:`ExtractableSequentialLanguageModel`
interface, therefore, is the method :func:`extract_by_src`.
:func:`extract_by_src` provides a means for the search code to rearrange the LM
state (`prev`) along the batch dimension, ``N``, in order to produce an updated
version of the state `updated`. `src` is a tensor of shape ``(N',)``, where
``N`` is not always equal to ``N'``, containing indices ``[0, N)`` to select
along the batch dimension of tensors in `prev` to produce `updated`. If a
tensor in `prev`, `prev_x`, has shape ``(*, N, *)``, then the corresponding
tensor in `updated`, `updated_x`, should be of shape ``(*, N', *)`` and have
values ``updated_x[..., src[n], ...] = prev_x[..., n, ...]``. This can normally
be accomplished with the function :func:`torch.Tensor.index_select`, as can be
seen above. For :class:`RNNLM`, we perform an index select along the batch
dimension (``0``) for both the hidden and cell states, returning an updated
dictionary.

Peeling the hood back a bit, search functions keep track of a number of
candidate paths, extending some and pruning others according to their
probabilities. The dimension ``N`` is actually a flattened combination of
``batch_size * previous_beam_width`` while ``N'`` is ``batch_size *
current_beam_width``. :func:`extract_by_src` allows the search to select the
states of the paths that survived. The takeaway from an implementation
perspective is that the batch size of any tensors in the methods of
:class:`RNNLM` are not guaranteed to match those of the tensors the module was
passed as arguments (`batch_size` above).

With the updates to the model code complete, the updated code for training and
decoding is as follows:

.. code-block:: python

    from pydrobert.torch.modules import BeamSearch

    torch.manual_seed(1)
    vocab_size, batch_size, sequence_length, epochs, eos = 520, 5, 15, 30, 0
    beam_width, pad = 5, -1
    lm = RNNLM(vocab_size)
    lens = torch.randint(sequence_length, (batch_size,))
    text = [torch.randint(1, vocab_size, (x + 1,)) for x in lens]
    for text_n in text:
        text_n[-1] = eos
    text = torch.nn.utils.rnn.pad_sequence(text, padding_value=pad)

    # training
    optim = torch.optim.Adam(lm.parameters())
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=pad)
    for epoch in range(epochs):
        optim.zero_grad()
        hist = text[:-1].clamp(min=0)
        logits = lm(hist)
        loss = ce_loss(logits.flatten(0, 1), text.flatten())
        loss.backward()
        optim.step()
    
    # decoding
    search = BeamSearch(lm, beam_width, eos)
    with torch.no_grad():
        y, y_lens, log_probs = search()
    print('top path:', y[:, 0], 'log_prob', log_probs[0])

The training code is similar to that we had before, except now we handle
sequences of different lengths with an end-of-sequence (`eos`) type and a
padding (`pad`) type. We append an end-of-sequence token to the end of each
token sequence, followed by as many padding tokens as is necessary to match the
length of every other sequence. The results are concatenated together by
:func:`torch.nn.utils.rnn.pad_sequence` into the tensor `text`. The loss
function ignores the padded values. This training code would work just as well
with our previous version of :class:`RNNLM`.

The decoding code is much simpler than that we used for the random walk. We
merely create a :class:`pydrobert.torch.modules.BeamSearch` module, pass the LM,
beam width, and end-of-sequence type to it, and then call the module. The first
argument to the module is `y_prev`. Usually this is just an empty tensor of
shape ``(0, batch_size)``, though it can be of size ``(S, batch_size)`` to pass
prefixes to the search to continue off of. Here, all the batch elements will
yield the same results because the search is deterministic and :class:`RNNLM`
is not conditioned on any other input. The search returns a triple ``y, lens,
log_probs``. ``y`` is of shape ``(S', batch_size, beam_width)`` where ``y[s, n,
k]`` is the ``s``-th token of the ``k``-th most probable path of the ``n``-th
batch element; ``lens`` is of shape ``(batch_size, beam_width)`` where
``lens[n, k]`` is the length of the ``k``-th most probable path of the ``n``-th
batch element in ``y`` (i.e. values in ``y[lens[n, k]:, n, k]`` are padding);
and ``log_probs`` is of shape ``(batch_size, beam_width)`` containing the
(pseudo-)log probabilities of each path.

Extending :class:`RNNLM` for a CTC prefix search with shallow fusion requires
implementing :class:`pydrobert.torch.modules.MixableSequentialLanguageModel`.
The interface adds only one additional method but is otherwise identical to the
previous implementation. For brevity, we forego rewriting the other methods
below:

.. code-block:: python

    import torch
    from pydrobert.torch.modules import MixableSequentialLanguageModel

    class RNNLM(MixableSequentialLanguageModel):

        # ...
        
        def mix_by_mask(self, prev_true, prev_false, mask):
            return {
                "hidden_state": torch.where(mask.unsqueeze(1), prev_true["hidden_state"], prev_false["hidden_state"]),
                "cell_state": torch.where(mask.unsqueeze(1), prev_true["cell_state"], prev_false["cell_state"]),
            }

The method :func:`mix_by_mask` allows the search to pick and choose parts of
two separate state dictionaries via a boolean switch. `mask` is a boolean
tensor of shape ``(N,)`` and the batch index of the tensors in *both*
`prev_true` and `prev_false` should also be equal to ``N``. The method returns
a merged state dictionary `updated` such that, for tensors `prev_true_x`,
`prev_false_x`, and `updated_x` in `prev_true`, `prev_false`, and `updated`,
respectively, all of shape ``(*, N, *)``, ``updated_x[..., n, ...] ==
prev_true_x[..., n, ...] if mask[n] == True else prev_false_x[..., n, ...]``.
This can usually be accomplished with :func:`torch.where`. The above
:func:`mix_by_mask` does so for both the hidden and cell states of the LSTM.

Why is this necessary? A CTC prefix search may sometimes choose to emit a token
which is reduced into the previously emitted token, i.e. when emitting a
duplicate or blank token. For these paths, we want to revert the state of the
LM to whatever it was before the token was emitted. Since we don't want to
revert the state for all paths (some may have emitted), we require the method
:func:`mix_by_mask`. A similar situation occurs in a beam search when one or
more paths have ended (via an `eos`) while others continue, but we don't bother
rolling back the LM state then because we ignore all the probabilities output
for those paths anyways. From an implementation perspective, it's worth keeping
in mind that `prev_true` and `prev_false` come from different steps in the
decoding process. This will matter if any of the state tensors change size over
subsequent steps, for example.

The training code is identical to above, so we forego it below. The decoding
code has been updated for CTC:

.. code-block:: python

    from pydrobert.torch.modules import CTCPrefixSearch

    torch.manual_seed(2)
    
    # ...

    # decoding
    ctc_logits = torch.randn(sequence_length + 10, batch_size, vocab_size + 1)
    ctc_lens = lens + 10
    search = CTCPrefixSearch(beam_width, lm=lm)
    with torch.no_grad():
        y, y_lens, probs = search(ctc_logits, ctc_lens)
    for n in range(batch_size):
        print(f'top path {n}:', y[:y_lens[n, 0], n, 0], 'prob', probs[n, 0])

`ctc_logits` is a tensor of shape ``(T, batch_size, vocab_size + 1)``
representing the output of an acoustic model. The vocabulary dimension is one
larger than the vocabulary size; the logits for the blank label are stored in
``ctc_logits[..., vocab_size]``. `ctc_lens` functions similarly to `y_lens`
above but for `ctc_logits` instead of `y`: the logits
``ctc_logits[ctc_lens[n]:, n]`` are all padding and thus should be ignored. We
no longer need to consider `eos` in decoding because the total number of steps
is dictated by the sequence dimension of `ctc_logits`, ``T``. The search is
passed `ctc_logits` and `ctc_lens`, returning a triplet. The only difference
between the interpretation of the returned values from :class:`BeamSearch` is
that the final element, `probs`, are the (pseudo-)probabilities rather than the
(pseudo-)log probabilities.

You may have noticed that the final implementation of :class:`RNNLM` is
entirely compatible with the previous usages: the :class:`RNNLM` for
:class:`CTCPrefixSearch` can be passed to :class:`BeamSearch`, and both those
versions can be used to perform a random walk or determine the probability of a
token sequence. For most cases, I suspect the only disadvantage implementing
:class:`MixableSequentialLanguageModel` over
:class:`ExtractableSequentialLanguageModel` over
:class:`SequentialLanguageModel` is a time commitment. Non-sequential language
models like BERT [bert2019]_ won't be able to implement any of them.

Extensions
----------

We can extend the above example in a few ways which we will cover here: the LM
architecture can be updated, the training pass made more efficient, or the beam
search can be modified.

There are a variety of LM architectures which can be considered sequential, at
least with respect to the output token sequences. A straightforward extension
to the :class:`RNNLM` above is to turn it into a encoder-decoder architecture.
An encoder-decoder, a mainstay in Neural Machine Translation (NMT) [cho2014]_
and ASR [chan2016]_, is effectively an RNN LM which conditions the token
sequence on some input `in_` via attention. More about attention is discussed
in :ref:`advanced-attn`. Here's an implementation:

.. code-block:: python

    import torch
    from pydrobert.torch.modules import (
        MixableSequentialLanguageModel,
        DotProductSoftAttention,
        BeamSearch,
    )

    class EncoderDecoder(MixableSequentialLanguageModel):

        def __init__(self, in_size, vocab_size, embed_size=128, hidden_size=512):
            super().__init__(vocab_size)
            self.hidden_size = hidden_size
            self.encoder = torch.nn.LSTM(in_size, hidden_size)
            self.attention = DotProductSoftAttention(hidden_size, 0)
            self.embed = torch.nn.Embedding(
                vocab_size + 1, embed_size, padding_idx=vocab_size
            )
            self.cell = torch.nn.LSTMCell(embed_size + hidden_size, hidden_size)
            self.ff = torch.nn.Linear(hidden_size, vocab_size)
        
        def update_input(self, prev, hist):
            if "in" not in prev:
                return prev  # already initialized
            in_ = prev["in"]  # (T, N, in_size)
            N = hist.size(1)
            assert N == in_.size(1)
            encoding = self.encoder(in_)[0]  # (T, N, hidden_size)
            zeros = self.ff.weight.new_zeros((N, self.hidden_size))
            return {"hidden_state": zeros, "cell_state": zeros, "encoding": encoding}
        
        def extract_by_src(self, prev, src):
            return {
                "hidden_state": prev["hidden_state"].index_select(0, src),
                "cell_state": prev["cell_state"].index_select(0, src),
                "encoding": prev["encoding"].index_select(1, src)
            }
        
        def mix_by_mask(self, prev_true, prev_false, mask):
            # the encoding doesn't change each step, so we don't bother with torch.where
            return {
                "hidden_state": torch.where(mask.unsqueeze(1), prev_true["hidden_state"], prev_false["hidden_state"]),
                "cell_state": torch.where(mask.unsqueeze(1), prev_true["cell_state"], prev_false["cell_state"]),
                "encoding": prev_true["encoding"]
            }

        def calc_idx_log_probs(self, hist, prev, idx):
            idx_zero = idx == 0
            if idx_zero.all():
                x = idx.new_full((hist.size(1),), self.vocab_size)
            elif not idx.dim():
                x = hist[idx - 1]
            else:
                x = hist.gather(0, (idx - 1).clamp(min=0).unsqueeze(0)).squeeze(0)
                x = x.masked_fill(idx_zero, self.vocab_size)
            x = self.embed(x)
            encoding = prev["encoding"]
            ctx = self.attention(prev["hidden_state"], encoding, encoding)
            x = torch.cat([x, ctx], 1)
            h_1, c_1 = self.cell(x, (prev["hidden_state"], prev["cell_state"]))
            logits = self.ff(h_1)
            return (
                torch.nn.functional.log_softmax(logits, -1),
                {"hidden_state": h_1, "cell_state": c_1, "encoding": encoding},
            )
        
    torch.manual_seed(3)
    vocab_size, batch_size, sequence_length, epochs, eos = 520, 5, 15, 100, 0
    beam_width, pad, in_size, in_length = 5, -1, 30, 20
    lm = EncoderDecoder(in_size, vocab_size)
    lens = torch.randint(sequence_length, (batch_size,))
    text = [torch.randint(1, vocab_size, (x + 1,)) for x in lens]
    for text_n in text:
        text_n[-1] = eos
    text = torch.nn.utils.rnn.pad_sequence(text, padding_value=pad)
    in_ = torch.randn(in_length, batch_size, in_size)

    # training
    optim = torch.optim.Adam(lm.parameters())
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=pad)
    for epoch in range(epochs):
        optim.zero_grad()
        hist = text[:-1].clamp(min=0)
        logits = lm(hist, {"in": in_})
        loss = ce_loss(logits.flatten(0, 1), text.flatten())
        loss.backward()
        optim.step()
    
    # decoding
    search = BeamSearch(lm, beam_width, eos)
    with torch.no_grad():
        y, y_lens, log_probs = search({"in": in_}, batch_size)
    for n in range(batch_size):
        print(f'top path {n}:', y[:y_lens[n, 0], n, 0], 'log_prob', log_probs[n, 0])

Here we take advantage of passing the initial state in both the call to the
`lm` and `search` instances to pass the initial input tensor `in_` to the LM.
On the first call to :func:`update_input`, the input tensor is fed into the
encoder network and the output, `encoding`, is passed alongside the decoder
LSTM states in the dictionary. The `encoding` is used in each call to
:func:`calc_idx_log_probs` to create a context vector `ctx`, which is
concatenated with the embedding and fed into the decoder LSTM. We've included
code for :class:`BeamSearch` decoding, but :class:`EncoderDecoder` is
compatible with :class:`CTCPrefixSearch` as well.

With a little effort, the RNNs in :class:`EncoderDecoder` can be replaced with
stacks of attention layers like a Transformer network [vaswani2017]_. The
encoder part can be handled the same way as above. The attention-based
auto-regressive decoder's recursion on states is generally difficult to
memoize, though it is possible to do so via this interface. It is much easier,
however, to implement an attention-based decoder which just recalculates all
its hidden states every time :func:`calc_idx_log_probs` is called using all the
values of `hist`.

The class :class:`pydrobert.torch.modules.LookupLanguageModel`, which loads
pre-trained n-gram language models, implements
:class:`MixableSequentialLanguageModel` and is therefore compatible with both
:class:`BeamSearch` and :class:`CTCPrefixSearch`.

We now move on to a key efficiency improvement applicable to all models covered
so far. Auto-regressive sequential language models are usually trained (as
above) by feeding the entire gold-standard token sequence as input to the LM,
disregarding the "auto-regressive" feedback loop. Having access to the entire
input sequence at once may allow the LM to use more efficient subroutines than
a simple for loop. :class:`SequentialLanguageModel` contains a method called
:func:`calc_full_log_probs` with a default implementation:

.. code-block:: python

    class SequentialLanguageModel(torch.nn.Module):

        # ...
        
        def calc_full_log_probs(self, hist, prev):
            log_probs = []
            for idx in torch.arange(hist.size(0) + 1, device=hist.device):
                log_probs_idx, prev = self.calc_idx_log_probs(hist, prev, idx)
                log_probs.append(log_probs_idx)
            return torch.stack(log_probs, 0)

The method returns a single tensor of shape ``(sequence_length, batch_size,
vocab_size)`` by stacking the results of successive calls to
:func:`calc_idx_log_probs`. A subclass may reimplement this method. For
example, our :class:`RNNLM` can implement it as:

.. code-block:: python

    import torch
    from pydrobert.torch.modules import MixableSequentialLanguageModel, BeamSearch

    class RNNLM(MixableSequentialLanguageModel):
        def __init__(self, vocab_size, embed_size=128, hidden_size=512):
            super().__init__(vocab_size)
            self.hidden_size = hidden_size
            self.embed = torch.nn.Embedding(
                vocab_size + 1, embed_size, padding_idx=vocab_size
            )
            self.cell = torch.nn.LSTMCell(embed_size, hidden_size)
            self.lstm = torch.nn.LSTM(embed_size, hidden_size)
            self.lstm.weight_ih_l0 = self.cell.weight_ih
            self.lstm.weight_hh_l0 = self.cell.weight_hh
            self.lstm.bias_ih_l0 = self.cell.bias_ih
            self.lstm.bias_hh_l0 = self.cell.bias_hh
            self.ff = torch.nn.Linear(hidden_size, vocab_size)
        
        # ...
        
        def calc_full_log_probs(self, hist, prev):
            hist = torch.cat([hist.new_full((1, hist.size(1)), self.vocab_size), hist], 0)
            x = self.embed(hist)
            x = self.lstm(x)[0]
            logits = self.ff(x)
            return torch.nn.functional.log_softmax(logits, -1)

We've shared weights between the :class:`torch.nn.LSTMCell` module instance
`cell` and a :class:`torch.nn.LSTM` module instance `lstm`. Calling the `lstm`
module on the full sequence allows access to more efficient backend routines. A
Transformer network can avoid the recurrence altogether by appropriate masking
of input.

The final extension I'll mention relates to :class:`BeamSearch`. There are a
variety of different flavours of beam search out there. :class:`BeamSearch` is
a no-frills variety which computes the probability of a path as the product of
the probabilities of its tokens and finishes when the most probable path in the
beam is also completed (i.e. ends with an `eos`). Other varieties of beam
search will modify the path probabilities and/or the stopping criteria.
:class:`BeamSearch` supports two additional stopping criteria: all paths in the
beam must be complete, or some cut-off length is achieved. Consult the class
documentation for more detail. More complicated stopping criteria will require
reimplementing beam search, at which point the low-level function
:func:`pydrobert.torch.functional.beam_search_advance` might be a good starting
point. Modifying path probabilities is much easier. To do so, one may sublclass
:class:`BeamSearch` and reimplement the method
:func:`pydrobert.torch.modules.BeamSearch.update_log_probs_for_step`. Here's an
example which normalizes the log probabilities of paths by their lengths:

.. code-block:: python

    from pydrobert.torch.modules import BeamSearch
    
    class LengthNormalizedBeamSearch(BeamSearch):

        def update_log_probs_for_step(
                self,
                log_probs_prev,
                log_probs_t,
                y_prev,
                y_prev_lens,
                eos_mask,
            ):
            num = y_prev_lens.to(log_probs_prev)
            denom = num + 1 - eos_mask.to(log_probs_prev)
            num, denom = num.clamp_(min=1), denom.clamp_(min=1)
            return (
                log_probs_prev * num / denom,
                log_probs_t / denom.unsqueeze(-1)
            )

`log_probs_prev` is the pseudo-log-probabilities of the paths up to the current
step (with normalization); `log_probs_t` contains the log-probabilities of the
tokens extending the paths (without normalization). To renormalize
`log_probs_prev` by the extended length (``y_prev_lens + 1``), we multiply by
the previous normalization constant (``y_prev_lens``) to de-normalize
`log_probs_prev`, then divide by the new one. Since `log_probs_t` is
unnormalized, we just divide by the new constant. When the results are added
together, the extended path pseudo-log-probability will be normalzied by
``y_prev_lens + 1``.

This is just one implementation of many. Consult the documentation of
:func:`pydrobert.torch.modules.BeamSearch.update_log_probs_for_step` for more
information.