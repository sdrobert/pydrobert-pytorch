
Language Modelling and Decoding
===============================

.. code-block:: python

  import torch
  from pydrobert.torch.layers import SequentialLanguageModel

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
though it is when pairing with a search algorithm.

Note at the beginning of the method that we check if `idx == 0`. This is for
when we're generating the first token. Since we can't extract a previous token
from the history to feed into our LSTM, we produce a special, start-of-sequence
token. We add the start-of-sequence type to end of the vocabulary (note the
size of the :class:`Embedding` layer) and replace ``hist[idx - 1]`` with a
tensor of start-of-sequence tokens whenever ``idx == 0``.

