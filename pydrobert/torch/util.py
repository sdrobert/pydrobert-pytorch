# Copyright 2019 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Utility functions'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"
__all__ = [
    'beam_search_advance',
    'optimizer_to',
]


def beam_search_advance(logits, width, log_prior=None, y_prev=None, eos=None):
    r'''Advance a beam search

    Suppose a model outputs a un-normalized log-probability distribution over
    the next element of a sequence in `logits` s.t.

    .. math::

        Pr(y_t = c) = exp(logits_c) / \sum_k exp(logits_k)

    `logits` is "auto-regressive" if it is conditioned upon a prefix sequence
    :math:`y_{<t}`. If `logits` is not auto-regressive, beam search (this
    function) is not needed, since `logits` will produce the same probability
    distribution over :math:`y_t` regardless of what it's predicted up to that
    point. So we assume from now on that :math:`logits = f(y_{<t})`.
    Critically, we assume that `logits` is conditionally independent of any
    `logits` from prior timesteps given :math:`y_{<t}`

    Beam search is a heuristic mechanism for determining a best path, i.e.
    :math:`\argmax_y Pr(y)` that maximizes the probability of the best path
    by keeping track of `width` high probability paths called "beams"
    (the aggregate of which for a given time step is named, unfortunately,
    "the beam").

    This function is called at every time step. It updates old beam
    log-probabilities (`log_prior`) with new ones (`score`), and updates
    us the class indices emitted between them (`y`). See the examples section
    for how this might work.

    Parameters
    ----------
    logits : torch.tensor
        The conditional probabilities over class labels for the current tim
        step. Either of shape ``(num_batches, old_width, num_classes)``,
        where ``old_width`` is the number of beams in the previous time step,
        or ``(num_batches, num_classes)``, where it is assumed that
        ``old_width == 1``
    width : int
        The number of beams in the beam to produce for the current time step.
        ``width <= num_classes``
    log_prior : torch.tensor, optional
        A tensor of (or proportional to) log prior probabilities of beams up
        to the previous time step. Either of shape ``(num_batches, old_width)``
        or ``(num_batches,)``. If unspecified, a uniform log prior will be used
    y_prev : torch.LongTensor, optional
        A tensor of shape ``(t - 1, num_batches, old_width)`` or
        ``(t - 1, num_batches)`` specifying :math:`y_{<t}`. If unspecified,
        it is assumed that ``t == 1``
    eos : int, optional
        If both `eos` and `y_prev` are specified, whenever
        ``y_prev[-1, i, j] == eos``, the indexed beam is considered "finished"
        and will not update its value with `logits`. `y` will copy the `eos`
        symbol into :math:`y_t`.


    Returns
    -------
    score, y, s : torch.tensor, torch.LongTensor, torch.LongTensor
        `score` is a tensor of shape ``(num_batches, width)`` of the
        log-joint probabilities of the new beams in the beam. `y` is a
        tensor of shape ``(t, num_batches, width)`` of indices of the class
        labels generated up to this point. `s` is a tensor of shape
        ``(num_batches, width)`` of indices of beams in the old beam which
        prefix the new beam. Note that beams in the new beam are sorted by
        descending probability

    Examples
    --------

    Decoding with beam search, assuming index class 0 is eos and -1 is start

    >>> N, I, C, T, W, H = 5, 5, 10, 100, 5, 10
    >>> cell = torch.nn.RNNCell(I + 1, H)
    >>> ff = torch.nn.Linear(H, C)
    >>> inp = torch.rand(T, N, I)
    >>> y = torch.full((1, N, 1), -1, dtype=torch.long)
    >>> h_t = torch.zeros(N, 1, H)
    >>> score = None
    >>> for inp_t in inp:
    >>>     y_tm1 = y[-1]
    >>>     old_width = y_tm1.shape[-1]
    >>>     inp_t = inp_t.unsqueeze(1).expand(N, old_width, I)
    >>>     x_t = torch.cat([inp_t, y_tm1.unsqueeze(2).float()], -1)
    >>>     h_t = cell(
    ...         x_t.view(N * old_width, I + 1),
    ...         h_t.view(N * old_width, H)
    ...     ).view(N, old_width, H)
    >>>     logits_t = ff(h_t)
    >>>     score, y, s_t = beam_search_advance(logits_t, W, score, y, 0)
    >>>     h_t = h_t.gather(1, s_t.unsqueeze(-1).expand(N, W, H))
    >>> bests = []
    >>> for batch_idx in range(N):
    >>>     best_beam_path = y[1:, batch_idx, 0]
    >>>     not_special_mask = best_beam_path.ne(0)
    >>>     best_beam_path = best_beam_path.masked_select(not_special_mask)
    >>>     bests.append(best_beam_path)
    '''
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)
    elif logits.dim() != 3:
        raise ValueError('logits must have dimension of either 2 or 3')
    num_batches, old_width, num_classes = logits.shape
    if log_prior is None:
        log_prior = torch.full(
            (num_batches, old_width),
            -torch.log(torch.tensor(float(num_classes))),
            dtype=logits.dtype, device=logits.device,
        )
    elif tuple(log_prior.shape) == (num_batches,) and old_width == 1:
        log_prior = log_prior.unsqueeze(1)
    elif log_prior.shape != logits.shape[:-1]:
        raise ValueError(
            'If logits of shape {} then log_prior must have shape {}'.format(
                (num_batches, old_width, num_classes),
                (num_batches, old_width),
            ))
    if y_prev is not None and y_prev.shape[1:] != log_prior.shape:
        raise ValueError(
            'If logits of shape {} then y_prev must have shape (*, {}, {})'
            ''.format(
                (num_batches, old_width, num_classes),
                num_batches, old_width,
            )
        )
    logits = torch.nn.functional.log_softmax(logits, 2)
    if y_prev is not None and eos is not None:
        if eos < 0:
            raise ValueError('eos must be a valid index')
        done_mask = y_prev[-1].eq(eos)
        num_done = done_mask.sum().item()
        if num_done:
            if num_done == old_width and old_width < width:
                warnings.warn(
                    'New beam width ({}) is wider than old beam width ({}), '
                    'but all paths are already done. Reducing new width.'
                    ''.format(width, old_width))
                width = num_done
            # we want finished beams to only ever be in the top k when the next
            # class in the beam is EOS, so we fill all the class labels of old
            # beams with -inf except EOS, which gets a 0
            done_classes = torch.full_like(logits[0, 0], -float('inf'))
            done_classes[eos] = 0.
            logits = torch.where(
                done_mask.unsqueeze(-1),
                done_classes,
                logits
            )
    joint = log_prior.unsqueeze(2) + logits
    score, idxs = torch.topk(joint.view(num_batches, -1), width, dim=1)
    s = idxs // num_classes
    y = (idxs % num_classes).unsqueeze(0)
    if y_prev is not None:
        tm1 = y_prev.shape[0]
        y_prev = y_prev.gather(
            2, s.unsqueeze(0).expand(tm1, num_batches, width))
        y = torch.cat([y_prev, y], 0)
    return score, y, s


def optimizer_to(optimizer, to):
    '''Move tensors in an optimizer to another device

    This function traverses the state dictionary of an `optimizer` and pushes
    any tensors it finds to the device implied by `to` (as per the semantics
    of ``torch.tensor.to``)
    '''
    state_dict = optimizer.state_dict()
    key_stack = [(x,) for x in state_dict.keys()]
    new_device = False
    while key_stack:
        last_dict = None
        val = state_dict
        keys = key_stack.pop()
        for key in keys:
            last_dict = val
            val = val[key]
        if isinstance(val, torch.Tensor):
            try:
                last_dict[keys[-1]] = val.to(to)
                if last_dict[keys[-1]].device != val.device:
                    new_device = True
            except Exception as e:
                print(e)
                pass
        elif hasattr(val, 'keys'):
            key_stack += [keys + (x,) for x in val.keys()]
    if new_device:
        optimizer.load_state_dict(state_dict)
