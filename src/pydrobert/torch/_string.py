# Copyright 2022 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import warnings

from typing import Optional

import torch

from . import config
from ._compat import script, pad_sequence


@script
def _lens_from_eos(tok: torch.Tensor, eos: int, dim: int) -> torch.Tensor:
    # length to first eos (exclusive)
    mask = tok.eq(eos)
    x = torch.cumsum(mask, dim, dtype=torch.long)
    max_, argmax = (x.eq(1) & mask).max(dim)
    return argmax.masked_fill(max_.eq(0), tok.shape[dim])


@script
def _string_matching(
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int],
    include_eos: bool,
    batch_first: bool,
    ins_cost: float,
    del_cost: float,
    sub_cost: float,
    warn: bool,
    norm: bool = False,
    return_mask: bool = False,
    return_prf_dsts: bool = False,
    exclude_last: bool = False,
    padding: int = config.INDEX_PAD_VALUE,
    return_mistakes: bool = False,
) -> torch.Tensor:
    assert not return_mask or not return_prf_dsts
    assert not exclude_last or (return_mask or return_prf_dsts)
    if ref.dim() != 2 or hyp.dim() != 2:
        raise RuntimeError("ref and hyp must be 2 dimensional")
    if ins_cost == del_cost == sub_cost > 0.0:
        # results are equivalent and faster to return
        ins_cost = del_cost = sub_cost = 1.0
        return_mistakes = False
    elif return_mistakes and warn:
        warnings.warn(
            "The behaviour for non-uniform error rates has changed after v0.3.0. "
            "Please switch to edit_distance functions for old behaviour. Set "
            "warn=False to suppress this warning"
        )
    if batch_first:
        ref = ref.t()
        hyp = hyp.t()
    mistakes = del_mat = mask = prefix_ers = torch.empty(0)
    ref = ref.detach()
    hyp = hyp.detach()
    max_ref_steps, batch_size = ref.shape
    max_hyp_steps, batch_size_ = hyp.shape
    device = ref.device
    if batch_size != batch_size_:
        raise RuntimeError(
            "ref has batch size {}, but hyp has {}".format(batch_size, batch_size_)
        )
    if eos is not None:
        ref_lens = _lens_from_eos(ref, eos, 0)
        hyp_lens = _lens_from_eos(hyp, eos, 0)
        if include_eos:
            ref_eq_mask = ref_lens == max_ref_steps
            ref_lens = ref_lens + 1
            if ref_eq_mask.any():
                if warn:
                    warnings.warn(
                        "include_eos=True, but a transcription in ref did not "
                        "contain the eos symbol ({}). To suppress this "
                        "warning, set warn=False".format(eos)
                    )
                ref_lens = ref_lens - ref_eq_mask.to(ref_lens.dtype)
            hyp_eq_mask = hyp_lens == max_hyp_steps
            hyp_lens = hyp_lens + 1
            if hyp_eq_mask.any():
                if warn:
                    warnings.warn(
                        "include_eos=True, but a transcription in hyp did not "
                        "contain the eos symbol ({}). To suppress this "
                        "warning, set warn=False".format(eos)
                    )
                hyp_lens = hyp_lens - hyp_eq_mask.to(hyp_lens.dtype)
    else:
        ref_lens = torch.full(
            (batch_size,), max_ref_steps, dtype=torch.long, device=ref.device
        )
        hyp_lens = torch.full(
            (batch_size,), max_hyp_steps, dtype=torch.long, device=ref.device
        )
    if return_mask:
        mask = torch.empty(
            (max_hyp_steps + (0 if exclude_last else 1), max_ref_steps, batch_size),
            device=device,
            dtype=torch.bool,
        )
        mask[0, 0] = 1
        mask[0, 1:] = 0
    elif return_prf_dsts:
        prefix_ers = torch.empty(
            (max_hyp_steps + (0 if exclude_last else 1), batch_size),
            device=device,
            dtype=torch.float,
        )
        prefix_ers[0] = ref_lens * (1.0 if return_mistakes else del_cost)
    # direct row down corresponds to insertion
    # direct col right corresponds to a deletion
    #
    # we vectorize as much as we can. Neither substitutions nor insertions require
    # values from the current row to be computed, and since the last row can't be
    # altered, we can easily vectorize there. To vectorize deletions, we use del_matrix.
    # It has entries
    #
    # 0   inf inf inf ...
    # d   0   inf inf ...
    # 2d  d   0   inf ...
    # ...
    #
    # Where "d" is del_cost. When we sum with the intermediate values of the next row
    # "v" (containing the minimum of insertion and subs costs), we get
    #
    # v[0]    inf     inf     inf ...
    # v[0]+d  v[1]    inf     inf ...
    # v[0]+2d v[1]+d  v[2]    inf ...
    # ...
    #
    # And we take the minimum of each row. The dynamic programming algorithm for
    # levenshtein would usually handle deletions as:
    #
    # for i=1..|v|:
    #     v[i] = min(v[i], v[i-1]+d)
    #
    # if we unroll the loop, we get the minimum of the elements of each row of the above
    # matrix
    row = torch.arange(max_ref_steps + 1, device=device, dtype=torch.float)  # (R+1, N)
    if return_mistakes:
        mistakes = row.unsqueeze(1).expand(max_ref_steps + 1, batch_size)
        row = row * del_cost
    else:
        row *= del_cost
        del_mat = row.unsqueeze(1) - row
        del_mat = del_mat + torch.full_like(del_mat, float("inf")).triu(1)
        del_mat = del_mat.unsqueeze(-1)  # (R + 1, R + 1, 1)
    row = row.unsqueeze(1).expand(max_ref_steps + 1, batch_size)
    for hyp_idx in range(1, max_hyp_steps + (0 if exclude_last else 1)):
        not_done = (hyp_idx - (0 if exclude_last else 1)) < hyp_lens
        last_row = row
        ins_mask = (hyp_lens >= hyp_idx).float()  # (N,)
        neq_mask = (ref != hyp[hyp_idx - 1]).float()  # (R + 1, N)
        row = last_row + ins_cost * ins_mask
        sub_row = last_row[:-1] + sub_cost * neq_mask
        if return_mistakes:
            # The kicker is substitutions over insertions or deletions.
            pick_sub = row[1:] >= sub_row
            row[1:] = torch.where(pick_sub, sub_row, row[1:])
            last_mistakes = mistakes
            mistakes = last_mistakes + ins_mask
            msub_row = last_mistakes[:-1] + neq_mask
            mistakes[1:] = torch.where(pick_sub, msub_row, mistakes[1:])
            # FIXME(sdrobert): the min function behaves non-determinically r.n.
            # (regardless of what the 1.7.0 docs say!) so techniques for extracting
            # indices from the min are a wash. If we can get determinism, we can flip
            # the 1 dimension if (del_mat + row) before the min and get the least idx
            # min, which should have the fewest number of deletions.
            for ref_idx in range(1, max_ref_steps + 1):
                del_ = row[ref_idx - 1] + del_cost
                pick_sub = del_ >= row[ref_idx]
                row[ref_idx] = torch.where(pick_sub, row[ref_idx], del_)
                mistakes[ref_idx] = torch.where(
                    pick_sub, mistakes[ref_idx], mistakes[ref_idx - 1] + 1.0
                )
            mistakes = torch.where(not_done, mistakes, last_mistakes)
        else:
            row[1:] = torch.min(row[1:], sub_row)
            row, _ = (del_mat + row).min(1)
        row = torch.where(not_done, row, last_row)
        if return_mask:
            # As proven in the OCD paper, the optimal targets are always the first
            # character of a suffix of the reference transcript that remains to be
            # aligned. The levenshtein operation corresponding to what we do with that
            # target would be a matched substitution (i.e. hyp's next token is the OCD
            # target, resulting in no change in cost from the prefix). Thus, given a
            # levenshtein matrix for one of these OCD targets (which is this matrix,
            # except for the final row), the minimal values on the final row sit on a
            # diagonal from the minimal values of the current row.
            mins = row.min(0, keepdim=True)[0]
            row_mask = row[:-1] == mins
            row_mask = row_mask & not_done
            mask[hyp_idx] = row_mask
        elif return_prf_dsts:
            if return_mistakes:
                prefix_ers[hyp_idx] = mistakes.gather(0, ref_lens.unsqueeze(0)).squeeze(
                    0
                )
            else:
                prefix_ers[hyp_idx] = row.gather(0, ref_lens.unsqueeze(0)).squeeze(0)
    if return_mask:
        mask = mask & (
            (
                torch.arange(max_ref_steps, device=device)
                .unsqueeze(1)
                .expand(max_ref_steps, batch_size)
                < ref_lens
            ).unsqueeze(0)
        )
        return mask
    elif return_prf_dsts:
        if norm:
            prefix_ers = prefix_ers / ref_lens.to(row.dtype)
            zero_mask = ref_lens.eq(0).unsqueeze(0)
            if zero_mask.any():
                if warn:
                    warnings.warn(
                        "ref contains empty transcripts. Error rates will be "
                        "0 for prefixes of length 0, 1 otherwise. To suppress "
                        "this warning, set warn=False"
                    )
                prefix_ers = torch.where(
                    zero_mask,
                    (
                        torch.arange(prefix_ers.size(0), device=device)
                        .gt(0)
                        .to(row.dtype)
                        .unsqueeze(1)
                        .expand_as(prefix_ers)
                    ),
                    prefix_ers,
                )
        prefix_ers = prefix_ers.masked_fill(
            (
                torch.arange(prefix_ers.size(0), device=device)
                .unsqueeze(1)
                .ge(hyp_lens + (0 if exclude_last else 1))
            ),
            padding,
        )
        if batch_first:
            prefix_ers = prefix_ers.t()
        return prefix_ers
    if return_mistakes:
        er = mistakes.gather(0, ref_lens.unsqueeze(0)).squeeze(0)
    else:
        er = row.gather(0, ref_lens.unsqueeze(0)).squeeze(0)
    if norm:
        er = er / ref_lens.to(er.dtype)
        zero_mask = ref_lens.eq(0)
        if zero_mask.any():
            if warn:
                warnings.warn(
                    "ref contains empty transcripts. Error rates for entries "
                    "will be 1 if any insertion and 0 otherwise. To suppress "
                    "this warning, set warn=False"
                )
            er = torch.where(zero_mask, hyp_lens.gt(0).to(er.dtype), er)
    return er


def error_rate(
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int] = None,
    include_eos: bool = False,
    norm: bool = True,
    batch_first: bool = False,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
    sub_cost: float = 1.0,
    warn: bool = True,
) -> torch.Tensor:
    """Functional version of ErrorRate

    See Also
    --------
    pydrobert.torch.layers.ErrorRate
    """
    return _string_matching(
        ref,
        hyp,
        eos,
        include_eos,
        batch_first,
        ins_cost,
        del_cost,
        sub_cost,
        warn,
        norm=norm,
        return_mistakes=True,
    )


def edit_distance(
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int] = None,
    include_eos: bool = False,
    norm: bool = False,
    batch_first: bool = False,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
    sub_cost: float = 1.0,
    warn: bool = True,
) -> torch.Tensor:
    """Functional version of EditDistance

    See Also
    --------
    pydrobert.torch.layers.EditDistance
        For information about this module and its associated parameters.
    """
    return _string_matching(
        ref,
        hyp,
        eos,
        include_eos,
        batch_first,
        ins_cost,
        del_cost,
        sub_cost,
        warn,
        norm=norm,
    )


@script
def optimal_completion(
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int] = None,
    include_eos: bool = True,
    batch_first: bool = False,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
    sub_cost: float = 1.0,
    padding: int = config.INDEX_PAD_VALUE,
    exclude_last: bool = False,
    warn: bool = True,
) -> torch.Tensor:
    """Functional version of OptimalCompletion
    
    See Also
    --------
    pydrobert.torch.layers.OptimalCompletion
        For information about this module and its associated parameters.
    """
    mask = _string_matching(
        ref,
        hyp,
        eos,
        include_eos,
        batch_first,
        ins_cost,
        del_cost,
        sub_cost,
        warn,
        return_mask=True,
        exclude_last=exclude_last,
    )
    max_hyp_steps_p1, _, batch_size = mask.shape
    targets = []
    if batch_first:
        for mask_bt, ref_bt in zip(mask.transpose(0, 2), ref):
            for mask_bt_hyp in mask_bt.t():
                targets.append(torch.unique(ref_bt.masked_select(mask_bt_hyp)))
    else:
        for mask_hyp in mask:
            for mask_hyp_bt, ref_bt in zip(mask_hyp.t(), ref.t()):
                targets.append(torch.unique(ref_bt.masked_select(mask_hyp_bt)))
    # the cast to float is a concession for scripting
    targets = pad_sequence(targets, padding_value=float(padding), batch_first=True)
    if batch_first:
        targets = targets.view(batch_size, max_hyp_steps_p1, -1)
    else:
        targets = targets.view(max_hyp_steps_p1, batch_size, -1)
    return targets


def prefix_error_rates(
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int] = None,
    include_eos: bool = True,
    norm: bool = True,
    batch_first: bool = False,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
    sub_cost: float = 1.0,
    padding: int = config.INDEX_PAD_VALUE,
    exclude_last: bool = False,
    warn: bool = True,
) -> torch.Tensor:
    """Functional version of PrefixErrorRates
    
    See Also
    --------
    pydrobert.torch.layers.PrefixErrorRates
        For information about this module and its associated parameters.
    """
    return _string_matching(
        ref,
        hyp,
        eos,
        include_eos,
        batch_first,
        ins_cost,
        del_cost,
        sub_cost,
        warn,
        norm=norm,
        return_prf_dsts=True,
        exclude_last=exclude_last,
        padding=padding,
        return_mistakes=True,
    )


def prefix_edit_distances(
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int] = None,
    include_eos: bool = True,
    norm: bool = False,
    batch_first: bool = False,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
    sub_cost: float = 1.0,
    padding: int = config.INDEX_PAD_VALUE,
    exclude_last: bool = False,
    warn: bool = True,
) -> torch.Tensor:
    """Functional version of PrefixEditDistances

    See Also
    --------
    pydrobert.torch.layers.PrefixEditDistances
        For information about this module and its associated parameters.
    """
    return _string_matching(
        ref,
        hyp,
        eos,
        include_eos,
        batch_first,
        ins_cost,
        del_cost,
        sub_cost,
        warn,
        norm=norm,
        return_prf_dsts=True,
        exclude_last=exclude_last,
        padding=padding,
        return_mistakes=False,
    )


@script
def _string_matching(
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int],
    include_eos: bool,
    batch_first: bool,
    ins_cost: float,
    del_cost: float,
    sub_cost: float,
    warn: bool,
    norm: bool = False,
    return_mask: bool = False,
    return_prf_dsts: bool = False,
    exclude_last: bool = False,
    padding: int = config.INDEX_PAD_VALUE,
    return_mistakes: bool = False,
):
    assert not return_mask or not return_prf_dsts
    assert not exclude_last or (return_mask or return_prf_dsts)
    if ref.dim() != 2 or hyp.dim() != 2:
        raise RuntimeError("ref and hyp must be 2 dimensional")
    mult = 1.0
    if ins_cost == del_cost == sub_cost > 0.0:
        # results are equivalent and faster to return
        if not return_mistakes:
            mult = ins_cost
        ins_cost = del_cost = sub_cost = 1.0
        return_mistakes = False
    elif return_mistakes and warn:
        warnings.warn(
            "The behaviour for non-uniform error rates has changed after v0.3.0. "
            "Please switch to edit_distance functions for old behaviour. Set "
            "warn=False to suppress this warning"
        )
    if batch_first:
        ref = ref.t()
        hyp = hyp.t()
    mistakes = del_mat = mask = prefix_ers = torch.empty(0)
    ref = ref.detach()
    hyp = hyp.detach()
    max_ref_steps, batch_size = ref.shape
    max_hyp_steps, batch_size_ = hyp.shape
    device = ref.device
    if batch_size != batch_size_:
        raise RuntimeError(
            "ref has batch size {}, but hyp has {}".format(batch_size, batch_size_)
        )
    if eos is not None:
        ref_lens = _lens_from_eos(ref, eos, 0)
        hyp_lens = _lens_from_eos(hyp, eos, 0)
        if include_eos:
            ref_eq_mask = ref_lens == max_ref_steps
            ref_lens = ref_lens + 1
            if ref_eq_mask.any():
                if warn:
                    warnings.warn(
                        "include_eos=True, but a transcription in ref did not "
                        "contain the eos symbol ({}). To suppress this "
                        "warning, set warn=False".format(eos)
                    )
                ref_lens = ref_lens - ref_eq_mask.to(ref_lens.dtype)
            hyp_eq_mask = hyp_lens == max_hyp_steps
            hyp_lens = hyp_lens + 1
            if hyp_eq_mask.any():
                if warn:
                    warnings.warn(
                        "include_eos=True, but a transcription in hyp did not "
                        "contain the eos symbol ({}). To suppress this "
                        "warning, set warn=False".format(eos)
                    )
                hyp_lens = hyp_lens - hyp_eq_mask.to(hyp_lens.dtype)
    else:
        ref_lens = torch.full(
            (batch_size,), max_ref_steps, device=ref.device, dtype=torch.long
        )
        hyp_lens = torch.full(
            (batch_size,), max_hyp_steps, device=ref.device, dtype=torch.long
        )
    if return_mask:
        mask = torch.empty(
            (max_hyp_steps + (0 if exclude_last else 1), max_ref_steps, batch_size),
            device=device,
            dtype=torch.bool,
        )
        mask[0, 0] = ref_lens > 0
        mask[0, 1:] = 0
    elif return_prf_dsts:
        prefix_ers = torch.empty(
            (max_hyp_steps + (0 if exclude_last else 1), batch_size),
            device=device,
            dtype=torch.float,
        )
        prefix_ers[0] = ref_lens * (1.0 if return_mistakes else del_cost)
    # direct row down corresponds to insertion
    # direct col right corresponds to a deletion
    #
    # we vectorize as much as we can. Neither substitutions nor insertions require
    # values from the current row to be computed, and since the last row can't be
    # altered, we can easily vectorize there. To vectorize deletions, we use del_matrix.
    # It has entries
    #
    # 0   inf inf inf ...
    # d   0   inf inf ...
    # 2d  d   0   inf ...
    # ...
    #
    # Where "d" is del_cost. When we sum with the intermediate values of the next row
    # "v" (containing the minimum of insertion and subs costs), we get
    #
    # v[0]    inf     inf     inf ...
    # v[0]+d  v[1]    inf     inf ...
    # v[0]+2d v[1]+d  v[2]    inf ...
    # ...
    #
    # And we take the minimum of each row. The dynamic programming algorithm for
    # levenshtein would usually handle deletions as:
    #
    # for i=1..|v|:
    #     v[i] = min(v[i], v[i-1]+d)
    #
    # if we unroll the loop, we get the minimum of the elements of each row of the above
    # matrix
    rrange = torch.arange(max_ref_steps + 1, device=device, dtype=torch.float)
    if return_mistakes:
        mistakes = rrange.unsqueeze(1).expand(max_ref_steps + 1, batch_size)
        row = rrange * del_cost
    else:
        row = rrange * del_cost
        del_mat = row.unsqueeze(1) - row
        del_mat = del_mat + torch.full_like(del_mat, float("inf")).triu(1)
        del_mat = del_mat.unsqueeze(-1)  # (R + 1, R + 1, 1)
    row = row.unsqueeze(1).expand(max_ref_steps + 1, batch_size)
    for hyp_idx in range(1, max_hyp_steps + (0 if exclude_last else 1)):
        not_done = (hyp_idx - (0 if exclude_last else 1)) < hyp_lens
        last_row = row
        ins_mask = (hyp_lens >= hyp_idx).float()  # (N,)
        neq_mask = (ref != hyp[hyp_idx - 1]).float()  # (R + 1, N)
        row = last_row + ins_cost * ins_mask
        sub_row = last_row[:-1] + sub_cost * neq_mask
        if return_mistakes:
            # The kicker is substitutions over insertions over deletions.
            pick_sub = row[1:] >= sub_row
            row[1:] = torch.where(pick_sub, sub_row, row[1:])
            last_mistakes = mistakes
            mistakes = last_mistakes + ins_mask
            msub_row = last_mistakes[:-1] + neq_mask
            mistakes[1:] = torch.where(pick_sub, msub_row, mistakes[1:])
            # FIXME(sdrobert): the min function behaves non-determinically r.n.
            # (regardless of what the 1.7.0 docs say!) so techniques for extracting
            # indices from the min are a wash. If we can get determinism, we can flip
            # the 1 dimension if (del_mat + row) before the min and get the least idx
            # min, which should have the fewest number of deletions.
            for ref_idx in range(1, max_ref_steps + 1):
                del_ = row[ref_idx - 1] + del_cost
                pick_sub = del_ >= row[ref_idx]
                row[ref_idx] = torch.where(pick_sub, row[ref_idx], del_)
                mistakes[ref_idx] = torch.where(
                    pick_sub, mistakes[ref_idx], mistakes[ref_idx - 1] + 1.0
                )
            mistakes = torch.where(not_done, mistakes, last_mistakes)
        else:
            row[1:] = torch.min(row[1:], sub_row)
            row, _ = (del_mat + row).min(1)
        row = torch.where(not_done, row, last_row)
        if return_mask:
            # As proven in the OCD paper, the optimal targets are always the first
            # character of a suffix of the reference transcript that remains to be
            # aligned. The levenshtein operation corresponding to what we do with that
            # target would be a matched substitution (i.e. hyp's next token is the OCD
            # target, resulting in no change in cost from the prefix). Thus, given a
            # levenshtein matrix for one of these OCD targets (which is this matrix,
            # except for the final row), the minimal values on the final row sit on a
            # diagonal from the minimal values of the current row.
            #
            # N.B. AFAICT this is the only case where we actually care what goes on in
            # the invalid range of the row. The below masking could always be applied,
            # but it's wasted effort otherwise.
            row = row.masked_fill(rrange.unsqueeze(1) > ref_lens, float("inf"))
            mins = row.min(0, keepdim=True)[0]
            row_mask = (row[:-1] == mins) & not_done
            mask[hyp_idx] = row_mask
        elif return_prf_dsts:
            if return_mistakes:
                prefix_ers[hyp_idx] = mistakes.gather(0, ref_lens.unsqueeze(0)).squeeze(
                    0
                )
            else:
                prefix_ers[hyp_idx] = row.gather(0, ref_lens.unsqueeze(0)).squeeze(0)
    if return_mask:
        mask &= (
            torch.arange(max_ref_steps, device=device)
            .unsqueeze(1)
            .expand(max_ref_steps, batch_size)
            < ref_lens
        ).unsqueeze(0)
        return mask
    elif return_prf_dsts:
        prefix_ers = prefix_ers * mult
        if norm:
            prefix_ers = prefix_ers / ref_lens.to(row.dtype)
            zero_mask = (ref_lens == 0).unsqueeze(0)
            if zero_mask.any():
                if warn:
                    warnings.warn(
                        "ref contains empty transcripts. Error rates will be "
                        "0 for prefixes of length 0, 1 otherwise. To suppress "
                        "this warning, set warn=False"
                    )
                prefix_ers = torch.where(
                    zero_mask,
                    (
                        torch.arange(prefix_ers.size(0), device=device)
                        .gt(0)
                        .to(row.dtype)
                        .unsqueeze(1)
                        .expand_as(prefix_ers)
                    ),
                    prefix_ers,
                )
        prefix_ers = prefix_ers.masked_fill(
            (
                torch.arange(prefix_ers.size(0), device=device)
                .unsqueeze(1)
                .ge(hyp_lens + (0 if exclude_last else 1))
            ),
            padding,
        )
        if batch_first:
            prefix_ers = prefix_ers.t()
        return prefix_ers
    if return_mistakes:
        er = mistakes.gather(0, ref_lens.unsqueeze(0)).squeeze(0)
    else:
        er = row.gather(0, ref_lens.unsqueeze(0)).squeeze(0)
    er = er * mult
    if norm:
        er = er / ref_lens.to(er.dtype)
        zero_mask = ref_lens.eq(0)
        if zero_mask.any():
            if warn:
                warnings.warn(
                    "ref contains empty transcripts. Error rates for entries "
                    "will be 1 if any insertion and 0 otherwise. To suppress "
                    "this warning, set warn=False"
                )
            er = torch.where(zero_mask, hyp_lens.gt(0).to(er.dtype), er)
    return er


_SM_ARGS = """\
`ref` is a long tensor of shape ``(max_ref_steps, batch_size)`` such that the
``n``-th reference (gold-standard) token sequence is stored in ``ref[:, n]``. `hyp`
is a long tensor of shape ``(max_hyp_steps, batch_size)`` containing the hypothesis
(machine-generated) sequences."""

_SM_PARAM_DICT = {
    "eos": """\
    eos : int or None, optional
        A special token in `ref` and `hyp` whose first occurrence in each batch
        indicates the end of a transcript. This allows for variable-length transcripts
        in the batch.
    """,
    "include_eos": """\
    include_eos : bool, optional
        Whether to include the first instance of `eos` found in both `ref` and `hyp` as
        valid tokens to be computed as part of the rate. This is useful when gauging
        if a model is learning to emit the `eos` properly, but is not usually included
        in an evaluation. Only the first `eos` per transcript is included.
    """,
    "norm": """\
    norm : bool, optional
        If :obj:`True`, will normalize the distance by the number of tokens in the
        reference sequence (making the returned value a divergence)
    """,
    "batch_first": """\
    batch_first : bool, optional
        If :obj:`True`, the first two dimensions of `ref`, `hyp`, and the return value
        are transposed from those above.
    """,
    "ins_cost": """\
    ins_cost : float, optional
        The cost of an adding an extra token to a sequence in `ref`
    """,
    "del_cost": """\
    del_cost : float, optional
        The cost of removing a token from a sequence in `ref`
    """,
    "sub_cost": """\
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`
    """,
    "warn": """\
    warn : bool, optional
        Whether to display warnings on irregularities. Currently, this can happen in
        three ways:

        1. If :obj:`True` and `ins_cost`, `del_cost`, or `sub_cost` is not 1, a warning
           about a difference in computations will be raised. See the below warning for
           more info.
        2. If :obj:`True` and `norm` is :obj:`True`, will warn when a reference
           transcription has zero length
        3. If `eos` is set and `include_eos` is :obj:`True`, will warn when a transcript
           does not include an `eos` symbol
    """,
    "padding": """\
    padding : int, optional
        The value to right-pad unequal-length sequences with. Defauls to
        :obj:`pydrobert.torch.config.INDEX_PAD_VALUE`.
    """,
    "exclude_last": """\
    exclude_last : bool, optional
        If true, will exclude the final prefix, consisting of the entire transcript,
        from the return value. It will be of shape ``(max_hyp_steps, batch_size,
        max_unique_next)``
    """,
}


class _StringMatching(torch.nn.Module, metaclass=abc.ABCMeta):
    __constants__ = [
        "eos",
        "include_eos",
        "batch_first",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "warn",
    ]

    eos: Optional[int]
    include_eos: bool
    batch_first: bool
    ins_cost: float
    del_cost: float
    sub_cost: float
    warn: bool

    def __init__(
        self, eos, include_eos, batch_first, ins_cost, del_cost, sub_cost, warn
    ):
        super().__init__()
        self.eos = eos
        self.include_eos = include_eos
        self.batch_first = batch_first
        self.ins_cost = ins_cost
        self.del_cost = del_cost
        self.sub_cost = sub_cost
        self.warn = warn

    def extra_repr(self) -> str:
        return ", ".join(f"{x}={getattr(self, x)}" for x in self.__constants__)

    @abc.abstractmethod
    def forward(self, ref: torch.Tensor, hyp: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EditDistance(_StringMatching):
    __constants__ = [
        "eos",
        "include_eos",
        "norm",
        "batch_first",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "warn",
    ]

    norm: bool

    def __init__(
        self,
        eos: Optional[int] = None,
        include_eos: bool = False,
        norm: bool = False,
        batch_first: bool = False,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        sub_cost: float = 1.0,
        warn: bool = True,
    ):
        super().__init__(
            eos, include_eos, batch_first, ins_cost, del_cost, sub_cost, warn
        )
        self.norm = norm

    __doc__ = f"""Compute an edit distance over a batch of references and hypotheses

    An `Edit Distance <https://en.wikipedia.org/wiki/Edit_distance>`__ quantifies
    how dissimilar two token sequences are as the total cost of transforming a
    reference sequence into a hypothesis sequence. There are three operations that can
    be performed, each with an associated cost: adding an extra token to the reference,
    removing a token from the reference, or swapping a token in the reference with a
    token in the hypothesis.

    When instantiated, this module has the signature::

        ed = edit_distance(ref, hyp)

    {_SM_ARGS}  The return value `ed` is a tensor of shape ``(batch_size,)`` storing the
    associated edit distances.

    Parameters
    ----------
    {"".join(_SM_PARAM_DICT[c] for c in __constants__)}

    Notes
    -----
    This module returns identical values (modulo a bug fix) to :func:`error_rate` up
    to `v0.3.0` (though the default of `norm` has changed to :obj:`False`). For more
    details on the distinction between this module and the new :func:`ErrorRate`,
    please see that module's documentation.
    """

    def forward(self, ref: torch.Tensor, hyp: torch.Tensor) -> torch.Tensor:
        return edit_distance(
            ref,
            hyp,
            self.eos,
            self.include_eos,
            self.norm,
            self.batch_first,
            self.ins_cost,
            self.del_cost,
            self.sub_cost,
            self.warn,
        )


class PrefixEditDistances(_StringMatching):

    __constants__ = [
        "eos",
        "include_eos",
        "norm",
        "batch_first",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "padding",
        "exclude_last",
        "warn",
    ]

    norm: bool
    padding: int
    exclude_last: bool

    def __init__(
        self,
        eos: Optional[int] = None,
        include_eos: bool = True,
        norm: bool = False,
        batch_first: bool = False,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        sub_cost: float = 1.0,
        padding: int = config.INDEX_PAD_VALUE,
        exclude_last: bool = False,
        warn: bool = True,
    ):
        super().__init__(
            eos, include_eos, batch_first, ins_cost, del_cost, sub_cost, warn
        )
        self.norm = norm
        self.padding = padding
        self.exclude_last = exclude_last

    __doc__ = f"""Compute the edit distance between ref and each prefix of hyp

    When instantiated, this module has the signature::

        prefix_eds = prefix_edit_distances(ref, hyp)
    
    {_SM_ARGS} The return value `prefix_eds` is of shape ``(max_hyp_steps + 1,
    batch_size)`` and contains the edit distances for each prefix of each hypothesis,
    starting from the empty prefix.

    Parameters
    ----------
    {"".join(_SM_PARAM_DICT[c] for c in __constants__)}
    
    Notes
    -----
    This module returns identical values (modulo a bug fix) to
    :func:`prefix_error_rates` (and :class:`PrefixErrorRates`) up to `v0.3.0` (though
    the default of `norm` has changed to :obj:`False`). For more details on the
    distinction between this module and the new :func:`prefix_error_rates`, please
    consult the documentation of :class:`ErrorRate`.
    """

    def forward(self, ref: torch.Tensor, hyp: torch.Tensor) -> torch.Tensor:
        return prefix_edit_distances(
            ref,
            hyp,
            self.eos,
            self.include_eos,
            self.norm,
            self.batch_first,
            self.ins_cost,
            self.del_cost,
            self.sub_cost,
            self.padding,
            self.exclude_last,
            self.warn,
        )


class ErrorRate(_StringMatching):
    __constants__ = [
        "eos",
        "include_eos",
        "norm",
        "batch_first",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "warn",
    ]

    norm: bool

    def __init__(
        self,
        eos: Optional[int] = None,
        include_eos: bool = False,
        norm: bool = True,
        batch_first: bool = False,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        sub_cost: float = 1.0,
        warn: bool = True,
    ):
        super().__init__(
            eos, include_eos, batch_first, ins_cost, del_cost, sub_cost, warn
        )
        self.norm = norm

    __doc__ = f"""Calculate error rates over a batch of references and hypotheses

    An error rate is the total number of insertions, deletions, and substitutions
    between a reference (gold-standard) and hypothesis (generated) transcription,
    normalized by the number of elements in a reference. Consult the Wikipedia article
    on the `Levenshtein distance <https://en.wikipedia.org/wiki/Levenshtein_distance>`__
    for more information.

    When instantiated, this module has the signature::

        er = error_rate(ref, hyp)

    {_SM_ARGS} The return value `er` is a tensor of shape ``(batch_size,)`` storing the
    associated error rates. `er` will not have a gradient, and is thus not directly
    suited to being a loss function.

    Parameters
    ----------
    {"".join(_SM_PARAM_DICT[c] for c in __constants__)}

    Warnings
    --------
    Up to and including `v0.3.0`, :func:`error_rate` computed a normalized
    `Edit distance <https://en.wikipedia.org/wiki/Edit_distance>`__ instead of an error
    rate. The latter can be considered the total weighted cost of insertions, deletions,
    and substitutions (as per `ins_cost`, `del_cost`, and `sub_cost`), whereas the
    former is the sum of the number of mistakes. The old behaviour of returning the cost
    is now in :func:`edit_distance` and :class:`EditDistance` (though `norm` is
    :obj:`False` by default). For speech recognition evaluation, this module or
    :func:`error_rate` is the one to use. However, if you are using the default costs,
    ``ins_cost == del_cost == sub_cost == 1``, there should be no numerical difference
    between the two.
    """

    def forward(self, ref: torch.Tensor, hyp: torch.Tensor) -> torch.Tensor:
        return error_rate(
            ref,
            hyp,
            self.eos,
            self.include_eos,
            self.norm,
            self.batch_first,
            self.ins_cost,
            self.del_cost,
            self.sub_cost,
            self.warn,
        )


class PrefixErrorRates(_StringMatching):
    __constants__ = [
        "eos",
        "include_eos",
        "norm",
        "batch_first",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "padding",
        "exclude_last",
        "warn",
    ]

    norm: bool
    padding: int
    exclude_last: bool

    def __init__(
        self,
        eos: Optional[int] = None,
        include_eos: bool = True,
        norm: bool = True,
        batch_first: bool = False,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        sub_cost: float = 1.0,
        padding: int = config.INDEX_PAD_VALUE,
        exclude_last: bool = False,
        warn: bool = True,
    ):
        super().__init__(
            eos, include_eos, batch_first, ins_cost, del_cost, sub_cost, warn
        )
        self.norm = norm
        self.padding = padding
        self.exclude_last = exclude_last

    __doc__ = f"""Compute the error rate between ref and each prefix of hyp

    When instantiated, this module has the signature::

        prefix_ers = prefix_error_rates(ref, hyp)
    
    {_SM_ARGS} The return value `prefix_ers` is of shape ``(max_hyp_steps + 1,
    batch_size)`` and contains the error rates for each prefix of each hypothesis,
    starting from the empty prefix.

    Parameters
    ----------
    {"".join(_SM_PARAM_DICT[c] for c in __constants__)}

    Warnings
    --------
    The values returned by :func:`prefix_error_rates` (and thus this module) changed
    after `v0.3.0`. The old behaviour can be found in :class:`PrefixEditDistances`
    (though with `norm` defaulting to :obj:`False`). Consult the warning in
    :class:`ErrorRate` for more info.
    """

    def forward(self, ref: torch.Tensor, hyp: torch.Tensor) -> torch.Tensor:
        return prefix_error_rates(
            ref,
            hyp,
            self.eos,
            self.include_eos,
            self.norm,
            self.batch_first,
            self.ins_cost,
            self.del_cost,
            self.sub_cost,
            self.padding,
            self.exclude_last,
            self.warn,
        )


class OptimalCompletion(_StringMatching):
    __constants__ = [
        "eos",
        "include_eos",
        "batch_first",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "padding",
        "exclude_last",
        "warn",
    ]

    padding: int
    exclude_last: bool

    def __init__(
        self,
        eos: Optional[int] = None,
        include_eos: bool = True,
        batch_first: bool = False,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        sub_cost: float = 1.0,
        padding: int = config.INDEX_PAD_VALUE,
        exclude_last: bool = False,
        warn: bool = True,
    ):
        super().__init__(
            eos, include_eos, batch_first, ins_cost, del_cost, sub_cost, warn
        )
        self.padding = padding
        self.exclude_last = exclude_last

    __doc__ = f"""Return a mask of next tokens of a minimum edit distance prefix
    
    When instantiated, this module has the signature::

        optimals = optimal_completion(ref, hyp)
    
    {_SM_ARGS} The return value `optimals` is a long tensor of shape ``(max_hyp_steps +
    1, batch_size, max_unique_next)``, where ``max_unique_next <= max_ref_steps``, of
    the unique tokens that could be added to the hypothesis prefix ``hyp[:prefix_len,
    batch]`` such that some remaining suffix concatenated to the prefix would result in
    a minimal edit distance. See below for an example.

    Parameters
    ----------
    {"".join(_SM_PARAM_DICT[c] for c in __constants__)}

    Examples
    --------

    Consider the reference text "foot" and the hypothesis text "bot". The below shows
    the matrix used to calculate edit distances between them::

        \ _ f o o t
        _ 0 1 2 3 4
        b 1 1 2 3 4
        o 2 2 1 2 3
        t 3 3 2 2 2

    If ``prefix_len == 0``, then the prefix is "", and "f" (from the suffix "foot") is
    the only subsequent token that would not increase the edit distance from that of the
    prefix (0). If ``prefix_len == 1``, then the prefix is "b". To arrive at the minimum
    edit distance for "b", one either treats "b" as an insertion or a substitution for
    "f", yielding suffixes "foot" and "oot". Thus, the subsequent token could be "f" or
    "o". For the prefix "bo", the minimum edit distance is achieved by first
    substituting "f" for "b", then substituting "o" for "o", resulting in the suffix
    "ot" and the next optimal character "o". Finally, for ``prefix_len == 3`` and prefix
    "bot", there are many operations that can produce the minimum edit distance of 2,
    resulting in one of the suffixes "ot", "t", and "". The latter suffix requires no
    more tokens and so any operation would increase the edit distance. Thus the optimal
    next tokens could be "o" or "t".

    Plugging "foot" and "bot" into this function, we get the prefixes:

    >>> ref_text, hyp_text = "foot", "bot"
    >>> ref = torch.tensor([ord(c) for c in ref_text]).unsqueeze(1)
    >>> hyp = torch.tensor([ord(c) for c in hyp_text]).unsqueeze(1)
    >>> optimal = optimal_completion(ref, hyp).squeeze(1)
    >>> for prefix_len, o_for_pr in enumerate(optimal):
    ...     o_for_pr = o_for_pr.masked_select(o_for_pr.ge(0)).tolist()
    ...     print('prefix={{}}: {{}}'.format(
    ...         hyp_text[:prefix_len], ','.join([chr(i) for i in o_for_pr])))
    prefix=: f
    prefix=b: f,o
    prefix=bo: o
    prefix=bot: o,t

    See Also
    --------
    pydrobert.torch.layers.HardOptimalCompletionDistillationLoss
        A loss function that uses these optimal completions to train a model
    """

    def forward(self, ref: torch.Tensor, hyp: torch.Tensor) -> torch.Tensor:
        return optimal_completion(
            ref,
            hyp,
            self.eos,
            self.include_eos,
            self.batch_first,
            self.ins_cost,
            self.del_cost,
            self.sub_cost,
            self.padding,
            self.exclude_last,
            self.warn,
        )


@script
def hard_optimal_completion_distillation_loss(
    logits: torch.Tensor,
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int] = None,
    include_eos: bool = True,
    batch_first: bool = False,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
    sub_cost: float = 1.0,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    ignore_index: int = -2,
    warn: bool = True,
) -> torch.Tensor:
    """Functional version of HardOptimalCompletionDistillationLoss

    See Also
    --------
    HardOptimalCompletionDistillationLoss
        The :class:`torch.nn.Module` version. Describes the arguments.
    """
    if logits.dim() != 3:
        raise RuntimeError("logits must be 3 dimensional")
    if logits.shape[:-1] != hyp.shape:
        raise RuntimeError("first two dims of logits must match hyp shape")
    if include_eos:
        if eos is not None and ((eos < 0) or (eos >= logits.size(-1))):
            raise RuntimeError(f"If include_eos=True, eos ({eos}) must be a class idx")
        if eos is not None and eos == ignore_index:
            raise RuntimeError(
                f"If include_eos=True, eos cannot equal ignore_index ({eos}"
            )
    optimals = optimal_completion(
        ref,
        hyp,
        eos=eos,
        include_eos=include_eos,
        batch_first=batch_first,
        ins_cost=ins_cost,
        del_cost=del_cost,
        sub_cost=sub_cost,
        padding=ignore_index,
        exclude_last=True,
        warn=warn,
    )
    max_unique_next = optimals.size(-1)
    logits = logits.unsqueeze(2).expand(-1, -1, max_unique_next, -1)
    logits = logits.contiguous()
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, -2),
        optimals.flatten(),
        weight=weight,
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(optimals)
    padding_mask = optimals == ignore_index
    loss = loss.masked_fill(padding_mask, 0.0).sum(2)
    loss = loss / (~padding_mask).sum(2).clamp_min(1)
    if reduction == "mean":
        seq_dim = 1 if batch_first else 0
        loss = (
            loss.sum(seq_dim) / (~padding_mask).any(2).sum(seq_dim).clamp_min(1)
        ).mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction != "none":
        raise RuntimeError(f"'{reduction}' is not a valid value for reduction")
    return loss


class HardOptimalCompletionDistillationLoss(torch.nn.Module):
    r"""A categorical loss function over optimal next tokens

    Optimal Completion Distillation (OCD) [sabour2018]_ tries to minimize the train/test
    discrepancy in transcriptions by allowing seq2seq models to generate whatever
    sequences they want, then assigns a per-step loss according to whatever next token
    would set the model on a path that minimizes the edit distance in the future.

    In its "hard" version, the version used in the paper, the OCD loss function is
    simply a categorical cross-entropy loss of each hypothesis token's distribution
    versus those optimal next tokens, averaged over the number of optimal next tokens:

    .. math::

        loss(logits_t) = \frac{-\log Pr(s_t|logits_t)}{|S_t|}

    Where :math:`s_t \in S_t` are tokens from the set of optimal next tokens given
    :math:`hyp_{\leq t}` and `ref`. The loss is decoupled from an exact prefix of `ref`,
    meaning that `hyp` can be longer or shorter than `ref`.

    When called, this loss function has the signature::

        loss(logits, ref, hyp)

    `hyp` is a long tensor of shape ``(max_hyp_steps, batch_size)`` if `batch_first` is
    :obj:`False`, otherwise ``(batch_size, max_hyp_steps)`` that provides the hypothesis
    transcriptions. Likewise, `ref` of shape ``(max_ref_steps, batch_size)`` or
    ``(batch_size, max_ref_steps)`` providing reference transcriptions. `logits` is a
    3-dimensional tensor of shape ``(max_hyp_steps, batch_size, num_classes)`` if
    `batch_first` is :obj:`False`, ``(batch_size, max_hyp_steps, num_classes)``
    otherwise. A softmax over the step dimension defines the per-step distribution over
    class labels.

    Parameters
    ----------
    eos : int or None, optional
        A special token in `ref` and `hyp` whose first occurrence in each
        batch indicates the end of a transcript
    include_eos : bool, optional
        Whether to include the first instance of `eos` found in both `ref` and
        `hyp` as valid tokens to be computed as part of the distance. `eos`
        must be a valid class index if `include_eos` is :obj:`True`
    batch_first : bool, optional
        Whether the batch dimension comes first, or the step dimension
    ins_cost : float, optional
        The cost of an adding a superfluous token to a transcript in `hyp`
    del_cost : float, optional
        The cost of missing a token from `ref`
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`
    weight : torch.Tensor or None, optional
        A float tensor of manual rescaling weight given to each class
    reduction : {'mean', 'none', 'sum'}, optional
        Specifies the reduction to be applied to the output. 'none': no
        reduction will be applied. 'sum': the output will be summed. 'mean':
        the output will be averaged.
    ignore_index : int, optional
        Specify a target value that is ignored and does not contribute to the input
        gradient. Should not be set to `eos` when `include_eos` is :obj:`True`.

    See Also
    --------
    pydrobert.torch.util.optimal_completion
        Used to determine the optimal next token set :math:`S`
    pydrobert.torch.util.random_walk_advance
        For producing a random `hyp` based on `logits` if the underlying
        model producing `logits` is auto-regressive. Also provides an example
        of sampling non-auto-regressive models
    """

    __constants__ = [
        "eos",
        "include_eos",
        "batch_first",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "reduction",
        "ignore_index",
    ]

    eos: Optional[int]
    include_eos: bool
    batch_first: bool
    ins_cost: float
    del_cost: float
    sub_cost: float
    reduction: str
    ignore_index: int

    def __init__(
        self,
        eos: Optional[int] = None,
        include_eos: bool = True,
        batch_first: bool = False,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        sub_cost: float = 1.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.eos = eos
        self.include_eos = include_eos
        self.batch_first = batch_first
        self.ins_cost = ins_cost
        self.del_cost = del_cost
        self.sub_cost = sub_cost
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight)

    def forward(
        self,
        logits: torch.Tensor,
        ref: torch.Tensor,
        hyp: torch.Tensor,
        warn: bool = True,
    ) -> torch.Tensor:
        return hard_optimal_completion_distillation_loss(
            logits,
            ref,
            hyp,
            self.eos,
            self.include_eos,
            self.batch_first,
            self.ins_cost,
            self.del_cost,
            self.sub_cost,
            self.weight,
            self.reduction,
            self.ignore_index,
            warn,
        )


@script
def minimum_error_rate_loss(
    log_probs: torch.Tensor,
    ref: torch.Tensor,
    hyp: torch.Tensor,
    eos: Optional[int] = None,
    include_eos: bool = True,
    sub_avg: bool = True,
    batch_first: bool = False,
    norm: bool = True,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
    sub_cost: float = 1.0,
    reduction: str = "mean",
    warn: bool = True,
) -> torch.Tensor:
    """Functional version of MinimumErrorRateLoss

    See Also
    --------
    MinimumErrorRateLoss
        The :class:`torch.nn.Module` version. Describes the arguments
    """
    if log_probs.dim() != 2:
        raise RuntimeError("log_probs must be 2 dimensional")
    if hyp.dim() != 3:
        raise RuntimeError("hyp must be 3 dimensional")
    if ref.dim() not in (2, 3):
        raise RuntimeError("ref must be 2 or 3 dimensional")
    if batch_first:
        batch_size, samples, max_hyp_steps = hyp.shape
        if ref.dim() == 2:
            ref = ref.unsqueeze(1).repeat(1, samples, 1)
        if (ref.shape[:2] != (batch_size, samples)) or (
            ref.shape[:2] != log_probs.shape
        ):
            raise RuntimeError(
                "ref and hyp batch_size and sample dimensions must match"
            )
        max_ref_steps = ref.size(-1)
        ref = ref.view(-1, max_ref_steps)
        hyp = hyp.view(-1, max_hyp_steps)
    else:
        max_hyp_steps, batch_size, samples = hyp.shape
        if ref.dim() == 2:
            ref = ref.unsqueeze(-1).repeat(1, 1, samples)
        if (ref.shape[1:] != (batch_size, samples)) or (
            ref.shape[1:] != log_probs.shape
        ):
            raise RuntimeError(
                "ref and hyp batch_size and sample dimensions must match"
            )
        max_ref_steps = ref.size(0)
        ref = ref.view(max_ref_steps, -1)
        hyp = hyp.view(max_hyp_steps, -1)
    if samples < 2:
        raise RuntimeError(f"Batch must have at least two samples, got {samples}")
    er = error_rate(
        ref,
        hyp,
        eos=eos,
        include_eos=include_eos,
        norm=norm,
        batch_first=batch_first,
        ins_cost=ins_cost,
        del_cost=del_cost,
        sub_cost=sub_cost,
        warn=warn,
    ).view(batch_size, samples)
    if sub_avg:
        er = er - er.mean(1, keepdim=True)
    loss = er * torch.nn.functional.softmax(log_probs, 1)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction != "none":
        raise RuntimeError(f"'{reduction}' is not a valid value for reduction")
    return loss


class MinimumErrorRateLoss(torch.nn.Module):
    r"""Error rate expectation normalized over some number of transcripts

    Proposed in [prabhavalkar2018]_ though similar ideas had been explored
    previously. Given a subset of all possible token sequences and their
    associated probability mass over that population, this loss calculates the
    probability mass normalized over the subset, then calculates the
    expected error rate over that normalized distribution. That is, given some
    sequences :math:`s \in S \subseteq P`, the loss for a given reference
    transcription :math:`s^*` is

    .. math::

        \mathcal{L}(s, s^*) = \frac{Pr(s) ER(s, s^*)}{\sum_{s'} Pr(s')}

    This is an exact expectation over :math:`S` but not over :math:`P`. The
    larger the mass covered by :math:`S`, the closer the expectation is to the
    population - especially so for an n-best list (though it would be biased).

    This loss function has the following signature::

        loss(log_probs, ref, hyp, warn=True)

    `log_probs` is a tensor of shape ``(batch_size, samples)`` providing the log joint
    probabilities of every path. `hyp` is a long tensor of shape ``(max_hyp_steps,
    batch_size, samples)`` if `batch_first` is :obj:`False` otherwise ``(batch_size,
    samples, max_hyp_steps)`` that provides the hypothesis transcriptions. `ref` is a 2-
    or 3-dimensional tensor. If 2D, it is of shape ``(max_ref_steps, batch_size)`` (or
    ``(batch_size, max_ref_steps)``). Alternatively, `ref` can be of shape
    ``(max_ref_steps, batch_size, samples)`` or ``(batch_size, samples,
    max_ref_steps)``.

    If `ref` is 2D, the loss is calculated as

    .. math::

        loss_{MER} = SoftMax(log\_probs)[ER(hyp_i, ref) - \mu_i]

    where :math:`\mu_i` is the average error rate along paths in the batch element
    :math:`i`. :math:`mu_i` can be removed by setting `sub_avg` to :obj:`False`. Note
    that each hypothesis is compared against the same reference as long as the batch
    element remains the same

    If `ref` is 3D, the loss is calculated as

    .. math::

        loss_{MER} = SoftMax(log\_probs)[ER(hyp_i, ref_i) - \mu_i]

    In this version, each hypothesis is compared against a unique reference

    Parameters
    ----------
    eos : int, optional
        A special token in `ref` and `hyp` whose first occurrence in each
        batch indicates the end of a transcript
    include_eos : bool, optional
        Whether to include the first instance of `eos` found in both `ref` and
        `hyp` as valid tokens to be computed as part of the distance.
    sub_avg : bool, optional
        Whether to subtract the average error rate from each pathwise error
        rate
    batch_first : bool, optional
        Whether batch/path dimensions come first, or the step dimension
    norm : bool, optional
        If :obj:`False`, will use edit distances instead of error rates
    ins_cost : float, optional
        The cost of an adding a superfluous token to a transcript in `hyp`
    del_cost : float, optional
        The cost of missing a token from `ref`
    sub_cost : float, optional
        The cost of swapping a token from `ref` with one from `hyp`
    reduction : {'mean', 'none', 'sum'}, optional
        Specifies the reduction to be applied to the output. 'none': no
        reduction will be applied. 'sum': the output will be summed. 'mean':
        the output will be averaged.

    Attributes
    ----------
    eos, ignore_index : int
    include_eos, sub_avg, batch_first, norm : bool
    ins_cost, del_cost, sub_cost : float
    reduction : {'mean', 'none', 'sum'}

    Notes
    -----

    A previous version of this module incorporated a Maximum Likelihood Estimate (MLE)
    into the loss as in [prabhavalkar2018]_, which required `logits` instead of
    `log_probs`. This was overly complicated, given the user can easily incorporate the
    additional loss term herself by using :class:`torch.nn.CrossEntropyLoss`. Take a
    look at the example below for how to recreate this

    Examples
    --------

    Assume here that `logits` is the output of some neural network, and that `hyp` has
    somehow been produced from that (e.g. a beam search or random walk). We combine this
    loss function with a cross-entropy/MLE term to sort-of recreate [prabhavalkar2018]_.

    >>> from pydrobert.torch.util import sequence_log_probs
    >>> steps, batch_size, num_classes, eos, padding = 30, 20, 10, 0, -1
    >>> samples, lmb = 10, .01
    >>> logits = torch.randn(
    ...     steps, samples, batch_size, num_classes, requires_grad=True)
    >>> hyp = torch.randint(num_classes, (steps, samples, batch_size))
    >>> ref_lens = torch.randint(1, steps + 1, (batch_size,))
    >>> ref_lens[0] = steps
    >>> ref = torch.nn.utils.rnn.pad_sequence(
    ...     [torch.randint(1, num_classes, (x,)) for x in ref_lens],
    ...     padding_value=padding,
    ... )
    >>> ref[ref_lens - 1, range(batch_size)] = eos
    >>> ref = ref.unsqueeze(1).repeat(1, samples, 1)
    >>> mer = MinimumErrorRateLoss(eos=eos)
    >>> mle = torch.nn.CrossEntropyLoss(ignore_index=padding)
    >>> log_probs = sequence_log_probs(logits, hyp, eos=eos)
    >>> l = mer(log_probs, ref, hyp)
    >>> l = l + lmb * mle(logits.view(-1, num_classes), ref.flatten())
    >>> l.backward()

    See Also
    --------
    pydrobert.torch.util.beam_search_advance
        For getting an n-best list into `hyp` and some `log_probs`.
    pydrobert.torch.util.random_walk_advance
        For getting a random sample into `hyp`
    pydrobert.torch.util.sequence_log_probs
        For converting token log probs (or logits) to sequence log probs
    """

    __constants__ = [
        "eos",
        "include_eos",
        "sub_avg",
        "batch_first",
        "norm",
        "ins_cost",
        "del_cost",
        "sub_cost",
        "reduction",
    ]

    eos: Optional[int]
    include_eos: bool
    sub_avg: bool
    norm: bool
    ins_cost: float
    del_cost: float
    sub_cost: float
    reduction: str

    def __init__(
        self,
        eos: Optional[int] = None,
        include_eos: bool = True,
        sub_avg: bool = True,
        batch_first: bool = False,
        norm: bool = True,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        sub_cost: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.eos = eos
        self.include_eos = include_eos
        self.sub_avg = sub_avg
        self.batch_first = batch_first
        self.norm = norm
        self.ins_cost = ins_cost
        self.del_cost = del_cost
        self.sub_cost = sub_cost
        self.reduction = reduction

    def forward(
        self,
        log_probs: torch.Tensor,
        ref: torch.Tensor,
        hyp: torch.Tensor,
        warn: bool = True,
    ) -> torch.Tensor:
        return minimum_error_rate_loss(
            log_probs,
            ref,
            hyp,
            self.eos,
            self.include_eos,
            self.sub_avg,
            self.batch_first,
            self.norm,
            self.ins_cost,
            self.del_cost,
            self.sub_cost,
            self.reduction,
            warn,
        )
