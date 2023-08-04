# Copyright 2023 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Boilerplate for checking argument values

These functions are intended for use primarily in :class:`torch.Module` definitions
and are not :mod:`torch.jit` safe.
"""

import os
import string

from typing import (
    Collection,
    Optional,
    TypeVar,
    Callable,
    Any,
    Type,
    Union,
)
from typing_extensions import overload, Literal, get_args
from pathlib import Path

import torch

V = TypeVar("V")
NumLike = Union[torch.Tensor, float, int]
N = TypeVar("N", *get_args(NumLike))
StrPathLike = TypeVar("StrPathLike", str, os.PathLike, Path)


def _nv(name: Optional[str], val: Any) -> str:
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return f"{val.item()}" if name is None else f"{name} ({val.item()})"
        else:
            return name if name is not None else "tensor"
    else:
        if isinstance(val, str):
            val = f"'{val}'"
        return f"{val}" if name is None else f"{name} ({val})"


def _type_check_factory(t: Type[V], *ts: type):
    ts = (t,) + ts

    @overload
    def _is_check(
        val: t, name: Optional[str] = None, allow_none: Literal[False] = False
    ) -> t:
        ...

    @overload
    def _is_check(
        val: Optional[t], name: Optional[str] = None, allow_none: Literal[True] = False
    ) -> Optional[t]:
        ...

    def _is_check(val, name=None, allow_none=False):
        if allow_none and val is None:
            return None
        if isinstance(val, _is_check.ts):
            return val if isinstance(val, _is_check.t) else _is_check.t(val)
        else:
            tname = _is_check.t.__name__
            x = "n" if tname.startswith(("a", "e", "i", "o", "u")) else ""
            y = " or None" if allow_none else ""
            raise ValueError(f"{_nv(name, val)} is not a{x} {tname}{y}")

    _is_check.t, _is_check.ts = t, ts

    return _is_check


is_str = _type_check_factory(str)
is_int = _type_check_factory(int)
is_bool = _type_check_factory(bool, int, float)
is_float = _type_check_factory(float, int)
is_tensor = _type_check_factory(torch.Tensor)
is_pathlike = _type_check_factory(os.PathLike)
is_path = _type_check_factory(Path)


@overload
def is_numlike(
    val: N, name: Optional[str] = None, allow_none: Literal[False] = False
) -> N:
    ...


@overload
def is_numlike(
    val: Optional[N], name: Optional[str] = None, allow_none: Literal[True] = False,
) -> Optional[N]:
    ...


def is_numlike(val, name=None, allow_none=False):
    if allow_none and val is None:
        return None
    if not isinstance(val, get_args(NumLike)):
        raise ValueError(f"{_nv(name, val)} is not num-like {get_args(N)}")
    return val


def is_token(
    val: str,
    name: Optional[str] = None,
    empty_okay: bool = False,
    whitespace: str = string.whitespace,
) -> str:
    val = is_str(val, name)
    if not empty_okay and not len(val):
        raise ValueError(f"{_nv(name, val)} is empty")
    else:
        for w in whitespace:
            if w in val:
                raise ValueError(f"{_nv(name, val)} contains '{w}'")
    return val


@overload
def is_a(
    val: V, t: Type[V], name: Optional[str] = None, allow_none: Literal[False] = False
) -> V:
    ...


@overload
def is_a(
    val: Optional[V],
    t: Type[V],
    name: Optional[str] = None,
    allow_none: Literal[True] = False,
) -> Optional[V]:
    ...


def is_a(val, t, name=None, allow_none=False):
    assert not issubclass(t, torch.nn.Module), f"what if {t} is scripted?"
    if allow_none and val is None:
        return None
    if not isinstance(val, t):
        suf = "n" if t.__name__.startswith(("a", "e", "i", "o", "u")) else ""
        raise ValueError(f"{_nv(name, val)} is not a{suf} {t.__name__}")
    return val


@overload
def is_in(
    val: V,
    collection: Collection[V],
    name: Optional[str] = None,
    allow_none: Literal[False] = False,
) -> V:
    ...


@overload
def is_in(
    val: Optional[V],
    collection: Collection[V],
    name: Optional[str] = None,
    allow_none: Literal[True] = False,
) -> Optional[V]:
    ...


def is_in(val, collection, name=None, allow_none=False):
    if allow_none and val is None:
        return None
    if val not in collection:
        raise ValueError(f"{_nv(name, val)} is not one of {collection}")
    return val


def is_exactly(
    val: V, other: V, name: Optional[str] = None, other_name: Optional[str] = None
) -> V:
    if val is not other:
        raise ValueError(f"{_nv(name, val)} is not {_nv(other_name, other)}")
    return val


def is_pos(val: N, name: Optional[str] = None) -> N:
    val_ = torch.as_tensor(is_numlike(val, name))
    if (val_ <= 0).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(f"{_nv(name, val)} is not {x}positive")
    return val


def is_neg(val: N, name: Optional[str] = None) -> N:
    val_ = torch.as_tensor(is_numlike(val, name))
    if (val_ >= 0).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(f"{_nv(name, val)} is not {x}negative")
    return val


def is_nonpos(val: N, name: Optional[str] = None) -> N:
    val_ = torch.as_tensor(is_numlike(val, name))
    if (val_ > 0).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(f"{_nv(name, val)} is not {x}non-positive")
    return val


def is_nonneg(val: N, name: Optional[str] = None) -> N:
    val_ = torch.as_tensor(is_numlike(val, name))
    if (val_ < 0).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(f"{_nv(name, val)} is not {x}non-negative")
    return val


def is_equal(
    val: N, other: NumLike, name: Optional[str] = None, other_name: Optional[str] = None
) -> V:
    val_ = torch.as_tensor(is_numlike(val, name))
    other_ = torch.as_tensor(is_numlike(other, other_name), device=val_.device)
    if (val_ != other_).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(
            f"{_nv(name, val)} is not {x}equal to {_nv(other_name, other)}"
        )
    return val


def is_lt(
    val: N,
    other: Union[float, int, torch.Tensor],
    name: Optional[str] = None,
    other_name: Optional[str] = None,
) -> N:
    val_ = torch.as_tensor(is_numlike(val, name))
    other_ = torch.as_tensor(is_numlike(other, other_name), device=val_.device)
    if (val_ >= other_).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(
            f"{_nv(name, val)} is not {x}less than {_nv(other_name, other)}"
        )
    return val


def is_gt(
    val: N,
    other: Union[float, int, torch.Tensor],
    name: Optional[str] = None,
    other_name: Optional[str] = None,
) -> N:
    val_ = torch.as_tensor(is_numlike(val, name))
    other_ = torch.as_tensor(is_numlike(other, other_name), device=val_.device)
    if (val_ <= other_).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(
            f"{_nv(name, val)} is not {x}greater than {_nv(other_name, other)}"
        )
    return val


def is_lte(
    val: N,
    other: Union[float, int, torch.Tensor],
    name: Optional[str] = None,
    other_name: Optional[str] = None,
) -> N:
    val_ = torch.as_tensor(is_numlike(val, name))
    other_ = torch.as_tensor(is_numlike(other, other_name), device=val_.device)
    if (val_ > other_).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(
            f"{_nv(name, val)} is not {x}less than or equal to "
            f"{_nv(other_name, other)}"
        )
    return val


def is_gte(
    val: N,
    other: Union[float, int, torch.Tensor],
    name: Optional[str] = None,
    other_name: Optional[str] = None,
) -> N:
    val_ = torch.as_tensor(is_numlike(val, name))
    other_ = torch.as_tensor(is_numlike(other, other_name), device=val_.device)
    if (val_ < other_).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(
            f"{_nv(name, val)} is not {x}greater than or equal to"
            f"{_nv(other_name, other)}"
        )
    return val


def is_btw(
    val: N,
    left: NumLike,
    right: NumLike,
    name: Optional[str] = None,
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
    left_inclusive: bool = False,
    right_inclusive: bool = False,
) -> N:
    val_ = torch.as_tensor(is_numlike(val, name))
    try:
        if left_inclusive:
            val_ = is_gte(val_, left, name, left_name)
        else:
            val_ = is_gt(val_, left, name, left_name)
        if right_inclusive:
            val_ = is_lte(val_, right, name, right_name)
        else:
            val_ = is_lt(val_, right, name, right_name)
    except ValueError:
        x = "entirely " if val_.numel() > 1 else ""
        y = "incl." if left_inclusive else "excl."
        z = "incl." if right_inclusive else "excl."
        raise ValueError(
            f"{_nv(name, val)} is not {x}within {_nv(left_name, left)} {y} and "
            f"{_nv(right_name, right)} {z}"
        )
    return val


def is_btw_open(
    val: N,
    left: NumLike,
    right: NumLike,
    name: Optional[str] = None,
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
) -> N:
    return is_btw(val, left, right, name, left_name, right_name, False, False)


def is_btw_closed(
    val: N,
    left: NumLike,
    right: NumLike,
    name: Optional[str] = None,
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
) -> N:
    return is_btw(val, left, right, name, left_name, right_name, True, True)


def is_open01(val: N, name: Optional[str] = None) -> N:
    return is_btw(val, 0, 1, name, None, None, False, False)


def is_closed01(val: N, name: Optional[str] = None) -> N:
    return is_btw(val, 0, 1, name, None, None, True, True)


def _numlike_special_factory(t: Type[N], *ts: Type[N]):

    _type_check = _type_check_factory(t, *ts)

    def _pos_check(val: t, name: Optional[str] = None) -> t:
        val = _pos_check.type_check(val, name)
        return is_pos(val, name)

    _pos_check.type_check = _type_check

    def _neg_check(val: t, name: Optional[str] = None) -> t:
        val = _neg_check.type_check(val, name)
        return is_neg(val, name)

    _neg_check.type_check = _type_check

    def _nonpos_check(val: t, name: Optional[str] = None) -> t:
        val = _nonpos_check.type_check(val, name)
        return is_nonpos(val, name)

    _nonpos_check.type_check = _type_check

    def _nonneg_check(val: t, name: Optional[str] = None) -> t:
        val = _nonneg_check.type_check(val, name)
        return is_nonneg(val, name)

    _nonneg_check.type_check = _type_check

    def _equal_check(
        val: t,
        other: NumLike,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
    ) -> t:
        val = _equal_check.type_check(val, name)
        return is_equal(val, other, name, other_name)

    _equal_check.type_check = _type_check

    def _lt_check(
        val: t,
        other: NumLike,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
    ) -> t:
        val = _lt_check.type_check(val, name)
        return is_lt(val, other, name, other_name)

    _lt_check.type_check = _type_check

    def _lte_check(
        val: t,
        other: NumLike,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
    ) -> t:
        val = _lte_check.type_check(val, name)
        return is_lte(val, other, name, other_name)

    _lte_check.type_check = _type_check

    def _gt_check(
        val: t,
        other: NumLike,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
    ) -> t:
        val = _gt_check.type_check(val, name)
        return is_gt(val, other, name, other_name)

    _gt_check.type_check = _type_check

    def _gte_check(
        val: t,
        other: NumLike,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
    ) -> t:
        val = _gte_check.type_check(val, name)
        return is_gte(val, other, name, other_name)

    _gte_check.type_check = _type_check

    def _btw_check(
        val: t,
        left: NumLike,
        right: NumLike,
        name: Optional[str] = None,
        left_name: Optional[str] = None,
        right_name: Optional[str] = None,
        left_inclusive: bool = False,
        right_inclusive: bool = False,
    ) -> t:
        val = _btw_check.type_check(val, name)
        return is_btw(
            val,
            left,
            right,
            name,
            left_name,
            right_name,
            left_inclusive,
            right_inclusive,
        )

    _btw_check.type_check = _type_check

    def _btw_open_check(
        val: t,
        left: NumLike,
        right: NumLike,
        name: Optional[str] = None,
        left_name: Optional[str] = None,
        right_name: Optional[str] = None,
    ) -> t:
        val = _btw_open_check.type_check(val, name)
        return is_btw_open(val, left, right, name, left_name, right_name, False, False)

    _btw_open_check.type_check = _type_check

    def _btw_closed_check(
        val: t,
        left: NumLike,
        right: NumLike,
        name: Optional[str] = None,
        left_name: Optional[str] = None,
        right_name: Optional[str] = None,
    ) -> t:
        val = _btw_closed_check.type_check(val, name)
        return is_btw_closed(val, left, right, name, left_name, right_name, True, True)

    _btw_closed_check.type_check = _type_check

    def _open01_check(val: t, name: Optional[str] = None) -> t:
        val = _open01_check.type_check(val, name)
        return is_open01(val, name)

    _open01_check.type_check = _type_check

    def _closed01_check(val: t, name: Optional[str] = None) -> t:
        val = _closed01_check.type_check(val, name)
        return is_nonneg(val, name)

    _closed01_check.type_check = _type_check

    return (
        _pos_check,
        _neg_check,
        _nonpos_check,
        _nonneg_check,
        _equal_check,
        _lt_check,
        _lte_check,
        _gt_check,
        _gte_check,
        _btw_check,
        _btw_open_check,
        _btw_closed_check,
        _open01_check,
        _closed01_check,
    )


(
    is_posi,
    is_negi,
    is_nonposi,
    is_nonnegi,
    is_equali,
    is_lti,
    is_ltei,
    is_gti,
    is_gtei,
    is_btwi,
    is_btw_openi,
    is_btw_closedi,
    is_open01i,
    is_closed01i,
) = _numlike_special_factory(int)
is_nat = is_posi
(
    is_posf,
    is_negf,
    is_nonposf,
    is_nonnegf,
    is_equalf,
    is_ltf,
    is_ltef,
    is_gtf,
    is_gtef,
    is_btwf,
    is_btw_openf,
    is_btw_closedf,
    is_open01f,
    is_closed01f,
) = _numlike_special_factory(float, int)
(
    is_post,
    is_negt,
    is_nonpost,
    is_nonnegt,
    is_equalt,
    is_ltt,
    is_ltet,
    is_gtt,
    is_gtet,
    is_btwt,
    is_btw_opent,
    is_btw_closedt,
    is_open01t,
    is_closed01t,
) = _numlike_special_factory(torch.Tensor)


def is_file(val: StrPathLike, name: Optional[str] = None) -> StrPathLike:
    if not os.path.isfile(val):
        raise ValueError(f"{_nv(name, val)} is not a file")
    return val


def is_dir(val: StrPathLike, name: Optional[str] = None) -> StrPathLike:
    if not os.path.isdir(val):
        raise ValueError(f"{_nv(name, val)} is not a directory")
    return val


def has_ndim(val: torch.Tensor, ndim: int, name: Optional[str] = None) -> torch.Tensor:
    if val.ndim != ndim:
        raise ValueError(
            f"Expected {_nv(name, val)} to have dimension {ndim}; got {val.ndim}"
        )
    return val


def is_nonempty(val: torch.Tensor, name: Optional[str] = None) -> torch.Tensor:
    if not val.numel():
        raise ValueError(f"Expected {_nv(name, val)} to be nonempty")
    return val


def _cast_factory(
    cast: Callable[[Any], V],
    check: Optional[Callable[[V, Optional[str]], V]] = None,
    cast_name: Optional[str] = None,
):
    def _cast(val: Any, name: Optional[str] = None) -> V:
        try:
            val = cast(val)
        except:
            suf = "n" if _cast.__name__.startswith(("a", "e", "i", "o", "u")) else ""
            raise TypeError(
                f"Could not cast {_nv(name, val)} as a{suf} {_cast.__name__}"
            )
        if _cast.check is not None:
            val = _cast.check(val, name)
        return val

    _cast.check = check
    _cast.__name__ = cast.__name__ if cast_name is None else cast_name

    return _cast


as_str = _cast_factory(str)
as_int = _cast_factory(int)
as_bool = _cast_factory(bool)
as_float = _cast_factory(float)
as_posf = _cast_factory(float, is_pos, cast_name="positive float")
as_nat = _cast_factory(int, is_pos, cast_name="natural number")
as_posi = _cast_factory(int, is_pos, cast_name="positive integer")
as_nonnegf = _cast_factory(float, is_nonneg, cast_name="non-negative float")
as_nonnegi = _cast_factory(int, is_nonneg, cast_name="non-negative integer")
as_negf = _cast_factory(float, is_neg, cast_name="negative float")
as_negi = _cast_factory(int, is_neg, cast_name="negative integer")
as_nonposf = _cast_factory(float, is_nonpos, cast_name="non-positive float")
as_nonposi = _cast_factory(int, is_nonpos, cast_name="non-positive integer")
as_ltf = _cast_factory(float, is_lt)
as_lti = _cast_factory(int, is_lt)
as_gtf = _cast_factory(float, is_gt)
as_gti = _cast_factory(int, is_gt)
as_open01 = _cast_factory(float, is_open01, cast_name="float within (0, 1)")
as_closed01 = _cast_factory(float, is_closed01, cast_name="float within [0, 1]")
as_path = _cast_factory(Path)
as_path_file = _cast_factory(Path, is_file, cast_name="readable file")
as_path_dir = _cast_factory(Path, is_dir, cast_name="readable directory")
as_file = _cast_factory(str, is_file, cast_name="readable file")
as_dir = _cast_factory(str, is_dir, cast_name="readable directory")
