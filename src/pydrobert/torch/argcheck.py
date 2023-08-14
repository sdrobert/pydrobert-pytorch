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

There are two broad types of function defined in this submodule: ``is_*`` and ``as_*``.

The ``is_*`` functions check if the passed value satisfies some requirement, usually
being of some type. They are intended to check arguments being passed to objects
(primarily :class:`torch.nn.Module` objects) on initialization. Some accept other types
(e.g. :func:`is_float` accepts :class:`int`, :class:`np.integer`, and
:class:`np.floating` in addition to :class:`float`) which will quietly be cast to the
expected type before returning. Most other ``is_*`` functions check whether the value
satisfies some condition (e.g. being a member of a collection, :func:`is_in`, or being
positive, :func:`is_pos`). Some, e.g. :func:`is_nat`, combine type checks with
conditions (:func:`is_int` and :func:`is_pos`).

The ``as_*`` functions are more agressive, casting their first argument to the type
immediately, then possibly checking a condition. They are intended primarly for use
as the ``type`` argument in :func:`argparser.ArgumentParser.add_argument`.
"""

import os
import string

from typing import (
    Collection,
    Concatenate,
    Optional,
    TypeVar,
    Callable,
    Any,
    Type,
    Union,
    Protocol,
    Generic,
    cast,
)
from typing_extensions import overload, Literal, get_args, ParamSpec
from pathlib import Path

import torch
import numpy as np

V1 = TypeVar("V1")
V2 = TypeVar("V2")
NumLike = Union[torch.Tensor, float, int, np.floating, np.integer]
N = TypeVar("N", bound=NumLike)
P = ParamSpec("P")
StrOrPathLike = Union[str, os.PathLike]


class _IsCheck(Protocol[V1]):

    # if allow_none is the literal "False" (also the default), we can narrow to the type
    # of interest. If allow_none is true or is variable, we have to assume it's optional
    @overload
    def __call__(
        self, val: V1, name: Optional[str] = None, allow_none: Literal[False] = False
    ) -> V1:
        ...

    @overload
    def __call__(
        self, val: Optional[V1], name: Optional[str] = None, allow_none: bool = False
    ) -> Optional[V1]:
        ...


def _is_check_allow_none(wrapped: Callable[..., V1]) -> _IsCheck[V1]:
    def wrapper(val, name=None, allow_none=False):
        if allow_none and val is None:
            return val
        return wrapped(val, name)

    return wrapper


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


def _type_check_factory(t: Type[V1], *ts: type):
    ts = (t,) + ts

    @_is_check_allow_none
    def _is_check(val, name=None) -> t:
        if isinstance(val, ts):
            return val if (type(val) is t) else t(val)
        else:
            tname = t.__name__
            x = "n" if tname.startswith(("a", "e", "i", "o", "u")) else ""
            raise ValueError(f"{_nv(name, val)} is not a{x} {tname}")

    return _is_check


is_str = _type_check_factory(str)
is_int = _type_check_factory(int, np.integer)
is_bool = _type_check_factory(bool)
is_float = _type_check_factory(float, int, np.integer, np.floating)
is_tensor = _type_check_factory(torch.Tensor)
is_path = _type_check_factory(Path, *get_args(StrOrPathLike))


@_is_check_allow_none
def is_numlike(val, name=None) -> NumLike:
    if not isinstance(val, get_args(NumLike)):
        raise ValueError(f"{_nv(name, val)} is not num-like {get_args(N)}")
    return val


@overload
def is_token(
    val: str,
    name: Optional[str] = None,
    empty_okay: bool = False,
    whitespace: str = string.whitespace,
    allow_none: Literal[False] = False,
) -> str:
    ...


@overload
def is_token(
    val: Optional[str],
    name: Optional[str] = None,
    empty_okay: bool = False,
    whitespace: str = string.whitespace,
    allow_none: bool = False,
) -> Optional[str]:
    ...


def is_token(
    val, name=None, empty_okay=False, whitespace=string.whitespace, allow_none=False
):
    if val is None and allow_none:
        return val
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
    val: V1,
    t: Type[V1],
    name: Optional[str],
    allow_none: Literal[False] = False,
) -> V1:
    ...


@overload
def is_a(
    val: Optional[V1],
    t: Type[V1],
    name: Optional[str] = None,
    allow_none: bool = False,
) -> Optional[V1]:
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
    val: V1,
    collection: Collection[V1],
    name: Optional[str] = None,
    allow_none: Literal[False] = False,
) -> V1:
    ...


@overload
def is_in(
    val: Optional[V1],
    collection: Collection[V1],
    name: Optional[str] = None,
    allow_none: bool = False,
) -> Optional[V1]:
    ...


def is_in(val, collection, name=None, allow_none=False):
    if allow_none and val is None:
        return None
    if val not in collection:
        raise ValueError(f"{_nv(name, val)} is not one of {collection}")
    return val


@_is_check_allow_none
def is_pos(val, name=None) -> NumLike:
    val_ = torch.as_tensor(is_numlike(val, name))
    if (val_ <= 0).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(f"{_nv(name, val)} is not {x}positive")
    return val


@_is_check_allow_none
def is_neg(val, name=None) -> NumLike:
    val_ = torch.as_tensor(is_numlike(val, name))
    if (val_ >= 0).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(f"{_nv(name, val)} is not {x}negative")
    return val


@_is_check_allow_none
def is_nonpos(val, name=None) -> NumLike:
    val_ = torch.as_tensor(is_numlike(val, name))
    if (val_ > 0).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(f"{_nv(name, val)} is not {x}non-positive")
    return val


@_is_check_allow_none
def is_nonneg(val: N, name=None) -> NumLike:
    val_ = torch.as_tensor(is_numlike(val, name))
    if (val_ < 0).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(f"{_nv(name, val)} is not {x}non-negative")
    return val


class _CompareProtocol(Protocol, Generic[V1, V2]):
    @overload
    def __call__(
        self,
        val: V1,
        other: V2,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
        allow_none: Literal[False] = False,
    ) -> V1:
        ...

    @overload
    def __call__(
        self,
        val: Optional[V1],
        other: V2,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
        allow_none: bool = False,
    ) -> Optional[V1]:
        ...


def _compare_allow_none(
    func: Callable[Concatenate[V1, V2, P], V1]
) -> _CompareProtocol[V1, V2]:
    def _allow_none(val, other, name=None, other_name=None, allow_none=False):
        if val is None and allow_none:
            return val
        return func(val, other, name, other_name)

    return _allow_none


@_compare_allow_none
def is_exactly(val: Any, other: Any, name=None, other_name=None) -> Any:
    if val is not other:
        raise ValueError(f"{_nv(name, val)} is not {_nv(other_name, other)}")
    return val


@_compare_allow_none
def is_equal(val: NumLike, other: NumLike, name=None, other_name=None) -> NumLike:
    val_ = torch.as_tensor(is_numlike(val, name))
    other_ = torch.as_tensor(is_numlike(other, other_name), device=val_.device)
    if (val_ != other_).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(
            f"{_nv(name, val)} is not {x}equal to {_nv(other_name, other)}"
        )
    return val


@_compare_allow_none
def is_lt(
    val: NumLike,
    other: NumLike,
    name=None,
    other_name=None,
) -> NumLike:
    val_ = torch.as_tensor(is_numlike(val, name))
    other_ = torch.as_tensor(is_numlike(other, other_name), device=val_.device)
    if (val_ >= other_).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(
            f"{_nv(name, val)} is not {x}less than {_nv(other_name, other)}"
        )
    return val


@_compare_allow_none
def is_gt(
    val: NumLike,
    other: NumLike,
    name: Optional[str] = None,
    other_name: Optional[str] = None,
) -> NumLike:
    val_ = torch.as_tensor(is_numlike(val, name))
    other_ = torch.as_tensor(is_numlike(other, other_name), device=val_.device)
    if (val_ <= other_).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(
            f"{_nv(name, val)} is not {x}greater than {_nv(other_name, other)}"
        )
    return val


@_compare_allow_none
def is_lte(
    val: NumLike,
    other: NumLike,
    name: Optional[str] = None,
    other_name: Optional[str] = None,
) -> NumLike:
    val_ = torch.as_tensor(is_numlike(val, name))
    other_ = torch.as_tensor(is_numlike(other, other_name), device=val_.device)
    if (val_ > other_).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(
            f"{_nv(name, val)} is not {x}less than or equal to "
            f"{_nv(other_name, other)}"
        )
    return val


@_compare_allow_none
def is_gte(
    val: NumLike,
    other: NumLike,
    name: Optional[str] = None,
    other_name: Optional[str] = None,
) -> NumLike:
    val_ = torch.as_tensor(is_numlike(val, name))
    other_ = torch.as_tensor(is_numlike(other, other_name), device=val_.device)
    if (val_ < other_).any():
        x = "entirely " if val_.numel() > 1 else ""
        raise ValueError(
            f"{_nv(name, val)} is not {x}greater than or equal to"
            f"{_nv(other_name, other)}"
        )
    return val


@overload
def is_btw(
    val: NumLike,
    left: NumLike,
    right: NumLike,
    name: Optional[str] = None,
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
    left_inclusive: bool = False,
    right_inclusive: bool = False,
    allow_none: Literal[False] = False,
) -> NumLike:
    ...


@overload
def is_btw(
    val: Optional[NumLike],
    left: NumLike,
    right: NumLike,
    name: Optional[str] = None,
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
    left_inclusive: bool = False,
    right_inclusive: bool = False,
    allow_none: bool = False,
) -> Optional[NumLike]:
    ...


def is_btw(
    val,
    left,
    right,
    name=None,
    left_name=None,
    right_name=None,
    left_inclusive=False,
    right_inclusive=False,
    allow_none=False,
):
    if allow_none and val is None:
        return val
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


@overload
def is_btw_open(
    val: NumLike,
    left: NumLike,
    right: NumLike,
    name: Optional[str] = None,
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
    allow_none: Literal[False] = False,
) -> NumLike:
    ...


@overload
def is_btw_open(
    val: Optional[NumLike],
    left: NumLike,
    right: NumLike,
    name: Optional[str] = None,
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
    allow_none: bool = False,
) -> Optional[NumLike]:
    ...


def is_btw_open(
    val,
    left,
    right,
    name=None,
    left_name=None,
    right_name=None,
    allow_none: bool = False,
):
    return is_btw(
        val, left, right, name, left_name, right_name, False, False, allow_none
    )


@overload
def is_btw_closed(
    val: NumLike,
    left: NumLike,
    right: NumLike,
    name: Optional[str] = None,
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
    allow_none: Literal[False] = False,
) -> NumLike:
    ...


@overload
def is_btw_closed(
    val: Optional[NumLike],
    left: NumLike,
    right: NumLike,
    name: Optional[str] = None,
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
    allow_none: bool = False,
) -> Optional[NumLike]:
    ...


def is_btw_closed(
    val,
    left,
    right,
    name=None,
    left_name=None,
    right_name=None,
    allow_none=False,
):
    return is_btw(val, left, right, name, left_name, right_name, True, True, allow_none)


def is_open01(val, name=None, allow_none=False):
    return is_btw(val, 0, 1, name, None, None, False, False, allow_none)


is_open01 = cast(_IsCheck[NumLike], is_open01)


def is_closed01(val, name=None, allow_none=False):
    return is_btw(val, 0, 1, name, None, None, True, True, allow_none)


is_closed01 = cast(_IsCheck[NumLike], is_closed01)


def _numlike_special_factory(t: Type[N], *ts: type):
    _type_check = _type_check_factory(t, *ts)

    @_is_check_allow_none
    def _pos_check(val: t, name=None) -> t:
        val = _type_check(val, name)
        return is_pos(val, name)

    @_is_check_allow_none
    def _neg_check(val: t, name=None) -> t:
        val = _type_check(val, name)
        return is_neg(val, name)

    @_is_check_allow_none
    def _nonpos_check(val: t, name=None) -> t:
        val = _type_check(val, name)
        return is_nonpos(val, name)

    @_is_check_allow_none
    def _nonneg_check(val: t, name=None) -> t:
        val = _type_check(val, name)
        return is_nonneg(val, name)

    @_compare_allow_none
    def _equal_check(val: t, other: NumLike, name=None, other_name=None) -> t:
        val = _type_check(val, name)
        return is_equal(val, other, name, other_name)

    @_compare_allow_none
    def _lt_check(val: t, other: NumLike, name=None, other_name=None) -> t:
        val = _type_check(val, name)
        return is_lt(val, other, name, other_name)

    @_compare_allow_none
    def _lte_check(
        val: t,
        other: NumLike,
        name=None,
        other_name=None,
    ) -> t:
        val = _type_check(val, name)
        return is_lte(val, other, name, other_name)

    @_compare_allow_none
    def _gt_check(val: t, other: NumLike, name=None, other_name=None) -> t:
        val = _type_check(val, name)
        return is_gt(val, other, name, other_name)

    @_compare_allow_none
    def _gte_check(val: t, other: NumLike, name=None, other_name=None) -> t:
        val = _type_check(val, name)
        return is_gte(val, other, name, other_name)

    @overload
    def _btw_check(
        val: t,
        left: NumLike,
        right: NumLike,
        name: Optional[str] = None,
        left_name: Optional[str] = None,
        right_name: Optional[str] = None,
        left_inclusive: bool = False,
        right_inclusive: bool = False,
        allow_none: Literal[False] = False,
    ) -> t:
        ...

    @overload
    def _btw_check(
        val: Optional[t],
        left: NumLike,
        right: NumLike,
        name: Optional[str] = None,
        left_name: Optional[str] = None,
        right_name: Optional[str] = None,
        left_inclusive: bool = False,
        right_inclusive: bool = False,
        allow_none: bool = False,
    ) -> Optional[t]:
        ...

    def _btw_check(
        val,
        left,
        right,
        name=None,
        left_name=None,
        right_name=None,
        left_inclusive=False,
        right_inclusive=False,
        allow_none=False,
    ):
        val = _type_check(val, name)
        return is_btw(
            val,
            left,
            right,
            name,
            left_name,
            right_name,
            left_inclusive,
            right_inclusive,
            allow_none,
        )

    @overload
    def _btw_open_check(
        val: t,
        left: NumLike,
        right: NumLike,
        name: Optional[str] = None,
        left_name: Optional[str] = None,
        right_name: Optional[str] = None,
        allow_none: Literal[False] = False,
    ) -> t:
        ...

    @overload
    def _btw_open_check(
        val: Optional[t],
        left: NumLike,
        right: NumLike,
        name: Optional[str] = None,
        left_name: Optional[str] = None,
        right_name: Optional[str] = None,
        allow_none: bool = False,
    ) -> Optional[t]:
        ...

    def _btw_open_check(
        val: t,
        left: NumLike,
        right: NumLike,
        name: Optional[str] = None,
        left_name: Optional[str] = None,
        right_name: Optional[str] = None,
        allow_none=False,
    ) -> t:
        val = _type_check(val, name)
        return is_btw(
            val, left, right, name, left_name, right_name, False, False, allow_none
        )

    @overload
    def _btw_closed_check(
        val: t,
        left: NumLike,
        right: NumLike,
        name: Optional[str] = None,
        left_name: Optional[str] = None,
        right_name: Optional[str] = None,
        allow_none: Literal[False] = False,
    ) -> t:
        ...

    @overload
    def _btw_closed_check(
        val: Optional[t],
        left: NumLike,
        right: NumLike,
        name: Optional[str] = None,
        left_name: Optional[str] = None,
        right_name: Optional[str] = None,
        allow_none: bool = False,
    ) -> Optional[t]:
        ...

    def _btw_closed_check(
        val: t,
        left: NumLike,
        right: NumLike,
        name: Optional[str] = None,
        left_name: Optional[str] = None,
        right_name: Optional[str] = None,
        allow_none=False,
    ) -> t:
        val = _type_check(val, name)
        return is_btw(
            val, left, right, name, left_name, right_name, True, True, allow_none
        )

    @_is_check_allow_none
    def _open01_check(val: t, name=None) -> t:
        val = _type_check(val, name)
        return is_open01(val, name)

    @_is_check_allow_none
    def _closed01_check(val: t, name=None) -> t:
        val = _type_check(val, name)
        return is_nonneg(val, name)

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
) = _numlike_special_factory(int, np.integer)
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
) = _numlike_special_factory(float, np.floating, int, np.integer)
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


@_is_check_allow_none
def is_file(val: StrOrPathLike, name: Optional[str] = None) -> StrOrPathLike:
    if not os.path.isfile(val):
        raise ValueError(f"{_nv(name, val)} is not a file")
    return val


@_is_check_allow_none
def is_dir(val: StrOrPathLike, name: Optional[str] = None) -> StrOrPathLike:
    if not os.path.isdir(val):
        raise ValueError(f"{_nv(name, val)} is not a directory")
    return val


@overload
def has_ndim(
    val: torch.Tensor,
    ndim: int,
    name: Optional[str] = None,
    allow_none: Literal[False] = False,
) -> torch.Tensor:
    ...


@overload
def has_ndim(
    val: Optional[torch.Tensor],
    ndim: int,
    name: Optional[str] = None,
    allow_none: bool = False,
) -> Optional[torch.Tensor]:
    ...


def has_ndim(val, ndim, name=None) -> torch.Tensor:
    if val.ndim != ndim:
        raise ValueError(
            f"Expected {_nv(name, val)} to have dimension {ndim}; got {val.ndim}"
        )
    return val


@_is_check_allow_none
def is_nonempty(val, name=None) -> torch.Tensor:
    if not val.numel():
        raise ValueError(f"Expected {_nv(name, val)} to be nonempty")
    return val


def _cast_factory(
    cast: Callable[[Any], V1],
    check: Optional[Callable[[V1, Optional[str]], V1]] = None,
    cast_name: Optional[str] = None,
):
    def _cast(val: Any, name: Optional[str] = None) -> V1:
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
as_tensor = _cast_factory(torch.as_tensor, cast_name="tensor")
as_posf = _cast_factory(float, is_pos, cast_name="positive float")
as_nat = _cast_factory(int, is_pos, cast_name="natural number")
as_posi = _cast_factory(int, is_pos, cast_name="positive integer")
as_nonnegf = _cast_factory(float, is_nonneg, cast_name="non-negative float")
as_nonnegi = _cast_factory(int, is_nonneg, cast_name="non-negative integer")
as_negf = _cast_factory(float, is_neg, cast_name="negative float")
as_negi = _cast_factory(int, is_neg, cast_name="negative integer")
as_nonposf = _cast_factory(float, is_nonpos, cast_name="non-positive float")
as_nonposi = _cast_factory(int, is_nonpos, cast_name="non-positive integer")
as_open01 = _cast_factory(float, is_open01, cast_name="float within (0, 1)")
as_closed01 = _cast_factory(float, is_closed01, cast_name="float within [0, 1]")
as_path = _cast_factory(Path)
as_path_file = _cast_factory(Path, is_file, cast_name="readable file")
as_path_dir = _cast_factory(Path, is_dir, cast_name="readable directory")
as_file = _cast_factory(str, is_file, cast_name="readable file")
as_dir = _cast_factory(str, is_dir, cast_name="readable directory")
