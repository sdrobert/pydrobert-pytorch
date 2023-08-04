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

"""Boilerplate for checking argument values"""

import os
import string

from typing import (
    Optional,
    TypeVar,
    Callable,
    Any,
    Collection,
    Type,
    List,
    TYPE_CHECKING,
)
from pathlib import Path
from typing_extensions import ParamSpec, Concatenate

import torch

from ._compat import script

V = TypeVar("V")
P = ParamSpec("P")
NumLike = TypeVar("NumLike", float, int, torch.Tensor)
StrPathLike = TypeVar("StrPathLike", str, os.PathLike, Path)


@script
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


def _is_check_factory(t: Type[V], *ts: type):
    ts = (t,) + ts
    suf = "n" if t.__name__.startswith(("a", "e", "i", "o", "u")) else ""

    @script
    def _is_check(val: t, name: Optional[str] = None) -> t:
        if torch.jit.is_scripting():
            return val
        elif isinstance(val, ts):
            return val if isinstance(val, t) else t(val)
        else:
            raise ValueError(f"{_nv(name, val)} is not a{suf} {t.__name__}")

    return _is_check


is_str = _is_check_factory(str)
is_int = _is_check_factory(int)
is_bool = _is_check_factory(bool, int, float)
is_float = _is_check_factory(float, int)
is_pathlike = _is_check_factory(os.PathLike)
is_path = _is_check_factory(Path)
is_tensor = _is_check_factory(torch.Tensor)


def _is_numcheck_factory(t: Type[NumLike], *ts: Type[NumLike]):
    ts = (t,) + ts

    @script
    def _is_poscheck(val: t, name: Optional[str] = None) -> t:
        if not torch.jit.is_scripting():
            if isinstance(val, ts):
                val = t(val)
            else:
                raise ValueError(f"{_nv(name, val)} is not a positive {t.__name__}")
        if val <= 0:
            raise ValueError(f"{_nv(name, val)} is not a positive {t.__name__}")
        return val

    @script
    def _is_negcheck(val: t, name: Optional[str] = None) -> t:
        if not torch.jit.is_scripting():
            if isinstance(val, ts):
                val = t(val)
            else:
                raise ValueError(f"{_nv(name, val)} is not a negative {t.__name__}")
        if val >= 0:
            raise ValueError(f"{_nv(name, val)} is not a negative {t.__name__}")
        return val

    @script
    def _is_nonposcheck(val: t, name: Optional[str] = None) -> t:
        if not torch.jit.is_scripting():
            if isinstance(val, ts):
                val = t(val)
            else:
                raise ValueError(f"{_nv(name, val)} is not a non-positive {t.__name__}")
        if val > 0:
            raise ValueError(f"{_nv(name, val)} is not a non-positive {t.__name__}")
        return val

    @script
    def _is_nonnegcheck(val: t, name: Optional[str] = None) -> t:
        if not torch.jit.is_scripting():
            if isinstance(val, ts):
                val = t(val)
            else:
                raise ValueError(f"{_nv(name, val)} is not a non-negative {t.__name__}")
        if val < 0:
            raise ValueError(f"{_nv(name, val)} is not a non-negative {t.__name__}")
        return val

    return _is_poscheck, _is_negcheck, _is_nonposcheck, _is_nonnegcheck


is_posi, is_negi, is_nonposi, is_nonnegi = _is_numcheck_factory(int)
is_nat = is_posi
is_posf, is_negf, is_nonposf, is_nonnegf = _is_numcheck_factory(float, int)


@script
def is_post(val: torch.Tensor, name: Optional[str] = None) -> torch.Tensor:
    if not torch.jit.is_scripting():
        if not isinstance(val, torch.Tensor):
            raise ValueError(
                f"{_nv(name, val)} is not a tensor of only positive values"
            )
    if (val <= 0).any():
        raise ValueError(f"{_nv(name, val)} is not a tensor of only positive values")
    return val


@script
def is_negt(val: torch.Tensor, name: Optional[str] = None) -> torch.Tensor:
    if not torch.jit.is_scripting():
        if not isinstance(val, torch.Tensor):
            raise ValueError(
                f"{_nv(name, val)} is not a tensor of only negative values"
            )
    if (val >= 0).any():
        raise ValueError(f"{_nv(name, val)} is not a tensor of only negative values")
    return val


@script
def is_nonpost(val: torch.Tensor, name: Optional[str] = None) -> torch.Tensor:
    if not torch.jit.is_scripting():
        if not isinstance(val, torch.Tensor):
            raise ValueError(
                f"{_nv(name, val)} is not a tensor of only non-positive values"
            )
    if (val > 0).any():
        raise ValueError(
            f"{_nv(name, val)} is not a tensor of only non-positive values"
        )
    return val


@script
def is_nonnegt(val: torch.Tensor, name: Optional[str] = None) -> torch.Tensor:
    if not torch.jit.is_scripting():
        if not isinstance(val, torch.Tensor):
            raise ValueError(
                f"{_nv(name, val)} is not a tensor of only non-negative values"
            )
    if (val < 0).any():
        raise ValueError(
            f"{_nv(name, val)} is not a tensor of only non-negative values"
        )
    return val


@script
def is_token(
    val: str,
    name: Optional[str] = None,
    empty_okay: bool = False,
    whitespace: str = string.whitespace,
) -> str:
    if not empty_okay and not len(val):
        raise ValueError(f"{_nv(name, val)} is empty")
    else:
        for w in whitespace:
            if w in val:
                raise ValueError(f"{_nv(name, val)} contains '{w}'")
    return val


@torch.jit.unused
def is_a(val: V, t: Type[V], name: Optional[str] = None) -> V:
    if not isinstance(val, t):
        suf = "n" if t.__name__.startswith(("a", "e", "i", "o", "u")) else ""
        raise ValueError(f"{_nv(name, val)} is not a{suf} {t.__name__}")
    return val


if TYPE_CHECKING:

    def is_pos(val: NumLike, name: Optional[str] = None) -> NumLike:
        ...

    def is_neg(val: NumLike, name: Optional[str] = None) -> NumLike:
        ...

    def is_nonpos(val: NumLike, name: Optional[str] = None) -> NumLike:
        ...

    def is_nonneg(val: NumLike, name: Optional[str] = None) -> NumLike:
        ...

    def is_equal(
        val: NumLike,
        other: NumLike,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
    ) -> NumLike:
        ...

    def is_exactly(
        val: V,
        other: Any,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
    ) -> V:
        ...

    def is_lt(
        val: NumLike,
        other: NumLike,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
        inclusive: bool = False,
    ) -> NumLike:
        ...

    def is_lte(
        val: NumLike,
        other: NumLike,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
    ) -> NumLike:
        ...

    def is_gt(
        val: NumLike,
        other: NumLike,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
        inclusive: bool = False,
    ) -> NumLike:
        ...

    def is_gte(
        val: NumLike,
        other: NumLike,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
    ) -> NumLike:
        ...

    def is_btw(
        val: NumLike,
        left: NumLike,
        right: NumLike,
        name: Optional[str] = None,
        left_name: Optional[str] = None,
        right_name: Optional[str] = None,
        left_inclusive: bool = False,
        right_inclusive: bool = False,
    ) -> NumLike:
        ...

    def is_in(val: V, choices: Collection[Any], name: Optional[str] = None) -> V:
        ...


else:

    @script
    def is_pos(val: Any, name: Optional[str] = None) -> Any:
        okay: Optional[bool] = None
        if isinstance(val, torch.Tensor):
            okay = bool((val > 0).all().item())
        if isinstance(val, (float, int)):
            okay = val > 0
        if okay is None:
            raise TypeError("type not implemented for is_pos")
        elif not okay:
            raise ValueError(f"{_nv(name, val)} is not positive")
        return val

    @script
    def is_neg(val: Any, name: Optional[str] = None) -> Any:
        okay: Optional[bool] = None
        if isinstance(val, torch.Tensor):
            okay = bool((val < 0).all().item())
        if isinstance(val, (float, int)):
            okay = val < 0
        if okay is None:
            raise TypeError("type not implemented for is_neg")
        elif not okay:
            raise ValueError(f"{_nv(name, val)} is not negative")
        return val

    @script
    def is_nonpos(val: Any, name: Optional[str] = None) -> Any:
        okay: Optional[bool] = None
        if isinstance(val, torch.Tensor):
            okay = bool((val <= 0).all().item())
        if isinstance(val, (float, int)):
            okay = val <= 0
        if okay is None:
            raise TypeError("type not implemented for is_nonpos")
        elif not okay:
            raise ValueError(f"{_nv(name, val)} is positive")
        return val

    @script
    def is_nonneg(val: Any, name: Optional[str] = None) -> Any:
        okay: Optional[bool] = None
        if isinstance(val, torch.Tensor):
            okay = bool((val >= 0).all().item())
        if isinstance(val, (float, int)):
            okay = val >= 0
        if okay is None:
            raise TypeError("type not implemented for is_nonneg")
        elif not okay:
            raise ValueError(f"{_nv(name, val)} is not negative")
        return val

    @script
    def is_equal(
        val: Any,
        other: Any,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
    ) -> Any:
        okay: Optional[bool] = None
        if isinstance(val, torch.Tensor) and isinstance(other, torch.Tensor):
            okay = bool((val == other).all().item())
        if isinstance(val, torch.Tensor) and isinstance(other, (int, float)):
            okay = bool((val == other).all().item())
        if isinstance(other, torch.Tensor) and isinstance(val, (int, float)):
            okay = bool((other == val).all().item())
        if isinstance(val, (float, int)) and isinstance(other, (float, int)):
            okay = float(val) == float(other)
        if okay is None:
            raise TypeError("type not implemented for is_equal")
        elif not okay:
            raise ValueError(
                f"{_nv(name, val)} does not equal {_nv(other_name, other)}"
            )
        return val

    @script
    def is_exactly(
        val: Any,
        other: Any,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
    ) -> Any:
        if val is not other:
            raise ValueError(f"{_nv(name, val)} is not {_nv(other_name, other)}")
        return val

    @script
    def is_lt(
        val: Any,
        other: Any,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
        inclusive: bool = False,
    ) -> Any:
        okay: Optional[bool] = None
        comp = ">"
        if inclusive:
            if isinstance(val, torch.Tensor) and isinstance(other, torch.Tensor):
                okay = bool((val <= other).all().item())
            if isinstance(val, torch.Tensor) and isinstance(other, (float, int)):
                okay = bool((val <= other).all().item())
            if isinstance(other, torch.Tensor) and isinstance(val, (float, int)):
                okay = bool((other >= val).all().item())
            if isinstance(val, (float, int)) and isinstance(other, (float, int)):
                okay = float(val) <= float(other)
        else:
            comp = ">="
            if isinstance(val, torch.Tensor) and isinstance(other, torch.Tensor):
                okay = bool((val < other).all().item())
            if isinstance(val, torch.Tensor) and isinstance(other, (float, int)):
                okay = bool((val < other).all().item())
            if isinstance(other, torch.Tensor) and isinstance(val, (float, int)):
                okay = bool((other > val).all().item())
            if isinstance(val, (float, int)) and isinstance(other, (float, int)):
                okay = float(val) < float(other)
        if okay is None:
            raise TypeError("type not implemented for is_lt")
        if not okay:
            raise ValueError(f"{_nv(name, val)} {comp} {_nv(other_name, other)}")
        return val

    def is_lte(
        val: Any,
        other: Any,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
    ) -> Any:
        return is_lt(val, other, name, other_name, True)

    @script
    def is_gt(
        val: Any,
        other: Any,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
        inclusive: bool = False,
    ) -> Any:
        okay: Optional[bool] = None
        comp = "<"
        if inclusive:
            if isinstance(val, torch.Tensor) and isinstance(other, torch.Tensor):
                okay = bool((val >= other).all().item())
            if isinstance(val, torch.Tensor) and isinstance(other, (float, int)):
                okay = bool((val >= other).all().item())
            if isinstance(other, torch.Tensor) and isinstance(val, (float, int)):
                okay = bool((other <= val).all().item())
            if isinstance(val, (float, int)) and isinstance(other, (float, int)):
                okay = float(val) >= float(other)
        else:
            comp = "<="
            if isinstance(val, torch.Tensor) and isinstance(other, torch.Tensor):
                okay = bool((val > other).all().item())
            if isinstance(val, torch.Tensor) and isinstance(other, (float, int)):
                okay = bool((val > other).all().item())
            if isinstance(other, torch.Tensor) and isinstance(val, (float, int)):
                okay = bool((other < val).all().item())
            if isinstance(val, (float, int)) and isinstance(other, (float, int)):
                okay = float(val) > float(other)
        if okay is None:
            raise TypeError("type not implemented for is_gt")
        if not okay:
            raise ValueError(f"{_nv(name, val)} {comp} {_nv(other_name, other)}")
        return val

    def is_gte(
        val: Any,
        other: Any,
        name: Optional[str] = None,
        other_name: Optional[str] = None,
    ) -> Any:
        return is_gt(val, other, name, other_name, True)

    def is_btw(
        val: Any,
        left: Any,
        right: Any,
        name: Optional[str] = None,
        left_name: Optional[str] = None,
        right_name: Optional[str] = None,
        left_inclusive: bool = False,
        right_inclusive: bool = False,
    ) -> Any:
        is_lt(val, right, name, right_name, right_inclusive)
        is_gt(val, left, name, left_name, left_inclusive)
        return val

    @script
    def is_in(val: Any, choices: List[Any], name: Optional[str] = None) -> Any:
        okay: bool = False
        if torch.jit.is_scripting():
            for choice in choices:
                if val is choice:
                    okay = True
        else:
            okay = val in choices
        if not okay:
            raise ValueError(f"{_nv(name, val)} not in {list(choices)}")
        return val


def is_posi(val: int, name: Optional[str] = None) -> int:
    return val


is_nat = is_posi


def is_negi(val: int, name: Optional[str] = None) -> int:
    val = is_int(val, name)
    is_neg(val, name)
    return val


def is_nonposi(val: int, name: Optional[str] = None) -> int:
    val = is_int(val, name)
    is_nonpos(val, name)
    return val


def is_nonnegi(val: int, name: Optional[str] = None) -> int:
    val = is_int(val, name)
    is_nonneg(val, name)
    return val


def is_posi(val: int, name: Optional[str] = None) -> int:
    val = is_int(val, name)
    is_pos(val, name)
    return val


is_nat = is_posi


def is_negi(val: int, name: Optional[str] = None) -> int:
    val = is_int(val, name)
    is_neg(val, name)
    return val


def is_nonposi(val: int, name: Optional[str] = None) -> int:
    val = is_int(val, name)
    is_nonpos(val, name)
    return val


def is_nonnegi(val: int, name: Optional[str] = None) -> int:
    val = is_int(val, name)
    is_nonneg(val, name)
    return val


def is_open01(val: float, name: Optional[str] = None) -> float:
    is_float(val, name)
    is_btw(val, 0.0, 1.0, name)
    return val


def is_closed01(val: float, name: Optional[str] = None) -> float:
    is_float(val, name)
    is_btw(val, 0.0, 1.0, name, left_inclusive=True, right_inclusive=True)
    return val


@torch.jit.unused
def is_file(val: StrPathLike, name: Optional[str] = None) -> StrPathLike:
    if not os.path.isfile(val):
        raise ValueError(f"{_nv(name, val)} is not a file")
    return val


@torch.jit.unused
def is_dir(val: StrPathLike, name: Optional[str] = None) -> StrPathLike:
    if not os.path.isdir(val):
        raise ValueError(f"{_nv(name, val)} is not a directory")
    return val


@script
def has_ndim(val: torch.Tensor, ndim: int, name: str = "tensor") -> torch.Tensor:
    if val.ndim != ndim:
        raise ValueError(
            f"Expected {_nv(name, val)} to have dimension {ndim}; got {val.ndim}"
        )
    return val


@script
def is_nonempty(val: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    if not val.numel():
        raise ValueError(f"Expected {name} to be nonempty")
    return val


def _cast_factory(
    cast: Callable[[Any], V],
    check: Optional[Callable[[V, Optional[str]], V]] = None,
    cast_name: Optional[str] = None,
):
    @torch.jit.unused
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
