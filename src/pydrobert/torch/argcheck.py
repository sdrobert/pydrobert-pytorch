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

from typing import Optional, TypeVar, Type, Callable, Any, Collection
from pathlib import Path
from typing_extensions import ParamSpec, Concatenate

import torch

V = TypeVar("V")
P = ParamSpec("P")
StrPathLike = TypeVar("StrPathLike", str, os.PathLike, Path)


def _nv(name: Optional[str], val) -> str:
    return f"'{val}'" if name is None else f"{name} ('{val}')"


def _simple_check_factory(t: Type[V], *ts: Type[Any]):
    ts = (t,) + ts
    suf = "n" if t.__name__.startswith(("a", "e", "i", "o", "u")) else ""

    def _check(val: V, *args, name: Optional[str] = None, **kwargs) -> V:
        if not isinstance(val, ts):
            raise ValueError(f"{_nv(name, val)} is not a{suf} {t.__name__}")
        return val

    return _check


is_str = _simple_check_factory(str)
is_int = _simple_check_factory(int)
is_bool = _simple_check_factory(bool)
is_float = _simple_check_factory(float, int)
is_pathlike = _simple_check_factory(os.PathLike)
is_path = _simple_check_factory(Path)


def _chain_factory(
    *checks: Callable[Concatenate[V, P], V]
) -> Callable[Concatenate[V, P], V]:
    def _chain(val: V, *args, **kwargs) -> V:
        for check in checks:
            val = check(val, *args, **kwargs)
        return val

    return _chain


def is_pos(val: V, *args, name: Optional[str] = None, **kwargs) -> V:
    if val <= 0:
        raise ValueError(f"{_nv(name, val)} is not positive")
    return val


is_posf = _chain_factory(is_float, is_pos)
is_nat = is_posi = _chain_factory(is_int, is_pos)


def is_nonneg(val: V, *args, name: Optional[str] = None, **kwargs) -> V:
    if val < 0:
        raise ValueError(f"{_nv(name, val)} is negative")
    return val


is_nonnegf = _chain_factory(is_float, is_nonneg)
is_nonnegi = _chain_factory(is_int, is_nonneg)


def is_neg(val: V, *args, name: Optional[str] = None, **kwargs) -> V:
    if val >= 0:
        raise ValueError(f"{_nv(name, val)} is not negative")
    return val


is_negf = _chain_factory(is_float, is_nonneg)
is_negi = _chain_factory(is_int, is_nonneg)


def is_nonpos(val: V, *args, name: Optional[str] = None, **kwargs) -> V:
    if val > 0:
        raise ValueError(f"{_nv(name, val)} is positive")
    return val


is_nonposf = _chain_factory(is_float, is_nonpos)
is_nonposi = _chain_factory(is_int, is_nonpos)


def is_lt(
    val: V,
    other,
    *args,
    name: Optional[str] = None,
    other_name: Optional[str] = None,
    inclusive: bool = False,
    **kwargs,
) -> V:
    if inclusive and val > other:
        raise ValueError(f"{_nv(name, val)} is greater than {_nv(other_name, other)}")
    elif val >= other:
        raise ValueError(
            f"{_nv(name, val)} is greater than or equal to {_nv(other_name, other)}"
        )
    return val


is_ltf = _chain_factory(is_float, is_lt)
is_lti = _chain_factory(is_int, is_lt)


def is_gt(
    val: V,
    other,
    *args,
    name: Optional[str] = None,
    other_name: Optional[str] = None,
    inclusive: bool = False,
    **kwargs,
) -> V:
    if inclusive and val < other:
        raise ValueError(f"{_nv(name, val)} is less than {_nv(other_name, other)}")
    elif val <= other:
        raise ValueError(
            f"{_nv(name, val)} is less than or equal to {_nv(other_name, other)}"
        )
    return val


is_gtf = _chain_factory(is_float, is_gt)
is_gti = _chain_factory(is_int, is_gt)


def is_btw(
    val: V,
    left,
    right,
    *args,
    name: Optional[str] = None,
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
    left_inclusive: bool = False,
    right_inclusive: bool = False,
    **kwargs,
) -> V:
    val = is_gt(val, left, name=name, other_name=left_name, inclusive=left_inclusive)
    val = is_lt(val, right, name=name, other_name=right_name, inclusive=right_inclusive)
    return val


is_btwf = _chain_factory(is_float, is_btw)
is_btwi = _chain_factory(is_int, is_btw)


def is_mag(val: float, *args, **kwargs) -> float:
    return is_btwf(val, 0, 1, *args, **kwargs)


is_open01 = is_mag


def is_closed01(val: float, *args, **kwargs) -> float:
    return is_btwf(
        val, 0, 1, *args, inclusive_left=True, inclusive_right=True, **kwargs
    )


def is_file(val: StrPathLike, *args, name: Optional[str], **kwargs) -> StrPathLike:
    if not os.path.isfile(val):
        raise ValueError(f"{_nv(name, val)} is not a file")
    return val


def is_dir(val: StrPathLike, *args, name: Optional[str], **kwargs) -> StrPathLike:
    if not os.path.isdir(val):
        raise ValueError(f"{_nv(name, val)} is not a directory")
    return val


def is_in(
    val: V, choices: Collection[V], *args, name: Optional[str] = None, **kwargs
) -> V:
    if val not in choices:
        raise ValueError(f"{_nv(name, val)} not in '{choices}'")
    return val


def _cast_factory(cast: Callable[[Any], V], *checks: Callable[Concatenate[V, P], V]):
    def _cast(val: Any, *args, name: Optional[str] = None, **kwargs) -> V:
        try:
            val = cast(val)
        except:
            raise ValueError(f"Could not cast {_nv(name, val)} as {cast.__name__}")
        for check in checks:
            val = check(val, *args, name=name, **kwargs)
        return val

    return _cast


as_str = _cast_factory(str)
as_int = _cast_factory(int)
as_bool = _cast_factory(bool)
as_float = _cast_factory(float)
as_posf = _cast_factory(float, is_pos)
as_nat = as_posi = _cast_factory(int, is_pos)
as_nonnegf = _cast_factory(float, is_nonneg)
as_nonnegi = _cast_factory(int, is_nonneg)
as_negf = _cast_factory(float, is_neg)
as_negi = _cast_factory(int, is_neg)
as_nonposf = _cast_factory(float, is_nonpos)
as_nonposi = _cast_factory(int, is_nonpos)
as_ltf = _cast_factory(float, is_lt)
as_lti = _cast_factory(int, is_lt)
as_gtf = _cast_factory(float, is_gt)
as_gti = _cast_factory(int, is_gt)
as_mag = as_open01 = _cast_factory(float, is_mag)
as_closed01 = _cast_factory(float, is_closed01)
as_path = _cast_factory(Path)
as_path_file = _cast_factory(Path, is_file)
as_path_dir = _cast_factory(Path, is_dir)
as_file = _cast_factory(str, is_file)
as_dir = _cast_factory(str, is_dir)
