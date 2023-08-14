# Copyright 2023 Sean Robertson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import pytest
import numpy as np

from pathlib import Path

from pydrobert.torch import argcheck


@pytest.mark.cpu
@pytest.mark.parametrize(
    "check,val,exp",
    [
        (argcheck.is_str, "a", "a"),
        (argcheck.is_str, "", ""),
        (argcheck.is_str, bytes("a", "utf-8"), None),
        (argcheck.is_int, 1, 1),
        (argcheck.is_int, -1, -1),
        (argcheck.is_int, np.uint32(1), 1),
        (argcheck.is_int, 1.0, None),
        (argcheck.is_int, np.float32(1.0), None),
        (argcheck.is_bool, True, True),
        (argcheck.is_bool, False, False),
        (argcheck.is_bool, 1, None),
        (argcheck.is_bool, "", None),
        (argcheck.is_float, 1.0, 1.0),
        (argcheck.is_float, np.inf, np.inf),
        (argcheck.is_float, 1, 1.0),
        (argcheck.is_float, np.float64(3.14), 3.14),
        (argcheck.is_float, np.uint8(255), 255.0),
        (argcheck.is_tensor, torch.ones(5), torch.ones(5)),
        (argcheck.is_tensor, 1, None),
        (argcheck.is_path, Path("."), Path(".")),
        (argcheck.is_path, ".", Path(".")),
        (argcheck.is_path, b".", None),
        (argcheck.is_numlike, 1, 1),
        (argcheck.is_numlike, 1.0, 1.0),
        (argcheck.is_numlike, "", None),
        (argcheck.is_token, "foo", "foo"),
        (argcheck.is_token, "", None),
        (argcheck.is_token, "foo bar", None),
        (argcheck.is_posi, 1, 1),
        (argcheck.is_posi, np.uint8(2), 2),
        (argcheck.is_posi, 1.0, None),
        (argcheck.is_posi, 0, None),
        (argcheck.is_nonposf, -1.0, -1.0),
        (argcheck.is_nonposf, -1, -1.0),
        (argcheck.is_nonposf, np.uint8(0), 0.0),
        (argcheck.is_nonposf, 1.0, None),
        (argcheck.is_closed01t, torch.arange(2), torch.arange(2)),
        (argcheck.is_closed01t, 1, None),
        (argcheck.is_closed01t, -torch.arange(2), None),
        (argcheck.is_open01t, torch.full((5,), 0.5), torch.full((5,), 0.5)),
        (argcheck.is_open01t, 0.5, None),
        (argcheck.is_open01t, torch.arange(2), None),
        (argcheck.is_file, __file__, __file__),
        (argcheck.is_file, Path.cwd(), None),
        (argcheck.is_dir, Path.cwd(), Path.cwd()),
        (argcheck.is_dir, __file__, None),
        (argcheck.is_nonempty, torch.ones(1), torch.ones(1)),
        (argcheck.is_nonempty, torch.ones(0), None),
    ],
)
def test_is_type(check, val, exp):
    if exp is None:
        with pytest.raises(ValueError):
            check(val)
    else:
        act = check(val)
        assert type(exp) is type(act)
        if isinstance(act, torch.Tensor):
            assert (exp == act).all()
        else:
            assert exp == act
        assert check(None, allow_none=True) is None


@pytest.mark.parametrize(
    "check,val,rest,good",
    [
        (argcheck.is_a, 1, (int,), True),
        (argcheck.is_a, 1, (float,), False),
        (argcheck.is_in, 2, (range(10),), True),
        (argcheck.is_in, "1", (range(10),), False),
        (argcheck.is_exactly, 1, (1,), True),
        (argcheck.is_exactly, 1, (1.0,), False),
        (argcheck.is_equal, 1, (1.0,), True),
        (argcheck.is_equal, torch.ones(2, 1), (torch.ones(1, 2),), True),
        (argcheck.is_equal, torch.ones(2, 1), (torch.arange(2),), False),
        (argcheck.is_lt, 1, (torch.arange(2) + 2,), True),
        (argcheck.is_lt, 1.0, (torch.arange(2) + 1,), False),
        (argcheck.is_gte, torch.full((5,), np.inf), (10_000,), True),
        (argcheck.is_gte, 1, (1.0,), True),
        (argcheck.is_gte, 0, (torch.arange(2),), False),
        (argcheck.is_btw, 30.5, (0.1, np.inf), True),
        (argcheck.is_btw, 30.5, (0.1, -np.inf), False),
        (argcheck.is_btw_open, 1, (0.999, 1.001), True),
        (argcheck.is_btw_open, 1, (0.999, 1), False),
        (argcheck.is_btw_closed, 1, (1, 1), True),
        (argcheck.is_btw_closed, 1.001, (1, 1), False),
        (argcheck.has_ndim, torch.empty(1, 2, 3), (3,), True),
        (argcheck.has_ndim, torch.empty(0), (3,), False),
    ],
)
def test_comparative(check, val, rest, good):
    if good:
        assert check(val, *rest) is val
    else:
        with pytest.raises(ValueError):
            check(val, *rest)
    assert check(None, *rest, allow_none=True) is None
