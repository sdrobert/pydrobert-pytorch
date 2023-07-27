# Copyright 2022 Sean Robertson
#
# Code for broadcast_shapes was adapted from PyTorch
# https://github.com/pytorch/pytorch/blob/2367face24afb159f73ebf40dc6f23e46132b770/torch/functional.py
# Code for TorchVersion was taken directly from PyTorch
# https://github.com/pytorch/pytorch/blob/b737e09f60dd56dbae520e436648e1f3ebc1f937/torch/torch_version.py
# Code for one_hot was taken directly from PyTorch.
# https://github.com/pytorch/pytorch/blob/89c844db9b3120223bc4e45a1dcbb2368301e956/torch/distributions/constraints.py
# See LICENSE_pytorch in project root directory for PyTorch license.
#
# Code for check_methods was taken directly from CPython
# https://github.com/python/cpython/blob/2085bd0877e17ad4d98a4586d5eabb6faecbb190/Lib/_collections_abc.py
# With the following PSF license
#
# Copyright 2007 Google, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.
#
# with the additional notices
# https://docs.python.org/3/copyright.html?highlight=copyright

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    NamedTuple,
    Set,
)

import torch

from . import config


__all__ = [
    "broadcast_shapes",
    "check_methods",
    "euler_constant",
    "jit_isinstance",
    "linalg_solve",
    "meshgrid",
    "one_hot",
    "pad_sequence",
    "script",
    "SpoofPackedSequence",
    "trunc_divide",
]


def check_methods(C, *methods):
    try:
        mro = C.__mro__
        for method in methods:
            for B in mro:
                if method in B.__dict__:
                    if B.__dict__[method] is None:
                        return NotImplemented
                    break
            else:
                return NotImplemented
    except AttributeError:
        for method in methods:
            if getattr(C, method, None) is None:
                return NotImplemented
    return True


# to avoid some scripting issues with torch.utils.nn.PackedSequence
class SpoofPackedSequence(NamedTuple):
    data: torch.Tensor
    batch_sizes: torch.Tensor
    sorted_indices: Optional[torch.Tensor]
    unsorted_indices: Optional[torch.Tensor]


try:
    from torch.torch_version import __version__ as _v  # type: ignore
except ModuleNotFoundError:
    from torch.version import __version__ as internal_version
    from pkg_resources import packaging  # type: ignore[attr-defined]

    Version = packaging.version.Version
    InvalidVersion = packaging.version.InvalidVersion

    class TorchVersion(str):
        """A string with magic powers to compare to both Version and iterables!
        Prior to 1.10.0 torch.__version__ was stored as a str and so many did
        comparisons against torch.__version__ as if it were a str. In order to not
        break them we have TorchVersion which masquerades as a str while also
        having the ability to compare against both packaging.version.Version as
        well as tuples of values, eg. (1, 2, 1)
        Examples:
            Comparing a TorchVersion object to a Version object
                TorchVersion('1.10.0a') > Version('1.10.0a')
            Comparing a TorchVersion object to a Tuple object
                TorchVersion('1.10.0a') > (1, 2)    # 1.2
                TorchVersion('1.10.0a') > (1, 2, 1) # 1.2.1
            Comparing a TorchVersion object against a string
                TorchVersion('1.10.0a') > '1.2'
                TorchVersion('1.10.0a') > '1.2.1'
        """

        # fully qualified type names here to appease mypy
        def _convert_to_version(
            self, inp: Union[packaging.version.Version, str, Iterable]
        ) -> packaging.version.Version:
            if isinstance(inp, Version):
                return inp
            elif isinstance(inp, str):
                return Version(inp)
            elif isinstance(inp, Iterable):
                # Ideally this should work for most cases by attempting to group
                # the version tuple, assuming the tuple looks (MAJOR, MINOR, ?PATCH)
                # Examples:
                #   * (1)         -> Version("1")
                #   * (1, 20)     -> Version("1.20")
                #   * (1, 20, 1)  -> Version("1.20.1")
                return Version(".".join((str(item) for item in inp)))
            else:
                raise InvalidVersion(inp)

        def __gt__(self, cmp):
            try:
                return Version(self).__gt__(self._convert_to_version(cmp))
            except InvalidVersion:
                # Fall back to regular string comparison if dealing with an invalid
                # version like 'parrot'
                return super().__gt__(cmp)

        def __lt__(self, cmp):
            try:
                return Version(self).__lt__(self._convert_to_version(cmp))
            except InvalidVersion:
                # Fall back to regular string comparison if dealing with an invalid
                # version like 'parrot'
                return super().__lt__(cmp)

        def __eq__(self, cmp):
            try:
                return Version(self).__eq__(self._convert_to_version(cmp))
            except InvalidVersion:
                # Fall back to regular string comparison if dealing with an invalid
                # version like 'parrot'
                return super().__eq__(cmp)

        def __ge__(self, cmp):
            try:
                return Version(self).__ge__(self._convert_to_version(cmp))
            except InvalidVersion:
                # Fall back to regular string comparison if dealing with an invalid
                # version like 'parrot'
                return super().__ge__(cmp)

        def __le__(self, cmp):
            try:
                return Version(self).__le__(self._convert_to_version(cmp))
            except InvalidVersion:
                # Fall back to regular string comparison if dealing with an invalid
                # version like 'parrot'
                return super().__le__(cmp)

    _v = TorchVersion(internal_version)

try:
    _v < "1.8.0"
except TypeError:
    # This occurs in autodoc when torch is being mocked.
    _v = ""

if config.USE_JIT:
    script = torch.jit.script
else:
    try:
        script = torch.jit.script_if_tracing
    except AttributeError:

        def script(obj, *args, **kwargs):
            return obj


if _v < "1.10.0":
    meshgrid = torch.meshgrid

    trunc_divide = torch.floor_divide
else:

    def trunc_divide(input: torch.Tensor, other: Any) -> torch.Tensor:
        if not torch.jit.is_scripting():
            return input.div(other, rounding_mode="trunc")
        elif torch.jit.isinstance(other, float):
            return input.div(other, rounding_mode="trunc")
        elif torch.jit.isinstance(other, int):
            return input.div(other, rounding_mode="trunc")
        elif torch.jit.isinstance(other, torch.Tensor):
            return input.div(other, rounding_mode="trunc")
        else:
            assert False

    def meshgrid(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.meshgrid(a, b, indexing="ij")
        return x[0], x[1]


if _v < "1.8.0":
    from torch.distributions.gumbel import euler_constant

    @script
    def pad_sequence(
        sequences: List[torch.Tensor],
        batch_first: bool = False,
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        shape = sequences[0].size()
        shape_rest = shape[1:]
        lens = [x.size(0) for x in sequences]
        max_len = max(lens)
        pad_shapes = [(max_len - x,) + shape_rest for x in lens]
        sequences = [
            torch.cat(
                [
                    seq,
                    torch.full(ps, padding_value, device=seq.device, dtype=seq.dtype),
                ],
                0,
            )
            for seq, ps in zip(sequences, pad_shapes)
        ]
        return torch.stack(sequences, 0 if batch_first else 1)

    def linalg_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.solve(B, A)[0]

    @torch.jit.ignore
    def jit_isinstance(obj: Any, x: type) -> bool:
        if isinstance(obj, torch.nn.utils.rnn.PackedSequence):
            obj = obj.data, obj.batch_sizes, obj.sorted_indices, obj.unsorted_indices
        origin = getattr(x, "__origin__", None)
        if origin is None:
            return isinstance(obj, x)
        if origin in {tuple, list, set, List, Set, Tuple}:
            args = getattr(x, "__args__", None)
            if not args:
                return (
                    (origin in {tuple, Tuple} and obj == tuple())
                    or (origin in {list, List} and obj == list())
                    or (origin in {set, Set} and obj == set())
                )
            if origin in {tuple, Tuple}:
                return (len(obj) is len(args)) and all(
                    jit_isinstance(*y) for y in zip(obj, args)
                )
            else:
                assert len(args) == 1
                return all(jit_isinstance(o, args[0]) for o in obj)
        elif origin is Union:
            args = x.__args__
            return any(jit_isinstance(obj, y) for y in args)
        return False

    from torch.distributions.constraints import Constraint

    class one_hot(Constraint):
        is_discrete = True
        event_dim = 1

        def check(self, value):
            is_boolean = (value == 0) | (value == 1)
            is_normalized = value.sum(-1).eq(1)
            return is_boolean.all(-1) & is_normalized


else:
    from torch.distributions.utils import euler_constant
    from torch.distributions.constraints import one_hot

    if config.USE_JIT:
        script = torch.jit.script
    else:
        script = torch.jit.script_if_tracing

    pad_sequence = torch.nn.utils.rnn.pad_sequence
    linalg_solve = torch.linalg.solve
    jit_isinstance = torch.jit.isinstance


if _v < "1.7.0":

    @script
    def movedim(a: torch.Tensor, source: int, dest: int) -> torch.Tensor:
        D = a.ndim
        if source < -D or source >= D:
            raise RuntimeError(
                f"Dimension 'source' expected to be in the range [{-D},{D - 1}], "
                f"got {source}"
            )
        source = (source + D) % D
        if dest < -D or dest >= D:
            raise RuntimeError(
                f"Dimension 'dest' expected to be in the range [{-D},{D - 1}], "
                f"got {dest}"
            )
        dest = (dest + D) % D
        if source < dest:
            for s in range(source, dest):
                a = a.transpose(s, s + 1)
        elif source > dest:
            for s in range(source, dest, -1):
                a = a.transpose(s - 1, s)
        return a


else:
    movedim = torch.movedim


if _v < "1.6.0":

    def logaddexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        max_, min_ = torch.max(a, b), torch.min(a, b)
        return torch.where(
            torch.isfinite(max_), (min_ - max_).exp().log1p() + max_, max_
        )


else:
    logaddexp = torch.logaddexp


def broadcast_shapes(a: List[int], b: List[int]) -> List[int]:
    scalar = torch.zeros((), device="cpu")
    tensor_a = scalar.expand(a)
    tensor_b = scalar.expand(b)
    tensor_a, tensor_b = torch.broadcast_tensors(tensor_a, tensor_b)
    return tensor_a.shape


@script
def unflatten(x: torch.Tensor, dim: int, shape: List[int]) -> torch.Tensor:
    ndim = x.dim()
    if dim < -ndim or dim > ndim - 1:
        raise RuntimeError(f"Expected dim to be between [{-ndim},{ndim-1}], got {dim}")
    dim = (dim + ndim) % ndim
    full_shape = list(x.shape)
    full_shape = full_shape[:dim] + shape + full_shape[dim + 1 :]
    return x.view(full_shape)
