# Copyright 2022 Sean Robertson
#
# Code for broadcast_shapes was adapted from PyTorch
# https://github.com/pytorch/pytorch/blob/2367face24afb159f73ebf40dc6f23e46132b770/torch/functional.py
# Code for TorchVersion was taken directly from PyTorch
# https://github.com/pytorch/pytorch/blob/b737e09f60dd56dbae520e436648e1f3ebc1f937/torch/torch_version.py
# See LICENSE_pytorch in project root directory for PyTorch license.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Iterable, List, Optional, Tuple, Union, NamedTuple, Set

import torch
import pydrobert.torch.config as config


__all__ = [
    "broadcast_shapes",
    "jit_isinstance",
    "linalg_solve",
    "meshgrid",
    "pad_sequence",
    "script",
    "SpoofPackedSequence",
    "trunc_divide",
]


# to avoid some scripting issues with torch.utils.nn.PackedSequence
class SpoofPackedSequence(NamedTuple):
    data: torch.Tensor
    batch_sizes: torch.Tensor
    sorted_indices: Optional[torch.Tensor]
    unsorted_indices: Optional[torch.Tensor]


if config.USE_JIT:
    script = torch.jit.script
else:
    try:
        script = torch.jit.script_if_tracing
    except AttributeError:

        def script(obj, *args, **kwargs):
            return obj


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

if _v < "1.8.0":

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

    @torch.jit.unused
    def _jit_isinstance(obj: Any, x: type) -> bool:
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
                    _jit_isinstance(*y) for y in zip(obj, args)
                )
            else:
                assert len(args) == 1
                return all(_jit_isinstance(o, args[0]) for o in obj)
        elif origin is Union:
            args = x.__args__
            return any(_jit_isinstance(obj, y) for y in args)
        return False

    def jit_isinstance(obj: Any, x: type) -> bool:
        if torch.jit.is_scripting():
            return isinstance(obj, x)
        else:
            return _jit_isinstance(obj, x)


else:
    pad_sequence = torch.nn.utils.rnn.pad_sequence
    linalg_solve = torch.linalg.solve
    jit_isinstance = torch.jit.isinstance


@torch.no_grad()
def broadcast_shapes(a: List[int], b: List[int]) -> List[int]:
    scalar = torch.zeros((), device="cpu")
    tensor_a = scalar.expand(a)
    tensor_b = scalar.expand(b)
    tensor_a, tensor_b = torch.broadcast_tensors(tensor_a, tensor_b)
    return tensor_a.shape


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

    def meshgrid(*tensors) -> Tuple[torch.Tensor, ...]:
        return torch.meshgrid(*tensors, indexing="ij")
