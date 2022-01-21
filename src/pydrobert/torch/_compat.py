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

from typing import List
import torch

__all__ = ["pad_sequence"]

if torch.__version__ < "1.8.0":
    from ._jit import script

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


else:
    pad_sequence = torch.nn.utils.rnn.pad_sequence

