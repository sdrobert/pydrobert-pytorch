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

"""Functions and classes which interface with :mod:`pytorch_lightning`

This functionality is WIP.

See `scpc <https://github.com/sdrobert/scpc>`_ for a working example.

Raises
------
ImportError
    If :mod:`pytorch_lightning` is not installed.
"""

from ._pl_data import (
    LitDataModule,
    LitDataModuleParams,
    LitDataModuleParamsMetaclass,
    LitSpectDataModule,
    LitSpectDataModuleParams,
)

__all__ = [
    "LitDataModule",
    "LitDataModuleParams",
    "LitDataModuleParamsMetaclass",
    "LitSpectDataModule",
    "LitSpectDataModuleParams",
]
