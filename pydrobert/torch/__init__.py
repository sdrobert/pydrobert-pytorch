# Copyright 2019 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"
__all__ = [
    'command_line',
    'data',
    'estimators',
    'INDEX_PAD_VALUE',
    'layers',
    'training',
    'util',
]


'''The value to pad index-based tensors with

Batched operations often involve variable-width input. This value is used to
right-pad indexed-based tensors with to indicate that this element should be
ignored.

The default value (-100) was chosen to coincide with the PyTorch 1.0 default
for ``ignore_index`` in the likelihood losses
'''
INDEX_PAD_VALUE = -100
