[![tox](https://github.com/sdrobert/pydrobert-pytorch/actions/workflows/tox.yml/badge.svg)](https://github.com/sdrobert/pydrobert-pytorch/actions/workflows/tox.yml)
[![Documentation Status](https://readthedocs.org/projects/pydrobert-pytorch/badge/?version=latest)](https://pydrobert-pytorch.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# pydrobert-pytorch

PyTorch utilities for Machine Learning. This is an eclectic mix of utilities
that I've used in my various projects. There is a definite leaning towards
speech, specifically end-to-end ASR. The primary benefit `pydrobert-pytorch`
has over other packages is modularity: you can pick and choose the
functionality you desire without subscribing to an entire ecosystem. You can
find out more about what the package offers in the documentation links below.

**This is student-driven code, so don't expect a stable API. I'll try to use
semantic versioning, but the best way to keep functionality stable is by
pinning the version in the requirements or by forking.**

## Documentation

- [Latest](https://pydrobert-pytorch.readthedocs.io/en/latest/)
- [v0.3.0](https://pydrobert-pytorch.readthedocs.io/en/v0.3.0/)

## Installation

`pydrobert-pytorch` is available through both Conda and PyPI.

``` bash
conda install -c sdrobert pydrobert-pytorch
pip install pydrobert-pytorch
```

## Licensing and How to Cite

Please see the [pydrobert page](https://github.com/sdrobert/pydrobert) for more
details on how to cite this package.

Implementations of
`pydrobert.torch._img.{polyharmonic_spline,sparse_image_warp}` are based off
Tensorflow's codebase, which is Apache 2.0 licensed.

Implementations of
`pydrobert.torch._compat.{broadcast_shapes,TorchVersion,one_hot}` were directly
taken from the PyTorch codebase. A number of methods and functions in
`pydrobert.torch._straight_through` modify PyTorch code (see the file for more
info). PyTorch has a BSD-style license which can be found in the file
`LICENSE_pytorch.txt`.

The implementation of `pydrobert.torch._compat.check_methods` was taken
directly from the CPython codebase, Copyright 2007 Google with additional
notices at <https://docs.python.org/3/copyright.html?highlight=copyright>.

The file `pydrobert.torch._textgrid,py` was taken with some minor modifications
from
[nltk_contrib](https://github.com/nltk/nltk_contrib/blob/95d1806e2f4e89e960b76a685b1fba2eaa7d5142/nltk_contrib/textgrid.py#L1).
It is Apache 2.0-licensed, with the specific license text saved to
`LICENSE_nltk.txt`.
