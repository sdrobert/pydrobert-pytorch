[![Build Status](https://travis-ci.com/sdrobert/pydrobert-pytorch.svg?branch=master)](https://travis-ci.com/sdrobert/pydrobert-pytorch)
[![Documentation Status](https://readthedocs.org/projects/pydrobert-pytorch/badge/?version=latest)](https://pydrobert-pytorch.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# pydrobert-pytorch

PyTorch utilities for Machine Learning. This is an eclectic mix of utilities
that I've used in my various projects, but have been tailored to be as generic
as possible.

**This is student-driven code, so don't expect a stable API. I'll try to use
semantic versioning, but the best way to keep functionality stable is by
pinning the version in the requirements or by forking.**

## Overview

Functionality is split by submodule. They include

- `pydrobert.torch.estimators`: Implements a number of popular gradient
  estimators in ML literature. Useful for RL tasks, or anything that needs
  discrete samples.
- `pydrobert.torch.training`: Utilities that should be useful to most model
  training loops, even the most esoteric. `TrainingStateController` can be used
  to persist model and optimizer states across runs, and manage
  non-determinism.
- `pydrobert.torch.data`: Primarily serves as a means to manipulate speech
  data. It contains subclasses of `torch.utils.data.DataLoader` for both
  random and sequential access of speech data, as well as examples of how to
  use them. `pydrobert.torch.data` also contains functions for transducing back
  and forth between tensors and transcriptions. In particular, this package
  comes with command line hooks for converting to and from
  [NIST sclite](http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm)
  file formats. Feature data and senone alignments from
  [Kaldi](http://kaldi-asr.org/) can be converted to this format using the
  command line hooks from
  [pydrobert-kaldi](https://github.com/sdrobert/pydrobert-kaldi).

## Documentation

- [Latest](https://pydrobert-pytorch.readthedocs.io/en/latest/)
- [Stable](https://pydrobert-pytorch.readthedocs.io/en/stable/)

## Installation

`pydrobert-pytorch` is available through both Conda and PyPI.

``` bash
conda install -c sdrobert pydrobert-pytorch
pip install pydrobert-pytorch
```
