# pydrobert-pytorch

PyTorch utilities for Machine Learning. This is an eclectic mix of utilities
that I've used in my various projects, but have been tailored to be as generic
as possible.

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

Consult the submodule docstrings for more information.

## Installation

`pydrobert-pytorch` is available through both Conda and PyPI.

``` bash
conda install -c sdrobert pydrobert-pytorch
pip install pydrobert-pytorch
```
