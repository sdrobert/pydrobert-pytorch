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
- `pydrobert.data`: Primarily serves as a means to manipulate speech data.

Consult the submodule docstrings for more information.

## Installation

`pydrobert-pytorch` is available through both Conda and PyPI.

``` bash
conda install -c sdrobert pydrobert-pytorch
pip install pydrobert-pytorch
```
