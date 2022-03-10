# Change log

## HEAD

- `SimpleRandomSamplingWithoutReplacement` has been added as a new
  distribution.
- `EnumerateEstimator`, `ImportanceSamplingEstimator`, and
  `IndependentMetropolisHastingsEstimator` have been added as a new estimators.
- `pydrobert.torch.estimators` has been rewritten from the ground-up, with old
  functionality deprecated. Distribution-related functions have been rewritten
  as `torch.distributions.Distribution` classes implementing a
  `ConditionalStraightThrough` interface and stored in
  `pydrobert.torch.distributions`. The REINFORCE and RELAX estimators now have
  an object-oriented interface subclassing `MonteCarloEstimator` as
  `DirectEstimator` and `RelaxEstimator`, respectively.  The REBAR control
  variate is now distribution-specific and found in `pydrobert.torch.modules`.
- Bug fixes to `OptimalCompletion` and `HardOptimalCompletionDistillationLoss`
  involving batch sizes.
- Refactored code to move modules to `pydrobert.torch.modules` and functions
  to `pydrobert.torch.functional`. Deprecated `pydrobert.torch.layers` as well
  as most of the contents of `pydrobert.torch.util`.
- Added a number of modules to `pydrobert.torch.modules` to wrap functional
  API. Moved docstrings to modules.
- Fixed a problem with `warp_1d_grid`/`SpecAugment` which made it sensitive
  to the length of other elements in the batch.
- Added compatibility wrappers to avoid warnings across supported pytorch
  versions.
- Refactored code and added tests to support JIT tracing and scripting for most
  functions/modules in pytorch >= 1.8.1.
  before the next release. I'll write up documentation shortly.
- Added `pydrobert.torch.config` to store constants used in the module.
- Removed `setup.py`.
- Deleted conda recipe in prep for [conda-forge](https://conda-forge.org/).
- Compatibility/determinism fixes for 1.5.1.
- Bump minimum PyTorch version to 1.5.1. Actually testing this minimum!
- `version.py` -> `_version.py`.
- A number of modifications and additions related to decoding and language
  models, including:
  - `beam_search_advance` has been simplified, with much of the end-of-sequence
    logic punted to `BeamSearch`
  - Rejigged `SequentialLanguageModel` and `LookupLanguageModel` to be both
    simpler and compatible with decoder interfaces.
  - `ctc_greedy_search` and `ctc_prefix_search` functions have been added.
  - `ExtractableSequentialLanguageModel`, `MixableSequentialLanguageModel`,
    `BeamSearch`, and `CTCPrefixSearch` modules have been added.
  - A new documentation page on how to deal with all of that.
- Fixed bug in controller that always compared thresholds against best, not the
  last point that reset the countdown (#55)
- Added `pad_variable` and `RandomShift` (#54)
- Modified `error_rate`, `prefix_error_rates` to actually compute error rates
  when non-default costs are used. Old functionality is now in `edit_distance`
  and `prefix_edit_distances` (#51)
- Fixed bug in how padding is handled in string matching utilities.
- Fixed logic errors in `compute-torch-token-data-dir-error-rates` (#50)
- Modified frame end in `pydrobert.torch.data.transcript_to_token` and added
  some notes on the ambiguity of the conversion.
- Added some more checks and a 'fix' flag to
  `pydrobert.torch.data.validate_spect_data_set`. Entry
  `get-torch-spect-data-dir-info` now has `--fix` flag, too.

## v0.3.0

A considerable amount of refactoring occurred for this build, chiefly to get
rid of Python 2.7 support. While the functionality did not change much for this
version, we have switched from a `pkgutil`-style `pydrobert` namespace to
PEP-420-style namespaces. As a result, *this package is not
backwards-compatible with previous `pydrobert` packages!* Make sure that if any
of the following are installed, they exceed the following version thresholds:

- `pydrobert-param >0.2.0`
- `pydrobert-kaldi >0.5.3`
- `pydrobert-speech >0.1.0`

Miscellaneous other stuff:

- Type hints everywhere
- Shifted python source to `src/`
- Black-formatted remaining source
- Removed `future` dependency
- Shifted most of the configuration to `setup.cfg`, leaving only a shell
  in `setup.py` to remain compatible with Conda builds
- Added `pyproject.toml` for [PEP
  517](https://www.python.org/dev/peps/pep-0517/).
- `tox.ini` for TOX testing
- Switched to AppVeyor for CI
- Added changelog :D
