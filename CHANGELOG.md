# Change log

## v0.4.0

- Added `pydrobert.torch.argcheck` and standardized module argument checking a
  bit
- `parse_arpa` can handle log-probs in scientific notation (e.g. `1e4`)
- Added `best_is_train` flag to `TrainingController.update_for_epoch`
- Refactored `get-torch-spect-data-dir-info` to be faster
- `subset-torch-spect-data-dir` command has been added
- `print-torch-{ali,ref}-data-dir-length-moments` commands have been added
- `LookupLanguageMode.prob_list` has been renamed to `prob_dicts`
- Added `ShallowFusionLanguageModel`, `ExtractableShallowFusionLanguageModel`,
  and `MixableShallowFusionLanguageModel`
- Slicing and chunking modules `SliceSpectData`, `ChunkBySlices`, and
  `ChunkTokenSequenceBySlices`, as well as the command
  `chunk-torch-spect-data-dir`, which puts them together.
- Code for handling TextGrid files, including the functions `read_textgrid` and
  `write_textgrid`, as well as the commands `torch-token-data-dir-to-textgrids`
  and `textgrids-to-torch-token-data-dir`.
- Commands for switching between ref and ali format:
  `torch-ali-data-dir-to-torch-token-data-dir` and
  `torch-token-data-dir-to-torch-ali-data-dir`.
- Added py 3.10 support; removed py 3.6 support.
- Initial (undocumented) support for
  [PyTorch-Lightning](https://www.pytorchlightning.ai/) in
  `pydrobert.torch.lightning` submodule. Will document when I get some time.
- Refactored much of `pydrobert.torch.data`. Best just to look at the API.
  `ContextWindowEvaluationDataLoader`, `ContextWindowTrainingDataLoader`,
  `SpectEvaluationDataLoader`, `SpectTrainingDataLoader`, `DataSetParams`,
  `SpectDataSetParams`, and `ContextWindowDataSetParams` are now deprecated.
  The data loaders have been simplified to `ContextWindowDataLoader` and
  `SpectDataLoader`. Keyword arguments (like `shuffle`) now control their
  behaviour. The `*DataSetParams` have been renamed `*DataLoaderParams` with
  some of the parameters moved around. Notably, `LangDataParams` now stores
  `sos`, `eos`, and `subset_ids` parameters, from which a number of parameter
  objects inherit. `SpectDataLoaderParams` inherits from
  `LangDataLoaderParams`, which in turn inherits from
  `DynamicLengthDataLoaderParams`. The latter allows the loader's batch
  elements to be bucketed by length using the new `BucketBatchSampler`. It and
  a number of other samplers inherit from `AbstractEpochSampler` to help
  facilitate the simplified loaders and better resemble the PyTorch API.
  Mean-variance normalization of features is possible through the loaders and
  the new `MeanVarianceNormalization` module. `LangDataSet` and
  `LangDataLoader` have been introduced to facilitate language mdoel training.
  Finally, loaders (and samplers) are compatible with `DistributedDataParallel`
  environments.
- Mean-variance statistics for normalization may be estimated from a data
  partition using the command `compute-mvn-stats-for-torch-feat-data-dir`.
- Added `torch-spect-data-dir-to-wds` to convert a data dir to a
  [WebDataset](https://github.com/webdataset/webdataset).
- Changed method of constructing random state in `EpochRandomSampler`.
  Rerunning training on this new version with the same seed will likely result
  in different results from the old version!
- `FeatureDeltas` now a module, in case you want to compute them online rather
  than waste disk space.
- Added `PadMaskedSequence`.
- Added  `FillAfterEndOfSequence`.
- Added `binomial_coefficient`, `enumerate_binary_sequences`,
  `enumerate_vocab_sequences`, and
  `enumerate_binary_sequences_with_cardinality`.
- Docstrings updated to hopefully be clearer. Use "Call Parameters" and
  "Returns" sections for pytorch modules.
- readthedocs updated.
- Fixed up formatting of CLI help documentation.
- Data sets can now initialize some of their parameters with the values in
  their associated param containers. For example, `sos` and `eos` are now
  set in `SpectDataSet` by passing an optional `SpectDataParam` instance. The
  old method (by argument) is now deprecated.
- Renamed `DataSetParams` to `DataLoaderParams` and deprecated former naming
  to better mesh with their use in data loaders.
- Moved `pydrobert.torch.util.parse_arpa_lm` to `pydrobert.torch.data`
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
  to `pydrobert.torch.functional`.
- Deprecated `pydrobert.torch.layers` and `pydrobert.torch.util`.
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
  - `beam_search_advance` and `random_walk_advance` have been simplified, with
    much of the end-of-sequence logic punted to their associated modules.
  - Rejigged `SequentialLanguageModel` and `LookupLanguageModel` to be both
    simpler and compatible with decoder interfaces.
  - `ctc_greedy_search` and `ctc_prefix_search_advance` functions have been
    added.
  - `ExtractableSequentialLanguageModel`, `MixableSequentialLanguageModel`,
    `BeamSearch`, `RandomWalk`, and `CTCPrefixSearch` modules have been added.
  - A `SequentialLanguageModelDistribution` wrapping `RandomWalk` which
    implements PyTorch's `Distribution` interface. Language models now work
    with estimators!
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
