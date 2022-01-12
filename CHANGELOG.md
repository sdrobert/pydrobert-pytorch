# Change log

## HEAD

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
- Added `util.pad_variable` and `layers.RandomShift` (#54)
- Modified `error_rate`, `prefix_error_rates` to actually compute error rates
  when non-default costs are used. Old functionality is now in
  `edit_distance` and `prefix_edit_distances` (#51)
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
- Added `pyproject.toml` for [PEP 517](https://www.python.org/dev/peps/pep-0517/).
- `tox.ini` for TOX testing
- Switched to AppVeyor for CI
- Added changelog :D
