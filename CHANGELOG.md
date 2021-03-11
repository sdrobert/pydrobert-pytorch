# v0.3.0

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
