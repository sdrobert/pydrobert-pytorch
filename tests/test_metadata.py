"""Test package metadata"""

import pydrobert.torch


def test_version():
    assert pydrobert.torch.__version__ != "inplace"
