from setuptools import setup

setup(
    setup_requires=["setuptools_scm"],
    use_scm_version={"write_to": "src/pydrobert/torch/version.py"},
)
