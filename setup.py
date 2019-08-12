from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from codecs import open
from os import path
from setuptools import setup

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"

PWD = path.abspath(path.dirname(__file__))
with open(path.join(PWD, 'README.md'), encoding='utf-8') as readme_file:
    LONG_DESCRIPTION = readme_file.read()

SETUP_REQUIRES = ['setuptools_scm']
if {'pytest', 'test', 'ptr'}.intersection(sys.argv):
    SETUP_REQUIRES += ['pytest-runner']


def main():
    setup(
        name='pydrobert-pytorch',
        description='PyTorch utilities for ML, specifically speech',
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        use_scm_version=True,
        zip_safe=False,
        url='https://github.com/sdrobert/pydrobert-pytorch',
        author=__author__,
        author_email=__email__,
        license=__license__,
        packages=['pydrobert', 'pydrobert.torch'],
        classifiers=[
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        install_requires=[
            'numpy', 'future', 'torch', 'param'
        ],
        setup_requires=SETUP_REQUIRES,
        tests_require=[
            'pytest', 'scipy', 'optuna',
        ],
        entry_points={
            'console_scripts': [
                'ctm-to-torch-token-data-dir = pydrobert.torch.command_line:'
                'ctm_to_torch_token_data_dir',
                'get-torch-spect-data-dir-info = pydrobert.torch.command_line:'
                'get_torch_spect_data_dir_info',
                'trn-to-torch-token-data-dir = pydrobert.torch.command_line:'
                'trn_to_torch_token_data_dir',
                'torch-token-data-dir-to-ctm = pydrobert.torch.command_line:'
                'torch_token_data_dir_to_ctm',
                'torch-token-data-dir-to-trn = pydrobert.torch.command_line:'
                'torch_token_data_dir_to_trn',
                'compute-torch-token-data-dir-error-rates = pydrobert.torch.'
                'command_line:compute_torch_token_data_dir_error_rates',
            ]
        },
    )


if __name__ == '__main__':
    main()
