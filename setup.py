#!/usr/bin/env python
import os
from typing import List
from setuptools import setup, find_packages

# Package name used to install via pip (shown in `pip freeze` or `conda list`)
MODULE_NAME = 'statistical_simulation_tools'

# How this module is imported in Python (name of the folder inside `src`)
MODULE_NAME_IMPORT = 'statistical_simulation_tools'

# Repository name
REPO_NAME = 'statistical-simulation-tools'


def get_version() -> str:
    with open(os.path.join('src', MODULE_NAME_IMPORT, 'resources', 'VERSION')) as f:
        return f.read().strip()


SETUP_ARGS = {
    "name": MODULE_NAME,
    "description": "Tools for fitting, and validating distributions",
    "version": get_version(),
    'package_dir': {'': 'src'},
    'packages': find_packages('src'),
    'python_requires': '>=3.9,<3.10',
    "install_requires": ['numpy', 'scipy', 'matplotlib'],
}


if __name__ == '__main__':
    setup(**SETUP_ARGS)
