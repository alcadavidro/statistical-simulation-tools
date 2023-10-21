#!/usr/bin/env python
import os
from typing import List
from setuptools import setup, find_packages

MODULE_NAME = 'statistical_simulation_tools'

MODULE_NAME_IMPORT = 'statistical_simulation_tools'

REPO_NAME = 'statistical-simulation-tools'


def get_version() -> str:
    with open(os.path.join('src', MODULE_NAME_IMPORT, 'resources', 'VERSION')) as f:
        return f.read().strip()
    
def parse_requirements(filename: str) -> List[str]:
    with open(filename, 'r') as pip:
        return [package.strip() for package in pip if not package.strip()]


SETUP_ARGS = {
    "name": MODULE_NAME,
    "description": "Tools for fitting, and validating distributions",
    "version": get_version(),
    'package_dir': {'': 'src'},
    'packages': find_packages('src'),
    'python_requires': '>=3.9,<3.10',
    "install_requires": parse_requirements('requirements.txt'),
}


if __name__ == '__main__':
    setup(**SETUP_ARGS)
