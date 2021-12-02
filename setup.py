"""Project: ModelMetricUncertainty

Created: 2021/09/24

Description:
    setup script to install ModelMetricUncertainty package.

Authors:
    Ralph Urlus [rurlus.dev@gmail.com]

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""

from pathlib import Path
from setuptools import setup

NAME = 'mmur'

MAJOR = 0
REVISION = 1
PATCH = 0
DEV = True

# note: also update README.rst

VERSION = '{major}.{revision}.{patch}'.format(major=MAJOR, revision=REVISION, patch=PATCH)
FULL_VERSION = VERSION
if DEV:
    FULL_VERSION += '.dev'

# read the contents of readme file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

def write_version_py(filename: str = f'{NAME}/version.py') -> None:
    """Write package version to version.py.

    This will ensure that the version in version.py is in sync with us.

    Parameters
    ----------
    filename : str
        the path the file to write the version.py

    """
    # Do not modify the indentation of version_str!
    version_str = """\"\"\"THIS FILE IS AUTO-GENERATED BY ModelMetricUncertaintyResearch SETUP.PY.\"\"\"

name = '{name!s}'
version = '{version!s}'
full_version = '{full_version!s}'
release = {is_release!s}
"""

    with open(filename, 'w') as version_file:
        version_file.write(
            version_str.format(name=NAME.lower(), version=VERSION, full_version=FULL_VERSION, is_release=not DEV)
        )


if __name__ == '__main__':
    write_version_py()
    setup(version=FULL_VERSION)
