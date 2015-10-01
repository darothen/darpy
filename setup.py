#!/usr/bin/env python
import os
import warnings
from setuptools import setup
from textwrap import dedent

MAJOR, MINOR, MICRO = 0, 1, 0
DEV = True
VERSION = "{}.{}.{}".format(MAJOR, MINOR, MICRO)

# Correct versioning with git info if DEV
if DEV:
    import subprocess

    pipe = subprocess.Popen(
        ['git', "describe", "--always", "--match", "v[0-9]*"],
        stdout=subprocess.PIPE)
    so, err = pipe.communicate()

    if pipe.returncode != 0:
        # no git or something wrong with git (not in dir?)
        warnings.warn("WARNING: Couldn't identify git revision, using generic version string")
        VERSION += ".dev"
    else:
        git_rev = so.strip()
        git_rev = git_rev.decode('ascii') # necessary for Python >= 3

        VERSION += ".dev-{}".format(git_rev)


NAME = 'marc_analysis'
LICENSE = 'BSD 3-Clause'
AUTHOR = 'Daniel Rothenberg'
AUTHOR_EMAIL = 'darothen@mit.edu'
URL = 'https://github.com/darothen/marc_analysis'
CLASSIFIERS = [
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Topic :: Scientific/Engineering',
]

DESCRIPTION = "Analysis toolkit for CESM/MARC simulations"
LONG_DESCRIPTION = """
**marc_analysis** is a suite of Python tools to help automate many
tasks related to analyzing CESM/MARC experiment output. It comes with
helper classes for extracting default and value-added/post-processed
variables from the simulation archive and for managing experimental
setups.
"""


def _write_version_file():

    fn = os.path.join(os.path.dirname(__file__), 'marc_analysis', 'version.py')

    version_str = dedent("""
        version = {}
        """)

    # Write version file
    with open(fn, 'w') as version_file:
        version_file.write(version_str.format(VERSION))

# Write version and install
_write_version_file()

setup(name=NAME,
      license=LICENSE,
      author=AUTHOR, author_email=AUTHOR_EMAIL,
      classifiers=CLASSIFIERS,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      url=URL)
