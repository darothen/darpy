#!/usr/bin/env python
import os
import warnings
from setuptools import setup
from textwrap import dedent

from datetime import datetime

# Use a date-stamp format for versioning
now = datetime.now()
VERSION = now.strftime("%Y-%m-%d")

NAME = 'darpy'
LICENSE = 'BSD 3-Clause'
AUTHOR = 'Daniel Rothenberg'
AUTHOR_EMAIL = 'darothen@mit.edu'
URL = 'https://github.com/darothen/darpy'
CLASSIFIERS = [
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Topic :: Scientific/Engineering',
]

DESCRIPTION = "Personal research and analysis toolkit climate data"
LONG_DESCRIPTION = """
**darpy** is a collection of analysis and plotting tools used in the research
work of Daniel Rothenberg.
"""


def _write_version_file():

    fn = os.path.join(os.path.dirname(__file__), 'darpy', 'version.py')

    version_str = dedent("""
        version = "{}"
        """)

    # Write version file
    with open(fn, 'w') as version_file:
        version_file.write(version_str.format(VERSION))

# Write version and install
_write_version_file()

setup(name=NAME,
      version=VERSION,
      license=LICENSE,
      author=AUTHOR, author_email=AUTHOR_EMAIL,
      classifiers=CLASSIFIERS,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      url=URL,
      packages=['darpy', ],
      package_data={
          'darpy': ['data/masks.nc',
                    'data/CESM_default_vars.p',
                    'data/ne_110m_ocean.shp',
                    'data/ne_110m_land.shp',
                    'data/quaas_regions.nc',
                    'data/landsea.nc'],
      },
      scripts=['scripts/calc_aerosol',
               'scripts/quick_plot',
               'scripts/simple_cat',
               'scripts/interp_pres',
               'scripts/interp_field',
               'scripts/reduce_dims',
               'scripts/global_avg', ],
)
