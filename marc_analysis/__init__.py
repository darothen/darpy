"""
A package for analzying an aerosol-indirect effect 
experiment with different activation parameterizations
in the CESM-MARC. 

"""

from __future__ import absolute_import, division

# Set up some default imports to bring some of the 
# tools to the marc_aie namespace.
# from . case_setup import *
from . experiment import *
from . extract import *
from . io import *
from . var import *
from . plot import *
from . convert import *
from . utilities import *

# Import some resources

# 1) Masks for land/ocean
try:
    import pkg_resources
    import xray
    import warnings

    _masks_fn = pkg_resources.resource_filename("marc_analysis",
                                                "data/masks.nc")
    masks = xray.open_dataset(_masks_fn, decode_cf=False,
                              mask_and_scale=False,
                              decode_times=False).squeeze()
except RuntimeError:
    warnings.warn("Unable to locate `masks` resource.")
    masks = None
