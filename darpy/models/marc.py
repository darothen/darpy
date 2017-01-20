""" Helper functions for handling MARC output. """

import logging
logger = logging.getLogger()

import pkg_resources
import pickle
import warnings

#: Aerosol modes
all_modes = ['NUC', 'AIT', 'ACC', 'OC', 'MOS', 'BC', 'MBS',
             'DST01', 'DST02', 'DST03', 'DST04',
             'SSLT01', 'SSLT02', 'SSLT03', 'SSLT04', ]

#: Default aerosol mode color scheme
mode_colors = {
    # Sulfate - red
    'NUC': '#fee0d2',
    'AIT': '#fc9272',
    'ACC': '#ef3b2c',
    # Organic - green
    'OC': '#4daf4a',
    'MOS': '#ffff33', # green + red = yellow
    # Black - blue,
    'BC': '#377eb8',
    'MBS': '#984ea3', # blue + red = purple
    # Dust - shades of brown
    'DST01': '#f6e8c3', 'DST02': '#dfc27d',
    'DST03': '#bf812d', 'DST04': '#8c510a',
    # Sea Salt - shades of teal
    'SSLT01': '#c7eae5', 'SSLT02': '#80cdc1',
    'SSLT03': '#35978f', 'SSLT04': '#01665e',
}


#######################################################################
## Utility functions

#: Placeholder for CESM var list
_CESM_VARS = None

def get_cesm_vars(reextract=True):
    """ Load in the saved dictionary of CESM vars. """

    global _CESM_VARS

    if reextract or (_CESM_VARS is None):
        try:
            CESM_defaults_fn = pkg_resources.resource_filename(
                "marc_analysis", "data/CESM_default_vars.p"
            )
            with open(CESM_defaults_fn, 'rb') as f:
                CESM_vars_dict = pickle.load(f)

            _CESM_VARS = CESM_vars_dict

        except FileNotFoundError:
            warnings.warn("Couldn't find CESM_default_vars archive")
            _CESM_vars_dict = None
    else:
        CESM_vars_dict = _CESM_VARS

    return CESM_vars_dict

def _extract_cesm_vars(nc_filename, out="CESM_default_vars.p"):
    """ Extract all the CESM variables and their metadata from
    a given dataset. """
    import netCDF4 as nc

    d = nc.Dataset(nc_filename, 'r')
    all_vars = { key: { att: data.__dict__[att] for att in data.ncattrs() } \
                 for key, data in d.variables.items() }
    logger.debug("... found %d variables" % (len(all_vars), ))

    with open(out, 'wb') as f:
        pickle.dump(all_vars, f, protocol=2)
    logger.debug("done. Saved to %s" % out)

def print_cesm_vars():
    """ Print a list of all the available CESM vars. """
    if _CESM_VARS is None:
        _ = get_cesm_vars()

    for k, v in sorted(_CESM_VARS.items()):

        logger.info(k)
        if ('long_name' in v) and ('units' in v) :
            logger.info("{} ({})".format(
                v['long_name'], v['units']
            ))
        else:
            logger.info("")
