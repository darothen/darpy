from utilities import extract_variable as extr_drv
from functools import partial

from case_setup import *

extract_variable = partial(extr_drv,
                           years_omit=3, re_extract=True,
                           act_cases=CASES_ACT,
                           aer_cases=CASES_AER )

if __name__ == "__main__":

    #############################
    ## BLOCK 1 - basic extraction

    mapping = ("newvar", "oldvar", "out_suffix",
               "attributes", "lev_bnds", "cdo_method", "ncap_str")

    proc_vars = [
        ("CLDLOW", "CLDLOW", "annavg",
            {"long_name": "Mean annual low cloud (850-960 mb)"},
            None, ["timmean"], ),
        ("PRECL", "PRECL", "annavg",
            {"long_name": "Mean annual large-scale precipitation"},
            None, ["timmean"], ),
        ("AREL_LOW", "AREL", "annavg",
            {"long_name": "Mean annual low cloud (850-960 mb) average droplet effective radius"},
            (23, 27), ["timmean", "vertmean"], ),
        ("AREL_960", "AREL", "annavg",
            {"long_name": "Mean annual low cloud (960 mb) average droplet effective radius"},
            (27, ), ["timmean",], ),
    ]

    for var in proc_vars:
        kw_args = { key: val for key, val in zip(mapping, var) }
        #print kw_args
        extract_variable(**kw_args)
