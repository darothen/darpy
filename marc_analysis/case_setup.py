"""
Script defining where the data is located and all the case combinations

"""
import os

## OS X
CASES_ACT = ["arg_min_smax", "arg_comp", ]
CASES_AER = ["F2000", "F1850"]
EMIS_MAP = { 'F2000': "PD", 'F1850': "PI" }

DATA_ROOT_DIR = "/Users/daniel/Desktop/MARC_AIE"
WORK_DIR = os.path.join(DATA_ROOT_DIR, "work")
VAR_ARCHIVE = "saved_vars"

AMWG_SRC = ""
AMWG_OUT = ""

## LEGION
# CASES_ACT = ["arg_comp", "arg_min_smax", "pcm_comp", "pcm_min_smax",
#              "nenes_comp", ]
# CASES_AER = ["F2000", "F1850"]
# EMIS_MAP = { 'F2000': "PD", 'F1850': "PI" }

# DATA_ROOT_DIR = "/net/s001/volume1/storage01/darothen/CESM/MARC_AIE"
# DATA_ROOT_DIR2 = "/storage02/darothen/CESM/MARC_AIE"
# WORK_DIR = os.path.join(DATA_ROOT_DIR2, "work")
# VAR_ARCHIVE = "saved_vars"

# AMWG_SRC = "/home/darothen/models/CESM/amwg_diag_20140804/"
# AMWG_OUT = os.path.join(DATA_ROOT_DIR2, "AMWG")

# CERES_DIR = os.path.join(DATA_ROOT_DIR2, "satellite", "CERES")
# MODIS_DIR = os.path.join(DATA_ROOT_DIR2, "satellite", "MODIS", "order3")
# BENNARTZ_DIR = os.path.join(DATA_ROOT_DIR2, "satellite", "bennartz")

## CESM/MARC metadata and helper info
lev_stdname = "atmosphere_hybrid_sigma_pressure_coordinate"

def case_path(act, aer):
    return os.path.join(DATA_ROOT_DIR, aer, act)
