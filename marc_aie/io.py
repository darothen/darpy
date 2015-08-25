
import os
import numpy as np
from xray import open_dataset, decode_cf

from . case_setup import WORK_DIR

__all__ = ['load_variable', ]

##################

def load_variable(var, act, aer, suffix="", save_dir=WORK_DIR,
                  method='xray', fix_times=True, extr_kwargs={}):
    """ Interface for loading a variable into memory, using
    either iris or xray 

    Parameters
    ----------
    var : Var
        The variable meta information
    act, aer : strings
        The activation and aerosol emissions case
    var_name : string
        The name of the variable to load
    suffix : string (optional)
        Particular file output suffix to look for
    save_dir : string
        Path to save directory; defaults to case WORK_DIR.
    method : string
        Choose between 'iris' or 'xray'
    fix_times : bool
        Correct the timestamps to the middle of the bounds
        in the variable metadata (CESM puts them at the right
        boundary which sucks!)
    extr_kwargs : dict
        Additional keyword arguments to pass to the extractor

    """

    var_name = var.varname

    # build filename
    fn = "%s_%s_%s" % (act, aer, var_name)
    if suffix: 
        fn += "_%s" % suffix
    fn += ".nc"

    print("Loading %s" % var) 
    print("from %s/%s" % (save_dir, fn))

    path_to_file = os.path.join(save_dir, fn)

    if method == "iris":

        raise NotImplementedError("`iris` deprecated with Python 3")

        # cf = lambda c : c.var_name == var_name
        # cubes = iris.load(path_to_file, iris.Constraint(cube_func=cf),
        #                   **extr_kwargs)
        #
        # if not cubes:
        #     raise RuntimeError("Could not find '%s' in cube" % var_name)
        #
        # assert len(cubes) == 1
        #
        # c = cubes[0]
        #
        # if fix_times:
        #     times = c.coord('time')
        #     assert hasattr(times, 'bounds')
        #
        #     bnds = times.bounds
        #     mean_times = np.mean(bnds, axis=1)
        #
        #     times.points = mean_times
        #
        # return c

    elif method == "xray":

        ds = open_dataset(path_to_file,
                               decode_cf=False,
                               **extr_kwargs)

        # Fix time unit, if necessary
        interval, timestamp = ds.time.units.split(" since ")
        timestamp = timestamp.split(" ")
        yr, mm, dy = timestamp[0].split("-")

        yr = int(yr)
        if yr < 1650: yr = str(2000)

        # Re-construct at Jan 01, 2000 and re-set
        timestamp[0] = "-".join([yr, mm, dy])
        new_units = " ".join([interval, "since"] + timestamp)
        ds.time.attrs['units'] = new_units

        if fix_times:
            assert hasattr(ds, 'time_bnds')
            bnds = ds.time_bnds.values
            mean_times = np.mean(bnds, axis=1)

            ds.time.values = mean_times

        # Lazy decode CF
        ds = decode_cf(ds)

        return ds
