
import numpy as np
from xray import open_dataset, decode_cf

__all__ = ['load_variable', ]

##################

def load_variable(var_name, path_to_file,
                  method='xray', fix_times=True, extr_kwargs={}):
    """ Interface for loading an extracted variable into memory, using
    either iris or xray. If `path_to_file` is instead a raw dataset,
    then the entire contents of the file will be loaded!

    Parameters
    ----------
    var_name : string
        The name of the variable to load
    path_to_file : string
        Location of file containing variable
    method : string
        Choose between 'iris' or 'xray'
    fix_times : bool
        Correct the timestamps to the middle of the bounds
        in the variable metadata (CESM puts them at the right
        boundary which sucks!)
    extr_kwargs : dict
        Additional keyword arguments to pass to the extractor

    """

    print("Loading %s from %s" % (var_name, path_to_file))

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

        ds = open_dataset(path_to_file, decode_cf=False, **extr_kwargs)

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
