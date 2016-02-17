
from glob import glob
import numpy as np
import xarray

import logging
logger = logging.getLogger()

__all__ = ['load_variable', 'load_netcdfs' ]

##################

def load_netcdfs(files, dim='time', transform_func=None, open_kws={}):
    """ Load/pre-process a set of netCDF files and concatenate along the
    given dimension. Useful for loading a portion of a large dataset into
    memory directly, even when that dataset spans many different files.

    This is based on the idiom provided in the xarray documentation at
    http://xarray.readthedocs.org/en/stable/io.html

    Parameters
    ----------
    files : str
        A string indicating either a single filename or a glob pattern
        to match multiple files.
    dim : str
        Name of dimensions to concatenate on; defaults to 'time'
    transform_func : func, optional
        Callback function to apply to each file opened before concatenation.
    open_kws: dict, optional
        Additional keyword arguments to apply when opening dataset

    Returns
    -------
    An xarray.Dataset with the transformed, subsetted data from the
    requested files.

    """

    def process_file(path):

        logger.debug("load_netcdfs: opening {}".format(path))

        # use a context manager, to ensure the file gets closed after use
        with xarray.open_dataset(path, **open_kws) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    paths = sorted(glob(files))
    logger.debug("load_netcdfs: found {} paths".format(len(paths)))

    datasets = [process_file(p) for p in paths]
    combined = xarray.concat(datasets, dim)

    return combined


def load_variable(var_name, path_to_file,
                  method='xarray', fix_times=True, extr_kwargs={}):
    """ Interface for loading an extracted variable into memory, using
    either iris or xarray. If `path_to_file` is instead a raw dataset,
    then the entire contents of the file will be loaded!

    Parameters
    ----------
    var_name : string
        The name of the variable to load
    path_to_file : string
        Location of file containing variable
    method : string
        Choose between 'iris' or 'xarray'
    fix_times : bool
        Correct the timestamps to the middle of the bounds
        in the variable metadata (CESM puts them at the right
        boundary which sucks!)
    extr_kwargs : dict
        Additional keyword arguments to pass to the extractor

    """

    logger.info("Loading %s from %s" % (var_name, path_to_file))

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

    elif method == "xarray":

        ds = xarray.open_dataset(path_to_file, decode_cf=False, **extr_kwargs)

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
        ds = xarray.decode_cf(ds)

        return ds
