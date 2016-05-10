
import os
import re
from glob import glob

import numpy as np
import pandas as pd
import xarray

import logging
logger = logging.getLogger()

__all__ = ['load_variable', 'load_netcdfs' ]

# Note - must compile with re.VERBOSE option; can't use advanced
# string formatting because of specified field lengths in regex!
OUTPUT_FN_REGEX = """
    (?P<name>\w+)    # Case name
    .
    (?P<comp>%s)\d?  # Model component - format wildcard, string
    .
    h(?P<hist>%1d)   # History tape number - format wildcard, int
    .
    (?P<year>\d{4})  # Year
    -
    (?P<month>\d{2}) # Month
    -?
    (?P<day>\d{2})?  # Day, if present
    -?
    (?P<time>\d{5})? # Timestamp, if present
    (.nc)$           # file suffix (netcdf)
"""
COMP_MAP = {
    'atm': 'cam',
    'rof': 'rtm',
    'ocn': 'pop',
    'lnd': 'clm',
}
def _format_regex(comp='cam', hist=0):
    """ Create matching regex with experiment output details hardcoded. """
    return OUTPUT_FN_REGEX % (comp, hist)

# TODO: Generate regex for matching output files
# def _build_regex(name=None, comp=None, hist=None, year=None, month=None,
#                  day=None, time=None):
#     """ Build a regular expression to match some subset of of default CESM
#     output tapes. """
#
#     regex = ""
#
#     name_str = "(?P<name>{})"
#     if name is None:
#         name_str = name_str.format("\w+")
#     else:
#         name_str = name_str.format("({})".format(name))
#
#     comp_str = "(?P<comp>{})\d?"
#     if comp is None:
#         comp_str = comp_str.format("")


def _match_file_list(output_dir, regex_str, years_omit=0):
    """ Match the files in a specified directory against a regex
    and process them, filtering based on how many years to omit.

    Note - this method serves as a kernel for a future expansion
    to analyze what files are present in the dataset, hence why
    it is overcomplicated.
    """

    # Compile the regular expression for matching
    comp_re = re.compile(regex_str, re.VERBOSE)

    # Process the files into a DataFrame
    all_files = os.listdir(output_dir)

    # Analyze the filenames using the passed regular expression
    matches = [ comp_re.match(f) for f in all_files ]
    groups, valid_files = zip(*[ (m.groupdict(), m.string) \
                                 for m in matches if m is not None ])
    groups = list(groups)
    valid_files = list(valid_files)

    # Postprocess - convert Nones, record monthly or sub-monthly
    for g in groups:
        g['monthly'] = False
        if g['day'] is None:
            g['day'] = 1; g['monthly'] = True
        if g['time'] is None: g['time'] = 0; g['monthly'] = True

    files_df = pd.DataFrame(groups)
    files_df['filename'] = valid_files

    # Postprocess - convert strings to ints
    for key in ['year', 'month', 'day', 'time', 'hist']:
        files_df[key] = files_df[key].apply(int)

    # Postprocess - sort
    files_df = files_df.sort_values(by=['hist', 'year', 'month', 'day', 'time'])

    ###################################################################

    # Determine 0th year and extract all years beyond 0th year + years_omit
    year0 = files_df.iloc[0].year
    year_start = year0 + years_omit
    filtered_files = (files_df[files_df['year'] >= year_start]
                         .filename
                         .tolist())

    return filtered_files


##################

def load_netcdfs(files, dim='time', transform_func=None, open_kws={}):
    """ Load/pre-process a set of netCDF files and concatenate along the
    given dimension. Useful for loading a portion of a large dataset into
    memory directly, even when that dataset spans many different files.

    This is based on the idiom provided in the xarray documentation at
    http://xarray.readthedocs.org/en/stable/io.html

    .. note:
        This is a poor man's functional version of xarray.open_mfdataset()


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
        if yr < 1650:
            yr = str(2001)

        # Re-construct at Jan 01, 2001 and re-set
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
