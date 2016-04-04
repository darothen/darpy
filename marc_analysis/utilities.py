
import logging
logger = logging.getLogger()

import os, sys
from collections import OrderedDict
from datetime import datetime
from functools import wraps
from subprocess import call, check_output

import numpy as np
from xarray import DataArray, Dataset

## DATA REFERENCE
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
all_modes = ['NUC', 'AIT', 'ACC', 'OC', 'MOS', 'BC', 'MBS',
             'DST01', 'DST02', 'DST03', 'DST04',
             'SSLT01', 'SSLT02', 'SSLT03', 'SSLT04', ]

#####################################################################
## VERSIONING FUNCTIONS

def get_git_versioning():
    """ Returns the currently checked out commit shortname. """
    return check_output(
        ['git', 'rev-parse', '--short', 'HEAD']
    ).strip()

def append_history(ds, call_str=None, extra_info=None):
    """ Add a line to a Dataset's history record indicating the
    current calling context or operation.

    Parameters
    ----------
    ds : xarray.Dataset
        The Dataset to append history information to
    call_str : str, optional
        The full calling path of the operation - including the script
        name and arguments. If not passed, will grab from sys.argv
    extra_info : str, optional
        Additional info to include in the history statement

    Returns
    -------
    Dataset with appended history

    """
    try:
        history = ds.attrs['history']
    except KeyError:
        history = ""
    now = datetime.now()

    # Create the call string if not passed explicitly
    if call_str is None:
        call_str = " ".join(sys.argv)
    # Append extra info if passed
    if extra_info is not None:
        call_str += " ({})".format(extra_info)

    history = (now.strftime("%a %b %d %H:%M:%S %Y") +
               ": {}\n".format(call_str) +
               history)
    ds.attrs['history'] = history

    return ds

def get_timestamp(time=True, date=True, fmt=None):
    """ Return the current timestamp in machine local time.

    Parameters:
    -----------
    time, date : Boolean
        Flag to include the time or date components, respectively,
        in the output.
    fmt : str, optional
        If passed, will override the time/date choice and use as
        the format string passed to `strftime`.

    """

    time_format = "%H:%M:%S"
    date_format = "%m-%m-%Y"

    if fmt is None:
        if time and date:
            fmt = time_format + " " + date_format
        elif time:
            fmt = time_format
        elif date:
            fmt = date_format
        else:
            raise ValueError("One of `date` or `time` must be True!")

    return datetime.now().strftime(fmt)

################################################################
## DATASET MANIPULATION FUNCTIONS

def copy_attrs(data_orig, data_new):
    """ Copy the attributes of a DataArray or a DataSet and its
    child DataArrays from one instance to another. If the second
    instance has reduced dimensionality due to some aggregation
    or operation, any truncated coordinates will be ignored.

    """

    if isinstance(data_orig, Dataset):

        # Variables
        for v in data_orig.data_vars:
            field = data_orig[v]
            for attr, val in field.attrs.items():
                data_new[v].attrs[attr] = val

        # Coordinates
        for c in data_orig.coords:
            coord = data_orig.coords[c]
            for attr, val in coord.attrs.items():
                if c in data_new.coords:
                    data_new.coords[c].attrs[attr] = val

        # Metadata
        for attr, val in data_orig.attrs.items():
            data_new.attrs[attr] = val

    elif isinstance(data_orig, DataArray):

        # Variable Metadata
        for att, val in data_orig.attrs.items():
            data_new.attrs[att] = val

        # Coordinates
        for c in data_orig.coords:
            coord = data_orig.coords[c]
            for attr, val in coord.attrs.items():
                if c in data_new.coords:
                    data_new.coords[c].attrs[attr] = val

    else:
        raise ValueError("Couldn't handle type %r" % type(data_orig))

    return data_new

def preserve_attrs(func):
    """ Apply the method `copy_attrs` to the input and output
    of a function.

    Example:
    --------

    >>> a.some_attr = 1
    >>> @preserve_attrs
    ... def add(a, b):
    ...     return a + b
    >>> c = add(a + b)
    >>> print(c.some_attr)
    1

    """

    @wraps(func)
    def apply_copy_attrs(*args, **kwargs):
        data_orig = args[0]
        data_new  = func(*args, **kwargs)
        return copy_attrs(data_orig, data_new)

    return apply_copy_attrs

def decompose_multikeys(keys):
    """ Given a list of keys that are all tuples representing
    multi-indices or multikeys, returns the sets of key
    components used to create them.

    Example, using `itertools.product()`::

        >>> a, b = ['a', 'b', 'c', 'd'], [1, 2, 3, 4]
        >>> keys = product(a, b)
        >>> a_new, b_new = decompose_multikeys(keys)
        >>> sorted(a_new) == sorted(a)
        True

    Parameters:
    -----------
    keys : iterable of tuples or iterables
        The set of multi-index keys to unravel

    Returns:
    --------
    list of sets of multikey components

    """
    return [list(set(s)) for s in zip(*keys)]

def shuffle_dims(d, first_dims='lev'):
    """ Re-order the dimensions of a DataArray or Dataset such that
    the given keys are first in order. """

    if isinstance(first_dims, str):
        first_dims = [first_dims, ]
    assert isinstance(first_dims, (list, tuple))

    dims = list(d.dims)
    for i, dim in enumerate(first_dims):
        dims.insert(i, dims.pop(dims.index(dim)))
    return d.transpose(*dims)

################################################################
## IRIS CUBE FUNCTIONS

# def trunc_coords(data, valid_coords=['longitude', 'latitude'],
#                  method=iris.analysis.MEAN):
#     """ Collapse all except desired coordinates by performing
#     an aggregation along them.
#
#     Parameters
#     ----------
#     data : iris.cube.Cube
#         A multi-dimensional Cube holding the data being analyzed
#     valid_coords : list of strs
#         The final set of coordinates wanted in the Cube
#     method : iris analysis func
#         Aggregation method to apply along collapsing dimensions
#
#     """
#     raise NotImplementedError("`iris` deprecated with Python 3")
#
#     # Sanity check - make sure the coordinates we want are
#     # actually in the dataset
#     for coord in valid_coords:
#         _ = data.coord(coord) # -> will raise CoordinateNotFoundError
#
#     to_trunc = [ c.long_name for c in data.coords() if
#                  c.long_name not in valid_coords ]
#     data = data.collapsed(to_trunc, method)
#
#     return data

################################################################
## OTHER ANALYSIS FUNCTIONS


def shift_lons(lons):
    """ Shift longitudes from [0, 360] to [-180, 10] """
    mask = lons > 180
    lons[mask] = -(360. - lons[mask])
    return lons


def area_grid(lon, lat, asarray=False):
    """ Compute the area of the grid specified by 1D arrays
    lon and lat. Returns the result as a 2D array (nlon, nlat)
    containing the area of each gridbox, in m^2.

    Parameters
    ----------
    lon, lat : array-like of floats
        Arrays containing the longitude and latitude values
        of the given grid.
    asarray : Boolean
        Return an array instead of DataArray object

    Returns
    -------
    areas : xarray.DataArray or array
        Array with shape (len(lon), len(lats)) containing the
        area (in m^2) of each cell on the grid.

    """

    #: Earth's radius (m)
    R_EARTH = 6375000.

    # Induce lon/lat to array-like
    lon_arr = np.asarray(lon)
    lat_arr = np.asarray(lat)

    ## grid parameters
    nlon, nlat = len(lon_arr), len(lat_arr)
    # mean delta lon between gridpoints (actually constant on regular grid)
    # converted to radians
    dlon  = np.abs(np.mean(lon_arr[1:]-lon_arr[:-1]))*np.pi/180.
    # same, for lat
    dlat  = np.abs(np.mean(lat_arr[1:]-lat_arr[:-1]))*np.pi/180.
    # convert latitudes from -90,90 -> -180, 0 and then to radians
    theta = (90. - lat_arr)*np.pi/180.

    areas = np.zeros([nlon, nlat])
    for lat_i in range(nlat):
        for lon_i in range(nlon):

            lat1 = theta[lat_i] - dlat/2.
            lat2 = theta[lat_i] + dlat/2.

            if (theta[lat_i] == 0) or (theta[lat_i] == np.pi):
                areas[lon_i, lat_i] = (R_EARTH**2.) \
                                     *np.abs(  np.cos(dlat/2.) \
                                             - np.cos(0.)     )*dlon
            else:
                areas[lon_i, lat_i] = (R_EARTH**2.) \
                                     *np.abs( np.cos(lat1) - np.cos(lat2) )*dlon

    areas = areas.T # re-order (lat, lon)

    if asarray:
        return areas
    else:
        # Construct DataArray from the result

        # For some reason, there are random issues with the order
        # of the coordinates, so we try both ways here.
        coords = OrderedDict(lon=lon, lat=lat)
        while True:
            try:
                areas = DataArray(
                    areas, coords, name='area',
                    attrs=dict(long_name="grid cell area", units="m^2")
                )

                break
            except ValueError:
                areas = areas.T

        return areas

def latitude_weights(lats):
    """ Compute the 'gauss weights' for computed area-weighted
    averages of CESM output data. These can be applied over
    the latitude axis of any output, as you would do with the
    'gw' field in the normal history tapes.

    Parameters
    ----------
    lats : array-like of floats
        Arrays containing the latitude values of the given grid.

    Returns
    -------
    weights : array-like
        Array with shape `len(lats)` containing the weights for
        each latitude band.

    """

    areas = area_grid(np.array([0., 360.]), lats)
    aw = areas.sum(axis=0)
    weights = 2.*aw/np.sum(aw)

    return weights

class Coord(object):
    """ A lightweight container for normalizing lat-lon coordinates.

    Attributes:
    -----------
    x, y : floats
        The coordinate x- and y-values, respectively
    """

    def __init__(self, coords):
        """ Default constructor from tuple. Assumes that the longitude
        spans from 0(equator) - 360, with 180 as the prime meridian,
        and that latitude is symmetric from -90-90. Converts the
        longitude to a -180-180 symmetric basis.

        Parameters:
        -----------
        coords : iterable of the x- and y- components
        """
        self.x = coords[0]
        self.y = coords[1]

        if 180. < self.x <= 360.:
            self.x = -360. + self.x

    @staticmethod
    def convert_latlon(latlon):
        """ Convert a latlon string to a numerical value in the ranges
        lat = (-90, 90) and lon = (-180, 180)

        """

        # Latitudes
        if "N" in latlon:
            latlon = float(latlon.split("N")[0])
        elif "S" in latlon:
            latlon = -1.*float(latlon.split("S")[0])

        # Longitudes
        elif "E" in latlon:
            latlon = float(latlon.split("E")[0])
        elif "W" in latlon:
            latlon = 360. - float(latlon.split("W")[0])

        # All else?
        else:
            latlon = float(latlon)

        return latlon

    def __repr__(self):

        return "(%3.1f, %3.1f)" % (self.x, self.y)

######################################################################
## Extraction/processing functions

def remove_intermediates(intermediates):
    """ Delete list of intermediate files. """
    for fn in intermediates:
        logger.debug("Removing", fn)
        os.remove(fn)

def arg_in_list(arg, arg_list):
    """ Returns true if `arg` is a partial match for any value in `arg_list`. """
    return reduce(lambda a, b: a or b, [arg in s for s in arg_list])

def cdo_func(args, fn_in, fn_out, silent=True):
    """ Execute a sequence of CDO functions in a single process. """
    call_args = ["cdo", "-O", ]
    if silent: call_args.append("-s")

    def _proc_arg(arg):
        if isinstance(arg, (list, tuple)):
            return ",".join(arg)
        else:
            return arg

    call_args.append(args[0])

    for arg in args[1:]:
        call_args.append("-" + _proc_arg(arg))
    call_args.extend([fn_in, fn_out])

    logger.debug("      CDO - %s" % " ".join(call_args))
    call(call_args)

    # Post-process using ncwa to remove variables which have been
    # averaged over
    # if arg_in_list("vert", args):
    #     call(['ncwa', '-O', '-a', "lev", fn_out, fn_out])
    if arg_in_list("tim", args):
        call(['ncwa', '-O', '-a', "time", fn_out, fn_out])
    # if arg_in_list("zon", args):
    #     call(['ncwa', '-O', '-a', "lon", fn_out, fn_out])
