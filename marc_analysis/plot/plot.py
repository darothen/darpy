
"""
Plotting functions. Recommended to use this module directly by
importing:
    import marc_analysis.plot as maplt

"""

import functools
import inspect
import warnings

import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry as sgeom
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from . common import *
from .. convert import cyclic_dataarray

## 1D PLOTS

def line_plot(darray, dim=None, ax=None, **kwargs):
    """
    Simple line plot of DataArray against a specified dimension.

    """
    darray = darray.squeeze()

    ndims = len(darray.dims)
    if (dim is None) and (ndims > 1):
        raise ValueError("If more than one dimension present, then must "
                         "explicitly specify plot dim")
    if (dim is not None):
        if not (dim in darray.dims):
            raise ValueError("'{}'".format(dim) + " was not in dims")
        x = darray.indexes[dim]
    else:
        dim, x = list(darray.indexes.items())[0]

    if ax is None:
        ax = plt.gca()

    lp = ax.plot(x, darray, **kwargs)

    # Set some labeling defaults
    ax.set_xlabel(dim)
    if darray.name is not None:
        lab_str = darray.name
        if hasattr(darray, 'units'):
            lab_str += " [{}]".format(darray.units)
        ax.set_ylabel(lab_str)

    # If this is a zonal plot (latitude is the dimension),
    # Do some additional plot tweaking
    if dim == 'lat':
        ax = format_zonal_axis(ax)

    # Format dates on xlabels if necessary
    if np.issubdtype(x.dtype, np.datetime64):
        plt.gcf().autofmt_xdate()

    sns.despine(offset=10)

    return ax, lp

## 2D PLOTS

def infer_x_y(darray, x=None, y=None):
    """
    Infer the ordering of the dimensions in the data field
    for plotting.
    """
    dims = set(darray.dims)
    ndims = len(dims)

    if (x is None) and (y is None):
        if ndims > 2:
            raise ValueError("Too many dimensions in data array; please "
                             "pass 'x' or 'y' to reduce plotting ambiguity")
        elif ndims == 1:
            # This will ultimately produce a lineplot
            x = darray.dims[0]
            return x, None
    elif (x is None):
            # Swap order for logic testing
            x = y
            y = None

    user_dims = set()
    for dim in [x, y]:
        if dim is None: continue

        if not (dim in dims):
            raise ValueError("Couldn't find {} in data coords".format(dim))
        user_dims.add(dim)

    # Infer 'y' from whatever is left in dims if necessary
    if (y is None):
        y = dims.difference(user_dims).pop()

    if not (x is None) and not (y is None):
        return x, y

    # At this point, both are 'None' so we need to infer
    # potential dims:
    # lon, lat, time, lev
    # 1) 2d geo plot: lon, lat
    if dims == set(['lon', 'lat']):
        return 'lon', 'lat'
    elif 'lon' in dims:
        y = dims.difference(['lon', ]).pop()
        return 'lon', y
    elif 'lat' in dims:
        y = dims.difference(['lat', ]).pop()
        return 'lat', y
    elif 'lev' in dims:
        x = dims.difference(['lev', ]).pop()
        return x, 'lev'
    else:
        return list(dims)


def geo_plot(darray, ax=None, method='contourf',
             projection='PlateCarree', grid=False, **kwargs):
    """ Create a global plot of a given variable.

    Parameters:
    -----------
    darray : xray.DataArray
        The darray to be plotted.
    ax : axis
        An existing axis instance, else one will be created.
    method : str
        String to use for looking up name of plotting function via iris
    projection : str or tuple
        Name of the cartopy projection to use and any args
        necessary for initializing it passed as a dictionary;
        see func:`make_geoaxes` for more information
    grid : bool
        Include lat-lon grid overlay
    **kwargs : dict
        Any additional keyword arguments to pass to the plotter.

    """

    # Check if darray needs a cyclic point added
    if not check_cyclic(darray, coord='lon'):
        darray = cyclic_dataarray(darray, coord='lon')

    # Set up plotting function
    if method in PLOTTYPE_ARGS:
        extra_args = PLOTTYPE_ARGS[method].copy()
    else:
        raise ValueError("Don't know how to deal with '%s' method" % method)
    extra_args.update(**kwargs)

    # Alias a plot function based on the requested method and the
    # datatype being plotted
    plot_func = plt.__dict__[method]

    # `transform` should be the ORIGINAL coordinate system -
    # which is always a simple lat-lon coordinate system in CESM
    # output
    extra_args['transform'] = ccrs.PlateCarree()

    # Was an axis passed to plot on?
    new_axis = ax is None

    if new_axis:
        ax = make_geoaxes(projection)
    else: # Set current axis to one passed as argument
        if not hasattr(ax, 'projection'):
            raise ValueError("Expected `ax` to be a GeoAxes instance")
        plt.sca(ax)

    # Setup map
    ax.set_global()
    ax.coastlines()

    try:
        gl = ax.gridlines(crs=extra_args['transform'], draw_labels=True,
                          linewidth=0.5, color='grey', alpha=0.8)
        LON_TICKS = [ -180, -90, 0, 90, 180 ]
        LAT_TICKS = [ -90, -60, -30, 0, 30, 60, 90 ]
        gl.xlabels_top   = False
        gl.ylabels_right = False
        gl.xlines = grid
        gl.ylines = grid
        gl.xlocator = mticker.FixedLocator(LON_TICKS)
        gl.ylocator = mticker.FixedLocator(LAT_TICKS)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
    except TypeError:
        warnings.warn("Could not label the given map projection.")

    # Infer colormap settings if not provided
    if not ('vmin' in kwargs):
        cmap_kws = infer_cmap_params(darray.data, **extra_args)
        extra_args.update(cmap_kws)

    gp = plot_func(darray.lon.values, darray.lat.values, darray.data,
                   **extra_args)

    return ax, gp


def vertical_plot(darray, ax=None, method='contourf', top=100., log_vert=False,
                  grid=False, **kwargs):
    """ Create a contour plot of a given variable with the vertical dimension
    on the y-axis.

    Parameters
    ----------
    darray : xray.Dataset or xray.DataArray
        The data to be plotted.
    ax : axis
        An existing axis instance, else one will be created.
    top : float, defaults to 100.
        Cut-off for y-axis plot limits
    grid : bool
        Include lat-lon grid overlay
    grid : bool
        Include lat-lon grid overlay
    plot_kws : dict
        Any additional keyword arguments to pass to the plotter.

    """

    # Set up plotting function
    if method in PLOTTYPE_ARGS:
        extra_args = PLOTTYPE_ARGS[method].copy()
    else:
        raise ValueError("Don't know how to deal with '%s' method" % method)
    extra_args.update(**kwargs)

    # Alias a plot function based on the requested method and the
    # datatype being plotted
    plot_func = plt.__dict__[method]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else: # Set current axis to one passed as argument
        plt.sca(ax)


    # Infer colormap settings if not provided
    if not ('vmin' in kwargs):
        cmap_kws = infer_cmap_params(darray.data)
        extra_args.update(cmap_kws)

    # Plot data
    x_dim = set(darray.dims).difference(['lev', ]).pop()
    xvals, yvals = darray.indexes[x_dim].values, darray.indexes['lev'].values
    zvals = darray.to_masked_array(copy=False)

    zvp = plot_func(xvals, yvals, zvals, **extra_args)

    # X-Axis
    if x_dim == 'lat':
        ax = format_zonal_axis(ax)
    elif x_dim == 'time':
        # ax = format_time_axis(ax)
        pass

    # Y-axis
    if log_vert:
        ax.set_yscale('log')

        # Some nice tick labeling utilities from matplotlib
        from matplotlib.ticker import FormatStrFormatter, MaxNLocator
        ax.yaxis.set_major_formatter(
            FormatStrFormatter("%d")
        )
        ax.yaxis.set_major_locator(MaxNLocator(10))
    ax.set_ylim(1000, top)
    ax.set_ylabel("Pressure (hPa)")

    if grid:
        plt.grid(linewidth=0.5, color='grey', alpha=0.8)

    return ax, zvp


def region_plot(regions, ax=None, colors=None, only_regions=False,
                figsize=(12, 5), projection='PlateCarree'):
    """

    Parameters:
    -----------
    regions : dict
        A mapping of region names -> coordinates defining the
        outline of the region. Assume that these represent the
        lower-left and upper-right coordinates of a box
    ax : Axes, optional
        An axes instance to plot on. If none is passed, will create one
        and return it
    colors : dict, optional
        A mapping of region names -> colors. If passed, then a thin border
        will be made around each region and they'll have the requested
        facecolor. If None, then each region will have a thicker black
        border and a transparent face.
    only_regions : boolean, optional
        Only plot the region boxes (don't add gridlines or continents)
    figsize : tuple, optional
        The figure size. Defaults to (12, 5)
    projection : str or tuple
        Name of the cartopy projection to use and any kwargs necessary
        for initializing it passed as a dictionary

    Returns:
    --------
    ax : Axes
        Instance containing the region map.

    """

    # Set up map projection information
    if isinstance(projection, (list, tuple)):
        if len(projection) != 2:
            raise ValueError("Expected 'projection' to only have 2 values")
        projection, proj_kwargs = projection[0], projection[1]
    else:
        proj_kwargs = {}

    # Was an axis passed to plot on?
    new_axis = ax is None

    if new_axis:
        ax = make_geoaxes(projection, **proj_kwargs)
    else: # Set current axis to one passed as argument
        plt.sca(ax)

    # Set up the map
    if not only_regions:
        ax.set_global()
        ax.coastlines(lw=1)
        gl = ax.gridlines(draw_labels=False)
        ax.add_feature(cfeature.LAND, facecolor='0.75')

    for region, coords in regions.items():
        ll, ur = coords

        # If the box spans the international dateline, then we need to
        # shift the coordinate system to account for this
        if ll.x*ur.x < 0 and ll.x > ur.x: # box crosses the Intl Dateline
            box_proj = ccrs.PlateCarree(central_longitude=180.)
            f = lambda x: -180. + x if x > 0 else 180. + x
            ll.x = f(ll.x)
            ur.x = f(ur.x)
        else:
            box_proj = ccrs.PlateCarree()

        # Generate polygon to plot
        box = sgeom.box(ll.x, ll.y, ur.x, ur.y)
        # box = sgeom.Polygon([(ll.x, ll.y), (ur.x, ll.y),
        #                     (ur.x, ur.y), (ll.x, ur.y)])


        # Setup arguments for styling the box
        box_args = dict(facecolor='none', edgecolor='black',
                        lw=2, alpha=1.0, label=region, zorder=10)
        if colors is not None:
            color = colors[region]
            box_args['facecolor'] = color
            box_args['lw'] = 1

        ax.add_geometries([box], box_proj, **box_args)

    return ax


def hovmoller_plot():
    raise NotImplementedError()


def default_plot(x, y, z, ax, **kwargs):
    """
    Default contour plot.

    """
    ax, cf = ax.contourf(x, y, z, **kwargs)
    return ax, cf

_PLOT2D_MAP = {
    ('lon', 'lat'): geo_plot,
    ('lat', 'lev'): vertical_plot,
    ('lat', 'time'): hovmoller_plot,
    ('lon', 'time'): hovmoller_plot,

}
def plot2d(darray, x=None, y=None, ax=None,
           vmin=None, vmax=None, cmap=None,
           center=None, robust=None, levels=None,
           func_kwargs={}, **kwargs):
    """
    Automatically plot an appropriate 2D plot based on the dimensions
    of a given DataArray

    Parameters
    ----------
    darray : xray.DataArray
        A 2D xray.DataArray with the field to plot.
    x : string, optional
        Coordinate for x-axis; else will infer
    y : string, optional
        Coordinate for y-axis; else will infer
    ax : matplotlib axes object or subclass, optional
        If none, will attempt to get the current axis for plotting
    vmin, vmax : floats
        Minimum and maximum value for coloring
    cmap : str
        Name of color map from matplotlib or seaborn
    center : float
        Value to fix at center of coloring scheme
    robust : bool (default = False)
        Infer maximum and minimum using a percentile estimate over all
        the data
    levels : int or iterable of floats
        Either the number of levels to use or the values of all the
        level demarcations
    func_kwargs : dict, optional
        Dictionary of kwargs to pass to the dispatch plot function
    **kwargs : optional
        Additional arguments passed to the dispatch plot function

    Returns
    -------
    axes, artist :
        A references to the axes object being plotted and the
        primitive generated by the plotting function
    """

    # Look up the plotting function
    DEFAULT_PLOT = True
    plotfunc = default_plot

    x, y = infer_x_y(darray, x, y)
    xvals = darray.indexes[x].values
    yvals = darray.indexes[y].values
    zvals = darray.to_masked_array(copy=False)

    try:
        plotfunc = _PLOT2D_MAP[(x, y)]
        DEFAULT_PLOT = False
    except KeyError:
        pass

    # Grab the default arguments for the dispatch plotting method
    # The callspec of each of these functions is (darray, ax,
    # **func_args, **kwargs), so strip off the first !2! args and the
    # the first !1! default value to construct the default dict
    default_func_kwargs = dict(
        zip(inspect.getargspec(plotfunc).args[2:],
            inspect.getargspec(plotfunc).defaults[1:])
    )
    default_func_kwargs.update(func_kwargs)
    func_kwargs = default_func_kwargs.copy()

    # Create an axes object to plot on if necessary
    if ax is None:
        if 'projection' in func_kwargs:
            ax = make_geoaxes(func_kwargs['projection'])
        else:
            ax = plt.gca()

    cmap_kwargs = {
        'plot_data': zvals.data,
        'vmin': vmin, 'vmax': vmax,
        'cmap': cmap, 'center': center, 'robust': robust,
        'levels': levels,
    }
    cmap_params = infer_cmap_params(**cmap_kwargs)

    # Begin constructing arguments for plotting function
    kwargs['levels'] = cmap_params['levels']
    kwargs.setdefault('norm', cmap_params['cnorm'])

    ## Choose the plotting function
    if not DEFAULT_PLOT:
        kwargs.update(**func_kwargs)
        return plotfunc(darray, ax=ax, **kwargs)
    else:
        return default_plot(xvals, yvals, zvals, ax=ax, **kwargs)
