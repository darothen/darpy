# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns

from . common import set_labels, make_colors, PLOTTYPE_ARGS

__all__ = [ 'zonal_plot', 'zonal_vert_plot' ]

def zonal_plot(data, ax=None, set_axes=True, check_coords=False, 
               plot_kws={}): 
    """ Create a zonal plot of a given variable. 

    Parameters
    ----------
    data : xray.DataArray
        The data to be plotted.
    ax : axis
        An existing axis instance, else one will be created.
    set_axes : boolean
        Label plot axes and set axis limits
    check_coords : bool
        Force truncation of extra coordinates (not only lat)
    plot_kws : dict
        Any additional keyword arguments to pass to the plotter.

    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # Check data
    # if check_coords:
    #     data = trunc_coords(data, ['latitude', ])

    zp = ax.plot(data.lat, data, **plot_kws)

    if set_axes:
        sns.despine(offset=10)
        
        set_labels(ax, 
                   xlabel="Latitude (°N)",
                   ylabel="%s (%s)" % (data.long_name, data.units) )

        ax.set_xlim(-90, 90)
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])

    if ax is not None:
        return zp
    else:
        return ax, zp

def zonal_vert_plot(data, ax=None, set_axes=True, levels=None, top=100.,
                    cmap='cubehelix_r', method='contourf', 
                    coloring='discrete', grid=False,
                    check_coords=False, plot_kws={}):
    """ Create a zonal-vertical plot of a given variable.

    Parameters
    ----------
    data : xray.Dataset or xray.DataArray
        The data to be plotted.
    ax : axis
        An existing axis instance, else one will be created.
    set_axes : boolean
        Label plot axes and set axis limits
    levels : arraylike of floats
        User-defined levels for colorbars.
    top : float, defaults to 100.
        Cut-off for y-axis plot limits
    cmap : str
        String to use for looking up colormap.
    grid : bool
        Include lat-lon grid overlay
    coloring : str
        Either 'continuous' or 'discrete' coloring option.
    grid : bool
        Include lat-lon grid overlay
    check_coords : bool
        Force truncation of extra coordinates (not lon/lat)
    plot_kws : dict
        Any additional keyword arguments to pass to the plotter.

    TODO: Need to figure out z-coordinate name! 

    """

    # Set up plotting function
    if method in PLOTTYPE_ARGS:
        extra_args = PLOTTYPE_ARGS[method].copy()
    else:
        raise ValueError("Don't know how to deal with '%s' method" % method)
    extra_args.update(plot_kws)

    # Alias a plot function based on the requested method and the
    # datatype being plotted
    plot_func = plt.__dict__[method]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else: # Set current axis to one passed as argument
        plt.sca(ax)

    # Check data
    # if check_coords:
    #    data = trunc_coords(data, ['longitude', 'latitude'])

    ## Plot data
    if levels is None:
        zvp = plot_func(data.lat, data.lev, data, cmap=plt.get_cmap(cmap),
                        **extra_args)

    else:
        cmap, norm = make_colors(levels, coloring, cmap)

        if method == 'contourf': extra_args['levels'] = levels
        if method == 'pcolormesh': extra_args['norm'] = norm
        zvp = plot_func(data.lat, data.lev, data,  cmap=cmap, norm=norm,
                        **extra_args)

    if set_axes:
        sns.despine()
        ax.set_xlim(-90, 90)
        ax.set_ylim(1000, top)
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        set_labels(ax, xlabel="Latitude (°N)", ylabel="Vertical level")

    if grid:
        plt.grid(linewidth=0.5, color='grey', alpha=0.8)

    if ax is not None:
        return zvp
    else:
        return ax, zvp