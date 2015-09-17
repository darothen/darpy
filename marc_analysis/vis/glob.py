
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry as sgeom
from cartopy.mpl.gridliner import ( LONGITUDE_FORMATTER,
                                    LATITUDE_FORMATTER   )
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from . common import PLOTTYPE_ARGS, make_colors, check_cyclic
from .. convert import cyclic_dataarray

__all__ = [ 'region_plot', 'global_plot' ]

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
        proj_kwargs = []
    proj = ccrs.__dict__[projection](*proj_kwargs)

    # Was an axis passed to plot on?
    new_axis = ax is None

    if new_axis:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=proj)
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
                        lw=2, alpha=0.5, label=region, zorder=10)
        if colors is not None:
            color = colors[region]
            box_args['facecolor'] = color
            box_args['lw'] = 1

        ax.add_geometries([box], box_proj, **box_args)

    return ax

def global_plot(data, ax=None, levels=None, 
                cmap='cubehelix_r', method='contourf',
                coloring='discrete', projection='PlateCarree',
                grid=False, check_coords=False, plot_kws={}):
    """ Create a global plot of a given variable.

    Parameters:
    -----------
    data : xray.DataArray
        The data to be plotted.
    ax : axis
        An existing axis instance, else one will be created.
    levels : arraylike of floats
        User-defined levels for colorbars.
    cmap : str
        String to use for looking up colormap.
    method : str
        String to use for looking up name of plotting function via iris 
    coloring : str
        Either 'continuous' or 'discrete' coloring option.
    projection : str or tuple
        Name of the cartopy projection to use and any args
        necessary for initializing it passed as a dictionary
    grid : bool
        Include lat-lon grid verlay
    check_coords : bool
        Force truncation of extra coordinates (not lon/lat)
    plot_kws : dict
        Any additional keyword arguments to pass to the plotter.

    """

    # TODO: enable access to Seaborn colormaps

    # Check if data needs a cyclic point added
    if not check_cyclic(data, coord='lon'):
        data = cyclic_dataarray(data, coord='lon')

    # Set up plotting function
    if method in PLOTTYPE_ARGS:
        extra_args = PLOTTYPE_ARGS[method].copy()
    else:
        raise ValueError("Don't know how to deal with '%s' method" % method)
    extra_args.update(plot_kws)

    # Alias a plot function based on the requested method and the 
    # datatype being plotted
    plot_func = plt.__dict__[method]
    
    # Set up map projection information
    if isinstance(projection, (list, tuple)):
        if len(projection) != 2:
            raise ValueError("Expected 'projection' to only have 2 values")
        projection, proj_kwargs = projection[0], projection[1]
    else:
        proj_kwargs = []
    proj = ccrs.__dict__[projection](*proj_kwargs)
    # `transform` should be the ORIGINAL coordinate system -
    # which is always a simple lat-lon coordinate system in CESM
    # output
    extra_args['transform'] = ccrs.PlateCarree()

    # Map the cube data if necessary
    # TODO: re-enable once iris is Python3 compliant
    # if projection != 'PlateCarree':
    #     data, extent = project(data, ccrs.PlateCarree())

    # Was an axis passed to plot on?
    new_axis = ax is None

    if new_axis:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=proj)
    else: # Set current axis to one passed as argument
        plt.sca(ax)

    # Check data
    # if check_coords:
    #     data = trunc_coords(data, ['longitude', 'latitude'])

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
    except TypeError as e:
        print("Could not label the given map projection.")

    ## Plot data
    if levels is None:
        gp = plot_func(data.lon.values, data.lat.values, data.data,
                       cmap=plt.get_cmap(cmap), **extra_args)

    else:
        cmap, norm = make_colors(levels, coloring, cmap)

        if method == 'contourf': extra_args['levels'] = levels
        if method == 'pcolormesh': extra_args['norm'] = norm 
        gp = plot_func(data.lon.values, data.lat.values, data.data,
                       cmap=cmap, **extra_args)

    if new_axis:
        return ax, gp
    else:
        return gp
