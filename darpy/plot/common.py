
# -*- coding: utf-8 -*-

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from matplotlib.colors import from_levels_and_colors, Normalize
from matplotlib.pyplot import get_cmap, colorbar, savefig
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from seaborn.apionly import color_palette

import numpy as np
import pandas as pd
import xarray

from textwrap import wrap

from .. utilities import cyclic_dataarray
from .. analysis import global_avg


PLOTTYPE_ARGS = {
    'pcolormesh': dict(linewidth='0'),
    'pcolor': dict(linewidth='0'),
    'contourf': dict(extend='both'),
    'imshow': dict(),
}

def get_projection(name, *args, **kwargs):
    """ Instantiate a Cartopy coordinate reference system for
    constructing a GeoAxes object.

    """
    return ccrs.__dict__[name](*args, **kwargs)

#: Default projection settings
PROJECTION = get_projection("PlateCarree", central_longitude=0.)
TRANSFORM = get_projection("PlateCarree")
SUBPLOT_KWS = dict(projection=PROJECTION, aspect='auto')
LAT_TICKS = [-90, -60, -30, 0, 30, 60, 90]
LON_TICKS = [-150, -90, -30, 30, 90, 150]
TICK_STYLE = {'size': 12}
GRID_STYLE = dict(linewidth=0.5, color='grey', alpha=0.3)


def get_figsize(nrows=1, ncols=1, size=4., aspect=16./10.):
    """ Compute figure size tuple given columns, rows, size, and aspect. """
    width = ncols*size*aspect
    height = nrows*size
    return (width, height)


def set_ytickformat(ax, fmt="%.4g"):
    """ Change y-axis tick format to use a more flexible notation. """
    yax = ax.get_yaxis()
    yax.set_major_formatter(
        mticker.FormatStrFormatter(fmt)
    )
    return ax


def format_zonal_axis(ax, axis='x', reverse=False):
    """ Properly format an axis with latitude labels.

    Parameters
    ----------
    ax : Axes object
        The axis to plot on
    axis : str (default, 'x')
        String indicating 'x' or 'y' axis to format
    reverse : logical (default, False)
        Reverse the zonal axis so that it runs from North->South

    Returns
    -------
    Axes object with formatted zonal axis/
    """

    ticks = [-90, -60, -30, 0, 30, 60, 90]
    lims = [-90, 90]

    locator = mticker.FixedLocator(ticks)
    def _fmt(x, pos):
        if x == 0:
            label = "{:d}".format(x)
        elif x > 0:
            label = "{:d}°N".format(x)
        else:
            label = "{:d}°S".format(-x)
        return label
    formatter = mticker.FuncFormatter(_fmt)

    if reverse:
        lims = lims[::-1]
        ticks = ticks[::-1]

    if axis == 'x':
        set_labels(ax, xlabel="Latitude")

        ax.set_xlim(*lims)
        # ax.set_xticks(ticks)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    else:
        set_labels(ax, ylabel="Latitude")

        ax.set_ylim(*lims)
        # ax.set_yticks(ticks)
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)

    return ax


def label_lon(ax, axis='x', transform=TRANSFORM):
    """ Label longitudes on a plot. """
    gl = ax.gridlines(crs=transform, draw_labels=True, **GRID_STYLE)
    if axis == 'x':
        gl.xlabels_top = gl.ylabels_left = gl.ylabels_right = False
        gl.xlocator = mticker.FixedLocator(LON_TICKS)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.xlabel_style = TICK_STYLE
    else:
        gl.xlabels_top = gl.xlabels_bottom = gl.ylabels_right = False
        gl.ylocator = mticker.FixedLocator(LON_TICKS)
        gl.yformatter = LONGITUDE_FORMATTER
        gl.ylabel_style = TICK_STYLE

    return ax


def label_lat(ax, axis='y', transform=TRANSFORM):
    """ Label latitudes on a plot. """
    gl = ax.gridlines(crs=transform, draw_labels=True, **GRID_STYLE)
    if axis == 'y':
        gl.xlabels_top = gl.xlabels_bottom = gl.ylabels_right = False
        gl.ylocator = mticker.FixedLocator(LAT_TICKS)
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylabel_style = TICK_STYLE
    else:
        gl.xlabels_top = gl.ylabels_left = gl.ylabels_right = False
        gl.xlocator = mticker.FixedLocator(LAT_TICKS)
        gl.xformatter = LATITUDE_FORMATTER
        gl.xlabel_style = TICK_STYLE

    return ax


def label_global_avg(g):
    """ Label global field avg in a box for a FacetGrid of plots. """
    print("Global averages")
    print("---------------")
    for ax, name_dict in zip(g.axes.flat, g.name_dicts.flat):
        if name_dict is None:
            continue

        d = g.data.loc[name_dict]
        avg = global_avg(d)
        avg = float(avg.data)

        # Convert to TeX scientific notation if exponential
        avg_str = "{:3.2g}".format(avg)
        if "e" in avg_str:
            coeff, exp = avg_str.split("e")
            exp = int(exp)
            avg_str = "{}x10".format(coeff) + "$^{%d}$" % exp
        s = "{} {}".format(avg_str, d.units)

        xl = ax.get_xlim()
        yl = ax.get_ylim()
        ax.text(xl[-1] - 5, yl[-1] - 15, s,
                size=11, ha='right',
                bbox=dict(facecolor='white', alpha=0.8))
        print("   ", name_dict, s, avg)

    return g


def geo_grid_plot(data, x, y, col=None, row=None, col_wrap=None,
                  label_grid=True, label_avg=True, **kws):

    plot_kws = {}
    if (col is not None) or (row is not None):
        plot_kws.update(dict(aspect=16./10., size=3.))
    plot_kws.update(kws)

    subplot_kws = dict(projection=PROJECTION, aspect='auto')

    g = data.plot.imshow(x=x, y=y, row=row, col=col, col_wrap=col_wrap,
                         subplot_kws=subplot_kws, **plot_kws)
    if label_grid:
        # try:
        g = geo_prettify(g, transform=TRANSFORM)
        # except TypeError:
        #     pass
    if label_avg:
        g = label_global_avg(g)

    plt.draw()

    return g


def geo_prettify(g, transform):
    """ Make longitude-latitude labels and grid for a FacetGrid of
    horizontal plots. """
    # TODO: Just iterate over axes and apply geo_prettify_ax()
    for ax in g.axes.flat:
        ax.coastlines()
        gl = ax.gridlines(crs=transform, **GRID_STYLE)
    # Bottom Row
    for ax in g._bottom_axes:
        ax = label_lon(ax, transform=transform)
    # Left column
    for ax in g._left_axes:
        ax = label_lat(ax, transform=transform)

    # Over-ride the default titles
    # TODO: This doesn't work because FacetGrid sets the right-hand column
    #       y-labels using annotate, so they're hard to remove. Will need to
    #       hack the xarray interface for this to work.
    # for ax, row_name in zip(g.axes[:, -1], g.row_names):
    #     ax.set_title(row_name, loc='right')
    # for ax, col_name in zip(g.axes[0, :], g.col_names):
    #     ax.set_title(col_name, loc='left')

    return g


def geo_prettify_ax(ax, transform=TRANSFORM, label_x=True, label_y=True):
    """ Make longitude-latitude labels for a single plot. """

    ax.coastlines()
    gl = ax.gridlines(crs=transform, **GRID_STYLE)
    if label_x:
        ax = label_lon(ax, transform=transform)
    if label_y:
        ax = label_lat(ax, transform=transform)

    return ax


def make_geoaxes(projection='PlateCarree', proj_kws={},
                 size=4., aspect=16./10.):
    """ Create a GeoAxes mapping object.

    Parameters
    ----------
    projection : str
        Name of the projection to use when creating the map
    proj_kws : dict (optional)
        Additional keyword arguments to pass to the projection
        creation function

    Returns
    -------
    ax : GeoAxes object with requeted mapping projection.

    """

    proj_kwargs = {}
    if proj_kws:
        proj_kwargs.update(proj_kws)

    proj = get_projection(projection, **proj_kwargs)
    fig = plt.figure(figsize=get_figsize(size, aspect))
    ax = fig.add_subplot(111, projection=proj, aspect='auto')

    return ax


def check_cyclic(data, coord='lon'):
    """ Checks if a DataArray already includes a cyclic point along the
    specified coordinate axis. If not, adds the cyclic point and returns
    the modified DataArray.

    """
    return np.all(data.isel(**{coord: 0}) == data.isel(**{coord: -1}))


class MidpointNorm(Normalize):
    """ A normalization tool for ensuring that '0' occurs in the
    middle of a colorbar.

    """
    def __init__(self, vmin, vmax, midpoint, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        #x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        #return np.ma.masked_array(np.interp(value, x, y))
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat>0] /= abs(vmax - midpoint)
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = np.ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result


def make_colors(levels, coloring, cmap):
    """ Generate colormaps and norms based on user-defined
    level sets.

    Parameters:
    -----------
    levels : iterable of floats
        List of level demarcations.
    coloring : str
        Either 'continuous' or 'discrete'; 'discrete' colorings
        have a set number of color elements, whereas 'continuous'
        ones span value in between the elements of `levels`.
    cmap : str
        The color palette / map name to use

    Returns:
    --------
    cmap
        A colormap that matplotlib can use to set plot colors
    norm
        The normalization controlling how the data is colored
        based on the user-defined levels.

    """

    ncolors = len(levels) + 1
    assert coloring in ['continuous', 'discrete']

    ## reverse the colorbar if its cubehelix and a negative scale
    if ("cubehelix" in cmap) and (levels[0] < 0):
        if "_r" in cmap: cmap = cmap.replace("_r", "")
        else: cmap = cmap+"_r"

    if coloring == 'continuous':
        norm = MidpointNorm(vmin=levels[0], vmax=levels[-1],
                            midpoint=levels[len(levels)/2])

    elif coloring == 'discrete':
        cmap = get_cmap(cmap)
        colors = [cmap(1.*i/ncolors) for i in range(ncolors)]

        cmap, norm = from_levels_and_colors(levels, colors, extend='both')
    else:
        raise ValueError("Received unknown coloring option (%r)" % coloring)

    return cmap, norm


def add_colorbar(mappable, fig=None, ax=None, thickness=0.025,
                 shrink=0.1, pad=0.05, orientation='horizontal'):
    """ Add a colorbar into an existing axis or figure. Need to pass
    either an Axis or Figure element to the appropriate keyword
    argument. Should elegantly handle multi-axes figures.

    Parameters
    ----------
    mappable : mappable
        The element set with data to tailor to the colorbar
    fig : Figure
    ax: Axis
    thickness: float
        The width/height of the colorbar in fractional figure area,
        given either vertical/horizontal orientation.
    shrink: float
        Fraction of the width/height of the figure to leave blank
    pad : float
        Padding between bottom/right subplot edge and the colorbar
    orientation : str
        The orientation of the colorbar

    """
    if (fig is None) and (ax is None):
        raise ValueError("Must pass either 'fig' or 'ax'")
    elif fig is None:
        # Plot on Axis
        cb = colorbar(mappable, ax=ax, pad=pad, orientation=orientation)
    else:
        # Plot onto Figure's set of axes
        axes = fig.axes

        # Get coordinates for making the colorbar
        ul = axes[0]
        lr = axes[-1]
        top = ul.get_position().get_points()[1][1]
        bot = lr.get_position().get_points()[0][1]
        right = lr.get_position().get_points()[1][0]
        left = ul.get_position().get_points()[0][0]

        # Calculate colorbar positioning and geometry
        if orientation ==  'vertical':
            cb_left = right + pad
            cb_width = thickness
            cb_bottom = bot + shrink
            cb_height = (top - shrink) - cb_bottom
        elif orientation == 'horizontal':
            cb_left = left + shrink
            cb_width = (right - shrink) - cb_left
            cb_height = thickness
            cb_bottom = (bot - pad) - cb_height
        else:
            raise ValueError("Uknown orientation '%s'" % orientation)

        cax = fig.add_axes([cb_left, cb_bottom,
                            cb_width, cb_height])
        cb = fig.colorbar(mappable, cax=cax, orientation=orientation)

    return cb


def multipanel_figure(nrow, ncol, aspect=16./10., size=3.,
                      sharex=True, sharey=True,
                      cbar_space=1., cbar_orientation='vertical',
                      **subplot_kw):
    """ Generate a Figure with subplots tuned to a user-specified
    grid and plot size. The Figure's tight_layout() method will
    automatically be wrapped to correctly set the inter-plot padding

    Parameters
    ----------
    nrow, ncol : float
        The number of rows and columns in the grid
    aspect : float, optional
        Aspect ratio of each subplot, so that ``aspect * size`` gives
        the width of each subplot in inches
    size : float, optional
        Height (in inches) of each subplot
    share{x,y} : logical (optional)
        Share x- or y-axes among plots
    cbar_space : float (optional)
        Extra space (in inches) to add to figure for colorbar.
    cbar_orientation : str, 'horizontal' or 'vertical'
        Designation for orientation of potential colorbar
    **kwargs : dict
        Any additional keyword arguments to pass to ``plt.subplots()``

    Returns
    -------
    fig, axes : matplotlib.fig.Figure object and list of axes objects

    """

    # Compute figure sizing
    fig_width = ncol * size * aspect
    fig_height = nrow * size

    if cbar_orientation == 'vertical':
        fig_width += cbar_space
    else:
        fig_height += cbar_space

    default_subplot_kw = dict(aspect='auto')
    default_subplot_kw.update(subplot_kw)

    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_width, fig_height),
                             sharex=sharex, sharey=sharey, squeeze=False,
                             subplot_kw=default_subplot_kw)

    # Slightly tweak spacing between subplots
    fig.subplots_adjust(hspace=0.3, wspace=0.2)

    return fig, axes


def set_title(ax, varname="", units="", l=None, r=None):
    """ Adds potentially two titles to a plot, including
    the variable name with units on the left, and if provided,
    the source or difference source on the right.

    """
    ax.set_title("%s (%s)" % (varname, units), loc='left')
    if (l is not None) and (r is not None):
        ax.set_title("%s - %s" % (l, r), loc='right')
    elif l is not None: # r is None
        ax.set_title(l, loc='right')
    return ax


def set_labels(ax, xlabel=None, ylabel=None, width=0):
    """ Add x- or y-axis labels to an axis instance.

    Parameters
    ----------
    ax : matplotlib.Axes instance
        The axis to modify labels on
    xlabel, ylabel : strings, optional
        The labels to attach to the indicated axis
    width : int
        If a value > 0 is passed, then the label will be wrapped at the
        indicated character width and spill over to a new line

    Returns
    -------
    ax : reference to the axis passed to the function

    """
    if xlabel is not None:
        if width > 0:
            xlabel = "\n".join(wrap(xlabel), width)
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        if width > 0:
            ylabel = "\n".join(wrap(ylabel), width)
        ax.set_ylabel(ylabel)
    return ax


def label_hanging(g, geo=False):
    """ Add x-labels to "hanging" plots """
    n_hanging = len(g.col_names) % g._col_wrap
    for i, ax in enumerate(g.axes[-2, -n_hanging:]):
        print("   {}) {}".format(i+1, ax.get_xlabel()))
        if geo:
            ax = label_lon(ax)
        else:
            for t in ax.xaxis.get_ticklabels():
                # print("      " + t.get_text())
                t.set_visible(True)
        ax.set_xlabel(g._bottom_axes[0].get_xlabel())
    return g


def save_figure(root, suffix="", fig=None, qual="quick"):
    """ Save a figure with presets for different output
    qualities.

    """

    fig_dict = {'bbox_inches': 'tight'}
    if qual == "production":
        format = "png"
        fig_dict['dpi'] = 300
    elif qual == "quick":
        format = "png"
        fig_dict['dpi'] = 100
    elif qual == "vector":
        format = 'pdf'
        fig_dict['transparent'] = True
    else:
        raise ValueError("'qual' should be either 'quick' or 'production'")

    if suffix:
        fn = "figs/{root:s}_{suffix:s}.{format:s}".format(root=root,
                                                          suffix=suffix,
                                                          format=format)
    else:
        fn = "figs/{root:s}.{format:s}".format(root=root, suffix=suffix,
                                               format=format)

    print("Saving figure %s..." % fn)
    if fig is None:
        savefig(fn, **fig_dict)
    else:
        fig.savefig(fn, **fig_dict)
    print("done.")


def colortext_legend(text_color_map, ax=None, text_labels=None, **kwargs):
    """ Add a custom-built legend to a plot, where all the items in
    the legend have colored text corresponding to colored elements
    in the figure.

    Parameters:
    -----------
    text_color_map : dict
        A mapping from labels -> colors which will be used to construct
        the patch elements to include in the legend.
    ax : Axes
        The axes instance on which to add the legend. If not passed,
        then the legend will simply be created and returned.
    text_labels : dict
        A mapping from labels -> longer descriptions to be used in their
        place in the legend.
    **kwargs : dict-like
        Additional arguments to pass to the legend function.

    Returns:
    --------
    leg : legend object
        The legend, for further customization

    """

    legend_elems = []
    labels = []
    for text, color in text_color_map.items():
        legend_elems.append(
            ( mpatches.Rectangle((0, 0), 1, 1,
                                 facecolor='none',
                                 edgecolor='none'),
              text )
        )
        labels.append(text)

    # Set up a legend with colored-text elements
    elems, texts = zip(*legend_elems)
    if ax is not None:
        ax.legend(elems, texts, **kwargs)
        leg = ax.get_legend()
    else:
        leg = plt.legend(elems, texts, **kwargs)

    # Change the label color
    for label in leg.get_texts():
        old_label = label.get_text()
        if text_labels is None:
            new_label = old_label
        else:
            try:
                new_label = text_labels[old_label]
            except KeyError:
                new_label = old_label

        plt.setp(label, color=text_color_map[old_label])
        label.set_text(new_label)

    return leg

def infer_cmap_params(plot_data, vmin=None, vmax=None, cmap=None,
                      center=None, robust=2.0, extend=None,
                      levels=None, filled=True, cnorm=None, **kwargs):
    """
    Use some heuristics to set good defaults for colorbar and range.

    Adapted from Seaborn:
    https://github.com/mwaskom/seaborn/blob/v0.6/seaborn/matrix.py#L158

    and xarray:
    https://github.com/xarray/xarray/blob/master/xarray/plot/plot.py#L253

    However, note that the "robust" parameter is handled differently here;
    instead of the typical behaior, it takes the percentile for the
    *minimum* value (b/t 0 and 100), and uses 100-robust for the maximum.

    Parameters
    ----------
    plot_data : Numpy array or DataArray
    vmin, vmax : floats
        Minimum and maximum value for coloring
    cmap : str
        Name of color map from matplotlib or seaborn
    center : float
        Value to fix at center of coloring scheme
    robust : float (default = 2.0)
        Percentile to use for inferring robust colormap limits; overridden
        by vmin/vmax
    levels : int or iterable of floats
        Either the number of levels to use or the values of all the
        level demarcations


    Returns
    -------
    cmap_params : dict
        Use depends on the type of the plotting function

    """
    import matplotlib as mpl

    # Unpack DataArray -> numpy array
    if isinstance(plot_data, xarray.DataArray):
        plot_data = plot_data.values
    calc_data = np.ravel(plot_data[~pd.isnull(plot_data)])

    # Legacy handling for old boolean robust
    if not robust:
        robust = 0. # set to min/max
    if vmin is None:
        vmin = np.percentile(calc_data, robust)
    if vmax is None:
        vmax = np.percentile(calc_data, 100 - robust)

    # Simple heuristics for whether these data should  have a divergent map
    divergent = ((vmin < 0) and (vmax > 0)) or center is not None

    # Now set center to 0 so math below makes sense
    if center is None:
        center = 0

    # A divergent map should be symmetric around the center value
    if divergent:
        vlim = max(abs(vmin - center), abs(vmax - center))
        vmin, vmax = -vlim, vlim

    # Now add in the centering value and set the limits
    vmin += center
    vmax += center

    # Choose default colormaps if not provided
    if cmap is None:
        if divergent:
            cmap = "RdBu_r"
        else:
            cmap = "cubehelix_r"
    # TODO: Allow viridis as default
    #         cmap = "viridis"
    #
    # # Allow viridis before matplotlib 1.5
    # if cmap == "viridis":
    #     cmap = _load_default_cmap()

    # Handle discrete levels
    if levels is not None:
        if isinstance(levels, int):
            ticker = mpl.ticker.MaxNLocator(levels)
            levels = ticker.tick_values(vmin, vmax)
        vmin, vmax = levels[0], levels[-1]

    if extend is None:
        extend = _determine_extend(calc_data, vmin, vmax)

    if levels is not None:
        cmap, cnorm = _build_discrete_cmap(cmap, levels, extend, filled)

    return dict(vmin=vmin, vmax=vmax, cmap=cmap, extend=extend,
                levels=levels, cnorm=cnorm)

def _determine_extend(calc_data, vmin, vmax):
    """ Infer the "extend" parameter based on the settings of
    vmin and vmax """
    extend_min = calc_data.min() < vmin
    extend_max = calc_data.max() > vmax
    if extend_min and extend_max:
        extend = 'both'
    elif extend_min:
        extend = 'min'
    elif extend_max:
        extend = 'max'
    else:
        extend = 'neither'
    return extend

def _build_discrete_cmap(cmap, levels, extend, filled):
    """
    Build a discrete colormap and normalization of the data.
    """
    import matplotlib as mpl

    if not filled:
        # non-filled contour plots
        extend = 'neither'

    if extend == 'both':
        ext_n = 2
    elif extend in ['min', 'max']:
        ext_n = 1
    else:
        ext_n = 0

    n_colors = len(levels) + ext_n - 1
    pal = _color_palette(cmap, n_colors)

    new_cmap, cnorm = mpl.colors.from_levels_and_colors(
        levels, pal, extend=extend)
    # copy the old cmap name, for easier testing
    new_cmap.name = getattr(cmap, 'name', cmap)

    return new_cmap, cnorm

def _color_palette(cmap, n_colors):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors_i = np.linspace(0, 1., n_colors)
    if isinstance(cmap, (list, tuple)):
        # we have a list of colors
        pal = color_palette(cmap, n_colors=n_colors)
    elif isinstance(cmap, str):
        # we have some sort of named palette
        try:
            pal = color_palette(cmap, n_colors=n_colors)
        except ValueError:
            # ValueError is raised when seaborn doesn't like a colormap (e.g. jet)
            # if that fails, use matplotlib
            try:
                # is this a matplotlib cmap?
                cmap = plt.get_cmap(cmap)
            except ValueError:
                # or maybe we just got a single color as a string
                cmap = ListedColormap([cmap], N=n_colors)
            pal = cmap(colors_i)
    else:
        # cmap better be a LinearSegmentedColormap (e.g. viridis)
        pal = cmap(colors_i)

    return pal

# Color manipulation functions - https://gist.github.com/ebressert/32806ac1c95461349522
import colorsys
def alter(color, col, factor=1.1):
    color = np.array(color)
    color[col] = color[col]*factor
    color[color > 1] = 1
    color[color < 0] = 0
    return tuple(color)
def rgb2hls(color):
    return colorsys.rgb_to_hls(*color)
def hls2rgb(color):
    return colorsys.hls_to_rgb(*color)
def lighten(color, increase=0.2):
    return hls2rgb(alter(rgb2hls(color), 1, factor=1+increase))
def darken(color, decrease=0.2):
    return hls2rgb(alter(rgb2hls(color), 1, factor=1-decrease))
def saturate(color, increase=0.2):
    return hls2rgb(alter(rgb2hls(color), 2, factor=1+increase))
def desaturate(color, decrease=0.2):
    return hls2rgb(alter(rgb2hls(color), 2, factor=1-decrease))
