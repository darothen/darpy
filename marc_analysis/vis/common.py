
import cartopy.crs as ccrs

from matplotlib.colors import from_levels_and_colors, Normalize
from matplotlib.pyplot import get_cmap, colorbar, savefig
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from seaborn.apionly import color_palette

import numpy as np
import pandas as pd
import xray

from .. convert import cyclic_dataarray


PLOTTYPE_ARGS = {
    'pcolormesh': dict(linewidth='0'),
    'pcolor': dict(linewidth='0'),
    'contourf': dict(extend='both'),
}

def get_projection(name, **kwargs):
    """ Instantiate a Cartopy coordinate reference system for
    constructing a GeoAxes object.

    """
    return ccrs.__dict__[name](**kwargs)


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
        axes = fig.get_axes()

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
                             sharex=True, sharey=True, squeeze=False,
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

def set_labels(ax, xlabel=None, ylabel=None):
    """ Add x- or y-axis labels to an axis instance. """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax
    
def save_figure(root, suffix="", fig=None, qual="quick"):
    """ Save a figure with presets for different output
    qualities.
    
    """

    fig_dict = {'bbox_inches': 'tight'}
    if qual == "production":
        format = "png"
        fig_dict['dpi'] = 600
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

def colortext_legend(text_color_map, ax, text_labels=None, **kwargs):
    """ Add a custom-built legend to a plot, where all the items in
    the legend have colored text corresponding to colored elements
    in the figure.

    Parameters:
    -----------
    text_color_map : dict
        A mapping from labels -> colors which will be used to construct
        the patch elements to include in the legend.
    ax : Axes
        The axes instance on which to add the legend.
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

    for text, color in text_color_map.items():
        legend_elems.append(
            ( mpatches.Rectangle((0, 0), 1, 1,
                                 facecolor='none',
                                 edgecolor='none'),
              text )
        )

    # Set up a legend with colored-text elements
    elems, texts = zip(*legend_elems)
    ax.legend(elems, texts, **kwargs)
    leg = ax.get_legend()

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
                      center=None, robust=False, extend=None,
                      levels=None, filled=True, cnorm=None):
    """
    Use some heuristics to set good defaults for colorbar and range.

    Adapted from Seaborn:
    https://github.com/mwaskom/seaborn/blob/v0.6/seaborn/matrix.py#L158

    and xray:
    https://github.com/xray/xray/blob/master/xray/plot/plot.py#L253

    Parameters
    ----------
    plot_data : Numpy array or DataArray
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


    Returns
    -------
    cmap_params : dict
        Use depends on the type of the plotting function

    """
    ROBUST_PERCENTILE = 2.0
    import matplotlib as mpl

    # Unpack DataArray -> numpy array
    if isinstance(plot_data, xray.DataArray):
        plot_data = plot_data.values
    calc_data = np.ravel(plot_data[~pd.isnull(plot_data)])

    if vmin is None:
        vmin = np.percentile(calc_data, ROBUST_PERCENTILE) if robust else calc_data.min()
    if vmax is None:
        vmax = np.percentile(calc_data, 100 - ROBUST_PERCENTILE) if robust else calc_data.max()

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