
from matplotlib.colors import from_levels_and_colors, Normalize
from matplotlib.pyplot import get_cmap, colorbar, savefig
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np

from .. convert import cyclic_dataarray

__all__ = [ 'add_colorbar', 'make_colors', 'save_figure' ]

PLOTTYPE_ARGS = {
    'pcolormesh': dict(linewidth='0'),
    'pcolor': dict(linewidth='0'),
    'contourf': dict(extend='both'),
}

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


def add_colorbar(ax, mappable):
    """ Add a colorbar into an existing axis. """
    cb = colorbar(mappable, ax=ax, pad=0.075, orientation='horizontal')
    return cb  
    
def set_title(ax, varname, units, l=None, r=None):
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