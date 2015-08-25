""" 
Specialized plot routines
"""

from numpy import ceil, linspace, max
import numpy as np

from functools import partial

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .. utilities import area_grid
from .. analysis import min_max_cubes, global_avg
from . glob import global_plot

__all__ = [ 'pd_vs_pi_summary', ]

def pd_vs_pi_summary(exp_pd, exp_pi, area=None,
                     level_trunc=None, abs_lim=None, pct_lim=None,
                     reverse_cmap=False, cmap="cubehelix_r", 
                     projection='PlateCarree', fig=None):
    """ Panel of four plots -
    
    UL: exp, PI      |        UR: exp, PD
    -----------------+-------------------
    LL: exp, PD-PI   | LR: exp, rel PD-PI
    
    Expect data as cubes.

    """

    def _fmt_str(val):
        val = np.abs(val)

        if 1. <= val < 1e4:
            return "%.1f"
        elif 0.01 < val < 1.:
            return "%.2f"
        else:
            return "%1.2e"

    # Set up area grid for weighted global averages
    if area is None:
        area = area_grid(exp_pd.lon, exp_pd.lat)
    _global_avg = partial(global_avg, weights=area)
    
    def _middle_colorbar(mappable, label, nticks=8):
        cax = plt.gcf().add_axes([0.125, 0.51, 0.775, 0.025])
        colorbar = plt.colorbar(mappable, cax, 
                                orientation='horizontal')
        colorbar.set_label(label)

        colorbar.locator = MaxNLocator(nticks)
        colorbar.update_ticks()

        return cax, colorbar

    def _left_colorbar(mappable, label):

        cax = plt.gcf().add_axes([0.05, 0.17, 0.02, 0.27])
        colorbar = plt.colorbar(mappable, cax, orientation='vertical')
        colorbar.ax.yaxis.set_ticks_position('left')
        colorbar.set_label(label)

        return cax, colorbar
    
    def _right_colorbar(mappable, label):
        cax = plt.gcf().add_axes([0.925, 0.17, 0.02, 0.27])
        colorbar = plt.colorbar(mappable, cax, orientation='vertical')
        colorbar.set_label(label)

        return cax, colorbar  

    abs_diff = exp_pd - exp_pi
    abs_diff.attrs['long_name'] = "Absolute difference (PD - PI)"
    abs_diff.attrs['units'] = exp_pi.attrs['units']

    rel_diff = 100.*abs_diff/exp_pi
    rel_diff.attrs['long_name'] = "Relative difference (PD - PI)"
    rel_diff.attrs['units'] = "%"

    if reverse_cmap:
        if cmap.endswith("_r"): 
            cmap = cmap.replace("_r", "")
        else:
            cmap = cmap + "_r"

    if fig is None:
        fig = plt.figure("4panel_PD-PI", figsize=(12, 8))

    proj_inst = ccrs.__dict__[projection]()

    gp_args = dict(projection=projection, method='contourf')

    lev_min, lev_max = min_max_cubes(exp_pd, exp_pi)
    if level_trunc is not None:
        #lev_max = min([lev_max, level_trunc])
        lev_max = level_trunc
    pc_levels = linspace(lev_min, lev_max, 30)

    ax_pi = fig.add_subplot(221, projection=proj_inst)
    gp = global_plot(exp_pi, ax=ax_pi, levels=pc_levels, cmap=cmap,
                     coloring='discrete', **gp_args)
    ax_pi.set_title("PI", loc='right')
    pi_avg = _global_avg(exp_pi)
    ax_pi.set_title("avg: " + _fmt_str(pi_avg) % pi_avg, loc='left')

    ax_pd = fig.add_subplot(222, projection=proj_inst)
    global_plot(exp_pd, ax=ax_pd, levels=pc_levels, cmap=cmap,
                coloring='discrete', **gp_args)
    ax_pd.set_title("PD", loc='right')
    pd_avg = _global_avg(exp_pd)
    ax_pd.set_title("avg: " + _fmt_str(pd_avg) % pd_avg, loc='left')
    
    ## Add a colorbar for the top two plots.
    #label = exp_pi.long_name + " (%s)" % exp_pi.units
    label = ""
    cax, colorbar = _middle_colorbar(gp, label)
    
    #######

    if abs_lim is None:
        abs_lim = ceil(1.5*max(abs_diff.data))
    abs_levels = linspace(-abs_lim, abs_lim, 21)
    ax_diff = fig.add_subplot(223, projection=proj_inst)
    gp = global_plot(abs_diff, ax=ax_diff, 
                     levels=abs_levels, cmap="RdYlBu_r",
                     coloring='continuous', **gp_args)
    ax_diff.set_title("Absolute diff (PD-PI)", loc='right')
    abs_avg = _global_avg(abs_diff)
    ax_diff.set_title("avg: " + _fmt_str(abs_avg) % abs_avg, loc='left')
    cax, colorbar = _left_colorbar(gp, abs_diff.units)
    
    if pct_lim is None:
        pct_lim = 50
    rel_levels = linspace(-pct_lim, pct_lim, 21)
    ax_rel_diff = fig.add_subplot(224, projection=proj_inst)
    gp = global_plot(rel_diff, ax=ax_rel_diff,
                     levels=rel_levels, cmap="RdBu_r",
                     coloring='continuous', **gp_args)
    ax_rel_diff.set_title("Relative diff (PD-PI)", loc='right')
    rel_avg = _global_avg(rel_diff)
    ax_rel_diff.set_title("avg: " + _fmt_str(rel_avg) % rel_avg, loc='left')
    cax, colorbar = _right_colorbar(gp, rel_diff.units)

    fig.suptitle("\n" + exp_pi.long_name + " (%s)" % exp_pi.units, 
                 fontsize=16)
    
    return fig