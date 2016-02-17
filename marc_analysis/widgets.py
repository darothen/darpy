""" Collection of interactive widgets/plotters for perusing
and analyzing results.

"""

import matplotlib.pyplot as plt
import numpy as np
from xarray import Dataset

from IPython.html import widgets
from IPython.display import clear_output, display

from . utilities import area_grid
from . plot import save_figure, pd_vs_pi_summary

def four_panel_horizontal(var, src_dir=WORK_DIR,
                          extract=True, extr_kwargs={},
                          load=True, load_kwargs={},
                          debug=False):

    var_name = var.varname

    # Extract the variables we want to plot, if necessary
    if extract:
        var.extract(src_dir,
                    act_cases=CASES_ACT, aer_cases=CASES_AER,
                    **extr_kwargs)
    else:
        if debug: print("skipping extraction")

    # Load into memory
    if load:
        var.load_datasets(src_dir, act_cases=CASES_ACT,
                          aer_cases=CASES_AER)
    else:
        if debug: print("skipping loading")

    # Coerce to DataArray
    sample = var.data[CASES_ACT[0], CASES_AER[0]]
    if isinstance(sample, Dataset):
        var.apply(lambda ds: ds[var.varname])
        if debug:
            print("converted to dataarray from %r" % type(sample))
    data_dict = var.data

    # Set up area grid for weighted global averages
    ds = data_dict[CASES_ACT[0], CASES_AER[0]]
    area = area_grid(ds.lon, ds.lat)
    total_area = np.sum(area.data)

    # Compute PD-PI difference
    max_diff = 0.
    max_level = 0.
    if debug: print("Reading...")
    for act in CASES_ACT:
        if debug: print("   ", act)
        exp_pd = data_dict[act, "F2000"]
        exp_pi = data_dict[act, "F1850"]
        abs_diff = exp_pi - exp_pd

        case_max_diff = np.max(abs_diff.data)
        if debug: print("   max diff", case_max_diff)
        if case_max_diff > max_diff:
            max_diff = case_max_diff

        _, case_max_level = min_max_cubes(exp_pd, exp_pi)
        if debug: print("   max level", case_max_level)
        if np.abs(case_max_level) > max_level:
            max_level = case_max_level

    if debug:
        print("Final max diff/level -", max_diff, max_level)

    #################################################################

    act_control = widgets.Dropdown(description="Activation Case",
                                   options=CASES_ACT)
    proj_control = widgets.Dropdown(description="Map Projection",
                                    options=["PlateCarree", "Robinson"])
    max_level_trunc = widgets.FloatText(description="Trunc level",
                                        value=max_level)

    abs_top = np.ceil(1.5*max_diff)
    step = 0.01 if abs_top <= 1. else 0.1
    if debug:
        print("absolute slider range - ", 0., abs_top)
        print("                 step - ", step)
    abs_control = widgets.FloatSlider(description="Max Abs. Difference",
                                      value=abs_top,
                                      min=0., max=abs_top,
                                      width=150., step=step)
    rel_control = widgets.IntSlider(description="Max Rel. Difference",
                                    value=50,
                                    min=0, max=150,
                                    width=150.)
    reverse_cmap = widgets.Checkbox(description="Reverse colormap",
                                    value=False, width=150.)
    colormap_text = widgets.Text(description="Colormap on PD/PI",
                                 value="cubehelix_r")
    plot_button = widgets.Button(description="Make Plot")
    save_quality = widgets.Dropdown(description="Save quality",
                                    options=["quick", "production",
                                             "vector"])
    save_filename = widgets.Text(description="Save filename",
                                 value="%s_horiz_PD-PI" % var_name)
    save_button = widgets.Button(description="Save Figure")

    form = widgets.VBox(width=1000)
    hbox_make = widgets.HBox()
    hbox_make.children = [act_control, proj_control, plot_button]
    hbox_diff = widgets.HBox()
    hbox_diff.children = [abs_control, rel_control]
    hbox_cmap = widgets.HBox()
    hbox_cmap.children = [max_level_trunc, reverse_cmap, colormap_text]
    hbox_save = widgets.HBox()
    hbox_save.children = [save_quality, save_filename, save_button]
    form.children = [hbox_make, hbox_diff, hbox_cmap, hbox_save]

    def _driver(*args, **kwargs):
        if not debug: clear_output()

        # Save the fig for later reference
        global fig
        fig = plt.figure("4panel_PD-PI", figsize=(12, 8))
        fig.clf()
        fig.set_rasterized(True)

        exp_pd = data_dict[act_control.value, "F2000"]
        exp_pi = data_dict[act_control.value, "F1850"]
        pd_vs_pi_summary(exp_pd, exp_pi, area,
                         max_level_trunc.value,
                         abs_control.value,
                         rel_control.value,
                         reverse_cmap.value, colormap_text.value,
                         proj_control.value, fig)

    def _savefig(*args, **kwargs):
        global fig
        save_figure(save_filename.value+".4panel",
                    fig=fig, qual=save_quality.value)

    plot_button.on_click(_driver)
    save_button.on_click(_savefig)

    return display(form)
