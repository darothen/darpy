# MARC Activation-Aerosol Indirect Effect Experiment

Source code repository for toolkit and analyses/exploration of a CESM/MARC 
experiment.

## Setup

After cloning this repository, setup a `conda` environment by executing
the command

```
cd [marc_aie_dir]
conda env create
```

which you can activate by calling

```
source activate marc_aie
```

If you don't already have Anaconda installed, you should install 
[Miniconda](http://conda.pydata.org/miniconda.html), a lightweight
version with minimal dependencies. 

## Toolkit Overview

Several utilities are provided in the `marc_aie` package to help process and
analyze the model output....

The scripts in this top-level folder hold all the tools for analyzing a simple CESM/MARC aerosol indirect effect experiment. A package, `marc_aie`, contains a number of useful utility functions for extracting, loading, and plotting data from the experiment. Other available folders (e.g. `global_greedy_activation`, `indirect_forcing`, etc) contain scripts for particular analyses.


## Experiment Details

The experiment contained eight 10-year simulations of the CESM/MARC coupled to a slab ocean, using 4 different activation schemes and 2 different aerosol emissions cases. The atmosphere output from each scheme was copied to legion; the location of this data is specified in the `cases_setup.py` script.

---

For each analysis, the work is divided up into two phases:

1. **Extraction of relevant variables** - rather than compute all fields in memory, it was decided to have a separate script process the raw model output and extract the necessary fields.
2. **Visualization** - all the visualizations will be contained in their own scripts, which could be automated with a Makefile or some other script later on.

In general, extraction should (at most) produce files with dimensions (time, lat, lon).

  **NOTE** -
  should upgrade the scripts to use the CDO interface through Python (https://code.zmaw.de/projects/cdo/wiki/Cdo%7Brbpy%7D); even better, wrap it with multiprocessing pools so that things run concurrently!

## Analyses
 
1) zonal_averages: reproduces figure 1 of [Gantt et al, 2014], with potential variations
2) indirect_forcing:


## MARC aerosol processing

Currently, two different sets of analyses look at the MARC aerosol data:

1. `aerosol_dists`
2. `global_greedy_activation`

They use a different system for extracting data; specifically, the pipeline is encoded in an IPython notebook in each folder. The order listed above indicates the order the pipelines were written, and each successive version implements some improvements over its predecessor, as additional constraints or data were necessary for certain analyses.

## Third-party dependencies

I'm leaning pretty heavily on third-party libraries to make life easier with this analysis package. For a complete summary, check the [**environment.yaml**](environment.yaml) file.

- [cartopy](http://scitools.org.uk/cartopy/docs/latest/) - visualization of maps and cartographic/geographic datasets
- [seaborn](http://stanford.edu/~mwaskom/software/seaborn/) - wrapper for matplotlib with fantastic aesthetic configuration and multi-factor plot analyses/layouts
- [windspharm](http://ajdawson.github.io/windspharm/) - wrapper for NCAR library of calculations of wind fields in spherical geometry
- [xray](http://xray.readthedocs.org) - package for manipulating structured data like NetCDF

## TODO:

1. Create IPython environment that automatically imports the snippet

    ```python
    import sys
    sys.path.insert(0, "../")
    ```

    for all notebooks

2. Contribute `xray` interface to windspharm

3. Interpolate vertical date coordinate from hybrid pressure level to mandatory pressure levels, a là AMWG (ncl method `vinth2p`). Should probably wait until xray supports interpolation over a dimension instead of just nearest-neighbor lookups, although it's entirely doable simply by building new arrays.

4. For some reason, `CESMVar` objects matriculate metadata (units) for variables like **time_bnds** while `CDOVar` doesn't. It probably has something to do with copying the attributes from the CESM output.


[Gantt et al, 2014]: http://www.atmos-chem-phys.net/14/7485/2014/

