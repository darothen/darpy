# Daniel's Research Analysis Toolkit

This toolkit contains useful helper functions and methods for both analyzing
and plotting climate model and other geospatial data. I've accumulated these
functions over half a decade of active research work in meteorology and
climate science; many of the methods here help automate a lot of the mundane
things necessary to re-align your data for analyses, set up figures and plots,
and much more. Nothing is particularly novel - for instance, I've spun off
projects like [experiment](https://github.com/darothen/experiment) from this
repository when they grew large and singular enough to stand alone.

Originally, this specific repository archived code for helping to analyze
CESM/MARC output.

## Toolkit Overview

[link to working](working/)

## Installation

Because this package builds on several widely-available toolkits, a conda environment is provided via `environment.yml` to manage its Python dependencies. To install, execute from the shell

```
$ conda env create -f environment.yml
```

This will create a `darpy` environment separate from your normal Python installation. This environment should be activated whenever you wish to use the toolkit. Then, we recommend installing the actual `darpy` toolkit as an ["editable" installation](http://pip-python3.readthedocs.org/en/latest/reference/pip_install.html#editable-installs) into the `darpy` environment:

```
$ source activate darpy
(darpy) $ pip install -e path/to/darpy
```

### Python Dependencies

- Python >= 3.5
- [cartopy >= 0.12](http://scitools.org.uk/cartopy/docs/latest/) - visualization of maps and cartographic/geographic datasets
- [xarray >= 0.8.2](http://xarray.pydata.org/en/stable/) - package for manipulating structured data like NetCDF
- seaborn - wrapper for matplotlib with fantastic aesthetic configuration and multi-factor plot analyses/layouts
- matplotlib
- numpy
- scipy

and more

### Other Dependencies

- [netCDF Operators](http://nco.sourceforge.net/) - command line tools for processing netCDF files
