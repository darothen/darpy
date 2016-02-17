# MARC Analysis Package

Source code repository for toolkit and analyses/exploration of a CESM/MARC experiment. This package is designed to automate much of the boilerplate code necessary to process, extract, and analyze variables from output from either a single CESM/MARC simulation or a designed experiment.

## Toolkit Overview

[link to working](working/)

## Installation

Because this package builds on several widely-available toolkits, a conda environment is provided via `environment.yml` to manage its Python dependencies. To install, execute from the shell

```
$ conda env create -f environment.yml
```

This will create a `marc_analysis` environment separate from your normal Python installation. This environment should be activated whenever you wish to use the toolkit. Then, we recommend installing the actual `marc_analysis` toolkit as an ["editable" installation](http://pip-python3.readthedocs.org/en/latest/reference/pip_install.html#editable-installs) into the `marc_analysis` environment:

```
$ source activate marc_analysis
(marc_analysis) $ pip install -e path/to/marc_analysis
```

### Python Dependencies

- Python >= 3.4
- [cartopy >= 0.12](http://scitools.org.uk/cartopy/docs/latest/) - visualization of maps and cartographic/geographic datasets
- [xarray >= 0.7](http://xarray.pydata.org/en/stable/) - package for manipulating structured data like NetCDF
- seaborn - wrapper for matplotlib with fantastic aesthetic configuration and multi-factor plot analyses/layouts
- matplotlib
- numpy

### Other Dependencies

- [netCDF Operators](http://nco.sourceforge.net/) - command line tools for processing netCDF files

## Usage

Implementing this package into a typical CESM/MARC analysis workflow requires three steps. First - and this is probably already done naturally when you download data to your home server - you need to layout your data in a hierarchy of directories which correspond to the factors in your experiment. For instance, suppose you had a two-factor experiment where you used three different physical parameterizations with two different emissions cases. Your layout may look something like

    DATA_FOLDER/
        emis_1/
            scheme_A/
            scheme_B/
            scheme_C/
        emis_2/
            scheme_A/
            scheme_B/
            scheme_C/

All the data files you care about live in the leaf directories. Each level is called a *case* in the nomenclature adopted by this package, and they are ordered from the left-most directory to the right-most or inner-most.

Once your data is organized like this, you must create an **Experiment** which encapsulates this data using **Case**s. This can be done very simply:

```python
import marc_analysis as ma

# Set up the experiment details
cases = [
    ma.Case('emis', 'greenhouse gas emissions scenario',
            [ 'emis_1', 'emis_2', ]),
    ma.Case('scheme', 'land carbon uptake scheme',
            [ 'scheme_A', 'scheme_B', 'scheme_C', ]),
]
exp = ma.Experiment('my_carbon_exp', cases,
                    data_dir=FULL_PATH_TO_DATA_FOLDER)

```

The final step involves identifying variables to process, extract, and load. This could vary based on your application, and there are many ways built-in to the module for combining variables, applying functions or netCDF/climate data operators (NCO/CDO), and manipulating. However, for a simple example, some functionality enabling all the CESM default history vars is included within the module:

```python
# Use the default surface temperature output
ts = ma.CESMVar("TS")

# Extract from the full dataset
ts.extract(exp)

# Load into memory
ts.load_data(exp)

# Create a master dataset
ts.master = ma.create_master(ts, exp)
```

The documentation goes into more detail on how these functions work, but in order:

1. We create a **Var** object, which lets us define how to manipulate a variable at a very low level

2. Extract that modified variable from the original dataset (it will save in a location defined within your **Experiment**, which can be customized)

3. Load the data into memory. This populates a few read-only attributes in the **Var**, but fundamentally it stores all the data for each case in a dictionary so that they can be manipulated individually (or simultaneously with the `Var.apply()` method)

4. Create a "master dataset"; this is a special `xray` dataset where each of the **Case**s defined previously is promoted to its own coordinate dimension, so that differences between cases can very easily be computed

From there on, you can analyze and process the variables. But all the boilerplate code for loading them into memory has been taken care of behind the scenes!

## TODO

2. Contribute `xray` interface to windspharm

3. Interpolate vertical date coordinate from hybrid pressure level to mandatory pressure levels, a là AMWG (ncl method `vinth2p`). Should probably wait until xray supports interpolation over a dimension instead of just nearest-neighbor lookups, although it's entirely doable simply by building new arrays.

4. For some reason, `CESMVar` objects matriculate metadata (units) for variables like **time_bnds** while `CDOVar` doesn't. It probably has something to do with copying the attributes from the CESM output.

5. Implement logging module instead of printing to console so that messages can be suppressed when desired

6. Context manager functionality so that you don't have to constantly pass the experiemnt argument when extracting / loading a variable

7. Build conda package for NCO, add to environment

8. Lazy open from simulation archive without extracting?

9. Replace NCO wrapper method with direct binding via [https://github.com/nco/pynco](https://github.com/nco/pynco)