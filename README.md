# MARC Analysis Package

Source code repository for toolkit and analyses/exploration of a CESM/MARC
experiment. This package is designed to automate much of the boilerplate
code necessary to process, extract, and analyze variables from output from
either a single CESM/MARC simulation or a designed experiment.

## Toolkit Overview

## Dependencies

- Python >= 3.1
- [netCDF Operators](http://nco.sourceforge.net/) - command line tools for processing netCDF files
- [cartopy >= 0.12](http://scitools.org.uk/cartopy/docs/latest/) - visualization of maps and cartographic/geographic datasets
- [xray >= 0.5.2](http://xray.readthedocs.org) - package for manipulating structured data like NetCDF
- seaborn - wrapper for matplotlib with fantastic aesthetic configuration and multi-factor plot analyses/layouts
- matplotlib
- numpy

## Usage

## TODO

1. Re-factor logic so that `Experiment` objects also load variable data in a simple, concise fashion

2. Contribute `xray` interface to windspharm

3. Interpolate vertical date coordinate from hybrid pressure level to mandatory pressure levels, a là AMWG (ncl method `vinth2p`). Should probably wait until xray supports interpolation over a dimension instead of just nearest-neighbor lookups, although it's entirely doable simply by building new arrays.

4. For some reason, `CESMVar` objects matriculate metadata (units) for variables like **time_bnds** while `CDOVar` doesn't. It probably has something to do with copying the attributes from the CESM output.

5. Implement logging module instead of printing to console so that messages can be suppressed when desired
