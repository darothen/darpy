""" Utilities developed for Aerocom activation work, that could be useful
someday to generalize for this library. """

import marc_analysis as ma
import pyresample as pr

import xarray as xr

# Set up for re-gridding
marc_masks = ma.regions._load_resource_nc("masks")
lon = marc_masks.lon.values.copy() - 180
lat = marc_masks.lat.values.copy()
lats, lons = np.meshgrid(lat, lon)
marc_grid = pr.geometry.GridDefinition(lons, lats)
@ma.preserve_attrs
def regrid_da_to_marc(da):
    """ Re-grid a DataArray to a 2-degree reference MARC grid. """
    lon = da.lon.values.copy() - 180
    lat = da.lat.values.copy()
    lats, lons = np.meshgrid(lat, lon)
    da_grid = pr.geometry.GridDefinition(lons, lats)

    da = ma.shuffle_dims(da, ['lon', 'lat'])
    arr = da.data
    regrid_arr = pr.kd_tree.resample_nearest(
        da_grid, arr, marc_grid, radius_of_influence=500000
    )

    # Construct re-gridded DataArray
    new_coords = {'lon': marc_masks.lon, 'lat': marc_masks.lat}
    for coord in set(da.dims).difference(['lon', 'lat']):
        new_coords[coord] = da[coord]
    regrid_da = xr.DataArray(regrid_arr, coords=new_coords, dims=da.dims)

    return regrid_da


def standard_parser(description, default_tape="2d_monthly"):
    """ Generate a standard parser, with defaults for "interactive" mode
    and an optional output filename. """
    parser = ArgumentParser(description=description,
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Enable interactive plotting mode")
    parser.add_argument("-o", "--output", type=str, metavar="[out_fn]",
                        help="Name of output file, else a default will be used")
    parser.add_argument("-P", "--progress", action="store_true",
                        help="If applicable, display a progress bar during work")
    parser.add_argument("-t", "--tape", type=str, default=default_tape,
                        help="Output tapes/experiment to load.")

    return parser
