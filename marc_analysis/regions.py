""" Utilities for extracting regions from MARC datasets

"""
import pkg_resources
import warnings

try:
    import geopandas
    from shapely.geometry import asPoint, MultiPolygon
    from shapely.prepared import prep
except ImportError:
    warnings.warn("Unable to load geopandas/shapely; ocean shape mask"
                  " not available.")
import numpy as np
import xarray
from xarray import DataArray, Dataset

from . utilities import shift_lons, shuffle_dims

_LANDSEA_MASK = None
_MASKS = None
_OCEAN_SHAPE = None
_QUAAS_REGIONS = None

__all__ = ['_load_resource_nc', 'landsea_mask']

def _load_resource_nc(name, resource=None):
    """ Load a named netCDF resource file. """

    try:
        _resource_fn = pkg_resources.resource_filename(
            "marc_analysis", "data/{}.nc".format(name)
        )

        resource = xarray.open_dataset(_resource_fn, decode_cf=False,
                                       mask_and_scale=False,
                                       decode_times=False).squeeze()
    except RuntimeError:  # xarray throws this if a file is not found
        warnings.warn("Unable to locate `{}` resource.".format(name))

    return resource


def extract_feature(ds, feature='ocean'):
    """ Extract a masked dataset with only the requested feature
    available for analysis.

    Parameters
    ----------
    ds : Dataset or DataArray
        The Dataset or DataArray to mask
    feature : str
        A string, either "ocean" or "land" indicating which
        feature to preserve in the dataset; the inverse of this feature
        will be masked.

    Returns
    -------
    masked_ds : Dataset or DataArray
        The original Dataset or DataArray with the inverse of the requested
        feature masked out.
    """

    global _MASKS

    _FEATURE_MAP = {
        'ocean': 0., 'land': 1., 'ice': 2.,
    }
    feature_key_str = " ".join(["'%s'" % s for s in _FEATURE_MAP])

    if not (feature in _FEATURE_MAP):
        raise ValueError("Expected one of [%s] as feature; got '%s'"
                         % (feature_key_str, feature))

    if _MASKS is None:
        _MASKS = _load_resource_nc("masks")
    if _MASKS is None:  # still?!?!? Must be broken/unavailable
        raise RuntimeError("Couldn't load masks resource")

    mask = (_MASKS['ORO'] == _FEATURE_MAP[feature])
    return ds.where(mask)


## Generalized landsea masks
# NCL has a contributed function, "landsea_mask", which takes an arbitrary
# lat-lon grid system and provides masks. It works by essentially performing a look-up
# of the provided gridpoints on a 1-degree grid reference dataset. But, it's
# very fast and convenient.
def landsea_mask(ds, feature='ocean'):
    """ Generate a lat-lon land-vs-sea mask for the lat-lon coordinate system
    in an arbitrary DataSet.

    This function cribs heavily from an NCL function -
        http://www.ncl.ucar.edu/Document/Functions/Shea_util/landsea_mask.shtml

    Instead of doing any sort of interpolation or re-gridding, we do a naive
    lookup of each grid-cell's coordinate on a 1-degree reference dataset
    where the masks have been pre-computed. However, this should generalize to
    an arbitrary grid.

    This routine expects that the latitude and longitude in your dataset obey
    certain conditions:

    1) lat: monotonically increasing from -90 to 90 (EQ at 0)
    2) lon: monotonically increasing from 0 to 360 (PM at 0)

    Parameters
    ----------
    ds : Dataset
        The Dataset to compute the mask for
    feature : str
        A string indicating either "land" or "ocean".

    Returns
    -------
    DataArray 

    """

    global _LANDSEA_MASK

    # Check to see if resource is loaded
    if _LANDSEA_MASK is None:
        _LANDSEA_MASK = _load_resource_nc("landsea")
    if _LANDSEA_MASK is None:  # still?!? Something terrible happened...
        raise RuntimeError("Couldn't load 'landsea' resource")

    feature_map = {
        'ocean': 0, 'land': 1
    }
    if feature not in feature_map.keys():
        raise ValueError("Unknown 'feature' provided to mask lookup")

    # Copy the coordinate system from your original Dataset
    lon1d = ds.lon.values.copy()
    lat1d = ds.lat.values.copy()
    orig_coords = [lon1d.copy(), lat1d.copy()]
    
    # Correct dimensions - eliminate > 360 longitudes, shift lats to (0, 180)
    lon1d[lon1d < 0] = lon1d[lon1d < 0] + 360.
    lat1d = ds.lat.values + 90.

    # Construct a 2D grid for the original coordinate system and chop to 
    # integers (which approximate indices on a 180x360 grid)
    lat2d, lon2d = np.meshgrid(lat1d, lon1d)
    lat2d = lat2d.astype(int)
    lon2d = lon2d.astype(int)

    # Correct edge/loop indices
    lat2d[lat2d < 0] = 0
    lat2d[lat2d > 179] = 179
    lon2d[lon2d >= 360] = 0

    # Construct mask. Our reference landsea dataset has coordinates in the order
    # (lat, lon), so we need to be careful about the order in which we constructed
    # and will now pass the indices for masking
    mask_raw = _LANDSEA_MASK['LSMASK'].data[[lat2d, lon2d]] == feature_map[feature]
    mask_da = DataArray(mask_raw, coords=orig_coords, dims=['lon', 'lat'])
    
    return mask_da
    

#################################
# Quaas regional analysis

Quass_region_names = {
    # id: (name, (label lon, label lat))
    'NPO': 3,
    'NAM': 4,
    'NAO': 5,
    'EUR': 6,
    'ASI': 7,
    'TPO': 8,
    'TAO': 9,
    'AFR': 10,
    'TIO': 11,
    'SPO': 12,
    'SAM': 13,
    'SAO': 14,
    'SIO': 15,
    'OCE': 16,
}

def extract_Quaas_region(ds, region):
    """ Extract one of the regions from Quaas et al (2009) from a given dataset.

    Parameters
    ----------
    ds : Dataset or DataArray
        The Dataset or DataArray to mask
    region : str
        A string representing the region to extract, following Quaas et al (2009)

    Returns
    -------
    The original Dataset or DataArray with all but the indicated region masked
    out.

    """

    global _QUAAS_REGIONS

    if not (region in Quass_region_names):
        raise ValueError("Don't know region '{}'".format(region))

    # Check to see if resource is loaded
    if _QUAAS_REGIONS is None:
        _QUAAS_REGIONS = _load_resource_nc("quaas_regions")
    if _QUAAS_REGIONS is None:  # still?!? Something terrible happened...
        raise RuntimeError("Couldn't load 'quaas_regions' resource")

    mask = (_QUAAS_REGIONS['reg'] == Quass_region_names[region])
    return ds.where(mask, drop=True)


#################################
# Point-by-point ocean extraction

def _is_in_ocean(p, oceans):
    """ Returns 'true' if the supplied shapely.geometry.Point is located in
    an ocean. """
    return oceans.contains(p)


def _get_ocean_shapefile():
    """ Load the ocean basins shapefile. """

    global _OCEAN_SHAPE

    if _OCEAN_SHAPE is None:
        try:
            ocean_shp_fn = pkg_resources.resource_filename(
                "marc_analysis", "data/ne_110m_ocean.shp"
            )
            _OCEAN_SHAPE = geopandas.read_file(ocean_shp_fn)
            _OCEAN_SHAPE = MultiPolygon(_OCEAN_SHAPE.geometry.values.tolist())
            _OCEAN_SHAPE = prep(_OCEAN_SHAPE)
        except OSError:
            warnings.warn("Unable to locate oceans shapefile; ocean"
                          " point mask is not available")

    return _OCEAN_SHAPE


def mask_ocean_points(dataset, oceans=None, pt_return=False,
                      longitude='lon', latitude='lat'):

    if oceans is None:
        oceans = _get_ocean_shapefile()
    if oceans is None:  # still?!? Must be broken
        raise RuntimeError("Couldn't load default ocean shapefile")

    lons, lats = dataset[longitude], dataset[latitude]
    if isinstance(dataset, (xarray.Dataset, xarray.DataArray)):
        lons.values = shift_lons(lons.values)
    else:
        lons = shift_lons(lons)

    points = [asPoint(point) for point in np.column_stack([lons, lats])]
    if pt_return:
        return points

    in_ocean = [_is_in_ocean(p, oceans) for p in points]
    return in_ocean
