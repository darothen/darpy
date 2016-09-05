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

from . utilities import shift_lons

_MASKS = None
_OCEAN_SHAPE = None
_QUAAS_REGIONS = None


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
