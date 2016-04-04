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


def _get_masks():
    """ Load the orography masks dataset. """

    global _MASKS

    if _MASKS is None:

        try:
            _masks_fn = pkg_resources.resource_filename(
                "marc_analysis", "data/masks.nc"
            )

            _MASKS = xarray.open_dataset(_masks_fn, decode_cf=False,
                                         mask_and_scale=False,
                                         decode_times=False).squeeze()
        except RuntimeError:  # xarray throws this if a file is not found
            warnings.warn("Unable to locate `masks` resource.")

    return _MASKS


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
    _FEATURE_MAP = {
        'ocean': 0., 'land': 1., 'ice': 2.,
    }
    feature_key_str = " ".join(["'%s'" % s for s in _FEATURE_MAP])

    if not (feature in _FEATURE_MAP):
        raise ValueError("Expected one of [%s] as feature; got '%s'"
                         % (feature_key_str, feature))

    if _MASKS is None:
        _get_masks()
    if _MASKS is None:  # still?!?!? Must be broken/unavailable
        raise RuntimeError("Couldn't load masks resource")

    mask = (_MASKS['ORO'] == _FEATURE_MAP[feature])
    return ds.where(mask)


def _is_in_ocean(p, oceans):
    """ Returns 'true' if the supplied shapely.geometry.Point is located in
    an ocean. """
    return oceans.contains(p)


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
