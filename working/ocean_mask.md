```python
import marc_analysis as ma
import xray

%matplotlib inline
import matplotlib.pyplot as plt
```

```python
data = xray.open_dataset(
    "/Users/daniel/Desktop/MARC_AIE/F2000/"
    "arg_comp/arg_comp.cam2.h0.0008-04.nc")
```

Use the total precipitation data as a test dataset.

```python
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
    

    mask = (ma.masks['ORO'] == _FEATURE_MAP[feature])
    return ds.where(mask)
    
d = extract_feature(data, 'land')
ma.geo_plot(d['TS'].squeeze())
```

```python
from marc_analysis.utilities import area_grid
from xray import DataArray
import numpy as np

def global_avg(data, weights=None, dims=['lon', 'lat']):
    """ Compute (area-weighted) global average over a DataArray
    or Dataset. If `weights` are not passed, they will be computed
    by using the areas of each grid cell in the dataset.

    .. note::
        Handles missing values (nans and infs).

    """

    if weights is None: # Compute gaussian weights in latitude
        weights = area_grid(data.lon, data.lat)
        gw = weights.sum('lon')
        weights = 2.*gw/gw.sum('lat')

    if isinstance(data, DataArray):

        is_null = ~np.isfinite(data)
        if np.any(is_null):
            data = np.ma.masked_where(is_null, data)
                                      
        # return np.average(data, weights=weights)
        # return np.sum(data*weights)/np.sum(weights)
        return (data*weights/weights.sum()).sum(dims)

ds = data['PRECC']

for f in ['land', 'ocean']:
    d = extract_feature(ds, f).squeeze()
    print (f)
    print (ds.mean()) 
#     print (global_avg(ds))
    print (d.mean()) 
#     print (global_avg(d))
```

Use the Natural Earth Data physical boundaries shapefiles to help extract the
continent/land mask.

```python
import geopandas
import os

continents = geopandas.read_file("/Users/daniel/Downloads/ne_110m_land/ne_110m_land.shp")
oceans = geopandas.read_file("/Users/daniel/Downloads/ne_110m_ocean/ne_110m_ocean.shp")

ocean_shapes = [(shape, n) for n, shape in enumerate(oceans.geometry)]
contient_shapes = [(shape, n) for n, shape in enumerate(continents.geometry)]
```

```python
o1 = oceans.geometry.ix[1]
lons, lats = o1.exterior.coords.xy

lons
```

Rasterize example using `rasterio` and `geopandas` following [Stephan
Hoyer](https://github.com/xray/xray/issues/501)

```python
from rasterio import features
from affine import Affine
import numpy as np


def shift_lons(lon):
    """ Convert a vector of longitudes in the 0 < lon < 360 coordinate
    system to longitudes in the -180 < 0 < 180 coordinate system. """
    lon = np.asarray(lon)
    mask = np.where(lon > 180)
    lon[mask] = -1.*(360. - lon[mask])

    return lon
    
def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def rasterize(shapes, coords, fill=np.nan, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude
    arrays.
    """
    lon = coords['lon'].data
    lat = coords['lat'].data
    
    # Check if in 0 < lon < 360 coordinates; if so, convert to
    # -180 < lon < 180
    if np.any(lon > 180): 
        lon = shift_lons(lon)
        
    transform = transform_from_latlon(lat, lon)
    out_shape = (len(lat), len(lon))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    return xray.DataArray(raster, coords=coords, dims=('lat', 'lon'))
```

```python
ocean_mask = rasterize(ocean_shapes, ds.coords)
print(ocean_mask.lon)

%matplotlib inline
import matplotlib.pyplot as plt

plt.figure()
ma.geo_plot(data.where(~np.isnan(ocean_mask))['PRECC'].squeeze())
plt.figure()
ma.geo_plot(data['PRECC'].squeeze())
```

```python
ds.lat.min()
```

```python

```
