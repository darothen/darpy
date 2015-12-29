
import pkg_resources
import warnings

import numpy as np
try:
    import geopandas
    from shapely.geometry import asPoint, MultiPolygon
    from shapely.prepared import prep
except ImportError:
    warnings.warn("Unable to load geopandas/shapely; ocean shape mask"
                  " not available.")
import xray
from xray import DataArray, Dataset

# from windspharm.standard import VectorWind
# from windspharm.tools import prep_data, recover_data, order_latdim

from . utilities import ( copy_attrs, preserve_attrs, area_grid,
                         shuffle_dims, shift_lons )

_MASKS  = None
def _get_masks():
    """ Load the orography masks dataset. """

    global _MASKS

    if _MASKS is None: 

        try:
            _masks_fn = pkg_resources.resource_filename("marc_analysis",
                                                        "data/masks.nc")
            
            _MASKS = xray.open_dataset(_masks_fn, decode_cf=False,
                                       mask_and_scale=False,
                                       decode_times=False).squeeze()
        except RuntimeError: # xray throws this if a file is not found
            warnings.warn("Unable to locate `masks` resource.")

    return _MASKS


_OCEAN_SHAPE = None
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

################################################################
## DATASET MANIPULATION FUNCTIONS

def hybrid_to_pressure(ds, stride='m', P0=100000.):
    """ Convert hybrid vertical coordinates to pressure coordinates
    corresponding to model sigma levels.

    Parameters
    ----------
    data : xray.Dataset
        The dataset to inspect for computing vertical levels
    stride : str, either 'm' or 'i'
        Indicate if the field is on the model level interfaces or
        middles for referencing the correct hybrid scale coefficients
    P0 : float, default = 1000000.
        Default reference pressure in Pa, used as a fallback.

    """

    # Grab necessary data fields
    a, b = ds['hya'+stride], ds['hyb'+stride] # A, B coefficients
    try:
        P0_ref = ds['P0'] # Reference pressure
    except KeyError:
        P0_ref = P0
    P0 = P0_ref
    PS = ds['PS'] # Surface pressure field

    pres_sigma = a*P0 + b*PS
    # Copy attributes and overwrite where different
    pres_sigma.attrs.update(PS.attrs)
    pres_sigma.attrs['long_name'] = "Pressure field on sigma levels"

    return pres_sigma

def _interp_scipy(data, pres_levs, new_pres_levs):

    """ Interpolate by aggregating all data into columns and
    applying scipy's interp1d method. """

    from scipy.interpolate import interp1d

    # Shuffle dims so that 'lev' is first for simplicity
    data = shuffle_dims(data)
    P = shuffle_dims(pres_levs)

    # Find the 'lev' axis for interpolating
    orig_shape = data.shape
    axis = data.dims.index('lev')
    nlev = orig_shape[axis]

    cols = int(np.product(data.shape)/nlev)
    nlev_new = len(new_pres_levs)

    # Re-shape original containers and create one for holding
    # interpolated data
    data_new = np.zeros((nlev_new, cols))
    temp_shape = (nlev, cols)
    data_prep = np.reshape(data.data, temp_shape)
    logP_prep = np.reshape(np.log10(P.data), temp_shape)


    for col in range(cols):

        # Create interpolater. Need to disable bounds error checking so that
        # it will automatically fill in missing values
        lin_interp = interp1d(logP_prep[:, col], data_prep[:, col],
                              axis=axis, bounds_error=False,
                              assume_sorted=True)

        # New coordinates (x'), mapped to log space
        logP_new = np.log10(new_pres_levs)

        # Compute interpolation on the new levels
        interped_logA = lin_interp(logP_new)
        data_new[:, col] = interped_logA

    final_shape = tuple([nlev_new, ] + list(orig_shape[1:]))
    data_new = np.reshape(data_new, final_shape)

    return data_new

def _interp_numpy(data, pres_levs, new_pres_levs):
    """ Interpolate all columns simultaneously by iterating over
    vertical dimension of original dataset, following methodology
    used in UV-CDAT. """

    # Shuffle dims so that 'lev' is first for simplicity
    data = shuffle_dims(data)
    pres_levs = shuffle_dims(pres_levs)

    # Find the 'lev' axis for interpolating
    orig_shape = data.shape
    axis = data.dims.index('lev')
    nlev = orig_shape[axis]

    n_sigma = nlev # Number of original sigma levels
    n_interp = len(new_pres_levs) # Number of interpolant levels

    data_interp_shape = [n_interp, ] + list(orig_shape[1:])
    data_new = np.zeros(data_interp_shape)

    # Shape of array at any given level
    flat_shape = pres_levs.isel(lev=0).shape


    # Loop over the interpolant levels
    for ilev in range(n_interp):

        lev = new_pres_levs[ilev]

        P_abv = np.ones(flat_shape)
        # Array on sigma level above, below
        A_abv, A_bel = -1.*P_abv, -1.*P_abv
        # Pressure on sigma level above, below
        P_abv, P_bel = -1.*P_abv, -1.*P_abv

        # Mask area where pressure == levels
        P_eq = np.ma.masked_equal(P_abv, -1)

        # Loop from the second sigma level to the last one
        for i in range(1, n_sigma):
            a = np.ma.greater_equal(pres_levs.isel(lev=i), lev) # sigma-P > lev?
            b = np.ma.less_equal(pres_levs.isel(lev=i-1), lev) # sigma-P < lev?

            # Now, if the interpolant level is between the two
            # sigma levels, then we can use these two levels for the
            # interpolation.
            a = (a & b)

            # Pressure on sigma level above, below
            # P_abv[a], P_bel[a] = P[i].data[a], P[i-1].data[a]
            P_abv = np.where(a, pres_levs[i], P_abv)
            P_bel = np.where(a, pres_levs[i-1], P_bel)
            # Array on sigma level above, below
            # A_abv[a], A_bel[a] = A[i].data[a], A[i-1].data[a]
            A_abv = np.where(a, data[i], A_abv)
            A_bel = np.where(a, data[i-1], A_bel)

            # sel = P[i] == lev
            # P_eq[sel] = A[i].data[sel]
            P_eq = np.where(pres_levs[i] == lev, data[i], P_eq)

        # If no data below, set to missing value; if there is, set to
        # (interpolating) level
        P_val = np.ma.masked_where((P_bel == -1), np.ones_like(P_bel)*lev)

        # Calculate interpolation
        tl = np.log(P_val/P_bel)/np.log(P_abv/P_bel)*(A_abv - A_bel) + A_bel
        tl.fill_value = np.nan

        # Copy into result array, masking where values are missing
        # because of bad interpolation (out of bounds, etc.)
        tl[tl.mask] = np.nan
        data_new[ilev] = tl

    return data_new

mandatory_levs = 100.*np.array([250., 300., 500., 700., 850., 925., 1000.])
def interp_to_pres_levels(data, pres_levs, new_pres_levs=mandatory_levs,
                          method="scipy"):
    """ Interpolate the vertical coordinaet of a given dataset to the
    requested pressure levels.

    """

    data = data.squeeze()
    pres_levs = pres_levs.squeeze()

    if method == "scipy":
        data_new = _interp_scipy(data, pres_levs, new_pres_levs)
    elif method == "numpy":
        data_new = _interp_numpy(data, pres_levs, new_pres_levs)
    else:
        raise ValueError("Don't know method '%s'" % method)

    # Create new DataArray based on interpolated data
    new_coords = {}
    dims = []
    for c in data.dims:
        if c == 'lev':
            new_coords[c] = new_pres_levs
        else:
            new_coords[c] = data.coords[c]
        dims.append(c)

    return xray.DataArray(data_new, coords=new_coords, dims=dims, )

def calc_eke(ds):
    """ Compute transient eddy kinetic energy.

    Eddy Kinetic Energy (EKE) is given as combination of the time-varying
    component of both the meridional and zonal velocities. In the ocean,
    a typical formula uses the geostrophic velocities $U_g'$ and $V_g'$:

    $$\text{EKE} = \frac{1}{2}(U_g'^2 + V_g'^2)$$

    See [Dave Randall's lecture notes](http://kiwi.atmos.colostate.edu/group/dave/pdf/Eddy_Kinetic_Energy.frame.pdf)
    for a detailed derivation in the context of atmospheric dynamics.

    In the default CESM output, we can easily reconstruct EKE:

    $$\text{EKE} = \text{VV + UU - $\frac{1}{2}$(V$^2$ + U$^2$)}$$

    """

    fields = ['V', 'VV', 'U', 'UU']
    for field in fields:
        if not (field in ds):
            raise KeyError('Expected to find key "%s" in dataset'
                           % field)

    V, VV = ds['V'], ds['VV']
    U, UU = ds['U'], ds['UU']

    eke = VV + UU - (V**2 + U**2)/2.

    copy_attrs(V, eke)
    eke.name = 'EKE'
    eke.attrs['long_name'] = 'Transient eddy kinetic energy'
    eke.attrs['units'] = 'm^2/s^2'

    return eke

def calc_mpsi(ds, diag_pressure=True, ptop=500., pbot=100500.):
    """ Compute meridional streamfunction.

    Based on theory of the zonally-averaged general circulation. Several
    codes are available to do this calculation, but none seem to have been
    wrapped in Python, and this particular quantity ($\overline\chi$ in
    Holton; $\overline\Psi$ elsewhere) is not easy to to diagnose from the
    vertical vector streamfunction. A simple implementation is available
    via the [NCL function `zonal_mpsi`][zonal_mpsi_ncl] (FORTRAN77
    implementation [here][zon_mpsi.f], which references a formula from
    an NCAR technical manual:

    $$\Psi(z, \phi, \theta) = \frac{2\pi a \cos(\phi)}{g} \int\limits_{p(z, \phi, \theta)}^{P_s(z, \phi, \theta)} V(z, \phi, \theta)dp$$

    Source:

        Buja, L. E. (1994)
        CCM Processor User's Guide(Unicos Version).
        NCAR Technical Note NCAR/TN-384+IA,
        pages B-17 to B-18.

    This integrates from the *top* of the atmosphere to the bottom. Note
    that this formula is only valid in the zonal mean. A nice
    implementation of a code which computes this integral and then
    performs the zonal mean operation is given in
    [David Stepaniak's **zmmsf.ncl**][zmmsf.ncl] script, and I've
    re-implemented that here.

    For consistency with the analyses in the AMWG diagnostics and in papers
    such as Ming et el (2011), we should apply time averaging *before*
    calculating $ \Psi $.

    [zonal_mpsi_ncl]: https://www.ncl.ucar.edu/Document/Functions/Built-in/zonal_mpsi.shtml
    [zon_mpsi.f]: https://github.com/yyr/ncl/blob/34bafa4a78ba69ce8852212f59546bb433ce40c6/ni/src/lib/nfpfort/zon_mpsi.f
    [zmmsf.ncl]: http://www.ncl.ucar.edu/Applications/Scripts/zmmsf.ncl

    """

    #: Gravitational acceleration, m/s^2
    g = 9.80616

    #: Radius of earth, m
    a = 6.37122e6

    # Make sure the necessary fields are available in the dataset
    fields = ['V', 'PS']
    if diag_pressure: fields.extend(['hyam', 'hybm', 'P0'])
    for field in fields:
        if not (field in ds):
            raise KeyError('Expected to find key "%s" in dataset'
                           % field)

    # Transpose some dimensions (move to front) to facilitate
    # vertical integration in vector form
    v_wind = shuffle_dims(ds['V'], ['lev', 'lat'])
    ps = shuffle_dims(ds['PS'], ['lat', ]) # in Pa already

    levs = v_wind.lev.data
    lats = v_wind.lat.data
    nlev = len(levs)

    # 1) Construct the pressure thickness of each level, such that
    # pressure levels are mapped into the odd index and level interfaces
    # are mapped into the even indices
    if diag_pressure:
        pres = ( ds['hyam']*ds['P0'] + ds['hybm']*ds['PS'] )
        pres = shuffle_dims(pres, ['lev', 'lat'])
    else:
        pres = v_wind.lev*100. # hPa -> Pa
    # print(pres)

    # Mapping pressure levels and interfaces
    ptmp = np.zeros([2*nlev+1, ] + list(pres.shape[1:]))
    ptmp[0] = ptop # Pa
    ptmp[1::2] = pres.data
    ptmp[2:-1:2] = 0.5*(pres[1:].data + pres[:-1].data)
    ptmp[-1] = pbot # Pa

    # Compute thicknesses
    dp = np.zeros_like(ptmp)
    dp[1:-1:2] = (ptmp[2::2] - ptmp[0:-2:2]) # Pa

    # 2) Perform vertical integral of V as a function of level and
    # latitude, broadcasting over additional dimensions (time,
    # longitude, etc)
    psitmp = np.zeros([2*nlev+1, ] + list(v_wind.shape[1:]))
    vtmp = np.zeros([2*nlev+1, ] + list(v_wind.shape[1:]))

    psitmp[0] = 0.
    psitmp[1:2*nlev] = np.nan
    vtmp[0:-1:2] = 0.
    vtmp[1::2] = v_wind

    c = 2.*np.pi*a*np.cos(lats*np.pi/180.)/g # integral constant pre-factor

    # Integration loop, from top of atmosphere to bottom, using
    # a central difference method and 1/2-levels
    for klvl in range(1, 2*nlev, 2):

        integrand = c * np.transpose(vtmp[klvl]*dp[klvl])
        psitmp[klvl+1] = psitmp[klvl-1] - np.transpose(integrand)

    # Second loop to apply boundary condition to bottom of atmosphere, where
    # below-ground interfaces are changed to reflection points.
    bad_points = []
    for klvl in range(1, 2*nlev, 2):
        if np.any(ptmp[klvl] > ps):
            iinds, jinds = np.where(ptmp[klvl] > ps)
            # for k in ind_iter: print(k.shape)
            for (ilat, jlon) in zip(iinds, jinds):
                if (ilat, jlon) in bad_points: continue

                psitmp[klvl+1, ..., ilat, jlon] = \
                    -1.*psitmp[klvl-1, ..., ilat, jlon]
                psitmp[klvl+2:, ..., ilat, jlon] = np.nan
                bad_points.append((ilat, jlon))

    # psi on pressure levels = average of surrounding interfaces
    psitmp[1::2] = 0.5*(psitmp[2::2] + psitmp[0:-2:2])
    # Fix sign convention
    psitmp[1::2] *= -1.

    # Promote to a DataArray; attach metadata
    mpsi = DataArray(psitmp[1::2], coords=v_wind.coords, name='MPSI')
    mpsi.attrs['long_name'] = "Meridional Streamfunction"
    mpsi.attrs['units'] = 'kg/s'

    return mpsi

# def calc_streamfunction(dataset):
#     """ Access streamfunction calculation routine from windspharm
#
#     .. note::
#         This is just a reference implementation to illustrate
#         how to handle the interface between xray and windspharm.
#         Ideally this logic should be packaged separately and
#         the analysis routines accessed directly from
#         windspharm.
#
#     """
#     lats, lons = dataset.lat, dataset.lon
#
#     uwnd = dataset['U']
#     vwnd = dataset['V']
#
#     uu, ui = prep_data(uwnd.data, 'tyx')
#     vv, vi = prep_data(vwnd.data, 'tyx')
#
#     lats, uu, vv = order_latdim(lats, uu, vv)
#     w = VectorWind(uu, vv)
#     sf = recover_data(w.streamfunction(), ui)
#
#
#     new_ds = Dataset()
#     new_ds['SF'] = (uwnd.dims, sf)
#
#     for coord in uwnd.coords:
#         new_ds.coords[coord] = uwnd.coords[coord]
#
#     return new_ds

def scale_order(data, order=0, latex_units=False):
    """ Scale data by a power of 10, but preserve attributes.

    Manually invoke copy_attrs since we modify the units en route
    to it.

    """
    scaled_data = data*(10.**order)
    copy_attrs(data, scaled_data)

    old_units = data.attrs['units']
    if latex_units:
        new_units = "10$^{%d}$"
    else:
        new_units = "10^%d"
    scaled_data.attrs['units'] = (new_units % (-1*order, ) +
                                  " " + old_units)

    return scaled_data

@preserve_attrs
def seasonal_avg(data, season=None):
    """ Compute a mean for a given season, or for all seasons in a
    dataset. """

    seasonal_means = (data.groupby('time.season')
                          .mean('time', keep_attrs=True))

    if season is None:
        return seasonal_means
    else:
        if season not in ["DJF", "MAM", "JJA", "SON"]:
            raise ValueError("Didn't understand season '%s'" % season)
        return seasonal_means.sel(season=season)

@preserve_attrs
def global_avg(data, weights=None, dims=['lon', 'lat']):
    """ Compute (area-weighted) global average over a DataArray
    or Dataset. If `weights` are not passed, they will be computed
    by using the areas of each grid cell in the dataset.

    .. note::
        Handles missing values (nans and infs).

    """

    if weights is None: # Compute gaussian weights in latitude
        weights = area_grid(data.lon, data.lat)
        # Saving for later - compute latitudinal weighting
        # gw = weights.sum('lon')
        # weights = 2.*gw/gw.sum('lat')

    if isinstance(data, DataArray):

        # If there are NaNs, then use individual weights
        is_nan = data.notnull()
        if is_nan.any():
            total_weights = weights.where(is_nan).sum(dims)
        else:
            total_weights = weights.sum(dims)

        return (data*weights).sum(dims)/total_weights

    elif isinstance(data, Dataset):

        # Create a new temporary Dataset
        new_data = Dataset()

        # Iterate over the contents of the original Dataset,
        # which are all DataArrays, and compute the global avg
        # on those elements.
        for v in data.data_vars:
            coords = data[v].coords
            if not ('lon' in coords): 
                new_data[v] = data[v]
            else:
                new_data[v] = global_avg(data[v], weights)

        # Collapse remaining lat, lon dimensions if they're here
        leftover_dims = [d for d in dims if d in new_data.coords]
        if leftover_dims:
            new_data = new_data.sum(leftover_dims)
        return new_data

    # elif isinstance(data, iris.cube.Cube):
    #     raise NotImplementedError("`iris` deprecated with Python 3")
    #     if weights is None:
    #         weights = area_grid(ds.lon.values, ds.lat.values,
    #                             as_cube=True).data
    #     return data.collapsed(['latitude', 'longitude'],
    #                           iris.analysis.MEAN, weights=weights.data).data

@preserve_attrs
def pd_minus_pi(ds, pd='F2000', pi='F1850'):
    """ Compute difference between present day and pre-industrial,
    given a Dataset/DataArray with the emissions scenario as an
    embedded dimension.

    """
    return ds.sel(aer=pd) - ds.sel(aer=pi)

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
        if _MASKS is None: # still?!?!? Must be broken/unavailable
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
    if oceans is None: # still?!? Must be broken
        raise RuntimeError("Couldn't load default ocean shapefile")

    lons, lats = dataset[longitude], dataset[latitude]
    if isinstance(dataset, (xray.Dataset, xray.DataArray)):
        lons.values = shift_lons(lons.values)
    else:
        lons = shift_lons(lons)

    points = [asPoint(point) for point in np.column_stack([lons, lats])]
    if pt_return:
        return points

    in_ocean = [_is_in_ocean(p, oceans) for p in points]
    return in_ocean

################################################################
## IRIS CUBE FUNCTIONS

def min_max_cubes(*cubes):
    """ Compute nice global min/max for a set of cubes. """

    mini = np.min([np.min(cube.data) for cube in cubes])
    maxi = np.max([np.max(cube.data) for cube in cubes])

    return np.floor(mini), np.ceil(maxi)
