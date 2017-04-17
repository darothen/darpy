
import warnings

import numpy as np
from scipy import signal

import xarray
import xarray.ufuncs as xu
from xarray import DataArray, Dataset

# from windspharm.standard import VectorWind
# from windspharm.tools import prep_data, recover_data, order_latdim

from . utilities import (copy_attrs, preserve_attrs, area_grid,
                         shuffle_dims, shift_lons)

################################################################
# DATASET MANIPULATION FUNCTIONS


def extract_cloud_top(data, cloud_data, cloud_thresh,
                      vert_dim='lev', nlev=None,
                      method='numpy', return_indices=False):
    """ Extract at-cloud-top values for a given DataArray, according to a
    corresponding DataArray for approximating where cloud top is occurring.

    In MARC/CESM, the microphysics defines "cloud top" as the vertical grid level
    where the lquid stratus cloud fraction in the grid box > 0.01, and where the
    in-cloud water mixing ratio in stratus > 1e-7. For simplicity, we can take the
    first of these two criterion by analyzing the 3D "FREQL" field in the model
    output. As an example:

    >>> ds = xarray.open_dataset(<PATH_TO_OUTPUT_FILE>)
    >>> T_at_cloud_top = extract_cloud_top(ds['T'],
    ...                                    cloud_data=ds['FREQL'],
    ...                                    cloud_thresh=0.01)
    <xarray.DataArray 'T' (time: 8, lat: 96, lon: 144)> ...

    `data` and `cloud_data` should have *identical* dimensions

    Parameters
    ----------
    data, cloud_data : xarray.DataArray
        The DataArrays containing the data to be extracted at cloud top and for
        identifying cloud top, respectively.
    cloud_thresh : float
        A threshold for indicating what consitutes "cloud" levels in `cloud_data`
    vert_dim : str
        Name of vertical dimension; defaults to 'lev'
    nlev : int
        Maximum number of levels deep into the atmosphere to consider (defaults
        to inspecting entire column)
    method : str (default: 'numpy')
        Either 'numpy' to use fast-array logic or 'column' to iterate over each
        column
    return_indices : boolean (default: False)
        Return the vertical indices where cloud top is located instead of the
        actual values from `data`

    Returns
    -------
    xarray.DataArray similar to `data`, but with the vertical dimension reduced
    to just the data at cloud top.

    """
    orig_shape = data.shape
    data = shuffle_dims(data, vert_dim)
    cloud_data = shuffle_dims(cloud_data, vert_dim)
    new_shape = data.shape

    # Get length of vertical dimension if necessary
    if nlev is None:
        nlev = len(new_shape[0])

    if method == "numpy":
        # Find the vertical axis for searching
        axis = data.dims.index(vert_dim)
        nlevs = new_shape[axis]

        # Default new data to sfc (lowest model lev)
        data_new = np.ones(cloud_data.shape[1:])*np.nan

        # Shape of array at any given level
        flat_shape = cloud_data.isel(**{vert_dim: 0}).shape

        # Loop over the interpolant levels
        mask = np.zeros(flat_shape, dtype=bool)
        for ilev in range(nlevs):
            if ilev > nlev:
                break

            # print(ilev, end=" ")
            # At each level, we see if we've encountered cloud top (cloud_field > cloud_thresh)
            # in each column. If we have, then record the value there from the data field, and
            # add this column to the mask.
            cloud_at_lev = (cloud_data.isel(**{vert_dim: ilev}) > cloud_thresh).data

            if return_indices:
                data_new[cloud_at_lev & ~mask] = ilev
            else:
                data_level = data[ilev].data
                data_new[cloud_at_lev & ~mask] = data_level[cloud_at_lev & ~mask]

            # Update mask to indicate the newly processed columns
            mask = mask | cloud_at_lev
            # print(mask.sum())

    elif method == "columns":
        # Unravel all the columns
        ncols = int(np.product(new_shape[1:]))
        nlevs = np.product(new_shape)//ncols
        data_new = np.ones((ncols, ))*np.nan
        data_by_col = np.reshape(data.data, (nlevs, ncols))

        # Iterate over all the columns, applying the simple seeking
        # algorithm
        count = 0
        for col in range(ncols):
            for ilev in range(nlevs):
                if ilev > nlev:
                    break

                if data_by_col[ilev, col] > cloud_thresh:
                    if return_indices:
                        data_new[col] = ilev
                    else:
                        data_new[col] = data_by_col[ilev, col]
                    count += 1
                    break
            else:
                data_new[col] = 0.
        # print("Mapped {:d} columns".format(count))

        # Re-shape the data back to match the shuffled original data
        data_new = np.reshape(data_new, data.shape[1:])
    else:
        raise ValueError("Don't know how to analyze with method '{}'".format(method))

    # Create a new DataArray out of the analyzed NumPy array
    new_coords = {}
    new_dims = []
    for c in data.dims:
        if c == vert_dim:
            continue
        new_coords[c] = data.coords[c]
        new_dims.append(c)
    data_new = xarray.DataArray(data_new, coords=new_coords, dims=new_dims)

    # Re-order to match diemnsion of original dataset
    data_new = shuffle_dims(data_new, [d for d in data.dims if d != vert_dim])

    # Copy attrs
    data_new = copy_attrs(data, data_new)
    if hasattr(data, 'name'):
        data_new.name = data.name

    return data_new


def hybrid_to_pressure(ds, stride='m', P0=100000.):
    """ Convert hybrid vertical coordinates to pressure coordinates
    corresponding to model sigma levels.

    Parameters
    ----------
    data : xarray.Dataset
        The dataset to inspect for computing vertical levels
    stride : str, either 'm' or 'i'
        Indicate if the field is on the model level interfaces or
        middles for referencing the correct hybrid scale coefficients
    P0 : float, default = 1000000.
        Default reference pressure in Pa, used as a fallback.

    """

    # Grab necessary data fields
    a, b = ds['hya'+stride], ds['hyb'+stride]  # A, B coefficients
    try:
        P0_ref = ds['P0']  # Reference pressure
    except KeyError:
        P0_ref = P0
    P0 = P0_ref
    PS = ds['PS']  # Surface pressure field

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


def _interp_numpy(data, coord_vals, new_coord_vals,
                  reverse_coord=False, interpolation='lin'):
    """ Interpolate all columns simultaneously by iterating over
    vertical dimension of original dataset, following methodology
    used in UV-CDAT.

    Parameters
    ----------
    data : xarray.DataArray
        The data (array) of values to be interpolated
    coord_vals : xarray.DataArray
        An array containing a 3D field to be used as an alternative vertical coordinate
    new_coord_vals : iterable
        New coordinate values to inerpolate to
    reverse_coord : logical, default=False
        Indicates that the coord *increases* from index 0 to n; should be "True" when
        interpolating pressure fields in CESM
    interpolation : str
        "log" or "lin", indicating the interpolation method

    Returns
    -------
    list of xarray.DataArrays of length equivalent to that of new_coord_vals, with the
    field interpolated to each value in new_coord_vals

    """

    # Shuffle dims so that 'lev' is first for simplicity
    data = shuffle_dims(data)
    coord_vals = shuffle_dims(coord_vals)

    # Find the 'lev' axis for interpolating
    orig_shape = data.shape
    axis = data.dims.index('lev')
    n_lev = orig_shape[axis]

    n_interp = len(new_coord_vals)  # Number of interpolant levels

    data_interp_shape = [n_interp, ] + list(orig_shape[1:])
    data_new = np.zeros(data_interp_shape)

    # Shape of array at any given level
    flat_shape = coord_vals.isel(lev=0).shape

    # Loop over the interpolant levels
    for ilev in range(n_interp):

        lev = new_coord_vals[ilev]

        P_abv = np.ones(flat_shape)
        # Array on level above, below
        A_abv, A_bel = -1.*P_abv, -1.*P_abv
        # Coordinate on level above, below
        P_abv, P_bel = -1.*P_abv, -1.*P_abv

        # Mask area where coordinate == levels
        P_eq = np.ma.masked_equal(P_abv, -1)

        # Loop from the second sigma level to the last one
        for i in range(1, n_lev):
            # TODO: This could be combined into a single statement using a "sign" function
            #       to merely detect when the bracketing layers are both above and below.
            # e.g,
            # a = np.sign((coord_vals.isel(lev=i) - lev)*(coord_vals.isel(lev=i-1) - lev))
            if reverse_coord:
                a = np.ma.greater_equal(coord_vals.isel(lev=i), lev)
                b = np.ma.less_equal(coord_vals.isel(lev=i - 1), lev)
            else:
                a = np.ma.less_equal(coord_vals.isel(lev=i), lev)
                b = np.ma.greater_equal(coord_vals.isel(lev=i - 1), lev)

            # Now, if the interpolant level is between the two
            # coordinate levels, then we can use these two levels for the
            # interpolation.
            a = (a & b)

            # Coordinate on level above, below
            P_abv = np.where(a, coord_vals[i], P_abv)
            P_bel = np.where(a, coord_vals[i - 1], P_bel)
            # Array on level above, below
            A_abv = np.where(a, data[i], A_abv)
            A_bel = np.where(a, data[i-1], A_bel)

            P_eq = np.where(coord_vals[i] == lev, data[i], P_eq)

        # If no data below, set to missing value; if there is, set to
        # (interpolating) level
        P_val = np.ma.masked_where((P_bel == -1), np.ones_like(P_bel)*lev)

        # Calculate interpolation
        if interpolation == 'log':
            tl = np.log(P_val/P_bel)/np.log(P_abv/P_bel)*(A_abv - A_bel) + A_bel
        elif interpolation == 'lin':
            tl = A_bel + (P_val-P_bel)*(A_abv - A_bel)/(P_abv - P_bel)
        else:
            raise ValueError("Don't know how to interpolate '{}'".format(interpolation))
        tl.fill_value = np.nan

        # Copy into result array, masking where values are missing
        # because of bad interpolation (out of bounds, etc.)
        tl[tl.mask] = np.nan
        data_new[ilev] = tl

    return data_new


mandatory_levs = 100.*np.array([250., 300., 500., 700., 850., 925., 1000.])
def interp_to_pres_levels(data, pres_levs, new_pres_levs=mandatory_levs,
                          method="numpy"):
    """ Interpolate the vertical coordinate of a given DataArray to the
    requested pressure levels.

    """

    data = data.squeeze()
    pres_levs = pres_levs.squeeze()

    new_pres_levs = np.asarray(new_pres_levs)

    if method == "scipy":
        data_new = _interp_scipy(data, pres_levs, new_pres_levs)
    elif method == "numpy":
        data_new = _interp_numpy(data, pres_levs, new_pres_levs,
                                 reverse_coord=True, interpolation='log')
    else:
        raise ValueError("Don't know method '%s'" % method)

    # Create new DataArray based on interpolated data, noting that the interpolated
    # levels are by default going to be the first dimension. They're also off by a
    # factor of 100 from the interpolation
    new_coords = {'lev': new_pres_levs/100.}
    dims = ['lev', ]
    for c in data.dims:
        if c == 'lev':
            continue
        new_coords[c] = data.coords[c]
        dims.append(c)

    data_new = xarray.DataArray(data_new, coords=new_coords, dims=dims)

    # Re-order to match dimension shape of original dataset
    data_new = shuffle_dims(data_new, data.dims)

    return data_new


def interp_by_field(data, coord, new_coord_levs,
                    reverse_coord=False, interpolation="lin"):
    """ Interpolate a given data field based on an auxiliary coordinate field of
    equivalent shape

    """
    data = data.squeeze()
    coord = coord.squeeze()
    new_coord_levs = np.asarray(new_coord_levs)

    # Interpolate coordinate field
    data_new = _interp_numpy(data, coord, new_coord_levs, reverse_coord, interpolation)

    new_coords = {'lev': new_coord_levs}
    dims = ['lev', ]
    for c in data.dims:
        if c == 'lev':
            continue
        new_coords[c] = data.coords[c]
        dims.append(c)

    data_new = xarray.DataArray(data_new, coords=new_coords, dims=dims)

    # Re-order to match dimension shape of original dataset
    data_new = shuffle_dims(data_new, data.dims)

    return data_new

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
#         how to handle the interface between xarray and windspharm.
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

    if isinstance(data, DataArray):

        if weights is None:  # Compute gaussian weights in latitude
            weights = area_grid(data.lon, data.lat)
            # Saving for later - compute latitudinal weighting
            # gw = weights.sum('lon')
            # weights = 2.*gw/gw.sum('lat')

        weights = weights.where(xu.isfinite(data))
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

                          
def detrend(darr, dim='time'):
    """ Remove a trend from one dimension of a DataArray. 

    Uses scipy.signal.filter() to linearly detrend a given
    DataArray. The detrended data is then re-centered around the 
    mean of the original data.

    Parameters
    ----------
    darr : xarray.DataArray
        The data to be detrended
    dim : str
        Name of the dimension along which to detrend

    Returns
    -------
    xarray.DataArray with the linear trend removed along the
    indicated dimension
    """    
      
    dim_idx = darr.dims.index(dim)
    
    # Compute mean and expand to include squeezed dim
    darr_mean = darr.mean(dim).values
    darr_mean = np.expand_dims(darr_mean, dim_idx)
    
    # Detrend
    detrended = signal.detrend(darr.values, dim_idx)
    
    # Add back the mean to center
    detrended = detrended + darr_mean
    
    darr.values = detrended
    return darr
                          
################################################################
# IRIS CUBE FUNCTIONS

def min_max_cubes(*cubes):
    """ Compute nice global min/max for a set of cubes. """

    mini = np.min([np.min(cube.data) for cube in cubes])
    maxi = np.max([np.max(cube.data) for cube in cubes])

    return np.floor(mini), np.ceil(maxi)
