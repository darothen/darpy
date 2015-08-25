
import numpy as np
import xray

# from windspharm.standard import VectorWind
# from windspharm.tools import prep_data, recover_data, order_latdim

from . utilities import copy_attrs, preserve_attrs, area_grid, shuffle_dims

################################################################
## DATASET MANIPULATION FUNCTIONS

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
    mpsi = xray.DataArray(psitmp[1::2], coords=v_wind.coords, name='MPSI')
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
#     new_ds = xray.Dataset()
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
def global_avg(data, weights=None, dims=['lon', 'lat']):
    """ Compute (area-weighted) global average over a DataArray
    or Dataset. If `weights` are not passed, they will be computed
    by using the areas of each grid cell in the dataset.

    .. note::
        Handles missing values (nans and infs).

    """

    if weights is None:
        weights = area_grid(data.lon, data.lat)

    if isinstance(data, xray.DataArray):

        is_null = ~np.isfinite(data)
        if np.any(is_null):
            data = np.ma.masked_where(is_null, data)

        # return np.average(data, weights=weights)
        # return np.sum(data*weights)/np.sum(weights)
        return (data*weights/weights.sum(dims)).sum(dims)

    elif isinstance(data, xray.Dataset):

        # Create a new temporary Dataset
        new_data = xray.Dataset()

        # Iterate over the contents of the original Dataset,
        # which are all DataArrays, and compute the global avg
        # on those elements.
        for v in data.data_vars:
            new_data[v] = global_avg(data[v], weights)

        # return (data*weights/weights.sum(dims)).sum(dims)
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

################################################################
## IRIS CUBE FUNCTIONS

def min_max_cubes(*cubes):
    """ Compute nice global min/max for a set of cubes. """

    mini = np.min([np.min(cube.data) for cube in cubes])
    maxi = np.max([np.max(cube.data) for cube in cubes])

    return np.floor(mini), np.ceil(maxi)
