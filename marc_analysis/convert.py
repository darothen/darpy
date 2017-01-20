""" Datatype converters
"""

from itertools import product
import warnings
try:
    from cartopy.util import add_cyclic_point
except ImportError:
    warnings.warn("cartopy not found!")

from numpy import empty, nditer
from xarray import DataArray, Dataset

__all__ = [ 'cyclic_dataarray', 'create_master', 'dataset_to_cube', ]

#: Hack for Py2/3 basestring type compatibility
if 'basestring' not in globals():
    basestring = str


def cyclic_dataarray(da, coord='lon'):
    """ Add a cyclic coordinate point to a DataArray along a specified
    named coordinate dimension.

    >>> from xarray import DataArray
    >>> data = DataArray([[1, 2, 3], [4, 5, 6]],
    ...                      coords={'x': [1, 2], 'y': range(3)},
    ...                      dims=['x', 'y'])
    >>> cd = cyclic_dataarray(data, 'y')
    >>> print cd.data
    array([[1, 2, 3, 1],
           [4, 5, 6, 4]])

    Parameters
    ----------
    da : DataArray
        The data to wrap with a cyclic point
    coord : str
        Coordinate to wrap; defaults to 'lon'

    """
    assert isinstance(da, DataArray)

    lon_idx = da.dims.index(coord)
    cyclic_data, cyclic_coord = add_cyclic_point(da.values,
                                                 coord=da.coords[coord],
                                                 axis=lon_idx)

    # Copy and add the cyclic coordinate and data
    new_coords = dict(da.coords)
    new_coords[coord] = cyclic_coord
    new_values = cyclic_data

    new_da = DataArray(new_values, dims=da.dims, coords=new_coords)

    # Copy the attributes for the re-constructed data and coords
    for att, val in da.attrs.items():
        new_da.attrs[att] = val
    for c in da.coords:
        for att in da.coords[c].attrs:
            new_da.coords[c].attrs[att] = da.coords[c].attrs[att]

    return new_da

def dataset_to_cube(ds, field):
    """ Construct an iris Cube from a field of a given
    xarray Dataset. """

    raise NotImplementedError("`iris` deprecated for Python 3")

    dsf = ds[field]

    ## Attach coordinates to the cube, using full dataset for lookup
    # dim_coords_and_dims = []
    # for coord_key in dsf.coords:
    #     coord = ds[coord_key]
    #     coord_attrs = deepcopy(coord.attrs)
    #     idx_num = dsf.get_axis_num(coord_key)
    #
    #     # Variable names
    #     standard_name, long_name, var_name = _get_dataset_names(ds,
    #                                                             coord_key)
    #     # remove protected keys from attributes
    #     if long_name is not None:
    #         del coord_attrs['long_name']
    #     if standard_name is not None:
    #         del coord_attrs['standard_name']
    #
    #     coord_units = _get_dataset_attr(coord, 'units')
    #     if coord_units is not None:
    #         del coord_attrs['units']
    #     else:
    #         coord_units = '1'
    #
    #     coord_bounds = _get_dataset_attr(coord, 'bounds')
    #     if coord_bounds is not None:
    #         coord_bounds = ds[coord_attrs['bounds']]
    #         del coord_attrs['bounds']
    #
    #     # Check if we need to coerce timestamps back to numerical
    #     # values
    #     if (
    #         ( coord_key == 'time' ) and
    #         ( 'datetime' in coord.dtype.name )
    #     ):
    #         points, unit, calendar = encode_cf_datetime(coord)
    #         coord_units = unit
    #     else:
    #         points = coord
    #
    #     dc = DimCoord(points, standard_name=standard_name,
    #                   long_name=long_name, var_name=var_name,
    #                   units=coord_units, attributes=coord_attrs)
    #
    #     dim_coords_and_dims.append( ( dc, idx_num ) )
    #
    # # Attach any attribute information to the cube
    # attributes = deepcopy(dsf.attrs)
    #
    # standard_name, long_name, var_name = _get_dataset_names(ds, field)
    # if long_name is not None:
    #     del attributes['long_name']
    # if standard_name is not None:
    #     del attributes['standard_name']
    #
    # units = _get_dataset_attr(dsf, 'units')
    # if units is not None:
    #     del attributes['units']
    # else:
    #     units = '1'
    #
    # cell_methods = _get_dataset_attr(dsf, 'cell_methods')
    # if cell_methods is not None:
    #     del attributes['cell_methods']
    #     cell_methods_list = []
    #     cell_methods = map(lambda s: s.strip(),
    #                        cell_methods.split(":"))
    #     for coord, method in zip(cell_methods[::2], cell_methods[1::2]):
    #         cell_methods_list.append(
    #             CellMethod(method, coord)
    #         )
    #
    #     cell_methods = cell_methods_list
    #
    # # Build / return the cube
    # return Cube(dsf, standard_name=standard_name, long_name=long_name,
    #             var_name=var_name, units=units, attributes=attributes,
    #             cell_methods=cell_methods,
    #             dim_coords_and_dims=dim_coords_and_dims)

def _get_dataset_attr(ds, attr_key):
    if attr_key in ds.attrs:
        return ds.attrs[attr_key]
    else:
        return None

def _get_dataset_names(ds, field):
    """ If possible, return the standard, long, and var names for a
    given selection from an xarray DataSet. """

    dsf = ds[field]

    standard_name, long_name, var_name = None, None, field
    long_name = _get_dataset_attr(dsf, 'long_name')
    standard_name = _get_dataset_attr(dsf, 'standard_name')

    return standard_name, long_name, var_name
