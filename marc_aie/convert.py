""" Datatype converters
"""

from copy import deepcopy
from itertools import product

from cartopy.util import add_cyclic_point
from numpy import empty, nditer
from xray import DataArray, Dataset
from xray.conventions import encode_cf_datetime

from . utilities import decompose_multikeys
from . var_data import Var

def cyclic_dataarray(da, coord='lon'):
    """ Add a cyclic coordinate point to a DataArray along a specified
    named coordinate dimension.

    >>> from xray import DataArray
    >>> from xray.conventions import encode_cf_datetime
    >>> data = DataArray([[1, 2, 3], [4, 5, 6]],
    ...                      coords={'x': [1, 2], 'y': range(3)},
    ...                      dims=['x', 'y'])
    >>> cd = cyclic_dataarray(data, 'y')
    >>> print cd.data
    array([[1, 2, 3, 1],
           [4, 5, 6, 4]])
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

def create_master(var, data_dict=None, new_fields=["PS", ]):
    """ Save a dictionary which holds variable data for all
    activation and aerosol case combinations to a dataset
    with those cases as auxiliary indices.

    Can be run with two major different configurations, and the 
    parameter `var` can either be a string or a :class:`Var`
    instance. In the first case, you must supply the `data_dict`
    which holds all of the data for all of the cases. For example,

    >>> from xray import Dataset as D
    >>> data_dict = {
    ...     ('a', 1): D({'data': [1, 2, 3]}),
    ...     ('b', 1): D({'data': [3, 3, 1]}),
    ...     ('a', 2): D({'data': [5, 2, 1]}),
    ...     ('b', 2): D({'data': [3, 4, 0]}),
    ... }
    >>> create_master_dataset("test_data", data_dict)
    <xray.Dataset>
    Dimensions:  (act: 2, aer: 2, data: 3)
    Coordinates:
      * act      (act) |S1 'a' 'b'
      * aer      (aer) int64 1 2
      * data     (data) int64 1 2 3
    Data variables:
        *empty*

    Parameters:
    -----------
    var : str or Var
        The name of the variable to extract from `data_dict`, or
        a Var object containing the data and cases to infer when
        creating the master dataset.
    data_dict : dict-like (optional)
        A multi-keyed dictionary containing the separate DataSets;
        note that the case information will be inferred from these
        multikeys, so they must cover the entire span of potential
        keys. 
    new_fields : list of strs (optional)
        A list of the keys in each DataSet to include in the 
        final multi-keyed master

    Returns:
    --------
    master : xray.DataSet 
        A DataSet combining all the sepearate cases into a single
        master, with the case information as auxiliary coordinates.

    """

    if isinstance(var, str):
        assert data_dict is not None
        acts, aers = decompose_multikeys(data_dict.keys())
        new_fields.append(var)
    elif isinstance(var, Var):
        data_dict = var.data
        acts = var.cases['act']
        aers = var.cases['aer']
        new_fields.append(var.varname)
        new_fields.extend(var.oldvar)
    else:
        raise ValueError("`var` must be a Var or a string.")

    for act, aer in product(acts, aers):
        assert (act, aer) in data_dict

    if isinstance(acts, str): acts = [acts, ]
    if isinstance(aers, str): aers = [aers, ]
    
    # Discover the type of the data passed into this method. If
    # it's an xray type, we'll preserve that. If it's an iris type,
    # then we need to crash for now.
    proto = data_dict[acts[0], aers[0]]
    if isinstance(proto, Dataset):
        return _master_dataset(data_dict, acts, aers, new_fields)
    elif isinstance(proto, DataArray):
        return _master_dataarray(data_dict, acts, aers)
    # elif isinstance(proto, Cube):
    #     raise NotImplementedError("Cube handling not yet implemented")
    else:
        raise ValueError("Data must be an xray or iris type")

def _master_dataarray(data_dict, acts, aers):

    proto = data_dict[acts[0], aers[0]]
    n_acts, n_aers = len(acts), len(aers)

    new_dims = ['act', 'aer', ] + [str(x) for x in proto.dims]
    new_values = empty([n_acts, n_aers] + list(proto.values.shape))
    
    it = nditer(empty((n_acts, n_aers)), flags=['multi_index', ])
    while not it.finished:
        i, j = it.multi_index
        act_i, aer_j = acts[i], aers[j]
        new_values[i, j, ...] = data_dict[act_i, aer_j].values
        it.iternext()

    # Copy and add the case coordinates
    new_coords = dict(proto.coords)
    new_coords['act'] = acts
    new_coords['aer'] = aers

    da_new = DataArray(new_values, dims=new_dims,
                         coords=new_coords)

    # Copy the attributes for act/aer coords, data itself
    for att, val in proto.attrs.items():
        da_new.attrs[att] = val
    da_new.coords['act'].attrs['long_name'] = "activation case"
    da_new.coords['aer'].attrs['long_name'] = "aerosol emission case"

    return da_new

def _master_dataset(data_dict, acts, aers, new_fields):

    proto = data_dict[acts[0], aers[0]]
    n_acts, n_aers = len(acts), len(aers)

    # Create the new Dataset to populate
    ds_new = Dataset()
    
    # Add act/aer case coord
    ds_new['act'] = acts
    ds_new['act'].attrs['long_name'] = "activation case"
    ds_new['aer'] = aers
    ds_new['aer'].attrs['long_name'] = "aerosol emission case"
    
    for f in proto.variables: 
        dsf = proto.variables[f]
       
        # Copy or update the coords/variable data
        if f in proto.coords:
            ds_new.coords[f] = (dsf.dims, dsf.values)
        else:
            if f in new_fields:
                
                new_dims = ['act', 'aer', ] + [str(x) for x in dsf.dims]
                new_values = empty([n_acts, n_aers] + list(dsf.values.shape))

                it = nditer(empty((n_acts, n_aers)), flags=['multi_index', ])
                while not it.finished:
                    i, j = it.multi_index
                    act_i, aer_j = acts[i], aers[j]
                    new_values[i, j, ...] = data_dict[act_i, aer_j].variables[f]
                    it.iternext()
                
                ds_new[f] = (new_dims, new_values)
            else:
                ds_new[f] = (dsf.dims, dsf.values)

        # Set attributes for the variable
        for att, val in dsf.attrs.items():
            ds_new[f].attrs[att] = val

    # Set global attributes
    for att, val in proto.attrs.items():
        ds_new.attrs[att] = val

    return ds_new

def dataset_to_cube(ds, field):
    """ Construct an iris Cube from a field of a given
    xray DataSet. """

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
    given selection from an xray DataSet. """

    dsf = ds[field]

    standard_name, long_name, var_name = None, None, field
    long_name = _get_dataset_attr(dsf, 'long_name')
    standard_name = _get_dataset_attr(dsf, 'standard_name')

    return standard_name, long_name, var_name