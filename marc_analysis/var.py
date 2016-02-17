
import logging
logger = logging.getLogger()

import os
import json
import pkg_resources
import pickle
import warnings

from . io import load_variable
from . extract import extract_variable

"""
Utilities for storing and archiving information for
extracting and plotting variable data from the experiment
data archives.
"""

__all__ = ['get_cesm_vars', 'VarList', 'Var',
           'CDOVar', 'CESMVar', 'MultiVar', 'common_vars', ]

_TAB = "    "
_CESM_VARS = None

#######################################################################
## Utility functions

def get_cesm_vars(reextract=True):
    """ Load in the saved dictionary of CESM vars. """

    global _CESM_VARS

    if reextract or (_CESM_VARS is None):
        try:
            CESM_defaults_fn = pkg_resources.resource_filename(
                "marc_analysis", "data/CESM_default_vars.p"
            )
            with open(CESM_defaults_fn, 'rb') as f:
                CESM_vars_dict = pickle.load(f)

            _CESM_VARS = CESM_vars_dict

        except FileNotFoundError:
            warnings.warn("Couldn't find CESM_default_vars archive")
            _CESM_vars_dict = None
    else:
        CESM_vars_dict = _CESM_VARS

    return CESM_vars_dict

def _extract_cesm_vars(nc_filename, out="CESM_default_vars.p"):
    """ Extract all the CESM variables and their metadata from
    a given dataset. """
    import netCDF4 as nc

    d = nc.Dataset(nc_filename, 'r')
    all_vars = { key: { att: data.__dict__[att] for att in data.ncattrs() } \
                 for key, data in d.variables.items() }
    logger.debug("... found %d variables" % (len(all_vars), ))

    with open(out, 'wb') as f:
        pickle.dump(all_vars, f, protocol=2)
    logger.debug("done. Saved to %s" % out)

def print_cesm_vars():
    """ Print a list of all the available CESM vars. """
    if _CESM_VARS is None:
        _ = get_cesm_vars()

    for k, v in sorted(_CESM_VARS.items()):

        logger.info(k)
        if ('long_name' in v) and ('units' in v) :
            logger.info("{} ({})".format(
                v['long_name'], v['units']
            ))
        else:
            logger.info("")

#######################################################################

class VarList(list):
    """ Special type of :class:`list` with some
    better handling for `Var` objects. Based on iris CubeLists
    as a learning example. """

    def __new__(cls, var_list=None):
        """ Create a VarList from a :class:`list` of Vars. """
        vl = list.__new__(cls, var_list)
        if not all( [isinstance(v, Var) for v in vl] ):
            raise ValueError("Some items were not Vars!")
        return vl

    def __str__(self):
        result = [ "%s: %s" % (i, v) for i, v in enumerate(self) ]

        if result:
            result = "\n".join(result)
        else:
            result = "No Vars found."

        return result

class Var(object):
    """ A container object for finding, extracting, modifying, and
    analyzing output from a CESM multi-run experiment.

    A Var allows you to quickly load a minimal set of variables
    necessary for an analysis into memory, and pipeline the
    extraction, analysis, and saving operations.

    Types extending Var add additional features, such as
    applying CDO operators or loading default variables from the
    CESM output which require no pre-processing.

    """
    # TODO: Extend `Mapping` interface so that one doesn't need to use the `data` property to access data
    # TODO: Implement `__enter__` and `__exit__` such that data is loaded and deleted from the `Var` instance
    # TODO: Logging of actions on `Var` instance for writing to new file history after analysis.
    # TODO:Change `oldvar` to automatically populate a 1-element list if not other values passed

    def __init__(self, varname,  oldvar="", long_name="",
                 units="", scale_factor=1., ncap_str="", **kwargs):

        """ Create a container in preparation for an analysis
        pipeline.

        Parameters
        ----------
        varname : str
            A string to use as a convenient, short alias for the
            variable name; this will be the name in the attached
            datasets for any new variable created during the
            extraction process.
        oldvar : str or list of strs, optional
            Either a str or a list of strs of the names of the
            variables which will be used to compose or create
            this new variable.
        long_name : str, optional
            A descriptive, long-form name for this variable.
        units : str, optional
            A udunits-compliant str describing the dimensional
            units of this variable.
        scale_factor : float, optional
            A value to use for re-scaling the output. If no value
            is passed, will default to `1.0`. This is particularly
            useful because it allows lazy conversion to new units.
        ncap_str : str, optional
            A string to be passed to the command line NCO `ncap2`
            for pre-processing during the extraction of the variable
            data
        """

        self.varname = varname
        if not oldvar:
            self.oldvar = varname
        else:
            self.oldvar = oldvar

        self.ncap_str = ncap_str

        # Set any additional arguments as attributes
        self.attributes = kwargs

        # Overwrite any attributes
        self.long_name = long_name
        if long_name:
            self.attributes['long_name'] = long_name

        self.units = units
        if units:
            self.attributes['units'] = units
        else:
            self.attributes['units'] = "1"

        self.scale_factor = scale_factor
        if scale_factor != 1.:
            self.attributes['scale_factor'] = scale_factor

        for attr, val in self.attributes.items():
            self.__dict__[attr] = val

        # Some useful properties to set
        self.name_change = self.varname == self.oldvar

        # Encapsulation of data set when variable is loaded
        self._data = None
        self._cases = None
        self._loaded = False

    def apply(self, func, *args, **kwargs):
        """ Apply a given function to every loaded cube/dataset
        attached to this Var instance. The given function should
        return a new Cube/DataSet instance. """

        if not self._loaded:
            raise Exception("Data is not loaded")
        for key, data in self.data.items():
            self.data[key] = func(data, *args, **kwargs)

    def extract(self, exp, years_omit=5, years_offset=0,
                re_extract=False):
        """ Extract the dataset for a given var. """

        extract_variable(exp, self, years_omit=years_omit,
                         years_offset=years_offset,
                         re_extract=re_extract)

    ## Deprecated on transition to `marc_analysis`
    #
    # def load_cubes(self, src_dir=WORK_DIR,
    #                act_cases=CASES_ACT, aer_cases=CASES_AER,
    #                fix_times=False, **kwargs):
    #     """ Load the data for this variable into iris cubes and attach
    #     them to the current instance.
    #
    #     Parameters
    #     ----------
    #     act_cases, aer_cases : strs or list of strs
    #         The names of the cases to load.
    #     src_dir : str
    #         The path to look for the extracted data in; will
    #         default to the **WORK_DIR** set in `cases_setup`.
    #     fix_times : bool
    #         Attempt to decode and replace timestamps in dataset
    #         with better values given the bounds attached to them.
    #     **kwargs : dict
    #         Additional keyword arguments to pass to the loader.
    #
    #     """
    #     return self._load('iris', src_dir, act_cases, aer_cases,
    #                       fix_times, **kwargs)

    def load_data(self, exp, method='xarray', fix_times=False, **kwargs):
        """ Load the data for this variable into xarray DataSets and
        attach them to the current instance.

        Parameters
        ----------
        exp : experiment.Experiment
            The container object detailing the experiment that
            produced this variable
        method : str
            String indicated to load `xarray` or `iris` data structures
        fix_times : bool
            Attempt to decode and replace timestamps in dataset
            with better values given the bounds attached to them.
        **kwargs : dict

            Additional keyword arguments to pass to the loader.

        """
        self._load(exp, method, fix_times, **kwargs)

    def _load(self, exp, method, fix_times, **kwargs):
        """ Loading workhorse method. """

        if self._loaded:
            raise Exception("Data is already loaded")

        # Save the cases
        self._cases = exp.cases

        # Get the location of the extracted variable data based
        # on the Experiment
        save_dir = exp.work_dir

        # Load the data
        self._data = dict()
        for case_bits in exp.all_cases():

            case_fn_comb = "_".join(case_bits)
            var_fn = "%s_%s.nc" % (case_fn_comb, self.varname)
            path_to_file = os.path.join(save_dir, var_fn)

            self._data[case_bits] = \
                load_variable(self.varname, path_to_file,
                              fix_times=fix_times, extr_kwargs=kwargs)
        self._loaded = True

    def to_dataarrays(self):
        """ Convert the data loaded using `self.load_datasets()`
        into DataArrays containing only the variable described by
        this. """

        self.apply(lambda ds: ds[self.varname])

    @property
    def cases(self, *keys):
        if not self._loaded:
            raise Exception("Data has not yet been loaded into memory")
        return self._cases
    @cases.deleter
    def cases(self):
        if not self._loaded:
            raise Exception("Data has not yet been loaded into memory")
        self._cases = None
        if self._data is not None:
            self._data = None
        self._loaded = False

    @property
    def data(self):
        if not self._loaded:
            raise Exception("Data has not yet been loaded into memory")
        return self._data
    @data.deleter
    def data(self):
        if not self._loaded:
            raise Exception("Data has not yet been loaded into memory")
        self._data = None
        if self._cases is not None:
            self._cases = None
        self._loaded = False

    @classmethod
    def from_json(cls, json_str):
        jd = json.loads(json_str)
        return Var(**jd)

    def to_json(self):
        """ Return JSON representation of variable info
        as a string. """
        return json.dumps(self.__dict__)

    def __str__(self):

        out = self.varname
        if hasattr(self, "long_name"):
            out += " (%s)" % self.long_name
        if hasattr(self, "units"):
            out += " [%s]" % self.units
        if self._loaded:
            data_type= type(next(iter(self.data.values()))).__name__
            out += "\n[ loaded -> %s(%r)]" % (data_type, self._cases)

        if not (self.oldvar == self.varname):
            olv_str = self.oldvar if isinstance(self.oldvar, str) \
                                  else ",".join(self.oldvar)
            out += "\n" + _TAB + "from fields " + olv_str
        if self.ncap_str:
            out += "\n" + _TAB + "NCAP func: " + self.ncap_str

        return out

    def _get_atts(self):
        """ Return list of uniquely-identifying attributes. """
        atts = [self.varname, self.units, self.ncap_str]
        if hasattr(self, 'lev_bnds'):
            atts += self.lev_bnds
        if hasattr(self, 'cdo_method'):
            atts += self.cdo_method
        return tuple(atts)

    def __eq__(self, other):
        self_atts, other_atts = self._get_atts(), other._get_atts()
        return self_atts == other_atts

    def __neq__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash( self._get_atts() )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

class CDOVar(Var):

    def __init__(self, varname,
                 lev_bnds=None, cdo_method=[], **kwargs):
        super(self.__class__, self).__init__(varname, **kwargs)

        if lev_bnds is not None:
            self.lev_bnds = lev_bnds
        if cdo_method:
            self.cdo_method = cdo_method

    def __str__(self):
        out = super(self.__class__, self).__str__()
        out += "\n" + _TAB + "CDO methods"

        if hasattr(self, 'lev_bnds'):
            out += " (levs %r): " % self.lev_bnds
        else:
            out += ": "

        if hasattr(self, 'cdo_method'):
            out += ",".join(self.cdo_method)

        return out

    @classmethod
    def from_var(cls, other, varname, lev_bnds=None, cdo_method=[]):
        """ Create a generic CDOVar by adding information on how to extract
        data. """

        assert isinstance(other, (Var, CESMVar))

        other_dict = dict(oldvar=other.oldvar,
                          long_name=other.long_name,
                          units=other.units,
                          scale_factor=other.scale_factor,
                          ncap_str=other.ncap_str)

        return cls(varname, lev_bnds, cdo_method, **other_dict)

    @classmethod
    def low_var(cls, other, lev_bnds=(23, 27), cdo_method=["vertmean", ]):
        """ Create a variable averaged over levels 23-27 """
        return cls.from_var(other, other.varname+"_low", lev_bnds, cdo_method)

class MultiVar(Var):
    """ Container for a collection of raw variables from the CESM output
    such that they can all be extracted together.
    """

    def __init__(self, collection_name="", list_of_vars=None):
        """ Default constructor from list of vars

        Parameters:
        -----------
        collection_name : str
            A name to add to the file. This will correspond to
            a scalar value of "0." in the file, but will be used
            in the Var machinery to suffix the output file for
            reference.
        list_of_vars : iterable of strings
            The list of varnames to include in the collection. Each
            item must correspond to an entry in the record of CESM
            variables available!
        """

        self.collection_name = collection_name
        self.list_of_vars = list_of_vars

        if _CESM_VARS is None:
            get_cesm_vars()
        for v in list_of_vars:
            assert v in _CESM_VARS

        varname = "temp" if not collection_name else collection_name
        super(self.__class__, self).__init__(varname, list_of_vars,
                                             ncap_str="%s=0." % varname)

        def __str__(self):

            out = "MultiVar containing:"
            for v in self.list_of_vars:
                out += "\n" + _TAB + \
                       "{v:s} ({long:s}) [{units:s}]".format(
                           v, _CESM_VARS[v]['long_name'],
                           _CESM_VARS[v]['units']
                       )

class CESMVar(Var):
    """ Raw variables from CESM which require no pre-processing,
    save for renaming operations.

    Use oldvar->varname renaming if it's passed as an argument to the
    constructor.

    """

    def __init__(self, varname, oldvar="", long_name="", units="", **kwargs):

        if _CESM_VARS is None:
            get_cesm_vars()

        ## Be sure there is a variable to extract!
        if oldvar:
            assert oldvar in _CESM_VARS
            var_attrs = _CESM_VARS[oldvar]
        else:
            assert varname in _CESM_VARS
            var_attrs = _CESM_VARS[varname]

        ## Replace name, units if applicable
        if (not long_name) and ('long_name' in var_attrs):
            long_name = var_attrs['long_name']
        if (not units) and ('units' in var_attrs):
            units = var_attrs['units']

        super(self.__class__, self).__init__(varname, oldvar,
                                             long_name=long_name,
                                             units=units, **kwargs)

## Deprecated on transition to `marc_analysis` package
#
# class VarArchive(object):
#     """ Handler for accessing archive of known variables
#     using the `shelve` database built-in.
#
#     A VarArchive should nested dictionary:
#     `VAR_ARCHIVE`
#         -> Var (as json string)
#             -> (activation case, aerosol case)
#     At the final level should be a single string which holds
#     the filename where this particular variable's analyzed
#     data is stored.
#
#     """
#
#     def __init__(self, archive_name, mode='w',
#                  open_db=False):
#         self.archive_name = archive_name
#         self.mode = mode
#
#         if open_db:
#             self._open_db()
#
#     def _open_db(self):
#         """ Open the databse from shelve """
#
#         try:
#             self.db = shelve.open(self.archive_name, self.mode,
#                                   writeback=True)
#             if self.mode == 'c': self.db['all_vars'] = []
#
#         except Exception as e: # since dbm uses a weird special one
#             if "open new db" in str(e):
#                 print("Creating new variable" \
#                       " archive -> %s" % self.archive_name)
#                 self.db = shelve.open(self.archive_name, "c",
#                                       writeback=True)
#             else:
#                 print("Could not open archive (%r)" % e)
#
#     def _create_empty_var(self, key):
#         empty_dict = { actaer: "" for actaer
#                                     in product(CASES_ACT,
#                                                CASES_AER) }
#         self.db[key] = empty_dict
#
#     def get(self, var, act=None, aer=None):
#         var_data = self.db[var.to_json()]
#
#         if (act is None) and (aer is None):
#             return var_data[(act, aer)]
#         elif (act is None):
#             return [ var_data[act,   a] for a in CASES_AER ]
#         elif (act is None):
#             return [ var_data[a  , aer] for a in CASES_ACT ]
#         else:
#             return var_data
#
#     def set(self, var, act, aer, item):
#         var_json = var.to_json()
#
#         if var_json not in self.db:
#             self._create_empty_var(var_json)
#
#         var_data = self.db[var_json]
#         var_data[(act, aer)] = item
#
#         if var not in self.db['all_vars']:
#             self.db['all_vars'].append(var)
#
#     def __enter__(self):
#
#         self._open_db()
#
#         return self
#
#     def __exit__(self, exc_type, exc_value, traceback):
#         self.db.close()
#
#         if isinstance(exc_value, TypeError):
#             print(traceback())
#             return False
#         else:
#             return True
#
#
#     def find_var(self, short_name=None, long_name=None):
#         """ Given the short or long name of a variable, search
#         for it in the saved archive. Return a `VarList` of any
#         partial matches. """
#
#         if (short_name is None) and (long_name is None):
#             raise ValueError("Expected only one of short or long name.")
#
#         all_vars = self.db['all_vars']
#         all_vars = [var for var in all_vars if var != 'all_vars']
#
#         if short_name is not None:
#             in_var = lambda var: short_name in var.varname
#         else: # short circuit to either/orr
#             in_var = lambda var: long_name in var.long_name
#
#         result = VarList(var for var in all_vars if in_var(var))
#
#         return result
#
#     @classmethod
#     def default(cls):
#         return VarArchive(_VAR_ARCHIVE)

class common_vars:

    CDNC_col = CESMVar("CDNC_col", "CDNUMC",
                       long_name="Total Column CDNC",
                       units='10^6 cm^-2', scale_factor=1e-4*1e-6)
    CF = CESMVar("CLDTOT", long_name="Total Cloud Fraction")
    # COT = CESMVar("COT", "TOT_ICLD_VISTAU")
    # COT = CDOVar("COT", lev_bnds=[23,30], cdo_method=['vertsum', ],
    #              long_name="In-cloud low-cloud optical thickness")
    LWP = CESMVar("TGCLDLWP")
    PRECL = CESMVar("PRECL",
                    long_name="Total large-scale precipitation",
                    scale_factor=1e3*3600.*24., units="mm/day")
    PRECIP = Var("PRECIP", ("PRECC", "PRECL"),
                  long_name="Total precipitation",
                  scale_factor=1e3*3600.*24., units="mm/day")
    SWCF = CESMVar("SWCF")
    TS = CESMVar("TS", long_name="Near-surface Temperature")





# BASE_VARS = [
#     CESMVar("AREL", "Average droplet effective radius", "micron"),
#     CESMVar("CCN3", 'CDNC at 0.1% Supersaturation', 'cm-3'),
#     Var("CDNC", ("AWNC", "FREQL"),
#         long_name="Cloud Droplet Number Concentration", units="cm-3",
#         scale_factor=1e-6,
#         ncap_str="CDNC=0.0*AWNC; where(FREQL>0.0) CDNC=AWNC/FREQL" ),
#     Var("CDNC_col", "CDNUMC", long_name="Column-integrated CDNC",
#         units="cm-3", scale_factor=1e-4),
#     CESMVar("CLOUD", "Cloud Fraction"),
#     CESMVar("T", "Temperature", "K"),
#     Var("VQ", 'Meridional Water Transport', 'kg/kg m/s'),
#     Var("VT", 'Meridional Heat Transport', 'K m/s'),
#     Var("VU", 'Meridional Zonal Momentum Flux', 'm2/s2'),
#     Var("CF", "CLDTOT", long_name="Total cloud fraction", units="1"),
#     CDOVar("COT", oldvar=("TOT_ICLD_VISTAU","FREQL"),
#            long_name="In-cloud low-cloud optical thickness", units="1",
#            lev_bnds=(23, 27), cdo_method=["vertsum", ]),

#     Var("LWP", "TGCLDLWP",
#         {"long_name": "Vertically-integrated cloud liquid water path",
#          "units": "kg/m2"},
#         None, ["vertsum"], None),
# ]
# BASE_VARS_TABLE = { v.newvar: v for v in BASE_VARS }

# # LOW_VARS_TABLE = { v.newvar: v for
# #     CDOVar.low_var(v for v )
# # }
# # ]

