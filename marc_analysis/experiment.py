"""
Container object specifying the design of a particulary experiment.

In general, it's convenient to use one of two different naming
conventions for sets of repeated CESM/MARC experiments:

1) Provide a unique identifier or name to all your experiments, and
store each in a separate folder, such as

    data/
        exp_a/
        exp_b/
        exp_c/
        exp_d/

alternatively,

2) Use a common name for fundamentally similar experiments but store
them in hierarchical folders which describe a particular parameter
which has been changed,

    data/
        factor_1-a/
            factor_2-a/
            factor_2-b/
        factor_1-b/
            factor_2-a/
            factor_2-b/

It's especially convenient to name each experimental run after the
leaf factor.

"""
from __future__ import print_function

import os
import warnings
from collections import OrderedDict, namedtuple
from itertools import product

from . extract import extract_variable
from . io import load_variable
from . convert import create_master

__all__ = ['Case', 'Experiment', 'SingleCaseExperiment', ]

Case = namedtuple('case', ['shortname', 'longname', 'vals'])

#: Hack for Py2/3 basestring type compatibility
if 'basestring' not in globals():
    basestring = str


class Experiment(object):
    """ CESM/MARC Experiment details.

    Experiment encapsulates information about a particular CESM/MARC
    experiment so that data can quickly and easily be accessed. It
    records the layout of the experiment (how many different cases),
    where the data directory resides, and some high-level details
    about how to process the data.

    Attributes
    ----------
    name : str
        The name of the experiment.
    cases : iterable of Case namedtuples
        The levels of the experimental cases being considered
    data_dir : str
        Path to directory containing the unanalyzed data for this
        experiment
    """

    def __init__(self, name, cases,
                 timeseries=False,
                 data_dir='./',
                 full_path=False,
                 output_prefix="",
                 output_suffix=".nc",
                 validate_data=True):

        """
        Parameters
        ----------
        name : str
            The name of the experiment.
        cases : iterable of Case namedtuples
            The levels of the experimental cases being considered
        timeseries : logical
            If "True", then the data is in "timeseries" form instead of
            "timeslice" form; that is, in the leaf folders of the archive
            hierarchy, the files are split by variable rather than snapshots
            of all fields at a given time.
        cases : str or list
        data_dir : str
            Path to directory containing the unanalyzed data for this experiment
        full_path : bool
            Indicates whether the data directory structure leads immediately
            to a folder containing the output data (if `False`) or to the
            hierarchical structure output by CESM/MARC by default
        output_prefix : str
            Global prefix for all output files as a string, which can optionally
            include named format directives indicated which case bit to supply
        output_suffix : str
            Suffix ending all output files. Defaults to ".nc"
        validate_data : bool, optional (default True)
            Validate that the specified case structure is reflected in the
            directory structure passed via `data_dir`
        """

        self.name = name
        self.full_path = full_path

        # Process the case data, which is an Iterable of Cases
        self._case_data = OrderedDict()
        try:
            for case in cases:
                assert isinstance(case, Case)
                self._case_data[case.shortname] = case
        except AttributeError:
            raise ValueError("Couldn't process `cases`")

        # Mapping to private information on case data
        self._cases = list(self._case_data.keys())
        self._case_vals = {case: self._case_data[case].vals for case in self._cases}
        self._casenames = {case: self._case_data[case].longname for case in self._cases}
        # Add cases to this instance for "Experiment.[case]" access
        for case, vals in self._case_vals.items():
            setattr(self.__class__, case, vals)

        self.timeseries = timeseries
        self.output_prefix = output_prefix
        self.output_suffix = output_suffix

        # Walk tree of directory containing existing data to ensure
        # that all the cases are represented
        self.data_dir = data_dir
        if validate_data:
            # Location of existing data
            assert os.path.exists(data_dir)
            self._validate_data()

    # Validation methods
    def _validate_data(self):
        """ Validate that the specified data directory contains
        a hierarchy of directories which match the specified
        case layout.

        """

        root = self.data_dir

        path_bits = self.all_cases()
        for bits in path_bits:
            full_path = os.path.join(root, *bits)
            try:
                assert os.path.exists(full_path)
            except AssertionError:
                raise AssertionError("Couldn't find data on path {}".format(full_path))

    # Properties and accessors
    @property
    def cases(self):
        """ Property wrapper for list of cases. Superfluous, but
        it's really important that it doesn't get changed.
        """
        return self._cases

    def itercases(self):
        """ Generator for iterating over the encapsulated case
        information for this experiment

        >>> for case_info in marc_analysis.itercases():
        ...     print(case_info)
        ('aer', 'aerosol emissions', ['F2000', 'F1850'])
        ('act', 'activation scheme', ['arg_comp', 'arg_min_smax'])

        """
        for case in self._cases:
            yield case, self._casenames[case], self._case_vals[case]

    def all_cases(self):
        """ Return an iterable of all the ordered combinations of the
        cases comprising this experiment.

        >>> for case in marc_analysis.all_cases():
        ...     print(case)
        ('F2000', 'arg_comp')
        ('F1850', 'arg_comp')
        ('F2000', 'arg_min_smax')
        ('F1850', 'arg_min_smax')

        """
        return product(*self.all_case_vals())

    def all_case_vals(self):
        """ Return a list of lists which contain all the values for
        each case.

        >>> for case_vals in marc_analysis.all_case_vals():
        ...     print(case_vals)
        ['F2000', 'F1850']
        ['arg_comp', 'arg_min_smax']
        """
        return [self._case_vals[case] for case in self._cases]

    def get_case_vals(self, case):
        """ Return a list of strings with the values associated
        with a particular case.

        Parameters
        ----------
        case : str
            The name of the case to fetch values for.

        """
        return self._case_vals[case]

    def get_case_bits(self, **case_kws):
        """ Return the given case keywords in the order they're defined in
        for this experiment. """
        return [case_kws[case] for case in self.cases]

    def get_case_kws(self, *case_bits):
        """ Return the given case bits as a dictionary. """
        return {name: val for name, val in zip(self.cases, case_bits)}

    def case_path(self, *case_bits, **case_kws):
        """ Return the path to a particular set of case's output from this
        experiment.

        """
        if case_kws:
            # Re-assemble into ordered bits
            case_bits = self.get_case_bits(**case_kws)
            return os.path.join(self.data_dir, *case_bits)
        elif case_bits:
            return os.path.join(self.data_dir, *case_bits)
        else:
            raise ValueError("Expected either a list or dict of case values")

    def case_prefix(self, **case_bits):
        """ Return the output prefix for a given case. """
        return self.output_prefix.format(**case_bits)

    # Loading methods
    def load(self, var, fix_times=False, master=False, **case_kws):
        """ Load a given variable from this experiment's output archive.

        Parameters
        ----------
        var : str or Var
            Either the name of a variable to load, or a Var instanced
            defining a specific output variable
        fix_times : logical
            Fix times if they fall outside an acceptable calendar
        master : logical
            Return a master dataset, with each case defined as a unique
            identifying dimension
        case_kws : dict (optional)
            Additional keywords, which will be interpreted as a specific
            case to load from the experiment.

        """
        if self.timeseries:
            return self._load_timeseries(var, fix_times, master, **case_kws)
        else:
            return self._load_timeslice(var, fix_times, master, **case_kws)

    def _load_timeslice(self, var, fix_times=False, master=False, **case_kws):
        raise NotImplementedError

    def _load_timeseries(self, var, fix_times=False, master=False, **case_kws):
        """ Load a timeseries dataset directly from the experiment output
        archive.

        See Also
        --------
        Experiment.load : sentinel for loading data
        """

        is_var = not isinstance(var, basestring)
        if is_var:
            field = var.varname
            is_var = True
        else:
            field = var

        if case_kws:
            # Load/return a single case
            prefix = self.case_prefix(**case_kws)

            path_to_file = os.path.join(
                self.case_path(**case_kws),
                self.case_prefix(**case_kws) + field + self.output_suffix,
            )
            ds = load_variable(field, path_to_file, fix_times=fix_times)

            return ds
        else:

            data = dict()

            # Load/return all cases
            for case_bits in self.all_cases():
                case_kws = self.get_case_kws(*case_bits)

                path_to_file = os.path.join(
                    self.case_path(*case_bits),
                    self.case_prefix(**case_kws) + field + self.output_suffix,
                )
                data[case_bits] = \
                    load_variable(field, path_to_file, fix_times=fix_times)
            if is_var:
                var._data = data
                var._loaded = True

            if master:
                ds_master = create_master(self, field, data)

                if is_var:
                    var.master = ds_master

                return ds_master

            return data

    # Extraction methods - deprecated!
    def extract(self, var, **kwargs):
        """ Extract a given variable.
        """
        raise DeprecationWarning

        extract_variable(self, var, **kwargs)

    def load_extracted(self, var, fix_times=False,
                       master=False, master_kwargs={}, **kwargs):
        """ Load the data previously extracted for this experiment and
        for a single variable into xarray Datasets and attach that data to
        the current Experiment.

        """
        raise DeprecationWarning

        if var._loaded:
            raise Exception("Data is already loaded")

        # Save the cases
        var._cases = self.cases

        # Load the data
        var._data = dict()
        for case_bits in self.all_cases():

            case_fn_comb = "_".join(case_bits)
            var_fn = "%s_%s.nc" % (case_fn_comb, var.varname)
            path_to_file = os.path.join(save_dir, var_fn)

            var._data[case_bits] = \
                load_variable(var.varname, path_to_file,
                              fix_times=fix_times, extr_kwargs=kwargs)
        var._loaded = True

        if master:
            var.master = create_master(self, var, **master_kwargs)

        # Attach to current Experiment
        self.__dict__[var.varname] = var

    def __repr__(self):
        base_str = "{} -".format(self.name)
        for case in self._cases:
            base_str += "\n    {}: ".format(self._casenames[case])
            base_str += " [" + \
                        ", ".join(val for val in self._case_vals[case]) + \
                        "]"
        return base_str


class SingleCaseExperiment(Experiment):
    """ Special case of Experiment where only a single model run
    is to be analyzed.

    """

    def __init__(self, name, **kwargs):
        """
        Parameters
        ---------
        name : str
            The name to use when referencing the model run

        """
        cases = [Case(name, name, [name, ]), ]
        super(self.__class__, self).__init__(name, cases, validate_data=False,
                                             **kwargs)

    def case_path(self, *args):
        """ Overridden case_path() method which simply returns the
        data_dir, since that's where the data is held.

        """

        return self.data_dir
