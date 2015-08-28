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

import os
from collections import OrderedDict, Iterable, namedtuple
from itertools import product

from . utilities import remove_intermediates, arg_in_list, cdo_func
from . extract import extract_variable
from . io import load_variable
from . convert import create_master

__all__ = [ 'Case', 'Experiment', ]

Case = namedtuple('case', ['shortname', 'longname', 'vals'])

class Experiment(object):

    def __init__(self, name, cases, data_dir='./',
                 naming_case='', archive='', work_dir='data/',
                 validate_data=True):

        """

        name : str
        cases : iterable of Cases
        data_dir: str

        """

        self.name = name

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
        self._case_vals = { case: self._case_data[case].vals \
                            for case in self._cases }
        self._casenames = { case: self._case_data[case].longname \
                            for case in self._cases }

        if not naming_case:
            self.naming_case = self._cases[-1]
        else:
            self.naming_case = naming_case

        # Location of existing data
        assert os.path.exists(data_dir)
        self.data_dir = data_dir

        # Walk tree of directory containing existing data to ensure
        # that all the cases are represented
        if validate_data:
            self._validate_data()

        # Location of working directory for saving intermediate
        # files
        if not os.path.exists(work_dir):
            os.mkdirs(work_dir)
        self.work_dir = work_dir

        # Name of var archive
        if not archive:
            self.var_archive = os.path.join(self.work_dir,
                                            name + '.va')

    ## Validation methods

    def _validate_data(self):
        """ Validate that the specified data directory contains
        a hierarchy of directories which match the specified
        case layout.

        """

        root = self.data_dir

        path_bits = self.all_cases()
        for bits in path_bits:
            full_path = os.path.join(root, *bits)
            assert os.path.exists(full_path)

    ## Properties and accessors

    @property
    def cases(self):
        """ Property wrapper for list of cases. Superfluous, but
        it's really important that it doesn't get changed.
        """
        return self._cases

    def itercases(self):
        """ Generator for iterating over the encapsulated case
        information for this experiment

        >>> for case_info in marc_aie.itercases():
        ...     print case_info
        ('aer', 'aerosol emissions', ['F2000', 'F1850'])
        ('act', 'activation scheme', ['arg_comp', 'arg_min_smax'])

        """
        for case in self._cases:
            yield case, self._casenames[case], self._case_vals[case]

    def all_cases(self):
        """ Return an iterable of all the ordered combinations of the
        cases comprising this experiment.

        >>> for case in marc_aie.all_cases():
        ...     print case
        ('F2000', 'arg_comp')
        ('F1850', 'arg_comp')
        ('F2000', 'arg_min_smax')
        ('F1850', 'arg_min_smax')

        """
        return product(*self.all_case_vals())

    def all_case_vals(self):
        """ Return a list of lists which contain all the values for
        each case.

        >>> for case_vals in marc_aieall_case_vals():
        ...     print case_vals
        ['F2000', 'F1850']
        ['arg_comp', 'arg_min_smax']
        """
        return [ self._case_vals[case] for case in self._cases ]

    def case_path(self, *case_bits):
        """ Return the path to a particular set of case's output from this
        experiment.

        """
        return os.path.join(self.data_dir, *case_bits)

    ## Instance methods

    def extract(self, var, **kwargs):
        """ Extract a given variable.
        """

        extract_variable(self, var, **kwargs)

    def load(self, var, fix_times=False, master=False, **kwargs):
        """ Load the data for a variable into xray Datasets and
        attach the variable and data to the current Experiment.

        """

        if var._loaded:
            raise Exception("Data is already loaded")

        # Save the cases
        var._cases = self.cases

        # Get the location of the extracted variable data based
        # on the Experiment
        save_dir = self.work_dir

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
            cmd_dict = dict()
            if 'new_fields' in kwargs:
                cmd_dict['new_fields'] = kwargs['new_fields']
            var.master = create_master(self, var, **cmd_dict)

        # Attach to current Experiment
        self.__dict__[var.varname] = var

