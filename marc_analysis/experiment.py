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
from collections import OrderedDict, Iterable
from itertools import product

from . utilities import remove_intermediates, arg_in_list, cdo_func

class Experiment(object):

    def __init__(self, name, cases, data_dir,
                 naming_case='', archive='', work_dir='data/',
                 validate_data=True):

        if isinstance(cases, OrderedDict):
            self.case_data = cases
        elif isinstance(cases, Iterable):
            self.case_data = OrderedDict()
            for case in cases:
                assert isinstance(case, Iterable)
                self.case_data[cases] = case
        else:
            raise ValueError("Couldn't process `cases`")

        self._cases = list(self.case_data.keys())

        if not naming_case:
            self.naming_case = self.cases[-1]
        else:
            self.naming_case = naming_case

        # Location of existing data
        assert os.path.exists(data_dir)
        self.data_dir = data_dir

        # Walk tree of directory containing existing data to ensure
        # that all the cases are represented
        if validate_data:
            self._validate_data(data_dir)

        # Location of working directory for saving intermediate
        # files
        if not os.path.exists(work_dir):
            os.mkdirs(work_dir)
        self.work_dir = work_dir

        # Name of var archive
        if not archive:
            self.var_archive = os.path.join(self.work_dir,
                                            name + '.va')

    def all_cases(self):
        """ Return an iterable of all the ordered combinations of the
        cases comprising this experiment.

        """
        return product(*[self.case_data[key] for key in self.cases])

    def case_path(self, *case_bits):
        """ Return the path to a particular set of case's output from this
        experiment.

        """
        return os.path.join(self.data_dir, *case_bits)

    def _validate_data(self):
        """ Validate that the specified data directory contains
        a hierarchy of directories which match the specified
        case layout.

        """

        root = self.data_dir

        path_bits = self.all_cases
        for bits in path_bits:
            full_path = os.path.join(root, *bits)
            assert os.path.exists(full_path)