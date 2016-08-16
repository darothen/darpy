from copy import copy, deepcopy
from textwrap import dedent
try:
    import cPickle as pickle
except ImportError:
    import pickle
from itertools import product
import os


from .. experiment import Case, Experiment

from . import unittest

case_act = Case('act', 'Activation Scheme',
                ['arg_comp', 'nenes_comp', 'pcm_gCCN3'])
case_aer = Case('aer', 'Aerosol Emissions',
                ['PD', 'PI'])
case_letter = Case('letter', 'Letter', ['x', 'y', 'z'])

class TestExperiment(unittest.TestCase):

    def test_repr(self):
        exp = Experiment("Test Experiment", [case_act, case_aer],
                         case_path="", validate_data=False)
        expected = dedent("""\
        Test Experiment -
            Activation Scheme:  [arg_comp, nenes_comp, pcm_gCCN3]
            Aerosol Emissions:  [PD, PI]"""
        )
        actual = "\n".join(x.rstrip() for x in repr(exp).split("\n"))
        print(expected, '\n', actual)
        self.assertEqual(expected, actual)

    def test_case_attrs(self):

        case_list = [case_act, case_aer]
        exp = Experiment("Test Experiment", case_list,
                         case_path="", validate_data=False)

        # Cases property
        self.assertEqual(exp.cases, [case.shortname for case in case_list])

        # itercases generator
        actual_cases = [(case.shortname, case.longname, case.vals)
                        for case in case_list]
        exp_cases = [case for case in exp.itercases()]
        self.assertEqual(exp_cases, actual_cases)

        # all_cases list
        actual_case_gen = product(*[case.vals for case in case_list])
        exp_case_gen = exp.all_cases()
        for expected, actual in zip(exp_case_gen, actual_case_gen):
            self.assertEqual(expected, actual)

    def test_walk_cases_default(self):
        """ Walk cases when no case_path is provided - so bits are
        ordered in the order passed as cases """
        case_list = [case_act, case_aer]
        exp = Experiment("Test Experiment", case_list,
                         validate_data=False)

        actual_paths = []
        for case_ordered in product(*[case.vals for case in case_list]):
            actual_paths.append(os.path.join(*case_ordered))
        exp_paths = exp._walk_cases()
        for expected, actual in zip(exp_paths, actual_paths):
            print(expected, actual)
            self.assertEqual(expected, actual)

    def test_walk_cases_custom(self):
        """ Walk cases when no case_path is provided - so bits are
        ordered in the order passed as cases """
        case_list = [case_act, case_aer]
        exp = Experiment("Test Experiment", case_list,
                         case_path="{act}/{aer}", validate_data=False)

        actual_paths = []
        for case_ordered in product(*[case.vals for case in case_list]):
            actual_paths.append(os.path.join(*case_ordered))
        exp_paths = exp._walk_cases()
        for expected, actual in zip(exp_paths, actual_paths):
            print(expected, actual)
            self.assertEqual(expected, actual)

        # Generic - no format arguments
        exp = Experiment("Test Experiment", case_list,
                         case_path="", validate_data=False)
        for p in exp._walk_cases():
            self.assertEqual(p, "")
