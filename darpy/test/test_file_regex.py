""" Test cases for file matching with regular expressions.

"""

import re
import unittest

from itertools import product
from random import randint

from .. extract import OUTPUT_FN_REGEX, _format_regex

class TestFileRegex(unittest.TestCase):

    def test_monthly_file(self):
        comp, hist = 'cam', 0

        regex = re.compile(_format_regex(comp, hist), re.VERBOSE)

        for year, month in product(range(1, 5), range(1, 13)):
            fn = "experiment_name.{comp:s}.h{hist:1d}.{year:04d}-{month:02d}.nc".format(comp=comp, hist=hist, year=year, month=month)

            print(fn)

            match = regex.match(fn)
            groups = match.groupdict()

            self.assertEqual('{:04d}'.format(year), groups['year'])
            self.assertEqual('{:02d}'.format(month), groups['month'])
            self.assertEqual('{:s}'.format(comp), groups['comp'])
            self.assertEqual('{:1d}'.format(hist), groups['hist'])

    def test_monthly_file_digit_comp(self):
        comp, hist, year = 'cam', 0, 2000

        regex = re.compile(_format_regex(comp, hist), re.VERBOSE)

        for digit, month in product(range(2), range(1, 13)):
            fn = "experiment_name.{comp:s}{digit:1d}.h{hist:1d}.{year:04d}-{month:02d}.nc".format(comp=comp, digit=digit, hist=hist, year=year, month=month)

            print(fn)

            match = regex.match(fn)
            groups = match.groupdict()

            self.assertEqual('{:04d}'.format(year), groups['year'])
            self.assertEqual('{:02d}'.format(month), groups['month'])
            self.assertEqual('{:s}'.format(comp), groups['comp'])
            self.assertEqual('{:1d}'.format(hist), groups['hist'])

    def test_daily_file(self):
        comp, hist, year, month = 'cam', 1, 2000, 6

        regex = re.compile(_format_regex(comp, hist), re.VERBOSE)

        for day in range(31):
            time = randint(0, int(1e5) - 1)
            fn = "experiment_name.{comp:s}.h{hist:1d}.{year:04d}-{month:02d}-{day:02d}-{time:05d}.nc".format(comp=comp, hist=hist, year=year, month=month, day=day, time=time)

            print(fn)

            match = regex.match(fn)
            groups = match.groupdict()

            self.assertEqual('{:04d}'.format(year), groups['year'])
            self.assertEqual('{:02d}'.format(month), groups['month'])
            self.assertEqual('{:s}'.format(comp), groups['comp'])
            self.assertEqual('{:1d}'.format(hist), groups['hist'])
            self.assertEqual('{:05d}'.format(time), groups['time'])

    def test_daily_file_digit(self):
        comp, hist, year, month = 'cam', 1, 2000, 6

        regex = re.compile(_format_regex(comp, hist), re.VERBOSE)

        for day, digit in product(range(10), range(2)):
            time = randint(0, int(1e5) - 1)

            fn = "experiment_name.{comp:s}{digit:1d}.h{hist:1d}.{year:04d}-{month:02d}-{day:02d}-{time:05d}.nc".format(comp=comp, digit=digit, hist=hist, year=year, month=month, day=day, time=time)

            print(fn)

            match = regex.match(fn)
            groups = match.groupdict()

            self.assertEqual('{:04d}'.format(year), groups['year'])
            self.assertEqual('{:02d}'.format(month), groups['month'])
            self.assertEqual('{:s}'.format(comp), groups['comp'])
            self.assertEqual('{:1d}'.format(hist), groups['hist'])
            self.assertEqual('{:05d}'.format(time), groups['time'])


if __name__ == "__main__":
    unittest.main()