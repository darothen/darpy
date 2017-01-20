""" Basic testing. """

import warnings
from contextlib import contextmanager

try:
    import unittest2 as unittest
except ImportError:
    import unittest
from unittest import TestCase
