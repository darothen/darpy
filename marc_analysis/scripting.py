"""
Set up some basic command-line parsers for analysis scripts
using features from this package.
"""

import logging
logger = logging.getLogger()

from argparse import ArgumentParser, RawDescriptionHelpFormatter

class BasicParser(ArgumentParser):
    """ Custom wrapper for `ArgumentParser` class which adds some
    default parsing options, including:

        -d, --debug      Enable debug/verbose logging

    """

    def __init__(self, description="Default program", **kwargs):

        super(self.__class__, self).__init__(
            description=description,
            formatter_class=RawDescriptionHelpFormatter,
            **kwargs
        )

        # Add some default options
        self.add_argument("-d", "--debug", action='store_true',
                          help="Enable debug/verbose logging")

    def parse_args(self, args=None, namespace=None):

        args = super(self.__class__, self).parse_args(
            args=args, namespace=namespace
        )

        # Enable logger debug statements
        if args.debug:
            logger.setLevel(logging.DEBUG)

        return args