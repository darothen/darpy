#!/usr/bin/env python
""" Test the basic scripting functionality. """

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

from marc_analysis.scripting import BasicParser

parser = BasicParser(__doc__)
parser.add_argument("name", help="Name to print to console")

if __name__ == "__main__":

    args = parser.parse_args()

    logger.info("Name is: {}".format(args.name))
    logger.debug("(DEBUG) Name is: {}".format(args.name))