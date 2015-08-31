#!/usr/bin/env python
""" Some setup to initialize a notebook for analyzing model output. """

import sys

import os
import numpy as np
import pandas as pd
import xray

DEFAULT_DATA_DIR = "data/"

# Set default aesthetic properties
import seaborn as sns
sns.set(style="white", rc={ 'axes.labelsize': 14,
                            'ytick.labelsize': 12,
                            'xtick.labelsize': 12,
                            'legend.fontsize': 13, } )

# Create a basic argument parser to handle some useful functions:
from argparse import ArgumentParser
parser = ArgumentParser(description="Load common settings for notebook analyses")

# 1) Potentially create a specified directory for saving data
parser.add_argument("-d", "--datadir", type=str, default=DEFAULT_DATA_DIR,
                    help="(relative) directory path for saving data")

# 2) Location of analysis package 
parser.add_argument("-a", "--analysis", type=str, default="../",
                    help="path to location of analysis package for import") 

if __name__ == "__main__":

    args = parser.parse_args()

    # 1) Set up the data directory
    DATA_DIR = os.path.join(os.getcwd(), args.datadir)

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print("Making data directory at %r" % DATA_DIR)

    # 2) Add analysis package to path and import
    sys.path.insert(0, args.analysis)
    import marc_analysis as ma

    