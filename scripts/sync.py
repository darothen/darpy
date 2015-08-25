#!/usr/bin/env

import os, sys

from subprocess import call
from argparse import ArgumentParser

"""
SYNC - sync the experiment analysis between local machine
       and remote server

       Should always be run from the local machine. If no file
       types specified, it'll at a minimum sync the README.

Author - Daniel Rothenberg <darothen@mit.edu>
Version - 1/26/2015
"""

## Remote path setup
REMOTE_SERVER = "legion.mit.edu"
REMOTE_USER   = "darothen"
REMOTE_PATH   = "/home/%s/workspace/MARC_AIE/" % REMOTE_USER

## File extensions for exclusion, based on command line arguments
OUTPUT_EXT = [ ".png", ".pdf", ]
DATA_EXT   = [ ".nc", ".hdf", ]
CODE_EXT   = [ ".pro", ".py", ".ipynb", ]

parser = ArgumentParser(description="Sync MARC_AIE experiment files")
parser.add_argument("-c", "--code", action="store_true",
                    help="Copy code files (dangerous!)")
parser.add_argument("-d", "--data", action="store_true",
                    help="Copy data files (%r)" % DATA_EXT)
parser.add_argument("-o", "--output", action="store_true",
                    help="Copy script output/results files (%r)" % OUTPUT_EXT)
parser.add_argument("--debug", action='store_true')
dest_group = parser.add_mutually_exclusive_group(required=True)
dest_group.add_argument("--to", action="store_true")
dest_group.add_argument("--from", action="store_false")

# Helper function to append carriage return to list of strings
add_nl = lambda l: "\n".join(["*" + elem for elem in l])

if __name__ == "__main__":

    args = parser.parse_args()

    ## Write excludes to file 
    print 
    with open(".include", 'w') as f:
        if args.data:
            f.writelines(add_nl(DATA_EXT))
            print "Including data files (%r)" % DATA_EXT
        if args.output:
            f.writelines(add_nl(OUTPUT_EXT))
            print "Including output files (%r)" % OUTPUT_EXT
        if args.code:
            f.writelines(add_nl(CODE_EXT))
            print "Including code files (%r)" % CODE_EXT
        f.write("\n")


    ## Execute rsync call
    if args.to:
        print "\nSyncing from local to %s on %s" % (REMOTE_PATH, REMOTE_SERVER)
        src  = os.getcwd()
        dest = "%s@%s:%s" % (REMOTE_USER, REMOTE_SERVER, REMOTE_PATH)

        sys.exit("Syncing to remote has been disabled.")
    else:
        print "\nSyncing from %s on %s to local" % (REMOTE_PATH, REMOTE_SERVER)
        src  = "%s@%s:%s" % (REMOTE_USER, REMOTE_SERVER, REMOTE_PATH)
        dest = os.getcwd()

    call_args = [src, dest, ]
    call_flags = ["a", "r", "u", "P"]
    if args.debug:
        call_flags.append("n")
    call([
        "rsync", "-"+"".join(call_flags),
        "--prune-empty-dirs",
        "--include-from", ".include",
        "--exclude-from", ".exclude",
        "-e", "ssh",
    ] + call_args)