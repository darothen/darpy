import os
from itertools import product
from subprocess import call

from .case_setup import case_path, WORK_DIR
from .utilities import _GIT_COMMIT
from functools import reduce

__all__ = ['extract_variable', ]

def extract_variable(var, out_suffix="", save_dir=WORK_DIR,
                     years_omit=5, years_offset=0, 
                     re_extract=False,
                     act_cases=[], aer_cases=[]):
    """ Extract a timeseries of data for one variable from the raw CESM/MARC output. 

    A few features bear some explanation:

    1) `years_offset` allows you to add a certain number of years to the time
       variable definition. By default, most simulations I do just start in year
       0001. However, this breaks some calendar-anaylsis routines, since the
       standard/Gregorian calendars are not defined prior to the 1500's. It's much
       easier to simply offset the years to the present day and take advantage
       of the analysis tools machinery than it would be to re-write parts of their
       functionality.

    2) By default, the first 5 years of the simulation data will be omitted.

    Parameters
    ----------
    var : (var_data.Var or subtype) or str
        The container for the variable info being extracted or just the
        name of a default CESM var
    out_suffix : string
        Suffix for final filename
    save_dir : string
        Path to save directory; defaults to case WORK_DIR.
    years_omit : int
        The number of years to omit from the beginning of the dataset
    years_offset : int
        The number of years to add to the time unit definition.
    re_extract : bool
        Force extraction even if files already exist
    act_cases, aer_cases : list of str
        Experiment names to extract

    """

    if isinstance(var, str):
        from .var_data import CESMVar
        var = CESMVar(var, )

    print() 
    print("---------------------------------")
    print("Processing %s from CESM output" %  var.oldvar)
    print("   for cases %r, %r" % (act_cases, aer_cases))
    if hasattr(var, 'lev_bnds'): 
        print("   for levels %r" % var.lev_bnds)
    if hasattr(var, 'cdo_method'):
        print("   applying methods %r" % var.cdo_method)
    else:
        print("   No cdo method applied")

    intermediates = []

    ## Combine existing variables, if ncessary
    combine = isinstance(var.oldvar, (list, tuple))
    if combine:
        comb_vars = var.oldvar[:]
        assert var.varname # make sure it was provided!
        print("   will combine %r to compute %s" % (comb_vars, var.varname))
        print("   using %s" % var.ncap_str if var.ncap_str else "simple addition")
        oldvar_extr = ",".join(comb_vars)
    else:
        oldvar_extr = var.oldvar
        if not var.varname: var.varname = var.oldvar

    for i, (act, aer) in enumerate(product(act_cases, aer_cases)):
        print()
        print("   %02d)" % (i+1, ), act, aer)

        retained_files = []
    
        # fn_extr :-> monthly file with variable subset
        fn_extr = "%s_%s_%s_monthly.nc" % (act, aer, var.varname)
        in_file = os.path.join(case_path(act, aer),
                      "%s.cam2.h0.00[0,1][0,%d-9]-*.nc" % (act, years_omit+1, ))
        out_file = os.path.join(save_dir, fn_extr)

        # fn_final :-> the end result of this output
        fn_final = "%s_%s_%s" % (act, aer, var.varname)
        if out_suffix: 
            fn_final += "_%s" % out_suffix
        fn_final += ".nc"
        out_file_final = os.path.join(save_dir, fn_final)

        retained_files.append(out_file)

        if ( re_extract or not os.path.exists(out_file_final) ): 
            print("      Extracting from original dataset")

            # These are important metadata vars (vertical coord system,
            # time, etc) which we will always want to save
            SAVE_VARS = ",time_bnds,hyam,hybm,PS,P0,gw"

            if not hasattr(var, 'lev_bnds'):
                call("ncrcat -O -v %s %s %s" % 
                         (oldvar_extr+SAVE_VARS, in_file, out_file),
                     shell=True)
            else: 
                if len(var.lev_bnds) == 2: lo, hi = var.lev_bnds
                else: lo, hi = var.lev_bnds[0], var.lev_bnds[0]
                call("ncrcat -O -d lev,%d,%d -v %s %s %s" % 
                      (lo, hi, oldvar_extr+SAVE_VARS, in_file, out_file),
                     shell=True)

            if combine:
                print("      combining vars")
                ncap2_args = ["ncap2", "-O", "-s", 
                              "%s=%s" % (var.varname, "+".join(comb_vars)), 
                              out_file, "-o", out_file]
                if var.ncap_str: ncap2_args[3] = "'%s'" % var.ncap_str
                #print ncap2_args
                call(" ".join(ncap2_args), shell=True)

            if hasattr(var, 'cdo_method'):
                if out_suffix: # override if its provided
                    cdo_bit = out_suffix
                else:
                    cdo_bit = "_".join(cdo_arg for cdo_arg in var.cdo_method)

                fn_cdo = "%s_%s_%s_%s.nc" % (act, aer, var.varname, cdo_bit)
                cdo_out_file = os.path.join(save_dir, fn_cdo)

                cdo_func(var.cdo_method, out_file, cdo_out_file)

                out_file = cdo_out_file

                retained_files.append(out_file)

            if (var.varname and (not var.name_change) and not combine):
                for of in retained_files:
                    call(['ncrename', '-v', ".%s,%s" % (var.oldvar, var.varname), of])

            ## Add attributes to variable and global info
            att_list = [ 
                "-a", "years_omit,global,o,i,%d" % (int(years_omit), ), 
                "-a", "years_offset,global,o,i,%d" % (int(years_offset), ), 
                "-a", "git_commit,global,o,c,%s" % _GIT_COMMIT,
            ]
            if hasattr(var, 'attributes'):
                print("      modifying var attributes")
                for att, val in var.attributes.items():
                    dtype = { int: 'i', 
                              str: 'c',
                              float: 'f', }[type(val)]
                    print("         %s: %s (%s)" % (att, val, dtype))

                    att_list.extend(["-a", "%s,%s,o,%s,%s" % \
                                     (att, var.varname, dtype, val)])
                if years_offset > 0:
                    print("         adding time offset")
                    att_list.extend(["-a", 'units,time,o,c,days since %04d-01-01 00:00:00' %
                                           years_offset])
            if att_list:
                for of in retained_files:
                    call(["ncatted", "-O", ] + att_list + [of, ])

            remove_intermediates(intermediates)

            call(['mv', out_file, out_file_final])
            print("      ...done -> %s" % out_file_final)

        else:
            print("      Extracted dataset already present.")

def remove_intermediates(intermediates):
    """ Delete list of intermediate files. """
    for fn in intermediates:
        print("Removing", fn)
        os.remove(fn)

def arg_in_list(arg, arg_list):
    """ Returns true if `arg` is a partial match for any value in `arg_list`. """
    return reduce(lambda a, b: a or b, [arg in s for s in arg_list])
    
def cdo_func(args, fn_in, fn_out, silent=True):
    """ Execute a sequence of CDO functions in a single process. """
    call_args = ["cdo", "-O", ]
    if silent: call_args.append("-s")

    def _proc_arg(arg):
        if isinstance(arg, (list, tuple)):
            return ",".join(arg)
        else:
            return arg

    call_args.append(args[0])

    for arg in args[1:]:
        call_args.append("-" + _proc_arg(arg))
    call_args.extend([fn_in, fn_out])

    print("      CDO - %s" % " ".join(call_args))
    call(call_args)

    # Post-process using ncwa to remove variables which have been 
    # averaged over
    # if arg_in_list("vert", args):
    #     call(['ncwa', '-O', '-a', "lev", fn_out, fn_out])
    if arg_in_list("tim", args):
        call(['ncwa', '-O', '-a', "time", fn_out, fn_out])
    # if arg_in_list("zon", args):
    #     call(['ncwa', '-O', '-a', "lon", fn_out, fn_out])