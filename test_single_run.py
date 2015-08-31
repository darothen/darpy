import marc_analysis as ma

## Set up the experiment details

exp = ma.SingleCaseExperiment('arg_comp',
                              data_dir="/Users/daniel/Desktop/MARC_AIE/F2000/arg_comp",
                              work_dir="/Users/daniel/Desktop/MARC_AIE/work_test/")

## Set up the variables to extract
var_dict = dict(
    # FNET = ma.Var("FNET", ["FSNT", "FLNT"], 
    #               long_name="Net total radiative flux at TOA",
    #               units="W/m2",
    #               ncap_str="FNET=FSNT-FLNT;"),
    # LWP = ma.CESMVar("LWP", "TGCLDLWP", scale_factor=1e3, units='g/m2'),
    TS = ma.CESMVar("TS"),
    # CLDLOW = ma.CESMVar("CLDLOW", scale_factor=100., units='%')
)
for v, vv in var_dict.items():
    print(vv)
print("--"*40, "\n")

## Extract the variables and load a master dataset into memory
for key, d in var_dict.items():
    exp.extract(d, re_extract=True, years_omit=0)
    exp.load(d, master=True)