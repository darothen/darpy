#!/usr/bin/env python
"""
Plot PDFs of number concentration and size for AIT, ACC, MOS, MBS aerosol modes; compare
between all levels and just a single level.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", font='serif', rc={'font.size': 12})

modes = ["AIT", "ACC", "MOS", "MBS"]
mu_modes = ["mu%s" % s for s in modes]
N_modes = ["n%s" % s for s in modes]

nbins = np.linspace(-2, 5, 33)
mubins = np.linspace(-5, 0, 31)

## Read in the data
df_all = pd.read_csv("B20norad.july_df_params_ti-000.csv")
df_single = pd.read_csv("B20norad.july.lev26_df_params_V-0020.csv")

## Extract number/mu, take log10
df_all = df_all[mu_modes + N_modes].apply(np.log10)
df_all.rename(columns={ s: "log10(%s)" % s for s in mu_modes + N_modes }, inplace=True)

df_single = df_single[mu_modes + N_modes].apply(np.log10)
df_single.rename(columns={ s: "log10(%s)" % s for s in mu_modes + N_modes }, inplace=True)

## Do some quick analysis, and plot histograms
head = "  mode  |   N (min,max)  |  mu (min,max)  \n" + \
       "----------------------------------------------------------"
str_f = "{0:^8s}| ({1:5.1f}, {2:5.1f}) | ({3:5.1f}, {4:5.1f})"

for s, df in zip(["all", "lev26"], [df_all, df_single]):
    print s
    print head
    for aer in modes:
        n = df["log10(n%s)" % aer]
        mu = df["log10(mu%s)" % aer]
        print str_f.format(aer, n.min(), n.max(), mu.min(), mu.max())
    print

## 2-panel plots
for aer in modes:
    fig, [axn, axmu] = plt.subplots(1, 2, figsize=(10, 3))
    plt.subplots_adjust(wspace=0.3, bottom=0.2)
    for s, df in zip(["all", "lev26"], [df_all, df_single]):
        print aer
        n = df["log10(n%s)" % aer]
        mu = df["log10(mu%s)" % aer]

        axn = sns.distplot(n, kde=True, bins=nbins, ax=axn, label=s,
                          axlabel="log10(Number) - %s, log10(cm$^{-3}$)" % aer)
        axmu = sns.distplot(mu, kde=True, bins=mubins, ax=axmu, label=s,
                           axlabel="log10(Mean Radius) - %s, log10(micron)" % aer)
    sns.despine()
    axn.set_xlim(nbins[0], nbins[-1])
    axmu.set_xlim(mubins[0], mubins[-1])
    for ax in [axn, axmu]:
        ax.legend(loc='upper right')
        ax.set_ylabel("Probability Density")
    plt.savefig("lev_pdf_compare_%s.pdf" % aer, transparent=True)

