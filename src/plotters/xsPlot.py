#!/usr/bin/python

import pylab as plt
import numpy as np


def xsPlot(xs, energyVec=None, micro=True, label='xs1', style='-', fnameOut='xsPlt', figNum=4):
    if not energyVec:
        # assume 10 energy grp structure by default
        energyVec = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    xsVec = np.append(np.array([0]), xs)
    plt.figure(figNum)
    plt.step(energyVec, xsVec, linewidth='4', label=label, linestyle=style)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Energy (eV)")
    if micro:
        plt.ylabel("Micro Xsection [barns]")
    else:
        plt.ylabel("Macro Xsection [1/cm]")
    plt.legend()
    plt.savefig(fnameOut)
