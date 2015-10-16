#!/usr/bin/python
#
# Used to plot flux vs Energy spectrum.
#
# Requres upper bin energies and flux vector

import pylab as plt
import numpy as np


def plotFluxE(fluxVec, energyVec=None, fnameOut='fluxE', figNum=0, label='fluxE'):
    if not energyVec:
        # assume 10 energy grp structure by default
        energyVec = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    fluxVec = np.append(np.array([0]), fluxVec)
    plt.figure(figNum)
    plt.step(energyVec, fluxVec / np.sum(fluxVec), linewidth='4', label=label)
    plt.ylim([5e-3, 1])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Energy (eV)")
    plt.ylabel("Log Flux (Arbitrary Scaling)")
    plt.legend()
    plt.savefig(fnameOut)
