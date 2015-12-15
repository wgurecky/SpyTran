import numpy as np
import h5py
from scipy import interpolate


def diff1D(fdfile, fefile):
    fd = h5py.File(fdfile)
    fe = h5py.File(fefile)
    fdmesh = fd['mesh']
    femesh = fe['mesh0']
    #
    # For each group:
    for g in range(10):
        # Construct interpolant on base mesh
        fdComp = fd['scalarFluxes'][:, g]
        #fdS = interpolate.interp1d(fdmesh, fdGsoln, kind='linear')
        feGsoln = fe['groupFlx' + str(g)]
        fdS = interpolate.interp1d(femesh, feGsoln, kind='linear')
        feComp = fdS(fdmesh)
        maxDiff = 100 * np.median(abs(fdComp - feComp) / feComp)
        rmsDiff = 100 * np.min(abs(fdComp - feComp) / fdComp)
        print("for group " + str(g + 1) + " diffs:")
        print("-----------------------------------")
        print("max rel percent diff = " + str(maxDiff))
        print("RMS diff = " + str(rmsDiff))


if __name__ == "__main__":
    diff1D('1d_7region.h5', '1dfedata.h5')
