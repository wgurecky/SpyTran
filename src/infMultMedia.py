#!/usr/bin/python
# Solve an infinite multiplying media problem.
# Use 10 energy groups
#
# Obtain:
#   - flux(E)
#   - k-eigenvalue
#
# Balance equation in matrix notation:
# A * b = (1/k) * F * b
# where:
#   A = Multigroup transport operator
#   b = Multigroup flux
#   k = Nuetron mult factor
#   F = fission matrix
#     = Chi.tranpose * nuFission
#
# And k is updated by the following
# k(i+1) = k(i) * (AF / F)

import materials.materialMixxer as mx
import numpy as np


def fluxSolve(H, F, k, flux):
    # iH = np.linalg.inv(H)
    fissionOp = (1 / k) * F
    # scale flux by fission op
    b = np.dot(fissionOp, flux)
    # update flux by multiplying both sides by Ainv
    newFlux = np.linalg.solve(H, b)
    return newFlux


def cramersRule(H, F):
    from scipy import linalg
    eigl, eigv = linalg.eig(H, F)
    return eigl, eigv


def kUpdate(nuFission, k, flux):
    newK = k * np.sum(nuFission * flux[-1]) / np.sum(nuFission * flux[-2])
    return newK


def checkConverge(k, flux):
    deltK = np.abs(k[-1] - k[-2])
    l2Flux = np.linalg.norm((flux[-1] - flux[-2]) / flux[-1])
    if deltK < 1e-6 and l2Flux < 1e-6:
        return deltK, l2Flux, True
    else:
        return deltK, l2Flux, False


def solveCrit(infMediaMat, k0=1.1, flux0=np.ones(10)):
    """
    Takes a mixed material class instance.
    Performs power iterations to obtain flux(E) and k for the inf. medium.
    """
    np.set_printoptions(linewidth=200)
    groups = len(infMediaMat.macroProp['Ntotal'])
    # setup multigroup transport operator
    A = np.eye(groups) * infMediaMat.macroProp['Ntotal'] - \
        infMediaMat.macroProp['Nskernel'][0]
    print("Multigroup transport operator:")
    print(A)
    # setup fission matrix
    F = np.dot(np.array([infMediaMat.macroProp['chi']]).T,
               np.array([infMediaMat.macroProp['Nnufission']]))
    print("Fission source matrix:")
    print(F)
    # initial guess for k and flux (keep track of past k estimates)
    k, flux = [k0], [flux0]
    kResid, fluxResid = [], []
    # Perform cramers rule for a check
    cramersRule(A, F)
    # Init stop criteria
    converged, i = False, 1
    print("_________K-Eigenvalue Inf Medium Solver Start_________")
    print("iteration |   k-eff   |   k-diff  |   l2Norm-flux     ")
    print("======================================================")
    print(str("%i" % 0) + "         " + str("%.5f" % k[-1]))
    while not converged:
        # Power iterations
        print(flux[-1])
        flux.append(fluxSolve(A, F, k[-1], flux[-1]))
        k.append(kUpdate(infMediaMat.macroProp['Nnufission'], k[-1], flux))
        deltK, l2Flux, converged = checkConverge(k, flux)
        kResid.append(deltK)
        fluxResid.append(l2Flux)
        print(str("%i" % i) + "         " + str("%.5f" % k[-1]))
        if i > 10:
            break
        i += 1
        converged = False
    return k, flux


if __name__ == "__main__":
    # Load xs database
    mx.genMaterialDict('./materials/hw2')
    import pinCellMatCalc as pcm
    fluxVec, kVec = solveCrit(pcm.createPinCellMat())
    # plot results
    pass
