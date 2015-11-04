#!/usr/bin/python
# Solve an infinite multiplying media problem.
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

import numpy as np
np.set_printoptions(linewidth=200)  # set print to screen opts


def fluxSolve(H, F, k, flux):
    """
    Update flux according to power iteration.
    """
    fissionOp = (1 / k) * F
    # scale flux by fission op
    b = np.dot(fissionOp, flux)
    # update flux by multiplying both sides by Ainv
    newFlux = np.linalg.solve(H, b)
    return newFlux


def cramersRule(H, F):
    """
    Compute eigenval using cramers rule.
    """
    from scipy import linalg
    eigl, eigv = linalg.eig(H, F)
    print("Cramers Rule k-effs:")
    print(eigl)
    return eigl, eigv


def kUpdate(nuFission, k, flux):
    """
    Update k according to power iteration.
    """
    newK = k * np.sum(nuFission * flux[-1]) / np.sum(nuFission * flux[-2])
    return newK


def checkConverge(k, flux):
    """
    Performes convergence check on flux and eigenvalue.
    """
    deltK = np.abs(k[-1] - k[-2])
    l2Flux = np.linalg.norm((flux[-1] - flux[-2]) / flux[-1])
    if deltK < 1e-6 and l2Flux < 1e-6:
        return deltK, l2Flux, True
    else:
        return deltK, l2Flux, False


#@profile
def solveCrit(infMediaMat, k0=1.1, flux0=np.ones(10), **kwargs):
    """
    Takes a mixed material class instance and initial guesses for the flux and
    eigenvalue.  Optionally find eigenvalues using kramers rule.
    Performs power iterations to obtain flux(E) and k for the inf. medium.
    """
    # Optional Arguments
    cramerCheck = kwargs.pop('cceck', False)
    verbosePrint = kwargs.pop('verbose', True)
    #
    #
    # setup multigroup transport operator
    Ngroups = len(infMediaMat.macroProp['Ntotal'])
    A = np.eye(Ngroups) * infMediaMat.macroProp['Ntotal'] - \
        infMediaMat.macroProp['Nskernel'][0]
    # setup fission matrix
    F = np.dot(np.array([infMediaMat.macroProp['chi']]).T,
               np.array([infMediaMat.macroProp['Nnufission']]))
    if verbosePrint:
        print("Multigroup transport operator:")
        print(A)
        print("Fission source matrix:")
        print(F)
    # initial guess for k and flux (keep track of past k estimates)
    k, flux = [k0], [flux0]
    kResid, fluxResid = [], []
    # Perform cramers rule for a check
    if cramerCheck:
        cramersRule(A, F)
    # Init stop criteria
    converged, i = False, 1
    print("======================================================")
    print("_________K-Eigenvalue Inf Medium Solver Start_________")
    print("iteration|    k-eff    |    delta-k   |   l2Norm-flux     ")
    print("======================================================")
    print(str("%i" % 0) + "         " + str("%.5f" % k[-1]))
    while not converged:
        # Power iterations
        flux.append(fluxSolve(A, F, k[-1], flux[-1]))
        k.append(kUpdate(infMediaMat.macroProp['Nnufission'], k[-1], flux))
        deltK, l2Flux, converged = checkConverge(k, flux)
        kResid.append(deltK)
        fluxResid.append(l2Flux)
        print(str("%i" % i) + "         " + str("%.6f" % k[-1]) + "       " +
              str("%.6e" % kResid[-1]) + "     " + str("%.6e" % fluxResid[-1]))
        if i > 10:
            break
        i += 1
    print("======================================================")
    print("Final Flux Vector Estimate:")
    print(flux[-1])
    print("======================================================")
    return k, flux


if __name__ == "__main__":
    # Load xs database
    import materials.materialMixxer as mx
    import utils.pinCellMatCalc as pcm
    mx.genMaterialDict('./materials/newXS')
    # Create pin cell material
    pinCellMaterial = pcm.createPinCellMat()
    modDict = {'u235': False, 'u238': False, 'zr90': False}
    resDict = {'h1': False, 'o16': False, 'zr90': False}
    ssPinCellMaterial = pinCellMaterial.selfSheild(modDict, resDict)
    # ssPinCellMaterial = pinCellMaterial.selfSheild()
    # Solve k-eigenvalue problem
    kVec, fluxVec = solveCrit(ssPinCellMaterial, k0=1.1)
    # Print f-factors for u238 and u235
    print(ssPinCellMaterial.microDat['u235']['f'])
    print(ssPinCellMaterial.microDat['u238']['f'])
    # Compute U235 and U238 reaction rates
    # numberDensity235 = pinCellMaterial.nDdict['u235']
    # numberDensity238 = pinCellMaterial.nDdict['u238']
    # u235 = mx.mixedMat({'u235': numberDensity235})
    # u238 = mx.mixedMat({'u238': numberDensity238})
    # Ru238 = np.sum(u238.macroProp['Nnufission'] * fluxVec[-1])
    # Ru235 = np.sum(u235.macroProp['Nnufission'] * fluxVec[-1])
    # fRR = np.sum([Ru238, Ru235])
    # print("Relative U238 fission reaction rate: " + str(Ru238 / fRR))
    # print("Relative U235 fission reaction rate: " + str(Ru235 / fRR))
    # plot results
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    import plotters.fluxEplot as flxPlt
    flxPlt.plotFluxE(fluxVec[-1][::-1], label='Self Shield ON')
    kVec, fluxVec = solveCrit(pinCellMaterial, k0=1.1)
    flxPlt.plotFluxE(fluxVec[-1][::-1], label='Self Shield OFF')
