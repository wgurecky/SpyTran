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
from copy import deepcopy
mx.genMaterialDict('./materials/hw2')


def computeVols():
    # Pin cell dimensions
    ppitch = 1.33  # [cm]
    rfuel = 0.412  # [cm]
    rclad = 0.475  # [cm]
    #
    volTot = ppitch ** 2.
    volFuel = np.pi * rfuel ** 2.
    volClad = np.pi * (rclad ** 2. - rfuel ** 2.)
    volMod = volTot - volFuel - volClad
    return [volFuel, volClad, volMod]


def createPinCellMat():
    # Create fuel mixture
    duUO2 = mx.mixedMat({'u238': 1. / 3., 'o16': 2 / 3.})
    duUO2.setDensity(10.35)
    heuUO2 = mx.mixedMat({'u235': 1 / 3., 'o16': 2 / 3.})
    heuUO2.setDensity(10.35)
    fuelMat = 0.964 * duUO2 + 0.036 * heuUO2
    #fuelMat = 0.6 * duUO2 + 0.4 * heuUO2
    fuelMat.setDensity(10.35)
    # create cladding mixture
    cladMat = mx.mixedMat({'zr90': 1.0})
    cladMat.setDensity(5.87)
    # create moderator mixture
    modrMat = mx.mixedMat({'h1': 2. / 3., 'o16': 1 / 3.})
    modrMat.setDensity(1.0)
    # Homogenize pin cell materials into one big soup
    regionVols = computeVols()
    massFuel = fuelMat.density * regionVols[0]
    massClad = cladMat.density * regionVols[1]
    massModr = modrMat.density * regionVols[2]
    volTot = sum(regionVols)
    massTot = massFuel + massClad + massModr
    pinCellMat = (massFuel / massTot) * deepcopy(fuelMat) + \
                 (massClad / massTot) * deepcopy(cladMat) + \
                 (massModr / massTot) * deepcopy(modrMat)
    #pinCellMat = (regionVols[0] / volTot) * deepcopy(fuelMat) + \
    #             (regionVols[1] / volTot) * deepcopy(cladMat) + \
    #             (regionVols[2] / volTot) * deepcopy(modrMat)
    return pinCellMat


def fluxSolve(H, F, k, flux):
    iH = np.linalg.inv(H)
    fissionOp = (1 / k) * F
    # scale flux by fission op
    b = np.dot(fissionOp, flux)
    # update flux by multiplying both sides by Ainv
    return newFlux


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


def solveCrit(infMediaMat):
    """
    Takes a mixed material class instance.
    Performs power iterations to obtain flux(E) and k for the inf. medium.
    """
    np.set_printoptions(linewidth=200)
    # setup multigroup transport operator
    A = np.eye(10) * infMediaMat.macroProp['Ntotal'] - \
        infMediaMat.macroProp['Nskernel'][0]
    print("Multigroup transport operator:")
    print(A)
    # setup fission matrix
    F = np.dot(np.array([infMediaMat.macroProp['chi']]).T,
               np.array([infMediaMat.macroProp['Nnufission']]))
    print("Fission source matrix:")
    print(F)
    # initial guess for k and flux (keep track of past k estimates)
    k, flux = [1.1], [np.ones(10)]
    kResid, fluxResid = [], []
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
        flux[-1] *= 10. / sum(flux[-1])
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
    fluxVec, kVec = solveCrit(createPinCellMat())
    # plot results
    pass
