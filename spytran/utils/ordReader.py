# Ordinate set parser
#

import numpy as np
import scipy.special as spc


def createLegArray(sNmu, lMax):
    legArray = np.zeros((lMax + 1, len(sNmu)))
    for l in range(lMax + 1):
        legArray[l, :] = spc.eval_legendre(l, sNmu)
    return legArray


def createSphrHarm(sNmu, omega, lMax):
    sphArray = np.zeros((lMax + 1, lMax + 1, len(sNmu), len(omega)))
    for m in range(lMax + 1):
        # loop over legendre order
        for l in range(lMax + 1):
            for muI, mu in enumerate(sNmu):
                for wI, w in enumerate(omega):
                    sphArray[m, l, muI, wI] = spc.sph_harm(m, l, mu, w).real
    return sphArray


def readOrdFile(inFile, sNords):
    ords = np.fromtxt(inFile)
    if len(ords) == sNords:
        ordinateSet = ords
    else:
        pass
    return ordinateSet


def levelSymQuadSet(sN):
    if sN == 4:
        mu = np.array([0.3500212, 0.8688903])
        wN = np.array([1 / 3.])
    elif sN == 6:
        mu = np.array([0.2666355, 0.6815076, 0.9261808])
        wN = np.array([0.1761263, 0.1572071])
    else:
        # default s8
        mu = np.array([0.2182179, 0.5773503, 0.7867958, 0.9511897])
        wN = np.array([0.1209877, 0.0907407, 0.0925926])
    ordinateSet = np.array([mu, wN])
    return ordinateSet


def gaussLegQuadSet(sNords):
    """
    Input number of ordinates. (should be even number)
    Compute symetric gauss-leg quadrature set
    """
    legWeights = np.zeros(sNords + 1)
    pnp1s = np.zeros(sNords)
    pprimes = np.zeros(sNords)
    legWeights[sNords] = 1.
    mus = np.polynomial.legendre.legroots(legWeights)
    for i, mu in enumerate(mus):
        pprimes[i] = spc.lpn(sNords, mu)[1][-1]
        pnp1s[i] = spc.lpn(sNords + 1, mu)[0][-1]
    weights = -2. / ((sNords + 1) * pnp1s * pprimes)
    #
    ordinateSet = np.array([mus[::-1], weights])
    return ordinateSet


if __name__ == "__main__":
    print(gaussLegQuadSet(8))
