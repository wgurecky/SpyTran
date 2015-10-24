# Ordinate set parser
#

import numpy as np
import scipy.special as spc


def readOrdFile(inFile, sNords):
    ords = np.fromtxt(inFile)
    if len(ords) == sNords:
        ordinateSet = ords
    else:
        pass
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
