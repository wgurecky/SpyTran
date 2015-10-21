# Use setup.py script to compile
#
# Use as traditional python module after setup.py is run.
# ex:
#   import scattSrc as scs

import numpy as np
cimport numpy as np
import scipy.special as spc


def evalScatterSource(cell, int g, np.ndarray skernel):
    """
    computes within group scattering sources:
        sigma_s(x, Omega.Omega')*flux_n(r, omega)
        where n is the group
    returns vector of scattered ordinate fluxes

    parameter : cell
        type : Cell1DSn object
    parameter : g
        type : int
        description : neutron group number
    parameter : skernel
        type : ndarray
        notes : neutron scattering kernel matrix
    """
    cdef int l
    cdef np.ndarray weights = np.zeros(cell.maxLegOrder + 1)
    def ggprimeInScatter(g, l):
        """
        Computes in-scattring into grp g reaction rate.
        """
        return np.sum(skernel[l, g, :] * evalVecLegFlux(cell, l))
    #
    for l in range(cell.maxLegOrder + 1):
        weights[l] = (2 * l + 1) * ggprimeInScatter(g, l)
    return np.polynomial.legendre.legval(cell.sNmu, weights)


cdef evalVecLegFlux(cell, int l):
    """
    Vectorized version of legendre moment of flux routine (must faster)
    """
    cdef np.ndarray legsum
    legsum = np.sum(spc.eval_legendre(l, cell.sNmu) *
                    cell.wN * (cell.ordFlux[:, 0, :]), axis=1)
    return 0.5 * legsum
