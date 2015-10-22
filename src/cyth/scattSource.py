import numpy as np
import scipy.special as spc


def evalScatterSource(cell, g, skernel):
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
    def ggprimeInScatter(g, l):
        """
        Computes in-scattring into grp g reaction rate.
        """
        return np.sum(skernel[l, g, :] * evalVecLegFlux(cell, l))
    #
    weights = np.array([np.zeros(cell.maxLegOrder + 1)])
    for l in range(cell.maxLegOrder + 1):
        weights[0][l] = (2 * l + 1) * ggprimeInScatter(g, l)
    #return np.polynomial.legendre.legval(cell.sNmu, weights)
    return np.sum(weights.T * cell.legArray, axis=0)


def evalVecLegFlux(cell, l):
    """
    Vectorized version of legendre moment of flux routine (must faster)
    """
    #legsum = np.sum(spc.eval_legendre(l, cell.sNmu) *
    #                cell.wN * (cell.ordFlux[:, 0, :]), axis=1)
    legsum = np.sum(cell.legArray[l, :] *
                    cell.wN * (cell.ordFlux[:, 0, :]), axis=1)
    return 0.5 * legsum


def sweepOrd(cell, skernel, chiNuFission, keff=1.0, depth=0, overRlx=1.0):
    """
    Use the scattering source iteration to sweep through sN discrete balance
    equations, one for each sN ordinate direction.
    Perform scattering source iteration

    Scattering source iteration:
    m = 0:
        [Omega'.grad + sigma_t] * qflux^(m)_g = fixed_source + fission_src
    when m>0:
        [Omega'.grad + sigma_t] * qflux^(m)_g =
        sum(o, sigma_s(r, g->g', Omega.Omega')_g * qflux^(m-1)_o)
    m is the scattering souce iteration index
    o is the direction ordinate
    g is the group index

    As m-> inf.  fewer and fewer neutrons will be around to contribute to the
    mth scattering source.  qflux^(m) should tend to 0 at large m.

    :Parameters:
        - :param arg1: descrition
        - :type arg1: type
        - :return: return desctipt
        - :rtype: return type
    """
    if depth >= 1:
        if depth >= 2:
            for g in range(cell.nG):
                cell.qin[g, 0, :] = overRlx * (evalScatterSource(cell, g, skernel) - cell.previousQin[g, 0, :]) + cell.previousQin[g, 0, :]
            cell.previousQin = cell.qin
        else:
            for g in range(cell.nG):
                cell.qin[g, 0, :] = evalScatterSource(cell, g, skernel)
            cell.previousQin = cell.qin
    elif cell.multiplying and depth == 0:
        for g in range(cell.nG):
            # compute gth group fission source
            cell.qin[g, 0, :] = cell._computeFissionSource(g, chiNuFission, keff)
        cell.resetTotOrdFlux()
    elif not cell.multiplying and depth == 0:
        cell.qin = cell.S
        cell.resetTotOrdFlux()
    return cell.qin
