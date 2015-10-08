import numpy as np
import sys


class Cell1DSn(object):
    """
    sN ordinates (sNords) dont have to be evenly distributed in mu-space.  can
    be specified to be biased to one particular direction, for instance, to
    represent a collumated beam more accurately.
    # define canned quadrature sets
    S2 Quadrature figure for example:

          (1) |  (2)
            \ | /
    mu=-1 ----------mu=1 (axis of sym)
    mu=cos(theta)
    in S2, bin by 90deg chunks
    """
    # STANDARD ORDINATES AND FLUX WEIGHTS FOR STD QUADRATURE SET
    sNwDict = {2: np.array([1.0, 1.0]),
               4: np.array([0.34785, 0.65214, 0.65214, 0.34785])
               }
    sNmuDict = {2: np.array([0.5773502691, -0.5773502691]),
                4: np.array([0.86113, 0.33998, -0.33998, -0.86113])
                }

    def __init__(self, xpos, deltaX, nGroups=10, legOrder=8, sNords=2, **kwargs):
        self.faceNormals = np.array([-1, 1])
        self.centroid = xpos
        self.deltaX = deltaX
        # store cell centered, and cell edge fluxes.  Store as
        # len(groups)x3xlen(sNords) matrix.
        self.sNords = sNords      # number of discrete dirs tracked
        self.wN = self.sNwDict[sNords]     # quadrature weights
        self.sNmu = self.sNmuDict[sNords]  # direction cosines
        self.maxLegOrder = legOrder  # remember to range(maxLegORder + 1)
        self.nG = nGroups         # number of energy groups
        #
        # ord flux vec: 0 is cell centered, 1 is left, 2 is right face
        self.ordFlux = np.ones((nGroups, 3, self.sNords))
        self.totOrdFlux = np.zeros((nGroups, 3, self.sNords))
        self.qin = np.ones((nGroups, 3, self.sNords))  # scatter/fission source computed by scattering source iteration
        # fixed volumetric source
        self.S = kwargs.pop('source', np.zeros((nGroups, 3, self.sNords)))
        if self.S == 'fission':
            self.multiplying = True
        else:
            self.multiplying = False
        # Cell can have 3 types of Bcs:
        #   - Fixed face flux
        #       - vaccume (incomming flux 0)
        #       - fixed flux (incomming flux = const)
        #   - reflecting flux
        #   - white
        self.boundaryCond = kwargs.pop('bc', None)  # none denotes interior cell

    def resetTotOrdFlux(self):
        self.totOrdFlux = np.zeros((self.nGroups, 3, self.sNords))

    def sweepOrd(self, skernel, chiNuFission, keff=1.0, depth=0):
        """
        Use the scattering source iteration to sweep through sN discrete balance
        equations, one for each sN ordinate direction.

        Scattering source iteration:
        l = 0:
            [Omega'.grad + sigma_t] * qflux^(m)_l = fixed_source + (fission_src?)
        when l>0:
            [Omega'.grad + sigma_t] * qflux^(m)_l =
            sum(l, sigma_s_l(r, Omega.Omega') * qflux^(m-1)_l)
        where m is the scattering souce iteration index
        l is the direction ordinate

        Note that the RHS does not require any space deriviatives to be computed.
        sum the dummy qfluxes to obtain the true flux.
        totOrdflux = sum(m, qflux^(m))

        As m-> inf.  fewer and fewer neutrons will be around to contribute to the
        mth scattering source.  qflux^(m) should tend to 0 at large m.

        :Parameters:
            - :param arg1: descrition
            - :type arg1: type
            - :return: return desctipt
            - :rtype: return type
        """
        if depth >= 1:
            for g in range(self.nG):
                # obtain group scattering sources
                self.qin[g, 0, :] = self._evalScatterSource(g, skernel)
        elif self.multiplying and depth == 0:
            for g in range(self.nG):
                # compute gth group fission source
                self.qin = self._computeFissionSource(g, chiNuFission, keff)
            self.qin = self.qin + self.S
        elif not self.multiplying and depth == 0:
            self.qin = self.S
        return self.qin

    def _computeFissionSource(self, g, chiNuFission, keff):
        if self.multiplying:
            # multiplying medium source
            # note fission source is isotripic so each ordinate fission source
            # flux is equivillent
            return (1 / keff) * chiNuFission[g] * self._evalScalarFlux(g)
        else:
            # need fixed source from user input
            print("Fission source requested for Non multiplying medium.  FATALITY")
            sys.exit()

    def _evalScatterSource(self, g, skernel):
        """
        computes within group scattering sources:
            sigma_s(x, Omega.Omega')*flux_n(r, omega)
            where n is the group
        returns vector of scattered ordinate fluxes
        """
        return self._evalLegSource(g, skernel)

    def _evalLegSource(self, g, skernel):
        """
        compute sum_l((2l+1) * P_l(mu) * sigma_l * flux_l)
        where l is the legendre order
        returns a vecotr of length = len(mu)  (number of ordinate dirs)
        Amazingly, legendre.legval function provides exactly this capability
        """
        weights = np.zeros(self.maxLegOrder)
        for l in range(self.maxLegOrder + 1):
            weights[l] = (2 * l + 1) * skernel[l] * self._evalLegFlux(g, l)
        return np.polynomial.legendre.legval(self.sNmu, weights)

    def _evalScalarFlux(self, g, pos=0):
        """
        group scalar flux evaluator
        scalar_flux_g = (1/2) * sum_n(w_n * flux_n)
        n is the ordinate iterate
        """
        scalarFlux = np.sum(self.sNw * self.ordFlux[g, pos, :], axis=2)
        return (1 / 2.) * scalarFlux

    def _evalLegFlux(self, g, l, pos=0):
        """
        group legendre group flux
        scalar_flux_lg = (1/2) * sum_n(w_n * P_l * flux_n)
        where l is the legendre order
        and n is the ordinate iterate
        """
        legweights = np.zeros(self.maxLegOrder)
        legweights[l] = 1.0
        legsum = np.sum(np.polynomial.legendre.legval(self.sNmu, legweights) *
                        self.sNw * self.ordFlux[g, pos, :])
        return (1 / 2.) * legsum

    def sweepEnergy(self, oi):
        pass

    def solveEnergy(self, H, qin, oi):
        """
        Instead of performing down/up-scatter sweeps:
        solve simple ngrp x ngrp Ax = b problem.

        Multigroup space and direction independent transport operator:
        H = Ntotal - Leg_skernel_in_direction_oi

        source term:
            For fission:
                qin = (1/k) * F * flux + FixedSource
                F = chi.T * nuFission

        Solves for all group fluxes in one ordinate direction.

        :Parameters:
            - :param arg1: descrition
            - :type arg1: type
            - :return: return desctipt
            - :rtype: return type
        """
        self.ordFlux[:, 0, oi] = np.linalg.solve(H, qin)


class Sn1Dbc(object):
    def __init__(self, cell, face, bcType, depth=0):
        if bcType == 'vac':
            self.applyVacBC(cell, face)
        else:
            pass

    def applyRefBC(cell, face):
        # reflects cell outgoing flux at boundary to incomming flux
        # ex: flux_2 == flux_1
        # faceDot = cell.sNmu * cell.faceNormals[face - 1]
        # look for equal magnitude but opposite sign when assigning direction
        # pairs
        directionPairs = []
        for inwardDir, outwardDir in directionPairs:
            cell.ordFlux[:, face, inwardDir] = cell.ordFlux[:, face, outwardDir]

    def applyVacBC(cell, face):
        # sets incomming flux to zero on designated face
        faceDot = cell.sNmu * cell.faceNormals[face - 1]
        inwardDirs = np.where(faceDot < 0)
        cell.ordFlux[:, face, inwardDirs] = 0.0

    def applyFixedFluxBC():
        # only applied to un-collieded flux iter: depth=0, vac for all
        # others.
        # sets incomming flux to user specified val
        pass

    def applyWhiteBC():
        # Summs all out going flux and redistributes evenly over all inward
        # facing directions.
        pass
