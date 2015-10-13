import numpy as np
import scipy.special as spc
import sys
np.set_printoptions(linewidth=200)  # set print to screen opts


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
               4: np.array([0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451])
               }
    sNmuDict = {2: np.array([0.5773502691, -0.5773502691]),
                4: np.array([0.8611363115, 0.3399810435, -0.3399810435, -0.8611363115])
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
        self.legweights = np.zeros(legOrder + 1)
        self.nG = nGroups         # number of energy groups
        #
        # ord flux vector: 0 is cell centered, 1 is left, 2 is right face
        iguess = kwargs.pop('iFlux', np.ones((nGroups, 3, self.sNords)))
        self.ordFlux = iguess
        self.totOrdFlux = iguess
        self.qin = np.ones((nGroups, 3, self.sNords))  # scatter/fission source computed by scattering source iteration
        # fixed volumetric source
        self.S = kwargs.pop('source', np.zeros((nGroups, 3, self.sNords)))
        if type(self.S) is str:
            if self.S == 'fission':
                self.multiplying = True
            else:
                self.S = np.zeros((nGroups, 3, self.sNords))
                self.multiplying = False
        # set bc, if any given
        self.boundaryCond = kwargs.pop('bc', None)  # none denotes interior cell

    def setBC(self, bc):
        """
        Assign boundary condition instance to cell.
        """
        self.boundaryCond = Sn1Dbc(bc)

    def applyBC(self, depth):
        """
        Enforce the boundary condition in cell.  Compute/adjust the cell face
        fluxes according to the boundary condition.
        """
        if self.boundaryCond is not None:
            self.boundaryCond.applyBC(self, depth)
            return True
        else:
            print("WARNING: You are trying to apply a boundary condition in an interior cell.")
            return False

    def resetTotOrdFlux(self):
        self.totOrdFlux = np.zeros((self.nG, 3, self.sNords))

    def resetOrdFlux(self):
        self.ordFlux = np.zeros((self.nG, 3, self.sNords))

    def sweepOrd(self, skernel, chiNuFission, keff=1.0, depth=0):
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
            for g in range(self.nG):
                # obtain group scattering sources
                self.qin[g, 0, :] = self._evalScatterSource(g, skernel)
        elif self.multiplying and depth == 0:
            for g in range(self.nG):
                # compute gth group fission source
                self.qin[g, 0, :] = self._computeFissionSource(g, chiNuFission, keff)
            self.resetTotOrdFlux()
        elif not self.multiplying and depth == 0:
            self.qin = self.S
            self.resetTotOrdFlux()
        return self.qin

    def _computeFissionSource(self, g, chiNuFission, keff):
        """
        Compute the withen group fission source.
        chiNuFission[g] is a row vector corresponding to all g'
        """
        if self.multiplying:
            # multiplying medium source
            # note fission source is isotripic so each ordinate fission source
            # flux is equivillent
            return (1 / keff / 2.0) * np.abs(self.wN) * \
                np.sum(chiNuFission[g] * self._evalTotScalarFlux(g))
            #return (1 / keff / 1.0) * \
            #    np.sum(chiNuFission[g] * self._evalTotScalarFlux(g))
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
        # remove diagonal entries from skernal.  We do not care about
        # g == g' scatter (within grp scatter).
        # skMultiplier = np.ones((self.nG, self.nG))
        # skMultiplier = np.ones((self.nG, self.nG)) - np.eye(self.nG)
        return self._evalLegSource(g, skernel)

    def _evalLegSource(self, g, skernel):
        """
        compute sum_l((2l+1) * P_l(mu) * sigma_l * flux_l)
        where l is the legendre order
        returns a vecotr of length = len(mu)  (number of ordinate dirs)
        Amazingly, legendre.legval function provides exactly this capability
        """
        def ggprimeInScatter(g, l):
            """
            Computes in-scattring into grp g reaction rate.
            """
            return np.sum(skernel[l, g, :] * self._evalVecLegFlux(l))
        #
        weights = np.zeros(self.maxLegOrder + 1)
        for l in range(self.maxLegOrder + 1):
            weights[l] = (2 * l + 1) * ggprimeInScatter(g, l)
        return (1 / 2.) * np.polynomial.legendre.legval(self.sNmu, weights)

    def _evalScalarFlux(self, g, pos=0):
        """
        group scalar flux evaluator
        scalar_flux_g = (1/2) * sum_n(w_n * flux_n)
        n is the ordinate iterate
        """
        scalarFlux = np.sum(self.wN * self.ordFlux[g, pos, :])
        return (1 / 2.) * scalarFlux

    def sumOrdFlux(self):
        self.totOrdFlux += self.ordFlux

    def _evalTotScalarFlux(self, g, pos=0):
        """
        group scalar flux evaluator
        scalar_flux_g = (1/2) * sum_n(w_n * flux_n)
        n is the ordinate iterate
        """
        scalarFlux = np.sum(self.wN * self.totOrdFlux[g, pos, :])
        return (1 / 2.) * scalarFlux

    def getTotScalarFlux(self, pos=0):
        scalarFlux = []
        for g in range(self.nG):
            scalarFlux.append(self._evalTotScalarFlux(g, pos))
        return np.array(scalarFlux)

    def _evalLegFlux(self, g, l, pos=0):
        """
        group legendre group flux
        scalar_flux_lg = (1/2) * sum_n(w_n * P_l * flux_n)
        where l is the legendre order
        and n is the ordinate iterate
        """
        legweights = np.zeros(self.maxLegOrder + 1)
        legweights[l] = 1.0
        legsum = np.sum(np.polynomial.legendre.legval(self.sNmu, legweights) *
                        self.wN * self.ordFlux[g, pos, :])
        return (1 / 2.) * legsum

    def _evalVecLegFlux(self, l, pos=0):
        """
        Vectorized version of legendre moment of flux routine (must faster)

        group legendre group flux
        scalar_flux_lg = (1/2) * sum_n(w_n * P_l * flux_n)
        where l is the legendre order
        and n is the ordinate iterate
        """
        #self.legweights *= 0.0
        #self.legweights[l] = 1.0
        #leg = sp.special.legendre(l)
        #legsum = np.sum(np.polynomial.legendre.legval(self.sNmu, self.legweights[:l+1]) *
        #                self.wN * (self.ordFlux[:, pos, :]), axis=1)
        legsum = np.sum(spc.eval_legendre(l, self.sNmu) *
                        self.wN * (self.ordFlux[:, pos, :]), axis=1)
        return legsum

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
    def __init__(self, bc):
        self.vacBC = bc.pop('vac', None)
        self.refBC = bc.pop('ref', None)
        self.whiteBC = bc.pop('white', None)

    def applyBC(self, cell, depth):
        if self.vacBC is not None:
            try:
                face = self.vacBC[0]
            except:
                face = self.vacBC
            self.applyVacBC(cell, face)
        elif self.refBC is not None:
            try:
                face = self.refBC[0]
            except:
                face = self.refBC
            self.applyRefBC(cell, face)
        else:
            pass

    def applyRefBC(self, cell, face):
        # reflects cell outgoing flux at boundary to incomming flux
        # ex: flux_2 == flux_1
        # look for equal magnitude but opposite sign when assigning direction
        # pairs
        # directionPairs = []
        faceDot = cell.sNmu * cell.faceNormals[face - 1]
        # when dot product is (-) direction is oposite of outward normal to face
        # when dot product is (+) direction is same dir as outward normal to face
        # "same direction as face" == "pointing out of domain" ONLY IF
        # BCs are applied at the START of the space-sweep!!! YUCK.  TODO:
        # Simplify BC assignment.  way to garly right now.
        # TODO: does not generalize to arbitary ordinate dirs.  If asymetric
        # Sn is performed we need another, better method for reflection...
        inDirs = np.where(faceDot < 0.)[0]
        #outDirs = np.where(faceDot > 0.)
        #uniqueOctantOrds = np.unique(np.abs(faceDot))
        #for uniqueOrd in uniqueOctantOrds:
        #    directionPairs.append(np.where(np.abs(faceDot) == uniqueOrd))
        for iDir in inDirs:
            # get negative mu in iDir
            negDir = -1 * cell.sNmu[iDir]
            outDir = np.where(negDir == cell.sNmu)
            cell.ordFlux[:, face, iDir] = cell.ordFlux[:, face, outDir[0][0]]

    def applyVacBC(self, cell, face):
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
