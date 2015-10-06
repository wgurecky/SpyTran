#!/usr/bin/python

# 1D specific Sn transport classes and methods
#
# Sweeps through energy, direction, and space on a 1D mesh
#
#

import numpy as np
import materials.materialMixxer as mx


class Domain(object):
    """
    Top level class that holds all the problem specifications.  May be initilized
    from an input deck that is written in yml format that is translated into
    a dict and fed into the domain class to initilize the problem.
    """
    def __init__(self, inputDict):
        # specify sub-domain extents.  For serial implementation, one sub-domain
        # is sufficient as all mesh-material regions will belong to the same CPU
        self.subDomains
        self.dim = inputDict.pop("dim", 1)
        #
        # specify mesh-material region extents and materials
        # as key val pairs
        defaultMat = mx.mixedMat({'u235': 0.1, 'h1': 0.6, 'o16': 0.3})
        defaultMat.setDensity(1.0)
        self.matRegions.pop("matRegions", {'glob1': ([0, 10], 0.01, defaultMat)})
        #
        # specify sN order.  Provide pre-computed quadrature, allow user defined
        # quadrature sets as well.
        self.sNord = inputDict.pop("sNorder", 2)  # S2 by default
        #
        # specify material xsdir (xs folder), default to default mat dir
        self.xsdir = inputDict.pop("xsdir", "./materials/hw2")

    def solveTransport(self):
        # call inner sweeps on all subdomains.
        pass


class MultipyingDomain(Domain):
    def __init__(self, inputDict):
        super(Domain, self).__init__(inputDict)

    def solveKeigen(self):
        # perform power iteration using the latest flux estimate.
        pass


class SubDomain(object):
    """
    Inner iterations (i.e. sweeps through energy, angle, and space) are performed
    on each subdomain.  Boundary conditions are specified on each subdomain.

    In the case of multi-core: a special subdomain-subdomain boundary condition
    must be specified to pass information back and forth across subdomains after
    each inner iteration is performed
    """
    def __init__(self, mat, bounds):
        pass

    def updateFlux(self):
        # perform inner sweeps on subdomain
        pass


class Mesh1Dsn(object):
    def __init__(self, bounds, deltaX, material, **kwargs):
        # sNorder = kwargs.pop("sNorder", 2)
        self.deltaX = deltaX
        self.depth = 0  # scattering source iteration
        # initilize all cells in the mesh.
        self.cells, self.cellXPos = [], []
        for i, pos in enumerate(np.arange(bounds[0], bounds[1] + deltaX, deltaX)):
            self.cellXPos.append(pos)
            self.cells.append(Cell1DSn())
        self.cellXPos = np.array(self.cellXPos)
        #
        self.totalXs = material.macroProp['Ntotal']
        self.skernel = material.macroProp['Nskernel']

    def buildCells(self):
        # save nearest neighbor information in preperation to generalize to 2D,
        # unstructured meshes.
        pass

    def makeSweepPattern(self):
        """
        returns a list of cell ids
        list ordered by sweep direction.
          in 1D returns 1D array

        ex:
            ------> sweep dir
        cellIDs = [2, 4, 1, 3, 5, 6]
        cellXPos = [1, 2, 3, 4, 5, 6]
        order by X dist from min(cellXPos).  Need to use dist formula in 2 and 3D.
        return = [2, 4, 1, 3, 5, 6]
        """
        pass

    def sweepMesh(self, bcLeft, bcRight):
        """
        Outer space iterations:
            March through the cells in the mesh, update the ordinate fluxes as we go
            Specify ordinate direction to travel in.  Go in +mu dir, then back
            in the -mu direction (if necissary) untill convergence.

        Inner sN - energy iterations:
            perform scattering source inner iterations on each cell untill convergence.
            distribute neutrons in angle and energy.
        """
        # Sweep space
        converged, i = False, 0
        while not converged:
            self._sweepDir(1, bcLeft)
            self._sweepDir(2, bcRight)
            i += 1
            if i > 3:
                converged = True
        self.depth += 1  # increment scattering source iter
        self._addOrdFlux()

    def _addOrdFlux(self):
        '''
        add Nth collided ord flux to running total
        '''
        for cell in self.cells:
            cell.totOrdFlux += cell.ordFlux

    def _sweepDir(self, f, bc):
        """
        f is either 1 or 2 in 1D
        f stands for face
        1 is left cell face,  2 is right face
        """
        lastCellFaceVal = bc  # set boundary condition on flux at edge
        if f == 2:
            cellList = reversed(self.cells)
        else:
            cellList = self.cells
        for cell in cellList:
            if cell.boundaryCond is not None:
                # Set cell boundary condition
                pass
            else:
                # Interior cell
                cell.ordFlux[:, f, :] = lastCellFaceVal
            # Sweep angle and energy within the cell to update qin (inscatter source)
            cell.sweepOrd(self.depth)
            # Step through space
            cell.ordFlux[:, 0, :] = (cell.ordFlux[:, f, :] + self.deltaX * cell.qin / (2. * np.abs(self.mu))) / \
                (1. + self.totalXs * self.deltaX / (2. * np.abs(self.mu)))
            if f == 1:
                cell.ordFlux[:, 2, :] = 2. * cell.ordFlux[:, 0, :] - cell.ordFlux[:, f, :]
                lastCellFaceVal = cell.ordFlux[:, 2, :]
            if f == 2:
                cell.ordFlux[:, 1, :] = 2. * cell.ordFlux[:, 0, :] - cell.ordFlux[:, f, :]
                lastCellFaceVal = cell.ordFlux[:, 1, :]


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

    def __init__(self, xpos, nGroups=10, legOrder=8, sNords=2, **kwargs):
        self.faceNormals = np.array([-1, 1])
        self.centroid = xpos
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
        # fixed volumetric source
        self.S = kwargs.pop('source', np.zeros((nGroups, 3, self.sNords)))

    def sweepOrd(self, depth=0):
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
        # 0th scattering soucre is given (fission or non-mult src term):
        sIn = np.zeros(np.shape(self.ordFlux))
        if depth >= 1:
            for oi in range(self.sNords):
                scatteringSourceVec = self._evalScatterSource(oi)
                sIn += np.sum(scatteringSourceVec)
        elif self.multiplying and depth == 0:
            sIn = self._computeFissionSource()
        elif not self.multiplying and depth == 0:
            sIn = self.S
        self.qin = sIn
        return sIn

    def _computeFissionSource(self, nuFission, keff):
        if self.multiplying:
            # multiplying medium source
            return (1 / keff) * np.sum(nuFission * self.ordFlux, axis=0)
            pass
        else:
            # need fixed source from user input
            pass

    def _evalScatterSource(self, oi):
        """
        computes scattering source:
            sigma_s(x, Omega.Omega')*flux_n(r, omega)
            where n is the group
        returns vector of scattered fluxes (flux after scattering op
        has acted upon it).
        """
        self.legOrder
        pass

    def _evalScalarFlux(self, g, pos=0):
        """
        group scalar flux evaluator
        scalar_flux_g
        """
        return (1 / 2.) * np.sum(self.sNw * self.ordFlux[g, pos, :])

    def _evalLegFlux(self, g, pos=0):
        """
        group legendre group flux
        scalar_flux_lg
        """
        legsum = 0
        for i in range(np.shape(self.ordFlux)[-1]):
            legsum += self._legval(self.mu[i], self.sNw[i] * self.ordFlux[g, pos, i])
        return (1 / 2.) * legsum

    def _legval(self, mu, wN, oflux):
        """
        :Parameters:
            - :param mu: len N direction cosine vector
            - :type mu: nparray
            - :param oflux: ordinate flux vector
            - :type oflux: nparray
            - :param wN: ordinate quadrature weight
            - :type wN: nparray
            - :return: len N vec. legendre poly evaluated at input mu vec, scaled by input weights
            - :rtype: return nparray
        """
        return np.polynomial.legendre.legval(mu, wN * oflux)

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
        faceDot = cell.sNmu * cell.faceNormals[face - 1]
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
