#!/usr/bin/python

# 1D specific Sn transport classes and methods
#
# Sweeps through energy, direction, and space on a 1D mesh
#
# Class hiarchy:
#
# - Domain
#   - sub-domain
#       - mesh-material region
#           - cells
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
        self.sNorder = kwargs.pop("sNorder", 2)
        self._setOrds()
        self.deltaX = deltaX
        # initilize all cells in the mesh.
        self.cells = []
        for i, pos in enumerate(np.arange(bounds[0], bounds[1] + deltaX, deltaX)):
            self.cellPos[i] = pos
            self.cells.append(Cell1DSn())

        self.totalXs = material.macroProp['Ntotal']
        self.skernel = material.macroProp['Nskernel']
        self.qin = []  # scattering source
        pass

    def _setOrds(self):
        if self.sNorder == 2:
            self.mu = np.array([-1., 1.])
        else:
            pass

    def buildCells(self):
        # save nearest neighbor information in preperation to generalize to 2D,
        # unstructured meshes.
        pass

    def sweepMesh(self, ordi):
        """
        March through the cells in the mesh, update the ordinate fluxes as we go
        Specify ordinate direction to travel in.  Go in +mu dir, then back
        in the -mu direction (if necissary) untill convergence
        """
        converged, i = False, 0
        while not converged:
            self._sweepDir(1)
            self._sweepDir(2)
            i += 1
            if i > 3:
                converged = True

    def _sweepDir(self, o):
        """
        o is either 1 or 2 in 1D
        1 is left cell edge,  2 is right edge
        """
        # note len(self.ordFlux) == len(self.mu)
        for cell in self.cells:
            cell.ordFlux[:, 0, :] = (cell.ordFlux[:, o, :] + self.deltaX * self.qin / (2. * np.abs(self.mu))) / \
                (1. + self.totalXs * self.deltaX / (2. * np.abs(self.mu)))
            if o == 1:
                cell.ordFlux[:, 2, :] = 2. * cell.ordFlux[:, 0, :] - cell.ordFlux[:, o, :]
            if o == 2:
                cell.ordFlux[:, 1, :] = 2. * cell.ordFlux[:, 0, :] - cell.ordFlux[:, o, :]


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
    sN2w = np.array([1.0, 1.0])
    sN2mu = np.array([0.5773502691, -0.5773502691])
    sN4w = np.array([0.65214, 0.34785, 0.34785, 0.65214])
    sN4mu = np.array([0.33998, 0.86113, -0.86113, -0.33998])

    def __init__(self, nGroups, legOrder, sNords, **kwargs):
        # store cell centered, and cell edge fluxes.  Store as
        # len(groups)x3xlen(sNords) matrix.
        self.sNords = sNords
        self.legOrder = legOrder
        self.nG = nGroups
        # ord flux vec: 0 is cell centered, 1 is left, 2 is right face
        self.ordFlux = np.ones((nGroups, 3, len(self.sNords)))

    def _sweepOrd(self):
        # Transport equation in sNords directions
        for oi in ordinateDirs:
            pass

    def _evalScalarFlux(self, g, pos=0):
        return (1 / 2.) * np.sum(self.sNw * self.ordFlux[g, pos, :])

    def _evalLegMom(self, g, pos=0):
        legsum = 0
        for i in range(np.shape(self.ordFlux)[-1]):
            legsum += self._legval(self.mu[i], self.sNw[i], self.ordFlux[g, pos, i])
        return (1 / 2.) * legsum

    def _legval(self, mu, wN, oflux):
        """
        takes direction cosine,
        ordinate weight
        ordinate flux
        """
        return np.polynomial.legendre.legval(mu, wN * oflux)

    def _sweepEnergy(self, oi):
        pass

    def _solveEnergy(self, H, qin, oi):
        """
        Instead of performing down/up-scatter sweeps:
        solve simple ngrp x ngrp Ax = b problem.

        Multigroup space and direction independent transport operator:
        H = Ntotal - Nskernel

        source term:
        qin = (1/k) * F * flux + FixedSource
        F = chi.T * nuFission
        """
        self.ordFlux[:, 0, oi] = np.linalg.solve(H, qin)
