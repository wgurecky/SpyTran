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


class Mesh1Dsn(object):
    def __init__(self, bounds, deltaX, material, **kwargs):
        # initilize all cells in the mesh.
        pass

    def buildCells(self):
        # save nearest neighbor information in preperation to generalize to 2D,
        # unstructured meshes.
        pass


class Cell(object):

    def __init__(self, legOrder, sNords, **kwargs):
        pass


class Cell1DSn(Cell):
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
    sN4w = np.array([0.65214, 0.34785, 0.34785, 0.65214])

    def __init__(self, nGroups, legOrder, sNords, **kwargs):
        super(Cell, self).__init__(legOrder, sNords, **kwargs)
        # store cell centered, and cell edge fluxes.  Store as
        # len(groups)x3xlen(sNords) matrix.
        self.ordFlux = np.ones((nGroups, 3, len(self.sNords)))
        pass


class Cell1DPn(Cell):
    def __init__(self):
        pass
