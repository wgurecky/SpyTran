#!/usr/bin/python

# 1D specific Sn transport classes and methods
#
# Sweeps through energy, direction, and space on a 1D mesh
#
#

import numpy as np
import materials.materialMixxer as mx
import sn1Dcell as snc1d


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
        self.cells = []
        for i, pos in enumerate(np.arange(bounds[0], bounds[1] + deltaX, deltaX)):
            self.cells.append(snc1d.Cell1DSn(pos, deltaX))
        #
        self.totalXs = material.macroProp['Ntotal']
        self.skernel = material.macroProp['Nskernel']
        if 'chi' in material.macroProp.keys():
            self.chiNuFission = np.dot(np.array([material.macroProp['chi']]).T,
                                       np.array([material.macroProp['Nnufission']]))
        else:
            self.chiNuFission = None

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
            if i > 5:
                # max number of space-angle sweeps to perform
                converged = True
        self.depth += 1  # increment scattering source iter
        self._addOrdFlux()

    def _addOrdFlux(self):
        '''
        add Nth collided ord flux to running total
        '''
        for cell in self.cells:
            cell.totOrdFlux += cell.ordFlux

    def resetFlux(self):
        """
        In k-eigenvalue problems.  after each outer iteration we have to
        reset the ordinate fluxes
        """
        for cell in self.cells:
            cell.resetTotOrdFlux()

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
            # Sweep angle within the cell to update qin (inscatter source)
            cell.sweepOrd(self.skernel, self.chiNuFisison, self.keff, self.depth)
            # Step through space
            cell.ordFlux[:, 0, :] = (cell.ordFlux[:, f, :] + self.deltaX * cell.qin / (2. * np.abs(self.mu))) / \
                (1. + self.totalXs * self.deltaX / (2. * np.abs(self.mu)))
            if f == 1:
                cell.ordFlux[:, 2, :] = 2. * cell.ordFlux[:, 0, :] - cell.ordFlux[:, f, :]
                lastCellFaceVal = cell.ordFlux[:, 2, :]
            if f == 2:
                cell.ordFlux[:, 1, :] = 2. * cell.ordFlux[:, 0, :] - cell.ordFlux[:, f, :]
                lastCellFaceVal = cell.ordFlux[:, 1, :]
