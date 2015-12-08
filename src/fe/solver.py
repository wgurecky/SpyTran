import numpy as np
import time
import utils.hdf5dump as h5d
from utils.ordReader import gaussLegQuadSet
from utils.ordReader import createLegArray
from utils.gmshPreproc import gmsh1DMesh
from mesh import SuperMesh
np.set_printoptions(linewidth=200)  # set print to screen opts


class SnFe1D(object):
    """
    High level solver tasks reside here. e.g:
        - Make transport operator (matirx A)
        - Make RHS vecotr (vector b)
        - solve flux (solve Ax=b)
        - Update scattering souce in each element
    Methods can be called when necissary by a controller script.
    """
    def __init__(self, geoFile, materialDict, bcDict, srcDict, nGroups=10,
                 legOrder=8, sNords=2):
        """
        Matierial dict in:
            {'material_str': material_class_instance, ...}
        format
        """
        quadSet = gaussLegQuadSet(sNords)                       # quadrature set
        self.sNords = sNords                                    # number of discrete dirs tracked
        self.sNmu, self.wN = quadSet[0], quadSet[1]             # quadrature weights
        self.maxLegOrder = legOrder                             # remember to range(maxLegORder + 1)
        self.nG = nGroups                                       # number of energy groups
        self.legArray = createLegArray(self.sNmu, self.maxLegOrder)  # Stores leg polys
        #
        gmshMesh = gmsh1DMesh(geoFile=geoFile)  # Run gmsh
        self.nodes = gmshMesh.nodes
        self.superMesh = SuperMesh(gmshMesh, materialDict, bcDict, srcDict, nGroups, sNords)    # build the mesh
        self.depth = 0  # scattering source iteration depth
        self.keff = 1
        self.buildTransOp()
        self.buildRHS()
        self.applyBCs()
        self.timeScatter, self.timeLinSolver = 0, 0

    def scatterSource(self):
        """
        Perform scattering souce iteration for all nodes in the mesh:
        for region in regions:
            for elements in region:
                element.scatter()
        """
        timeStart = time.time()
        self.superMesh.scatter(self.depth, self.keff)
        if self.depth == 1:
            # rebuild transport Op if scattering depth == 1
            self.buildTransOp()
        self.buildRHS()  # build RHS after scatter
        self.applyBCs()  # apply BCs after scatter
        self.depth += 1
        self.timeScatter = (time.time() - timeStart)

    def buildTransOp(self):
        """
        Construct transport operator, A.
        Note A is not the complete transport operator, it only moves neutrons through space,
        not in energy or angle.  The scattering souce iteration takes care of energy
        and angle redistribution.
        """
        self.superMesh.buildSysMatrix(self.depth)

    def buildRHS(self):
        self.superMesh.buildSysRHS()

    def applyBCs(self):
        self.superMesh.applyBCs(self.depth)

    def solveFlux(self, tol=1.0e-6):
        """
        Solve Ax=b.
        Returns flux norm
        """
        timeStart = time.time()
        self.norm, resid = self.superMesh.sweepFlux(tol)
        self.timeLinSolver = (time.time() - timeStart)
        return self.norm, (self.timeScatter, self.timeLinSolver)

    def writeData(self, outFileName='1Dfeout.h5'):
        """
        Write solution state to hdf5 file.
            - keff (if applicable)
            - mesh
                - elements (nodes in element)
                - node positions
            - flux field
        """
        # write [[nodeID, nodeX, nodeY, nodeZ],...] vector  (this is gmshMesh.nodes)
        # write [[nodeID, fluxValue]...] vector  (this is the totFluxField)
        # write eigenvalue
        h5data = {'nodes': self.nodes, 'ordFluxes': self.superMesh.totFluxField,
                  'keff': self.keff, 'fluxNorm': self.norm,
                  'mu': self.sNmu, 'weights': self.wN,
                  'nGrp': self.nG, 'scrIters': self.depth}
        h5d.writeToHdf5(h5data, outFileName)
