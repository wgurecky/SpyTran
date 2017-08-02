import numpy as np
import warnings
import time
import spytran.utils.hdf5dump as h5d
from spytran.utils.ordReader import gaussLegQuadSet
from spytran.utils.ordReader import D2quadSet
from spytran.utils.gmshPreproc import gmsh1DMesh
from spytran.utils.gmshPreproc import gmsh2DMesh
from mesh import SuperMesh
np.set_printoptions(linewidth=200)  # set print to screen opts
warnings.filterwarnings("ignore")


class SnFeSlv(object):
    """
    High level solver tasks reside here. e.g:
        - Make transport operator (matirx A)
        - Make RHS vecotr (vector b)
        - solve flux (solve Ax=b)
        - Update scattering souce in each element
    Methods can be called when necissary by a controller script.
    """
    def __init__(self, geoFile, materialDict, bcDict, srcDict, nGroups=10,
                 legOrder=8, sN=4, dim=1):
        """
        Matierial dict in:
            {'material_str': material_class_instance, ...}
        format
        """
        if dim == 1:
            quadSet = gaussLegQuadSet(sN)                      # quadrature set
            self.sNords = sN                                   # number of discrete dirs tracked
            self.wN = quadSet[1]
        elif dim == 2:
            quadSet = D2quadSet(sN)
            self.sNords, self.wN = quadSet.sNords, quadSet.wN
        self.maxLegOrder = legOrder                             # remember to range(maxLegORder + 1)
        self.nG = nGroups                                       # number of energy groups
        #
        if dim == 1:
            gmshMesh = gmsh1DMesh(geoFile=geoFile)  # Run gmsh
        elif dim == 2:
            gmshMesh = gmsh2DMesh(geoFile=geoFile)  # Run gmsh
        gmshMesh.enable_connectivity()
        self.nodes = gmshMesh.nodes
        self.superMesh = SuperMesh(gmshMesh, materialDict, bcDict, srcDict,
                                   nGroups, self.sNords, quadSet, dim)    # build the mesh
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

    def _initkEig(self, sFactor=1.0):
        if not hasattr(self, 'fissionSrc'):
            print("Init Keff: " + str(self.keff))
            self.fissionSrc = []
            self.superMesh.initFlux(sFactor)  # scaling factor
            self.fissionSrc.append(self.superMesh.getFissionSrc())

    def kEig(self, rTol=1e-6, kTol=1e-3, finalIter=False, verbosity=1):
        """
        Perform a single k-eigen update.  If k is stationary, return true for kconverged
        """
        self._initkEig()
        for i in range(150):
            # perform scattering src iterations until flux tol falls below spcified rtol
            self.scatterSource()
            self.solveFlux()
            if verbosity == 1 and i % 1 == 0:
                print("{0: <3}".format(str(i)) + "        " + "{:.4e}".format(self.norm) + "              " +
                      "{:.2e}".format(self.timeLinSolver) + "           " +
                      "{:.2e}".format(self.timeScatter))
            if self.norm <= rTol:
                break
        self.depth = 0
        # update keff
        self.fissionSrc.append(self.superMesh.getFissionSrc())
        kold = self.keff
        self.keff = self.keff * (self.fissionSrc[-1] / self.fissionSrc[-2])
        if np.abs(kold - self.keff) < kTol:
            kconv = True
        else:
            if finalIter is False:
                self.superMesh.resetMeshFlux()
            kconv = False
        return self.keff, kconv, self.norm

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
                  'keff': self.keff, 'fluxNorm': self.norm, 'weights': self.wN,
                  'nGrp': self.nG, 'scrIters': self.depth}
        h5d.writeToHdf5(h5data, outFileName)
