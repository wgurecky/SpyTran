import numpy as np
#import scattSource as scs
from copy import deepcopy
# import scattSrc as scs
# import scipy.special as spc
import scipy.sparse as sps
import sys
from utils.ordReader import gaussLegQuadSet
np.set_printoptions(linewidth=200)  # set print to screen opts


#class FE1Dbc(object):
#    def __init__(self, **kwargs):
#        pass

class FE1DSnMesh(object):
    def __init__():
        pass

    def initFlux(self):
        # the RHS and flux scalar field are dense
        pass

    def constructA(self, g):
        """
        Construct the system matrix A for group g.  This must only be done once
        as the system matrix A is not dependent on the flux.
        For each angle and energy the matrix A is only dependent on the
        total cross section (energy).
        Since A is very sparse, use scipy's sparse matrix class to save memory.
        """
        A = sps.eye(self.nVerts, format='dok')
        for element in self.elements:
            nodeIDs, sysVals = element.getElemMatrix(g)
            for nodeID, sysVal in zip(nodeIDs, sysVals):
                A[nodeID] = sysVal
        return A

    def constructRHS(self, g, o):
        """
        Must be performed before each spatial flux solve.  RHS contains
        source terms and boundary values.  Source terms are dependent on the
        previous scattering iterations flux values.
        """
        for element in self.elements:
            nodeIDs, RHSvals = element.getRHS(g, o)
            for nodeID, RHSval in zip(nodeIDs, RHSvals):
                self.RHS[g][o][nodeID] = RHSval

    def buildSysMatrix(self):
        self.sysA = []
        for g in range(self.nG):
            self.sysA.append(self.constructA(g))

    def sweepFlux(self):
        """
        For each angle and energy, solve a system of linear equations
        to update the flux scalar field on the mesh.
        """
        for g in range(self.nG):
            for o in range(self.sNords):
                self.scFluxField[g][o] = sps.linalg.gmres(self.sysA[g], self.RHS[g][o], tol=1e-5)

    def dirichletBCtoA(self, nodeID, A):
        """
        Inside A, the row corresponding to the boundary node is 0 everywhere
        but at the diagonal for a dirichlet BC.
        """
        A[nodeID] *= 0.0
        A[nodeID, nodeID] = 1.0

    def dirichletBCtoRHS(self, nodeID, RHS, value):
        RHS[nodeID] = value


class FE1DSnElement(object):
    """
    Finite element 1D Sn class.

    A 1D finite element contains flux support points on either end.  Source
    terms are computed at the center of the element. linear interpolation is used
    to find the value of the flux at the center of the node (required for computing
    the souce term at the finite element centroid.
    """

    def __init__(self, nodes, nGroups=10, legOrder=8, sNords=2, **kwargs):
        """
        takes ([nodeIDs], [nodePos]) tuple.
        Optionally specify number of groups, leg order and number of ordinates
        """
        #
        # Basic data needed for scattering source calcs
        quadSet = gaussLegQuadSet(sNords)                       # quadrature set
        self.sNords = sNords                                    # number of discrete dirs tracked
        self.sNmu, self.wN = quadSet[0], quadSet[1]             # quadrature weights
        self.maxLegOrder = legOrder                             # remember to range(maxLegORder + 1)
        self.nG = nGroups                                       # number of energy groups
        self.legArray = self._createLegArray(self.maxLegOrder)  # Stores leg polys
        #
        # Store node IDs in element and node positions
        self.nodeIDs, self.nodeVs = nodes
        self.deltaX = self._computeDeltaX()  # Compute deltaX
        #
        # Store material properties
        self.totalXs = kwargs.pop("totalXs", np.ones(self.nG))
        #
        # Flux and source storage
        # initial flux guess
        iguess = kwargs.pop('iFlux', np.ones((nGroups, 3, self.sNords)))
        #
        # Ord flux vector: 0 is cell centered, 1 is left, 2 is right node
        self.ordFlux = iguess
        self.totOrdFlux = iguess
        #
        # Scattering Source term(s)
        self.qin = np.zeros((nGroups, 3, self.sNords))  # init scatter/fission source
        #self.qin[0, 0, :] = 5e10
        self.previousQin = np.ones((nGroups, 3, self.sNords))  # init scatter/fission source
        #
        # optional volumetric source (none by default, fission or user-set possible)
        self.S = kwargs.pop('source', np.zeros((nGroups, 3, self.sNords)))

    def _computeDeltaX(self):
        return np.abs(self.nodeVs[0] - self.nodeVs[1])

    def _computeCentroidFlux(self):
        """
        Centroid flux is required to compute scattering souce in this
        finite element.
        """
        self.ordFlux[:, 0, :] = (self.ordFlux[:, 1, :] + self.ordFlux[:, 2, :]) / 2.
        self.totOrdFlux[:, 0, :] = (self.totOrdFlux[:, 1, :] + self.totOrdFlux[:, 2, :]) / 2.

    def setEleFlux(self, fluxVec, fluxType='scattered'):
        """
        Set all group and ord fluxes at both supports points _and_ centroid
        in the 1D finite element.
        """
        if fluxVec.shape != self.ordFlux.shape:
            sys.exit("FATALITY: flux vec shape mismatch when setting FE flux")
        if fluxType is 'scattered':
            self.ordFlux = fluxVec
        elif fluxType is 'total':
            self.totOrdFlux = fluxVec
        self._computeCentroidFlux()

    def getElemMatrix(self, g):
        """
        Returns element matrix for group g foran _interior_ element.
        Remember to account for boundary conditions elsewhere!

        return ID and value matricies.
        ID matrix:
            ID11,   ID12
            ID21,   ID22
        gives row collum tuples of where to locate the element matrix entries in
        the complete system 'A' matrix.
        """
        elemIDmatrix = [(self.nodeIDs[0], self.nodeIDs[0]), (self.nodeIDs[0], self.nodeIDs[1]),
                        (self.nodeIDs[1], self.nodeIDs[0]), (self.nodeIDs[1], self.nodeIDs[1])]
        feI = np.array([[1, -1], [-1, 1]])
        elemMatrix = (1 / self.deltaX) * feI + (self.totalXs[g] / 2.) * feI
        return elemIDmatrix, elemMatrix.flatten()

    def getRHS(self, g, o):
        """
        Produces right hand side of neutron balance for this element.
        """
        elemIDRHS = np.array([self.nodeIDs[0], self.nodeIDs[1]])
        elemRHS = 0.5 * np.array([self.qin[g, 0, o], self.S[g, 0, o]])
        return elemIDRHS, elemRHS

    def sweepOrd(self, skernel, chiNuFission, keff=1.0, depth=0, overRlx=1.0):
        """
        Perform scattering source iteration in this finite element

        :Parameters:
            :param arg1: arg1 nodes
            :type arg1: type
            :return: return notes
            :rtype: return type
        """
        weights = np.array([np.zeros(self.maxLegOrder + 1)])
        lw = np.arange(self.maxLegOrder + 1)
        if depth >= 1:
            if depth >= 2:
                for g in range(self.nG):
                    self.qin[g, 0, :] = overRlx * (self.evalScatterSourceImp(g, skernel, weights, lw) - self.previousQin[g, 0, :]) + self.previousQin[g, 0, :]
                self.previousQin = self.qin
            else:
                for g in range(self.nG):
                    self.qin[g, 0, :] = self.evalScatterSourceImp(g, skernel, weights, lw)
                self.previousQin = self.qin
        elif self.multiplying and depth == 0:
            for g in range(self.nG):
                # compute gth group fission source
                self.qin[g, 0, :] = self._computeFissionSource(g, chiNuFission, keff)
            self.resetTotOrdFlux()
        elif not self.multiplying and depth == 0:
            self.qin = deepcopy(self.S)
            self.resetTotOrdFlux()
        return self.qin

    def evalScatterSourceImp(self, g, skernel, weights, lw):
        """
        Impoved version of eval scatter source.  Performs same
        operations with 0 _python_ for loops.  all in numpy!
        """
        b = 0.5 * np.dot(self.wN * self.legArray[:, :], self.ordFlux[:, 0, :].T)
        ggprimeInScatter = np.sum(skernel[:, g, :].T * b.T, axis=0)
        weights[0][:] = (2 * lw + 1) * ggprimeInScatter
        return np.sum(weights.T * self.legArray, axis=0)
