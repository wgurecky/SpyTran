import numpy as np
#import scattSource as scs
from copy import deepcopy
# import scattSrc as scs
import scipy.sparse as sps
import sys
from utils.ordReader import gaussLegQuadSet
from utils.ordReader import createLegArray
from utils.gmshPreproc import gmsh1DMesh
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
    def __init__(self, geoFile, materialDict, nGroups=10, legOrder=8, sNords=2):
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
        self.legArray = createLegArray(self.snMu, self.maxLegOrder)  # Stores leg polys
        #
        gmshMesh = gmsh1DMesh(geoFile=geoFile)  # Run gmsh
        self.superMesh = superMesh(gmshMesh, materialDict)    # build the mesh
        self.depth = 0  # scattering source iteration depth
        self.buildTransOp()

    def scatterSource(self):
        """
        Perform scattering souce iteration for all nodes in the mesh:
        for region in regions:
            for elements in region:
                element.scatter()
        """
        self.superMesh.scatter(self.depth, self.keff)
        self.depth += 1

    def buildTransOp(self):
        """
        Construct transport operator A:
        for region in regions:
            for elements in region:
                stamp elmA into sysA
        """
        self.superMesh.buildSysMatrix()

    def solveFlux(self):
        """
        Solve Ax=b.
        """
        pass

    def writeFlux(self):
        """
        Write solution state to hdf5 file.
            - keff (if applicable)
            - mesh
                - elements (nodes in element)
                - node positions
            - flux field
        """
        pass


class superMesh(object):
    """
    Contains all region meshes.
    Contains mappings betwen array/matrix field representation and element class
    representation.
    """
    def __init__(self, gmshMesh, materialDict):
        self.regions = {}     # mesh subregion dictionary
        self.nNodes = 0    # number of nodes in mesh
        for regionID, gmshRegion in gmshMesh.regions.iteritems():
            if gmshRegion['type'] == 'interior':
                self.regions[regionID] = regionMesh(gmshRegion, materialDict[gmshRegion['material']])
                self.nNodes += gmshRegion['nodes'].shape[0]
            elif gmshRegion['type'] == 'bc':
                # mark boundary nodes
                pass
            else:
                print("Unknown region type sepecified in gmsh input. Ignoring")

    def scatter(self, depth, keff):
        for regionID, region in self.regions.iteritems():
            region.scatterSrc(depth, keff)

    def buildSysMatrix(self):
        self.sysA = []
        for g in range(self.nG):
            self.sysA.append(self.constructA(g))

    def constructA(self, g):
        A = sps.eye(self.nNodes, format='dok')
        for regionID, region in self.regions.iteritems():
            A = region.buildRegionA(A, g)
        return A

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


class regionMesh(object):
    def __init__(self, gmshRegion, material, **kwargs):
        """
        Each region requires a material specification.

        Each region requires a node layout specification.
        A 1D mesh has the following structure:
        [[elementID1, x1, x2],
         [elementID2, x2, x3]
         ...
        ]
        """
        self.totalXs = material.macroProp['Ntotal']
        self.skernel = material.macroProp['Nskernel']
        if 'chi' in material.macroProp.keys():
            self.nuFission = material.macroProp['Nnufission']
            self.chiNuFission = np.dot(np.array([material.macroProp['chi']]).T,
                                       np.array([material.macroProp['Nnufission']]))
            source = 'fission'
        else:
            self.nuFission = None
            self.chiNuFission = None
            source = kwargs.pop("source", None)
        # Build elements in the region mesh
        self.buildElements(gmshRegion, source, **kwargs)
        self.linkBoundaryElements(gmshRegion)

    def buildElements(self, gmshRegion, **kwargs):
        """
        Initilize and store interior elements.
        """
        self.elements = {}
        for element in gmshRegion['elements']:
            nodeIDs = element[1:]
            nodePos = [gmshRegion['nodes'][nodeID][1] for nodeID in nodeIDs]
            self.elements[element[0]] = InteriorElement((nodeIDs, nodePos), **kwargs)

    def linkBoundaryElements(self, gmshRegion):
        """
        Store boundary elements that border this region.  Link the interior element
        with its corrosponding boundary element
        """
        self.belements = {}  # boundary element dict (empty if subregion contains no boundaries)
        for bctype, bcElms in gmshRegion['bcElms'].iteritems():
            for bcElmID, nodeIDs in bcElms.iteritems():
                nodePos = [gmshRegion['nodes'][nodeID][1] for nodeID in nodeIDs]
                self.belements[bcElmID] = BoundaryElement(bctype, (nodeIDs, nodePos), self.elements[bcElmID])

    def buildRegionA(self, A, g):
        """
        Populate matrix A for group g for nodes in this region.
        This must only be done once
        as the system matrix A is not dependent on the flux.
        For each angle and energy the matrix A is only dependent on the
        total cross section (energy).
        Since A is very sparse, use scipy's sparse matrix class to save memory.
        """
        for elementID, element in self.elements.iteritems():
            nodeIDs, sysVals = element.getElemMatrix(g, self.totalXs)
            for nodeID, sysVal in zip(nodeIDs, sysVals):
                A[nodeID] = sysVal
        return A

    def buildRegionRHS(self, RHS, g, o):
        """
        Must be performed before each spatial flux solve.  RHS contains
        source terms and boundary values.  Source terms are dependent on the
        previous scattering iterations flux values.
        """
        for elementID, element in self.elements.iteritems():
            nodeIDs, RHSvals = element.getRHS(g, o)
            for nodeID, RHSval in zip(nodeIDs, RHSvals):
                RHS[g][o][nodeID] = RHSval
        return RHS

    def setRegionBCsA(self, A, g):
        for belementID, belement in self.belements.iteritems():
            pass

    def setRegionBCsRHS(self, RHS, g, o):
        for belementID, belement in self.belements.iteritems():
            pass

    def scatterSrc(self, depth, keff):
        """
        Perform scattering souce iteration for all elements in region.
        """
        for elementID, element in self.elements.iteritems():
            element.sweepOrd(self.skernel, self.chiNuFission, keff, depth)


class InteriorElement(object):
    """
    Finite element 1D Sn class.

    A 1D finite element contains flux support points on either end.  Source
    terms are computed at the center of the element. linear interpolation is used
    to find the value of the flux at the center of the node (required for computing
    the souce term at the finite element centroid.
    """

    def __init__(self, nodes, **kwargs):
        """
        takes ([nodeIDs], [nodePos]) tuple.
        Optionally specify number of groups, leg order and number of ordinates
        """
        #
        # Basic data needed for scattering source calcs
        self.sNords = kwargs.pop("sNords", 2)                                    # number of discrete dirs tracked
        quadSet = kwargs.pop("quadSet", gaussLegQuadSet(self.sNords))            # quadrature set
        self.sNmu, self.wN = quadSet[0], quadSet[1]                              # quadrature weights
        self.maxLegOrder = kwargs.pop("legOrder", 8)                             # remember to range(maxLegORder + 1)
        self.nG = kwargs.pop("nGroups", 10)                                      # number of energy groups
        self.legArray = kwargs.pop("legP", createLegArray(self.sNmu, self.maxLegOrder))     # Stores leg polys
        #
        # Store node IDs in element and node positions
        self.nodeIDs, self.nodeVs = nodes
        self._computeDeltaX()  # Compute deltaX
        #
        # Flux and source storage
        # initial flux guess
        iguess = kwargs.pop('iFlux', np.ones((self.nG, 3, self.sNords)))
        #
        # Ord flux vector: 0 is cell centered, 1 is left, 2 is right node
        self.ordFlux = iguess
        self.totOrdFlux = iguess
        #
        # Scattering Source term(s)
        self.qin = np.zeros((self.nG, 3, self.sNords))  # init scatter/fission source
        self.previousQin = np.ones((self.nG, 3, self.sNords))  # init scatter/fission source
        #
        # optional volumetric source (none by default, fission or user-set possible)
        self.S = kwargs.pop('source', np.zeros((self.nG, 3, self.sNords)))
        self.multiplying = False
        if type(self.S) is str:
            if self.S == 'fission':
                self.multiplying = True
            else:
                self.S = np.zeros((self.nG, 3, self.sNords))
        elif self.S is None:
            self.S = np.zeros((self.nG, 3, self.sNords))
        elif type(self.S) is np.ndarray:
            if self.S.shape != (self.nG, 3, self.sNords):
                sys.exit("FATALITY: Invalid shape of source vector. Shape must be (nGrps, 3, sNords).")

    def _computeDeltaX(self):
        self.deltaX = np.abs(self.nodeVs[0] - self.nodeVs[1])

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

    def getElemMatrix(self, g, totalXs):
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
        elemMatrix = (1 / self.deltaX) * feI + (totalXs[g] * 2. / self.deltaX) * feI
        return elemIDmatrix, elemMatrix.flatten()

    def getRHS(self, g, o):
        """
        Produces right hand side of neutron balance for this element.
        """
        elemIDRHS = np.array([self.nodeIDs[0], self.nodeIDs[1]])
        elemRHS = 0.5 * np.array([self.qin[g, 0, o], self.qin[g, 0, o]])
        return elemIDRHS, elemRHS

    def resetTotOrdFlux(self):
        self.totOrdFlux = np.zeros((self.nG, 3, self.sNords))

    def resetOrdFlux(self):
        self.ordFlux = np.zeros((self.nG, 3, self.sNords))

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


class BoundaryElement(object):
    """
    In 1D boundary conditions are specified on a node.
    Equivillent to a boundary element in 2D.
    Contains methods to compute face normals needed for setting vacuume and
    reflective boundary conditions.

    Notes:
        Boundary condition assignment only works for CONVEX shapes.
        No interior corners allowed!!
    """
    def __init__(self, bcType, nodes, parentElement):
        """
        bcType is a str:  "vac" or "ref"
        Requres interior element on which the boundary node resides.
        This is required for computing the outward normals.

        x---Eb----o----Ei----o----Ei--

        where x is the boundary node, o are interior nodes and Eb is the
        boundary element.
        """
        # Store node IDs in element and node positions
        self.nodeIDs, self.nodeVs = nodes
        # Link to the parent element.  parent element class contains
        # ordinate direction and node location info required to determine
        # outward normals
        self.parent = parentElement
        self.bcType = bcType

    def computeOutNormals(self):
        pass
