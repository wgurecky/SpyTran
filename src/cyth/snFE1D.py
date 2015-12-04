import numpy as np
#import scattSource as scs
from copy import deepcopy
# import scattSrc as scs
import scipy.sparse as sps
import scipy.sparse.linalg as spl
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
        self.superMesh = superMesh(gmshMesh, materialDict, bcDict, srcDict, nGroups, sNords)    # build the mesh
        self.depth = 0  # scattering source iteration depth
        self.keff = 1
        self.buildTransOp()
        self.buildRHS()
        self.applyBCs()

    def scatterSource(self):
        """
        Perform scattering souce iteration for all nodes in the mesh:
        for region in regions:
            for elements in region:
                element.scatter()
        """
        self.superMesh.scatter(self.depth, self.keff)
        if self.depth == 1:
            self.buildTransOp()
        self.superMesh.buildSysRHS()
        self.applyBCs()
        self.depth += 1

    def buildTransOp(self):
        """
        Construct transport operator, A:
        for region in regions:
            for elements in region:
                stamp elmA into sysA
        Note A is not the complete transport operator, it only moves neutrons through space,
        not in energy or angle.  The scattering souce iteration takes care of energy
        and angle redistribution.
        """
        self.superMesh.buildSysMatrix()

    def buildRHS(self):
        self.superMesh.buildSysRHS()

    def applyBCs(self):
        self.superMesh.applyBCs(self.depth)

    def solveFlux(self):
        """
        Solve Ax=b.
        Returns flux norm
        """
        norm, resid = self.superMesh.sweepFlux()
        return norm, resid

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
    def __init__(self, gmshMesh, materialDict, bcDict, srcDict, nG, sNords):
        self.nG, self.sNords = nG, sNords
        self.nNodes = int(np.max(gmshMesh.regions.values()[0]['nodes'][:, 0] + 1))
        self.sysRHS = np.zeros((self.nG, self.sNords, self.nNodes))        # source vector
        self.scFluxField = np.zeros((self.nG, self.sNords, self.nNodes))   # scattered flux field
        self.totFluxField = np.zeros((self.nG, self.sNords, self.nNodes))  # total flux field
        fluxStor = (self.scFluxField, self.totFluxField)
        self.regions = {}     # mesh subregion dictionary
        for regionID, gmshRegion in gmshMesh.regions.iteritems():
            if gmshRegion['type'] == 'interior':
                self.regions[regionID] = regionMesh(gmshRegion, fluxStor, materialDict[gmshRegion['material']],
                                                    bcDict, srcDict.pop(gmshRegion['material'], None))
            elif gmshRegion['type'] == 'bc':
                # mark boundary nodes
                pass
            else:
                print("Unknown region type sepecified in gmsh input. Ignoring")
        print("Number of nodes in mesh: " + str(self.nNodes))

    def scatter(self, depth, keff):
        for regionID, region in self.regions.iteritems():
            region.scatterSrc(depth, keff)

    def buildSysRHS(self):
        self.sysRHS = np.zeros((self.nG, self.sNords, self.nNodes))        # reset source vector
        for regionID, region in self.regions.iteritems():
            for g in range(self.nG):
                for o in range(self.sNords):
                    self.sysRHS = region.buildRegionRHS(self.sysRHS, g, o)

    def buildSysMatrix(self):
        self.sysA = np.empty((self.nG, self.sNords), dtype=sps.dok.dok_matrix)
        for g in range(self.nG):
            for o in range(self.sNords):
                self.sysA[g, o] = self.constructA(g, o)

    def constructA(self, g, o):
        A = sps.eye(self.nNodes, format='csr') * 0.
        A = sps.dok_matrix(A)
        for regionID, region in self.regions.iteritems():
            A = region.buildRegionA(A, g, o)
        return A

    def sweepFlux(self):
        """
        For each angle and energy, solve a system of linear equations
        to update the flux scalar field on the mesh.
        """
        innerResid = 0
        for g in range(self.nG):
            for o in range(self.sNords):
                self.scFluxField[g, o], Aresid = \
                    spl.gmres(sps.csc_matrix(self.sysA[g, o]), self.sysRHS[g, o], tol=1e-5)
                innerResid += Aresid
        self.totFluxField += self.scFluxField
        for regionID, region in self.regions.iteritems():
            fluxStor = (self.scFluxField, self.totFluxField)
            region.updateEleFluxes(fluxStor)
        return np.linalg.norm(self.scFluxField), innerResid

    def applyBCs(self, depth):
        for regionID, region in self.regions.iteritems():
            self.sysA, self.sysRHS = region.setBCs(self.sysA, self.sysRHS, depth)


class regionMesh(object):
    def __init__(self, gmshRegion, fluxStor, material, bcDict, source, **kwargs):
        """
        Each region requires a material specification.

        Each region requires a node layout specification.
        A 1D mesh has the following structure:
        [[elementID1, x1, x2],
         [elementID2, x2, x3]
         ...
        ]
        """
        self.bcDict = bcDict
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
            #source = kwargs.pop("source", None)
        # Build elements in the region mesh
        self.buildElements(gmshRegion, fluxStor, source, **kwargs)
        self.linkBoundaryElements(gmshRegion)

    def buildElements(self, gmshRegion, fluxStor, source, **kwargs):
        """
        Initilize and store interior elements.
        """
        self.elements = {}
        for element in gmshRegion['elements']:
            nodeIDs = element[1:]
            nodePos = [gmshRegion['nodes'][nodeID][1] for nodeID in nodeIDs]
            self.elements[element[0]] = InteriorElement((nodeIDs, nodePos), fluxStor, source, **kwargs)

    def linkBoundaryElements(self, gmshRegion):
        """
        Store boundary elements that border this region.  Link the interior element
        with its corrosponding boundary element
        """
        self.belements = {}  # boundary element dict (empty if subregion contains no boundaries)
        for bctype, bcElms in gmshRegion['bcElms'].iteritems():
            if type(bcElms) is dict:
                for bcElmID, nodeIDs in bcElms.iteritems():
                    nodePos = [gmshRegion['nodes'][nodeID][1] for nodeID in nodeIDs]
                    self.belements[bcElmID] = BoundaryElement(self.bcDict[bctype], (nodeIDs, nodePos), self.elements[bcElmID])

    def buildRegionA(self, A, g, o):
        """
        Populate matrix A for group g for nodes in this region.
        This must only be done once
        as the system matrix A is not dependent on the flux.
        For each angle and energy the matrix A is only dependent on the
        total cross section (energy).
        Since A is very sparse, use scipy's sparse matrix class to save memory.
        """
        for elementID, element in self.elements.iteritems():
            nodeIDs, sysVals = element.getElemMatrix(g, o, self.totalXs)
            for nodeID, sysVal in zip(nodeIDs, sysVals):
                # add values to A
                # TODO: We have to rebuild A for the first 2 scatter iters :(
                # due to bcs changing slightly between the 0th and 1st iter
                A[nodeID] += sysVal
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
                #TODO: Zero-out RHS between outer iters. rebuild it every scatter
                # iteration
                # RHS[g, o, nodeID] = RHSval
                RHS[g, o, nodeID] += RHSval
        return RHS

    def setBCs(self, A, RHS, depth):
        A = self.setRegionBCsA(A, depth)
        RHS = self.setRegionBCsRHS(RHS, depth)
        return A, RHS

    def setRegionBCsA(self, A, depth):
        for belementID, belement in self.belements.iteritems():
            A = belement.applyBC2A(A, depth)
        return A

    def setRegionBCsRHS(self, RHS, depth):
        for belementID, belement in self.belements.iteritems():
            RHS = belement.applyBC2RHS(RHS, depth)
        return RHS

    def scatterSrc(self, depth, keff):
        """
        Perform scattering souce iteration for all elements in region.
        """
        for elementID, element in self.elements.iteritems():
            element.sweepOrd(self.skernel, self.chiNuFission, keff, depth)

    def updateEleFluxes(self, fluxStor):
        for elementID, element in self.elements.iteritems():
            element.updateFluxes(fluxStor)


class InteriorElement(object):
    """
    Finite element 1D Sn class.

    A 1D finite element contains flux support points on either end.  Source
    terms are computed at the center of the element. linear interpolation is used
    to find the value of the flux at the center of the node (required for computing
    the souce term at the finite element centroid.
    """

    def __init__(self, nodes, fluxStor, source, **kwargs):
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
        self.setEleScFlux(fluxStor[0])
        self.setEleTotFlux(fluxStor[1])
        #
        # Scattering Source term(s)
        self.qin = np.zeros((self.nG, self.sNords))         # init scatter/fission source
        self.previousQin = np.ones(self.qin.shape)  # required for over relaxation
        #
        # Volumetric source
        self.S = source
        self.multiplying = False
        if type(self.S) is str:
            if self.S == 'fission':
                self.multiplying = True
            else:
                self.S = np.zeros((self.nG, self.sNords))
        elif self.S is None:
            self.S = np.zeros((self.nG, self.sNords))
        elif type(self.S) is np.ndarray:
            if self.S.shape != (self.nG, self.sNords):
                sys.exit("FATALITY: Invalid shape of source vector. Shape must be (nGrps, 3, sNords).")
            else:
                pass

    def updateFluxes(self, fluxStor):
        self.setEleScFlux(fluxStor[0])
        self.setEleTotFlux(fluxStor[1])

    def setEleScFlux(self, scFluxField):
        """
        Storage for scattered fluxs
        node scattered flux vector is a [ngrp, nord, nNodes] array
        """
        self.nodeScFlux = scFluxField[:, :, self.nodeIDs]  # scatteredFlux
        self.centScFlux = np.average(self.nodeScFlux, axis=2)

    def setEleTotFlux(self, totFluxField):
        """
        Storage for total flux (sum of all scattered fluxes)
        """
        self.nodeTotFlux = totFluxField[:, :, self.nodeIDs]
        self.centTotFlux = np.average(self.nodeTotFlux, axis=2)

    def _computeDeltaX(self):
        self.deltaX = np.abs(self.nodeVs[0] - self.nodeVs[1])

    def getElemMatrix(self, g, o, totalXs):
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
        elemMatrix = (np.abs(self.sNmu[o]) / self.deltaX) * feI + (totalXs[g] * 2. / self.deltaX) * feI
        return elemIDmatrix, elemMatrix.flatten()

    def getRHS(self, g, o):
        """
        Produces right hand side of neutron balance for this element.
        """
        elemIDRHS = np.array([self.nodeIDs[0], self.nodeIDs[1]])
        elemRHS = 0.5 * np.array([self.qin[g, o], self.qin[g, o]])
        return elemIDRHS, elemRHS

    def resetTotOrdFlux(self):
        self.centTotFlux *= 0
        self.nodeTotFlux *= 0

    def resetOrdFlux(self):
        self.centScFlux *= 0
        self.nodeScFlux *= 0

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
                    self.qin[g, :] = overRlx * (self.evalScatterSourceImp(g, skernel, weights, lw) - self.previousQin[g, :]) + self.previousQin[g, :]
                self.previousQin = self.qin
            else:
                for g in range(self.nG):
                    self.qin[g, :] = self.evalScatterSourceImp(g, skernel, weights, lw)
                self.previousQin = self.qin
        elif self.multiplying and depth == 0:
            for g in range(self.nG):
                # compute gth group fission source
                self.qin[g, :] = self._computeFissionSource(g, chiNuFission, keff)
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
        b = 0.5 * np.dot(self.wN * self.legArray[:, :], self.centScFlux[:, :].T)
        ggprimeInScatter = np.sum(skernel[:, g, :].T * b.T, axis=0)
        weights[0][:] = (2 * lw + 1) * ggprimeInScatter
        return np.sum(weights.T * self.legArray, axis=0)

    def _computeFissionSource(self, g, chiNuFission, keff):
        """
        Compute the withen group fission source.
        chiNuFission[g] is a row vector corresponding to all g'
        """
        if self.multiplying:
            return (1 / keff / 8.0 / (1.0)) * \
                np.sum(chiNuFission[g] * self._evalCentTotAngleInt(g))
        else:
            # need fixed source from user input
            print("Fission source requested for Non multiplying medium.  FATALITY")
            sys.exit()

    def _evalCentTotAngleInt(self, g):
        """
        group scalar flux evaluator
        scalar_flux_g = (1/2) * sum_n(w_n * flux_n)
        n is the ordinate iterate
        """
        scalarFlux = np.sum(self.wN * self.centTotFlux[g, :])
        return 0.5 * scalarFlux


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
    def __init__(self, bcData, nodes, parentElement):
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
        self.bcData = bcData  # could be a string e.g: 'vac', or could be a dict
        self.computeOutNormal()
        self.computeInOrds()

    def applyBC2RHS(self, RHS, depth):
        if self.bcData == 'vac':
            return self.vacBC(RHS)
        elif self.bcData == 'ref':
            return self.refBC(RHS)
        elif type(self.bcData) is np.ndarray:
            if self.bcData.shape != self.parent.centTotFlux.shape:
                print("WARNING: BC flux shape mismatch.")
            if depth == 0:
                return self.diricletBC(RHS)
            else:
                return self.vacBC(RHS)
        else:
            print("WARNING: BC assignment failed.  Assuming free boundary.")
        return RHS

    def applyBC2A(self, A, depth):
        """
        TODO: finish bc assignment to A
        """
        if depth >= 2:
            # do nothing after 2nd scattering iteration. A does not change past
            # this point
            return A
        if self.bcData == 'vac':
            return self.vacBC2A(A)
        elif self.bcData == 'ref':
            pass
        elif type(self.bcData) is np.ndarray:
            if depth == 0:
                return self.fixedBC2A(A)
            else:
                return self.vacBC2A(A)

    def computeOutNormal(self):
        # obtain node(s) that are common between parent ele and boundary.
        commonNodeV = np.intersect1d(self.parent.nodeVs, self.nodeVs)
        self.internalBCnodeIDs = np.array([np.where(cV == self.parent.nodeVs) for cV in commonNodeV])[:, 0, 0]
        # obtain odd man out node
        lonelyNodeV = np.setdiff1d(self.parent.nodeVs, self.nodeVs)
        # In 2D and 3D we must find orthogonal line to surface/line
        self.outwardNormal = (commonNodeV - lonelyNodeV) / np.abs(commonNodeV - lonelyNodeV)

    def computeInOrds(self):
        """ Compute inward ordinate directions.  Those ords with dot product with the
        outward normal negative """
        # Works for 1D only at the moment, simple inspection of the
        # magnitude of multiplication of direction
        # cosines works fine.
        faceDot = self.parent.sNmu * self.outwardNormal
        self.inOs = np.where(faceDot < 0)
        self.outOs = np.where(faceDot >= 0)

    def vacBC2A(self, A):
        """
        Vaccume boundary only applied to inward ordinates at boundary.
        This manifests itself as a row in the A matrix being 0 everywhre but
        at the ordinaties facing inwards at the boundary node.
        """
        # system A is structured as:
        # [group, ordinate, n, n], where n is number of nodes in the problem
        for o in range(self.parent.sNords):
            if o in self.inOs:
                for g in range(self.parent.nG):
                    A[g, o][self.nodeIDs, :] = 0.
                    A[g, o][self.nodeIDs, self.nodeIDs] = 1.
        return A

    def fixedBC2A(self, A):
        """
        Since the flux is _specified_ for the bc nodes (atleast on scatter iter 0)
        for the boundary node in question; zero out its row in the A matrix everywhere
        but at the diagonal.  We are after 1 * flux_o_g = specified_flux_o_g
        """
        for o in range(self.parent.sNords):
            for g in range(self.parent.nG):
                A[g, o][self.nodeIDs, :] = 0.
                A[g, o][self.nodeIDs, self.nodeIDs] = 1.
        return A

    def vacBC(self, RHS):
        """
        RHS for all groups, g,  and for inward facing ordinates, are set to
        zero.
        """
        RHS[:, self.inOs, self.nodeIDs] = 0
        return RHS

    def refBC(self, RHS):
        """
        Apply reflected flux dirichletBC.  Take banked fluxes on the outer surface
        and add to the RHS.
        """
        RHS[:, self.inOs, self.nodeIDs] = self.bankedRefFlux[:, self.inOs, self.internalBCnodeIDs]
        return RHS

    def diricletBC(self, RHS):
        """
        Fixed flux bc.
        """
        for bcNodeID in self.nodeIDs:
            RHS[:, :, bcNodeID] = self.bcData
        return RHS

    def postSweepRefBC(self, RHS):
        """
        Bank outward ordinate fluxes in parent cell for use in next
        source iteration.
        Perform this action after each spacial flux solve is complete.
        """
        self.bankedRefFlux = np.zeros(self.parent.nodeScFlux.shape)
        for iDir in self.inOs[0]:
            # get negative mu in iDir
            negDir = -1 * self.parent.sNmu[iDir]
            outDir = np.where(np.round(negDir, 6) == np.round(self.parent.sNmu, 6))
            # only look at nodes which lie on boundary
            self.bankedRefFlux[:, iDir, self.internalBCnodeIDs] = \
                self.parent.nodeScFlux[:, outDir[0][0], self.internalBCnodeIDs]


if __name__ == "__main__":
    # Construct mesh
    # Inspect the system matrix A
    pass
