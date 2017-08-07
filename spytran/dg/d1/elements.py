import numpy as np
from copy import deepcopy
import sys
from spytran.utils.ordReader import createLegArray
np.set_printoptions(linewidth=200)  # set print to screen opts


class d1InteriorElement(object):
    """!
    @brief Finite element 1D Sn class.

    A 1D finite element contains flux support points on either end.  Source
    terms are computed at the center of the element. linear interpolation is used
    to find the value of the flux at the center of the node (required for computing
    the souce term at the finite element centroid.
    """
    def __init__(self, nodes, fluxStor, source, gmsh_dg_element, **kwargs):
        """
        takes ([nodeIDs], [nodePos]) tuple.
        Optionally specify number of groups, leg order and number of ordinates
        """
        #
        self.gmsh_dg_element = gmsh_dg_element
        # Basic data needed for scattering source calcs
        self.sNords = kwargs.pop("sNords", 4)                                    # number of discrete dirs tracked
        quadSet = kwargs.pop("quadSet")                                          # quadrature set
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
            if self.S is 'fission':
                self.multiplying = True
            else:
                self.S = np.zeros((self.nG, self.sNords))
        elif self.S is None:
            self.S = np.zeros((self.nG, self.sNords))
        elif type(self.S) is np.ndarray:
            if self.S.shape != (self.nG, self.sNords):
                sys.exit("FATALITY: Invalid shape of source vector. Shape must be (nGrps, sNords).")
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
        self.sortedNodeIndex = np.argsort(self.nodeVs)

    def getElemMatrix(self, g, o, totalXs):
        """
        Returns element matrix for group g for an interior element.
        Remember to account for boundary conditions elsewhere!

        return ID and value matricies.
        ID matrix:
            ID11,   ID12
            ID21,   ID22
        gives row collum tuples of where to locate the element matrix entries in
        the complete system 'A' matrix.
        """
        internal_id_matrix = [(self.nodeIDs[0], self.nodeIDs[0]), (self.nodeIDs[0], self.nodeIDs[1]),
                              (self.nodeIDs[1], self.nodeIDs[0]), (self.nodeIDs[1], self.nodeIDs[1])]
        # check element orientation
        if self.nodeVs[0] < self.nodeVs[1]:
            feI = np.array([[-1, 1], [-1, 1]])
        else:
            feI = np.array([[1, -1], [1, -1]])
        feI2 = np.array([[1, 0.5], [0.5, 1]])
        internal_matrix = (-0.5 * self.sNmu[o]) * feI + ((1 / 3.) * totalXs[g] * self.deltaX) * feI2
        return internal_id_matrix, internal_matrix.flatten()

    def getNeighborMatrix(self, g, o, totalXs, numerical_flux='upwind'):
        """!
        @brief  Couples the neighboring elements with the current element
        through boundary-upwinded or boundary-average fluxes.
        @param g  int. group id
        @param o  int.  ordinate id
        @param totalXs  float.  Total XS for this energy group in this element
        @param numerical_flux string.  either 'upwind' or 'avg'
            Note: The 'upwind' method is known to 'lock' in the diffusion limit.
            TODO: automatically select between methods depending on value for
            totalXs
        For upwind: for each edge the upwind element is determined.
        Next, the term:
        \f[
        \int_{\partial K} \hat{F} \cdot \mathbf n \phi dx
        \f]
        Where \f[ \hat F \cdot \mathbf n \f] is given by
        \f[
        \Omega \cdot \mathbf n \psi^{\uparrow}
        \f]
        Where \f[ \mathbf n \f] is the outward edge normal
        and \f[\Omega \f] is the current ordinate direction

        @return  two lists:
            boundary_id_matrix:  a list of tuples that determines linkages between nodes
                                 in the global system matrix.  Identifies the element (i, j)
                                 in the system matrix A to place the values in the
                                 other returned list: boundary_matrix
            boundary_matrix:  holds the coupling coefficients.
        The complete A matrix will be a banded sparse matrix in the case of CG or DG
        with DG having more degrees of freedom (all internal nodes are multi-valued in the DG case)
        """
        boundary_id_matrix = []
        boundary_matrix = []
        for k, neighbor_edge_id in enumerate(self.gmsh_dg_element['neighbors']['neighbor_edge_ids']):
            parent_edge_id = self.gmsh_dg_element['neighbors']['parent_edge_ids'][k]
            parent_edge_global_node_ids = self.gmsh_dg_element['neighbors']['parent_edge_global_node_ids'][k]
            neighbor_edge_global_node_ids = self.gmsh_dg_element['neighbors']['neighbor_edge_global_node_ids'][k]
            # Build edge coupling matrix
            p = parent_edge_global_node_ids[0]
            n = neighbor_edge_global_node_ids[0]
            boundary_id_matrix_k = [(p, p), (p, n),
                                    (n, p), (n, n)]
            # Obtain edge outward normal
            parent_edge = self.gmsh_dg_element['edges'][parent_edge_id]
            edge_normal = parent_edge['edge_normal']
            out_normal_dot_mu = np.dot(np.array([self.sNmu[o], 0., 0.]), edge_normal)
            if out_normal_dot_mu > 0:
                # edge normal is in same dir as ordinate dir
                # Therefore use the parent element's flux at this edge to
                # determine boundary flux
                boundary_matrix_k = np.array([1.0, 0.0,
                                              0.0, 0.0])
                if numerical_flux == 'avg':
                    boundary_matrix_k = np.array([0.5, 0.0,
                                                  0.5, 0.0])
            else:
                # edge normal is in oposite dir as ordinate dir
                # Therefore use the neighbor element's flux at this edge to
                # determine boundary flux
                boundary_matrix_k = np.array([0.0, 0.0,
                                              -1.0, 0.0])
                if numerical_flux == 'avg':
                boundary_matrix_k = np.array([-0.5, 0.0,
                                              -0.5, 0.0])
            boundary_id_matrix += boundary_id_matrix_k
            boundary_matrix += list(boundary_matrix_k * np.abs(out_normal_dot_mu))
        return boundary_id_matrix, boundary_matrix

    def getRHS(self, g, o):
        """
        Produces right hand side of neutron balance for this element.
        """
        elemIDRHS = np.array([self.nodeIDs[0], self.nodeIDs[1]])
        elemRHS = 0.5 * self.deltaX * np.array([self.qin[g, o], self.qin[g, o]])
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
            for g in range(self.nG):
                self.qin[g, :] = self.evalScatterSourceImp(g, skernel, weights, lw)
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
        # legArray[l, ord]
        fluxM = 0.5 * np.dot(self.wN * self.legArray[:, :], self.centScFlux[:, :].T)
        ggprimeInScatter = np.sum(skernel[:, g, :].T * fluxM.T, axis=0)
        weights[0][:] = (2 * lw + 1) * ggprimeInScatter
        scSource = np.sum(weights.T * self.legArray, axis=0)
        return scSource

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


class d1BoundaryElement(object):
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
        if self.bcData is 'vac':
            return self.vacBC(RHS)
        elif self.bcData is 'ref':
            if depth == 0:
                return self.vacBC(RHS)
            else:
                self.postSweepRefBC()
                return self.refBC(RHS)
        elif type(self.bcData) is np.ndarray:
            if self.bcData.shape != self.parent.centTotFlux.shape:
                print("WARNING: BC flux shape mismatch.")
            if depth == 0:
                return self.dirichletBC(RHS)
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
        if self.bcData is 'vac':
            return self.vacBC2A(A)
        elif self.bcData is 'ref':
            return self.vacBC2A(A)
            #if depth == 0:
            #    return self.vacBC2A(A)
            #else:
            #    return self.fixedBC2A(A)
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
        # TODO: THIS IS THE FRIGGIN DG BUG.  OUTWARD NORMAL IS
        # NO LONGER COMPUTED CORRECTLY HERE (MOVED TO MESH DEF)
        # import pdb; pdb.set_trace()
        # self.outwardNormal = (commonNodeV - lonelyNodeV) / np.abs(commonNodeV - lonelyNodeV)  # broke
        self.outwardNormal = -(commonNodeV - lonelyNodeV) / np.abs(commonNodeV - lonelyNodeV)  # tmp fix

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
            if o in self.inOs[0]:
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

    def dirichletBC(self, RHS):
        """
        Fixed flux bc.
        """
        for bcNodeID in self.nodeIDs:
            RHS[:, :, bcNodeID] = self.bcData
        return RHS

    def postSweepRefBC(self):
        """
        Bank outward ordinate fluxes in parent cell for use in next
        source iteration.
        Perform this action after each space flux solve is complete.
        """
        self.bankedRefFlux = np.zeros(self.parent.nodeScFlux.shape)
        for iDir in self.inOs[0]:
            # get negative mu in iDir
            negDir = -1 * self.parent.sNmu[iDir]
            outDir = np.where(np.round(negDir, 6) == np.round(self.parent.sNmu, 6))
            # only look at nodes which lie on boundary
            self.bankedRefFlux[:, iDir, self.internalBCnodeIDs] = \
                self.parent.nodeScFlux[:, outDir[0][0], self.internalBCnodeIDs]
