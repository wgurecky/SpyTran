import numpy as np
from copy import deepcopy
import sys
np.set_printoptions(linewidth=200)  # set print to screen opts


class d2InteriorElement(object):
    """
    Finite element 2D Sn class.

    A 2D finite element contains flux support points on triangle nodes.  Source
    terms are computed at the center of the element. linear interpolation is used
    to find the value of the flux at the center of the element (required for computing
    the souce term at the finite element centroid.
    """

    def __init__(self, nodes, fluxStor, source, **kwargs):
        """
        takes ([nodeIDs], [nodePos]) tuple.
        Optionally specify number of groups, leg order and number of ordinates
        """
        #
        # Basic data needed for scattering source calcs
        self.quadSet = kwargs.get("quadSet")
        self.sNords = self.quadSet.sNords
        self.sNmu, self.sNeta, self.wN = self.quadSet.mus, self.quadSet.etas, self.quadSet.wN
        self.maxLegOrder = kwargs.pop("legOrder", 8)                             # remember to range(maxLegORder + 1)
        self.nG = kwargs.pop("nGroups", 10)                                      # number of energy groups
        self.C = np.zeros(self.maxLegOrder + 1)
        self.C[0] = 1.
        #
        # Store node IDs in element and node positions
        self.nodeIDs, self.nodeVs = nodes
        self._computeArea()  # Compute area
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
        # Use vandermonde matrix to obtain coeffs of lin interpolant for
        # _all_ scalar flux fields
        #C = np.dot(self.vI, self.nodeScFlux)  # check dims!
        #self.centScFlux = np.dot(C, self.centAroid)
        self.centScFlux = np.average(self.nodeScFlux, axis=2)

    def setEleTotFlux(self, totFluxField):
        """
        Storage for total flux (sum of all scattered fluxes)
        """
        self.nodeTotFlux = totFluxField[:, :, self.nodeIDs]
        self.centTotFlux = np.average(self.nodeTotFlux, axis=2)

    def _computeArea(self):
        #self.sortedNodeIndexX = np.argsort(self.nodeVs[:, 0])
        #self.sortedNodeIndexY = np.argsort(self.nodeVs[:, 1])
        self.centroid = np.array([np.average(self.nodeVs[:, 0]), np.average(self.nodeVs[:, 1])])
        # pre-compute vandermonde and vandermonde matrix inverse on element init
        self.vV = np.vstack([np.ones(3), self.nodeVs[:, 0], self.nodeVs[:, 1]]).T
        try:
            self.vI = np.linalg.inv(self.vV)
        except:
            sys.exit("SINGULAR vandermonde matrix.  Mangled mesh.")
        self.area = 0.5 * abs(np.linalg.det(self.vV))

    def getElemMatrix(self, g, o, totalXs):
        """
        Returns element matrix for group g foran _interior_ element.
        Remember to account for boundary conditions elsewhere!
        gives row collum tuples of where to locate the element matrix entries in
        the complete system 'A' matrix.
        """
        elemIDmatrix = [(self.nodeIDs[0], self.nodeIDs[0]), (self.nodeIDs[0], self.nodeIDs[1]), (self.nodeIDs[0], self.nodeIDs[2]),
                        (self.nodeIDs[1], self.nodeIDs[0]), (self.nodeIDs[1], self.nodeIDs[1]), (self.nodeIDs[1], self.nodeIDs[2]),
                        (self.nodeIDs[2], self.nodeIDs[0]), (self.nodeIDs[2], self.nodeIDs[1]), (self.nodeIDs[2], self.nodeIDs[2])]
        feI = - np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1]])
        feI2 = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
        elemMatrix = ((1 / 3.) * self.sNmu[o]) * feI + ((1 / 3.) * self.sNeta[o]) * feI + \
            ((1 / 24.) * totalXs[g] * (0.5 * self.area)) * feI2
        return elemIDmatrix, elemMatrix.flatten()

    def getRHS(self, g, o):
        """
        Produces right hand side of neutron balance for this element.
        """
        elemIDRHS = np.array([self.nodeIDs[0], self.nodeIDs[1], self.nodeIDs[2]])
        #J = np.array([[np.nodeVs[1, 0] - np.nodeVs[0, 0]], [np.nodeVs[2, 0] - np.nodeVs[0, 0]],
        #              [np.nodeVs[1, 1] - np.nodeVs[0, 1]], [np.nodeVs[2, 1] - np.nodeVs[0, 1]]])
        elemRHS = (1 / 6.) * (1. * self.area) * np.array([self.qin[g, o], self.qin[g, o], self.qin[g, o]])
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
        #weights = np.array([np.zeros(self.maxLegOrder + 1)])
        weights = np.array([np.zeros((self.maxLegOrder + 1, self.maxLegOrder + 1))])
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

    def evalScatterSourceImp(self, G, skernel, weights, lw):
        """
        Impoved version of eval scatter source.  Performs same
        operations with 0 _python_ for loops.  all in numpy!
        """
        # Ylm[m, l, ord]
        #b = 0.25 * np.dot(self.wN * self.legArray[:, :], self.centScFlux[:, :].T)
        fluxM = 0.25 * np.dot(self.wN * self.quadSet.Ylm[:, :, :], self.centScFlux[:, :].T)
        ggprimeInScatter = np.sum(skernel[:, G, :] * fluxM, axis=2)
        weights[0] = (2 - self.C) * ggprimeInScatter
        scSource = np.sum(weights.T * self.quadSet.Ylm, axis=(0, 1))
        #
        S = np.zeros(self.qin.shape[1])
        for n in range(0, self.qin.shape[1]):
            for gp in range(self.nG):
                for l in range(self.maxLegOrder + 1):
                    S[n] += skernel[l, G, gp] * self._innerSum(n, l, gp)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        #return S
        return scSource

    def _evalFluxMoments(self, l, m, gp):
        return 0.25 * np.sum(self.wN * self.quadSet.Ylm[m, l, :] * self.centScFlux[gp, :])

    def _innerSum(self, n, l, gp):
        isum = 0.
        for m in range(0, l + 1):
            isum += np.sum((2 - self.C[m]) * self.quadSet.Ylm[m, l, n] * self._evalFluxMoments(l, m, gp))
        return isum

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
        return 0.25 * scalarFlux


class d2BoundaryElement(object):
    """
    In 2D boundary conditions are specified on a node.
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
        self.nGv = np.arange(self.parent.nG)
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
        commonNodeV = self._computeArrayIntersection(self.parent.nodeVs, self.nodeVs)
        self.internalBCnodeIDs = np.array([np.where(cV == self.parent.nodeVs) for cV in commonNodeV])[:, 0, 0]
        lonelyNodeV = self._computeArrayDiff(self.parent.nodeVs, self.nodeVs)
        # determine outward normal
        deltaX = self.nodeVs[1, 0] - self.nodeVs[0, 0]
        deltaY = self.nodeVs[1, 1] - self.nodeVs[0, 1]
        # we have 2 canidate vectors: (in 3D we would use np.cross to find perp
        # vectors to a surface, but in 2d its simpler than that)
        #c1 = np.array([-deltaX, deltaY])
        #c2 = np.array([deltaX, -deltaY])
        #
        c1 = np.array([-deltaY, deltaX])
        c2 = np.array([deltaY, -deltaX])
        # but only 1 can be the correct outward normal vec (the other is inward
        # normal)
        # where is the lonely node located in relation to the two border nodes?
        t1 = np.array([lonelyNodeV[0, 0] - self.nodeVs[0, 0], lonelyNodeV[0, 1] - self.nodeVs[0, 1]])
        if np.dot(c1, t1) > 0:
            # found it!
            vec = c1
        else:
            # its the other one.
            vec = c2
        vec = vec / np.linalg.norm(vec)   # norm vec
        # given as standard [nu, eta, ksi] set
        self.outwardNormal = np.array([vec[0], vec[1], 0])
        self.outwardNormal[abs(self.outwardNormal) < 1e-4] = 0

    def _computeArrayIntersection(self, a, b):
        """ np.intersect1d wont work for 2d arrays, so this dirty hack does the
        trick """
        av = a.view([('', a.dtype)] * a.shape[1]).ravel()
        bv = b.view([('', b.dtype)] * b.shape[1]).ravel()
        return np.intersect1d(av, bv).view(a.dtype).reshape(-1, a.shape[1])

    def _computeArrayDiff(self, a, b):
        """ np.setdiff1d wont work for 2d arrays, so this dirty hack does the
        trick """
        av = a.view([('', a.dtype)] * a.shape[1]).ravel()
        bv = b.view([('', b.dtype)] * b.shape[1]).ravel()
        return np.setdiff1d(av, bv).view(a.dtype).reshape(-1, a.shape[1])

    def computeInOrds(self):
        self.outOs, self.inOs = self.parent.quadSet.dirCosine(self.outwardNormal)

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
        # TODO: FIX multiple element assignment in 2d!!
        RHS[np.ix_(self.nGv, self.inOs, self.nodeIDs)] = 0
        #RHS[:, self.inOs, self.nodeIDs] = 0   # does not work if len(inOs) != len(nodeIDs)
        return RHS

    def refBC(self, RHS):
        """
        Apply reflected flux dirichletBC.  Take banked fluxes on the outer surface
        and add to the RHS.
        """
        RHS[np.ix_(self.nGv, self.inOs, self.nodeIDs)] = \
            self.bankedRefFlux[np.ix_(self.nGv, self.inOs, self.internalBCnodeIDs)]
        #RHS[:, self.inOs, self.nodeIDs] = self.bankedRefFlux[:, self.inOs, self.internalBCnodeIDs]
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
        if np.allclose(self.outwardNormal, np.array([1, 0, 0])) or np.allclose(self.outwardNormal, np.array([-1, 0, 0])):
            inDirs = np.array(self.parent.quadSet.xzpairs[self.inOs][:, 0], dtype=int)
            outDirs = np.array(self.parent.quadSet.xzpairs[self.inOs][:, 1], dtype=int)
        elif np.allclose(self.outwardNormal, np.array([0, 1, 0])) or np.allclose(self.outwardNormal, np.array([0, -1, 0])):
            inDirs = np.array(self.parent.quadSet.yzpairs[self.inOs][:, 0], dtype=int)
            outDirs = np.array(self.parent.quadSet.yzpairs[self.inOs][:, 1], dtype=int)
        else:
            print("Can only handle boundaries perpendicular to x or y axis at the moment")
            print("Future: add arbitrary bc orientation capability")
            sys.exit()
        #self.bankedRefFlux[:, inDirs, self.internalBCnodeIDs] = \
        #    self.parent.nodeScFlux[:, outDirs, self.internalBCnodeIDs]
        self.bankedRefFlux[np.ix_(self.nGv, inDirs, self.internalBCnodeIDs)] = \
            self.parent.nodeScFlux[np.ix_(self.nGv, outDirs, self.internalBCnodeIDs)]
