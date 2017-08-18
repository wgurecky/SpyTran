import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl
from d1.elements import d1InteriorElement
from d1.elements import d1BoundaryElement
from d2.elements import d2InteriorElement
from d2.elements import d2BoundaryElement
np.set_printoptions(linewidth=200)  # set print to screen opts


class SuperMesh(object):
    """
    Contains all region meshes.
    Contains mappings betwen array/matrix field representation and element class
    representation.
    """
    def __init__(self, gmshMesh, materialDict, bcDict, srcDict, nG, sNords, quadSet, dim=1):
        self.nG, self.sNords = nG, sNords
        self.nNodes = gmshMesh.total_dg_nodes
        self.sysRHS = np.zeros((self.nG, self.sNords, self.nNodes))        # source vector
        self.scFluxField = np.zeros((self.nG, self.sNords, self.nNodes))   # scattered flux field
        self.totFluxField = np.zeros((self.nG, self.sNords, self.nNodes))  # total flux field
        fluxStor = (self.scFluxField, self.totFluxField)
        self.regions = {}     # mesh subregion dictionary
        for regionID, gmshRegion in gmshMesh.regions.iteritems():
            if gmshRegion['type'] == 'interior':
                self.regions[regionID] = RegionMesh(gmshRegion, fluxStor, materialDict[gmshRegion['material']],
                                                    bcDict, srcDict.get(gmshRegion['material'], None),
                                                    nGroups=self.nG, sNords=self.sNords, quadSet=quadSet, dim=dim)
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

    def buildSysMatrix(self, depth):
        self.sysA = np.empty((self.nG, self.sNords), dtype=sps.lil.lil_matrix)
        self.sysP = np.empty((self.nG, self.sNords), dtype=object)
        for g in range(self.nG):
            for o in range(self.sNords):
                self.sysA[g, o] = self.constructA(g, o)
                if depth <= 1:
                    self.computePrecon(g, o)
                if depth == 1:
                    self.sysA[g, o] = sps.csc_matrix(self.sysA[g, o])

    def computePrecon(self, g, o):
        M_x = lambda x: spl.spsolve(self.sysA[g, o] * sps.eye(self.nNodes), x)
        self.sysP[g, o] = spl.LinearOperator((self.nNodes, self.nNodes), M_x)

    def constructA(self, g, o):
        A = sps.lil_matrix((self.nNodes, self.nNodes))
        for regionID, region in self.regions.iteritems():
            A = region.buildRegionA(A, g, o)
        return A

    def sweepFlux(self, tolr):
        """
        For each angle and energy, solve a system of linear equations
        to update the flux scalar field on the mesh.
        """
        innerResid, gmres_status = 0, 0
        for g in range(self.nG):
            for o in range(self.sNords):
                self.scFluxField[g, o], gmres_status = \
                    spl.gmres(self.sysA[g, o], self.sysRHS[g, o], tol=tolr, M=self.sysP[g, o])
                if gmres_status > 0:
                    print("WARNING: Linear system solve failed.  Terminated at gmres iter: " + str(gmres_status))
        self.totFluxField += self.scFluxField
        for regionID, region in self.regions.iteritems():
            fluxStor = (self.scFluxField, self.totFluxField)
            region.updateEleFluxes(fluxStor)
        return np.linalg.norm(self.scFluxField) / np.linalg.norm(self.totFluxField), innerResid

    def applyBCs(self, depth):
        for regionID, region in self.regions.iteritems():
            self.sysA, self.sysRHS = region.setBCs(self.sysA, self.sysRHS, depth)

    def initFlux(self, scFactor):
        fluxStor = (self.scFluxField, (0.0 * self.totFluxField + 1.0) * scFactor)
        for regionID, region in self.regions.iteritems():
            region.updateEleFluxes(fluxStor)

    def getFissionSrc(self):
        fissionSrc = 0
        for regionID, region in self.regions.iteritems():
            fissionSrc += region.getFissionSrc()
        return fissionSrc

    def resetMeshFlux(self):
        self.scFluxField = np.zeros((self.nG, self.sNords, self.nNodes))
        self.totFluxField = np.zeros((self.nG, self.sNords, self.nNodes))

    @property
    def global_node_list(self):
        global_node_list = []
        for regionID, region in self.regions.iteritems():
            global_node_list.append(region.node_list)
        return np.vstack(tuple(global_node_list))


class RegionMesh(object):
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
        self.gmshRegion = gmshRegion
        self.dim = kwargs.get("dim")
        self.nG = kwargs.get("nGroups")
        self.bcDict = bcDict
        self.totalXs = material.macroProp['Ntotal']
        self.skernel = material.macroProp['Nskernel']
        if 'chi' in material.macroProp.keys():
            self.nuFission = material.macroProp['Nnufission']
            self.chiNuFission = np.dot(np.array([material.macroProp['chi']]).T,
                                       np.array([material.macroProp['Nnufission']]))
            source = 'fission'
        else:
            self.nuFission = np.zeros(self.nG)
            self.chiNuFission = None
            #source = kwargs.pop("source", None)
        # Build elements in the region mesh
        self.buildElements(gmshRegion, fluxStor, source, **kwargs)
        self._gen_node_list(gmshRegion, **kwargs)
        self.linkBoundaryElements(gmshRegion)

    def _gen_node_list(self, gmshRegion, **kwargs):
        self.node_list = []
        for element in gmshRegion['elements']:
            nodeIDs = element[1:]
            gmsh_dg_element = gmshRegion['dg_elements'][int(element[0])]
            global_nodeIDs = gmsh_dg_element['global_nodeIDs']
            global_nodePos = gmsh_dg_element['vertex_pos']
            element_centroid = gmsh_dg_element['centroid']
            for glb_id, glb_pos in zip(global_nodeIDs, global_nodePos):
                self.node_list.append(
                        [glb_id] + list(element_centroid) + list(glb_pos)
                        )
        self.node_list = np.array(self.node_list)

    def buildElements(self, gmshRegion, fluxStor, source, **kwargs):
        """
        Initilize and store interior elements.
        """
        self.elements = {}
        for element in gmshRegion['elements']:
            nodeIDs = element[1:]
            gmsh_dg_element = gmshRegion['dg_elements'][int(element[0])]
            global_nodeIDs = gmsh_dg_element['global_nodeIDs']
            if self.dim == 1:
                global_nodePos = gmsh_dg_element['vertex_pos'][:, 0]
                self.elements[element[0]] = d1InteriorElement((global_nodeIDs, global_nodePos), fluxStor, source, gmsh_dg_element, **kwargs)
            else:
                global_nodePos = gmsh_dg_element['vertex_pos'][:, 0:2]
                self.elements[element[0]] = d2InteriorElement((global_nodeIDs, global_nodePos), fluxStor, source, gmsh_dg_element, **kwargs)

    def linkBoundaryElements(self, gmshRegion):
        """
        Store boundary elements that border this region.  Link the interior element
        with its corrosponding boundary element
        """
        self.belements = {}  # boundary element dict (empty if subregion contains no boundaries)
        for bctype, bcElms in gmshRegion['bcElms'].iteritems():
            if type(bcElms) is dict:
                for bcElmID, nodeIDs in bcElms.iteritems():
                    dg_element_node = gmshRegion['dg_elements'][bcElmID]
                    global_nodeIDs, global_nodePos = [], []
                    for global_id, gmsh_node_id, gmsh_node_pos in zip(dg_element_node['global_nodeIDs'],
                                                           dg_element_node['gmsh_nodeIDs'],
                                                           dg_element_node['vertex_pos']):
                        if gmsh_node_id in nodeIDs:
                            global_nodeIDs.append(global_id)
                            if self.dim == 1:
                                global_nodePos.append(gmsh_node_pos[0])
                            elif self.dim == 2:
                                global_nodePos.append(gmsh_node_pos[0:2])
                    if self.dim == 1:
                        self.belements[bcElmID] = d1BoundaryElement(self.bcDict[bctype], (global_nodeIDs,
                                                                    global_nodePos), self.elements[bcElmID])
                    else:
                        self.belements[bcElmID] = d2BoundaryElement(self.bcDict[bctype], (global_nodeIDs,
                                                                    np.array(global_nodePos)), self.elements[bcElmID])

    def buildRegionA(self, A, g, o):
        """
        @breif Populate matrix A for group g and ordinate o
        for nodes in this region.
        This must only be done once as the system matrix A does not depend on the flux.
        Since A is very sparse we use scipy's sparse matrix class.
        @param A  scipy.sparse.lil_matrix  A row-based linked list sparse
            matrix so that modifying the elements is rather fast.
            later this is converted to a
            scipy.sparse.csc_matrix before solving linear system.
        @param g  int. energy group.
        @param o  int.  discrete ordinate id.
        @return A  filled system A matrix
        """
        n_boundary_elements = 0
        n_interior_elements = 0
        for elementID, element in self.elements.iteritems():
            nodeIDs, sysVals = element.getElemMatrix(g, o, self.totalXs)
            neighbor_nodeIDs, neighbor_sysVals = element.getNeighborMatrix(g, o, self.totalXs)
            if len(neighbor_nodeIDs) == 1:
                n_boundary_elements += 1
            elif len(neighbor_nodeIDs) == 2:
                n_interior_elements += 1
            for nodeID, sysVal in zip(nodeIDs, sysVals):
                A[nodeID] += sysVal
            for neighbor_nodeID, neighbor_sysVal in zip(neighbor_nodeIDs, neighbor_sysVals):
                A[neighbor_nodeID] += neighbor_sysVal
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

    def getFissionSrc(self):
        fissionSrc = 0
        for elementID, element in self.elements.iteritems():
            for g in range(self.nG):
                if self.dim == 1:
                    fissionSrc += self.nuFission[g] * element.deltaX * element._evalCentTotAngleInt(g)
                elif self.dim == 2:
                    fissionSrc += self.nuFission[g] * element.area * element._evalCentTotAngleInt(g)
        return fissionSrc
