#!/usr/bin/python

# Creates 1D and 2D finite element descriptions of geometry
# using the GMSH program.
# Requires gmsh >= 2.8.5
#
# TODO: 3D meshs
#
#
from __future__ import division
from collections import defaultdict
import subprocess
import re
import sys
import numpy as np


# ============================================================================ #
class gmshMesh(object):
    """!
    @brief Parses 1D and 2D inp files from GMSH.
    Only supports triangular elements in 2D!
    """
    def __init__(self, geoFile, inpFileName=None):
        """!
        @param geoFile  Input GMSH compatible geo file.
        @param inpFileName  Custom name for .inp gmsh output (optional)
        """
        if not inpFileName:
            self.inpFileName = geoFile + '.inp'
        else:
            self.inpFileName = inpFileName + '.inp'
        self.geoFile = geoFile
        self.parseGEO()
        self.dg_element_dict = {}

    def runGMSH(self, dim=1):
        self.dim = dim
        print("Constructing the mesh.  Executing GMSH.")
        subprocess.call(['gmsh', str(self.geoFile), '-' + str(dim), '-o', self.inpFileName, '-v', '0'])
        print("Meshing complete.")
        self.inpFL = fileToList(self.inpFileName)
        self.parseINP()
        self.regionNodes()
        self.markRegionBCs()

    def parseGEO(self):
        self.regionInfo = {}
        geoFL = fileToList(self.geoFile)
        reFlags = {'PhysicalEnt': "Physical.+\((\d+)\).+\s+\/\/(.+)=(.+)"}
        for key, val in reFlags.iteritems():
            reFlags[key] = re.compile(val)
        for i, line in enumerate(geoFL):
            check = checkLine(line, reFlags)
            if check:
                regionID = int(check[1].group(1))    # region id (int)
                regionType = check[1].group(2).strip()   # bc or mat
                regionData = check[1].group(3).strip()   # matrial str or bc type
                if regionType == 'mat' or regionType == 'material':
                    self.regionInfo[regionID] = {'type': 'interior', 'info': regionData}
                elif regionType == 'bc' or regionType == 'bound':
                    self.regionInfo[regionID] = {'type': 'bc', 'info': regionData}
                else:
                    sys.exit("FATAL: Unknown region specification in .geo file")
            else:
                pass

    def parseINP(self):
        # Look for keywords
        reFlags = {'Node': "\*Node",
                   'Elm': "\*Element",
                   'ELSET': "\*ELSET"}
        for key, val in reFlags.iteritems():
            reFlags[key] = re.compile(val, flags=re.I)
        # Find locations of keywords in file
        # WARNING: there can be multiple instances of the *ELEMENT keyword
        # due to different mesh element types: tets are C3D6 octs are C3D8
        flaggedDict = defaultdict(list)
        for i, line in enumerate(self.inpFL):
            check = checkLine(line, reFlags)
            if check:
                flaggedDict[check[0]].append({'i': i, 'match': check[1]})
            else:
                pass
        # *Nodes
        self.createNodes(flaggedDict)
        # *Elements
        self.createElements(flaggedDict)
        # *ELSETs
        self.createRegions(flaggedDict)

    def createNodes(self, flaggedDict):
        """
        Round node coordinates and create self.nodes array
        """
        nodes = []
        nodeDefLineStart = flaggedDict['Node'][0]['i'] + 1
        nodeDefLineEnd = flaggedDict['Elm'][0]['i']
        for j, line in enumerate(self.inpFL[nodeDefLineStart: nodeDefLineEnd]):
            words = line.split()
            # fix
            for k, word in enumerate(words[1:]):
                try:
                    words[k + 1] = '%.10e' % round(float(word.strip(',')), 10)
                except:
                    break
                if (k + 1) < 3:
                    words[k + 1] += ','
            try:
                nodes.append([float(x.strip(', ')) for x in words])
            except:
                pass
        self.nodes = np.array(nodes)
        self.nodes[:, 0] -= 1  # fix annoying off by 1 indexing

    def createElements(self, flaggedDict):
        """
        The *Element section contains element IDs.  Each element ID
        contains node IDs which mark the verticies of the element.
        """
        elements = []
        elementDefLineStart = flaggedDict['Elm'][0]['i'] + 1
        elementDefLineEnd = flaggedDict['ELSET'][0]['i']
        for j, line in enumerate(self.inpFL[elementDefLineStart: elementDefLineEnd]):
            words = line.split()
            #words[0] = str(j + 1) + ', '
            try:
                elements.append([int(x.strip(', ')) for x in words])
                if len(elements[-1]) == 3 and self.dim == 2:
                    elements[-1] += [-100]  # add padding to line elements
            except:
                pass
        self.elements = np.array(elements) - 1

    def createRegions(self, flaggedDict):
        """
        At the end of an INP file, *ELSET lines define the element numbers
        making up the region.
        """
        self.regions = {}
        for k, elset in enumerate(flaggedDict['ELSET']):
            elsetDefStart = flaggedDict['ELSET'][k]['i']
            try:
                elsetDefEnd = flaggedDict['ELSET'][k + 1]['i']
            except:
                elsetDefEnd = None
            # perform region type and material assignment for each region
            regionStr = re.match('[^ \d]+(\d+)', self.inpFL[elsetDefStart])
            regionID = int(regionStr.group(1))
            elements = []
            for j, line in enumerate(self.inpFL[elsetDefStart + 1: elsetDefEnd]):
                words = line.split()
                #words[0] = str(j + 1) + ', '
                #elements.append([int(x.strip(', ')) - 1 for x in words])
                elements += [int(x.strip(', ')) - 1 for x in words]
            self.regions[regionID] = {}
            self.regions[regionID]['elementIDs'] = np.array(elements).flatten()
            self.regions[regionID]['type'] = self.regionInfo[regionID]['type']
            if self.regionInfo[regionID]['type'] == 'interior':
                self.regions[regionID]['material'] = self.regionInfo[regionID]['info']
            else:
                self.regions[regionID]['bc'] = self.regionInfo[regionID]['info']

    def regionNodes(self):
        """
        Identify nodes that reside in each region.
        """
        for regionID, region in self.regions.iteritems():
            if region['type'] == 'bc' and self.dim == 1:
                # Degenerate case in 1D.  Boundary element is just made up of
                # one node.
                # A boundary node in 1D is techically not an 'element' therefore
                # has no 'elements'
                self.regions[regionID]['nodeIDs'] = region['elementIDs']
            else:
                regionElementIndexs = np.unique([np.where(self.elements[:, 0] == i) for i in region['elementIDs']])
                #self.regions[regionID]['elements'] = np.array([self.elements[row] for row in regionElementIndexs])
                regionEles = np.array([self.elements[row] for row in regionElementIndexs])
                if regionEles[-1, -1] < 0:
                    regionEles = regionEles[:, :-1]
                self.regions[regionID]['elements'] = regionEles
                self.regions[regionID]['nodeIDs'] = np.unique(self.regions[regionID]['elements'][:, 1:].flatten())
            #self.regions[regionID]['nodes'] = np.take(self.nodes, self.regions[regionID]['nodeIDs'] - 1, axis=0)
            self.regions[regionID]['nodes'] = self.nodes

    def markRegionBCs(self):
        """
        If a region contains boundary nodes, store them.
        """
        # store boundary nodes
        boundaryRegions = []
        for regionID, region in self.regions.iteritems():
            if region['type'] == 'bc':
                boundaryRegions.append([region['bc'], region['nodeIDs']])
        # for each region
        for regionID, region in self.regions.iteritems():
            # check for elements containing boundary nodes
            if region['type'] == 'interior':
                self.regions[regionID]['bcElms'] = {}
                for boundaryRegion in boundaryRegions:
                    # Identify bounday nodes
                    boundingNodes = np.intersect1d(region['nodeIDs'], boundaryRegion[1])
                    # find element(s) to which the boundary nodes corrospond
                    # if dim == 1, we have a single node bound to a single ele
                    # if dim == 2, the boundary _element_ only shares its two
                    # verticies with ONE element
                    boundingEleDict = self.linkBele2Iele(region, boundingNodes)
                    bcType = boundaryRegion[0]
                    if len(boundingNodes) != 0:
                        self.regions[regionID]['bcElms'][bcType] = boundingEleDict
                    else:
                        self.regions[regionID]['bcElms'][bcType] = None

    def linkBele2Iele(self, region, boundingNodes):
        """
        Link boundary element to interior element.
        """
        bEdict = {}
        for element in region['elements']:
            bNodeIDs = np.intersect1d(boundingNodes, element[1:])
            if len(bNodeIDs) == self.dim:
                bEdict[element[0]] = bNodeIDs
            else:
                pass
        return bEdict

    def enable_connectivity(self):
        """!
        @brief Builds shared edges dict and global DG
        element mesh.  Verticies are multiply defined in this scheme.
        Note: Very expensive method (N^2).
        Must be manually called since mesh connectivity is not always desired.
        """
        interior_mesh_elements = []
        for regionID, region in self.regions.iteritems():
            try:
                interior_mesh_elements.append(self.regions[regionID]['elements'])
            except:
                pass
        interior_mesh_elements = np.concatenate(interior_mesh_elements, axis=0)
        dg_element_dict = self._build_global_dg_mesh(interior_mesh_elements)
        # find neighbors
        tmp_edge_list = self._edge_list(dg_element_dict)
        #
        for ele_id, ele in dg_element_dict.iteritems():
            ele_neighbors, ele_edge_neighbors = \
                    self._find_edge_neighbors(ele_id, dg_element_dict, tmp_edge_list)
            dg_element_dict[ele_id]['neighbors'] = {'elements': ele_neighbors,
                                                    'edges': ele_edge_neighbors}
        self.dg_element_dict = dg_element_dict

    def _build_global_dg_mesh(self, interior_mesh_elements):
        """!
        @brief Generates a dictionary of all elements in the mesh.
        Each element is asscoiated with bounding edges and verticies.
        @return dictionary of elements
        """
        self.global_to_gmsh_table = {}
        dg_element_dict = {}
        global_node_id_idx = 0
        edge_id = 0
        for ele in interior_mesh_elements:
            el_id = int(ele[0])
            dg_element_dict[el_id] = {'gmsh_nodeIDs': ele[1:]}
            dg_element_dict[el_id]['vertex_pos'] = []
            for gmsh_node_id in ele[1:]:
                dg_element_dict[el_id]['vertex_pos'].append( \
                        self.nodes[int(np.argwhere(gmsh_node_id == self.nodes[:, 0]))][1:])
            dg_element_dict[el_id]['vertex_pos'] = \
                np.array(dg_element_dict[el_id]['vertex_pos'])
            # dg_element_dict[el_id]['nodePos'] =
            element_global_node_ids = np.zeros(len(ele[1:]), dtype=int)
            local_global_node_ids = np.zeros(len(ele[1:]), dtype=int)
            local_node_id_idx = 0
            for i, nodeID in enumerate(ele[1:]):
                self.global_to_gmsh_table[global_node_id_idx] = nodeID
                element_global_node_ids[i] = global_node_id_idx
                local_global_node_ids[i] = local_node_id_idx
                global_node_id_idx += 1
                local_node_id_idx += 1
            dg_element_dict[el_id]['global_nodeIDs'] = element_global_node_ids
            dg_element_dict[el_id]['local_nodeIDs'] = local_global_node_ids
            dg_element_dict[el_id]['centroid'] = \
                    np.sum(dg_element_dict[el_id]['vertex_pos'], axis=0) / \
                    len(dg_element_dict[el_id]['vertex_pos'])
            # label edges
            dg_element_dict[el_id]['edges'] = {}
            if self.dim == 1:
                for edge_pos, edge_id in zip(dg_element_dict[el_id]['vertex_pos'],
                                             dg_element_dict[el_id]['global_nodeIDs']):
                    dg_element_dict[el_id]['edges'][edge_id] = {'edge_node_ids': (edge_id,),
                                                                'edge_centroid': edge_pos}
                    ele_centroid = dg_element_dict[el_id]['centroid']
                    dg_element_dict[el_id]['edges'][edge_id]['edge_normal'] = \
                            (edge_pos - ele_centroid) / np.linalg.norm(edge_pos - ele_centroid)
                    edge_id += 1
            elif self.dim == 2:
                # TODO: FIX 2D CASE
                for i in range(len(dg_element_dict[el_id]['global_nodeIDs'])):
                    dg_element_dict[el_id]['edges'][edge_id] = {'edge_node_ids': [],
                                                                'edge_centroid': 0.}
                    try:
                        edge_node_id_pair = (dg_element_dict[el_id]['global_nodeIDs'][i], \
                            dg_element_dict[el_id]['global_nodeIDs'][i+1])
                    except:
                        edge_node_id_pair = (dg_element_dict[el_id]['global_nodeIDs'][i], \
                            dg_element_dict[el_id]['global_nodeIDs'][0])
                    dg_element_dict[el_id]['edges']['edge_node_ids'].append(edge_node_id_pair)
                    # dg_element_dict[el_id]['edges']['edge_centroids'].append()
                    edge_id += 1
            else:
                raise RuntimeError("Dim must be 1 or 2")
        return dg_element_dict

    def _edge_list(self, dg_element_dict):
        """!
        @brief Generates a flat edge list view from dict
        @return list of edges
        """
        edge_list = []
        for ele_id, ele in dg_element_dict.iteritems():
            for edge_id, edge in ele['edges'].iteritems():
                edge_list.append([ele_id, edge_id,
                                  edge['edge_centroid'][0],
                                  edge['edge_centroid'][1],
                                  edge['edge_centroid'][2]])
        return np.array(edge_list)

    def _find_edge_neighbors(self, ele_id, dg_element_dict, edge_list):
        """!
        @brief Find the neighbors of a given element by
        shared edge inspection.
        @param ele_id  global element id to inspect for neighbors
        @param dg_element_dict  element dictionary
        @param edge_list  list of edges
        @return neighboring elements, neighboring edges
        """
        edge_mask = (edge_list[:, 0] != ele_id)
        neighbor_list = []
        neighbor_edge_list = []
        for edge_id, edge in dg_element_dict[ele_id]['edges'].iteritems():
            # compute the distance to each other edge centroid in the mesh
            edge_dists = np.linalg.norm(edge_list[edge_mask][:, 2:] - edge['edge_centroid'], axis=1)
            nearest_edge_idx = np.argmin(edge_dists)
            if edge_dists[nearest_edge_idx] == 0.:
                nearest_ele_id = int(edge_list[edge_mask][nearest_edge_idx][0])
                nearest_ele_shared_edge_id = int(edge_list[edge_mask][nearest_edge_idx][1])
                neighbor_list.append(nearest_ele_id)
                neighbor_edge_list.append(nearest_ele_shared_edge_id)
        return neighbor_list, neighbor_edge_list


    @property
    def region_ids(self):
        """!
        @brief Regions ids list
        """
        return list(self.regions.keys())


# ============================================================================ #
class gmsh1DMesh(gmshMesh):
    def __init__(self, **kwargs):
        super(gmsh1DMesh, self).__init__(**kwargs)
        self.runGMSH(1)

    def gmsh1Dparse(self):
        pass


# ============================================================================ #
class gmsh2DMesh(gmshMesh):
    def __init__(self, **kwargs):
        super(gmsh2DMesh, self).__init__(**kwargs)
        self.runGMSH(2)

    def gmsh2Dparse(self):
        pass


# ============================================================================ #
def fileToList(infile):
    '''
    list generator helper function so we only need to read the file
    in once and store its contents in some variable.  No need to
    open and close a file a bunch.

    :param infile:  File name
    :type infile: string
    :returns:  lines (list): list of strings that comprise the file
    '''
    try:
        f = open(infile, 'r')
        lines = [line for line in f]
        f.close()
        return lines
    except:
        return None


def checkLine(line, reFlags):
    """
    Checks lines for flagged strings.  Checks each line
    against regular expession dictionary.
    """
    for flagName, reF in reFlags.iteritems():
        match = reF.match(line)
        if match:
            return (flagName, match)
        else:
            pass
    return None

if __name__ == "__main__":
    Mesh1D = gmsh1DMesh(geoFile='testline2.geo')
    Mesh1D.enable_connectivity()
    pass
