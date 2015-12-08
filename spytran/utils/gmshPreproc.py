#!/usr/bin/python

# Creates 1D and 2D finite element descriptions of geometry
# using the GMSH program.
# Requires gmsh >= 2.8.5
#
# TODO: 3D meshs
#
#

from collections import defaultdict
import subprocess
import re
import sys
import numpy as np


class gmshMesh(object):
    def __init__(self, geoFile, inpFileName=None):
        if not inpFileName:
            self.inpFileName = geoFile + '.inp'
        else:
            self.inpFileName = inpFileName + '.inp'
        self.geoFile = geoFile
        self.parseGEO()

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
                self.regions[regionID]['elements'] = np.array([self.elements[row] for row in regionElementIndexs])
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


class gmsh1DMesh(gmshMesh):
    def __init__(self, **kwargs):
        super(gmsh1DMesh, self).__init__(**kwargs)
        self.runGMSH(1)

    def gmsh1Dparse(self):
        """
        Create regions and elements from .inp file gmsh output file
        Each region should be ascossiated with a material using the
        .geo input file.
        """
        pass


class gmsh2DMesh(gmshMesh):
    def __init__(self, **kwargs):
        super(gmsh2DMesh, self).__init__(**kwargs)
        self.runGMSH(2)

    def gmsh2Dparse(self):
        pass


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
