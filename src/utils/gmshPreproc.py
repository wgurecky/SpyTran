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
        subprocess.call(['gmsh', str(self.geoFile), '-' + str(dim), '-o', self.inpFileName, '-v', '0'])
        self.inpFL = fileToList(self.inpFileName)

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
            reFlags[key] = re.compile(val)
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
        ROUND NODES COORDINATES and create self.nodes array
        """
        nodeDefLineStart = flaggedDict['Node'][0]['i'] + 1
        nodeDefLineEnd = flaggedDict['Elm'][0]['i']
        for j, line in enumerate(self.inpFL[nodeDefLineStart: nodeDefLineEnd]):
            words = line.split()
            # fix
            for k, word in enumerate(words[1:]):
                words[k + 1] = '%.10e' % round(float(word.strip(',')), 10)
                if (k + 1) < 3:
                    words[k + 1] += ','
            self.nodes.append([float(x.strip(', ')) for x in words])
        self.nodes = np.array(self.nodes)

    def createElements(self, flaggedDict):
        """
        The *Element section contains element IDs.  Each element ID
        contains node IDs which mark the verticies of the element.
        """
        elementDefLineStart = flaggedDict['Elm'][0]['i'] + 1
        elementDefLineEnd = flaggedDict['ELSET'][0]['i']
        for j, line in enumerate(self.inpFL[elementDefLineStart: elementDefLineEnd]):
            words = line.split()
            words[0] = str(j + 1) + ', '
            try:
                self.elements.append([int(x.strip(', ')) for x in words])
            except:
                pass
        self.elements = np.array(self.elements)

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
                elsetDefEnd = -1
            # perform region type and material assignment for each region
            regionStr = re.match('[^ \d]+(\d+)', self.inpFL(elsetDefStart))
            regionID = int(regionStr.group(1))
            elements = []
            for j, line in enumerate(self.inpFL[elsetDefStart + 1: elsetDefEnd]):
                words = line.split()
                words[0] = str(j + 1) + ', '
                elements.append([int(x.strip(', ')) for x in words])
            self.regions[regionID]['elementIDs'] = np.array(elements)
            self.regions[regionID]['type'] = self.regionInfo[regionID]['type']
            if self.regionInfo[regionID]['type'] == 'interior':
                self.regions[regionID]['material'] = self.regionInfo[regionID]['info']
            else:
                self.regions[regionID]['bc'] = self.regionInfo[regionID]['info']

    def regionNodes(self):
        for regionID, region in self.regions.iteritems():
            if region['type'] == 'bc' and self.dim == 1:
                self.regions[regionID]['nodeIDs'] = region['elementIDs']
            else:
                regionElementIndexs = np.unique([np.where(self.elements[:, 0] == i) for i in region['elementIDs']])
                self.regions[regionID]['elements'] = np.array([self.elements[row] for row in regionElementIndexs])
                self.regions[regionID]['nodeIDs'] = self.regions[regionID]['elements'][:, 1:]

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
                self.regions[regionID]['bcNodes'] = {}
                for boundaryRegion in boundaryRegions:
                    boundingNodes = np.intersect1d(region['nodeIDs'].flatten(), boundaryRegion[1].flatten())
                    bcType = boundaryRegion[0]
                    if boundingNodes.any():
                        self.regions[regionID]['bcNodes'][bcType] = boundingNodes
                    else:
                        self.regions[regionID]['bcNodes'][bcType] = None


class gmsh1DMesh(gmshMesh):
    def __init__(self, **kwargs):
        super(gmsh1DMesh, self).__init__(**kwargs)
        self.runGMSH(1)
        self.parseINP()

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
