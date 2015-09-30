import numpy as np
import re


def readXS(inFileName):
    '''
    Populate a dictionary of material properties given a
    file formatted in a basic xs. format.
    '''
    #propsDict = {'total': None,
    #             'chi': None,
    #             'nufission': None,
    #             'skernel': None
    #             }
    propsDict = {}
    reFlags = {'total': "TOTAL",
               'chi': "CHI",
               'nufission': "NUFISSION",
               'skernel': "SKERNEL"
               }
    for key, val in reFlags.iteritems():
        reFlags[key] = re.compile(val)
    fileLineList = fileToList(inFileName)
    for i, line in enumerate(fileLineList):
        check = checkLine(line, reFlags)
        if check:
            propsDict[check[0]] = table2NP(i, fileLineList)
    # Restructure skernel table into matrix format
    # Do all materials have skernel? yes, but need a check here.
    if 'skernel' in propsDict.keys():
        propsDict['skernel'] = matrixSkernel(propsDict['skernel'], len(propsDict['total']))
        propsDict['skernel'] = np.transpose(propsDict['skernel'], axes=[0, 2, 1])
    # Trim and flip arrays (NJOY group ordering fuckerry)
    # group 1 in NJOY is slowest group.
    # Reorder so group 1 is fastest group (traditional way)
    if 'chi' in propsDict.keys():
        propsDict['chi'] = np.flipud(propsDict['chi'][:, 1])
    if 'total' in propsDict.keys():
        propsDict['total'] = np.flipud(propsDict['total'][:, 1])
    if 'nufission' in propsDict.keys():
        propsDict['nufission'] = np.flipud(propsDict['nufission'][:, 1])
    return propsDict


def matrixSkernel(skernelTable, Ngroups):
    """
    takes number of groups (len(totxs_array)) and skernel
    table from ascii file.
    Generates a g x g x len(legendreM) numpy array
    """
    NlegMoments = np.shape(skernelTable)[1] - 2
    skernelMatrix = np.zeros((NlegMoments, Ngroups, Ngroups))
    for row in skernelTable:
        skernelMatrix[:, int(row[0]) - 1, int(row[1]) - 1] = row[2:]
    return skernelMatrix


def table2NP(i, fileLines):
    '''
    Takes line number and list of lines.  Starting from line i,
    reads contiguous table to numpy array.
    '''
    table = []
    for line in fileLines[i+1:]:
        npArray = np.fromstring(line.strip(), sep=' ')
        if len(npArray) == 1 or len(npArray) == 0:
            break
        else:
            table.append(np.fromstring(line.strip(), sep=' '))
    return np.array(table)


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
        print("WARNING: Input file does not exist")
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
    """
    Test functionality on simple xs file
    """
    data = readXS('hw2/u235.xs')
    print(data)
