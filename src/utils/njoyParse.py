#!/usr/bin/python2.7
#
# Parses njoy output for group constants
#
# Output:
#
# TOTAL
#
# FFACTOR (= (sig(b)/sig(inf))
#
# CHI
#
# NUFISSION
#
# SKERNEL
#
#
# NOTES:
# group struct is ordered lowest energy first.  (njoy format)
#
#

import re
import numpy as np


def readNJOY(inFileName, outFileName='parseout'):
    '''
    '''
    propsDict = {}
    reFlags = {'total_dilution': ".*mf\s+3.*mt\s+1\s+",
               'skernel': ".*mf\s+6.*mt\s+2\s+",
               'fission': ".*mf\s+3.*mt\s+18\s+",
               'chi': ".*mf\s+6.*mt\s+18\s+",
               'nu': ".*mf\s+3.*mt452\s+"
               }
    for key, val in reFlags.iteritems():
        reFlags[key] = re.compile(val)
    fileLineList = fileToList(inFileName)
    for i, line in enumerate(fileLineList):
        check = checkLine(line, reFlags)
        if check:
            # match success, run parser on table
            # every table block begins with a line starting with the word
            # "group".
            # read from here untill we hit our second non-number containing line
            propsDict[check[0]] = getDataTable(fileLineList[i+1:])
    xsWrite(propsDict, outFileName)


def getDataTable(fileLines):
    blankLines, tableRows, tableStart = 0, [], False
    for line in fileLines:
        if line.strip() == '' and tableStart is False:
            pass
        elif tableStart is False:
            if line.split()[0] == 'group':
                tableStart = True
            else:
                pass
        elif tableStart is True:
            # start parsing table
            if line.strip() == '':
                blankLines += 1
            else:
                tableRows.append(line.strip())
            if blankLines == 2:
                break
    return convertTableRows2np(tableRows)


def convertTableRows2np(tableRows):
    table = []
    try:
        for row in tableRows:
            tablerow = list(np.fromstring(sciNoteFix(row.strip()), sep=' '))
            if tablerow == [-1]:
                pass
            else:
                table.append(tablerow)
        if table != []:
            return np.array(table)
        else:
            # if table is empty, this is expected for the chi table, which is
            # formatted differently in the njoy output than the others.
            for row in tableRows:
                tablerow = list(np.fromstring(sciNoteFix(row[16:].strip()), sep=' '))
                if tablerow == [-1]:
                    pass
                else:
                    table.append(tablerow)
            if table is None:
                print("Warning: Table conversion failure. Blank table.")
            else:
                return np.array(table[0] + table[1])
    except:
        # we have a str lurking in the table
        print("Warning: Failure to convert the table into a numpy array")


def sciNoteFix(numberStr):
    pattern = r'\d\+'
    repl = r'e+'
    numberStr = re.sub(pattern, repl, numberStr)
    pattern = r'\d\-'
    repl = r'e-'
    return re.sub(pattern, repl, numberStr)


def xsWrite(propsDict, outFileName):
    f = open(outFileName, 'w')
    for prop, table in propsDict.iteritems():
        if prop == 'total_dilution':
            j = 0
            f.write('TOTAL \n')
            for i, tot in enumerate(table):
                if tot[1] == 0:
                    # only take 0th order leg here
                    f.write(str(j + 1) + "  " + str("%.4e" % tot[2]) + "\n")
                    j += 1
            f.write('\n')
            f.write('FFACTOR \n')
            j = 0
            for i, tot in enumerate(table):
                if tot[1] == 0:
                    sig_infty = tot[2]
                    f.write(str(j + 1) + "  ")
                    for sig_b in tot[3:]:
                        f.write("  " + str("%.4e" % (sig_b / sig_infty)))
                    j += 1
                    f.write('\n')
            f.write('\n')
        elif prop == 'chi':
            f.write('CHI \n')
            for i, chi in enumerate(table):
                f.write(str(i + 1) + "  " + str("%.4e" % chi) + "\n")
            f.write('\n')
        elif prop == 'fission':
            f.write('NUFISSION \n')
            for i, fission in enumerate(table):
                f.write(str(i + 1) + "  " + str("%.4e" % (table[i][1] * propsDict['nu'][i][1])) + "\n")
            f.write('\n')
        elif prop == 'skernel':
            lord = max([row[2] for row in propsDict['skernel']])
            f.write('SKERNEL \n')
            for i, entry in enumerate(table):
                if entry[2] == 0:
                    for l in range(int(lord) + 1):
                        if l == 0:
                            f.write(str(int(table[i + l][0])) + "  " + str(int(table[i + l][1])) + "  " +
                                    str("%.4e" % table[i + l][3]) + "  ")
                        else:
                            f.write(str("%.4e" % table[i + l][3]) + "  ")
                    f.write('\n')
                else:
                    pass
            f.write('\n')
    f.close()


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
    readNJOY('njoyout_9228', 'u235.xs')
    readNJOY('njoyout_2631', 'fe56.xs')
