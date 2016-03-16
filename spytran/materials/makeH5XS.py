#!/usr/bin/python
import h5py
import argparse
import readXSdatabase as rxs
import os
dir = os.path.dirname(os.path.abspath(__file__))

###
# Given a directory, this script will look for all
# .xs files.  It uses the readXS.py script to parse
# the .xs files and converts them to a single HDF5 file
###

def loadXS(inpath):
    return rxs.readXSFolder(inpath)

if __name__ == "__main__":
    """
    Converts .XS txt files to hdf5
    """
    parser = argparse.ArgumentParser(description='Creates HDF5 XS file from txt based .XS files')
    parser.add_argument('-i', type=str, help='input directory path')
    parser.add_argument('-o', type=str, help='output hdf5 xs file name')
    args = parser.parse_args()
    outfile, inpath = "", ""
    if args.o:
        outfile = args.o
    else:
        outfile = "xsdefault.h5"
    if args.i:
        inpath = args.i
    else:
        inpath = os.path.join(dir, "newXS")

    h5f = h5py.File(outfile, 'w')
    xsdir = loadXS(inpath)
    for isoName, isoXSdata in xsdir.iteritems():
        grp = h5f.create_group(str(isoName))
        for dataName, dataXS in isoXSdata.iteritems():
            grp.create_dataset(str(dataName), data=dataXS)
    h5f.close()
