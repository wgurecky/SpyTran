import h5py
import numpy as np


def writeToHdf5(data, fnameout):
    """
    Dumps dictionary of np arrays to h5 data file
    """
    h5f = h5py.File(fnameout, 'w')
    for dataname, dataset in data.iteritems():
        h5f.create_dataset(dataname, data=dataset)
    h5f.close()
