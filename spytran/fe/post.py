import numpy as np
import h5py
import utils.hdf5dump as h5d
from plotters import scalarFluxPlot as sfp


class Fe1DOutput(object):
    def __init__(self, dataFile):
        f = h5py.File(dataFile, 'r')
        self.nodes = f['nodes']
        self.ordFlux = f['ordFluxes']
        self.wN = f['weights']
        self.nG = f['nGrp'].value
        self.nNodes = self.ordFlux.shape[2]
        self.computeAngleIntFlux()
        self.computeTotFlux()

    def computeAngleIntFlux(self):
        """
        Integrate over angle.  Requires ordinate weights.
        """
        self.angleIntFlux = np.zeros((self.ordFlux.shape[0], self.ordFlux.shape[2]))
        for g in range(self.nG):
            for i in range(self.nNodes):
                self.angleIntFlux[g, i] = 0.5 * np.sum(self.wN * self.ordFlux[g, :, i])

    def computeTotFlux(self):
        """
        Sum over all energy groups
        """
        self.totFlux = np.sum(self.angleIntFlux, axis=0)

    def genFluxTable(self, fname="1dFEhdfoutput.h5"):
        """
        Create position vs flux table. Human readable / paste into excel
        """
        h5data = {}
        for g in range(10):
            plotData = np.array([self.nodes[:, 1], self.angleIntFlux[g]])
            plotData = plotData[:, np.argsort(plotData[0])]
            h5data["mesh" + str(g)] = plotData[0]
            h5data["groupFlx" + str(g)] = plotData[1]
        h5d.writeToHdf5(h5data, fname)

    def plotScalarFlux(self, g, fname='scflx'):
        """
        Plots 1D scalar flux for grp g
        """
        plotData = np.array([self.nodes[:, 1], self.angleIntFlux[g]])
        plotData = plotData[:, np.argsort(plotData[0])]
        sfp.plot1DScalarFlux(plotData[1], plotData[0], label='G' + str(g), fnameOut=fname)

    def plotTotalFlux(self, fname='totflx'):
        plotData = np.array([self.nodes[:, 1], self.totFlux])
        plotData = plotData[:, np.argsort(plotData[0])]
        sfp.plot1DScalarFlux(plotData[1], plotData[0], label='tot', fnameOut=fname)


class Fe2DOutput(object):
    def __init__(self, dataFile):
        f = h5py.File(dataFile, 'r')
        self.nodes = f['nodes']
        self.ordFlux = f['ordFluxes']
        self.wN = f['weights']
        self.nG = f['nGrp'].value
        self.nNodes = self.ordFlux.shape[2]
        self.computeAngleIntFlux()
        self.computeTotFlux()

    def computeAngleIntFlux(self):
        """
        Integrate over angle.  Requires ordinate weights.
        """
        self.angleIntFlux = np.zeros((self.ordFlux.shape[0], self.ordFlux.shape[2]))
        for g in range(self.nG):
            for i in range(self.nNodes):
                self.angleIntFlux[g, i] = 0.25 * np.sum(self.wN * self.ordFlux[g, :, i])

    def computeTotFlux(self):
        """
        Sum over all energy groups
        """
        self.totFlux = np.sum(self.angleIntFlux, axis=0)

    def writeToVTK(self, fname):
        from pyevtk.hl import pointsToVTK
        pointsToVTK(fname, self.nodes[:, 1], self.nodes[:, 2], self.nodes[:, 3],
                    data={"grp1": self.angleIntFlux[0, :],
                          "grp2": self.angleIntFlux[1, :],
                          "grp3": self.angleIntFlux[2, :],
                          "grp4": self.angleIntFlux[3, :],
                          "grp5": self.angleIntFlux[4, :],
                          "grp6": self.angleIntFlux[5, :],
                          "grp7": self.angleIntFlux[6, :],
                          "grp8": self.angleIntFlux[7, :],
                          "grp9": self.angleIntFlux[8, :],
                          "grp10": self.angleIntFlux[9, :],
                          "tot": self.totFlux[:]
                          }
                    )
