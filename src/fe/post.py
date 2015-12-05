import numpy as np
import h5py
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

    def genFluxTable(self):
        """
        Create position vs flux table. Human readable / paste into excel
        """
        pass

    def plotScalarFlux(self, g):
        """
        Plots 1D scalar flux for grp g
        """
        plotData = np.sort(np.array([self.nodes[:, 1], self.angleIntFlux[g]]), axis=1)
        sfp.plot1DScalarFlux(plotData[1], plotData[0], label='G'+str(g))

    def plotTotalFlux(self):
        plotData = np.sort(np.array([self.nodes[:, 1], self.totFlux]), axis=1)
        sfp.plot1DScalarFlux(plotData[1], plotData[0], label='tot')
