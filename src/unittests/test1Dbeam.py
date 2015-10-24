import unittest
import numpy as np
import sn1D as sn

# Load xs database
import materials.materialMixxer as mx
import utils.pinCellMatCalc as pcm
mx.genMaterialDict('./materials/newXS')
pinMaterial = pcm.createPinCellMat()

# plotters
import plotters.fluxEplot as flxPlt
import plotters.scalarFluxPlot as sfp
import plotters.plotOrdFlux as pof


class test1Dbeam(unittest.TestCase):

    def testAtten(self):
        print("\n========= INITIATING BEAM TEST ==========")
        width, dX = 50.0, 0.4
        sNord = 10
        attnMat = mx.mixedMat({'c12': 1.0})
        attnMat.setDensity(2.24)
        print(attnMat.nDdict)
        mesh1D = sn.Mesh1Dsn([0, width], dX, attnMat, sN=sNord)
        # define fixed boundary cond
        srcStrength = 1.e6  # [n / cm**2-s]
        # energy distribution of source (all born at 0.1MeV
        srcEnergy = np.array([0.0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0])
        bcs = {0: {'fixN': (1, [srcStrength, srcEnergy])},
               -1: {'vac': (2, 0)}}
        mesh1D.setBCs(bcs)
        for si in range(180):
            resid = mesh1D.sweepMesh(1)
            if resid < 1e-5:
                break
        scalarFlux = mesh1D.getScalarFlux()
        for g in range(len(srcEnergy)):
            sfp.plot1DScalarFlux(scalarFlux[:][:, g], np.arange(0, width + dX, dX), label='Group ' + str(g + 1))
            sfp.plot1DNeutronND(scalarFlux[:][:, g], np.arange(0, width + dX, dX), g)
        flxPlt.plotFluxE(scalarFlux[-1][::-1])  # flux vs E at left edge
        # plot ord fluxes at leading edge
        ordFlux = mesh1D.getOrdFlux()
        angles = np.arccos(mesh1D.cells[0].sNmu)
        g = 2
        mag = ordFlux[1][g, 0, :] / sum(ordFlux[1][g, 0, :])
        pof.compass(angles, mag, figName='polar_grp3')
        # plot ord fluxes at mid plane
        g = 3
        mag = ordFlux[1][g, 0, :] / sum(ordFlux[1][g, 0, :])
        pof.compass(angles, mag, figName='polar_grp4')


if __name__ == "__main__":
    unittest.main()
