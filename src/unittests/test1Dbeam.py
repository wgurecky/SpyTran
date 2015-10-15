import unittest
import numpy as np
import sn1D as sn

# Load xs database
import materials.materialMixxer as mx
import utils.pinCellMatCalc as pcm
mx.genMaterialDict('./materials/hw2')
pinMaterial = pcm.createPinCellMat()

# plotters
import plotters.fluxEplot as flxPlt
import plotters.scalarFluxPlot as sfp


class test1Dbeam(unittest.TestCase):

    def testAtten(self):
        print("\n========= INITIATING BEAM TEST ==========")
        width, dX = 40.0, 1.0
        sNord = 4
        #attnMat = mx.mixedMat({'c12': 0.99999, 'b10': 0.00001})
        attnMat = mx.mixedMat({'c12': 1.0})
        #attnMat = mx.mixedMat({'h1': 2.0 / 3., 'o16': 1. / 3.})
        attnMat.setDensity(2.26)
        #attnMat.setDensity(1.0)
        mesh1D = sn.Mesh1Dsn([0, width], dX, attnMat, sN=sNord)
        # define fixed boundary cond
        srcStrength = 1.e6  # [n / cm**2-s]
        # energy distribution of source (all born at 0.1MeV
        srcEnergy = np.array([0.0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0])
        bcs = {0: {'fixN': (1, [srcStrength, srcEnergy])},
               -1: {'vac': (2, 0)}}
        mesh1D.setBCs(bcs)
        for si in range(180):
            # print("Scattering Iteration: " + str(si))
            resid = mesh1D.sweepMesh(30)
            if resid < 1e-5:
                break
        scalarFlux = mesh1D.getScalarFlux()
        for g in range(len(srcEnergy)):
            sfp.plot1DScalarFlux(scalarFlux[:][:, g], np.arange(0, width + dX, dX), label='Group ' + str(g + 1))
            sfp.plot1DNeutronND(scalarFlux[:][:, g], np.arange(0, width + dX, dX), g)
        flxPlt.plotFluxE(scalarFlux[-1][::-1])  # flux vs E at left edge


if __name__ == "__main__":
    unittest.main()
