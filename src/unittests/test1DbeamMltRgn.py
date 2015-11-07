import unittest
import numpy as np
import sn1D as sn

# Load xs database
import materials.materialMixxer as mx
mx.genMaterialDict('./materials/newXS')

# plotters
import plotters.fluxEplot as flxPlt
import plotters.scalarFluxPlot as sfp
import plotters.plotOrdFlux as pof


class test1Dbeam(unittest.TestCase):

    def testAtten(self):
        print("\n========= INITIATING MULT REGION BEAM TEST ==========")
        sNord = 18
        # define fixed source boundary cond
        srcStrength = 1.e6  # [n / cm**2-s]
        # energy distribution of source (all born at 0.1MeV
        srcEnergy = np.array([1.0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0])
        #
        # ## REGION 1 - GRAPHITE ###
        width1, dX1 = 25.0, 1.0
        region1Mat = mx.mixedMat({'c12': 1.0})
        region1Mat.setDensity(1.96)
        region1mesh1D = sn.Mesh1Dsn([0, width1], dX1, region1Mat, sN=sNord)
        bcs1 = {0: {'fixN': (1, [srcStrength, srcEnergy])}}
        region1mesh1D.setBCs(bcs1)
        # ## REGION 2 - BORATED CARBON ###
        width2, dX2 = 25.0, 0.05
        region2Mat = mx.mixedMat({'c12': 0.99, 'b10': 0.01})
        region2Mat.setDensity(1.9)
        region2mesh1D = sn.Mesh1Dsn([width1 + dX1 / 2. + dX2 / 2., width1 + dX1 / 2. + dX2 / 2. + width2], dX2, region2Mat, sN=sNord)
        bcs2 = {-1: {'vac': (2, 0)}}
        region2mesh1D.setBCs(bcs2)
        #
        # ## BUILD DOMAIN ###
        domain = sn.SubDomain()
        domain.addRegion(region1mesh1D)
        domain.addRegion(region2mesh1D)
        domain.buildSweepTree()
        #
        # ## SWEEP DOMAIN ###
        for si in range(180):
            resid = domain.sweepSubDomain(1)
            if resid < 1e-5:
                break
        scalarFlux = domain.getScalarFlux()
        for g in range(len(srcEnergy)):
            pass
        flxPlt.plotFluxE(scalarFlux[-1][::-1])  # flux vs E at left edge
        centroids = [cell.centroid for cell in domain.regions[0].cells]
        centroids += [cell.centroid for cell in domain.regions[1].cells]
        for g in range(len(srcEnergy)):
            sfp.plot1DScalarFlux(scalarFlux[:][:, g], centroids, label='Group ' + str(g + 1))
            # sfp.plot1DNeutronND(scalarFlux[:][:, g], centroids, g)
        # plot ord fluxes at leading edge
        ordFlux = domain.getOrdFlux()
        angles = np.arccos(domain.regions[0].cells[0].sNmu)
        g = 2
        mag = ordFlux[1][g, 0, :] / sum(ordFlux[1][g, 0, :])
        pof.compass(angles, mag, figName='polar_grp3')
        # plot ord fluxes at mid plane
        g = 3
        mag = ordFlux[1][g, 0, :] / sum(ordFlux[1][g, 0, :])
        pof.compass(angles, mag, figName='polar_grp4')


if __name__ == "__main__":
    unittest.main()
