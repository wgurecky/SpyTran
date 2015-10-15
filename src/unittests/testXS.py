import unittest
import numpy as np
# Load xs database
import materials.materialMixxer as mx
mx.genMaterialDict('./materials/testXS')

# plotters
import plotters.xsPlot as xsplt


class test1Dbeam(unittest.TestCase):

    def testXSplot(self):
        """ Test ability to read XS file, and
        plot xs at multiple dilutions.
        """
        feMat = mx.mixedMat({'fe56': 1.0})
        feMat.setDensity(8.0)
        ffactors = feMat.microDat['fe56']['ffactor']
        totxs = feMat.microDat['fe56']['total']
        # plot at infinite dilution
        xsplt.xsPlot(totxs[::-1], label='Fe56 inf dilution')
        # plot at 10e-1 dilution
        totxs_ssheild = totxs * ffactors[:, 5]
        xsplt.xsPlot(totxs_ssheild[::-1], label='Fe56 10e-1 dilution', style='--')
        # Verify thermal XS is close to true val
        xsAt00253ev = 14.96
        print("Thermal XS diff: " + str(xsAt00253ev - totxs_ssheild[-2]) + " [b]")
        xsplt.xsPlot(np.ones(10) * xsAt00253ev, label='XS at 0.0253eV [KAERI]')

    def testSSheild(self):
        testMat = mx.mixedMat({'fe56': 0.5, 'u235': 0.5})
        testMat.setDensity(10.0)
        testMat._computeBackgroundXsec()
        print("Ffactors for fe56")
        print(testMat.microDat['fe56']['f'])
        print("Ffactors for u235")
        print(testMat.microDat['u235']['f'])
        ssTestMat = testMat.selfSheild()
        print("Self-sheilded Macro cross sections [1/cm] for fe56, u235 mix:")
        print(ssTestMat.macroProp['Ntotal'])
        print("Inf Dilute cross sections [1/cm] for fe56, u235 mix:")
        print(testMat.macroProp['Ntotal'])

if __name__ == "__main__":
    unittest.main()
