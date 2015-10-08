# To execute unittests ensure your pwd is in ./src
# exec:
#   python -m unittest unittest.test1DsnCell
#
# Author: William Gurecky
# Contact: william.gurecky@utexas.edu
#

import unittest
import numpy as np
import sn1Dcell as snc1d

# Load xs database
import materials.materialMixxer as mx
import utils.pinCellMatCalc as pcm
mx.genMaterialDict('./materials/hw2')
pinMaterial = pcm.createPinCellMat()


class test1DsnCell(unittest.TestCase):

    def testCheckScalarFlux(self):
        cell = snc1d.Cell1DSn(1.0, 1.0, source='fission', places=4)
        # group 1 scalar flux:
        self.assertAlmostEqual(cell._evalScalarFlux(1), 1., places=4)
        # group 9 scalar flux:
        self.assertAlmostEqual(cell._evalScalarFlux(9), 1.0)
        # sum over all groups (default is 10 energy grps)
        grpScalarFlux = []
        for g in range(10):
            grpScalarFlux.append(cell._evalScalarFlux(g))
        # 10 energy groups with flux == 1.0 in each group should sum to 10.
        self.assertAlmostEqual(np.sum(np.array(grpScalarFlux)), 10.)

    def testLegFlux(self):
        # S2
        print("\n S2 leg moments of flat unit group 1 flux in mu space")
        cell = snc1d.Cell1DSn(1.0, 1.0, 10, 8, 2, source='fission')
        # group 1 0th legendre moment of flux should be 1.0 as leg(0) = 1.0
        self.assertAlmostEqual(cell._evalLegFlux(1, 0), 1.0, places=4)
        #
        for l in range(9):
            lthLegMomentOfFlux = cell._evalLegFlux(1, l)
            print(str(l) + "th leg moment of flux = " + str(lthLegMomentOfFlux))
        #
        # repeat for S4
        print("\n S4 leg moments of flat unit group 1 flux in mu space")
        cell = snc1d.Cell1DSn(1.0, 1.0, 10, 8, 4, source='fission')
        for l in range(9):
            lthLegMomentOfFlux = cell._evalLegFlux(1, l)
            print(str(l) + "th leg moment of flux = " + str(lthLegMomentOfFlux))

    def testScatterSource(self):
        skernel = pinMaterial.macroProp['Nskernel']
        cell = snc1d.Cell1DSn(1.0, 1.0, 10, 8, 4, source='fission')
        #
        grpScatSource = cell._evalScatterSource(1, skernel)
        print("\n Direction cosines: " + str(cell.sNmu))
        print("Group 1 scatter source with uniform flux in mu space")
        print(grpScatSource)
        grpScatSource = cell._evalScatterSource(8, skernel)
        print("Group 8 scatter source with uniform flux in mu space")
        print(grpScatSource)

    def testFissionSource(self):
        chiNuFission = np.dot(np.array([pinMaterial.macroProp['chi']]).T,
                              np.array([pinMaterial.macroProp['Nnufission']]))
        cell = snc1d.Cell1DSn(1.0, 1.0, 10, 8, 4, source='fission')
        print("\n Direction cosines: " + str(cell.sNmu))
        for g in range(10):
            fissionSrc = cell._computeFissionSource(g, chiNuFission, 1.0)
            print("Fission source for grp " + str(g))
            print(fissionSrc)

    def testGroupSweep(self):
        pass


if __name__ == "__main__":
    unittest.main()
