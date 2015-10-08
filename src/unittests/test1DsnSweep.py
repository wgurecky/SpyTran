import unittest
import numpy as np
import sn1D as sn

# Load xs database
import materials.materialMixxer as mx
import utils.pinCellMatCalc as pcm
mx.genMaterialDict('./materials/hw2')
pinMaterial = pcm.createPinCellMat()


class test1DsnCell(unittest.TestCase):

    def testSetBCs(self):
        print("\n========= CHECKING BOUNDARY CONDITION ASSIGNMENT ==========")
        mesh1D = sn.Mesh1Dsn([0, 100], 1.0, pinMaterial, sN=4)
        bcs = {0: {'vac': (1, 0)}, -1: {'vac': (2, 0)}}
        mesh1D.setBCs(bcs)
        #
        # exterior cells
        self.assertEqual(mesh1D.cells[0].applyBC(0), True)
        self.assertEqual(mesh1D.cells[-1].applyBC(0), True)
        self.assertEqual(mesh1D.cells[0].ordFlux[:, 1, :].all(), 0.0)
        self.assertEqual(mesh1D.cells[0].ordFlux[:, 0, :].all(), 1.0)
        self.assertEqual(mesh1D.cells[0].ordFlux[:, 2, :].all(), 1.0)
        self.assertEqual(mesh1D.cells[-1].ordFlux[:, 2, :].all(), 0.0)
        self.assertEqual(mesh1D.cells[-1].ordFlux[:, 1, :].all(), 1.0)
        self.assertEqual(mesh1D.cells[-1].ordFlux[:, 0, :].all(), 1.0)
        #
        # should produce warning, no bc as this is an interior cell
        print("EXPECT WARNING:")
        self.assertEqual(mesh1D.cells[-2].applyBC(0), False)

    def testMeshSweep(self):
        print("\n========= INITIATING MESH SWEEP TEST ==========")
        mesh1D = sn.Mesh1Dsn([0, 10], 0.1, pinMaterial, sN=4)
        bcs = {0: {'vac': (1, 0)}, -1: {'vac': (2, 0)}}
        mesh1D.setBCs(bcs)
        #
        # Perform source iterations
        nSourceIterations = 1
        for si in range(nSourceIterations):
            mesh1D.sweepMesh(2)

    def testKeigenSweep(self):
        print("\n========= INITIATING K-EIGEN TEST ==========")
        mesh1D = sn.Mesh1Dsn([0, 100], 0.5, pinMaterial, sN=4)
        bcs = {0: {'vac': (1, 0)}, -1: {'vac': (2, 0)}}
        mesh1D.setBCs(bcs)
        #
        mesh1D.setKeff(1.)
        for pI in range(4):
            # Perform source iterations
            nSourceIterations = 4
            fissionSourceOld = np.sum(mesh1D.fissionSrc())  # todo mult by width
            for si in range(nSourceIterations):
                mesh1D.sweepMesh(1)
            fissionSourceNew = np.sum(mesh1D.fissionSrc())
            knew = mesh1D.keff * (fissionSourceNew / fissionSourceOld)
            print("Outter iteration: " + str(pI) + "  k-eff :" + str(knew))
            mesh1D.resetFlux()
            mesh1D.setKeff(knew)


if __name__ == "__main__":
    unittest.main()
