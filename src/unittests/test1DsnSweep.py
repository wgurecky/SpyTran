import unittest
#import numpy as np
import sn1D as sn

# Load xs database
import materials.materialMixxer as mx
import utils.pinCellMatCalc as pcm
mx.genMaterialDict('./materials/hw2')
pinMaterial = pcm.createPinCellMat()


class test1DsnCell(unittest.TestCase):

    def testSetBCs(self):
        mesh1D = sn.Mesh1Dsn([0, 100], 1.0, pinMaterial, sN=4)
        bcs = {0: {'vac': (1, 0)}, -1: {'vac': (2, 0)}}
        mesh1D.setBCs(bcs)
        #
        # exterior cells
        self.assertEqual(mesh1D.cells[0].applyBC(0), True)
        self.assertEqual(mesh1D.cells[-1].applyBC(0), True)
        #
        # should produce warning, no bc as this is an interior cell
        self.assertEqual(mesh1D.cells[-2].applyBC(0), False)

    def testMeshSweep(self):
        print("========= INITIATING MESH SWEEP TEST ==========")
        mesh1D = sn.Mesh1Dsn([0, 100], 1.0, pinMaterial, sN=4)
        bcs = {0: {'vac': (1, 0)}, -1: {'vac': (2, 0)}}
        mesh1D.setBCs(bcs)
        #
        # Perform sweep
        mesh1D.sweepMesh()


if __name__ == "__main__":
    unittest.main()
