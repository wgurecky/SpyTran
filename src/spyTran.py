#!/usr/bin/python

from cyth import snFE1D as fe1D
import numpy as np

# Load xs database
import materials.materialMixxer as mx
mx.genMaterialDict('./materials/newXS')
np.set_printoptions(linewidth=200)  # set print to screen opts

# Top level 1D and 2D controller script.
# Recives input decks and runs the appropriate solvers.
#


def inputPreProc(inFile):
    """
    Parse input file.  Initilize variables.
    """
    pass


def initSolve(geoFile, materialDict, bcDict, srcDict, **kwargs):
    nG = kwargs.pop('nG', 10)
    lOrder = kwargs.pop('lOrder', 8)
    sNords = kwargs.pop('sN', 2)
    solver = fe1D.SnFe1D(geoFile, materialDict, bcDict, srcDict, nG, lOrder, sNords)
    return solver


def nonMultiplying(slv):
    pass


def multiplying():
    pass


if __name__ == "__main__":
    # Solver settings
    sN, nG = 2, 10
    # Geometry
    geoFile = 'utils/testline2.geo'
    # Materials
    material1 = mx.mixedMat({'h1': 3.35e22 / 1e24, 'o16': 1.67e22 / 1e24})
    material2 = mx.mixedMat({'h1': 3.35e22 / 1e24, 'o16': 1.67e22 / 1e24, 'b10': 1.e5 / 1e24})
    materialDict = {'mat_1': material1,
                    'mat_2': material2}
    # Boundary conditions
    fixedFlux1 = np.zeros((nG, sN))
    fixedFlux1[1, 0] = 1e10    # 1e10 flux in ord 0, grp 1
    bcDict = {'bc1': fixedFlux1,
              'bc2': 'vac'}
    # Volumetric sources
    srcDict = {'mat_1': None,
               'mat_2': None}
    slv = initSolve(geoFile, materialDict, bcDict, srcDict, nG=nG, sN=sN)
