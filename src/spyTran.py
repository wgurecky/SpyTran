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


def main(geoFile, materialDict, **kwargs):
    nG = kwargs.pop('nG', 10)
    lOrder = kwargs.pop('lOrder', 8)
    sNords = kwargs.pop('sNords', 2)
    solver = fe1D.SnFe1D(geoFile, materialDict, nG, lOrder, sNords)


if __name__ == "__main__":
    geoFile = 'utils/testline2.geo'
    material1 = mx.mixedMat({'h1': 3.35e22 / 1e24, 'o16': 1.67e22 / 1e24})
    material2 = mx.mixedMat({'h1': 3.35e22 / 1e24, 'o16': 1.67e22 / 1e24, 'b10': 2.e21 / 1e24})
    materialDict = {'mat_1': material1,
                    'mat_2': material2}
    main(geoFile, materialDict)
