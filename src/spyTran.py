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
    """
    Pseudo Code:
        1. Initilize the mesh
        2. Assign materials to regions
        3. Assign boundary conditions to boundary elements
        4. initilize problem
            4.1. Zero flux
            4.2. Build A
            4.3. set BCs on A (A is stationary if material props dont change)
    """
    nG = kwargs.pop('nG', 10)
    lOrder = kwargs.pop('lOrder', 8)
    sNords = kwargs.pop('sN', 2)
    solver = fe1D.SnFe1D(geoFile, materialDict, bcDict, srcDict, nG, lOrder, sNords)
    return solver


def nonMultiplying(slv):
    """
    Pseudo Code non mult solve:
        1. Init problem (see initSolve)
        2. Perform scatter
            2.1. Distribute neutrons in angle and energy (qin for each ele known)
                2.1.1 if Volumetric source: RHS of 0th iteration =/= 0.
                      else:  There is no flux so scattering does nothing (RHS=0)
        3. Apply BCs on RHS
            3.1 Use source terms created by step 2 (scattering) to create RHS
            3.2 Accounts for dirichlet BCs on faces (fixed face flux bc), RHS
                may no longer be 0.
        4. Solve Ax=b for new ith scattered flux
        5. Add ith scattered flux to 'tot' flux (tot flux is sum of all scattered fluxes)
        6. Retun to 2.  Repeat untill scattered flux is stationary (measured by L2 norm)
    """
    for i in range(0, 5):
        slv.scatterSource()
        norms, resid = slv.solveFlux()
        print(norms)


def multiplying():
    """
    Pseudo Code multiplying problem:
        1. Guess flux and eigenvalue
        2. Obtain 1st fission source distribution (like a volume src).  Technically
           this is the 0th scattered neutron step.
        3. Solve Ax=b for 0th scatterd distribution of flux
        3.1 Add 0th scattered flux to 'tot' flux
            4. Goto step 2. of non-mult problem:
                step 2 from non-mult problem is modified as follows:
                    The scattering source term now includes a contribution from
                    fission reactions.
            5. Apply BCs on RHS
            6. Solve Ax=b to update flux
            7. Add ith scattered flux to 'tot' flux
        8. update eigenvalue
        9. Return to 4 untill eigenvalue and flux are stationary
    """
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
    nonMultiplying(slv)
