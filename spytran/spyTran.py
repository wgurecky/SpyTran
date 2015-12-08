#!/usr/bin/python

from fe import solver as fe1D
from fe.post import Fe1DOutput as fe1Dplt
import numpy as np
np.set_printoptions(linewidth=200)  # set print to screen opts


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


def nonMultiplying(slv, residTol=0.5e-5):
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
    totScTime, totLsTime = 0, 0
    print("====================================================================")
    print(" ||ScFlux||/||TotFlux|| -- Lin Solve Time [s] -- Scattering Time [s]")
    print("====================================================================")
    for i in range(0, 180):
        slv.scatterSource()
        norms, times = slv.solveFlux()
        totScTime += times[0]
        totLsTime += times[1]
        print("       " + "{:.4e}".format(norms) + "              " +
              "{:.2e}".format(times[1]) + "           " +
              "{:.2e}".format(times[0]))
        if norms < residTol:
            break
    print("====================================================================")
    print("    TOTAL SOLVE TIME [S]       " + "{:.2e}".format(totLsTime) + "           " +
          "{:.2e}".format(totScTime))
    print("====================================================================")


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
    # Load xs database
    import materials.materialMixxer as mx
    mx.genMaterialDict('./materials/newXS')
    # Solver settings
    sN, nG = 8, 10
    # Geometry
    geoFile = 'utils/testline2.geo'
    # Materials
    #material1 = mx.mixedMat({'h1': 3.35e22 / 1e24, 'o16': 1.67e22 / 1e24})
    #material2 = mx.mixedMat({'h1': 3.35e22 / 1e24, 'o16': 1.67e22 / 1e24, 'b10': 1.e5 / 1e24})
    attnMat = mx.mixedMat({'c12': 1.0})
    attnMat.setDensity(2.24)
    materialDict = {'mat_1': attnMat,
                    'mat_2': attnMat}
    # Boundary conditions
    fixedFlux1 = np.zeros((nG, sN))
    fixedFlux1[0, 0] = 16 * 1e6    # 1e6 flux in ord 0, grp 1
    bcDict = {'bc1': fixedFlux1,
              'bc2': 'vac'}
    # Volumetric sources
    srcDict = {'mat_1': None,
               'mat_2': None}
    slv = initSolve(geoFile, materialDict, bcDict, srcDict, nG=nG, sN=sN)
    nonMultiplying(slv)
    slv.writeData('1Dtestout.h5')
    plotter = fe1Dplt('1Dtestout.h5')
    for i in range(nG):
        plotter.plotScalarFlux(i)
    plotter.plotTotalFlux()
