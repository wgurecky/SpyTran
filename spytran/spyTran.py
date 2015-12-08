#!/usr/bin/python

from fe import solver as feslv
import numpy as np
np.set_printoptions(linewidth=200)  # set print to screen opts


class D1solver(object):
    def __init__(self, geoFile, materialDict, bcDict, srcDict, **kwargs):
        nG = kwargs.pop('nG', 10)
        lOrder = kwargs.pop('lOrder', 8)
        sNords = kwargs.pop('sN', 2)
        dim = kwargs.pop('dim', 1)
        self.solver = feslv.SnFeSlv(geoFile, materialDict, bcDict, srcDict, nG, lOrder, sNords, dim)

    def trSolve(self, residTol=0.5e-5):
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
            self.solver.scatterSource()
            norms, times = self.solver.solveFlux()
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

    def kSolve(self, residTol=0.5e-5, kTol=1e-4, outerIterMax=15):
        finalIter = False
        print("====================================================================")
        print("=                         K-EIGEN SOLVER                           =")
        print("====================================================================")
        for i in range(1, outerIterMax):
            if i == outerIterMax - 1:
                finalIter = True
            keff, converged, norms = self.solver.kEig(residTol, kTol, finalIter)
            print("====================================================================")
            print("Outter iteration: " + str(i) + "  k-eff :" + str(keff))
            print("====================================================================")
            if converged:
                print("Keff convergence reached!")
                break
        if i == outerIterMax - 1:
            print("Failed to converge k-eigenvalue.")

    def writeData(self, outFile):
        self.solver.writeData(outFile)


class D2solver(object):
    def __init__(self, geoFile, materialDict, bcDict, srcDict, **kwargs):
        nG = kwargs.pop('nG', 10)
        lOrder = kwargs.pop('lOrder', 8)
        sNords = kwargs.pop('sN', 2)
        self.solver = feslv.SnFeSlv(geoFile, materialDict, bcDict, srcDict, nG, lOrder, sNords, 2)


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
    solver = feslv.SnFeSlv(geoFile, materialDict, bcDict, srcDict, nG, lOrder, sNords)
    return solver


if __name__ == "__main__":
    pass
