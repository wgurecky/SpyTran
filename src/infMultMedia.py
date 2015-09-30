# Solve an infinite multiplying media problem.
# Use 10 energy groups
#
# Obtain:
#   - flux(E)
#   - k-eigenvalue
#
# Balance equation in matrix notation:
# A * b = (1/k) * F * b
# where:
#   A = Multigroup transport operator
#   b = Multigroup flux
#   k = Nuetron mult factor
#   F = fission matrix
#     = Chi.tranpose * nuFission
#
# And k is updated by the following
# k(i+1) = k(i) * (AF / F)

import materials.materialMixxer as mx
import numpy as np

def createPinCellMat():
    # Create fuel mixture
    duUO2 = mx.mixedMat({'u238': 1. / 3., 'o16': 2 / 3.})
    heuUO2 = mx.mixedMat({'u235': 1 / 3., 'o16': 2 / 3.})
    fuelMat = 0.964 * duUO2 + 0.036 * heuUO2
    fuelMat.setDensity(10.35)
    # create cladding mixture
    cladMat = mx.mixedMat({'zr90', 1.0})
    cladMat.setDensity(5.87)
    # create moderator mixture
    modrMat = mx.mixedMat({'1h': 2. / 3., 'o16': 1 / 3.})
    modrMat.setDensity(1.0)
    # Homogenize pin cell materials into one big soup
    return pinCellMat

if __name__ == "__main__":
    pass
