# Computes atom fraction from pin cell dimensions


import numpy as np
import materials.materialMixxer as mx
from copy import deepcopy


def computeVols():
    # Pin cell dimensions
    ppitch = 1.33  # [cm]
    rfuel = 0.412  # [cm]
    rclad = 0.475  # [cm]
    #
    volTot = ppitch ** 2.
    volFuel = np.pi * rfuel ** 2.
    volClad = np.pi * (rclad ** 2. - rfuel ** 2.)
    volMod = volTot - volFuel - volClad
    return [volFuel, volClad, volMod]


def createPinCellMat():
    # Create fuel mixture
    duUO2 = mx.mixedMat({'u238': 1. / 3., 'o16': 2 / 3.})
    duUO2.setDensity(10.35)
    heuUO2 = mx.mixedMat({'u235': 1 / 3., 'o16': 2 / 3.})
    heuUO2.setDensity(10.35)
    fuelMat = 0.964 * duUO2 + 0.036 * heuUO2
    #fuelMat = 0.6 * duUO2 + 0.4 * heuUO2
    fuelMat.setDensity(10.35)
    # create cladding mixture
    cladMat = mx.mixedMat({'zr90': 1.0})
    cladMat.setDensity(5.87)
    # create moderator mixture
    modrMat = mx.mixedMat({'h1': 2. / 3., 'o16': 1 / 3.})
    modrMat.setDensity(1.0)
    #j
    # Homogenize pin cell materials into one big soup
    regionVols = computeVols()
    volTot = sum(regionVols)
    pinCellMat = (regionVols[0] / volTot) * deepcopy(fuelMat) + \
                 (regionVols[1] / volTot) * deepcopy(cladMat) + \
                 (regionVols[2] / volTot) * deepcopy(modrMat)
    return pinCellMat
