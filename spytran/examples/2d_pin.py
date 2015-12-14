# to execute:
# From spytran dir:
#   python -m examples.1d_keigen
#

import spyTran as spytran
#import numpy as np
import os
pwdpath = os.path.dirname(os.path.realpath(__file__))

# Load xs database
import materials.materialMixxer as mx
mx.genMaterialDict('./materials/newXS')

# Solver settings
sN, nG = 4, 10

# Geometry
geoFile = pwdpath + '/geometry/2d_pin.geo'

# Materials
# FUEL
duUO2 = mx.mixedMat({'u238': 1. / 3., 'o16': 2 / 3.})
duUO2.setDensity(10.35)
heuUO2 = mx.mixedMat({'u235': 1 / 3., 'o16': 2 / 3.})
heuUO2.setDensity(10.35)
fuelMat = 0.964 * duUO2 + 0.036 * heuUO2
fuelMat.setDensity(10.35)
# create cladding mixture
cladMat = mx.mixedMat({'zr90': 1.0})
cladMat.setDensity(5.87)
# create moderator mixture
modrMat = mx.mixedMat({'h1': 2. / 3., 'o16': 1 / 3.})
modrMat.setDensity(1.0)


materialDict = {'mat_1': fuelMat,
                'mat_2': cladMat,
                'mat_3': modrMat}

# Boundary conditions
bcDict = {'bc1': 'ref',
          'bc2': 'ref',
          'bc3': 'ref',
          'bc4': 'ref'}

# Volumetric sources
srcDict = {'mat_1': 'fission',
           'mat_2': None,
           'mat_3': None}

# Init solver
slv = spytran.D1solver(geoFile, materialDict, bcDict, srcDict,
                       nG=nG, sN=sN, dim=2)

# Solve
slv.kSolve(residTol=1e-6, kTol=1e-5, outerIterMax=5)
slv.writeData(pwdpath + '/output/2Dpintest.h5')

# Plot
from fe.post import Fe2DOutput as fe2Dplt
plotter = fe2Dplt(pwdpath + '/output/2Dpintest.h5')
plotter.writeToVTK(fname=pwdpath + '/output/2Dpin')
