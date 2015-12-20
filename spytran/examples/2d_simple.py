# to execute:
# From spytran dir:
#   python -m examples.1d_keigen
#

import spytran.spyTran as spytran
import os
pwdpath = os.path.dirname(os.path.realpath(__file__))

# Load xs database
import spytran.materials.materialMixxer as mx
mx.genMaterialDict('../materials/newXS')

# Solver settings
sN, nG = 6, 10

# Geometry
geoFile = pwdpath + '/geometry/2d_simple.geo'

# Materials
import spytran.utils.pinCellMatCalc as pcm
pinMaterial = pcm.createPinCellMat()
modMat = mx.mixedMat({'h1': 3.35e22 / 1e24, 'o16': 1.67e22 / 1e24})
modMat.setDensity(1.0)
materialDict = {'mat_1': pinMaterial,
                'mat_2': modMat}

# Boundary conditions
bcDict = {'bc1': 'ref',
          'bc2': 'ref',
          'bc3': 'ref',
          'bc4': 'ref'}

# Volumetric sources
srcDict = {'mat_1': 'fission',
           'mat_2': None}

# Init solver
slv = spytran.D1solver(geoFile, materialDict, bcDict, srcDict,
                       nG=nG, sN=sN, dim=2)

# Solve
slv.kSolve(residTol=1e-6, kTol=1e-5, outerIterMax=4)
slv.writeData(pwdpath + '/output/2Dfistestout.h5')

# Plot
from spytran.fe.post import Fe2DOutput as fe2Dplt
plotter = fe2Dplt(pwdpath + '/output/2Dfistestout.h5')
plotter.writeToVTK(fname=pwdpath + '/output/2dfisplot')
