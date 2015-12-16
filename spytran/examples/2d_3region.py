# to execute:
# From spytran dir:
#   python -m examples.1d_volsrc_slabs
#

import spyTran as spytran
import numpy as np
import os
pwdpath = os.path.dirname(os.path.realpath(__file__))

# Load xs database
import materials.materialMixxer as mx
mx.genMaterialDict('./materials/newXS')

# Solver settings
sN, nG = 4, 10

# Geometry
geoFile = pwdpath + '/geometry/2d_3region.geo'

# Materials
modMat = mx.mixedMat({'h1': 3.35e22 / 1e24, 'o16': 1.67e22 / 1e24})
borMat = mx.mixedMat({'h1': 3.35e22 / 1e24, 'o16': 1.67e22 / 1e24, 'b10': 2.e21 / 1e24})
materialDict = {'mat_1': modMat,
                'mat_2': borMat}

# Boundary conditions
bcDict = {'bc1': 'ref',
          'bc2': 'vac',
          'bc3': 'ref',
          'bc4': 'vac'}

# Volumetric sources
src = np.zeros((10, 12))
srcStrength = 1.e10  # [n / cm**3-s]
src[0, :] = srcStrength
srcDict = {'mat_1': None,
           'mat_2': src}

# Init solver
slv = spytran.D1solver(geoFile, materialDict, bcDict, srcDict, nG=nG, sN=sN, dim=2)

# Solve
slv.trSolve(residTol=1e-5)
slv.writeData(pwdpath + '/output/2Dtestout.h5')

# Plot
# Plot
from fe.post import Fe2DOutput as fe2Dplt
plotter = fe2Dplt(pwdpath + '/output/2Dtestout.h5')
plotter.writeToVTK(fname=pwdpath + '/output/2Dmregion')
