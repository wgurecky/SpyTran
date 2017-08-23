# to execute:
# From spytran dir:
#   python -m examples.1d_graphite_beam_b10
#

import spytran.spyTran as spytran
import numpy as np
import os
pwdpath = os.path.dirname(os.path.realpath(__file__))

# Load xs database
import spytran.materials.materialMixxer as mx
mx.genMaterialDict('../materials/newXS')

# Solver settings
sN, nG = 8, 10

# Geometry
geoFile = pwdpath + '/geometry/1d_graphite_beam_b10.geo'

# Materials
attnMat = mx.mixedMat({'c12': 1.0})
attnMat.setDensity(2.24)
borMat = mx.mixedMat({'b10': 1.0})
borMat.setDensity(5.00)
materialDict = {'mat_1': attnMat,
                'mat_2': attnMat,
                'mat_3': borMat}

# Boundary conditions
fixedFlux1 = np.zeros((nG, sN))
fixedFlux1[0, -1] = 16 * 1e6    # 1e6 flux in ord 0, grp 1
bcDict = {'bc1': fixedFlux1,
          'bc2': 'vac'}

# Volumetric sources
srcDict = {'mat_1': None,
           'mat_2': None}

# Init solver
slv = spytran.SnSolver(geoFile, materialDict, bcDict, srcDict, nG=nG, sN=sN, space='dg')

# Solve
slv.trSolve(residTol=1e-6)
slv.writeData(pwdpath + '/output/1Dtestout.h5', False)

# Plot
from spytran.fe.post import Fe1DOutput as fe1Dplt
plotter = fe1Dplt(pwdpath + '/output/1Dtestout.h5')
for i in range(nG):
    plotter.plotScalarFlux(i, fname=pwdpath + '/output/scflux.png')
plotter.plotTotalFlux(fname=pwdpath + '/output/totflux.png')
