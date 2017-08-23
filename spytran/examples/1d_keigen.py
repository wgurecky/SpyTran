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
sN, nG = 2, 10

# Geometry
geoFile = pwdpath + '/geometry/1d_keigen.geo'

# Materials
import spytran.utils.pinCellMatCalc as pcm
pinMaterial = pcm.createPinCellMat()
materialDict = {'mat_1': pinMaterial,
                'mat_2': pinMaterial}

# Boundary conditions
bcDict = {'bc1': 'ref',
          'bc2': 'ref'}

# Volumetric sources
srcDict = {'mat_1': 'fission',
           'mat_2': 'fission'}

# Init solver
slv = spytran.SnSolver(geoFile, materialDict, bcDict, srcDict, nG=nG, sN=sN, space='dg')

# Solve
slv.kSolve(residTol=1e-6, kTol=1e-5)
slv.writeData(pwdpath + '/output/1Dtestout.h5')

# Plot
from spytran.fe.post import Fe1DOutput as fe1Dplt
plotter = fe1Dplt(pwdpath + '/output/1Dtestout.h5')
for i in range(nG):
    plotter.plotScalarFlux(i, fname=pwdpath + '/output/scflux.png')
plotter.plotTotalFlux(fname=pwdpath + '/output/totflux.png')
