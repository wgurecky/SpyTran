Class: Problem Domain
=====================

Initilized via input-deck.

Contains multiple regions.  Assigns dimensions and materials to those regions.

Assigns boundary conditions to all sides of each region.

def solveDomain():

    - transport solve
      Essentially the "main" solve function.

Subclass: Multiplying Domain
----------------------------

def powerIteration():

    - performs a power iteration after the region flux has been updated.

Subclass: Fixed Source Domain
-----------------------------

No special code required.  Perform region sweeps untill convergence.


Class: Region
=============

Holds a mesh.  

Holds a material.  

def selfSheildRegion():

    - modifies region material xs to account for self sheilding

    - optional: updateSelfShield():

        - requires additional function to compute leakage out of region

        - uses leakage information out of the region to compute new and improved
          self sheilded cross sections.

def sweepRegion():

    - performs an "inner" flux iteration.  sweep through all angle and all space.

    - future work: If multiple regions in problem allow parrallel sweeps via mpi


Class: Mesh
===========

Holds multiple cells.

When constructing the mesh assign cell types. 

Stores (and computes) cell connectivity data (nearest neighbors).

Contains ability to set boundary conditions. conditions on currents and fluxes possible.

Holds scalar field data, which can be generalized:

    - ordinate fluxes in the case of SN

    - legendre moments in case of PN

Evaluate spatial deriviatives.  Contains a variety of stencil options ideally. 

    - 1st, 2nd order differences schemes


Class: Cell
===========

Angle discretization & Legendre order.

Stores type of cell:

    - boundary cell

        - reflective
        - white
        - region2region boundary
        - vaccume

    - interior cell


Subclass: 1D Cell
-----------------

Special quadrature sets for 1D

Stores ordinate directions

Stores ordinate fluxes


Subclass: 2D Cell
-----------------

Special quadrature sets for 2D.

Stores ordinate directions

Stores ordinate fluxes

Supporting Code
===============
