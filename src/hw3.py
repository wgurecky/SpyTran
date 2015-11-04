import numpy as np
import sn1D as sn

# Load xs database
import materials.materialMixxer as mx
mx.genMaterialDict('./materials/newXS')

# plotters
import plotters.fluxEplot as flxPlt
import plotters.scalarFluxPlot as sfp
import plotters.plotOrdFlux as pof


def testSlab():
    print("\n========= INITIATING MULT REGION TEST ==========")
    ngrps = 10
    sNord = 8
    srcStrength = 1.e10  # [n / cm**3-s]
    # ## MATERIAL DEFS ##
    modMat = mx.mixedMat({'h1': 3.35e22 / 1e24, 'o16': 1.67e22 / 1e24})
    borMat = mx.mixedMat({'h1': 3.35e22 / 1e24, 'o16': 1.67e22 / 1e24, 'b10': 2.e21 / 1e24})
    # ## REGION WIDTHS
    width1, dx1 = 4, 0.1
    end1 = 0 + width1 - dx1
    #
    width2, dx2 = 2, 0.02
    start2 = end1 + dx1 / 2 + dx2 / 2
    end2 = start2 + width2 - dx2
    #
    width3, dx3 = 3, 0.1
    start3 = end2 + dx2 / 2 + dx3 / 2
    end3 = start3 + width3 - dx3
    #
    width4, dx4 = 2, 0.02
    start4 = end3 + dx3 / 2 + dx4 / 2
    #start4 = end3 + dx3 / 1. + dx4 / 1. + 0.04
    end4 = start4 + width4 - dx4
    #
    width5, dx5 = 3, 0.1
    start5 = end4 + dx4 / 2 + dx5 / 2
    end5 = start5 + width5 - dx5
    #
    width6, dx6 = 2, 0.02
    start6 = end5 + dx5 / 2 + dx6 / 2
    end6 = start6 + width6 - dx6
    #
    width7, dx7 = 4, 0.1
    start7 = end6 + dx6 / 2 + dx7 / 2
    end7 = start7 + width7 - dx7
    #
    # ## REGIONS Defs ###
    region1mesh1D = sn.Mesh1Dsn([0, end1 + dx1], dx1, modMat, sN=sNord)
    src = np.zeros((10, 3, sNord))
    src[0, 0, :] = srcStrength
    bcs1 = {0: {'vac': (1, 0)}}
    region1mesh1D.setBCs(bcs1)
    region2mesh1D = sn.Mesh1Dsn([start2, end2], dx2, borMat, sN=sNord, source=src)
    region3mesh1D = sn.Mesh1Dsn([start3, end3], dx3, modMat, sN=sNord)
    region4mesh1D = sn.Mesh1Dsn([start4, end4], dx4, borMat, sN=sNord, source=src)
    region5mesh1D = sn.Mesh1Dsn([start5, end5], dx5, modMat, sN=sNord)
    region6mesh1D = sn.Mesh1Dsn([start6, end6], dx6, borMat, sN=sNord, source=src)
    region7mesh1D = sn.Mesh1Dsn([start7, end7 + dx7], dx7, modMat, sN=sNord)
    bcs2 = {-1: {'vac': (2, 0)}}
    region7mesh1D.setBCs(bcs2)
    # ## BUILD DOMAIN ###
    domain = sn.SubDomain()
    domain.addRegion(region1mesh1D)
    domain.addRegion(region2mesh1D)
    domain.addRegion(region3mesh1D)
    domain.addRegion(region4mesh1D)
    domain.addRegion(region5mesh1D)
    domain.addRegion(region6mesh1D)
    domain.addRegion(region7mesh1D)
    domain.buildSweepTree()
    #
    # ## SWEEP DOMAIN ###
    for si in range(100):
        resid = domain.sweepSubDomain(1)
        if resid < 3.1e-4:
            break
    scalarFlux = domain.getScalarFlux()
    flxPlt.plotFluxE(scalarFlux[-1][::-1])  # flux vs E at left edge
    centroids = domain.getCentroids()
    # plot all grp fluxes vs space
    for g in range(ngrps):
        sfp.plot1DScalarFlux(scalarFlux[:][:, g], centroids, label='Group ' + str(g + 1), legend=False)
    # plot ord fluxes at center of first absorber strip
    ordFlux = domain.getOrdFlux()
    angles = np.arccos(domain.regions[1].cells[49].sNmu)
    for g in range(10):
        mag = ordFlux[70][g, 0, :] / sum(ordFlux[70][g, 0, :])
        pof.compass(angles, mag, figName='hw3_polar_grp' + str(g + 1))
    # plot absorption rate
    absRate = domain.getAbsRate()
    sfp.plot1DScalarFlux(absRate, centroids, label='absRate', legend=True, fnameOut='absRate', figNum=20)
    # Rate of leakage out of left and right faces
    leftGrpCurrent, rightGrpCurrent = 0, 0
    for g in range(ngrps):
        leftGrpCurrent += 0.5 * np.sum((domain.regions[0].cells[0].wN[:] * np.abs(domain.regions[0].cells[0].sNmu[:]) *
                                       domain.regions[0].cells[0].totOrdFlux[g, 1, :])[sNord/2:])
        rightGrpCurrent += 0.5 * np.sum((domain.regions[6].cells[-1].wN[:] * np.abs(domain.regions[6].cells[-1].sNmu[:]) *
                                         domain.regions[6].cells[-1].totOrdFlux[g, 2, :])[:sNord/2])
    # total neutron production rate (3x abs pins of width 2cm)
    totProd = 1e10 * 2 * 3   # (n/s-cm^3) * (cm)  -> n/s-cm^2
    # fraction out left and right face
    lfl = leftGrpCurrent / totProd
    rfl = rightGrpCurrent / totProd
    print("Fraction of source neutrons leaking left= " + str(lfl))
    print("Fraction of source neutrons leaking right= " + str(rfl))
    # non leakage prob
    nlp = 1 - (leftGrpCurrent + rightGrpCurrent) / totProd
    print("Non Leakage Probability= " + str(nlp))


def homogenized():
    print("\n========= INITIATING HOMOGENIZED REGION TEST ==========")
    ngrps = 10
    sNord = 8
    srcStrength = 1.e10  # [n / cm**3-s]
    # ## MATERIAL DEFS ##
    modMat = mx.mixedMat({'h1': 3.35e22 / 1e24, 'o16': 1.67e22 / 1e24})
    borMat = mx.mixedMat({'h1': 3.35e22 / 1e24, 'o16': 1.67e22 / 1e24, 'b10': 2.e21 / 1e24})
    homoMat = (6. / 20) * borMat + (14. / 20) * modMat
    # ## REGION WIDTHS
    width1, dx1 = 4, 0.1
    end1 = 0 + width1 - dx1
    #
    width2, dx2 = 2, 0.02
    start2 = end1 + dx1 / 2 + dx2 / 2
    end2 = start2 + width2 - dx2
    #
    width3, dx3 = 3, 0.1
    start3 = end2 + dx2 / 2 + dx3 / 2
    end3 = start3 + width3 - dx3
    #
    width4, dx4 = 2, 0.02
    start4 = end3 + dx3 / 2 + dx4 / 2
    #start4 = end3 + dx3 / 1. + dx4 / 1. + 0.04
    end4 = start4 + width4 - dx4
    #
    width5, dx5 = 3, 0.1
    start5 = end4 + dx4 / 2 + dx5 / 2
    end5 = start5 + width5 - dx5
    #
    width6, dx6 = 2, 0.02
    start6 = end5 + dx5 / 2 + dx6 / 2
    end6 = start6 + width6 - dx6
    #
    width7, dx7 = 4, 0.1
    start7 = end6 + dx6 / 2 + dx7 / 2
    end7 = start7 + width7 - dx7
    #
    # ## REGIONS Defs ###
    region1mesh1D = sn.Mesh1Dsn([0, end1 + dx1], dx1, homoMat, sN=sNord)
    src = np.zeros((10, 3, sNord))
    src[0, 0, :] = srcStrength
    bcs1 = {0: {'vac': (1, 0)}}
    region1mesh1D.setBCs(bcs1)
    region2mesh1D = sn.Mesh1Dsn([start2, end2], dx2, homoMat, sN=sNord, source=src)
    region3mesh1D = sn.Mesh1Dsn([start3, end3], dx3, homoMat, sN=sNord)
    region4mesh1D = sn.Mesh1Dsn([start4, end4], dx4, homoMat, sN=sNord, source=src)
    region5mesh1D = sn.Mesh1Dsn([start5, end5], dx5, homoMat, sN=sNord)
    region6mesh1D = sn.Mesh1Dsn([start6, end6], dx6, homoMat, sN=sNord, source=src)
    region7mesh1D = sn.Mesh1Dsn([start7, end7 + dx7], dx7, homoMat, sN=sNord)
    bcs2 = {-1: {'vac': (2, 0)}}
    region7mesh1D.setBCs(bcs2)
    # ## BUILD DOMAIN ###
    domain = sn.SubDomain()
    domain.addRegion(region1mesh1D)
    domain.addRegion(region2mesh1D)
    domain.addRegion(region3mesh1D)
    domain.addRegion(region4mesh1D)
    domain.addRegion(region5mesh1D)
    domain.addRegion(region6mesh1D)
    domain.addRegion(region7mesh1D)
    domain.buildSweepTree()
    #
    # ## SWEEP DOMAIN ###
    for si in range(100):
        resid = domain.sweepSubDomain(1)
        if resid < 3.1e-4:
            break
    scalarFlux = domain.getScalarFlux()
    flxPlt.plotFluxE(scalarFlux[-1][::-1])  # flux vs E at left edge
    centroids = domain.getCentroids()
    # plot all grp fluxes vs space
    for g in range(ngrps):
        sfp.plot1DScalarFlux(scalarFlux[:][:, g], centroids, label='Group ' + str(g + 1), legend=False)
    # plot ord fluxes at center of first absorber strip
    ordFlux = domain.getOrdFlux()
    angles = np.arccos(domain.regions[0].cells[0].sNmu)
    for g in range(10):
        mag = ordFlux[70][g, 0, :] / sum(ordFlux[70][g, 0, :])
        pof.compass(angles, mag, figName='hw3_polar_grp' + str(g + 1))
    # plot absorption rate
    absRate = domain.getAbsRate()
    sfp.plot1DScalarFlux(absRate, centroids, label='absRate', legend=True, fnameOut='absRate', figNum=20)
    # Rate of leakage out of left and right faces
    leftGrpCurrent, rightGrpCurrent = 0, 0
    for g in range(ngrps):
        leftGrpCurrent += 0.5 * np.sum((domain.regions[0].cells[0].wN[:] * np.abs(domain.regions[0].cells[0].sNmu[:]) *
                                       domain.regions[0].cells[0].totOrdFlux[g, 1, :])[sNord/2:])
        rightGrpCurrent += 0.5 * np.sum((domain.regions[6].cells[-1].wN[:] * np.abs(domain.regions[6].cells[-1].sNmu[:]) *
                                         domain.regions[6].cells[-1].totOrdFlux[g, 2, :])[:sNord/2])
    # total neutron production rate (3x abs pins of width 2cm)
    totProd = 1e10 * 2 * 3   # (n/s-cm^3) * (cm)  -> n/s-cm^2
    # fraction out left and right face
    lfl = leftGrpCurrent / totProd
    rfl = rightGrpCurrent / totProd
    print("Fraction of source neutrons leaking left= " + str(lfl))
    print("Fraction of source neutrons leaking right= " + str(rfl))
    # non leakage prob
    nlp = 1 - (leftGrpCurrent + rightGrpCurrent) / totProd
    print("Non Leakage Probability= " + str(nlp))


if __name__ == "__main__":
    testSlab()
    #homogenized()
