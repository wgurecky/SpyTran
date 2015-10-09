import pylab as plt


def plot1DScalarFlux(fluxVec, meshX, fnameOut='fluxS', figNum=0):
    plt.figure(1)
    plt.plot(meshX, fluxVec, linewidth='4')
    plt.ylabel("Flux (Arbitrary Scaling)")
    plt.xlabel("Position [cm]")
    plt.savefig(fnameOut)
