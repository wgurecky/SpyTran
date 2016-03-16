# Ordinate set parser
#
import sys
import numpy as np
import scipy.special as spc
import scipy.misc as msc


class D2quadSet(object):
    """
    manipulates a 2D ordinate set.
    Octant numbering:

          x
        1 |  0
        ------y
        2 |  3

    Usage:
        In the transport code ordinates are stored as a 1D array.
        We need to query D2ordSet with an ordinate ID and retrive angle,
        reflective pair, and quadrature weight information.
    """
    octWeightMap = {8: [[0.1209877, ((0, 0), (0, 3), (3, 0))],
                        [0.0907407, ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1))],
                        [0.0925926, ((1, 1), )]],
                    6: [[0.1761263, ((0, 0), (0, 2), (2, 0))],
                        [0.1572071, ((0, 1), (1, 0), (1, 1))]],
                    4: [[1 / 3., ((0, 0), (0, 1), (1, 0))]]
                    }

    def __init__(self, sN):
        self.sN = sN
        self.sNmu, self.wN = levelSymQuadSet(sN)
        self.sNords = (sN * (sN + 2) / 8) * 4        # no of ordinates
        self.ords = np.empty(self.sNords, dtype=object)
        self._computeVariations()
        self.sgn = np.array([[1,  1],
                             [-1, 1],
                             [-1, -1],
                             [1, -1]])
        for o in range(4):
            self._computeAngles(o)
        for ordi in self.ords:
            ordi.computeReflectivePartners(self.ords)
        self.collectProps()
        self.Ylm = createSphrHarm(self.mus, self.omegas, 8)
        self.wN = self.wgts

    def _computeAngles(self, o):
        """ compute and ascosiate mu and eta for each ordinate.
        Input octant is in [0, 1, 2, 3] """
        for j, varpair in enumerate(self.octVariations):
            # (mu, eta)
            iD = o * (self.sNords / 4.) + j
            self.ords[iD] = \
                Ordinate(self.sNmu[varpair[0]] * self.sgn[o, 0],
                         self.sNmu[varpair[1]] * self.sgn[o, 1],
                         self.octWeights[j], o, iD)

    def _computeVariations(self):
        """ produces ord ID pairs in an octant """
        self.octVariations, self.octWeights = [], []
        for i in range(self.sN / 2):
            for j in range(self.sN / 2 - i):
                self.octVariations.append([i, j])
                self.octWeights.append(self.findWeight(i, j))
        self.octVariations = np.array(self.octVariations)

    def findWeight(self, i, j):
        for row in self.octWeightMap[self.sN]:
            for idpair in row[1]:
                if (i, j) == idpair:
                    return row[0]

    def dirCosine(self, testOmega):
        """ return ordinates that have component in same dir as test direciton """
        outNormalIDs, inNormalIDs, i = [], [], 0
        for mu, eta, zed in zip(self.mus, self.etas, self.zeds):
            if np.dot(testOmega, np.array([mu, eta, zed])) > 0:
                outNormalIDs.append(i)
            else:
                inNormalIDs.append(i)
            i += 1
        return np.array(outNormalIDs), np.array(inNormalIDs)

    def collectProps(self):
        self.wgts = np.zeros(len(self.ords))
        self.mus = np.zeros(len(self.ords))
        self.etas = np.zeros(len(self.ords))
        self.zeds = np.zeros(len(self.ords))
        self.omegas = np.zeros(len(self.ords))
        self.xzpairs = np.zeros((len(self.ords), 2))
        self.yzpairs = np.zeros((len(self.ords), 2))
        for i, ordi in enumerate(self.ords):
            self.wgts[i] = ordi.wgt
            self.mus[i] = ordi.mu
            self.etas[i] = ordi.eta
            self.zeds[i] = ordi.zed
            self.omegas[i] = ordi.omega
            self.xzpairs[i] = ordi.xzpair
            self.yzpairs[i] = ordi.yzpair

    def plotOrds(self, figname='3dords.png'):
        """ Plots the ordinate set """
        import ordplot
        ordplot.ordPlot(self.mus, self.etas, self.zeds, figname)

    def plotOrdFlux(self, ordFlux):
        """ Plots the ordinate set vectors scaled by their ordinate flux.
        usefull for showing ordinate flux plots in all their 3d glory. """
        if len(ordFlux) == len(self.mus):
            # check ordFlux len
            pass


class Ordinate(object):
    def __init__(self, mu, eta, w, oc, iD):
        # mu is direction cosine of theta (projection of Omega with x axis)
        # mu is sin(theta)*cos(omega), omega is angle between y axis
        self.wgt = w
        self.mu, self.eta = mu, eta
        self.oc, self.iD = oc, iD
        self.theta = np.arccos(self.mu)
        self.omega = np.arccos(self.eta * (1 - self.mu ** 2) ** (-1 / 2.))
        self.zed = (1 - self.mu ** 2) ** (1 / 2.) * np.sin(self.omega)

    def computeReflectivePartners(self, ords):
        # reflect about the x,z and y,z plane - need a reflective partner for
        # both situations in 2D
        for canidateOrd in ords:
            if self.mu == -1 * canidateOrd.mu and self.eta == canidateOrd.eta:
                self.xzpair = np.array([self.iD, canidateOrd.iD])
            if self.eta == -1 * canidateOrd.eta and self.mu == canidateOrd.mu:
                self.yzpair = np.array([self.iD, canidateOrd.iD])


def levelSymQuadSet(sN):
    if sN == 4:
        mu = np.array([0.3500212, 0.8688903])
        wN = np.array([1 / 3.])
    elif sN == 6:
        mu = np.array([0.2666355, 0.6815076, 0.9261808])
        wN = np.array([0.1761263, 0.1572071])
    elif sN == 8:
        mu = np.array([0.2182179, 0.5773503, 0.7867958, 0.9511897])
        wN = np.array([0.1209877, 0.0907407, 0.0925926])
    else:
        sys.exit("Choose 4, 6, or 8 ordinates for 2D SN")
    ordinateSet = np.array([mu, wN])
    return ordinateSet


def createLegArray(sNmu, lMax):
    legArray = np.zeros((lMax + 1, len(sNmu)))
    for l in range(lMax + 1):
        legArray[l, :] = spc.eval_legendre(l, sNmu)
    return legArray


def createSphrHarm(mu, omega, lMax=8):
    sphArray = np.zeros((lMax + 1, lMax + 1, len(mu)))
    for l in range(lMax + 1):
        for m in range(lMax + 1):
            # loop over legendre order
            for i, (mmu, om) in enumerate(zip(mu, omega)):
                try:
                    C = (2 * l + 1) * float(msc.factorial(l - m)) / \
                        float(msc.factorial(l + m))
                    #sphArray[m, l, i] = np.real(C ** (0.5) * spc.lpmv(m, l, mmu) * np.exp(complex(om * m)))
                    sphArray[m, l, i] = np.real(C ** (0.5) * spc.lpmv(m, l, mmu) * np.cos(om * m))
                    #sphArray[m, l, i] = spc.sph_harm(m, l, mmu, om).real
                except:
                    pass
    return sphArray


def readOrdFile(inFile, sNords):
    ords = np.fromtxt(inFile)
    if len(ords) == sNords:
        ordinateSet = ords
    else:
        pass
    return ordinateSet


def gaussLegQuadSet(sNords):
    """
    Input number of ordinates. (should be even number)
    Compute symetric gauss-leg quadrature set
    """
    legWeights = np.zeros(sNords + 1)
    pnp1s = np.zeros(sNords)
    pprimes = np.zeros(sNords)
    legWeights[sNords] = 1.
    mus = np.polynomial.legendre.legroots(legWeights)
    for i, mu in enumerate(mus):
        pprimes[i] = spc.lpn(sNords, mu)[1][-1]
        pnp1s[i] = spc.lpn(sNords + 1, mu)[0][-1]
    weights = -2. / ((sNords + 1) * pnp1s * pprimes)
    #
    ordinateSet = np.array([mus[::-1], weights])
    return ordinateSet


if __name__ == "__main__":
    test2D = D2quadSet(4)
    test2D.plotOrds()
    testOrd = np.array([1, 0, 0])
    outOrds, inOrds = test2D.dirCosine(testOrd)
    print("Ords IDs with positive dot product with given test vector")
    print(outOrds)
    print("Ord IDs with negative dot product with given test vector")
    print(inOrds)
