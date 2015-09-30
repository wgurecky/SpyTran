import numpy as np
import pylab as pl


E1 = np.array([9.5598e0, 2.7005e-2, 3.4415e-5, 0, 0, 0, 0, 0, 0])
E2 = np.array([8.0042e0, 1.9376e0, 7.7967e-1, 1.8668e-1, 5.6568e-2,
               4.2337e-3, 5.2808e-5, 4.0947e-7, -9.6399e-7])


def legval(maxL, muVec, sigSlVec):
    """
    muVec must be on [-1, 1]
    sigmaSl Vec must be of length maxL
    """
    vecL = np.arange(0, maxL + 1)
    return np.polynomial.legendre.legval(muVec, (2 * vecL + 1) * sigSlVec)


def main():
    cosVec = np.linspace(0, 180., 100)
    muVec = np.cos(np.deg2rad(cosVec))
    # case 1 E1 2nd order approx
    case1Vec = legval(2, muVec, E1[0:3])
    # case 2 E2 2nd order approx
    case2Vec = legval(2, muVec, E2[0:3])
    # case 3 E1 8th order approx
    case3Vec = legval(8, muVec, E1)
    # case 4 E2 8th order approx
    case4Vec = legval(8, muVec, E2)
    pl.figure(1)
    pl.title("2nd order scattering kernel")
    pl.xlabel("scattering angle (deg)")
    pl.ylabel("cross section [barns]")
    plt1 = pl.plot(cosVec, case1Vec, linewidth=4, color='r', label="E1")
    plt2 = pl.plot(cosVec, case2Vec, linewidth=4, color='black', label="E2")
    pl.legend()
    pl.figure(2)
    pl.title("8th order scattering kernel")
    pl.ylabel("cross section [barns]")
    pl.xlabel("scattering angle (deg)")
    plt1, = pl.plot(cosVec, case3Vec, linewidth=4, color='r', label="E1")
    plt2, = pl.plot(cosVec, case4Vec, linewidth=4, color='black', label="E2")
    pl.legend()
    pl.show()


if __name__ == "__main__":
    main()
