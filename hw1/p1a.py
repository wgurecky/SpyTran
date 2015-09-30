import numpy as np
import pylab as pl


no = 1e10
N = 1e24
sigma = 1e-24


def expfun(u):
    """
    eval exp function.
    takes a single real or np.vector
    """
    return no * np.exp(-N*sigma*u)


def main():
    u = np.linspace(0, 10., 100)
    nu = expfun(u)
    pl.xlabel("distance [cm]")
    pl.ylabel("number density [#/cc]")
    pl.plot(u, nu, linewidth=5)
    attenFrac = 1 - expfun(10.) / no
    print("Attenuated fraction = " + str(attenFrac))
    pl.show()


if __name__=="__main__":
    main()
