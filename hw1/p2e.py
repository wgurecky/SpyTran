import numpy as np
import pylab as pl


no = 1e10
N = 1e24
sigma = 1e-24
C = 2e-47
v0 = 1e9
m = 1.67e-31


def expfun(u):
    """
    eval exp function.
    takes a single real or np.vector
    """
    #c1 = no / ((m * v0) * np.exp(-sigma*m*v0/C))
    #return c1*(N*C*u + m * v0) * np.exp(-sigma*(C*N*u+m*v0)/C)
    c1 = -no*m*v0/np.exp(m*sigma*v0/C)
    return c1*np.exp(m*sigma*v0/C-N*sigma*u)/(C*N*u-m*v0)


def expfun1(u):
    """
    eval exp function.
    takes a single real or np.vector
    """
    return no * np.exp(-N*sigma*u)


def main():
    u = np.linspace(0, 10., 500)
    nu = expfun(u)
    nu1 = expfun1(u)
    pl.xlabel("distance [cm]")
    pl.ylabel("number density [#/cc]")
    pl.ylim([0, 0.1e10])
    pl.plot(u, nu, linewidth=5)
    pl.plot(u, nu1, linewidth=3, color='r')
    pl.show()


if __name__ == "__main__":
    main()
