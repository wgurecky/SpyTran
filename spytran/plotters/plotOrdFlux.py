import numpy as np
import matplotlib.pyplot as plt


def cart2pol(x, y):
    """!
    @brief Convert from Cartesian to polar coordinates.
    @param x np_1darray. cart x coord
    @param y np_1darray. cart y coord
    Example
    -------
    >>> theta, radius = pol2cart(x, y)
    """
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return theta, radius


def compass(angles, radii, arrowprops=dict(color='blue', linewidth=3), figNum=6, figName='polarFlux'):
    """!
    @brief Polar plot of angular flux
    @param angles np_1darray of angles in radians
    @param radii np_1darray scaled radii
    """
    plt.figure(figNum)
    # angles, radii = cart2pol(u, v)
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    kw = dict(arrowstyle="->", color='k')
    if arrowprops:
        kw.update(arrowprops)
    [ax.annotate("", xy=(angle, radius), xytext=(0, 0),
                 arrowprops=kw) for
     angle, radius in zip(angles, radii)]
    ax.set_ylim(0, np.max(radii))
    plt.savefig(figName)
    return fig, ax
