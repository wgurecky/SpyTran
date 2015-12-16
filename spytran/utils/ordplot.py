from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure(100)
ax = fig.gca(projection='3d')
ax.set_aspect("equal")
ax.set_ylabel(r'$\eta$')
ax.set_xlabel(r'$\mu$')
ax.set_zlabel('z')

#draw sphere
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="r", linewidth=0.2)

#draw a point
ax.scatter([0], [0], [0], color="g", s=100)

#draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = []
        for x, y, z in zip(xs, ys, zs):
            self._verts3d.append([[0, x], [0, y], [0, z]])

    def draw(self, renderer):
        #xs3d, ys3d, zs3d = self._verts3d
        for coords in self._verts3d:
            xs3d, ys3d, zs3d = coords[0], coords[1], coords[2]
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)


def ordPlot(xs, ys, zs, figname="3dords.png"):
    a = Arrow3D(xs, ys, zs,  mutation_scale=20,  lw=1,  arrowstyle="-|>",  color="k")
    ax.add_artist(a)
    plt.savefig(figname)


if __name__ == "__main__":
    # test 3d ord plot
    a = Arrow3D([1, -1], [1, -1], [1, -1],  mutation_scale=20,  lw=1,  arrowstyle="-|>",  color="k")
    ax.add_artist(a)
    plt.show()
