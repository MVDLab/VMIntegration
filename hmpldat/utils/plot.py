from itertools import product, combinations
from pprint import pprint

import numpy as np
import pandas as pd

import matplotlib.animation as animation
from matplotlib.patches import Rectangle, PathPatch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.mplot3d.art3d as art3d

# use this to set your backend: ["WXAgg", "TkAgg", "Qt5Agg"]
# matplotlib.use("Qt5Agg")
# print(plt.get_backend())

# define style
plt.style.use("seaborn-bright")


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def screen(ax):
    """ 
    Plot screen

    """

    # define screen
    x = np.linspace(-2490, 2490, 500)
    y = np.linspace(0, 3000, 500)

    xc, yc = np.meshgrid(x, y)
    zc = -np.sqrt(2490 ** 2 - xc ** 2)

    ## plot screen as surface (again swaping y and z values)
    ax.plot_surface(xc, zc, yc, alpha=0.1, rstride=20, cstride=10, shade=True)


def vector(ax, a, b, *args):
    """
    a = tail
    b = head

    TODO: allow options from user
    """

    # tail
    x, y, z = a

    # direction
    q, r, s = b

    # flip y and z for ploting
    vector = Arrow3D(
        [x, q], [z, s], [y, r], mutation_scale=20, lw=1, arrowstyle="-|>", color="b"
    )
    ax.add_artist(vector)


def compare_etg(a, b, screen=False):
    """
    plot eyetracking glasses plane in 3D space

    use this to compare sets of 3D points

    Note:
        z & y values are swapped 
        - y is up in mocap space, matplotlib expects z to be up by default
        - axis labels reflect this change

    """
    print(a.index.shape)
    print(b.index.shape)

    print(a.columns.value_counts().size)
    print(b.columns.shape)

    if isinstance(a.index, pd.MultiIndex):
        # count levels & drop all but -1
        a = a.reset_index(level=list(range(0, a.index.ndim - 1)), drop=True)
    if isinstance(b.index, pd.MultiIndex):
        # count levels & drop all but -1
        b = b.reset_index(level=list(range(0, b.index.ndim - 1)), drop=True)
    if isinstance(a.columns, pd.MultiIndex):
        # count levels & drop all but -1
        a.columns = a.columns.droplevel(level=0)
    if isinstance(b.columns, pd.MultiIndex):
        # count levels & drop all but -1
        b.columns = b.columns.droplevel(level=0)

    print("plotting")
    print(a)
    print(b)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # plot a and b (swaping x and y values)
    # ax.scatter3D(a["x"], a["z"], a["y"], c='m', marker='o', depthshade=False)
    ax.scatter3D(
        a["x"][["camera_vector_origin", "left_vector_origin"]],
        a["z"][["camera_vector_origin", "left_vector_origin"]],
        a["y"][["camera_vector_origin", "left_vector_origin"]],
        c="c",
        marker="o",
        depthshade=False,
    )
    ax.scatter3D(
        a["x"]["camera"],
        a["z"]["camera"],
        a["y"]["camera"],
        c="m",
        marker="o",
        depthshade=False,
    )

    ax.scatter3D(
        b["x"][["lhead", "rhead", "fhead"]],
        b["z"][["lhead", "rhead", "fhead"]],
        b["y"][["lhead", "rhead", "fhead"]],
        c="g",
        marker="o",
        depthshade=False,
    )

    # ax.scatter3D(b["x"][["camera_vector_origin", "left_vector_origin"]], b["z"][["camera_vector_origin", "left_vector_origin"]], b["y"][["camera_vector_origin", "left_vector_origin"]], c='b', marker='o', depthshade=False)
    # ax.scatter3D(b["x"]["camera"], b["z"]["camera"], b["y"]["camera"],c='r', marker='o', depthshade=False)

    ax.set_xlim(-150, 150)
    ax.set_ylim(150, -150)  # z-limits
    ax.set_zlim(1500, 1800)  # y-limits

    # plot screen
    if screen:
        plot_screen(ax)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")  # swap axis labels to match our environment
    ax.set_zlabel("Y")
    # ax.view_init(elev=-160, azim=-70)

    plt.show(block=False)
    plt.pause(0.01)

    while True:
        if input("hit [SPACE BAR] to end.") == " ":
            plt.close()
            break
        else:
            print("y no pres spcebarr?")


def get_cube():
    phi = np.arange(1, 10, 2) * np.pi / 4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi) * np.sin(Theta)
    y = np.sin(Phi) * np.sin(Theta)
    z = np.cos(Theta) / np.sqrt(2)
    return x, y, z


def cross(ax, origin=(0, 0, 0)):
    """add cross to 3d plot
    
    Args:
        origin: center of cross
        ax: matplotlib 3D axis
    
    """

    ### surface plot
    # TODO: smash into single plot surface call
    q, r, s = origin

    x, y, z = get_cube()

    size = 3

    vx, vy, vz = x * 10 * size + q, y * 100 * size + r, z * 10 * size + s
    hx, hy, hz = x * 100 * size + q, y * 10 * size + r, z * 10 * size + s
    ax.plot_surface(vx, vz, vy, color="k")
    ax.plot_surface(hx, hz, hy, color="k")

    ### wireframe plot
    # # vertical rectangle
    # h = 100 #mm
    # w = 10 #mm
    # d = 10 #mm

    # xs = [x - w, x + w]
    # ys = [y - h, y + h]
    # zs = [z - d, z + d]

    # pprint(list(combinations(np.array(list(product(xs, zs, ys))), 2)))

    # edges = []
    # for s, e in combinations(np.array(list(product(xs, zs, ys))), 2):
    #     if np.linalg.norm(s-e) % 10 == 0:
    #         edges.append(np.array([s,e]))
    #         ax.plot3D(*zip(s, e), color="b")

    # pprint(edges)
    # print(len(edges))

    # ### horizontal rectangle
    # h = 10 #mm
    # w = 100 #mm
    # d = 10 #mm

    # xs = [x - w, x + w]
    # ys = [y - h, y + h]
    # zs = [z - d, z + d]

    # edges = []
    # for s, e in combinations(np.array(list(product(xs, zs, ys))), 2):
    #     if np.linalg.norm(s-e) % 10 == 0:
    #         edges.append(np.array([s,e]))
    #         ax.plot3D(*zip(s, e), color="b")

    # pprint(edges)
    # print(len(edges))


def glasses(df, ax):
    """
    plot camera model

    df where columns are points and rows = [x,y,z]
    ax = 3D matplot axis


    """

    m = 150

    centroid = df.mean(axis=1)

    for col in df.columns:
        x, y, z = df[col].values
        ax.scatter(x, z, y, label=col)

    ax.set_xlim(centroid.x - m, centroid.x + m)
    ax.set_ylim(centroid.z + m, centroid.z - m)  # z-limits
    ax.set_zlim(centroid.y - m, centroid.y + m)  # y-limits
    ax.set_xlabel("X")
    ax.set_ylabel("Z")  # swap axis labels to match our environment
    ax.set_zlabel("Y")


if __name__ == "__main__":

    ### example plotting a vector and screen
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    cross(ax, np.array([0, 2203.42428348, -2489.32029622]))
    # vector(ax, np.array([15.68271436, 1657.27412637, -151.62469477]), np.array([-58.17613639, 2203.42428348, -2489.32029622]))

    plot_screen(ax)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")  # swap axis labels to match our environment
    ax.set_zlabel("Y")

    ax.set_xlim3d(-2500, 2500)
    ax.set_ylim3d(200, -2500)
    ax.set_zlim3d(0, 3000)

    # plt.show()

    plt.show(block=False)
    plt.pause(0.001)
    input("hit [enter] to end.")
    plt.close()
