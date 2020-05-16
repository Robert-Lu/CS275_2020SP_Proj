# This file contains some plotting helper functions that I like to use.
# Author: Wuyue Lu

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ENABLE_PLOT = True
ENABLE_BLOCK = True


parula_cmap_color = [[0.2422, 0.1504, 0.6603],
    [0.2504, 0.1650, 0.7076],
    [0.2578, 0.1818, 0.7511],
    [0.2647, 0.1978, 0.7952],
    [0.2706, 0.2147, 0.8364],
    [0.2751, 0.2342, 0.8710],
    [0.2783, 0.2559, 0.8991],
    [0.2803, 0.2782, 0.9221],
    [0.2813, 0.3006, 0.9414],
    [0.2810, 0.3228, 0.9579],
    [0.2795, 0.3447, 0.9717],
    [0.2760, 0.3667, 0.9829],
    [0.2699, 0.3892, 0.9906],
    [0.2602, 0.4123, 0.9952],
    [0.2440, 0.4358, 0.9988],
    [0.2206, 0.4603, 0.9973],
    [0.1963, 0.4847, 0.9892],
    [0.1834, 0.5074, 0.9798],
    [0.1786, 0.5289, 0.9682],
    [0.1764, 0.5499, 0.9520],
    [0.1687, 0.5703, 0.9359],
    [0.1540, 0.5902, 0.9218],
    [0.1460, 0.6091, 0.9079],
    [0.1380, 0.6276, 0.8973],
    [0.1248, 0.6459, 0.8883],
    [0.1113, 0.6635, 0.8763],
    [0.0952, 0.6798, 0.8598],
    [0.0689, 0.6948, 0.8394],
    [0.0297, 0.7082, 0.8163],
    [0.0036, 0.7203, 0.7917],
    [0.0067, 0.7312, 0.7660],
    [0.0433, 0.7411, 0.7394],
    [0.0964, 0.7500, 0.7120],
    [0.1408, 0.7584, 0.6842],
    [0.1717, 0.7670, 0.6554],
    [0.1938, 0.7758, 0.6251],
    [0.2161, 0.7843, 0.5923],
    [0.2470, 0.7918, 0.5567],
    [0.2906, 0.7973, 0.5188],
    [0.3406, 0.8008, 0.4789],
    [0.3909, 0.8029, 0.4354],
    [0.4456, 0.8024, 0.3909],
    [0.5044, 0.7993, 0.3480],
    [0.5616, 0.7942, 0.3045],
    [0.6174, 0.7876, 0.2612],
    [0.6720, 0.7793, 0.2227],
    [0.7242, 0.7698, 0.1910],
    [0.7738, 0.7598, 0.1646],
    [0.8203, 0.7498, 0.1535],
    [0.8634, 0.7406, 0.1596],
    [0.9035, 0.7330, 0.1774],
    [0.9393, 0.7288, 0.2100],
    [0.9728, 0.7298, 0.2394],
    [0.9956, 0.7434, 0.2371],
    [0.9970, 0.7659, 0.2199],
    [0.9952, 0.7893, 0.2028],
    [0.9892, 0.8136, 0.1885],
    [0.9786, 0.8386, 0.1766],
    [0.9676, 0.8639, 0.1643],
    [0.9610, 0.8890, 0.1537],
    [0.9597, 0.9135, 0.1423],
    [0.9628, 0.9373, 0.1265],
    [0.9691, 0.9606, 0.1064],
    [0.9769, 0.9839, 0.0805],]


def parula_cmap():
    """
    Return a plt-compatible colormap that similar to `parula`,
    which is the default colormap in MATLAB
    """
    colors = np.array(parula_cmap_color)
    N = len(parula_cmap_color)
    colors = np.hstack([colors, np.ones((N, 1))])

    return LinearSegmentedColormap.from_list("parula" , colors, N)


def single_color_cmap(color=None):
    """
    Return a single-color colormap with given color
    color should be a list or ndarray of size not less than 3
    """
    if color is None:
        color = [0, 0, 0]
    colors = np.ones((2, 4))
    for i in range(3):
        colors[0, i] = color[i]
        colors[1, i] = color[i]

    return LinearSegmentedColormap.from_list("single_color" , colors, 2)


def grid_figure_size(r, c, magnitude=None):
    fig_size = plt.rcParams.get('figure.figsize').copy()
    if magnitude is None:
        magnitude = max(4 / (r + c), 1)
    fig_size[0] *= c * magnitude
    fig_size[1] *= r * magnitude
    return fig_size


def imagesc(mat, cmap=parula_cmap(), vmin=None, vmax=None, title=None, colorbar=True):
    """ Show a matrix """

    if not ENABLE_PLOT:
        return

    if cmap == "parula":
        cmap = parula_cmap()
    _im = plt.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(_im)
    if title is not None:
        plt.title(title)
    plt.show(block=True and ENABLE_BLOCK)

def show_2_image(img1, img2, cmap=parula_cmap(), vmin=None, vmax=None,
                 cmap2=parula_cmap(), vmin2=None, vmax2=None, title=None,
                 block=False):
    if not ENABLE_PLOT:
        return

    if cmap == "parula":
        cmap = parula_cmap()
    if cmap2 == "parula":
        cmap2 = parula_cmap()
    f = plt.figure(figsize=grid_figure_size(1, 2))
    f.add_subplot(1, 2, 1)
    plt.imshow(img1, cmap=cmap, vmin=vmin, vmax=vmax)
    f.add_subplot(1, 2, 2)
    plt.imshow(img2, cmap=cmap2, vmin=vmin2, vmax=vmax2)
    if title is not None:
        plt.suptitle(title)
    plt.show(block=block and ENABLE_BLOCK)


def show_sdf_and_zls(im, phi, cmap=parula_cmap(), vmin=None, vmax=None,
                     levels=None, zls_color=None, title=None, no_colorbar=True,
                     block=False, fig_handle=None, force_amend=False, not_show=False):
    """
    :param vmin: range of the colormap
    :param vmax: range of the colormap
    :param levels: used for zero level set (contour)
    :param zls_color: used for zero level set (contour)
    :param block: when show, use block=block
    :param fig_handle: if given and block=False, enable amend mode
    :param force_amend: force to enable amend mode
    :param not_show: do not show or draw, return handle only

    Note:
        * Amend mode means using the previous figure and draw on it.
        * By default, amend <=> not block & fig_handle is not None.
        * If fig_handle not given but force_amend triggered, not considered.
    """

    if not ENABLE_PLOT:
        return

    if zls_color is None:
        zls_color = [0, 0, 0]
    if levels is None:
        levels = [0]
    if cmap == "parula":
        cmap = parula_cmap()

    zls_cmap = zls_color
    if len(levels) == 1:
        zls_cmap = single_color_cmap(zls_color)

    amend_mode = False
    if fig_handle is None:
        f = plt.figure(figsize=grid_figure_size(1, 1))
    else:
        f = fig_handle
        amend_mode = not block
        f.clear()
    if (force_amend):
        amend_mode = True

    _im = plt.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
    if not no_colorbar:
        plt.colorbar(_im)
    plt.contour(phi, levels=levels, cmap=zls_cmap, linewidths=4)
    if title is not None:
        plt.title(title)

    if not_show:
        return f

    if amend_mode:
        plt.draw()
        plt.pause(0.001)
    else:
        plt.show(block=block and ENABLE_BLOCK)
    return f


def show_sdf_and_2_zls(im, phi, phi2, cmap=parula_cmap(), vmin=None, vmax=None,
                     levels=None, zls_color=None, zls_color2=None, title=None, no_colorbar=False,
                     block=False, fig_handle = None, linewidths=2):
    """ See show_sdf_and_zls """

    if not ENABLE_PLOT:
        return

    if zls_color is None:
        zls_color = [0, 0, 0]
    if levels is None:
        levels = [0]
    if cmap == "parula":
        cmap = parula_cmap()
    zls_cmap = single_color_cmap(zls_color)
    zls_cmap2 = single_color_cmap(zls_color2)

    amend_mode = False
    if fig_handle is None:
        f = plt.figure(figsize=grid_figure_size(1, 1))
    else:
        f = fig_handle
        amend_mode = not block
        f.clear()

    _im = plt.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
    if not no_colorbar:
        plt.colorbar(_im)
    plt.contour(phi, levels=levels, cmap=zls_cmap, linewidths=linewidths)
    plt.contour(phi2, levels=levels, cmap=zls_cmap2, linestyles="dashed", linewidths=linewidths)
    if title is not None:
        plt.title(title)

    if amend_mode:
        plt.draw()
        plt.pause(0.001)
    else:
        plt.show(block=block and ENABLE_BLOCK)
    return f

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def show_chan_vese(im, phi, cmap=parula_cmap(), vmin=None, vmax=None,
                     levels=None, zls_color=None, title=None, no_colorbar=True,
                     block=False, fig_handle=None, force_amend=False, not_show=False):
    """
    :param vmin: range of the colormap
    :param vmax: range of the colormap
    :param levels: used for zero level set (contour)
    :param zls_color: used for zero level set (contour)
    :param block: when show, use block=block
    :param fig_handle: if given and block=False, enable amend mode
    :param force_amend: force to enable amend mode
    :param not_show: do not show or draw, return handle only

    Note:
        * Amend mode means using the previous figure and draw on it.
        * By default, amend <=> not block & fig_handle is not None.
        * If fig_handle not given but force_amend triggered, not considered.
    """

    if not ENABLE_PLOT:
        return

    if zls_color is None:
        zls_color = [0, 0, 0]
    if levels is None:
        levels = [0]
    if cmap == "parula":
        cmap = parula_cmap()

    zls_cmap = zls_color
    if len(levels) == 1:
        zls_cmap = single_color_cmap(zls_color)

    amend_mode = False
    if fig_handle is None:
        f = plt.figure(figsize=grid_figure_size(2, 2))
    else:
        f = fig_handle
        amend_mode = not block
        f.clear()
    if (force_amend):
        amend_mode = True

    if not_show:
        return f

    if title is not None:
        f.suptitle(title, fontsize=25)

    f.add_subplot(2, 2, 1)
    _im = plt.imshow(im, cmap="gray", vmin=0, vmax=1)
    # _im = plt.imshow(im, cmap=cmap, vmin=0, vmax=1)

    if not no_colorbar:
        plt.colorbar(_im)
    _c = plt.contour(phi, levels=levels, cmap=zls_cmap, linewidths=2)

    f.add_subplot(2, 2, 2)
    _im = plt.imshow(phi, cmap=parula_cmap())
        
    ax = f.add_subplot(2, 2, 3, projection='3d')
    # 
    size = phi.shape
    X = np.arange(0, phi.shape[1], 1)
    Y = np.arange(0, phi.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    Z = phi
    surf = ax.contour(X, Y, Z, cmap=parula_cmap(), antialiased=False)
    ax.plot_surface(X, Y, Z, cmap=parula_cmap(), rstride=10, cstride=10, alpha=0.3)
    # ax.invert_zaxis()
    # ax.set_zlim(-50, 100)
    # 

    eps = np.finfo(float).eps
    f.add_subplot(2, 2, 4)
    interior_area = np.flatnonzero(phi <= 0) # interior points
    exterior_area = np.flatnonzero(phi > 0)  # exterior points
    c_in  = np.sum(im.flat[interior_area]) / (len(interior_area) + eps)  # interior mean
    c_out = np.sum(im.flat[exterior_area]) / (len(exterior_area) + eps)  # exterior mean
    cvim = np.zeros(phi.shape)
    cvim[phi >  0] = c_out
    cvim[phi <= 0] = c_in
    _im = plt.imshow(cvim, cmap="gray", vmin=0, vmax=1)

    if amend_mode:
        plt.draw()
        plt.pause(0.001)
    else:
        plt.show(block=block and ENABLE_BLOCK)
    return f