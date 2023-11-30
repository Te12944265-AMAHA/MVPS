"""
Take in 3 images, recover normal map and albedo map
1. view 1 image with flash
2. view 1 image without flash
3. view 2 image without flash
"""

import numpy as np
from skimage.io import imread, imsave
import cv2
import matplotlib.pyplot as plt
import copy
import argparse
import os
import patchify
from scipy.ndimage import gaussian_filter
from cp_hw2 import lRGB2XYZ
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import lstsq

lamb1 = 1
lamb2 = 1



def get_spherical_harmonics(ns):
    """
    ``ns``: (p, 3)
    Return h(n) shape (p, 9)
    """
    p = ns.shape[0]
    h1 = np.ones(p)
    h2 = ns[:, 0]
    h3 = ns[:, 1]
    h4 = ns[:, 2]
    h5 = h2 * h3
    h6 = h2 * h4
    h7 = h3 * h4
    h8 = h2**2 - h3**2
    h9 = 3 * h4**2 - 1
    h = np.vstack([h1, h2, h3, h4, h5, h6, h7, h8, h9]).T
    return h


def compute_global_lighting(normals, mnf, mf):
    """
    Solve l' in Nl' = m

    ``normals``: (h, w, 3)
    ``mnf``: (h, w)
    ``mf``: (h, w)

    Return l' with shape (9, )
    """
    h, w, _ = normals.shape
    ns = np.reshape(normals, (h * w, 3))
    # TODO check dimension
    N = get_spherical_harmonics(ns) / ns[:, 2]  # p x 9
    m = (mnf / (mf - mnf)).reshape((-1, 1))  # p x 1
    _, _, vh = np.linalg.svd(N)
    lp = vh[-1, :]
    return lp


def calc_shading_confidence(mnf, mf):
    r = mf / mnf
    mu = np.mean(r)
    sig = np.std(r)
    weight = np.exp(-((r - mu) ** 2) / (2 * sig**2))
    return weight


def calc_cost_shading(ns, lp, mnf, mf):
    hn = get_spherical_harmonics(ns)
    # (p, )
    cost = (hn @ lp).flatten() + ns[:, 2] * (mf / (mf - mnf)).flatten()
    return cost


def calc_cost_normal(ns, ns0):
    """
    ``ns``, ``ns0``: (p, 3)
    """
    cost = 1 - np.sum(ns * ns0, axis=-1)
    return cost


def calc_cost_unit_len(ns):
    cost = 1 - np.sum(ns * ns, axis=-1)
    return cost


def calc_cost(normals, normals0, lp, mnf, mf):
    h, w, _ = normals.shape
    ns = np.reshape(normals, (h * w, 3))
    ns0 = np.reshape(normals0, (h * w, 3))
    weight = calc_shading_confidence(mnf, mf)
    cost = (
        weight * calc_cost_shading(ns, lp, mnf, mf)
        + lamb1 * calc_cost_normal(ns, ns0)
        + lamb2 * calc_cost_unit_len(ns)
    )
    return cost
