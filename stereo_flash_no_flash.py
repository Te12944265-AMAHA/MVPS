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
from utils import gamma_decode
from scipy.optimize import least_squares
from scipy.linalg import lstsq

lamb1 = 0.1
lamb2 = 0.1



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
    Solve global lighting parameters l' in N @ l' = m

    ``normals``: (h, w, 3)
    ``mnf``: (h, w)
    ``mf``: (h, w)

    Return l' with shape (9, )
    """
    h, w, _ = normals.shape
    ns = np.reshape(normals, (h * w, 3))
    N = get_spherical_harmonics(ns) / ns[:, 2].reshape((-1, 1))  # p x 9
    m = get_ratio_img(mnf, mf).flatten()  # (p, )
    lp = lstsq(N, m)[0]
    return lp

def get_ratio_img(mnf, mf):
    denom = mf - mnf
    denom = np.where(denom == 0, 1e-5, denom)
    r = mnf / denom
    r = np.where((mf - mnf) == 0, 0, r)
    r = np.minimum(r, 1.0)
    return r

def calc_shading_confidence(mnf, mf):
    denom = np.where(mnf == 0, 1e-5, mnf)
    r = mf / denom
    r = np.where(mnf == 0, 0, r)
    mu = np.mean(r[mnf != 0])
    sig = np.std(r[mnf != 0])
    weight = np.exp(-((r - mu) ** 2) / (2 * sig**2))
    weight = np.where(mnf == 0, 0, weight)
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
    """
    Calculate the cost of the energy function, containing 3 terms: 
    1) shading constraint
    2) normal constraint
    3) unit length constraint

    ``normals``: (h,w,3) normal map, normalized
    ``normals0``: (h,w,3) initial condition of normal map
    ``lp``: (9,) global lighting
    ``mnf``, ``mf``: (h,w) luminance of no-flash or flash image

    Return (h*w, ) residual per pixel
    """
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

def calc_cost_wrapper(x, x0, lp, mnf, mf):
    """
    Rewrite calc_cost function in a way that least_square optimizer understands

    ``x``: (h*w*3, ) normal parameters to be optimized
    ``x0``: (h*w*3, ) initial condition
    ``lp``: (9, ) global lighting
    ``mnf``, ``mf``: (h, w) luminance of no-flash or flash image

    Return (h*w, ) residual per pixel
    """
    h, w = mnf.shape[0], mnf.shape[1]
    normals = x.reshape((h, w, 3))
    normals0 = x0.reshape((h, w, 3))
    cost = calc_cost(normals, normals0, lp, mnf, mf)
    return cost


def optimize(normals0, lp, mnf, mf):
    """
    Given initial guess of the normal map, refine normal map

    ``normals0``: (h, w, 3) initial guess
    ``lp``: (9, ) global lighting
    ``mnf``, ``mf``: (h, w) luminance of no-flash or flash image

    Return (h, w, 3) refined normal map
    """
    h, w, _ = normals0.shape
    x0 = normals0.flatten()
    res = least_squares(calc_cost_wrapper, x0, method="trf", args=(x0, lp, mnf, mf))
    if res.success:
        normals = np.reshape(res.x, (h, w, 3))
        residuals = np.reshape(res.fun, (h, w))
        print(residuals.max(), residuals.min(), residuals.mean())
        # print(res.fun)
        # print(res.cost)
    return normals

def run(normals0, mnf, mf):
    # Step 1: compute global lighting
    lp = compute_global_lighting(normals0, mnf, mf)
    print(lp)
    quit()

    # Step 2: optimize energy function
    normals = optimize(normals0, lp, mnf, mf)
    plt.imshow(normals)
    plt.show()
    plt.close("all")

if __name__ == "__main__":
    img_nf = imread("data/bunny_nf.png").astype(np.float32) / 255.0
    img_f = imread("data/bunny_f.png").astype(np.float32) / 255.0
    img_nf = lRGB2XYZ(gamma_decode(img_nf))[:,:,1]
    img_f = lRGB2XYZ(gamma_decode(img_f))[:,:,1]
    normals0 = imread("data/bunny_normal_map.png").astype(np.float32) / 255.0
    # TODO transform to camera coordinate
    plt.imshow(normals0, cmap="gray")
    plt.title("initial normal map")
    plt.axis("off")
    plt.show()
    plt.close("all")
    #run(normals0, img_nf, img_f)
    # r = get_ratio_img(img_nf, img_f)
    # plt.imshow(r, cmap="gray")
    # plt.title("ratio image")
    # plt.axis("off")
    # plt.show()
    # plt.close("all")