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
from scipy.optimize import least_squares, minimize, rosen
from scipy.linalg import lstsq
import time

lamb1 = 0.1
lamb2 = 10.0

patch_size_x = 25
patch_size_y = 25

count = 0

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


def compute_global_lighting(normals, mnf, mf, fg_mask):
    """
    Solve global lighting parameters l' in N @ l' = m

    ``normals``: (h, w, 3)
    ``mnf``: (h, w)
    ``mf``: (h, w)
    ``fg_mask``: (h, w) bool mask, True if is foreground

    Return l' with shape (9, )
    """
    fg_pix_idxs = np.argwhere(fg_mask.flatten()).flatten()
    ns = np.reshape(normals, (-1, 3)) # (h*w, 3)
    ns = ns[fg_pix_idxs, :] # (p, 3)
    N = get_spherical_harmonics(ns) / ns[:, 2].reshape((-1, 1))  # (p, 9)
    m = get_ratio_img(mnf, mf).flatten()[fg_pix_idxs]  # (p, )
    lp = lstsq(N, m)[0]
    return lp

def get_ratio_img(mnf, mf):
    denom = mf - mnf
    denom = np.where(denom == 0, 1e-5, denom)
    r = mnf / denom
    r = np.where((mf - mnf) == 0, 0, r)
    r = np.minimum(r, 1.0)
    return r

def calc_shading_confidence(mnf, mf, fg_mask):
    denom = np.where(mnf == 0, 1e-5, mnf)
    r = mf / denom
    r = np.where(mnf == 0, 0, r)
    #r = get_ratio_img(mnf, mf)
    # plt.imshow(r)
    # plt.title("ratio")
    # plt.axis("off")
    # plt.show()
    # plt.close("all")
    mu = np.mean(r[fg_mask])
    sig = np.std(r[fg_mask])
    weight = np.exp(-((r - mu) ** 2) / (2 * sig**2))
    weight = np.where(fg_mask, weight, 0)
    # plt.imshow(weight)
    # plt.title("confidence weight")
    # plt.axis("off")
    # plt.show()
    # plt.close("all")
    # quit()
    return weight


def calc_cost_shading(ns, lp, mnf, mf):
    hn = get_spherical_harmonics(ns)
    # (h*w, )
    cost = (hn @ lp).flatten() + ns[:, 2] * get_ratio_img(mnf, mf).flatten()
    return cost ** 2


def calc_cost_normal(ns, ns0):
    """
    ``ns``, ``ns0``: (p, 3)
    """
    cost = 1 - np.sum(ns * ns0, axis=-1)
    return cost ** 2


def calc_cost_unit_len(ns):
    cost = 1 - np.sum(ns * ns, axis=-1)
    return cost ** 2


def calc_cost(normals, normals0, lp, mnf, mf, fg_mask):
    """
    Calculate the cost of the energy function, containing 3 terms: 
    1) shading constraint
    2) normal constraint
    3) unit length constraint

    ``normals``: (h,w,3) normal map, normalized
    ``normals0``: (h,w,3) initial condition of normal map
    ``lp``: (9,) global lighting
    ``mnf``, ``mf``: (h,w) luminance of no-flash or flash image
    ``fg_mask``: (h, w) bool mask, True if is foreground

    Return (h*w, ) residual per pixel
    """
    global count
    count += 1
    h, w, _ = normals.shape
    ns = np.reshape(normals, (h * w, 3))
    ns0 = np.reshape(normals0, (h * w, 3))
    weight = calc_shading_confidence(mnf, mf, fg_mask).flatten()
    cost_shading = calc_cost_shading(ns, lp, mnf, mf)
    cost_normal = calc_cost_normal(ns, ns0)
    cost_unit_len = calc_cost_unit_len(ns)
    cost = (
        weight * cost_shading
        + lamb1 * cost_normal
        + lamb2 * cost_unit_len
    ) # (h*w, )
    cost = np.where(fg_mask.flatten(), cost, 0.0) # mask out background
    # if count == 1 or count == 1000:
    #     print(normals0[0,0,:])
    if count == 2000:
        print(count)
        plt.subplot(2, 4, 1)
        plt.imshow((normals+1)/2)
        plt.title("estimated")
        plt.axis("off")
        plt.subplot(2, 4, 2)
        plt.imshow((normals0+1)/2)
        plt.title("initial")
        plt.axis("off")
        plt.subplot(2, 4, 3)
        plt.imshow(weight.reshape((h, w)))
        plt.title("weight")
        plt.axis("off")
        plt.subplot(2, 4, 4)
        denom = np.where(mnf == 0, 1e-5, mnf)
        r = mf / denom
        r = np.where(mnf == 0, 0, r)
        plt.imshow(r.reshape((h, w)))
        plt.title("ratio")
        plt.axis("off")
        plt.subplot(2, 4, 5)
        plt.imshow(cost_shading.reshape((h, w)))
        plt.title("cost_shading")
        plt.axis("off")
        plt.subplot(2, 4, 6)
        plt.imshow(cost_normal.reshape((h, w)))
        plt.title("cost_normal")
        plt.axis("off")
        plt.subplot(2, 4, 7)
        plt.imshow(cost_unit_len.reshape((h, w)))
        plt.title("cost_unit_len")
        plt.axis("off")
        plt.subplot(2, 4, 8)
        plt.imshow(cost.reshape(h, w))
        plt.title("cost")
        plt.axis("off")
        plt.show()
        plt.close("all")
        quit()
    return cost

def calc_cost_wrapper(x, x0, lp, mnf, mf, fg_mask):
    """
    Rewrite calc_cost function in a way that least_square optimizer understands

    ``x``: (h*w*3, ) normal parameters to be optimized
    ``x0``: (h*w*3, ) initial condition
    ``lp``: (9, ) global lighting
    ``mnf``, ``mf``: (h, w) luminance of no-flash or flash image
    ``fg_mask``: (h, w) bool mask, True if is foreground

    Return (h*w, ) residual per pixel
    """
    h, w = mnf.shape[0], mnf.shape[1]
    normals = x.reshape((h, w, 3))
    normals0 = x0.reshape((h, w, 3))
    cost = calc_cost(normals, normals0, lp, mnf, mf, fg_mask)
    return cost


def optimize(normals0, lp, mnf, mf, fg_mask):
    """
    Given initial guess of the normal map, refine normal map

    ``normals0``: (h, w, 3) initial guess
    ``lp``: (9, ) global lighting
    ``mnf``, ``mf``: (h, w) luminance of no-flash or flash image
    ``fg_mask``: (h, w) bool mask, True if is foreground

    Return (h, w, 3) refined normal map
    """
    global count
    h, w, _ = normals0.shape
    #x0 = normals0.flatten()
    # TODO break into patches
    normals = np.ones_like(normals0)
    residuals = np.zeros_like(mnf)
    num_pix_patch = patch_size_y*patch_size_x
    jac_sparsity_patch = np.zeros((num_pix_patch, num_pix_patch*3))
    for i in range(num_pix_patch):
        jac_sparsity_patch[i, i*3 : i*3+3] = np.ones(3)
    for i in range(0, h, patch_size_y):
        for j in range(0, w, patch_size_x):
            if i != 400 or j != 300:
                continue
            i_end = i + patch_size_y
            j_end = j + patch_size_x
            fg_mask_patch = fg_mask[i:i_end, j:j_end]
            if not np.any(fg_mask_patch):
                continue
            print(i, j)
            x0_patch = normals0[i:i_end, j:j_end, :]
            mnf_patch = mnf[i:i_end, j:j_end]
            mf_patch = mf[i:i_end, j:j_end]
            res = least_squares(calc_cost_wrapper, x0_patch.flatten(), method="trf", jac_sparsity=jac_sparsity_patch, args=(x0_patch.flatten(), lp, mnf_patch, mf_patch, fg_mask_patch))
            if res.success:
                count = 0
                print("num evaluations:", res.nfev)
                normals_patch = np.reshape(res.x, (patch_size_y*patch_size_x, 3))
                fg_pix_idxs = np.argwhere(fg_mask_patch.flatten()).flatten() # (p, )
                normals_patch[fg_pix_idxs, :] = np.ones_like(normals_patch[fg_pix_idxs, :])
                normals_patch = np.reshape(normals_patch, (patch_size_y, patch_size_x, 3))
                residuals_patch = np.reshape(res.fun, (patch_size_y, patch_size_x))
                plt.subplot(1, 2, 1)
                plt.imshow(normals_patch)
                plt.title("estimated")
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(x0_patch)
                plt.title("initial")
                plt.axis("off")
                plt.show()
                plt.close("all")
                quit()
                normals[i:i_end, j:j_end, :] = normals_patch
                residuals[i:i_end, j:j_end] = residuals_patch
            else:
                print("error")
    residuals[fg_mask==False] = 0.0
    print(residuals[fg_mask==False].max(), residuals[fg_mask==False].min(), residuals[fg_mask==False].mean())
    # print(res.fun)
    # print(res.cost)
    return normals

def run(normals0, mnf, mf, fg_mask):
    # Step 1: compute global lighting
    lp = compute_global_lighting(normals0, mnf, mf, fg_mask)
    # print(lp)
    # quit()

    # Step 2: optimize energy function
    normals = optimize(normals0, lp, mnf, mf, fg_mask)
    plt.imshow(normals)
    plt.show()
    plt.close("all")

def fun_rosenbrock(x):
    return 10.0 * (x[1] - x[0]**2)+ (3*x[2]+1.0)**2

if __name__ == "__main__":
    c = np.ones((1, 3))
    t_start = time.time()
    for i in range(800):
        for j in range(600):
            if j % 50 == 0:
                print(i, j)
            #x0 = np.array([2.0, 2.0, 2.0]) + (np.random.random(3)-0.5)*3
            x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
            ops = {'maxiter': 10, "gtol":1e-2}
            res = minimize(rosen, x0, method="BFGS", tol=1e-3, options=ops)
            print(res.success)
            print(res.status)
            print(res.x)
            print(res.fun)
            print(res.nit)
            quit()
    print(time.time() - t_start)
    quit()
    img_nf = imread("data/bunny_nf.png").astype(np.float64) / 255.0
    img_f = imread("data/bunny_f.png").astype(np.float64) / 255.0
    img_nf = lRGB2XYZ(gamma_decode(img_nf))[:,:,1]
    img_f = lRGB2XYZ(gamma_decode(img_f))[:,:,1]
    cam_matrix_save_path = "data/Camera.002_matrix.npz"
    maps_save_path = "data/Camera.002_maps.npz"
    with np.load(cam_matrix_save_path) as X:
        K, RT = [X[i].astype(np.float64) for i in ("K", "RT")]
    #print(RT)
    with np.load(maps_save_path) as X:
        normals0, depth_map = [X[i].astype(np.float64) for i in ("normal_map", "depth_map")]
    h, w, _ = normals0.shape
    #print(normals0.max(), normals0.min())
    norm = np.linalg.norm(normals0, axis=2).reshape((h, w, 1))
    #print(norm.max(), norm.min())
    fg_mask = np.where(norm == 0, False, True).squeeze()
    denom = np.where(norm == 0, 1e-5, norm)
    normals0 = np.where(norm == 0, 0.0, normals0/denom)
    normals0 = (RT[:3,:3] @ normals0.reshape((h*w, 3)).T).T.reshape((h, w, 3))
    print(normals0[400,300,:])
    normals_vis = (normals0 + 1) / 2.0
    normals_vis = np.where(norm == 0, 1.0, normals_vis)
    #print(normals_vis.max(), normals_vis.min())
    plt.imshow(normals_vis, cmap="gray")
    plt.title("initial normal map")
    plt.axis("off")
    plt.show()
    plt.close("all")
    run(normals0, img_nf, img_f, fg_mask)
    # r = get_ratio_img(img_nf, img_f)
    # plt.imshow(r, cmap="gray")
    # plt.title("ratio image")
    # plt.axis("off")
    # plt.show()
    # plt.close("all")