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
from scipy.ndimage import gaussian_filter
from cp_hw2 import lRGB2XYZ
from mpl_toolkits.mplot3d import Axes3D
from utils import gamma_decode
from scipy.optimize import least_squares, minimize, rosen
from scipy.linalg import lstsq
import time
from multiprocessing import Pool
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from configs import imgs_dir, cams_dir, maps_dir, bg_val

lamb1 = 0.1
lamb2 = 0.1

patch_size_x = 25
patch_size_y = 25

#count = 0

processes_count = 10

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
    N = get_spherical_harmonics(ns) / -ns[:, 2].reshape((-1, 1))  # (p, 9)
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
    """
    [Deprecated]
    """
    hn = get_spherical_harmonics(ns)
    # (h*w, )
    cost = (hn @ lp).flatten() + ns[:, 2] * get_ratio_img(mnf, mf).flatten()
    return cost ** 2


def calc_cost_normal(ns, ns0):
    """
    [Deprecated]

    ``ns``, ``ns0``: (p, 3)
    """
    cost = 1 - np.sum(ns * ns0, axis=-1)
    return cost ** 2


def calc_cost_unit_len(ns):
    """
    [Deprecated]
    """
    cost = 1 - np.sum(ns * ns, axis=-1)
    return cost ** 2


def calc_cost(normals, normals0, lp, mnf, mf, fg_mask):
    """
    [Deprecated]

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
    #global count
    #count += 1
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
    return cost

def calc_cost_wrapper(x, x0, lp, mnf, mf, fg_mask):
    """
    [Deprecated]

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

def calc_cost_pix(x, _x0, _lp, _weight, _ratio_image):
    """
    ``x``: (3, ) Variable to be optimized
    ``_x0``: (3, )
    ``_lp``: (9, )
    ``_weight``, ``_ratio_image``: scalar

    Return a float
    """
    _hn = get_spherical_harmonics(x.reshape((1,3))).flatten()
    # cost_shading = calc_cost_shading(x, _lp, _mnf, _mf)
    cost_shading = np.dot(_hn, _lp) + x[2] * _ratio_image

    # cost_normal = calc_cost_normal(x, _x0)
    cost_normal = 1 - np.dot(x, _x0)

    #cost_unit_len = calc_cost_unit_len(ns)
    cost_unit_len = 1 - np.dot(x, x)

    cost = (
        _weight * cost_shading**2
        + lamb1 * cost_normal**2
        + lamb2 * cost_unit_len**2
    )
    return cost

def optimize_patch(args):
    #assert len(args) == 2
    global normals0, lp, fg_mask, weight, ratio_image
    i, j = args
    i_end = i + patch_size_y
    j_end = j + patch_size_x
    fg_mask_patch = fg_mask[i:i_end, j:j_end]
    x0_patch = normals0[i:i_end, j:j_end, :]
    weight_patch = weight[i:i_end, j:j_end]
    ratio_image_patch = ratio_image[i:i_end, j:j_end]

    normals_patch = copy.deepcopy(x0_patch)
    for r in range(patch_size_y):
        for c in range(patch_size_x):
            if not fg_mask_patch[r,c]:
                continue
            x0_pix = x0_patch[r, c, :]
            res = minimize(calc_cost_pix, x0_pix, method="BFGS", args=(x0_pix, lp, weight_patch[r,c], ratio_image_patch[r,c]))
            if res.success:
                # fill in pixel
                normals_patch[r, c, :] = res.x
    return (i, j), normals_patch

def init_worker(_normals0, _lp, _fg_mask, _weight, _ratio_image):
    global normals0, lp, fg_mask, weight, ratio_image
    normals0 = _normals0
    lp = _lp
    fg_mask = _fg_mask
    weight = _weight
    ratio_image = _ratio_image


def optimize(_normals0, _lp, _mnf, _mf, _fg_mask):
    """
    Given initial guess of the normal map, refine normal map

    ``normals0``: (h, w, 3) initial guess
    ``lp``: (9, ) global lighting
    ``mnf``, ``mf``: (h, w) luminance of no-flash or flash image
    ``fg_mask``: (h, w) bool mask, True if is foreground

    Return (h, w, 3) refined normal map
    """
    #global count
    h, w, _ = _normals0.shape
    #x0 = normals0.flatten()

    normals = copy.deepcopy(_normals0)
    _weight = calc_shading_confidence(_mnf, _mf, _fg_mask)
    _ratio_image = get_ratio_img(_mnf, _mf)

    # Run normal refinement for each patch in parallel.
    # In each patch, run optimization for each pixel sequentially
    pool = Pool(processes_count, initializer=init_worker, initargs=(_normals0, _lp, _fg_mask, _weight, _ratio_image, ))
    args = []
    for i in range(0, h, patch_size_y):
        for j in range(0, w, patch_size_x):
            args.append((i, j))
    ret_data_all = list(tqdm(pool.imap_unordered(optimize_patch, args), total=len(args)))
    # assemble
    for ret in ret_data_all:
        i, j = ret[0]
        i_end = i + patch_size_y
        j_end = j + patch_size_x
        normals[i:i_end, j:j_end, :] = ret[1]

    return normals

def run(normals0, mnf, mf, fg_mask):
    """
    ``normals_t`` is only for experiment
    """
    # Step 1: compute global lighting
    lp = compute_global_lighting(normals0, mnf, mf, fg_mask)
    # lp_t = compute_global_lighting(normals_t, mnf, mf, fg_mask)
    # print("lp:")
    # print(lp)
    # print(lp_t)
    # print(lp_t / lp)
    # print(np.linalg.norm(lp_t-lp))
    # print(lp)
    # quit()

    # Step 2: optimize energy function
    normals = optimize(normals0, lp, mnf, mf, fg_mask)
    h, w = mnf.shape
    norm = np.linalg.norm(normals, axis=2).reshape((h, w, 1))
    # check which pixels has normal magnitude that's very off
    # diff = norm.squeeze() - np.ones_like(norm.squeeze())
    # diff = np.where(norm.squeeze() == 0, 0.0, diff)
    # plt.imshow(diff)
    # plt.title("normal magnitude diff")
    # plt.axis("off")
    # plt.show()
    # plt.close("all")
    denom = np.where(norm == 0, 1e-5, norm)
    normals = np.where(norm == 0, 0.0, normals/denom)

    return lp, normals


def run_one_view(view_id, fg_mask_in=None):
    img_id = f"Camera.{view_id:03d}"
    print(f"Processing {img_id}...")
    img_nf_path = f"{imgs_dir}/no_flash/{img_id}.png"
    img_f_path = f"{imgs_dir}/flash/{img_id}.png"
    #cam_matrix_save_path = f"{cams_dir}/{img_id}.npz"
    maps_save_path = f"{maps_dir}/{img_id}.npz"

    img_nf_in = imread(img_nf_path).astype(np.float64) / 255.0
    img_f_in = imread(img_f_path).astype(np.float64) / 255.0
    img_nf = lRGB2XYZ(gamma_decode(img_nf_in))[:,:,1]
    img_f = lRGB2XYZ(gamma_decode(img_f_in))[:,:,1]
    # with np.load(cam_matrix_save_path) as X:
    #     K, RT = [X[i].astype(np.float64) for i in ("K", "RT")]


    # coarse normal and depth
    with np.load(maps_save_path) as X:
        normals0, depth_map0 = [X[i].astype(np.float64) for i in ("normal_map", "depth_map")]
    h, w, _ = normals0.shape

    # Find fg mask, normalize, transform to camera frame
    norm = np.linalg.norm(normals0, axis=2).reshape((h, w, 1))

    # using pixel value to find the bg and fg
    #fg_mask = np.all(np.where(np.abs(img_f_in - bg_val) < 1e-5, False, True), axis=2)
    #fg_mask = np.where(norm == 0, False, True).squeeze()
    if fg_mask_in is None:
        fg_mask = np.where(norm == 0, False, True).squeeze()
    else:
        fg_mask = copy.deepcopy(fg_mask_in)
    denom = np.where(norm == 0, 1e-5, norm)
    normals0 = np.where(norm == 0, 0.0, normals0/denom)
    
    # normals0 = (RT[:3,:3] @ normals0.reshape((h*w, 3)).T).T.reshape((h, w, 3))

    # plt.imshow(fg_mask)
    # plt.title("fg mask")
    # plt.axis("off")
    # plt.show()
    # plt.close("all")
    # quit()

    # invert normals0
    #normals0 = -normals0

    # normals0 = gaussian_filter(normals0, sigma=1.0)
    # norm_b = np.linalg.norm(normals0, axis=2).reshape((h, w, 1))
    # denom = np.where(fg_mask.reshape((h,w,1)), norm_b, 1e-5)
    # normals0 = np.where(fg_mask.reshape((h,w,1)), normals0/denom, 0.0)
    # plt.imshow((normals0 + 1.0)/2.0)
    # plt.title("normal0 map")
    # plt.axis("off")
    # plt.show()
    # plt.close("all")

    lp, normals = run(normals0, img_nf, img_f, fg_mask)

    visualize_normals(normals, fg_mask)

    return lp, normals, fg_mask, normals0, img_nf_in, img_f_in

def visualize_normals(normals, fg_mask, vis=True):
    h, w, _ = normals.shape
    normals_vis = (normals + 1) / 2.0
    normals_vis = np.where(fg_mask.reshape((h, w, 1)), normals_vis, 1.0)
    normals_vis = np.clip(normals_vis, a_min=0.0, a_max=1.0)
    #print(normals_vis.max(), normals_vis.min())
    if vis:
        plt.imshow(normals_vis)
        plt.title("normal map")
        plt.axis("off")
        plt.show()
        plt.close("all")
    return normals_vis


if __name__ == "__main__":
    img_nf = imread("data/bunny_nf.png").astype(np.float64) / 255.0
    img_f = imread("data/bunny_f.png").astype(np.float64) / 255.0
    img_nf = lRGB2XYZ(gamma_decode(img_nf))[:,:,1]
    img_f = lRGB2XYZ(gamma_decode(img_f))[:,:,1]
    # cam_matrix_save_path = "data/Camera.002_matrix.npz"
    # maps_save_path = "data/Camera.002_maps.npz"
    # with np.load(cam_matrix_save_path) as X:
    #     K, RT = [X[i].astype(np.float64) for i in ("K", "RT")]
    # #print(RT)
    # with np.load(maps_save_path) as X:
    #     normals0, depth_map = [X[i].astype(np.float64) for i in ("normal_map", "depth_map")]
    # h, w, _ = normals0.shape
    # #print(normals0.max(), normals0.min())
    # norm = np.linalg.norm(normals0, axis=2).reshape((h, w, 1))
    # #print(norm.max(), norm.min())
    # fg_mask = np.where(norm == 0, False, True).squeeze()
    # denom = np.where(norm == 0, 1e-5, norm)
    # normals0 = np.where(norm == 0, 0.0, normals0/denom)
    # normals0 = (RT[:3,:3] @ normals0.reshape((h*w, 3)).T).T.reshape((h, w, 3))
    # print(normals0[400,300,:])
    # normals_vis = (normals0 + 1) / 2.0
    # normals_vis = np.where(norm == 0, 1.0, normals_vis)
    # #print(normals_vis.max(), normals_vis.min())
    # plt.imshow(normals_vis)
    # plt.title("initial normal map")
    # plt.axis("off")
    # plt.show()
    # plt.close("all")
    # run(normals0, img_nf, img_f, fg_mask)
    fo = img_f - img_nf
    r = get_ratio_img(img_nf, img_f)
    plt.subplot(1, 3, 1)
    plt.imshow(img_nf, cmap="gray")
    plt.title("a) no-flash image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(fo, cmap="gray")
    plt.title("b) flash-only image")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(r, cmap="gray")
    plt.title("c) ratio image")
    plt.axis("off")
    plt.show()
    plt.close("all")