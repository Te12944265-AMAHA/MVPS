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
from tqdm import tqdm
from stereo_flash_no_flash import run_one_view, visualize_normals

from configs import data_folder, cams_dir, maps_true_dir, objName, cams_dir_true

save_dir = f"outputs/{objName}/saved"
os.makedirs(save_dir, exist_ok=True)
fg_mask_path = f"{save_dir}/fg_masks.npz"
normals_path = f"{save_dir}/normals_all.npz"
normals_t_path = f"{save_dir}/normals_t_all.npz"
normals0_path = f"{save_dir}/normals0_all.npz"
errors_path = f"{save_dir}/errors.npz"
errors0_path = f"{save_dir}/errors0.npz"

f_all_path = f"{save_dir}/f_all.npz"
nf_all_path = f"{save_dir}/nf_all.npz"


def compute_angular_error(normals_t, normals, fg_mask, vis=False):
    error = np.arctan2(np.linalg.norm(np.cross(normals_t,normals), axis=-1), np.sum(normals_t*normals, axis=-1))
    error = np.where(fg_mask, error, 0.0)
    if vis:
        plt.imshow(error)
        plt.title("angular error")
        plt.axis("off")
        plt.show()
        plt.close("all")
    return error


def run_evaluate(ids):
    errors = []
    errors0 = []
    fg_masks = []
    normals_all = []
    normals_t_all = []
    normals0_all = []

    f_all = []
    nf_all = []

    sum_errors = 0.0
    sum_errors_0 = 0.0
    sum_fg_pix = 0.0

    for cam_id in ids:
        img_id = f"Camera.{cam_id:03d}"
        cam_matrix_save_path = f"{cams_dir_true}/{img_id}.npz"

        # get ground truth
        maps_save_path = f"{maps_true_dir}/{img_id}.npz"
        with np.load(maps_save_path) as X:
            normals_t, depth_map_t = [X[i].astype(np.float64) for i in ("normal_map", "depth_map")]

        # we use this to transform the true normals to camera frame
        with np.load(cam_matrix_save_path) as X:
            K, RT = [X[i].astype(np.float64) for i in ("K", "RT")]

        h, w, _ = normals_t.shape

        # Find true fg mask, normalize, transform to camera frame 
        norm = np.linalg.norm(normals_t, axis=2).reshape((h, w, 1))
        fg_mask_t = np.where(norm == 0, False, True).squeeze()
        denom = np.where(norm == 0, 1e-5, norm)
        normals_t = np.where(norm == 0, 0.0, normals_t/denom)
        # the true normal needs to be transformed to camera frame as below
        # but the estimated normal is already in camera frame
        normals_t = (RT[:3,:3] @ normals_t.reshape((h*w, 3)).T).T.reshape((h, w, 3))

        lp, normals, fg_mask, normals0, img_nf_in, img_f_in = run_one_view(cam_id, fg_mask_t)


        fg_mask_both = fg_mask & fg_mask_t

        # mask out bg areas
        normals_t = np.where(fg_mask_both.reshape((h, w, 1)), normals_t, 0.0)
        normals = np.where(fg_mask_both.reshape((h, w, 1)), normals, 0.0)
        normals0 = np.where(fg_mask_both.reshape((h, w, 1)), normals0, 0.0)

        error = compute_angular_error(normals_t, normals, fg_mask_both)
        print("Estimated:")
        print("view mean error:", np.mean(error[fg_mask_both]))
        print("view max error:", np.max(error[fg_mask_both]))
        print("view min error:", np.min(error[fg_mask_both]))

        error0 = compute_angular_error(normals_t, normals0, fg_mask_both)
        print("Initial:")
        print("view mean error:", np.mean(error0[fg_mask_both]))
        print("view max error:", np.max(error0[fg_mask_both]))
        print("view min error:", np.min(error0[fg_mask_both]))

        errors.append(error)
        errors0.append(error0)

        fg_masks.append(fg_mask_both)

        normals_all.append(normals)
        normals_t_all.append(normals_t)
        normals0_all.append(normals0)

        sum_errors += np.sum(error)
        sum_errors_0 += np.sum(error0)
        sum_fg_pix += np.sum(fg_mask_both)

        f_all.append(np.where(fg_mask_both.reshape((h, w, 1)), img_f_in, 1.0))
        nf_all.append(np.where(fg_mask_both.reshape((h, w, 1)), img_nf_in, 1.0))

    mean_ang_error = sum_errors / sum_fg_pix
    print("Estimated total mean ang error:", mean_ang_error)
    mean_ang_error0 = sum_errors_0 / sum_fg_pix
    print("Initial total mean ang error:", mean_ang_error0)

    # save
    res_out = {"data": np.array(normals_t_all)}
    np.savez(normals_t_path, **res_out)
    res_out = {"data": np.array(normals_all)}
    np.savez(normals_path, **res_out)
    res_out = {"data": np.array(normals0_all)}
    np.savez(normals0_path, **res_out)

    res_out = {"data": np.array(errors)}
    np.savez(errors_path, **res_out)
    res_out = {"data": np.array(errors0)}
    np.savez(errors0_path, **res_out)
    res_out = {"data": np.array(fg_masks)}
    np.savez(fg_mask_path, **res_out)

    res_out = {"data": np.array(f_all)}
    np.savez(f_all_path, **res_out)
    res_out = {"data": np.array(nf_all)}
    np.savez(nf_all_path, **res_out)


def run_evaluate_captured(ids):
    fg_masks = []
    normals_all = []
    normals0_all = []

    f_all = []
    nf_all = []

    for cam_id in ids:
        img_id = f"Camera.{cam_id:03d}"

        lp, normals, fg_mask, normals0, img_nf_in, img_f_in = run_one_view(cam_id, fg_mask_in=None)

        h, w = fg_mask.shape

        fg_mask_both = fg_mask

        # mask out bg areas
        normals = np.where(fg_mask_both.reshape((h, w, 1)), normals, 0.0)
        normals0 = np.where(fg_mask_both.reshape((h, w, 1)), normals0, 0.0)

        fg_masks.append(fg_mask_both)

        normals_all.append(normals)
        normals0_all.append(normals0)

        f_all.append(np.where(fg_mask_both.reshape((h, w, 1)), img_f_in, 1.0))
        nf_all.append(np.where(fg_mask_both.reshape((h, w, 1)), img_nf_in, 1.0))

    # save
    res_out = {"data": np.array(normals_all)}
    np.savez(normals_path, **res_out)
    res_out = {"data": np.array(normals0_all)}
    np.savez(normals0_path, **res_out)

    res_out = {"data": np.array(fg_masks)}
    np.savez(fg_mask_path, **res_out)

    res_out = {"data": np.array(f_all)}
    np.savez(f_all_path, **res_out)
    res_out = {"data": np.array(nf_all)}
    np.savez(nf_all_path, **res_out)

def visualize_captured():
    normals_all = np.load(normals_path)["data"]
    normals0_all = np.load(normals0_path)["data"]

    f_all = np.load(f_all_path)["data"]
    nf_all = np.load(nf_all_path)["data"]

    fg_masks = np.load(fg_mask_path)["data"]

    use_idxs = [0,1]
    fig = plt.figure(constrained_layout=True, figsize=(13, 4))
    subfigs = fig.subfigures(1, 2, wspace=0.03, hspace=0.01)

    _, h, w = fg_masks.shape

    for idx, sample in enumerate(use_idxs):
        axs = subfigs[idx].subplots(2, 2)

        fg_mask = fg_masks[sample,:,:]
        fg_coords = np.argwhere(fg_mask)
        fg_y_min, fg_x_min = np.min(fg_coords, axis=0)
        fg_y_max, fg_x_max = np.max(fg_coords, axis=0)

        fg_mask2 = fg_mask.reshape((h, w, 1))
        
        for i, ax in enumerate(axs.flat):           
            #ax.set_title("set title")
            if i == 0:
                nf_img = nf_all[sample,:,:,:]
                nf_img = np.where(fg_mask2, nf_img, 1.0)
                ax.imshow(nf_img[fg_y_min:fg_y_max, fg_x_min:fg_x_max, :])
                ax.set_title("No-Flash Image")
                ax.set_axis_off()
            elif i == 1:
                normals0 = (normals0_all[sample,:,:,:] +1.0) / 2.0
                normals0 = np.where(fg_mask2, normals0, 1.0)
                ax.imshow(normals0[fg_y_min:fg_y_max, fg_x_min:fg_x_max, :])
                ax.set_title("Coarse")
                ax.set_axis_off()
            elif i == 3:
                normals = (normals_all[sample,:,:,:] +1.0) / 2.0
                normals = np.where(fg_mask2, normals, 1.0)
                ax.imshow(normals[fg_y_min:fg_y_max, fg_x_min:fg_x_max, :])
                ax.set_title("Refined")
                ax.set_axis_off()
            elif i == 2:
                f_img = f_all[sample,:,:,:]
                f_img = np.where(fg_mask2, f_img, 1.0)
                ax.imshow(f_img[fg_y_min:fg_y_max, fg_x_min:fg_x_max, :])
                ax.set_title("Flash Image")
                ax.set_axis_off()

    plt.show()
    plt.close("all")

def visualize():
    normals_t_all = np.load(normals_t_path)["data"]
    normals_all = np.load(normals_path)["data"]
    normals0_all = np.load(normals0_path)["data"]

    errors = np.load(errors_path)["data"]
    errors0 = np.load(errors0_path)["data"]

    f_all = np.load(f_all_path)["data"]
    nf_all = np.load(nf_all_path)["data"]

    fg_masks = np.load(fg_mask_path)["data"]

    use_idxs = [1,3]

    fig = plt.figure(constrained_layout=True, figsize=(13, 4))
    subfigs = fig.subfigures(1, 2, wspace=0.03, hspace=0.01)

    _, h, w = fg_masks.shape

    for idx, sample in enumerate(use_idxs):
        axs = subfigs[idx].subplots(2, 4)

        fg_mask = fg_masks[sample,:,:]
        fg_coords = np.argwhere(fg_mask)
        fg_y_min, fg_x_min = np.min(fg_coords, axis=0)
        fg_y_max, fg_x_max = np.max(fg_coords, axis=0)

        fg_mask2 = fg_mask.reshape((h, w, 1))

        error = errors[sample,:,:]
        error_max = np.max(error)
        error_min = np.min(error[fg_mask])

        error0 = errors0[sample,:,:]
        error0_max = np.max(error0)
        error0_min = np.min(error0[fg_mask])

        
        for i, ax in enumerate(axs.flat):           
            #ax.set_title("set title")
            if i == 0:
                nf_img = nf_all[sample,:,:,:]
                nf_img = np.where(fg_mask2, nf_img, 1.0)
                ax.imshow(nf_img[fg_y_min:fg_y_max, fg_x_min:fg_x_max, :])
                ax.set_title("No-Flash Image")
                ax.set_axis_off()
            elif i == 1:
                normals0 = (normals0_all[sample,:,:,:] +1.0) / 2.0
                normals0 = np.where(fg_mask2, normals0, 1.0)
                ax.imshow(normals0[fg_y_min:fg_y_max, fg_x_min:fg_x_max, :])
                ax.set_title("Coarse")
                ax.set_axis_off()
            elif i == 2:
                normals = (normals_all[sample,:,:,:] +1.0) / 2.0
                normals = np.where(fg_mask2, normals, 1.0)
                ax.imshow(normals[fg_y_min:fg_y_max, fg_x_min:fg_x_max, :])
                ax.set_title("Refined")
                ax.set_axis_off()
            elif i == 3:
                normals_t = (normals_t_all[sample,:,:,:] +1.0) / 2.0
                normals_t = np.where(fg_mask2, normals_t, 1.0)
                ax.imshow(normals_t[fg_y_min:fg_y_max, fg_x_min:fg_x_max, :])
                ax.set_title("Ground Truth")
                ax.set_axis_off()
            elif i == 4:
                f_img = f_all[sample,:,:,:]
                f_img = np.where(fg_mask2, f_img, 1.0)
                ax.imshow(f_img[fg_y_min:fg_y_max, fg_x_min:fg_x_max, :])
                ax.set_title("Flash Image")
                ax.set_axis_off()
            elif i == 5:
                ax.imshow(error0[fg_y_min:fg_y_max, fg_x_min:fg_x_max], vmin=error0_min, vmax=error0_max)
                err = np.mean(error0[fg_mask])
                ax.set_title(f"MangE: {err:.3f}")
                ax.set_axis_off()
            elif i == 6:
                im = ax.imshow(error[fg_y_min:fg_y_max, fg_x_min:fg_x_max], vmin=error_min, vmax=error_max)
                err = np.mean(error[fg_mask])
                ax.set_title(f"MangE: {err:.3f}")
                ax.set_axis_off()
            elif i == 7:
                fig.colorbar(im, cax=ax)

        #subfigs[sample // 2, sample % 2].suptitle(f"Pixel {sample}")
    #fig.suptitle("Dark pixel histograms")
    #plt.savefig("outputs/bunny_out.png")
    plt.show()
    plt.close("all")



    # fig = plt.figure()
    # ax = fig.add_subplot(2, 4, 1, projection="3d")
    # ax.set_title("Original")
    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # for ax in axes.flat:
    #     im = ax.imshow(np.random.random((10,10)), vmin=0, vmax=1)

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)

    # plt.show()



if __name__ == "__main__":
    ids = [0, 2, 3, 6]
    ids = [23, 27]
    visualize_captured()
    #run_evaluate_captured(ids)