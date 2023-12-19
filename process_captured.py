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
import cv2

from configs import img_captured_dir, imgs_dir

if __name__ == "__main__":
    f_img_dir = os.path.join(img_captured_dir, "flash")
    nf_img_dir = os.path.join(img_captured_dir, "no_flash")
    f_img_paths = sorted([
        os.path.join(f_img_dir, fn)
        for fn in os.listdir(f_img_dir)
        if fn.endswith("JPG")
    ])

    nf_img_paths = sorted([
        os.path.join(nf_img_dir, fn)
        for fn in os.listdir(nf_img_dir)
        if fn.endswith("JPG")
    ])
    assert len(f_img_paths) == len(nf_img_paths)
    step = 4
    for view_id in range(len(f_img_paths)):
        img_id = f"Camera.{view_id:03d}"
        print(img_id)
        # read both, subsample, save
        f_img = imread(f_img_paths[view_id])
        nf_img = imread(nf_img_paths[view_id])
        f_img = f_img[::step,::step, :]
        nf_img = nf_img[::step,::step, :]
        f_save_dir = f"{imgs_dir}/flash"
        nf_save_dir = f"{imgs_dir}/no_flash"
        os.makedirs(f_save_dir, exist_ok=True)
        os.makedirs(nf_save_dir, exist_ok=True)
        imsave(f"{f_save_dir}/{img_id}.png", f_img)
        imsave(f"{nf_save_dir}/{img_id}.png", nf_img)
