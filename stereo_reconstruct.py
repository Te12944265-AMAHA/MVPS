"""
Take in 2 ambient images, recover normal map
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


def get_coarse_normal(img1, img2):
    """
    From stereo matching and planePCA

    ``img1``: view 1 no flash
    ``img2``; view 2 no flash
    """

    return


img1 = cv2.imread("data/img1.jpg")
img2 = cv2.imread("data/img2.jpg")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img1 = cv2.resize(img1, (0, 0), 0.5, 0.5, cv2.INTER_LINEAR)
img2 = cv2.resize(img2, (0, 0), 0.5, 0.5, cv2.INTER_LINEAR)


# # Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()
print("here1")

# disparity range is tuned for 'aloe' image pair
win_size = 1
min_disp = 16
max_disp = min_disp * 9
num_disp = max_disp - min_disp  # Needs to be divisible by 16
# stereo = cv2.StereoSGBM(
#     minDisparity=min_disp,
#     numDisparities=num_disp,
#     SADWindowSize=win_size,
#     uniquenessRatio=10,
#     speckleWindowSize=100,
#     speckleRange=32,
#     disp12MaxDiff=1,
#     P1=8 * 3 * win_size**2,
#     P2=32 * 3 * win_size**2,
#     fullDP=True,
# )
print("here2")

# NOTE: Code returns a 16bit signed single channel image,
# CV_16S containing a disparity map scaled by 16. Hence it
# is essential to convert it to CV_32F and scale it down 16 times.
disparity_map = stereo.compute(img1, img2).astype(np.float32) / 16.0
print("here3")

disparity_n = (disparity_map - min_disp) / num_disp

cv2.imshow("disp", disparity_n)
cv2.waitKey(0)
