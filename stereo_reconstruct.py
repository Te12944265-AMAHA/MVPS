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
    ``img2``: view 2 no flash
    """

    return

def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1src.shape
    img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
    img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color

def rectify_from_correspondences(pts1, pts2, img1, img2):
    assert len(pts1) >= 5 and len(pts1) == len(pts2)
    E, mask = cv2.findEssentialMat(pts1, pts2, np.eye(3), method=cv2.RANSAC)
    #print(E)
    #n, R, t, _ = cv2.recoverPose(E, pts1, pts2, np.eye(3), 1e9, mask=mask)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # We select only inlier points
    pts1 = pts1[inliers.ravel() == 1]
    pts2 = pts2[inliers.ravel() == 1]

    h1, w1 = img1.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1)
    )
    return F, H1, H2

def visualize_epilines(F, pts1, pts2, img1, img2):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(
        pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(
        pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.suptitle("Epilines in both images")
    plt.show()

def visualize_rectify(H1, H2, img1, img2):
    h1, w1 = img1.shape[0:2]
    h2, w2 = img2.shape[0:2]

    # Undistort (rectify) the images and save them
    # Adapted from: https://stackoverflow.com/a/62607343
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1), borderMode=cv2.BORDER_REPLICATE)
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2), borderMode=cv2.BORDER_REPLICATE)

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(img1_rectified)
    axes[1].imshow(img2_rectified)
    axes[0].axhline(100)
    axes[1].axhline(100)
    axes[0].axhline(210)
    axes[1].axhline(210)
    plt.suptitle("Rectified images")
    plt.savefig("rectified_images.png")
    plt.show()
    

def get_disparity_vis(src: np.ndarray, scale: float = 1.0) -> np.ndarray:
    '''Replicated OpenCV C++ function

    Found here: https://github.com/opencv/opencv_contrib/blob/b91a781cbc1285d441aa682926d93d8c23678b0b/modules/ximgproc/src/disparity_filters.cpp#L559
    
    Arguments:
        src (np.ndarray): input numpy array
        scale (float): scale factor

    Returns:
        dst (np.ndarray): scaled input array
    '''
    dst = np.clip(src.astype(np.float32) * scale/16.0, a_min=0, a_max=255)
    print(dst.max(), dst.min())
    dst = dst.astype(np.uint8)
    return dst


def disparity_from_rectified(img1, img2):
    """
    ``img1``, ``img2``: grayscale images, rectified
    """
    # disparity range is tuned for 'aloe' image pair
    win_size = 3
    min_disp = 0
    max_disp = num_disp = 64

    if False:
        # Applying stereo image rectification on the left image
        img1 = cv2.remap(img1,
                Left_Stereo_Map_x,
                Left_Stereo_Map_y,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0)
        
        # Applying stereo image rectification on the right image
        img2 = cv2.remap(img2,
                Right_Stereo_Map_x,
                Right_Stereo_Map_y,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0)

    # filtering with confidence
    if False:
        max_disp/=2
        if max_disp%16!=0:
            max_disp += 16-(max_disp%16)
        img1 = cv2.resize(img1, (0, 0), 0.5, 0.5, cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (0, 0), 0.5, 0.5, cv2.INTER_LINEAR)

    # Creating an object of StereoBM algorithm
    stereo1 = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=max_disp,
        blockSize=win_size,
        P1=8 * 3 * win_size**2,
        P2=32 * 3 * win_size**2,
        preFilterCap=40,
        speckleWindowSize=0,
        mode=2, # MODE_SGBM_3WAY
    )

    stereo2 = cv2.StereoSGBM_create(
        minDisparity=-(min_disp+num_disp)+1,
        numDisparities=num_disp,
        blockSize=win_size,
        P1=8 * 3 * win_size**2,
        P2=32 * 3 * win_size**2,
        preFilterCap=40,
        speckleWindowSize=0,
        mode=2, # MODE_SGBM_3WAY
    )

    disparity1 = stereo1.compute(img1, img2)
    disparity2 = stereo2.compute(img2, img1)


    # Filtering wls
    lamb = 8000.0
    sig = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo1)
    wls_filter.setLambda(lamb)
    wls_filter.setSigmaColor(sig)
    disparity1_filtered = wls_filter.filter(disparity1, img1, disparity_map_right=disparity2)

    conf_map = wls_filter.getConfidenceMap()

    #roi = wls_filter.getROI()
    # upscale raw disparity and ROI back
    #disparity1 = cv2.resize(disparity1, (0,0), 2.0,2.0, cv2.INTER_LINEAR)
    #disparity1 = disparity1.astype(np.float32) * 2.0
    #roi = cv2.Rect(roi.x*2, roi.y*2, roi.width*2, roi.height*2)
    #disparity1_filtered = cv2.resize(disparity1_filtered, (0,0), 2.0,2.0, cv2.INTER_LINEAR)
    #disparity1_filtered = disparity1_filtered.astype(np.float32) * 8.0

    # Visualization
    #cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE) 
    vis_mult = 2.0
    raw_disp_vis = get_disparity_vis(disparity1, vis_mult)
    cv2.imshow("img", raw_disp_vis)
    cv2.waitKey(0)
    filtered_disp_vis = get_disparity_vis(disparity1_filtered, vis_mult)
    cv2.imshow("img", filtered_disp_vis)
    cv2.waitKey(0)
    cv2.imshow("img",conf_map)
    cv2.waitKey(0)

    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it
    # is essential to convert it to CV_32F and scale it down 16 times.
    # disparity_map = stereo.compute(img1, img2).astype(np.float32) / 16.0
    # print("here3")

    # disparity_n = (disparity_map - min_disp) / num_disp

    # cv2.imshow("disp", disparity_n)
    # cv2.waitKey(0)


if __name__ == "__main__":
    img1 = cv2.imread("data/img1.jpg")
    img2 = cv2.imread("data/img2.jpg")
    # img1 = cv2.imread("data/desk1.png")
    # img2 = cv2.imread("data/desk2.png")

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    disparity_from_rectified(img1, img2)