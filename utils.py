import numpy as np
import copy
import scipy.interpolate
import open3d as o3d
import matplotlib.pyplot as plt
import os

def gamma_decode(img_in):
    """
    Decode a nonlinear image (normalized)
    """
    img_out = np.where(
        img_in <= 0.0404482,
        img_in / 12.92,
        np.power((img_in + 0.055) / 1.055, 2.4),
    )
    return img_out