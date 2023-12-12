import sys
import os
import bpy
import numpy as np
from skimage.io import imsave

sys.path.append("/home/tina/Documents/courses/15862/MVPS/blender/scripts/")

def register():
    pass

def unregister():
    pass

from utils import print
from export_camera_matrix import get_3x4_P_matrix_from_blender
from export_depth_and_normal import get_normal, get_depth


if __name__ == "__main__":
    save_dir = "/home/tina/Documents/courses/15862/MVPS/blender/test"
    os.makedirs(save_dir, exist_ok=True)
    
    
    camera_names = [cam.name for cam in bpy.data.cameras]
    for camera_name in camera_names:
        print(f"Processing {camera_name}...")
        # get camera matrix
        cam_matrix_save_path = os.path.join(save_dir, f"{camera_name}_matrix.npz")
        cam = bpy.data.objects[camera_name]
        P, K, RT = get_3x4_P_matrix_from_blender(cam)
        res_out = {
            "K": np.array(K).astype(np.float32),
            "RT": np.array(RT).astype(np.float32),
        }
        np.savez(cam_matrix_save_path, **res_out)
        
        # get images
        bpy.context.scene.camera = bpy.context.scene.objects[camera_name]
        nmap = get_normal()
        h, w, _ = nmap.shape
        nmap_norm = np.linalg.norm(nmap, axis=2)
        print((nmap_norm.max(), nmap_norm.min()))
        #nmap = np.where(nmap_norm.reshape((h, w, 1)) == 0, 1.0, nmap / nmap_norm.reshape((h, w, 1)))
        #print((nmap.min(), nmap.max()))
        nmap_vis = (nmap + 1) / 2.0
        nmap_vis = np.clip(nmap_vis * 255.0, 0.0, 255.0).astype(np.uint8)
        nmap_fname = f"{camera_name}_normal_map.png"
        imsave(os.path.join(save_dir, f"{camera_name}_normal_map.png"), nmap_vis)

        dmap = get_depth()
        dmap_vis = np.clip(dmap * 255.0, 0.0, 255.0).astype(np.uint8)
        imsave(os.path.join(save_dir, f"{camera_name}_depth_map.png"), dmap_vis)
        
        maps_save_path = os.path.join(save_dir, f"{camera_name}_maps.npz")
        res_out = {
            "depth_map": dmap,
            "normal_map": nmap,
        }
        np.savez(maps_save_path, **res_out)