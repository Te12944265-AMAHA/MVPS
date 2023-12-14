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
    base_dir = "/home/tina/Documents/courses/15862/MVPS"
    save_dir = f"{base_dir}/blender/generated_coarse"
    f_dir = f"{save_dir}/flash"
    nf_dir = f"{save_dir}/no_flash"
    cam_dir = f"{save_dir}/cam"
    maps_dir = f"{save_dir}/maps"
    maps_vis_dir = f"{save_dir}/maps_vis"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f_dir, exist_ok=True)
    os.makedirs(nf_dir, exist_ok=True)
    os.makedirs(cam_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(maps_vis_dir, exist_ok=True)
    
    scene = bpy.context.scene
    # hide render all cameras and point lights
    cam_objs = [obj for obj in bpy.data.objects if obj.name.startswith("Camera")]
    light_objs = [obj for obj in bpy.data.objects if obj.name.startswith("Point")]
    for obj in light_objs:
        obj.hide_render = True
    for cam in cam_objs:
        obj.hide_render = True
    
    # Render images for each camera view
    for cam in cam_objs:
        camera_name = cam.name
        print(f"Processing {camera_name}...")
        # activate camera
        cam.hide_render = False
        scene.camera = cam
        
        # get rendered flash and no-flash views
        num = camera_name.split(".")[-1]
        light_name = f"Point.{num}"
        light = bpy.data.objects[light_name]
        for light_str in ["flash", "no_flash"]:
            if light_str == "flash":
                light.hide_render = False
            else:
                light.hide_render = True
            scene.render.filepath = f"{save_dir}/{light_str}/{camera_name}.png"
            bpy.ops.render.render(False, animation=False, write_still=True)
        
        scene.render.filepath = f"{save_dir}/tmp.png"
        # get camera matrix
        cam_matrix_save_path = os.path.join(cam_dir, f"{camera_name}.npz")
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
        imsave(os.path.join(maps_vis_dir, f"{camera_name}_normal_map.png"), nmap_vis)

        dmap = get_depth()
        dmap_vis = np.clip(dmap * 255.0, 0.0, 255.0).astype(np.uint8)
        imsave(os.path.join(maps_vis_dir, f"{camera_name}_depth_map.png"), dmap_vis)
        
        maps_save_path = os.path.join(maps_dir, f"{camera_name}.npz")
        res_out = {
            "depth_map": dmap,
            "normal_map": nmap,
        }
        np.savez(maps_save_path, **res_out)
        
        cam.hide_render = True