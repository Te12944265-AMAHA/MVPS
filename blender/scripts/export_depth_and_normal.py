import bpy
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from skimage.io import imsave
import os

def print(data):
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'CONSOLE':
                override = {'window': window, 'screen': screen, 'area': area}
                bpy.ops.console.scrollback_append(override, text=str(data), type="OUTPUT") 
                
def get_depth():
    """Obtains depth map from Blender render.
    :return: The depth map of the rendered camera view as a numpy array of size (H,W).
    """
    bpy.context.scene.node_tree.nodes["Switch"].check = False
    bpy.ops.render.render()
    z = bpy.data.images['Viewer Node']
    w, h = z.size
    dmap = np.array(z.pixels[:], dtype=np.float32) # convert to numpy array
    print(len(np.unique(dmap)))
    dmap = np.reshape(dmap, (h, w, 4))[:,:,0]
    dmap = np.rot90(dmap, k=2)
    dmap = np.fliplr(dmap)
    return dmap

def get_normal():
    """Obtains normal map from Blender render.
    :return: The normal map of the rendered camera view as a numpy array of size (H,W,3).
    """
    bpy.context.scene.node_tree.nodes["Switch"].check = True
    bpy.ops.render.render()
    z = bpy.data.images['Viewer Node']
    w, h = z.size
    dmap = np.array(z.pixels[:], dtype=np.float32) # convert to numpy array
    print(len(np.unique(dmap)))
    print((dmap.max(), dmap.min()))
    dmap = np.reshape(dmap, (h, w, 4))[:,:,:3]
    dmap = np.rot90(dmap, k=2)
    dmap = np.fliplr(dmap)
    return dmap

save_dir = "/home/tina/Documents/courses/15862/MVPS/blender/test"
os.makedirs(save_dir, exist_ok=True)

camera_names = [cam.name for cam in bpy.data.cameras]
for camera_name in camera_names:
    bpy.context.scene.camera = bpy.context.scene.objects[camera_name]
    nmap = get_normal()
    h, w, _ = nmap.shape
    nmap = np.where(np.linalg.norm(nmap, axis=2).reshape((h, w, 1)) == 0, 1.0, nmap)
    nmap_vis = (nmap + 1) / 2.0
    nmap_vis = np.clip(nmap_vis * 255.0, 0.0, 255.0).astype(np.uint8)
    imsave(os.path.join(save_dir, f"{camera_name}_normal_map.png"), nmap_vis)

    dmap = get_depth()
    dmap = np.clip(dmap * 255.0, 0.0, 255.0).astype(np.uint8)
    imsave(os.path.join(save_dir, f"{camera_name}_depth_map.png"), dmap)