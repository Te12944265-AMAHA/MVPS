import os

objName = "shoe"
data_folder = "datasets"
maps_dir = f"{data_folder}/{objName}/recon/maps"
os.makedirs(maps_dir, exist_ok=True)

maps_true_dir = f"{data_folder}/{objName}/generated/maps"

cams_dir_raw = f"{data_folder}/{objName}/sfm/cam"
os.makedirs(cams_dir_raw, exist_ok=True)
cams_dir = f"{data_folder}/{objName}/recon/cam"
os.makedirs(cams_dir, exist_ok=True)
cams_dir_true = f"{data_folder}/{objName}/generated/cam"

pcd_path = f"{data_folder}/{objName}/recon/reconstructed_filtered.pcd"

imgs_dir = f"{data_folder}/{objName}/generated"
os.makedirs(imgs_dir, exist_ok=True)

img_captured_dir = f"{data_folder}/{objName}/captured"

# background value
bg_val = 61.0 / 255.0