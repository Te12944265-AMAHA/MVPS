# Multi-View Photometric Stereo

This repo contains the photometric normal refinement routines.

For geometric coarse normal estimation, see [this repo](https://github.com/Te12944265-AMAHA/Hierarchical-Localization).

## Dependencies

python>=3.7
numpy
skimage
matplotlib
scipy
python-opencv
open3d==0.16.0

## File structure

Implemented in `stereo_flash_no_flash.py`, `evaluate.py`, `pointcloud_processing.py`. 

Blender simulation and exporting scripts in `blender/`.

Configuration is in `configs.py`.

Helper functions are in `utils.py`, `cp_hw2.py`.

Place the data in `datasets/`.

## How to run

To perform reconstruction, and run the following steps:

### Step 0: Change config

Set the object name in `configs.py`. Currently supported: `bunny`, `buddha`, `shoe`.

### Step 1: Process point cloud

Turn point cloud from SfM toolbox into a mesh and then into a coarse normal map.

```sh
python3 pointcloud_processing.py
```

### Step 2: Evaluate normal refinement

```sh
python3 evaluate.py
```