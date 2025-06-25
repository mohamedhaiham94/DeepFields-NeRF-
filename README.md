# NeRF – 3D Volume Reconstruction

A pipeline for neural radiance field (NeRF) based 3D volume reconstruction.

---
### Python Environment Setup (Python 3.11)

**1. Create a Virtual Environment**
```bash
python -m venv .venv
```
**2. Activate the Virtual Environment**
```bash 
# Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

**3.  Install Project Dependencies**
```bash
pip install -r requirements.txt
```
**4. Install [Pytorch](https://pytorch.org/get-started/locally/) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for [Instant-NGP](https://github.com/NVlabs/instant-ngp)**

**4.1  Install PyTorch (CUDA 12.6)**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
**4.2 Install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)**
```bash
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn\bindings\torch
# Install via setup script
tiny-cuda-nn/bindings/torch$ python setup.py install
```
 Make sure torch is installed before building tiny-cuda-nn, and your system has a compatible CUDA toolkit (e.g., 12.6).

**5. Install [Colmap](https://github.com/colmap/colmap)**
Download and install COLMAP for structure-from-motion and multi-view stereo reconstruction: 
[COLMAP Releases (Windows/Linux/macOS)](https://github.com/colmap/colmap/releases)

---

**Project Directory Structure**

```text
project_root/
├── data/                           # Contains raw input images and colmap outputs
│   └── scene_name
│       ├── images/                 # Resized input images (used for training)
│       ├── images_original/        # Original high-resolution input images
│       └── colmap_text_original/   # COLMAP sparse reconstruction (in text format)
│
├── configs/                        # YAML configuration files for each scene scene
│   ├── blender_scene.yml           # Config file for the blender scene  
│   ├── dataS7.yml                  # Config file for the DataS7 scene  
│   ├── real_scene.yml              # Config file for the Subset scene  
│   └── <scene_name>.yml            # Config file for the corresponding scene
│
├── outputs/                        # Outputs from training or extraction
│   └── <scene_name>
│       ├── checkpoints/            # Saved model checkpoints
│       └── logs                    # TensorBoard logs
│
├── scripts/    
│   ├── instantngp.py               # Instant-NGP variant of NeRF
│   ├── vanilla_nerf.py             # Standard NeRF implementation
│   ├── resize_images.py            # Resizes input images to desired resolution
│   ├── dataset.py                  # Dataset class for NeRF data loading
│   ├── normalize_with_aabb.py      # Normalizes scene and write the transforms_<scene_name>.json
│   ├── precompute_rays.py          # Precomputes ray directions as <scene_name>_ray_data.npz
│   ├── sampling.py                 # Sampler class for stratified/hierarchical sampling 
│   ├── train.py                    # Training script for NeRF 
│   ├── extract_vol.py              # Extracts colored occupancy volume from trained model
│   └── show_slice_vol.py           # Utility to visualize occupancy slices
│
├── outputs/                        # Created output files from from training
│   └── <scene_name>/               # Outputs related to the specific scene
│       ├── checkpoints/            # Trained model checkpoints
│       └── logs/                   # TensorBoard logs for training visualization
│
└── README.md                        # Project documentation
```


### Running the Pipeline.

1. Resize the images:
```bash
python scripts\resize_images.py --cfg_path cfg\<scene_name>.yml
```
The resized images will be saved in the ``data/<scene_name>/images`` folder specified in your config file.

2. Run COLMAP on the resized images and export the scene in text format inside the image directory, in a folder named colmap_text_original ```colmap_text_original```.


3. Create the transform.json file
```bash
python scripts\normalize_with_aabb.py --cfg_path cfg\<scene_name>.yml
```
**Rotation Matrix Initialization**
Each scene allows you to define a custom rotation matrix.
Press the R key in the VisPy viewer to print the current rotation matrix in YAML format. This makes it easy to copy and paste into your configuration file (e.g., rotation_initial: in a .yml file).

**Manual AABB Adjustment**
The Axis-Aligned Bounding Box (AABB) can be manually specified in your config file to override the default normalization. This is useful for cropping, focusing on a subregion, or fine-tuning reconstruction bounds.

```python
# transform.json
# python scripts\normalize_with_aabb.py
rotation: True
rotation_initial: #null
  - [0.77892503, -0.25715966, 0.57196565]
  - [-0.62710305, -0.32549969, 0.70766639]
  - [0.00419140, -0.90990047, -0.41480546]

#compute_percentile_bbox(points, lower=1.0, upper=99.8, padding=0.17)
percentile_bbox:
  lower: 1.0
  upper: 99.8
  padding: 0.17

rot_order: [0, 1, 2] # apply order rotation
# axis order x, y, z - alpha, beta, gamma
angles: [75, 0, 0] # alpha, beta, gamma
shift: [-0.1, -0.2, -0.3]
visualize: True

# aabb
aabb_adjust:
  aabb_min: [0, 0, 0.85]
  aabb_max: [0, 0, 0.1]

```
<p align="center">
  <img src="docs\post_aabb.PNG" alt="Ray 1" width="400px">
</p>


4. Precompute the camera rays for the entire dataset and save them to a file for faster training or evaluation ```<scene_name>_ray_data.npz ```
```bash
python scripts\precompute_rays.py --cfg_path cfg\<scene_name>.yml
```

5. Train the nerf model
```bash
python scripts\train.py --cfg_path cfg\<scene_name>.yml  
```
After training, the NeRF model will be saved to:
```outputs\<scene_name>\checkpoints\nerf_final.pth```

6. Extract volume with the pretrained network
```bash
python python scripts\extract_vol.py --cfg_path cfg\<scene_name>.yml
```
The volume wil be saved to: ```outputs\<scene_name>\volume.pth```

7. Postprocess/clean the volume
Since we sample density values across the entire 3D volume of the cube, we clean the volume via slicing the it using the axis-aligned bounding box (AABB).
```bash
python scripts\post_process_vol.py --cfg_path cfg\<scene_name>.yml --visualize sliced
```

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="docs/pre_forest.PNG" alt="Pre Forest" width="400px"><br>
        <sub><b>Before Slicing</b></sub>
      </td>
      <td align="center">
        <img src="docs/post_forest.PNG" alt="Post Forest" width="400px"><br>
        <sub><b>After Slicing</b></sub>
      </td>
    </tr>
  </table>
</p>

8. Create the volume format for Paraview

```bash
python scripts\post_process_vol.py --cfg_path cfg\<scene_name>.yml --visualize sliced
```