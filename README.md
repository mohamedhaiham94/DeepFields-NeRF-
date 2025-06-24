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
│   ├── extract_occupancy_volume.py # Extracts colored occupancy volume from trained model
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
python -m  optimized.normalize_with_aabb --cfg_path path/to/cfg.yml
```

1. precompute the rays for the whole dataset of images and store them into a ```$Scence_ray_data.npz ```
```python
python -m  optimized.precompute_rays --cfg_path path/to/cfg.yml
```

1. train the nerf model
```$Scence_ray_data.npz ```
```python
python -m  optimized.train --cfg_path path/to/cfg.yml
python optimized\train.py --cfg_path cfg\blender_scene.yml  
```

1. extract volume with the pretrained network
```python
python optimized\extract_occupancy_volume.py --checkpoint path\to\trained_model --output colored_occupancy_{scene_name}.pth --resolution 512 --extract-rgb
```