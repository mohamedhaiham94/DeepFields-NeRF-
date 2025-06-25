from omegaconf import OmegaConf
import torch
import numpy as np
import tifffile
import napari
import argparse
from pathlib import Path


def create_dir(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")


def create_volumes_from_points(points, rgbs, grid_size, fill_value=255):
    """Create both binary and RGB volumes from point data"""
    # Create empty volumes
    binary_volume = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    rgb_volume = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.uint8)

    # Convert normalized coordinates [-1, 1] to voxel indices [0, grid_size-1]
    indices = ((points + 1.0) * 0.5 * (grid_size - 1)).astype(int)
    indices = np.clip(indices, 0, grid_size - 1)

    # Fill binary volume
    binary_volume[indices[:, 0], indices[:, 1], indices[:, 2]] = fill_value

    # RGB values from [0, 1] -> [0, 255].
    rgb_values = (rgbs * 255).astype(np.uint8)

    # Fill RGB volume
    rgb_volume[indices[:, 0], indices[:, 1], indices[:, 2]] = rgb_values

    return binary_volume, rgb_volume


def save_volume_as_tiff(volume, path, paraview_format=False):
    path = Path(path)

    print(f"Input volume range before saving: [{volume.min()}, {volume.max()}]")
    print(f"Input volume dtype: {volume.dtype}")

    
    if paraview_format:
        print(f"Saving volume in ParaView format to {path}")
        # ParaView format - NO METADATA to avoid interpretation issues
        if volume.ndim == 3:
            # Binary volume: transpose to ZYX
            volume_paraview = np.transpose(volume, (2, 1, 0))
            # Ensure still uint8 after transpose
            volume_paraview = volume_paraview.astype(np.uint8)
            print(
                f"ParaView binary volume range after transpose: [{volume_paraview.min()}, {volume_paraview.max()}]"
            )
            print(f"ParaView binary volume shape: {volume_paraview.shape}")
            print(f"ParaView binary volume dtype: {volume_paraview.dtype}")

            # Save without any metadata that might confuse ParaView
            tifffile.imwrite(
                path,
                volume_paraview,
                imagej=False,
                compression=None,
                photometric="minisblack",
            )
        elif volume.ndim == 4:
            # RGB volume: transpose to ZYXC
            volume_paraview = np.transpose(volume, (2, 1, 0, 3))
            # Ensure still uint8 after transpose
            volume_paraview = volume_paraview.astype(np.uint8)
            print(
                f"ParaView RGB volume range after transpose: [{volume_paraview.min()}, {volume_paraview.max()}]"
            )
            print(f"ParaView RGB volume shape: {volume_paraview.shape}")
            print(f"ParaView RGB volume dtype: {volume_paraview.dtype}")

            # Save without metadata that might confuse ParaView
            tifffile.imwrite(
                path,
                volume_paraview,
                imagej=False,
                compression=None,
                photometric="rgb" if volume_paraview.shape[-1] == 3 else "minisblack",
            )

    else:  
        # Standard TIFF format
        volume_standard = volume.astype(np.uint8)
        tifffile.imwrite(path, volume_standard, imagej=False, compression=None)
        
    print(f"Volume saved as {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export volume data to a tiff format.")
    parser.add_argument(
        "--cfg_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=512,
        help="Size of the grid for the volume data. Default is 512.",
    )
    parser.add_argument(
        "--visualize",
        choices=["binary", "rgb", "both"],
        default=None,
        help="Flag to visualize the volume data using Napari.",
    )
    return parser.parse_args()


def load_volume_data(volume_path):
    print(f"Loading volume data from {volume_path}")
    return torch.load(volume_path, weights_only=False)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg_path)

    print("=" * 40)
    volume_data = load_volume_data(cfg.sliced_vol_path)
    points = volume_data["points_normalized"]
    rgbs = volume_data["rgbs"]

    print(f"Points shape: {points.shape}")  # (N, 3)
    print(f"RGB shape: {rgbs.shape}")  # (N, 3)
    print(f"RGB values range: [{rgbs.min()}, {rgbs.max()}]")
    print(f"Grid size: {args.grid_size}")
    print("=" * 40)

    # Create volumes from points
    binary_volume, rgb_volume = create_volumes_from_points(
        points, rgbs, grid_size=args.grid_size
    )
    print(f"Binary volume shape: {binary_volume.shape}")
    print(f"RGB volume shape: {rgb_volume.shape}")
    print(f"Binary volume range: [{binary_volume.min()}, {binary_volume.max()}]")
    print(f"RGB volume range: [{rgb_volume.min()}, {rgb_volume.max()}]")
    print(f"Binary volume dtype: {binary_volume.dtype}")
    print(f"RGB volume dtype: {rgb_volume.dtype}")
    print(f"Occupied voxels: {np.sum(binary_volume > 0)} / {binary_volume.size}")
    print(
        f"Occupancy ratio: {np.sum(binary_volume > 0) / binary_volume.size * 100:.2f}%"
    )

    # Setup output directories
    base_name = Path(cfg.sliced_vol_path).stem
    output_dir = Path(cfg.output_dir)
    napari_dir = output_dir / "napari"
    paraview_dir = output_dir / "paraview"
    for dir_path in [napari_dir, paraview_dir]:
        create_dir(dir_path)

    # Save files in Napari format
    print("\nSaving Napari format files...")
    save_volume_as_tiff(binary_volume, napari_dir / f"{base_name}_binary.tif")
    save_volume_as_tiff(rgb_volume, napari_dir / f"{base_name}_rgb.tif")

    # Save files in ParaView format (.tiff extension)
    print("\nSaving ParaView format files...")
    print("Saving binary volume for ParaView...")
    save_volume_as_tiff(binary_volume, paraview_dir / f"{base_name}_binary.tiff", paraview_format=True)
    print("\nSaving RGB volume for ParaView...")
    save_volume_as_tiff(rgb_volume, paraview_dir / f"{base_name}_rgb.tiff", paraview_format=True)
    

if __name__ == "__main__":
    main()
