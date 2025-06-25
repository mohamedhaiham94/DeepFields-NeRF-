import argparse
from html import parser
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import json
import time
from typing import Tuple, Optional, Union

from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from instantngp import InstantNGP


def load_instant_ngp_model(
    checkpoint_path: str, nerf_type: str = "large", device: str = "cuda"
) -> InstantNGP:
    """
    Load the trained instant ngp.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Loaded fine NeRF model
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("\n" + "=" * 50)
    print(f"Loading checkpoint from: {checkpoint_path}")
    print("\n" + "=" * 50)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_fine = InstantNGP(nerf_type)

    # Load the fine model state dict
    if "nerf_fine_state_dict" in checkpoint:
        model_fine.load_state_dict(checkpoint["nerf_fine_state_dict"])
        print(
            f"Loaded fine model from iteration {checkpoint.get('iteration', 'unknown')}"
        )
    else:
        raise KeyError("Fine model state dict not found in checkpoint")

    model_fine.eval()
    return model_fine


def generate_3d_grid(bounds: Tuple[float, float], resolution: int) -> torch.Tensor:
    """
    Generate a uniform 3D grid of points within the specified bounds.

    Args:
        bounds: (min_bound, max_bound) for the cube
        resolution: Number of points along each axis

    Returns:
        Grid points of shape [resolution^3, 3]
    """
    min_bound, max_bound = bounds

    # Create 1D coordinate arrays
    coords = torch.linspace(min_bound, max_bound, resolution)

    # Create 3D meshgrid
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")

    # Flatten and stack to get [N, 3] array
    grid_points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

    print(f"Generated 3D grid: {resolution}³ = {len(grid_points):,} points")
    print(f"Bounds: [{min_bound:.2f}, {max_bound:.2f}]³")

    return grid_points


def volume_extraction(
    model, points, canonical_view, batch_size=8192, device="cuda", show_progress=True
):
    model.eval()
    total_points = len(points)
    all_densities = []
    all_rgbs = []

    pbar = tqdm(range(0, total_points, batch_size), desc="Querying densities and RGB")

    with torch.no_grad():
        for i in pbar:
            end_idx = min(i + batch_size, total_points)
            batch_points = points[i:end_idx].to(device)
            batch_size_actual = end_idx - i

            # Use custom view direction
            batch_views = (
                canonical_view.to(device).unsqueeze(0).expand(batch_size_actual, -1)
            )

            # Query both RGB and density using full forward pass
            rgb, density = model(batch_points, batch_views)

            # Move back to CPU to save GPU memory
            all_densities.append(density.cpu())
            all_rgbs.append(rgb.cpu())

            pbar.set_postfix(
                {
                    "batch": f"{i//batch_size + 1}/{(total_points-1)//batch_size + 1}",
                    "points": f"{end_idx}/{total_points}",
                }
            )

    return torch.cat(all_densities, dim=0), torch.cat(all_rgbs, dim=0)


def apply_otsu_threshold(
    densities: torch.Tensor, fallback_percentile: float = 50.0
) -> Tuple[torch.Tensor, float]:
    """
    Apply Otsu's thresholding to convert densities to binary occupancy.

    Args:
        densities: Density values [N, 1] or [N]
        fallback_percentile: Percentile to use if Otsu fails

    Returns:
        Binary occupancy [N] and threshold value used
    """
    densities_np = densities.flatten().numpy()
    # Apply Otsu's method
    threshold = threshold_otsu(densities_np)
    print(f"Otsu threshold: {threshold:.6f}")

    # Apply threshold
    binary_occupancy = densities.flatten() > threshold

    # Print statistics
    occupied_ratio = binary_occupancy.float().mean().item()
    print(
        f"Occupied voxels: {occupied_ratio:.1%} ({binary_occupancy.sum().item():,}/{len(binary_occupancy):,})"
    )

    return binary_occupancy, threshold


def save_volume(
    occupancy: torch.Tensor,
    rgb_values: torch.Tensor,
    output_path: str,
    resolution: int,
    bounds: Tuple[float, float],
    threshold: float,
    metadata: Optional[dict] = None,
):
    """
    Save the binary occupancy volume with RGB colors and metadata.

    Args:
        occupancy: Binary occupancy [N] (flattened)
        rgb_values: RGB values [N, 3] (flattened)
        output_path: Path to save the volume
        resolution: Grid resolution
        bounds: Bounds used for sampling
        threshold: Threshold value used
        metadata: Additional metadata to save
    """
    output_path = Path(output_path)

    # Reshape to 3D volumes
    occupancy_volume = occupancy.reshape(resolution, resolution, resolution)
    rgb_volume = rgb_values.reshape(resolution, resolution, resolution, 3)

    # Prepare metadata (convert all values to JSON-serializable types)
    save_metadata = {
        "resolution": int(resolution),
        "bounds": [float(bounds[0]), float(bounds[1])],
        "threshold": float(threshold),
        "occupied_ratio": float(occupancy.float().mean().item()),
        "total_voxels": int(len(occupancy)),
        "occupied_voxels": int(occupancy.sum().item()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    save_metadata.update(metadata)

    if output_path.suffix.lower() == ".pth":
        # Save as PyTorch tensor with metadata
        torch.save(
            {
                "occupancy_volume": occupancy_volume.byte(),
                "rgb_volume": (rgb_volume * 255).byte(),
                "metadata": save_metadata,
            },
            output_path,
        )

        print(f"Saved colored occupancy volume with metadata: {output_path}")
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract binary occupancy volume from trained NeRF"
    )

    # Required arguments
    parser.add_argument("--cfg_path", type=str, default=None)
    # Optional arguments
    parser.add_argument(
        "--resolution",
        "-r",
        type=int,
        default=512,
        help="Grid resolution (default: 512)",
    )
    parser.add_argument(
        "--bounds",
        nargs=2,
        type=float,
        default=[-1.0, 1.0],
        help="Sampling bounds as min max (default: -1.0 1.0)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8192,
        help="Batch size for density queries (default: 8192)",
    )
    parser.add_argument(
        "--percentile-threshold",
        type=float,
        default=50.0,
        help="Percentile threshold if Otsu fails (default: 50.0)",
    )
    parser.add_argument(
        "--view-direction",
        nargs=3,
        type=float,
        default=[0.0, 0.0, -1.0],
        help="View direction for RGB extraction (default: 0.0 0.0 -1.0)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.resolution <= 0:
        raise ValueError("Resolution must be positive")

    if args.bounds[0] >= args.bounds[1]:
        raise ValueError("Invalid bounds: min must be less than max")

    return args


def main():
    args = get_args()
    cfg = OmegaConf.load(args.cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Step 1: Load the trained model
        print("\n" + "=" * 50)
        print("STEP 1: Loading trained NeRF model")
        print("=" * 50)
        model = load_instant_ngp_model(cfg.checkpoint, device=str(device))
        print("Model loaded successfully")

        # Step 2: Generate 3D grid
        print("\n" + "=" * 50)
        print("STEP 2: Generating 3D sampling grid")
        print("=" * 50)
        grid_points = generate_3d_grid(tuple(args.bounds), args.resolution)

        # Step 3: Query densities and RGB (if requested)
        print("\n" + "=" * 50)
        print("STEP 3: Querying density and RGB values")
        print("=" * 50)
        # Update the canonical view direction based on user input
        canonical_view = torch.tensor(args.view_direction, dtype=torch.float32)
        canonical_view = canonical_view / torch.norm(canonical_view)  # Normalize
        densities, rgb_values = volume_extraction(
            model, grid_points, canonical_view, args.batch_size, str(device)
        )

        # Print statistics
        print(f"Density statistics:")
        print(f"  Min: {densities.min().item():.6f}")
        print(f"  Max: {densities.max().item():.6f}")
        print(f"  Mean: {densities.mean().item():.6f}")
        print(f"  Std: {densities.std().item():.6f}")

        print(f"RGB statistics:")
        print(f"  Min: {rgb_values.min().item():.6f}")
        print(f"  Max: {rgb_values.max().item():.6f}")
        print(f"  Mean: {rgb_values.mean().item():.6f}")
        print(f"  Std: {rgb_values.std().item():.6f}")

        # Step 4: Apply thresholding
        print("\n" + "=" * 50)
        print("STEP 4: Applying binary thresholding")
        print("=" * 50)
        binary_occupancy, threshold = apply_otsu_threshold(
            densities, args.percentile_threshold
        )

        # Step 5: Save results
        print("\n" + "=" * 50)
        print("STEP 5: Saving occupancy volume")
        print("=" * 50)

        metadata = {
            "checkpoint_path": str(cfg.checkpoint),
            "device_used": str(device),
            "batch_size": int(args.batch_size),
            "density_stats": {
                "min": float(densities.min().item()),
                "max": float(densities.max().item()),
                "mean": float(densities.mean().item()),
                "std": float(densities.std().item()),
            },
            "view_direction": args.view_direction,
            "rgb_stats": {
                "min": float(rgb_values.min().item()),
                "max": float(rgb_values.max().item()),
                "mean": float(rgb_values.mean().item()),
                "std": float(rgb_values.std().item()),
            },
        }

        save_volume(
            binary_occupancy,
            rgb_values,
            cfg.volume_output_path,
            args.resolution,
            tuple(args.bounds),
            threshold,
            metadata,
        )

        print("\n" + "=" * 50)
        print("EXTRACTION COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Output saved to: {cfg.volume_output_path}")
        print("Colored occupancy volume with RGB values extracted")

    except Exception as e:
        print(f"\nError during extraction: {e}")


if __name__ == "__main__":
    main()
