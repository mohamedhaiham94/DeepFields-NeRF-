#!/usr/bin/env python3
"""
Binary Occupancy Volume Extraction from Trained NeRF

This script extracts a binary occupancy volume from a trained NeRF model by:
1. Loading the trained fine model from checkpoint
2. Generating a uniform 3D grid in the specified bounds
3. Querying density values (and optionally RGB colors) at each grid point in batches
4. Applying Otsu's thresholding to convert to binary occupancy
5. Saving the result as .npy or .pth file with optional visualization

Usage:
    # Basic binary occupancy extraction
    python extract_occupancy_volume.py --checkpoint path/to/model.pth --output occupancy.npy
    
    # Extract colored occupancy volume with RGB values
    python extract_occupancy_volume.py --checkpoint path/to/model.pth --output colored_occupancy.pth --extract-rgb
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import json
import time
from typing import Tuple, Optional, Union

# Import the model architecture (assuming it's in the same directory)
try:
    from model import get_nerf_models, OptimizedVanillaNeRF
except ImportError:
    print("Error: Could not import model.py. Make sure it's in the same directory.")
    exit(1)

# Optional imports for advanced features
try:
    from skimage.filters import threshold_otsu
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available. Using percentile-based thresholding.")
    SKIMAGE_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available. Visualization disabled.")
    MATPLOTLIB_AVAILABLE = False


def load_nerf_model(checkpoint_path: str, device: str = 'cuda') -> OptimizedVanillaNeRF:
    """
    Load the trained NeRF fine model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        Loaded fine NeRF model
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model with default parameters (adjust if needed)
    ngp =True
    if ngp:
        from instantngp import InstantNGP
        model_fine = InstantNGP("large")
    else:       
        _, model_fine = get_nerf_models(device=device)
    
    # Load the fine model state dict
    if 'nerf_fine_state_dict' in checkpoint:
        model_fine.load_state_dict(checkpoint['nerf_fine_state_dict'])
        print(f"Loaded fine model from iteration {checkpoint.get('iteration', 'unknown')}")
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
    x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
    
    # Flatten and stack to get [N, 3] array
    grid_points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
    
    print(f"Generated 3D grid: {resolution}³ = {len(grid_points):,} points")
    print(f"Bounds: [{min_bound:.2f}, {max_bound:.2f}]³")
    
    return grid_points


def query_density_and_rgb_batch(model: OptimizedVanillaNeRF, 
                               points: torch.Tensor, 
                               batch_size: int = 8192,
                               device: str = 'cuda',
                               show_progress: bool = True):
    """
    Query density and RGB values for a large set of points in batches.
    
    Args:
        model: Trained NeRF model
        points: Points to query [N, 3]
        batch_size: Number of points to process per batch
        device: Device for computation
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (density values [N, 1], RGB values [N, 3])
    """
    model.eval()
    total_points = len(points)
    all_densities = []
    all_rgbs = []
    
    # Create progress bar
    pbar = tqdm(range(0, total_points, batch_size), 
                desc="Querying densities and RGB", 
                disable=not show_progress)
    
    # We need dummy view directions for RGB computation
    # Use a canonical view direction (0, 0, -1) for all points
    canonical_view = torch.tensor([0.0, 0.0, -1.0], device=device)
    
    with torch.no_grad():
        for i in pbar:
            end_idx = min(i + batch_size, total_points)
            batch_points = points[i:end_idx].to(device)
            batch_size_actual = end_idx - i
            
            # Create view directions (same for all points in batch)
            batch_views = canonical_view.unsqueeze(0).expand(batch_size_actual, -1)
            
            # Query both RGB and density using full forward pass
            rgb, density = model(batch_points, batch_views)
            
            # Move back to CPU to save GPU memory
            all_densities.append(density.cpu())
            all_rgbs.append(rgb.cpu())
            
            # Update progress bar with current batch info
            pbar.set_postfix({
                'batch': f"{i//batch_size + 1}/{(total_points-1)//batch_size + 1}",
                'points': f"{end_idx}/{total_points}"
            })
    
    return torch.cat(all_densities, dim=0), torch.cat(all_rgbs, dim=0)


def query_density_batch(model: OptimizedVanillaNeRF, 
                       points: torch.Tensor, 
                       batch_size: int = 8192,
                       device: str = 'cuda',
                       show_progress: bool = True) -> torch.Tensor:
    """
    Query density values for a large set of points in batches.
    
    Args:
        model: Trained NeRF model
        points: Points to query [N, 3]
        batch_size: Number of points to process per batch
        device: Device for computation
        show_progress: Whether to show progress bar
        
    Returns:
        Density values [N, 1]
    """
    model.eval()
    total_points = len(points)
    all_densities = []
    
    # Create progress bar
    pbar = tqdm(range(0, total_points, batch_size), 
                desc="Querying densities", 
                disable=not show_progress)
    
    with torch.no_grad():
        for i in pbar:
            end_idx = min(i + batch_size, total_points)
            batch_points = points[i:end_idx].to(device)
            
            # Query density (we don't need features for occupancy)
            density, _ = model.query_density(batch_points)
            
            # Move back to CPU to save GPU memory
            all_densities.append(density.cpu())
            
            # Update progress bar with current batch info
            pbar.set_postfix({
                'batch': f"{i//batch_size + 1}/{(total_points-1)//batch_size + 1}",
                'points': f"{end_idx}/{total_points}"
            })
    
    return torch.cat(all_densities, dim=0)


def apply_otsu_threshold(densities: torch.Tensor, 
                        fallback_percentile: float = 50.0) -> Tuple[torch.Tensor, float]:
    """
    Apply Otsu's thresholding to convert densities to binary occupancy.
    
    Args:
        densities: Density values [N, 1] or [N]
        fallback_percentile: Percentile to use if Otsu fails
        
    Returns:
        Binary occupancy [N] and threshold value used
    """
    densities_np = densities.flatten().numpy()
    
    if SKIMAGE_AVAILABLE:
        try:
            # Apply Otsu's method
            threshold = threshold_otsu(densities_np)
            print(f"Otsu threshold: {threshold:.6f}")
        except Exception as e:
            print(f"Otsu thresholding failed: {e}")
            print(f"Using {fallback_percentile}th percentile instead")
            threshold = np.percentile(densities_np, fallback_percentile)
    else:
        # Fallback to percentile-based thresholding
        threshold = np.percentile(densities_np, fallback_percentile)
        print(f"Using {fallback_percentile}th percentile threshold: {threshold:.6f}")
    
    # Apply threshold
    binary_occupancy = densities.flatten() > threshold
    
    # Print statistics
    occupied_ratio = binary_occupancy.float().mean().item()
    print(f"Occupied voxels: {occupied_ratio:.1%} ({binary_occupancy.sum().item():,}/{len(binary_occupancy):,})")
    
    return binary_occupancy, threshold


def save_colored_occupancy_volume(occupancy: torch.Tensor,
                                 rgb_values: torch.Tensor,
                                 output_path: str,
                                 resolution: int,
                                 bounds: Tuple[float, float],
                                 threshold: float,
                                 metadata: Optional[dict] = None):
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
        'resolution': int(resolution),
        'bounds': [float(bounds[0]), float(bounds[1])],
        'threshold': float(threshold),
        'occupied_ratio': float(occupancy.float().mean().item()),
        'total_voxels': int(len(occupancy)),
        'occupied_voxels': int(occupancy.sum().item()),
        'has_rgb': True,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if metadata:
        save_metadata.update(metadata)
    
    # Save based on file extension
    if output_path.suffix.lower() == '.npy':
        # Save occupancy and RGB as separate files
        np.save(output_path, occupancy_volume.numpy().astype(np.uint8))
        
        # Save RGB volume
        rgb_path = output_path.with_name(output_path.stem + '_rgb.npy')
        np.save(rgb_path, (rgb_volume.numpy() * 255).astype(np.uint8))
        
        # Save metadata as JSON
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(save_metadata, f, indent=2)
        
        print(f"Saved occupancy volume: {output_path}")
        print(f"Saved RGB volume: {rgb_path}")
        print(f"Saved metadata: {metadata_path}")
        
    elif output_path.suffix.lower() == '.pth':
        # Save as PyTorch tensor with metadata
        torch.save({
            'occupancy_volume': occupancy_volume.byte(),
            'rgb_volume': (rgb_volume * 255).byte(),
            'metadata': save_metadata
        }, output_path)
        
        print(f"Saved colored occupancy volume with metadata: {output_path}")
    
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")


def save_occupancy_volume(occupancy: torch.Tensor, 
                         output_path: str,
                         resolution: int,
                         bounds: Tuple[float, float],
                         threshold: float,
                         metadata: Optional[dict] = None):
    """
    Save the binary occupancy volume with metadata.
    
    Args:
        occupancy: Binary occupancy [N] (flattened)
        output_path: Path to save the volume
        resolution: Grid resolution
        bounds: Bounds used for sampling
        threshold: Threshold value used
        metadata: Additional metadata to save
    """
    output_path = Path(output_path)
    
    # Reshape to 3D volume
    volume = occupancy.reshape(resolution, resolution, resolution)
    
    # Prepare metadata (convert all values to JSON-serializable types)
    save_metadata = {
        'resolution': int(resolution),
        'bounds': [float(bounds[0]), float(bounds[1])],
        'threshold': float(threshold),
        'occupied_ratio': float(occupancy.float().mean().item()),
        'total_voxels': int(len(occupancy)),
        'occupied_voxels': int(occupancy.sum().item()),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if metadata:
        save_metadata.update(metadata)
    
    # Save based on file extension
    if output_path.suffix.lower() == '.npy':
        # Save as numpy array
        np.save(output_path, volume.numpy().astype(np.uint8))
        
        # Save metadata as JSON
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(save_metadata, f, indent=2)
        
        print(f"Saved occupancy volume: {output_path}")
        print(f"Saved metadata: {metadata_path}")
        
    elif output_path.suffix.lower() == '.pth':
        # Save as PyTorch tensor with metadata
        torch.save({
            'occupancy_volume': volume.byte(),
            'metadata': save_metadata
        }, output_path)
        
        print(f"Saved occupancy volume with metadata: {output_path}")
    
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")


def visualize_colored_volume(occupancy: torch.Tensor,
                           rgb_values: torch.Tensor,
                           resolution: int,
                           bounds: Tuple[float, float],
                           max_points: int = 10000):
    """
    Create a 3D visualization of the colored occupancy volume.
    
    Args:
        occupancy: Binary occupancy [N] (flattened)
        rgb_values: RGB values [N, 3] (flattened)
        resolution: Grid resolution
        bounds: Bounds used for sampling
        max_points: Maximum points to plot (for performance)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping visualization.")
        return
    
    print("Creating 3D colored visualization...")
    
    # Get occupied voxel indices
    occupied_indices = torch.where(occupancy)[0]
    
    if len(occupied_indices) == 0:
        print("No occupied voxels to visualize.")
        return
    
    # Subsample if too many points
    if len(occupied_indices) > max_points:
        subsample_indices = torch.randperm(len(occupied_indices))[:max_points]
        occupied_indices = occupied_indices[subsample_indices]
        print(f"Subsampled to {max_points} points for visualization")
    
    # Convert flat indices to 3D coordinates
    z_coords = occupied_indices % resolution
    y_coords = (occupied_indices // resolution) % resolution
    x_coords = occupied_indices // (resolution * resolution)
    
    # Convert to world coordinates
    min_bound, max_bound = bounds
    coord_range = max_bound - min_bound
    
    x_world = min_bound + (x_coords.float() / (resolution - 1)) * coord_range
    y_world = min_bound + (y_coords.float() / (resolution - 1)) * coord_range
    z_world = min_bound + (z_coords.float() / (resolution - 1)) * coord_range
    
    # Get RGB colors for occupied voxels
    occupied_colors = rgb_values[occupied_indices].numpy()
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot occupied voxels with their colors
    ax.scatter(x_world.numpy(), y_world.numpy(), z_world.numpy(), 
              c=occupied_colors, alpha=0.8, s=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Colored Occupancy Volume\n{resolution}³ grid, {len(occupied_indices):,} occupied voxels')
    
    plt.tight_layout()
    plt.show()


def visualize_volume(occupancy: torch.Tensor, 
                    resolution: int,
                    bounds: Tuple[float, float],
                    max_points: int = 10000):
    """
    Create a 3D visualization of the occupancy volume.
    
    Args:
        occupancy: Binary occupancy [N] (flattened)
        resolution: Grid resolution
        bounds: Bounds used for sampling
        max_points: Maximum points to plot (for performance)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping visualization.")
        return
    
    print("Creating 3D visualization...")
    
    # Get occupied voxel indices
    occupied_indices = torch.where(occupancy)[0]
    
    if len(occupied_indices) == 0:
        print("No occupied voxels to visualize.")
        return
    
    # Subsample if too many points
    if len(occupied_indices) > max_points:
        subsample_indices = torch.randperm(len(occupied_indices))[:max_points]
        occupied_indices = occupied_indices[subsample_indices]
        print(f"Subsampled to {max_points} points for visualization")
    
    # Convert flat indices to 3D coordinates
    z_coords = occupied_indices % resolution
    y_coords = (occupied_indices // resolution) % resolution
    x_coords = occupied_indices // (resolution * resolution)
    
    # Convert to world coordinates
    min_bound, max_bound = bounds
    coord_range = max_bound - min_bound
    
    x_world = min_bound + (x_coords.float() / (resolution - 1)) * coord_range
    y_world = min_bound + (y_coords.float() / (resolution - 1)) * coord_range
    z_world = min_bound + (z_coords.float() / (resolution - 1)) * coord_range
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot occupied voxels
    ax.scatter(x_world.numpy(), y_world.numpy(), z_world.numpy(), 
              c='red', alpha=0.6, s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Binary Occupancy Volume\n{resolution}³ grid, {len(occupied_indices):,} occupied voxels')
    
    plt.tight_layout()
    plt.show()


def analyze_scene_bounds(transforms_path: str) -> Tuple[float, float]:
    """
    Analyze camera poses to suggest reasonable scene bounds.
    
    Args:
        transforms_path: Path to transforms.json file
        
    Returns:
        Suggested (min_bound, max_bound) for the scene
    """
    try:
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        
        # Extract camera positions
        positions = []
        for frame in transforms['frames']:
            transform_matrix = np.array(frame['transform_matrix'])
            position = transform_matrix[:3, 3]  # Translation component
            positions.append(position)
        
        positions = np.array(positions)
        
        # Calculate bounds with some padding
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        
        # Add padding (20% of range)
        range_size = max_pos - min_pos
        padding = range_size * 0.2
        
        scene_min = min_pos - padding
        scene_max = max_pos + padding
        
        # Suggest a cubic bound
        overall_min = scene_min.min()
        overall_max = scene_max.max()
        
        print(f"Analyzed {len(positions)} camera poses:")
        print(f"Camera position range: X[{min_pos[0]:.3f}, {max_pos[0]:.3f}], "
              f"Y[{min_pos[1]:.3f}, {max_pos[1]:.3f}], Z[{min_pos[2]:.3f}, {max_pos[2]:.3f}]")
        print(f"Suggested cubic bounds: [{overall_min:.3f}, {overall_max:.3f}]")
        
        return overall_min, overall_max
        
    except Exception as e:
        print(f"Could not analyze scene bounds: {e}")
        print("Using default bounds [-1, 1]")
        return -1.0, 1.0


def main():
    parser = argparse.ArgumentParser(description="Extract binary occupancy volume from trained NeRF")
    
    # Required arguments
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to the trained NeRF checkpoint (.pth file)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output path for the occupancy volume (.npy or .pth)')
    
    # Optional arguments
    parser.add_argument('--resolution', '-r', type=int, default=128,
                       help='Grid resolution (default: 128)')
    parser.add_argument('--bounds', nargs=2, type=float, default=[-1.0, 1.0],
                       help='Sampling bounds as min max (default: -1.0 1.0)')
    parser.add_argument('--batch-size', '-b', type=int, default=8192,
                       help='Batch size for density queries (default: 8192)')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--transforms', '-t', type=str, default=None,
                       help='Path to transforms.json to analyze scene bounds')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Show 3D visualization of the result')
    parser.add_argument('--percentile-threshold', type=float, default=50.0,
                       help='Percentile threshold if Otsu fails (default: 50.0)')
    parser.add_argument('--extract-rgb', action='store_true',
                       help='Extract RGB colors along with density (creates colored occupancy volume)')
    parser.add_argument('--view-direction', nargs=3, type=float, default=[0.0, 0.0, -1.0],
                       help='View direction for RGB extraction (default: 0.0 0.0 -1.0)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.resolution <= 0:
        raise ValueError("Resolution must be positive")
    
    if args.bounds[0] >= args.bounds[1]:
        raise ValueError("Invalid bounds: min must be less than max")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Analyze scene bounds if transforms file provided
    if args.transforms:
        suggested_bounds = analyze_scene_bounds(args.transforms)
        print(f"Consider using bounds: --bounds {suggested_bounds[0]:.3f} {suggested_bounds[1]:.3f}")
    
    try:
        # Step 1: Load the trained model
        print("\n" + "="*50)
        print("STEP 1: Loading trained NeRF model")
        print("="*50)
        model = load_nerf_model(args.checkpoint, device=str(device))
        
        # Step 2: Generate 3D grid
        print("\n" + "="*50)
        print("STEP 2: Generating 3D sampling grid")
        print("="*50)
        grid_points = generate_3d_grid(tuple(args.bounds), args.resolution)
        
        # Step 3: Query densities and RGB (if requested)
        print("\n" + "="*50)
        if args.extract_rgb:
            print("STEP 3: Querying density and RGB values")
            print("="*50)
            # Update the canonical view direction based on user input
            canonical_view = torch.tensor(args.view_direction, dtype=torch.float32)
            canonical_view = canonical_view / torch.norm(canonical_view)  # Normalize
            
            # Temporarily store the canonical view in the function
            original_canonical_view = None
            if hasattr(query_density_and_rgb_batch, '__code__'):
                # Modify the function to use custom view direction
                def custom_query_density_and_rgb_batch(model, points, batch_size=8192, device='cuda', show_progress=True):
                    model.eval()
                    total_points = len(points)
                    all_densities = []
                    all_rgbs = []
                    
                    pbar = tqdm(range(0, total_points, batch_size), 
                                desc="Querying densities and RGB", 
                                disable=not show_progress)
                    
                    with torch.no_grad():
                        for i in pbar:
                            end_idx = min(i + batch_size, total_points)
                            batch_points = points[i:end_idx].to(device)
                            batch_size_actual = end_idx - i
                            
                            # Use custom view direction
                            batch_views = canonical_view.to(device).unsqueeze(0).expand(batch_size_actual, -1)
                            
                            # Query both RGB and density using full forward pass
                            rgb, density = model(batch_points, batch_views)
                            
                            # Move back to CPU to save GPU memory
                            all_densities.append(density.cpu())
                            all_rgbs.append(rgb.cpu())
                            
                            pbar.set_postfix({
                                'batch': f"{i//batch_size + 1}/{(total_points-1)//batch_size + 1}",
                                'points': f"{end_idx}/{total_points}"
                            })
                    
                    return torch.cat(all_densities, dim=0), torch.cat(all_rgbs, dim=0)
                
                densities, rgb_values = custom_query_density_and_rgb_batch(model, grid_points, args.batch_size, str(device))
            else:
                densities, rgb_values = query_density_and_rgb_batch(model, grid_points, args.batch_size, str(device))
            
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
            
        else:
            print("STEP 3: Querying density values")
            print("="*50)
            densities = query_density_batch(model, grid_points, args.batch_size, str(device))
            rgb_values = None
            
            # Print density statistics
            print(f"Density statistics:")
            print(f"  Min: {densities.min().item():.6f}")
            print(f"  Max: {densities.max().item():.6f}")
            print(f"  Mean: {densities.mean().item():.6f}")
            print(f"  Std: {densities.std().item():.6f}")
        
        # Step 4: Apply thresholding
        print("\n" + "="*50)
        print("STEP 4: Applying binary thresholding")
        print("="*50)
        binary_occupancy, threshold = apply_otsu_threshold(densities, args.percentile_threshold)
        
        # Step 5: Save results
        print("\n" + "="*50)
        print("STEP 5: Saving occupancy volume")
        print("="*50)
        
        metadata = {
            'checkpoint_path': str(args.checkpoint),
            'device_used': str(device),
            'batch_size': int(args.batch_size),
            'extract_rgb': bool(args.extract_rgb),
            'density_stats': {
                'min': float(densities.min().item()),
                'max': float(densities.max().item()),
                'mean': float(densities.mean().item()),
                'std': float(densities.std().item())
            }
        }
        
        if args.extract_rgb:
            metadata['view_direction'] = args.view_direction
            metadata['rgb_stats'] = {
                'min': float(rgb_values.min().item()),
                'max': float(rgb_values.max().item()),
                'mean': float(rgb_values.mean().item()),
                'std': float(rgb_values.std().item())
            }
            save_colored_occupancy_volume(binary_occupancy, rgb_values, args.output, args.resolution, 
                                        tuple(args.bounds), threshold, metadata)
        else:
            save_occupancy_volume(binary_occupancy, args.output, args.resolution, 
                                tuple(args.bounds), threshold, metadata)
        
        # Step 6: Optional visualization
        if args.visualize:
            print("\n" + "="*50)
            print("STEP 6: Creating visualization")
            print("="*50)
            if args.extract_rgb:
                visualize_colored_volume(binary_occupancy, rgb_values, args.resolution, tuple(args.bounds))
            else:
                visualize_volume(binary_occupancy, args.resolution, tuple(args.bounds))
        
        print("\n" + "="*50)
        print("EXTRACTION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Output saved to: {args.output}")
        if args.extract_rgb:
            print("✓ Colored occupancy volume with RGB values extracted")
        else:
            print("✓ Binary occupancy volume extracted")
        
    except Exception as e:
        print(f"\nError during extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

