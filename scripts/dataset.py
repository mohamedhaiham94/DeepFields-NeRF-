import torch
from torch.utils.data import Dataset

import numpy as np
from pathlib import Path


class OptimizedNeRFRayDataset(Dataset):
    """
    Optimized PyTorch Dataset for precomputed NeRF ray data.

    Key optimizations:
    - Keeps data on CPU, transfers only batches to GPU
    - Supports memory mapping for large datasets
    - Efficient tensor operations
    """

    def __init__(self, data_path, use_memmap=False):
        """
        Initialize the optimized NeRF ray dataset.

        Args:
            data_path: Path to .npz file containing precomputed ray data
            use_memmap: Whether to use memory mapping for large datasets
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.use_memmap = use_memmap

        self._load_data()

    def _load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        print(f"Loading ray from: {self.data_path}")

        if self.use_memmap:
            # Use memory mapping for large datasets
            data = np.load(self.data_path, mmap_mode="r")
            self.rgb = data["rgbs"]
            self.ray_origins = data["rays_o"]
            self.ray_directions = data["rays_d"]
            self.t_near = data["t_near"]
            self.t_far = data["t_far"]
        else:
            # Load to CPU memory
            data = np.load(self.data_path)
            self.rgb = torch.from_numpy(data["rgbs"]).float()
            self.ray_origins = torch.from_numpy(data["rays_o"]).float()
            self.ray_directions = torch.from_numpy(data["rays_d"]).float()
            self.t_near = torch.from_numpy(data["t_near"]).float()
            self.t_far = torch.from_numpy(data["t_far"]).float()

        self.num_rays = len(self.ray_origins)

        if not self.use_memmap:
            t_near_min, t_near_max = float(torch.min(self.t_near)), float(
                torch.max(self.t_near)
            )
            t_far_min, t_far_max = float(torch.min(self.t_far)), float(
                torch.max(self.t_far)
            )
        else:
            t_near_min, t_near_max = float(np.min(self.t_near)), float(
                np.max(self.t_near)
            )
            t_far_min, t_far_max = float(np.min(self.t_far)), float(np.max(self.t_far))

        print(f"Loaded {self.num_rays} rays")
        print(f"t_near range: [{t_near_min:.3f}, {t_near_max:.3f}]")
        print(f"t_far range: [{t_far_min:.3f}, {t_far_max:.3f}]")

    def __len__(self):
        """Return the total number of rays."""
        return self.num_rays

    def __getitem__(self, idx):
        """
        Get a single ray sample.

        Returns CPU tensors - DataLoader will handle GPU transfer efficiently.
        """
        if self.use_memmap:
            # Convert from memory-mapped arrays to tensors
            sample = {
                "ray_origin": torch.from_numpy(self.ray_origins[idx].copy()).float(),
                "ray_direction": torch.from_numpy(
                    self.ray_directions[idx].copy()
                ).float(),
                "rgb": torch.from_numpy(self.rgb[idx].copy()).float(),
                "t_near": torch.from_numpy(self.t_near[idx].copy()).float(),
                "t_far": torch.from_numpy(self.t_far[idx].copy()).float(),
            }
        else:
            # Data already in tensor format
            sample = {
                "ray_origin": self.ray_origins[idx],
                "ray_direction": self.ray_directions[idx],
                "rgb": self.rgb[idx],
                "t_near": self.t_near[idx],
                "t_far": self.t_far[idx],
            }

        return sample
