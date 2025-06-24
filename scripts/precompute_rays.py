import argparse
import json
from omegaconf import OmegaConf
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm


def ray_cube_intersection_numpy(
    ray_origins, ray_directions, cube_min=-1.0, cube_max=1.0
):
    """
    Compute ray-cube intersection using NumPy.

    Args:
        ray_origins: Array of shape [N, 3] - ray origins in world space
        ray_directions: Array of shape [N, 3] - ray directions in world space (should be normalized)
        cube_min: Minimum coordinate of the cube (default: -1.0)
        cube_max: Maximum coordinate of the cube (default: 1.0)

    Returns:
        intersects: Boolean array of shape [N] - True if ray intersects cube
        t_near: Array of shape [N] - near intersection parameter (entry point)
        t_far: Array of shape [N] - far intersection parameter (exit point)
    """
    # Ensure inputs are numpy arrays
    ray_origins = np.asarray(ray_origins, dtype=np.float32)
    ray_directions = np.asarray(ray_directions, dtype=np.float32)

    # Handle near-zero direction components to avoid division by zero
    epsilon = 1e-8
    ray_directions = np.where(
        np.abs(ray_directions) < epsilon,
        np.sign(ray_directions) * epsilon,
        ray_directions,
    )

    # Compute intersection parameters for each axis
    t_min_x = (cube_min - ray_origins[:, 0]) / ray_directions[:, 0]
    t_max_x = (cube_max - ray_origins[:, 0]) / ray_directions[:, 0]

    t_min_y = (cube_min - ray_origins[:, 1]) / ray_directions[:, 1]
    t_max_y = (cube_max - ray_origins[:, 1]) / ray_directions[:, 1]

    t_min_z = (cube_min - ray_origins[:, 2]) / ray_directions[:, 2]
    t_max_z = (cube_max - ray_origins[:, 2]) / ray_directions[:, 2]

    # Ensure t_min <= t_max for each axis (swap if direction is negative)
    t_min_x, t_max_x = np.minimum(t_min_x, t_max_x), np.maximum(t_min_x, t_max_x)
    t_min_y, t_max_y = np.minimum(t_min_y, t_max_y), np.maximum(t_min_y, t_max_y)
    t_min_z, t_max_z = np.minimum(t_min_z, t_max_z), np.maximum(t_min_z, t_max_z)

    # Find the intersection interval
    t_near = np.maximum(np.maximum(t_min_x, t_min_y), t_min_z)
    t_far = np.minimum(np.minimum(t_max_x, t_max_y), t_max_z)

    # Ray intersects cube if t_near <= t_far and t_far >= 0
    intersects = (t_near <= t_far) & (t_far >= 0)

    # Ensure t_near is non-negative (clamp to 0 if ray starts inside cube)
    t_near = np.maximum(t_near, 0.0)

    return intersects, t_near, t_far


def get_rays(H, W, K, camera_poses, colmap=True, normalize_directions=True):
    """
    Generate rays for a given camera setup.
        args:
            H (int): Height of the image.
            W (int): Width of the image.
            K (np.ndarray): Camera intrinsic matrix of shape (3, 3).
            camera_poses (np.ndarray): Camera poses of shape (M, 4, 4) where M is the number of cameras.
            colmap (bool): If True, apply the Colmap flip transformation.
    """

    F = np.diag([1, -1, -1]) if colmap else np.eye(3)
    K_inv = np.linalg.inv(K)
    u, v = np.meshgrid(np.arange(0, W), np.arange(0, H))

    u = u.flatten()
    v = v.flatten()
    pixels_hom = np.stack(
        [u, v, np.ones_like(u)], axis=0
    )  # Shape: (3, height * width) = (3, N)

    camera_coords = K_inv @ pixels_hom  # Shape: (3, N)

    # Apply coordinate flip transformation (OpenCV -> OpenGL/NeRF)
    directions = F @ camera_coords  # Shape: (3, N)

    if normalize_directions:
        norms = np.linalg.norm(directions, axis=0, keepdims=True)  # Shape: (1, H*W)
        norms = np.clip(norms, 1e-8, None)  # Avoid division by zero

        directions = directions / norms  # Shape: (3, H*W)

    M = camera_poses.shape[0]  # Number of cameras

    # Extract rotation and translation from camera pose
    rotations = camera_poses[
        :, :3, :3
    ]  # Shape: (M, 3, 3) Rotation matrix of all cameras
    translations = camera_poses[
        :, :3, 3
    ]  # Shape: (M, 3) Translation vector of all cameras

    # Transform ray directions from camera to world coordinates
    # Einsum for proper broadcasting: (M, 3, 3) @ (3, H*W) → (M, 3, H*W)
    rays_d = np.einsum("mij,jn->min", rotations, directions)  # Shape: (M, 3, H*W)

    # Reshape to (M, H, W, 3)
    rays_d = rays_d.transpose(0, 2, 1).reshape(M, H, W, 3)

    # Ray origins are the camera positions (same for all rays from a camera)
    rays_o = translations[:, None, None, :]  # Shape: (M, 1, 1, 3)
    rays_o = np.broadcast_to(rays_o, (M, H, W, 3))  # Shape: (M, H, W, 3)

    return rays_o, rays_d


def load_poses_from_json(file_path, device):
    with open(file_path, "r") as f:
        data = json.load(f)
    frames = data["frames"]
    poses = []
    imgs = []
    for frame in tqdm(frames, desc="Loading poses and images", total=len(frames)):
        img = np.array(Image.open(frame["file_path"]).convert("RGB")).astype(np.float32)
        img /= 255.0  # Normalize to [0, 1]
        pose = frame["transform_matrix"]

        imgs.append(img)
        poses.append(pose)
    return np.array(poses), np.array(imgs)


def save_rays_to_file_npz(rays_o, rays_d, rgbs, filename):
    """
    Save rays and RGB values to a compressed .npz file.

    Args:
        rays_o (np.ndarray): Ray origins of shape (N, H, W, 3).
        rays_d (np.ndarray): Ray directions of shape (N, H, W, 3).
        rgbs (np.ndarray): RGB values of shape (N, H, W, 3).
        filename (str): Path to save the .npz file.
    """
    print("Saving rays data to file...")
    np.savez_compressed(filename, rays_o=rays_o, rays_d=rays_d, rgbs=rgbs)
    print(f"Saved rays data to {filename}")
    print(
        f"{len(rays_o)} rays with origins and directions, and {len(rgbs)} RGB values."
    )


def save_rays_to_file_pth(rays_o, rays_d, rgbs, filename):
    """
    Save rays and RGB values to a .pt file using PyTorch.

    Args:
        rays_o (np.ndarray or torch.Tensor): Ray origins of shape (N, H, W, 3).
        rays_d (np.ndarray or torch.Tensor): Ray directions of shape (N, H, W, 3).
        rgbs (np.ndarray or torch.Tensor): RGB values of shape (N, H, W, 3).
        filename (str): Path to save the .pt file.
    """
    print("Saving rays data to file...")

    # Convert to torch tensors if necessary

    rays_o = torch.from_numpy(rays_o)
    rays_d = torch.from_numpy(rays_d)
    rgbs = torch.from_numpy(rgbs)

    # Save as dictionary
    torch.save({"rays_o": rays_o, "rays_d": rays_d, "rgbs": rgbs}, filename)

    print(f"Saved rays data to {filename}")
    print(
        f"{len(rays_o)} rays with origins and directions, and {len(rgbs)} RGB values."
    )


def precompute_save_ray():
    """
    Example usage of the get_rays function.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_path)

    scene_name = cfg.scene_name
    transforms_file = f"transforms_{scene_name}.json"
    with open(transforms_file, "r") as f:
        data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = data["h"]
    W = data["w"]
    K = data["K"]

    K = np.array(K)
    poses, rgbs = load_poses_from_json(transforms_file, device)
    rays_o, rays_d = get_rays(H, W, K, poses)

    rays_o_flat = rays_o.reshape(-1, 3)
    rays_d_flat = rays_d.reshape(-1, 3)
    rgbs_flat = rgbs.reshape(-1, 3)

    print("Rays for intersection check.")
    print(f"{len(rays_d_flat)} rays with origins and directions.")
    print()

    intersects, t_near, t_far = ray_cube_intersection_numpy(rays_o_flat, rays_d_flat)
    ray_origins = rays_o_flat[intersects]
    ray_directions = rays_d_flat[intersects]
    rgbs = rgbs_flat[intersects]
    t_near = t_near[intersects]
    t_far = t_far[intersects]

    N = len(ray_origins)
    filename = f"{scene_name}_ray_data.npz"
    print(f"{N} rays with origins and directions.")
    print("Saving rays data to file...")
    np.savez_compressed(filename, rays_o=ray_origins, rays_d=ray_directions, rgbs=rgbs, t_near=t_near, t_far = t_far)
    print(f"Saved rays data to {filename}.")


if __name__ == "__main__":
    precompute_save_ray()
