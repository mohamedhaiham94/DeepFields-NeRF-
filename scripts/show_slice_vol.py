import argparse
from omegaconf import OmegaConf
import torch
from vispy import scene, app
import numpy as np
import os

def slice_aabb(points, rgbs, aabb_meta):
    # Filter using AABB
    aabb_min = np.array(aabb_meta["aabb_min"])
    aabb_max = np.array(aabb_meta["aabb_max"])

    inside_x = (points[:, 0] >= aabb_min[0]) & (points[:, 0] <= aabb_max[0])
    inside_y = (points[:, 1] >= aabb_min[1]) & (points[:, 1] <= aabb_max[1])
    # inside_y = (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    # inside_z = (points[:, 2] >= aabb_min[2]) & (points[:, 2] <= aabb_max[2])
    inside_z = (points[:, 2] >= -0.1) & (points[:, 2] <= aabb_max[2])
    mask = inside_x & inside_y & inside_z

    points = points[mask]
    rgbs = rgbs[mask]

    return points, rgbs


def visualize(points, rgbs):
    canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", show=True)
    view = canvas.central_widget.add_view()

    # Add scatter plot with RGB colors
    scatter = scene.visuals.Markers()
    scatter.set_data(points, size=2.0, edge_width=0.0, face_color=rgbs)
    scatter.set_gl_state("translucent")  # optional: disable depth testing
    view.add(scatter)
    # Add axis
    axis = scene.visuals.XYZAxis(parent=view.scene)
    # Camera setup
    view.camera = scene.TurntableCamera(
        fov=45, azimuth=30, elevation=30, distance=4.0, center=(0, 0, 0)
    )
    # Show scene
    app.run()


def save_volume(points, rgbs, aabb_meta, sliced_vol_path):
    torch.save(
        {"points_normalized": points, "rgbs": rgbs, "aabb_meta": aabb_meta},
        sliced_vol_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default=None)
    parser.add_argument(
        "--visualize",
        type=str,
        default=None,
        choices=["raw", "sliced"],
        help="Choose which volume to visualize: 'raw' or 'sliced'",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_path)

    if os.path.exists(cfg.sliced_vol_path) and args.visualize == "sliced":
        print("Loading sliced volume...")
        sliced_data = torch.load(cfg.sliced_vol_path, weights_only=False)
        points = sliced_data["points_normalized"]
        rgbs = sliced_data["rgbs"]
    else:
        print("Generating volume")
        data = torch.load(cfg.vol_extracted_dir)
        volume = data["occupancy_volume"]  # shape: [512, 512, 512]
        rgb_volume = data["rgb_volume"]  # shape: [512, 512, 512, 3]
        rgb_volume = rgb_volume / rgb_volume.max()

        print(f"RGB volume range: [{rgb_volume.min()}, {rgb_volume.max()}]")
        
        points = np.argwhere(volume.numpy())
        rgbs = rgb_volume[points[:, 0], points[:, 1], points[:, 2]].numpy()  # [N, 3]

        # Normalize coordinates to [-1, 1]
        res = volume.shape[0]
        points = (points / (res - 1)) * 2 - 1  # [N, 3]

        aabb_meta = None
        if cfg.aabb_slice:
            print("Slice volume with aabb")
            print(f"Transform: {cfg.transforms_json}")
            transform = OmegaConf.load(cfg.transforms_json)

            aabb_meta = transform.scene_aabb
            points, rgbs = slice_aabb(points, rgbs, aabb_meta)

            save_volume(points, rgbs, aabb_meta, cfg.sliced_vol_path)
            print(f"Sliced volume saved to: {cfg.sliced_vol_path}")

    visualize(points, rgbs)