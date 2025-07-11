import argparse
import json
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from vispy import app, scene
from vispy.scene import visuals

from scipy.spatial.transform import Rotation as R
import open3d as o3d
import yaml


def rotation_matrix_x(theta):
    """
    Rotation matrix for rotation around the X-axis by angle theta (in radians).
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def rotation_matrix_y(theta):
    """
    Rotation matrix for rotation around the Y-axis by angle theta (in radians).
    """
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def rotation_matrix_z(theta):
    """
    Rotation matrix for rotation around the Z-axis by angle theta (in radians).
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def draw_cube():
    # [x, y, z]
    return np.array(
        [
            [1, -1, -1],
            [1, 1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, 1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, -1],
            [1, 1, 1],
            [1, -1, 1],
            [1, -1, 1],
            [-1, -1, 1],
            [-1, -1, 1],
            [-1, 1, 1],
            [-1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [-1, 1, 1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, 1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, 1],
        ]
    )


def draw_aabb(aabb_min, aabb_max):
    """
    Generate line segments for visualizing an AABB.

    Args:
        aabb_min: [x_min, y_min, z_min]
        aabb_max: [x_max, y_max, z_max]

    Returns:
        Array of line segments for AABB visualization
    """
    x_min, y_min, z_min = aabb_min
    x_max, y_max, z_max = aabb_max

    # Define the 8 corners of the AABB
    corners = np.array(
        [
            [x_min, y_min, z_min],  # 0
            [x_max, y_min, z_min],  # 1
            [x_max, y_max, z_min],  # 2
            [x_min, y_max, z_min],  # 3
            [x_min, y_min, z_max],  # 4
            [x_max, y_min, z_max],  # 5
            [x_max, y_max, z_max],  # 6
            [x_min, y_max, z_max],  # 7
        ]
    )

    # Define the 12 edges of the AABB
    edges = [
        # Bottom face
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        # Top face
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        # Vertical edges
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    # Create line segments
    lines = []
    for edge in edges:
        lines.extend([corners[edge[0]], corners[edge[1]]])

    return np.array(lines)


def load_3d_points_txt(path) -> dict:
    """Load 3D points from a text file."""
    points = []
    rbgs = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            elems = line.strip().split()
            if len(elems) < 7:
                continue
            x, y, z = map(float, elems[1:4])
            r, g, b = map(int, elems[4:7])
            points.append([x, y, z])
            rbgs.append([r / 255.0, g / 255.0, b / 255.0])

    return np.array(points), np.array(rbgs)


def load_camera_pose_txt(
    pose_path: Path, image_dir: Path, scale=None, center=None
) -> list:
    """
    Load camera poses from COLMAP images.txt.
    Returns a dict: {'image_path', 'transform_matrix' matrix (4x4)}
    transform_matrix is the Camera-to-World transform c2w.

    FIXED: Apply scaling and centering consistently with coordinate transformations.
    """
    frames = []

    with open(pose_path, "r") as f:
        i = 0

        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            i = i + 1

            if i % 2 == 1:

                elems = line.split(" ")
                qw, qx, qy, qz = map(float, elems[1:5])
                tx, ty, tz = map(float, elems[5:8])
                image_name = elems[9]

                # Convert to rotation matrix
                rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
                tvec = np.array([tx, ty, tz])

                # World-to-Camera transform
                w2c = np.eye(4)
                w2c[:3, :3] = rot
                w2c[:3, 3] = tvec
                # Invert to get Camera-to-World transform
                c2w = np.linalg.inv(w2c)

                # Apply scaling and centering if provided
                # FIXED: Apply transformations in correct order
                if scale is not None and center is not None:
                    # Scale and center the translation vector
                    c2w[:3, 3] = scale * (c2w[:3, 3] - center)

                pose = {
                    "file_path": str(image_dir / image_name),
                    "transform_matrix": c2w.tolist(),
                }
                frames.append(pose)

    return frames


def flip_colmap_to_nerf_correct(
    points,
    frames,
    shift=[0, 0, 0],
    angles=[0, 0, 0],
    rot_order=[0, 0, 0],
    rotation_initial=None,
    rotation=True,
):
    """
    FIXED: Correct implementation of COLMAP to NeRF coordinate transformation.

    Method: Transform both points and camera poses consistently using the same
    coordinate transformation matrix.
    """
    # Coordinate transformation matrix: flip Y and Z axes
    flip = np.diag([1, -1, -1])
    flip_4x4 = np.eye(4)
    flip_4x4[:3, :3] = flip

    # Transform 3D points
    points_transformed = points @ flip

    if rotation:
        Rot = (
            rotation_matrix_z(np.radians(angles[2]))
            @ rotation_matrix_y(np.radians(angles[1]))
            @ rotation_matrix_x(np.radians(angles[0]))
        )
        rotation_funcs = [
            rotation_matrix_x(np.radians(angles[0])),
            rotation_matrix_y(np.radians(angles[1])),
            rotation_matrix_z(np.radians(angles[2])),
        ]
        Rot = np.eye(3)

        if rotation_initial is not None:
            Rot = np.array(rotation_initial)
            print(f"Rotation matrix after applying initial rotation:")
            print(Rot)

        for axis in rot_order:
            Rot = rotation_funcs[axis] @ Rot

        I = np.eye(3)
        print(f"Rotation matrix is identity: {np.all(np.abs(Rot - I) < 1e-6)}")

        points_transformed = points_transformed @ Rot.T
        pass

    points_transformed = points_transformed + shift

    # Transform camera poses correctly
    for pose in frames:
        T = np.array(pose["transform_matrix"])
        # Correct transformation: T_new = flip @ T @ flip.T
        T_transformed = flip_4x4 @ T @ flip_4x4.T

        if rotation:
            T_transformed = apply_world_rotation(T_transformed, Rot)

        T_transformed[:3, 3] = T_transformed[:3, 3] + shift
        pose["transform_matrix"] = T_transformed.tolist()

    return points_transformed, frames


def compute_scene_aabb(
    points, aabb_adjust, percentile_bounds=(1.0, 99.0), padding=0.02, remove_upper = True
):
    """
    Compute the Axis-Aligned Bounding Box (AABB) of the scene points.

    Args:
        points: Nx3 array of 3D points in normalized coordinates
        percentile_bounds: (lower, upper) percentiles for robust bounds computation
        padding: Additional padding around the AABB (as fraction of range)

    Returns:
        dict: AABB information with min/max bounds and metadata
    """
    if len(points) == 0:
        raise ValueError("Cannot compute AABB for empty point cloud")

    # Compute robust bounds using percentiles
    lower_percentile, upper_percentile = percentile_bounds
    aabb_min = np.percentile(points, lower_percentile, axis=0)
    aabb_max = np.percentile(points, upper_percentile, axis=0)

    # Add padding
    if padding > 0:
        ranges = aabb_max - aabb_min
        padding_amount = ranges * padding
        aabb_min -= padding_amount
        aabb_max += padding_amount

    # Ensure AABB is within the normalized cube [-1, 1]³
    aabb_min = np.maximum(aabb_min, -1.0)
    aabb_max = np.minimum(aabb_max, 1.0)

    # Compute additional metadata
    center = (aabb_min + aabb_max) / 2.0
    size = aabb_max - aabb_min
    volume = np.prod(size)

    # Compute efficiency metrics
    cube_volume = 8.0  # Volume of [-1,1]³ cube
    volume_efficiency = volume / cube_volume

    # Compute efficiency metrics
    cube_volume = 8.0  # Volume of [-1,1]³ cube
    volume_efficiency = volume / cube_volume

    # aabb_min = aabb_min + np.array(aabb_adjust["aabb_min"])
    aabb_min = np.array([-1, -1, -1]) + np.array(aabb_adjust["aabb_min"])

    # aabb_max = aabb_max + np.array(aabb_adjust["aabb_max"])
    z_offset = 0.1
    z_axis = aabb_max[2] + z_offset
    z_axis = 1 if z_axis > 1 else z_axis
    
    if remove_upper:
        aabb_max = np.array([1, 1, z_axis]) + np.array(aabb_adjust["aabb_max"])
    else:
        aabb_max = np.array([1,1,1])
        
    aabb_info = {
        "aabb_min": aabb_min.tolist(),
        "aabb_max": aabb_max.tolist(),
        "aabb_center": center.tolist(),
        "aabb_size": size.tolist(),
        "aabb_volume": float(volume),
        "volume_efficiency": float(volume_efficiency),
        "percentile_bounds": percentile_bounds,
        "padding": float(padding),
        "num_points": len(points),
    }

    print(f"Scene AABB computed:")
    print(f"  Min bounds: [{aabb_min[0]:.3f}, {aabb_min[1]:.3f}, {aabb_min[2]:.3f}]")
    print(f"  Max bounds: [{aabb_max[0]:.3f}, {aabb_max[1]:.3f}, {aabb_max[2]:.3f}]")
    print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  Size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")
    print(f"  Volume efficiency: {volume_efficiency:.1%} of full cube")

    return aabb_info


def display_points(
    points,
    rgbs=None,
    poses=None,
    aabb_info=None,
    shift=[0, 0, 0],
    size=7,
    clip_to_unit_cube=False,
):
    """
    Display a 3D point cloud with camera poses and AABB using VisPy.

    FIXED: Added scene_scale parameter to make frustum size proportional to scene scale.
    FIXED: Corrected camera forward direction visualization.
    ADDED: AABB visualization support.

    Parameters:
        points: Nx3 array of 3D points.
        rgbs: Nx3 array of RGB colors for each point (optional).
        poses: List of camera poses, each a dict with 'transform_matrix' (4x4 matrix).
        aabb_info: AABB information dictionary (optional).
        shift: Translation to apply to the point cloud.
    """

    points = points + np.array(shift)
    canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black")
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.ArcballCamera(fov=45, distance=3)

    # Draw point cloud
    scatter = visuals.Markers()

    if rgbs is not None and len(rgbs) == len(points):
        scatter.set_data(
            points, face_color=rgbs, size=size, edge_color=None, symbol="o"
        )
    else:
        scatter.set_data(
            points, face_color="blue", size=size, edge_color="black", symbol="o"
        )

    view.add(scatter)

    # Draw camera poses
    if poses is not None:
        for pose in poses:
            T = np.array(pose["transform_matrix"])
            origin = T[:3, 3]
            forward = (
                -T[:3, 2] * 0.1
            )  # Forward camera vector is -Z-axis, we flip it for visualization
            up = T[:3, 1] * 0.05  # Up vector is Y-axis
            right = T[:3, 0] * 0.05  # Right vector is X-axis

            frustum_lines = np.array(
                [origin, origin + forward, origin, origin + up, origin, origin + right]
            )

            colors = np.array(
                [
                    [0, 0, 1],  # Z (forward) → blue
                    [0, 0, 1],
                    [0, 1, 0],  # Y (up) → green
                    [0, 1, 0],
                    [1, 0, 0],  # X (right) → red
                    [1, 0, 0],
                ]
            )

            frustum = visuals.Line(
                pos=frustum_lines, color=colors, method="gl", width=2
            )
            view.add(frustum)

    # Add axis & unit cube
    axis = scene.visuals.XYZAxis(parent=view.scene)
    cube_lines = draw_cube()

    cube = visuals.Line(pos=cube_lines, color="red", method="gl", width=6)
    view.add(cube)

    # Draw AABB if provided
    if aabb_info is not None:
        aabb_min = np.array(aabb_info["aabb_min"]) + np.array(shift)
        aabb_max = np.array(aabb_info["aabb_max"]) + np.array(shift)
        aabb_lines = draw_aabb(aabb_min, aabb_max)

        aabb_visual = visuals.Line(pos=aabb_lines, color="green", method="gl", width=4)
        view.add(aabb_visual)

        print(f"AABB visualization added (green box)")

    # Define key press event
    @canvas.events.key_press.connect
    def on_key_press(event):
        if event.key == "R":
            transform_matrix = view.camera.transform.matrix
            rotation_matrix = transform_matrix[:3, :3]
            print("Current Rotation Matrix (on 'R'):\n", rotation_matrix)
            # Convert to Python list of lists (for YAML)
            print("\nRotation_initial:")
            for row in rotation_matrix:
                formatted_row = ", ".join(f"{val: .8f}" for val in row)
                print(f"  - [{formatted_row}]")

    app.run()


def filter_statistical_outliers(points, nb_neighbors=20, std_ratio=2.0):
    """Filter statistical outliers using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    filtered_points = np.asarray(pcd.points)
    return filtered_points, ind


def filter_radius_outliers(points, nb_points=16, radius=0.05):
    """Filter outliers based on radius search."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    filtered_points = np.asarray(pcd.points)
    return filtered_points, ind


def adaptive_percentile_bounds(points, target_retention=0.95):
    """
    Adaptively determine percentile bounds based on point cloud characteristics.
    
    Args:
        points: Nx3 array of 3D points
        target_retention: Target fraction of points to retain (0.9-0.99)
    
    Returns:
        (lower_percentile, upper_percentile): Percentile bounds
    """
    if len(points) < 100:
        # For small point clouds, be more conservative
        margin = (1.0 - target_retention) / 2.0
        return margin * 100, (1.0 - margin) * 100
    
    # Compute point density statistics
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    
    # Use distance distribution to determine outlier threshold
    median_dist = np.median(distances)
    mad = np.median(np.abs(distances - median_dist))  # Median Absolute Deviation
    
    # Adaptive bounds based on distribution spread
    if mad < median_dist * 0.1:
        # Tight distribution - use more aggressive filtering
        margin = (1.0 - target_retention) / 2.0
        return margin * 100, (1.0 - margin) * 100
    else:
        # Spread distribution - use conservative filtering
        margin = (1.0 - min(target_retention + 0.02, 0.99)) / 2.0
        return margin * 100, (1.0 - margin) * 100


def robust_bbox_computation(points, method="adaptive", **kwargs):
    """
    Compute robust bounding box with multiple strategies.
    
    Args:
        points: Nx3 array of 3D points
        method: "adaptive", "percentile", "statistical", "hybrid"
        **kwargs: Additional parameters for specific methods
    
    Returns:
        center, scale, info_dict
    """
    if len(points) == 0:
        raise ValueError("Cannot compute bounding box for empty point cloud")
    
    info = {"method": method, "original_points": len(points)}
    
    if method == "adaptive":
        # Adaptive percentile selection
        target_retention = kwargs.get("target_retention", 0.95)
        lower, upper = adaptive_percentile_bounds(points, target_retention)
        padding = kwargs.get("padding", 0.1)
        
        mins = np.percentile(points, lower, axis=0)
        maxs = np.percentile(points, upper, axis=0)
        center = (mins + maxs) / 2.0
        
        # Use per-axis scaling for better cube utilization
        ranges = maxs - mins
        max_range = np.max(ranges)
        scale = (2.0 * (1.0 - padding)) / max_range
        
        info.update({
            "percentile_bounds": (lower, upper),
            "padding": padding,
            "ranges": ranges.tolist(),
            "max_range": float(max_range)
        })
        
    elif method == "statistical":
        # Statistical outlier removal first
        nb_neighbors = kwargs.get("nb_neighbors", 20)
        std_ratio = kwargs.get("std_ratio", 2.0)
        padding = kwargs.get("padding", 0.1)
        
        filtered_points, indices = filter_statistical_outliers(
            points, nb_neighbors, std_ratio
        )
        
        if len(filtered_points) < len(points) * 0.5:
            print(f"Warning: Statistical filtering removed {len(points) - len(filtered_points)} points")
        
        mins = np.min(filtered_points, axis=0)
        maxs = np.max(filtered_points, axis=0)
        center = (mins + maxs) / 2.0
        
        ranges = maxs - mins
        max_range = np.max(ranges)
        scale = (2.0 * (1.0 - padding)) / max_range
        
        info.update({
            "filtered_points": len(filtered_points),
            "removed_points": len(points) - len(filtered_points),
            "padding": padding,
            "ranges": ranges.tolist(),
            "max_range": float(max_range)
        })
        
    elif method == "hybrid":
        # Combine statistical filtering with percentile bounds
        nb_neighbors = kwargs.get("nb_neighbors", 20)
        std_ratio = kwargs.get("std_ratio", 2.5)
        target_retention = kwargs.get("target_retention", 0.95)
        padding = kwargs.get("padding", 0.1)
        
        # First pass: statistical filtering
        filtered_points, indices = filter_statistical_outliers(
            points, nb_neighbors, std_ratio
        )
        
        # Second pass: percentile bounds on filtered points
        lower, upper = adaptive_percentile_bounds(filtered_points, target_retention)
        mins = np.percentile(filtered_points, lower, axis=0)
        maxs = np.percentile(filtered_points, upper, axis=0)
        center = (mins + maxs) / 2.0
        
        ranges = maxs - mins
        max_range = np.max(ranges)
        scale = (2.0 * (1.0 - padding)) / max_range
        
        info.update({
            "filtered_points": len(filtered_points),
            "percentile_bounds": (lower, upper),
            "padding": padding,
            "ranges": ranges.tolist(),
            "max_range": float(max_range)
        })
        
    else:  # method == "percentile"
        # Original percentile method with improvements
        lower = kwargs.get("lower", 1.0)
        upper = kwargs.get("upper", 99.0)
        padding = kwargs.get("padding", 0.1)
        
        mins = np.percentile(points, lower, axis=0)
        maxs = np.percentile(points, upper, axis=0)
        center = (mins + maxs) / 2.0
        
        ranges = maxs - mins
        max_range = np.max(ranges)
        scale = (2.0 * (1.0 - padding)) / max_range
        
        info.update({
            "percentile_bounds": (lower, upper),
            "padding": padding,
            "ranges": ranges.tolist(),
            "max_range": float(max_range)
        })
    
    # Validation checks
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError(f"Invalid scale computed: {scale}")
    
    if not np.all(np.isfinite(center)):
        raise ValueError(f"Invalid center computed: {center}")
    
    # Test normalization
    test_points = (points - center) * scale
    test_range = np.max(test_points, axis=0) - np.min(test_points, axis=0)
    max_test_range = np.max(test_range)
    
    info.update({
        "final_scale": float(scale),
        "final_center": center.tolist(),
        "test_max_range": float(max_test_range),
        "normalization_success": bool(max_test_range <= 2.1)  # Allow small margin
    })
    
    print(f"Robust bbox computation ({method}):")
    print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  Scale: {scale:.6f}")
    print(f"  Test max range: {max_test_range:.3f}")
    print(f"  Normalization success: {info['normalization_success']}")
    
    return center, scale, info


def compute_percentile_bbox(points, lower=1.0, upper=99.8, padding=0.17):
    """
    IMPROVED: Compute robust center and scale using percentiles, with validation.
    
    This is a backward-compatible wrapper around the new robust_bbox_computation.
    """
    try:
        center, scale, info = robust_bbox_computation(
            points, 
            method="percentile",
            lower=lower,
            upper=upper,
            padding=padding
        )
        return center, float(scale)
    except Exception as e:
        print(f"Warning: Robust computation failed ({e}), falling back to simple method")
        # Fallback to simple computation
        mins = np.percentile(points, lower, axis=0)
        maxs = np.percentile(points, upper, axis=0)
        center = (mins + maxs) / 2.0
        scene_size = np.max(maxs - mins)
        scale = (2.0 * (1.0 - padding)) / scene_size
        return center, float(scale)


def apply_world_rotation(T, R_new):
    T_new = np.eye(4)
    T_new[:3, :3] = R_new @ T[:3, :3]
    T_new[:3, 3] = R_new @ T[:3, 3]
    return T_new


def load_camera_intrinsics_txt(camera_path: Path, aabb_scale=1) -> dict:
    """
    Load camera intrinsics from COLMAP cameras.txt.
    Assuming SIMPLE_RADIAL model.
    """
    with open(camera_path, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            
        els = line.split(" ")
        camera = {}
        camera["w"] = int(els[2])
        camera["h"] = int(els[3])
        camera["fl_x"] = float(els[4])
        camera["fl_y"] = float(els[4])
        camera["k1"] = 0
        camera["k2"] = 0
        camera["k3"] = 0
        camera["k4"] = 0
        camera["p1"] = 0
        camera["p2"] = 0
        camera["cx"] = camera["w"] / 2
        camera["cy"] = camera["h"] / 2
        camera["is_fisheye"] = False
        if els[1] == "SIMPLE_PINHOLE":
            camera["cx"] = float(els[5])
            camera["cy"] = float(els[6])
        elif els[1] == "PINHOLE":
            camera["fl_y"] = float(els[5])
            camera["cx"] = float(els[6])
            camera["cy"] = float(els[7])
        elif els[1] == "SIMPLE_RADIAL":
            camera["cx"] = float(els[5])
            camera["cy"] = float(els[6])
            camera["k1"] = float(els[7])
        elif els[1] == "RADIAL":
            camera["cx"] = float(els[5])
            camera["cy"] = float(els[6])
            camera["k1"] = float(els[7])
            camera["k2"] = float(els[8])
        elif els[1] == "OPENCV":
            camera["fl_y"] = float(els[5])
            camera["cx"] = float(els[6])
            camera["cy"] = float(els[7])
            camera["k1"] = float(els[8])
            camera["k2"] = float(els[9])
            camera["p1"] = float(els[10])
            camera["p2"] = float(els[11])
        elif els[1] == "SIMPLE_RADIAL_FISHEYE":
            camera["is_fisheye"] = True
            camera["cx"] = float(els[5])
            camera["cy"] = float(els[6])
            camera["k1"] = float(els[7])
        elif els[1] == "RADIAL_FISHEYE":
            camera["is_fisheye"] = True
            camera["cx"] = float(els[5])
            camera["cy"] = float(els[6])
            camera["k1"] = float(els[7])
            camera["k2"] = float(els[8])
        elif els[1] == "OPENCV_FISHEYE":
            camera["is_fisheye"] = True
            camera["fl_y"] = float(els[5])
            camera["cx"] = float(els[6])
            camera["cy"] = float(els[7])
            camera["k1"] = float(els[8])
            camera["k2"] = float(els[9])
            camera["k3"] = float(els[10])
            camera["k4"] = float(els[11])

        K = np.array(
            [
                [camera["fl_x"], 0, camera["cx"]],
                [0, camera["fl_y"], camera["cy"]],
                [0, 0, 1],
            ]
        )
        print(f"Loaded camera intrinsics from {camera_path}:\n{K}")
        camera["K"] = K.tolist()

    return camera


def compute_near_far_bounds(points, poses, min_percentile=0.1, max_percentile=99.9):
    near, far = float("inf"), float("-inf")
    for pose in poses:
        c2w = np.array(pose["transform_matrix"])
        cam_origin = c2w[:3, 3]
        dists = np.linalg.norm(points - cam_origin, axis=1)
        near = min(
            near, np.percentile(dists, min_percentile)
        )  # optionally use percentile for robustness
        far = max(far, np.percentile(dists, max_percentile))
    return float(near), float(far)


def write_transform_to_json(transform, output_path):
    """
    Write the transform dictionary to a JSON file.
    """
    with open(output_path, "w") as f:
        json.dump(transform, f, indent=4)

    print(f"Transform saved to {output_path}")


def main():
    """
    IMPROVED: Main function with robust normalization computation and better error handling.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_path)

    scene_name = cfg.scene_name
    WORKDIR = Path("data", scene_name, "colmap_text")
    points_path = WORKDIR / "points3D.txt"
    poses_path = WORKDIR / "images.txt"
    image_dir = WORKDIR.parent / "images"
    intrinsic_path = WORKDIR / "cameras.txt"

    print("Loading and processing 3D points...")
    points, rgbs = load_3d_points_txt(points_path)
    print(f"Loaded {len(points)} 3D points from {points_path}")

    if len(points) == 0:
        raise ValueError("No 3D points loaded. Check the points3D.txt file.")

    # IMPROVED: Use robust bbox computation with multiple strategies
    normalization_method = cfg.get("normalization_method", "hybrid")
    print(f"Computing normalization parameters using '{normalization_method}' method...")
    
    try:
        if normalization_method == "original":
            # Use original method for backward compatibility
            cleaned_points, ind = filter_statistical_outliers(points)
            center, scale = compute_percentile_bbox(
                cleaned_points,
                lower=cfg.percentile_bbox["lower"],
                upper=cfg.percentile_bbox["upper"],
                padding=cfg.percentile_bbox["padding"],
            )
            normalization_info = {
                "method": "original",
                "filtered_points": len(cleaned_points),
                "removed_points": len(points) - len(cleaned_points)
            }
        else:
            # Use improved robust computation
            center, scale, normalization_info = robust_bbox_computation(
                points,
                method=normalization_method,
                target_retention=cfg.get("target_retention", 0.95),
                padding=cfg.percentile_bbox.get("padding", 0.1),
                nb_neighbors=cfg.get("outlier_nb_neighbors", 20),
                std_ratio=cfg.get("outlier_std_ratio", 2.0)
            )
            
            if not normalization_info["normalization_success"]:
                print("Warning: Normalization validation failed, trying fallback method...")
                center, scale, normalization_info = robust_bbox_computation(
                    points,
                    method="adaptive",
                    target_retention=0.9,
                    padding=0.15
                )
    
    except Exception as e:
        print(f"Error in robust normalization: {e}")
        print("Falling back to simple percentile method...")
        center, scale = compute_percentile_bbox(
            points,
            lower=1.0,
            upper=99.0,
            padding=0.15
        )
        normalization_info = {"method": "fallback", "error": str(e)}

    scale *= cfg.scale
    
    print(f"Normalization center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"Normalization scale: {scale:.6f}")

    # Apply normalization to all points
    points_normalized = (points - center) * scale

    # Load camera data
    print("Loading camera intrinsics and poses...")
    intrinsics = load_camera_intrinsics_txt(intrinsic_path)
    frames = load_camera_pose_txt(poses_path, image_dir, scale, center)
    print(f"Loaded {len(frames)} camera poses from {poses_path}")

    # Apply coordinate transformations
    print("Applying coordinate transformations...")
    points_transformed, frames_transformed = flip_colmap_to_nerf_correct(
        points_normalized,
        frames,
        shift=cfg.shift,
        angles=cfg.angles,
        rot_order=cfg.rot_order,
        rotation_initial=cfg.rotation_initial,
        rotation=cfg.rotation,
    )

    # Compute scene AABB after all transformations
    print("Computing scene AABB...")
    aabb_info = compute_scene_aabb(
        points_transformed,
        aabb_adjust=cfg.aabb_adjust,
        percentile_bounds=(1.0, 99.0),
        padding=0.02,
    )

    # Compute near/far bounds
    print("Computing near/far bounds...")
    near, far = compute_near_far_bounds(
        points_transformed, frames_transformed, max_percentile=99
    )
    print(f"Computed near: {near:.6f}, far: {far:.6f}")

    # Create complete transform dictionary with AABB information
    transform = {
        **intrinsics,
        "near": near,
        "far": far,
        "frames": frames_transformed,
        "scene_aabb": aabb_info,
        "normalization": {
            "original_center": center.tolist(),
            "scale": float(scale),
            "total_points": len(points),
            "method_info": normalization_info,
        },
    }

    # Save to JSON
    output_path = Path(f"transforms_{scene_name}.json")
    write_transform_to_json(transform, output_path)

    print(f"\nSummary:")
    print(f"  Total 3D points: {len(points)}")
    print(f"  Normalization method: {normalization_info.get('method', 'unknown')}")
    print(f"  Camera poses: {len(frames_transformed)}")
    print(f"  Scene AABB volume efficiency: {aabb_info['volume_efficiency']:.1%}")
    print(f"  Near/far bounds: {near:.6f} / {far:.6f}")
    print(f"  Transform saved to: {output_path}")

    # Visualize results
    if cfg.get("visualize", False):
        print("\nLaunching visualization...")
        display_points(points_transformed, rgbs, frames_transformed, aabb_info)


if __name__ == "__main__":
    main()

