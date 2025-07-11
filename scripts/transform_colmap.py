import argparse
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R


def load_camera_poses_txt(path):
    """Load camera poses from COLMAP images.txt file."""
    poses = []

    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or line == "":
            i += 1
            continue

        # Parse image line
        elems = line.split()
        if len(elems) < 10:
            i += 1
            continue

        image_id = int(elems[0])
        qw, qx, qy, qz = map(float, elems[1:5])
        tx, ty, tz = map(float, elems[5:8])
        camera_id = int(elems[8])
        image_name = elems[9]

        # Read the points2D line (next line)
        points2d_line = ""
        if i + 1 < len(lines):
            points2d_line = lines[i + 1].strip()
        
        # Skip to next image (skip the points2D line)
        i += 2

        poses.append(
            {
                "image_id": image_id,
                "quat": np.array([qw, qx, qy, qz]),
                "trans": np.array([tx, ty, tz]),
                "camera_id": camera_id,
                "image_name": image_name,
                "points2d_line": points2d_line,  # Store original points2D data
            }
        )

    return poses


def load_3d_points_txt(path):
    """Load 3D points from a COLMAP points3D.txt file."""
    points = []
    point_data = []
    
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            elems = line.strip().split()
            if len(elems) < 7:
                continue
            
            point_id = int(elems[0])
            x, y, z = map(float, elems[1:4])
            r, g, b = map(int, elems[4:7])
            error = float(elems[7])
            
            # Store the complete point data for reconstruction
            point_data.append({
                'id': point_id,
                'xyz': np.array([x, y, z]),
                'rgb': np.array([r, g, b]),
                'error': error,
                'track': elems[8:]  # Track information
            })
            
            points.append([x, y, z])
    
    return np.array(points), point_data


def compute_centroid(points, lower_percentile=2.0, upper_percentile=98.0):
    """Compute centroid of 3D points excluding outliers using percentile clipping."""
    if len(points) == 0:
        raise ValueError("Cannot compute centroid for empty point cloud")

    # Mask points within percentile range per axis
    mins = np.percentile(points, lower_percentile, axis=0)
    maxs = np.percentile(points, upper_percentile, axis=0)

    # Filter points within bounds
    mask = np.all((points >= mins) & (points <= maxs), axis=1)
    filtered_points = points[mask]

    if len(filtered_points) == 0:
        raise ValueError("No points left after filtering for centroid computation")

    centroid = np.mean(filtered_points, axis=0)
    return centroid


def compute_average_camera_direction(poses):
    """
    Compute the average camera viewing direction from camera poses.
    
    In COLMAP, the camera coordinate system has:
    - X pointing right
    - Y pointing down  
    - Z pointing forward (viewing direction)
    
    The camera-to-world rotation matrix R_cw transforms points from camera to world coordinates.
    The camera's viewing direction in world coordinates is the third column of R_cw.
    """
    viewing_directions = []
    
    for pose in poses:
        quat = pose['quat']  # [qw, qx, qy, qz]
        # Convert to scipy format [qx, qy, qz, qw]
        R_cw = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        
        # Camera viewing direction in world coordinates (third column of rotation matrix)
        viewing_dir = R_cw[:, 2]  # Z-axis of camera in world coordinates
        viewing_directions.append(viewing_dir)
    
    # Compute average direction
    viewing_directions = np.array(viewing_directions)
    avg_direction = np.mean(viewing_directions, axis=0)
    
    # Normalize the average direction
    avg_direction = avg_direction / np.linalg.norm(avg_direction)
    
    print(f"Average camera viewing direction: [{avg_direction[0]:.6f}, {avg_direction[1]:.6f}, {avg_direction[2]:.6f}]")
    return avg_direction


def compute_rotation_to_align_z_axis(target_direction):
    """
    Compute rotation matrix to align the positive Z-axis with the target direction.
    
    Args:
        target_direction: 3D vector representing the desired direction for positive Z-axis
        
    Returns:
        3x3 rotation matrix that transforms the coordinate system such that 
        the positive Z-axis points in the target direction
    """
    target_direction = target_direction / np.linalg.norm(target_direction)
    z_axis = np.array([0, 0, 1])
    
    # If target direction is already aligned with Z-axis, return identity
    if np.allclose(target_direction, z_axis):
        print("Target direction already aligned with Z-axis, no rotation needed")
        return np.eye(3)
    
    # If target direction is opposite to Z-axis, rotate 180 degrees around X-axis
    if np.allclose(target_direction, -z_axis):
        print("Target direction opposite to Z-axis, rotating 180 degrees around X-axis")
        return R.from_euler('x', 180, degrees=True).as_matrix()
    
    # Compute rotation axis (cross product of current Z and target direction)
    rotation_axis = np.cross(z_axis, target_direction)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Compute rotation angle
    cos_angle = np.dot(z_axis, target_direction)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # Create rotation matrix using axis-angle representation
    rotation_matrix = R.from_rotvec(angle * rotation_axis).as_matrix()
    
    print(f"Rotation angle: {np.degrees(angle):.2f} degrees")
    print(f"Rotation axis: [{rotation_axis[0]:.6f}, {rotation_axis[1]:.6f}, {rotation_axis[2]:.6f}]")
    
    return rotation_matrix


def transform_camera_poses(poses, translation_vector, rotation_matrix=None):
    """
    Transform camera poses by applying a translation and optional rotation to the world coordinate system.
    
    When we shift the world coordinate system by -translation_vector, 
    the camera positions in the new coordinate system become:
    C_new = C_old - translation_vector
    
    And the new translation vectors become:
    t_new = -R @ C_new = -R @ (C_old - translation_vector) = t_old + R @ translation_vector
    
    When we apply a rotation R_global to the world coordinate system:
    - The new camera rotation matrix becomes: R_new = R_old @ R_global^T
    - The new translation vector becomes: t_new = R_global @ t_old
    """
    transformed_poses = []
    
    for pose in poses:
        quat = pose['quat']  # [qw, qx, qy, qz]
        t = pose['trans']
        
        # Step 1: Convert w2c to c2w
        R_wc = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        R_cw = R_wc.T
        C = -R_cw @ t  # camera center in world coords

        # Step 2: Apply transformation in world frame
        C_new = C - translation_vector
        if rotation_matrix is not None:
            C_new = rotation_matrix @ C_new
            R_cw_new = rotation_matrix @ R_cw

            # Re-orthogonalize to ensure valid rotation
            U, _, Vt = np.linalg.svd(R_cw_new)
            R_cw_new = U @ Vt
        else:
            R_cw_new = R_cw

        # Step 3: Convert back to w2c
        R_wc_new = R_cw_new.T
        t_new = -R_wc_new @ C_new

        quat_new = R.from_matrix(R_wc_new).as_quat()  # [qx, qy, qz, qw]
        quat_new = np.array([quat_new[3], quat_new[0], quat_new[1], quat_new[2]])  # [qw, qx, qy, qz]

        new_pose = pose.copy()
        new_pose['quat'] = quat_new
        new_pose['trans'] = t_new
        transformed_poses.append(new_pose)
    
    return transformed_poses

def transform_points(points, translation_vector, rotation_matrix=None):
    """
    Transform 3D points by subtracting the translation vector and applying optional rotation.
    
    Args:
        points: Nx3 array of 3D points or single 3D point
        translation_vector: 3D vector to subtract from points
        rotation_matrix: Optional 3x3 rotation matrix to apply after translation
        
    Returns:
        Transformed points
    """
    
    # Apply translation
    transformed = points - translation_vector
    
    # Apply rotation if provided
    if rotation_matrix is not None:
        transformed = (rotation_matrix @ transformed.T).T
    
    return transformed


def write_points3d_txt(point_data, output_path):
    """Write transformed points to points3D.txt format."""
    with open(output_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(point_data)}\n")
        
        for point in point_data:
            x, y, z = point['xyz']
            r, g, b = point['rgb']
            track_str = ' '.join(point['track'])
            f.write(f"{point['id']} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {point['error']:.6f} {track_str}\n")


def write_images_txt(poses, output_path):
    """Write transformed camera poses to images.txt format."""
    with open(output_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(poses)}\n")
        
        for pose in poses:
            qw, qx, qy, qz = pose['quat']
            tx, ty, tz = pose['trans']
            
            # Write image data line
            f.write(f"{pose['image_id']} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} "
                   f"{tx:.6f} {ty:.6f} {tz:.6f} {pose['camera_id']} {pose['image_name']}\n")
            
            # Write points2D line (preserve original if available, otherwise empty)
            if 'points2d_line' in pose and pose['points2d_line']:
                f.write(f"{pose['points2d_line']}\n")
            else:
                f.write("\n")  # Empty line for points2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_path)
    
    workspace = Path(cfg.workspace) / "colmap_text"
    points3d_txt = workspace / "points3d.txt"
    images_txt = workspace / "images.txt"

    # Load data
    points, point_data = load_3d_points_txt(points3d_txt)
    poses = load_camera_poses_txt(images_txt)
    print(f"Loaded {len(points)} 3D points from {points3d_txt}")
    print(f"Loaded {len(poses)} camera poses from {images_txt}")
    
    # Step 1: Translate scene to center at origin
    print("\n=== Step 1: Centering scene at origin ===")
    centroid = compute_centroid(points)
    
    # Transform 3D points
    print("Transforming 3D points...")
    for point in point_data:
        point['xyz'] = point['xyz'] - centroid
    
    # Transform camera poses (translation only)
    print("Transforming camera poses...")
    translated_poses = transform_camera_poses(poses, centroid)
    
    # Step 2: Rotate scene to align Z-axis with average camera direction
    print("\n=== Step 2: Aligning Z-axis with camera direction ===")
    avg_camera_direction = compute_average_camera_direction(translated_poses)
    rotation_matrix = compute_rotation_to_align_z_axis(avg_camera_direction)
    
    # Apply rotation to 3D points
    print("Applying rotation to 3D points...")
    for point in point_data:
        point['xyz'] = transform_points(point['xyz'].reshape(1, -1), 
                                      np.zeros(3), rotation_matrix).flatten()
    
    # Apply rotation to camera poses
    print("Applying rotation to camera poses...")
    final_poses = []
    for pose in translated_poses:
        # Apply rotation to the already translated pose
        rotated_pose = transform_camera_poses([pose], np.zeros(3), rotation_matrix)[0]
        final_poses.append(rotated_pose)

    print(f"Input poses: {len(poses)}")
    print(f"Final transformed poses: {len(final_poses)}")
    
    # Verify pose count consistency
    if len(final_poses) != len(poses):
        print(f"WARNING: Pose count mismatch! Input: {len(poses)}, Output: {len(final_poses)}")
    else:
        print("Pose count preserved correctly")
    
    # Write output files
    print("\n=== Writing output files ===")
    print("Writing transformed points3D.txt...")
    write_points3d_txt(point_data, points3d_txt)
    print("Writing transformed images.txt...")
    write_images_txt(final_poses, images_txt)
    
    # Print summary
    print(f"\nTransformation summary:")
    print(f"- Translated scene by centroid: [{-centroid[0]:.6f}, {-centroid[1]:.6f}, {-centroid[2]:.6f}]")
    print(f"- Rotated to align Z-axis with average camera direction: [{avg_camera_direction[0]:.6f}, {avg_camera_direction[1]:.6f}, {avg_camera_direction[2]:.6f}]")
    print(f"- Final Z-axis should point towards the average camera viewing direction")
    print(f"- Preserved {len(final_poses)} camera poses")


if __name__ == "__main__":
    main()

