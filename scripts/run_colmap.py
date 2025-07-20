import argparse
import os
import shutil
import sys
from pathlib import Path
from glob import glob


def do_system(arg, cwd=None):
    print(f"==== running: {arg}")
    err = os.system(f'cd /d "{cwd}" && {arg}') if cwd else os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


def write_dir(path):
    """Create or clear a directory."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    print(f"Creating directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_colmap_binary(colmap_path: str = "tmp/colmap/COLMAP.bat"):
    colmap_path = Path(colmap_path).resolve()
    if not colmap_path.exists():
        print(f"FATAL: COLMAP.bat not found at: {colmap_path}")
        sys.exit(1)
    return str(colmap_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run COLMAP reconstruction pipeline and export to transforms.json")
    parser.add_argument("--workspace", default="data\\blender_scene", type=Path, help="Working directory")
    parser.add_argument("--colmap_matcher", default="exhaustive", choices=["exhaustive", "sequential", "spatial", "transitive", "vocab_tree"])
    parser.add_argument("--colmap_db", default="colmap.db", help="COLMAP database filename")
    parser.add_argument("--colmap_camera_model", default="OPENCV", choices=[
        "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV",
        "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"
    ])
    parser.add_argument("--colmap_camera_params", default="", help="Camera intrinsics (optional)")
    parser.add_argument("--images", default="images", help="Path to input images")
    parser.add_argument("--aabb_scale", default=1, choices=["1", "2", "4", "8", "16", "32", "64", "128"])
    parser.add_argument("--colmap_bat", default="tmp/colmap/COLMAP.bat", help="Path to COLMAP binary")
    parser.add_argument("--vocab_path", default="", help="Optional vocab tree path")
    return parser.parse_args()


def run_colmap(args):
    workspace = args.workspace.resolve()
    print("workspace:", workspace)
    db_path = workspace / args.colmap_db
    images_dir = workspace / args.images
    sparse_dir = workspace / "sparse"
    text_dir = workspace / "colmap_text"

    for dir_path in [sparse_dir, text_dir]:
        write_dir(dir_path)

    colmap_binary = find_colmap_binary(args.colmap_bat)
    colmap_dir = Path(colmap_binary).parent

    print(f"Using COLMAP binary: {colmap_binary}")

    # Feature extraction
    do_system(
        f'"{colmap_binary}" feature_extractor '
        f'--ImageReader.camera_model {args.colmap_camera_model} '
        f'--SiftExtraction.estimate_affine_shape=true '
        f'--SiftExtraction.domain_size_pooling=true '
        f'--ImageReader.single_camera 1 '
        f'--database_path "{db_path}" '
        f'--image_path "{images_dir}"',
        cwd=colmap_dir
    )

    # Feature matching
    match_cmd = (
        f'"{colmap_binary}" {args.colmap_matcher}_matcher '
        f'--SiftMatching.guided_matching=true '
        f'--database_path "{db_path}"'
    )
    if args.vocab_path:
        match_cmd += f' --VocabTreeMatching.vocab_tree_path "{args.vocab_path}"'
    do_system(match_cmd, cwd=colmap_dir)

    # Mapping
    do_system(
        f'"{colmap_binary}" mapper '
        f'--database_path "{db_path}" '
        f'--image_path "{images_dir}" '
        f'--output_path "{sparse_dir}"',
        cwd=colmap_dir
    )

    # Bundle adjustment
    do_system(
        f'"{colmap_binary}" bundle_adjuster '
        f'--input_path "{sparse_dir}/0" '
        f'--output_path "{sparse_dir}/0" '
        f'--BundleAdjustment.refine_principal_point 1',
        cwd=colmap_dir
    )

    # Convert model to TXT
    do_system(
        f'"{colmap_binary}" model_converter '
        f'--input_path "{sparse_dir}/0" '
        f'--output_path "{text_dir}" '
        f'--output_type TXT',
        cwd=colmap_dir
    )

    print("\n" + "="*40)
    print(f"Running COLMAP with:")
    print(f"DB path     : {db_path}")
    print(f"Images path : {images_dir}")
    print(f"Sparse path : {sparse_dir}")
    print(f"Text path   : {text_dir}")
    print("\nCOLMAP pipeline completed successfully.")
    print("="*40 + "\n")


def main():
    args = parse_args()
    run_colmap(args)


if __name__ == "__main__":
    main()
