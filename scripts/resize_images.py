# import argparse
# from PIL import Image
# from pathlib import Path
# from omegaconf import OmegaConf

# def resize_images_in_place(folder, size):
#     """
#     Resize all images in the folder and overwrite them in-place.

#     Parameters:
#         folder (str): Path to the folder containing images.
#         size (tuple): New size, e.g., (800, 600).
#     """
#     image_extensions = (".png", ".jpg", ".jpeg", ".bmp")

#     folder = Path(folder)
#     for filename in folder.iterdir():
#         if filename.name.lower().endswith(image_extensions):
#             img = Image.open(filename).resize(size, Image.Resampling.LANCZOS)
#             img.save(filename)  # Overwrite original
#             print(f"Resized and overwritten: {filename}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cfg_path", type=str, default=None)
#     args = parser.parse_args()

#     cfg = OmegaConf.load(args.cfg_path)
#     resize_images_in_place(
#         folder=cfg.image_dir,
#         outdir=cfg.tmp_image_dir,
#         resize=cfg.resize_images,
#         size=cfg.newSize,
#     )

import argparse
import shutil
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf

def process_images(folder, outdir, resize, size):
    """
    Resize or copy images from folder to outdir.

    Parameters:
        folder (str): Path to the folder containing images.
        outdir (str): Path to the output folder.
        resize (bool): Whether to resize images.
        size (tuple): New size, e.g., (800, 600).
    """
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp")

    folder = Path(folder)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for filename in folder.iterdir():
        if filename.suffix.lower() in image_extensions:
            dest_file = outdir / filename.name
            if resize:
                img = Image.open(filename).resize(size, Image.Resampling.LANCZOS)
                img.save(dest_file)
                print(f"Resized and saved: {dest_file}")
            else:
                shutil.copy(filename, dest_file)
                print(f"Copied: {dest_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg_path)
    # print(cfg.tmp_image_dir)
    
    process_images(
        folder=cfg.image_dir,
        outdir=cfg.tmp_image_dir,
        resize=cfg.resize_images,
        size=tuple(cfg.newSize),
    )
