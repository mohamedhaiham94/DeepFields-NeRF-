import argparse
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf

def resize_images_in_place(folder, size):
    """
    Resize all images in the folder and overwrite them in-place.

    Parameters:
        folder (str): Path to the folder containing images.
        size (tuple): New size, e.g., (800, 600).
    """
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp")

    folder = Path(folder)
    for filename in folder.iterdir():
        if filename.name.lower().endswith(image_extensions):
            img = Image.open(filename).resize(size, Image.Resampling.LANCZOS)
            img.save(filename)  # Overwrite original
            print(f"Resized and overwritten: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg_path)
    resize_images_in_place(
        folder=cfg.image_dir,
        size=cfg.newSize,
    )