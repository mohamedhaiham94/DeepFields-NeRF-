import argparse
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf


def resize_images(input_folder, output_folder, size):
    """
    Resize all images in the input_folder and save them to the output_folder.

    Parameters:
        input_folder (str): Path to the folder containing images.
        output_folder (str): Path to save resized images.
        size (tuple): New size, e.g., (800, 600).
    """
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp")

    output_folder = Path(output_folder)
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    input_folder = Path(input_folder)
    for filename in input_folder.iterdir():
        if filename.name.lower().endswith(image_extensions):
            img = Image.open(filename).resize(size, Image.Resampling.LANCZOS)

            output_path = output_folder / filename.name
            img.save(output_path)
            print(f"Resized and saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg_path)
    resize_images(
        input_folder=cfg.img_dir_original,
        output_folder=cfg.new_images_dir,
        size=cfg.newSize,
    )
