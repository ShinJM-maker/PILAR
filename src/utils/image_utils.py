"""Image processing utilities."""
from pathlib import Path
from typing import Tuple
from PIL import Image


def crop_region(image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    """Crop a region from an image given (x0, y0, x1, y1) in pixel coordinates."""
    x0, y0, x1, y1 = [int(c) for c in bbox]
    return image.crop((x0, y0, x1, y1))


def save_crop(image: Image.Image, bbox: Tuple[float, float, float, float],
              output_path: str | Path) -> str:
    """Crop and save a region from an image."""
    crop = crop_region(image, bbox)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    crop.save(str(output_path))
    return str(output_path)


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")
