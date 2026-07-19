import numpy as np
from PIL import Image
import cv2
import json
from skimage.morphology import binary_dilation
import os

def load_coordinates_from_json(json_file_path):
    """Load coordinates from a JSON file.

    Args:
        json_file_path (str): Path to JSON file containing coordinates

    Returns:
        list: List of coordinate pairs [[y, x], [y, x], ...]
    """
    if json_file_path is None:
        return None

    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Coordinate file not found: {json_file_path}")

    with open(json_file_path, 'r') as f:
        coordinates = json.load(f)

    # Validate that coordinates is a list of [y, x] pairs
    if not isinstance(coordinates, list):
        raise ValueError(f"Coordinates must be a list, got {type(coordinates)}")

    for i, coord in enumerate(coordinates):
        if not isinstance(coord, list) or len(coord) != 2:
            raise ValueError(f"Each coordinate must be a list of 2 values [y, x], got {coord} at index {i}")
        if not all(isinstance(val, (int, float)) for val in coord):
            raise ValueError(f"Coordinate values must be numbers, got {coord} at index {i}")

    return coordinates

def load_image(img):
    """Load image from path, numpy array, or PIL Image as numpy array."""
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
        return np.array(img)
    elif isinstance(img, np.ndarray):
        return img
    else:
        # Try to convert PIL Image to numpy
        try:
            return np.array(img)
        except Exception:
            raise ValueError("Input must be a file path, numpy array, or PIL Image.")


def get_naive_mask(collage_img, original_img, threshold=5):
    """
    Loads two images and returns a binary mask that is True (1) everywhere the difference
    exceeds the threshold, and False (0) elsewhere.

    Args:
        collage_img: Input image (file path, numpy array, or PIL Image)
        original_img: Original image (file path, numpy array, or PIL Image)
        threshold: Pixel difference threshold (default: 10). Pixels differing by more than
                  this value (in any channel) will be marked as different.
    """

    collage_arr = load_image(collage_img)
    original_arr = load_image(original_img)

    # Ensure same shape
    if collage_arr.shape != original_arr.shape:
        raise ValueError(f"Image shapes do not match: {collage_arr.shape} vs {original_arr.shape}")

    # Compute absolute difference
    diff = np.abs(collage_arr.astype(np.float32) - original_arr.astype(np.float32))

    # Compute mask: True where any channel difference exceeds threshold
    if collage_arr.ndim == 3:
        mask = np.any(diff > threshold, axis=-1)
    else:
        mask = diff > threshold

    # convert to uint8
    mask = mask.astype(np.uint8) * 255
    # downsample mask to 64x64
    mask = cv2.resize(mask, (64, 64))
    # convert back to bool
    mask = mask > 128
    masks = []
    for i in range(3):
        # dilate mask
        dilated_mask = binary_dilation(mask, np.ones((14 - 4 * i, 14 - 2 * i)))
        masks.append(dilated_mask.astype(np.uint8))
    masks.append(mask.astype(np.uint8))

    return masks