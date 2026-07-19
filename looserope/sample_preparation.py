"""
Sample preparation utilities for LooseRoPE pipeline.
Extracts masks, saliency maps, trimaps, and coordinate samples from input images.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from scipy.ndimage import distance_transform_edt
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import os


def get_zero_opacity_mask(image_path):
    """
    Given a path to an image, returns a binary numpy array mask where pixels with opacity=0 are 1 (i.e. the "hole"),
    and all others are 0.
    """
    with Image.open(image_path) as im:
        # Ensure the image has an alpha channel
        im_rgba = im.convert("RGBA")
        alpha = np.array(im_rgba)[:, :, 3]
        # Mask: 1 where alpha == 0, 0 otherwise
        mask = (alpha == 0).astype(np.uint8)
    return mask


def get_random_points(mask, num_points, inside=True, seed=42):
    """
    Sample random points from inside or outside a mask region.

    Args:
        mask: Binary mask array
        num_points: Number of points to sample
        inside: If True, sample from inside mask (where mask==1), else outside
        seed: Random seed for reproducibility

    Returns:
        List of [y, x] coordinate pairs
    """
    mask_np = np.array(mask)
    if inside:
        coords = np.argwhere(mask_np == 1)
    else:
        coords = np.argwhere(mask_np == 0)
    rng = np.random.default_rng(seed)
    if len(coords) >= num_points:
        return rng.choice(coords, size=num_points, replace=False).tolist()
    else:
        return coords.tolist()


def save_points(points, path):
    """Save coordinate points to JSON file."""
    with open(path, 'w') as f:
        json.dump(points, f)


def build_predictor():
    """Build Detectron2 predictor for instance segmentation and feature extraction."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return DefaultPredictor(cfg)


@torch.no_grad()
def fpn_feature_saliency(predictor: DefaultPredictor, img_bgr):
    """
    Compute feature map saliency from FPN backbone.

    Returns:
        Feature map saliency [H,W] in [0,1] (upsampled to the predictor's input size)
    """
    model = predictor.model
    device = next(model.parameters()).device

    # Use the predictor's augment + normalization so feature shapes match
    aug_img = predictor.aug.get_transform(img_bgr).apply_image(img_bgr)
    img_chw = torch.as_tensor(aug_img.transpose(2, 0, 1)).to(device)
    inputs = [{"image": img_chw, "height": img_bgr.shape[0], "width": img_bgr.shape[1]}]

    # Preprocess + backbone forward (same as GeneralizedRCNN)
    images = model.preprocess_image(inputs)             # ImageList
    feats = model.backbone(images.tensor)               # dict: {"p2":T,...}, each [B,C,h,w]
    # pick a level (P2: highest resolution)
    f = feats["p2"]                                     # [1,C,h,w]
    # simple saliency: channel energy
    sal = f.abs().mean(dim=1, keepdim=True)             # [1,1,h,w]
    # normalize and upsample to input image
    sal = (sal - sal.amin()) / (sal.amax() - sal.amin() + 1e-8)
    sal = F.interpolate(sal, size=img_chw.shape[-2:], mode="bilinear", align_corners=False)[0, 0]  # [H',W']
    return sal


def get_detectron_foreground_mask(img_path, mask_path):
    """
    Extract foreground mask using Detectron2 instance segmentation.

    Args:
        img_path: Path to input image
        mask_path: Path to mask .npy file

    Returns:
        Binary foreground mask (H, W) with values 0 or 255
    """
    img = Image.open(img_path).convert("RGB")
    mask = np.load(mask_path)

    # Resize mask to match the original image dimensions
    img_w, img_h = img.size
    mask_img_full = Image.fromarray(mask)
    mask_img_full = mask_img_full.resize((img_w, img_h), resample=Image.NEAREST)
    mask_full = np.array(mask_img_full)

    # Set pixels outside of the mask to zero in the original image
    img_np = np.array(img)
    if mask_full.ndim == 2:
        mask_full_3c = np.stack([mask_full]*3, axis=-1)
    else:
        mask_full_3c = mask_full
    img_masked = img_np * mask_full_3c
    img_masked_cv2 = cv2.cvtColor(np.array(img_masked), cv2.COLOR_RGB2BGR)

    pred = build_predictor()
    outputs = pred(img_masked_cv2)

    masks = outputs["instances"].pred_masks.cpu().numpy()  # [N, H, W]
    foreground_mask = np.any(masks, axis=0).astype(np.uint8) * 255

    return foreground_mask


def get_detectron_saliency(img_path, mask_path, zero_opacity_mask, gaussian_blur=False):
    """
    Compute saliency map using Detectron2 FPN features.

    Args:
        img_path: Path to input image
        mask_path: Path to mask .npy file
        zero_opacity_mask: Binary mask of transparent pixels
        gaussian_blur: Whether to apply Gaussian blur to the saliency map

    Returns:
        Saliency map downsampled to 64x64, normalized to [0,1]
    """
    img = Image.open(img_path).convert("RGB")
    mask = np.load(mask_path)

    # Remove pixels where zero_opacity_mask is positive from the 'mask'
    if zero_opacity_mask is not None:
        # Resize zero_opacity_mask to match 'mask' shape
        if zero_opacity_mask.shape != mask.shape:
            zero_opacity_mask_resized = cv2.resize(
                zero_opacity_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        else:
            zero_opacity_mask_resized = zero_opacity_mask

        # Set pixels in mask to 0 where zero_opacity_mask_resized > 0
        mask = mask.copy()
        mask[zero_opacity_mask_resized > 0] = 0

    # Resize mask to match the original image dimensions
    img_w, img_h = img.size
    mask_img_full = Image.fromarray(mask)
    mask_img_full = mask_img_full.resize((img_w, img_h), resample=Image.NEAREST)
    mask_full = np.array(mask_img_full)

    # Set pixels outside of the mask to zero in the original image
    img_np = np.array(img)
    if mask_full.ndim == 2:
        mask_full_3c = np.stack([mask_full]*3, axis=-1)
    else:
        mask_full_3c = mask_full
    img_masked = img_np * mask_full_3c
    img_masked_cv2 = cv2.cvtColor(np.array(img_masked), cv2.COLOR_RGB2BGR)

    pred = build_predictor()
    sal = fpn_feature_saliency(pred, img_masked_cv2).cpu().numpy()

    mask_resized = cv2.resize(mask_full, (sal.shape[1], sal.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Set elements outside of the mask to the minimal value within the mask
    mask_inside = mask_resized > 0
    if np.any(mask_inside):
        min_val_in_mask = sal[mask_inside].min()
    else:
        min_val_in_mask = sal.min()
    sal_out = sal.copy()
    sal_out[~mask_inside] = min_val_in_mask

    # normalize to [0,1]
    sal_out = (sal_out - sal_out.min()) / (sal_out.max() - sal_out.min())

    # Gradually dampen values near the edges using a distance transform
    # Compute distance from each pixel to the nearest zero (edge of mask)
    edge_dist = distance_transform_edt(mask_resized)
    # Normalize distances to [0, 1] (0 at edge, 1 at farthest point from edge)
    if edge_dist.max() > 0:
        edge_weight = edge_dist / edge_dist.max()
    else:
        edge_weight = edge_dist

    # Optionally, sharpen the falloff by raising to a power (e.g., 1.5 or 2)
    edge_weight = edge_weight ** 0.6

    # downscale to 64x64
    sal_out = cv2.resize(sal_out, (64, 64), interpolation=cv2.INTER_AREA)

    # if gaussian blur is enabled, apply it
    if gaussian_blur:
        sal_out = cv2.GaussianBlur(sal_out, (5, 5), 0)

    # normalize to [0,1]
    sal_out = (sal_out - sal_out.min()) / (sal_out.max() - sal_out.min())

    return sal_out


# Trimap constants
TRIMAP_LOW = 1
TRIMAP_MID = 2
TRIMAP_HIGH = 3


def get_saliency_trimap(saliency_map, mask_path):
    """
    Given a saliency map and a path to a binary mask, assign trimap values:
    - Lower third of saliency values within the mask -> TRIMAP_LOW
    - Middle third -> TRIMAP_MID
    - Top third -> TRIMAP_HIGH

    Returns a trimap of the same shape as saliency_map.
    """
    # Load mask
    mask = np.load(mask_path)
    # Resize mask to match saliency_map if needed
    if mask.shape != saliency_map.shape:
        mask = cv2.resize(mask.astype(np.float32), (saliency_map.shape[1], saliency_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.uint8)

    # Get saliency values inside the mask
    sal_in_mask = saliency_map[mask > 0]
    if sal_in_mask.size == 0:
        # If mask is empty, return all LOW
        return np.full_like(saliency_map, TRIMAP_LOW, dtype=np.uint8)

    # Compute thresholds for thirds
    sorted_sal = np.sort(sal_in_mask)
    n = len(sorted_sal)
    low_th = sorted_sal[n // 3]
    mid_th = sorted_sal[2 * n // 3]

    # Ensure trimap is zero outside the mask (not TRIMAP_LOW)
    trimap = np.full_like(saliency_map, TRIMAP_LOW, dtype=np.uint8)
    trimap[mask == 0] = 0
    # Middle third: > low_th and <= mid_th
    trimap[(saliency_map > low_th) & (saliency_map <= mid_th) & (mask > 0)] = TRIMAP_MID
    # Top third: > mid_th
    trimap[(saliency_map > mid_th) & (mask > 0)] = TRIMAP_HIGH

    return trimap


def prepare_sample(input_image_path, original_image_path=None, example_folder=None, num_points=5, seed=42, verbose=True):
    """
    Prepare all required files for a sample in the example folder.

    The crop mask can be provided directly as ``crop_mask.npy``. If it is
    missing, ``original_image_path`` is used to derive it from the input image.

    This function creates:
    - crop_mask.npy: Binary mask showing the cropped region
    - outside.json: Random coordinate samples outside the mask
    - inside.json: Random coordinate samples inside the mask
    - saliency.npy: Saliency map computed from Detectron2 features
    - trimap.npy: Trimap derived from saliency values
    - foreground_mask.npy: Foreground mask extracted using Detectron2 instance segmentation

    Args:
        input_image_path: Path to the input image (with transparent regions or modifications)
        original_image_path: Optional path to the original unmodified image
        example_folder: Folder where output files will be saved
        num_points: Number of random points to sample for inside/outside coordinates
        seed: Random seed for reproducibility
        verbose: Whether to print progress messages

    Returns:
        Dictionary with paths to all created files
    """
    if example_folder is None:
        example_folder = os.path.dirname(input_image_path)

    if verbose:
        print(f"[PREP] Preparing sample in folder: {example_folder}")

    # Create output folder if it doesn't exist
    os.makedirs(example_folder, exist_ok=True)

    # Define output paths
    crop_mask_path = os.path.join(example_folder, "crop_mask.npy")
    outside_json_path = os.path.join(example_folder, "outside.json")
    inside_json_path = os.path.join(example_folder, "inside.json")
    saliency_path = os.path.join(example_folder, "saliency.npy")
    trimap_path = os.path.join(example_folder, "trimap.npy")
    foreground_mask_path = os.path.join(example_folder, "foreground_mask.npy")

    # 1. Load or generate crop mask
    if os.path.exists(crop_mask_path):
        if verbose:
            print("[PREP] Using existing crop_mask.npy")
        mask = np.load(crop_mask_path)
    else:
        if original_image_path is None:
            raise ValueError(
                "crop_mask.npy is missing and original_image_path was not provided; "
                "provide either original.png or crop_mask.npy."
            )
        from looserope.utils import get_naive_mask

        if verbose:
            print("[PREP] Generating crop mask from original image...")
        mask = get_naive_mask(input_image_path, original_image_path)[-1]
        np.save(crop_mask_path, mask)
        if verbose:
            print("[PREP] ✓ Saved crop_mask.npy")

    mask = np.asarray(mask)

    # 2. Get zero opacity mask
    zero_opacity_mask = get_zero_opacity_mask(input_image_path)

    # 3. Generate and save outside points
    if verbose:
        print("[PREP] Sampling outside coordinates...")
    outside_coords = get_random_points(mask, num_points, inside=False, seed=seed)
    save_points(outside_coords, outside_json_path)
    if verbose:
        print(f"[PREP] ✓ Saved outside.json ({len(outside_coords)} points)")

    # 4. Generate and save inside points
    if verbose:
        print("[PREP] Sampling inside coordinates...")
    inside_coords = get_random_points(mask, num_points, inside=True, seed=seed)
    save_points(inside_coords, inside_json_path)
    if verbose:
        print(f"[PREP] ✓ Saved inside.json ({len(inside_coords)} points)")

    # 5. Generate and save saliency map
    if verbose:
        print("[PREP] Computing saliency map (this may take a moment)...")
    sal = get_detectron_saliency(input_image_path, crop_mask_path, zero_opacity_mask, gaussian_blur=True)
    np.save(saliency_path, sal)
    if verbose:
        print("[PREP] ✓ Saved saliency.npy")

    # 6. Generate and save trimap
    if verbose:
        print("[PREP] Generating trimap...")
    trimap = get_saliency_trimap(sal, crop_mask_path)
    np.save(trimap_path, trimap)
    if verbose:
        print("[PREP] ✓ Saved trimap.npy")

    # 7. Generate and save foreground mask
    if verbose:
        print("[PREP] Extracting foreground mask...")
    foreground_mask = get_detectron_foreground_mask(input_image_path, crop_mask_path)
    np.save(foreground_mask_path, foreground_mask)
    if verbose:
        print("[PREP] ✓ Saved foreground_mask.npy")

    if verbose:
        print("[PREP] ✓ Sample preparation complete!")

    return {
        'crop_mask': crop_mask_path,
        'outside_json': outside_json_path,
        'inside_json': inside_json_path,
        'saliency': saliency_path,
        'trimap': trimap_path,
        'foreground_mask': foreground_mask_path
    }


def check_required_files(example_folder):
    """
    Check if all required files exist in the example folder.

    Args:
        example_folder: Path to the folder to check

    Returns:
        Tuple of (all_exist: bool, missing_files: list)
    """
    required_files = [
        'crop_mask.npy',
        'inside.json',
        'outside.json',
        'saliency.npy',
        'trimap.npy',
        'foreground_mask.npy'
    ]

    missing_files = []
    for filename in required_files:
        filepath = os.path.join(example_folder, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)

    all_exist = len(missing_files) == 0
    return all_exist, missing_files

