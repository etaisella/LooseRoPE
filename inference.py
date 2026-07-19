import torch
import argparse
import os
import glob
import random
import yaml
import wandb
import time
from datetime import datetime
from PIL import Image

# Monkey-patch for transformers 5.0 compatibility with older diffusers
import transformers.utils
if not hasattr(transformers.utils, 'FLAX_WEIGHTS_NAME'):
    transformers.utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
    print("Applied FLAX_WEIGHTS_NAME monkey-patch for transformers 5.0 compatibility")

from diffusers import FluxKontextPipeline
from looserope.looserope_pipeline import LooseRoPEPipeline
from diffusers.utils import load_image
from looserope.attention_processor import AttentionConfig
import numpy as np
from looserope.looserope_pipeline import callback_store_latents, callback_calculate_metrics
from looserope.sample_preparation import prepare_sample, check_required_files

MODEL_DEFAULTS = {
    "kontext": {
        "guidance_scale": 2.5,
        "true_cfg_scale": 1.0,
        "num_inference_steps": 28,
        "pretrained_path": "black-forest-labs/FLUX.1-Kontext-dev",
    },
    "qwen": {
        "guidance_scale": None,
        "true_cfg_scale": 4.0,
        "num_inference_steps": 50,
        "pretrained_path": "Qwen/Qwen-Image-Edit",
    },
}

DEFAULT_INPUT_PATH = "data/demo/giraffeduck"
DEFAULT_ATTN_CONFIG_FILE = "configs/attn_config.yaml"

def seed_everything(seed):
    """Seed all random number generators for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set deterministic behavior for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeded all random number generators with seed: {seed}")

def initialize_wandb(args, attention_config=None):
    """Initialize Weights & Biases logging with experiment configuration"""
    # Create wandb config from command line arguments
    config = vars(args).copy()

    # Load and add attention config parameters if provided
    if attention_config is not None and args.attn_config_file is not None:
        with open(args.attn_config_file, 'r') as f:
            attn_config_dict = yaml.safe_load(f)
        # Add attention config with prefix to avoid name conflicts
        for key, value in attn_config_dict.items():
            config[f"attn_{key}"] = value

    # Generate run name with current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"

    # Initialize wandb
    wandb.init(project="looserope", config=config, name=run_name)
    return wandb

def find_image_files(folder_path):
    """Find all .png, .jpg, and .jpeg files in a folder (case-insensitive)"""
    image_files = []
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']

    for ext in extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))

    # Remove duplicates and sort
    return sorted(list(set(image_files)))

def resolve_original_path(original_path, fallback_folder=None):
    """Resolve an optional original image path, folder, or colocated original.png."""
    if original_path is None:
        if fallback_folder is not None:
            candidate = os.path.join(fallback_folder, "original.png")
            if os.path.exists(candidate):
                return candidate
        return None
    if os.path.isfile(original_path):
        return original_path
    if os.path.isdir(original_path):
        candidate = os.path.join(original_path, "original.png")
        if os.path.exists(candidate):
            return candidate
    return None


def crop_mask_path_for_image(input_image_path):
    return os.path.join(os.path.dirname(input_image_path), "crop_mask.npy")


def has_crop_mask_for_image(input_image_path):
    return os.path.exists(crop_mask_path_for_image(input_image_path))


def can_prepare_or_run(input_image_path, original_image_path):
    return original_image_path is not None or has_crop_mask_for_image(input_image_path)


def sample_input_path(folder_path, input_filename="input.png"):
    return os.path.join(folder_path, input_filename)


def is_sample_folder(folder_path, input_filename="input.png", original_path=None):
    if not os.path.isdir(folder_path):
        return False
    input_path = sample_input_path(folder_path, input_filename)
    if not os.path.exists(input_path):
        return False
    original_image_path = resolve_original_path(original_path, folder_path)
    return can_prepare_or_run(input_path, original_image_path)


def prepare_missing_sample_files(input_image_path, original_image_path, example_folder, seed, verbose=True):
    all_exist, missing_files = check_required_files(example_folder)
    if all_exist:
        if verbose:
            print(f"[PREP] ✓ All required files found in '{example_folder}'")
        return True

    print(f"[PREP] Missing files: {', '.join(missing_files)}. Preparing sample files...")
    if "crop_mask.npy" in missing_files and original_image_path is None:
        print("[PREP] Error: crop_mask.npy is missing and original_path was not provided.")
        print("[PREP] Provide either original.png or crop_mask.npy.")
        return False

    prepare_sample(input_image_path, original_image_path, example_folder, num_points=5, seed=seed, verbose=verbose)
    return True


def find_subfolders_with_image_pairs(folder_path, input_filename="input.png"):
    """
    Find subfolders that contain the input image and either original.png or crop_mask.npy.
    Returns tuples of (subfolder_path, input_image_path, original_image_path_or_none).
    """
    subfolders_with_pairs = []

    if not os.path.isdir(folder_path):
        return subfolders_with_pairs

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            input_path = os.path.join(item_path, input_filename)
            original_path = os.path.join(item_path, "original.png")
            crop_mask_path = os.path.join(item_path, "crop_mask.npy")

            if os.path.exists(input_path) and (os.path.exists(original_path) or os.path.exists(crop_mask_path)):
                subfolders_with_pairs.append((
                    item_path,
                    input_path,
                    original_path if os.path.exists(original_path) else None,
                ))

    return sorted(subfolders_with_pairs)


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

def remove_crop_background(input_image, original_image, foreground_mask_path):
    """
    Remove crop background by pasting foreground pixels from input onto original.

    Args:
        input_image: PIL Image of the input (cropped/modified) image
        original_image: PIL Image of the original image
        foreground_mask_path: Path to the foreground_mask.npy file

    Returns:
        PIL Image with foreground from input pasted onto original
    """
    # Load foreground mask
    foreground_mask = np.load(foreground_mask_path)

    # Convert images to numpy arrays
    input_np = np.array(input_image)
    original_np = np.array(original_image)

    # Resize foreground mask to match image dimensions if needed
    if foreground_mask.shape[:2] != input_np.shape[:2]:
        from PIL import Image as PILImage
        mask_img = PILImage.fromarray(foreground_mask)
        mask_img = mask_img.resize((input_np.shape[1], input_np.shape[0]), resample=PILImage.NEAREST)
        foreground_mask = np.array(mask_img)

    # Create binary mask (foreground = 1, background = 0)
    binary_mask = (foreground_mask > 127).astype(np.uint8)

    # Expand mask to 3 channels if needed
    if input_np.ndim == 3:
        binary_mask_3c = np.stack([binary_mask] * 3, axis=-1)
    else:
        binary_mask_3c = binary_mask

    # Composite: use input where mask is 1, original where mask is 0
    result_np = np.where(binary_mask_3c, input_np, original_np).astype(np.uint8)

    # Convert back to PIL Image
    result_image = Image.fromarray(result_np, mode='RGB')

    return result_image

def write_timing_summary(output_folder, example_name, timing_data, global_timings=None):
    """Write a neat timing breakdown summary to a txt file."""
    txt_path = os.path.join(output_folder, "timing_summary.txt")
    with open(txt_path, 'w') as f:
        f.write(f"{'='*70}\n")
        f.write(f"  TIMING BREAKDOWN: {example_name}\n")
        f.write(f"{'='*70}\n\n")

        if global_timings:
            f.write(f"--- One-time setup costs (shared across all examples) ---\n")
            for key, val in global_timings.items():
                f.write(f"  {key:<40s} {val:>8.2f}s\n")
            f.write(f"\n")

        f.write(f"--- Pipeline call summary ---\n")
        total = timing_data.get('total_call', 0)
        f.write(f"  {'Total pipeline call':<40s} {total:>8.2f}s\n")
        f.write(f"  {'Prompt encoding':<40s} {timing_data.get('prompt_encoding', 0):>8.2f}s\n")
        f.write(f"  {'Image preprocessing':<40s} {timing_data.get('image_preprocessing', 0):>8.2f}s\n")
        f.write(f"  {'Latent preparation':<40s} {timing_data.get('latent_preparation', 0):>8.2f}s\n")
        f.write(f"  {'Denoising loop (total)':<40s} {timing_data.get('denoising_loop', 0):>8.2f}s\n")
        f.write(f"  {'VAE decode':<40s} {timing_data.get('vae_decode', 0):>8.2f}s\n")
        f.write(f"\n")

        # Per-timestep breakdown
        timestep_times = timing_data.get('timestep_times', [])
        if timestep_times:
            f.write(f"--- Per-timestep breakdown ({len(timestep_times)} steps) ---\n")
            f.write(f"  {'Step':<6s} {'Total':>8s} {'Transformer':>12s} {'Neg CFG':>10s} {'VLM':>8s}\n")
            f.write(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*10} {'-'*8}\n")
            sum_transformer = 0.0
            sum_neg_cfg = 0.0
            sum_vlm = 0.0
            for entry in timestep_times:
                step = entry['step']
                t_total = entry['total']
                t_trans = entry['transformer_forward']
                t_neg = entry['neg_cfg_forward']
                t_vlm = entry['vlm_total']
                sum_transformer += t_trans
                sum_neg_cfg += t_neg
                sum_vlm += t_vlm
                f.write(f"  {step:<6d} {t_total:>8.2f}s {t_trans:>11.2f}s {t_neg:>9.2f}s {t_vlm:>7.2f}s\n")
            f.write(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*10} {'-'*8}\n")
            avg_trans = sum_transformer / len(timestep_times) if timestep_times else 0
            f.write(f"  {'SUM':<6s} {'':>8s} {sum_transformer:>11.2f}s {sum_neg_cfg:>9.2f}s {sum_vlm:>7.2f}s\n")
            f.write(f"  {'AVG':<6s} {'':>8s} {avg_trans:>11.2f}s\n")
            f.write(f"\n")

        # VLM verdict breakdown
        vlm_times = timing_data.get('vlm_times', [])
        vlm_tries = timing_data.get('vlm_tries', 0)
        if vlm_times:
            f.write(f"--- VLM verdict breakdown ({len(vlm_times)} calls, {vlm_tries} retries) ---\n")
            f.write(f"  {'Try':<5s} {'Step':<6s} {'Inference':>10s} {'Offload':>10s} {'Total':>8s} {'Verdict'}\n")
            f.write(f"  {'-'*5} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*12}\n")
            total_vlm_inference = 0.0
            total_vlm_offload = 0.0
            for entry in vlm_times:
                total_vlm_inference += entry['inference_time']
                total_vlm_offload += entry['offload_time']
                f.write(f"  {entry['try']:<5d} {entry['step']:<6d} "
                        f"{entry['inference_time']:>9.2f}s {entry['offload_time']:>9.2f}s "
                        f"{entry['total_time']:>7.2f}s {entry['verdict']}\n")
            f.write(f"  {'-'*5} {'-'*6} {'-'*10} {'-'*10} {'-'*8}\n")
            f.write(f"  {'TOTAL':<12s} {total_vlm_inference:>9.2f}s {total_vlm_offload:>9.2f}s "
                    f"{total_vlm_inference + total_vlm_offload:>7.2f}s\n")
            f.write(f"\n")
        elif timing_data.get('vlm_tries', 0) == 0 and not vlm_times:
            f.write(f"--- VLM: not triggered for this example ---\n\n")

        # x0 prediction times
        x0_times = timing_data.get('x0_prediction_times', [])
        if x0_times:
            f.write(f"--- x0 prediction times ({len(x0_times)} predictions) ---\n")
            total_x0 = 0.0
            for entry in x0_times:
                note = f"  ({entry['note']})" if 'note' in entry else ""
                f.write(f"  Step {entry['step']:<4d} {entry['time']:>8.2f}s{note}\n")
                total_x0 += entry['time']
            f.write(f"  {'TOTAL':<10s} {total_x0:>8.2f}s\n")
            f.write(f"\n")

        # High-level cost breakdown
        f.write(f"{'='*70}\n")
        f.write(f"  HIGH-LEVEL COST BREAKDOWN\n")
        f.write(f"{'='*70}\n")
        denoising_pure = sum(e['transformer_forward'] for e in timestep_times)
        neg_cfg_total = sum(e['neg_cfg_forward'] for e in timestep_times)
        vlm_total = sum(e['total_time'] for e in vlm_times) if vlm_times else 0
        x0_total = sum(e['time'] for e in x0_times) if x0_times else 0
        encoding = timing_data.get('prompt_encoding', 0)
        preprocess = timing_data.get('image_preprocessing', 0)
        latent_prep = timing_data.get('latent_preparation', 0)
        vae = timing_data.get('vae_decode', 0)
        accounted = denoising_pure + neg_cfg_total + vlm_total + x0_total + encoding + preprocess + latent_prep + vae
        other = total - accounted

        items = [
            ("Transformer forward passes", denoising_pure),
            ("Negative CFG forward passes", neg_cfg_total),
            ("VLM verdict (inference + offload)", vlm_total),
            ("x0 predictions", x0_total),
            ("Prompt encoding", encoding),
            ("Image preprocessing", preprocess),
            ("Latent preparation", latent_prep),
            ("VAE decode", vae),
            ("Other (scheduling, callbacks, etc.)", other),
        ]
        for label, val in items:
            pct = (val / total * 100) if total > 0 else 0
            f.write(f"  {label:<40s} {val:>8.2f}s  ({pct:>5.1f}%)\n")
        f.write(f"  {'-'*60}\n")
        f.write(f"  {'TOTAL':<40s} {total:>8.2f}s  (100.0%)\n")
        f.write(f"{'='*70}\n")

    print(f"[TIMING] Summary written to {txt_path}")
    return txt_path


def process_single_image(
        pipe,
        input_image_path,
        original_image_path,
        output_folder,
        prompt,
        guidance_scale,
        true_cfg_scale,
        negative_prompt,
        override=False,
        attention_config=None,
        seed=0,
        log_to_wandb=True,
        reconstruct_input=False,
        reconstruct_original=False,
        log_metrics=False,
        remove_crop_bg=False,
        global_timings=None,
        pipe_extra_kwargs=None,
    ):
    """Process a single image with the pipeline"""
    _t_example_start = time.time()
    # Load input image
    zero_opacity_mask = get_zero_opacity_mask(input_image_path)
    input_image = load_image(input_image_path, convert_method=lambda x: x.convert("RGB"))
    original_image = load_image(original_image_path) if original_image_path is not None else None

    crop_mask_path = crop_mask_path_for_image(input_image_path)

    # Remove crop background if flag is enabled
    if remove_crop_bg:
        if original_image is None:
            print("[REMOVE_BG] Error: --remove_crop_bg requires original_path/original.png")
            return None

        containing_folder = os.path.dirname(input_image_path)
        foreground_mask_path = os.path.join(containing_folder, "foreground_mask.npy")

        if os.path.exists(foreground_mask_path):
            print(f"[REMOVE_BG] Removing crop background using foreground mask...")
            input_image = remove_crop_background(input_image, original_image, foreground_mask_path)
            print(f"[REMOVE_BG] ✓ Background removed, using composite image as input")
        else:
            print(f"[REMOVE_BG] Warning: foreground_mask.npy not found at {foreground_mask_path}")
            print(f"[REMOVE_BG] Continuing without background removal")

    # Fill transparent areas (where mask == 1) with average color
    if np.any(zero_opacity_mask):
        print(f"[KONTEXT] Detected {zero_opacity_mask.sum()} pixels with zero opacity")
        img_rgb = np.array(input_image)

        # Calculate average color from non-transparent pixels (where mask == 0)
        non_transparent_mask = zero_opacity_mask == 0
        if np.any(non_transparent_mask):
            # Get average color from opaque pixels
            opaque_pixels = img_rgb[non_transparent_mask]
            avg_color = opaque_pixels.mean(axis=0).astype(np.uint8)
            print(f"[KONTEXT] Average color of opaque pixels: RGB{tuple(avg_color)}")
        else:
            # If all pixels are transparent, use white
            avg_color = np.array([255, 255, 255], dtype=np.uint8)
            print(f"[KONTEXT] All pixels transparent, using white fill")

        # Fill transparent areas with average color
        transparent_mask = zero_opacity_mask == 1
        img_rgb[transparent_mask] = avg_color

        # Convert back to PIL Image
        input_image = Image.fromarray(img_rgb, mode='RGB')
        print(f"[KONTEXT] ✓ Filled transparent pixels with average color")

    # Generate output filename
    input_filename = os.path.splitext(os.path.basename(input_image_path))[0]
    output_path = os.path.join(output_folder, "output.png")

    # if output path already exists, skip (unless override is enabled)
    if os.path.exists(output_path) and not override:
        print(f"Skipping {input_image_path} because {output_path} already exists")
        return output_path

    # Get coordinates using attention config
    inside_coords = None
    outside_coords = None
    if attention_config is not None:
        if os.path.exists(crop_mask_path):
            mask_np = np.load(crop_mask_path)
        elif original_image is not None:
            from looserope.utils import get_naive_mask
            mask_np = np.array(get_naive_mask(input_image, original_image)[-1])
            np.save(crop_mask_path, mask_np)
            print(f"[PREP] Saved derived crop_mask.npy to {crop_mask_path}")
        else:
            print("[KONTEXT] Error: attention config requires crop_mask.npy or original_path/original.png")
            return None
        inside_coords, outside_coords = attention_config.get_coordinates(mask_np)

    print(f"inside_coords: {inside_coords}")
    print(f"outside_coords: {outside_coords}")

    # Only store attention if config file is provided (which means attention_config is not None)
    if attention_config is not None:
        text_offsets = pipe.get_text_offsets(prompt)
        words_to_save = prompt.split(" ")
        attention_config.set_text_info(text_offsets, words_to_save)
        attention_config.set_masks()
        attention_save_folder = os.path.join(output_folder, "attention_maps")
        if any([
            attention_config.save_attention_maps,
            attention_config.save_norm_maps,
            attention_config.record_attention_locality,
        ]):
            os.makedirs(attention_save_folder, exist_ok=True)
        pipe.set_attn_processor_to_looserope(attention_save_folder, attention_config)

        if getattr(attention_config, "use_simplified_instruction", False):
            if os.path.isfile(crop_mask_path):
                pipe.crop_mask = np.load(crop_mask_path)
                print(f"[VLM] Loaded crop_mask.npy for simplified verdict ({crop_mask_path})")
            else:
                print(f"[VLM] Warning: use_simplified_instruction but crop_mask.npy missing at {crop_mask_path}")

    # Create generator for reproducible results
    generator = torch.Generator(device="cuda").manual_seed(seed)

    if reconstruct_input:
        print("Reconstructing input image...")
        pipe.set_temp_latent_dir(os.path.join(output_folder, "input_latents"))
        image = pipe(
            image=input_image,
            prompt="reconstruct the input image",
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            callback_on_step_end=callback_store_latents
        ).images[0]
        image.save(os.path.join(output_folder, "input_recon.png"))

    if reconstruct_original:
        if original_image is None:
            print("[RECONSTRUCT] Error: --reconstruct_original requires original_path/original.png")
            return None
        print("Reconstructing original image...")
        pipe.set_temp_latent_dir(os.path.join(output_folder, "original_latents"))
        image = pipe(
            image=original_image,
            prompt="reconstruct the input image",
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            callback_on_step_end=callback_store_latents
        ).images[0]
        image.save(os.path.join(output_folder, "original_recon.png"))

    metrics_callback = None
    if log_metrics and log_to_wandb:
        containing_folder = os.path.dirname(input_image_path)
        original_latent_dir = os.path.join(containing_folder, "original_latents")
        input_latent_dir = os.path.join(containing_folder, "input_latents")
        trimap = np.load(os.path.join(containing_folder, "trimap.npy"))
        crop_mask = np.load(os.path.join(containing_folder, "crop_mask.npy"))
        pipe.set_metrics_arguments(original_latent_dir, input_latent_dir, trimap, crop_mask)
        metrics_callback = callback_calculate_metrics

    # Process the image
    pipe_call_kwargs = dict(
        image=input_image,
        prompt=prompt,
        guidance_scale=guidance_scale,
        true_cfg_scale=true_cfg_scale,
        negative_prompt=negative_prompt,
        generator=generator,
        callback_on_step_end=metrics_callback,
    )
    if pipe_extra_kwargs:
        pipe_call_kwargs.update(pipe_extra_kwargs)
    image = pipe(**pipe_call_kwargs).images[0]

    # Save the result
    image.save(output_path)

    # Write timing summary
    example_name = os.path.basename(os.path.dirname(input_image_path))
    if hasattr(pipe, 'timing_data'):
        pipe.timing_data['total_example_wall_time'] = time.time() - _t_example_start
        write_timing_summary(output_folder, example_name, pipe.timing_data, global_timings=global_timings)

    # Log images to wandb if enabled. Do not write preview artifacts into the sample output folder.
    if log_to_wandb and wandb.run is not None:
        log_payload = {
            "input_image": wandb.Image(input_image, caption=f"Input: {os.path.basename(input_image_path)}"),
            "output_image": wandb.Image(image, caption="Output: output.png"),
        }
        if original_image is not None:
            log_payload["original_image"] = wandb.Image(original_image, caption=f"Original: {os.path.basename(original_image_path)}")
        elif os.path.exists(crop_mask_path):
            log_payload["crop_mask"] = wandb.Image(np.load(crop_mask_path), caption="Crop mask")
        wandb.log(log_payload, step=28)
        print("Images logged to wandb successfully")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Run FLUX Kontext pipeline on an input image or folder of images")
    parser.add_argument("input_path", type=str, nargs="?", default=DEFAULT_INPUT_PATH, help=f"Path to an input image, sample folder, or folder containing sample subfolders (default: {DEFAULT_INPUT_PATH})")
    parser.add_argument("original_path", type=str, nargs='?', default=None, help="Optional original image/folder; not needed when crop_mask.npy is available")
    parser.add_argument("--attn_config_file", type=str, default=DEFAULT_ATTN_CONFIG_FILE,
                       help=f"Path to YAML file containing attention configuration (default: {DEFAULT_ATTN_CONFIG_FILE})")
    parser.add_argument("--output_folder", type=str, default="outputs",
                       help="Output folder path (default: outputs)")
    parser.add_argument("--prompt", type=str, default="blend the cropped objects into the image in a convincing manner without changing the style of the image",
                       help="Prompt for image processing (default: 'blend the cropped objects into the image in a convincing manner without changing the style of the image')")
    parser.add_argument("--guidance_scale", type=float, default=2.5,
                       help="Guidance scale for processing (default: 2.5)")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for processing (default: False)")
    parser.add_argument("--use_orig_pipeline", action="store_true", help="Use original pipeline (default: False)")
    parser.add_argument("--attention_folder", type=str, default="attention_maps",
                       help="Attention map folder path (default: attention_maps)")
    parser.add_argument("--true_cfg_scale", type=float, default=1.0,
                       help="True CFG scale for processing (default: 1.0)")
    parser.add_argument("--negative_prompt", type=str, default=None,
                       help="Negative prompt for processing (default: None)")
    parser.add_argument("--override", action="store_true",
                       help="Override existing output files instead of skipping them (default: False)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for reproducible results (default: 0)")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable wandb logging (default: False)")
    parser.add_argument("--reconstruct_input", action="store_true",
                       help="Reconstruct the input image and store latents (default: False)")
    parser.add_argument("--reconstruct_original", action="store_true",
                       help="Reconstruct the original image and store latents (default: False)")
    parser.add_argument("--log_metrics", action="store_true",
                       help="Log metrics to wandb (default: False)")
    parser.add_argument("--remove_crop_bg", action="store_true",
                       help="Remove crop background by pasting foreground mask pixels from input onto original (default: False)")
    parser.add_argument("--run_rembg", action="store_true",
                       help="Use pasted_rembg.png instead of input.png as the input image (default: False)")
    parser.add_argument("--model", type=str, default="kontext", choices=["kontext", "qwen"],
                       help="Underlying model to use (default: kontext)")
    args = parser.parse_args()

    # Apply model-specific defaults for params not explicitly set by user
    model_defs = MODEL_DEFAULTS[args.model]
    if args.guidance_scale == 2.5 and args.model != "kontext":
        args.guidance_scale = model_defs["guidance_scale"]
    if args.true_cfg_scale == 1.0 and args.model != "kontext":
        args.true_cfg_scale = model_defs["true_cfg_scale"]

    # Seed everything for reproducible results
    seed_everything(args.seed)

    input_filename = "pasted_rembg.png" if args.run_rembg else "input.png"

    # Prepare missing sample files. A sample needs either original.png/original_path
    # or an existing crop_mask.npy.
    if os.path.isdir(args.input_path):
        if is_sample_folder(args.input_path, input_filename=input_filename, original_path=args.original_path):
            example_folder = args.input_path
            prep_input_path = sample_input_path(args.input_path, input_filename)
            prep_original_path = resolve_original_path(args.original_path, args.input_path)
            if not prepare_missing_sample_files(
                prep_input_path,
                prep_original_path,
                example_folder,
                args.seed,
                verbose=True,
            ):
                return
        else:
            subfolders_with_pairs = find_subfolders_with_image_pairs(args.input_path, input_filename=input_filename)
            if not subfolders_with_pairs:
                print(f"Error: no LooseRoPE samples found in folder: {args.input_path}")
                print(f"Expected either {input_filename} plus original.png/crop_mask.npy, or subfolders with that layout.")
                return

            print(f"[PREP] Found {len(subfolders_with_pairs)} subfolders with input plus original/mask")
            for subfolder_path, input_img_path, original_img_path in subfolders_with_pairs:
                if not prepare_missing_sample_files(
                    input_img_path,
                    original_img_path,
                    subfolder_path,
                    args.seed,
                    verbose=False,
                ):
                    return
    else:
        example_folder = os.path.dirname(args.input_path)
        prep_original_path = resolve_original_path(args.original_path, os.path.dirname(args.input_path))
        if not prepare_missing_sample_files(
            args.input_path,
            prep_original_path,
            example_folder,
            args.seed,
            verbose=True,
        ):
            return

    # Initialize wandb logging if not disabled
    use_wandb = not args.no_wandb

    # Load attention configuration if provided
    attention_config = None
    if args.attn_config_file is not None:
        # For subfolder processing, use the parent folder as example_folder
        # For single file/folder, example_folder was already set above
        if os.path.isdir(args.input_path):
            if is_sample_folder(args.input_path, input_filename=input_filename, original_path=args.original_path):
                example_folder = args.input_path
            else:
                subfolders_with_pairs = find_subfolders_with_image_pairs(args.input_path, input_filename=input_filename)
                if subfolders_with_pairs:
                    # Use the first subfolder as reference for attention config
                    example_folder = subfolders_with_pairs[0][0]
                else:
                    example_folder = args.input_path
        else:
            example_folder = os.path.dirname(args.input_path)

        attention_config = AttentionConfig(config_path=args.attn_config_file, example_folder=example_folder, output_folder=args.output_folder, use_wandb=use_wandb)
        print(f"Loaded attention configuration from {args.attn_config_file}")
    if use_wandb:
        initialize_wandb(args, attention_config)
        print("Wandb logging initialized successfully")
    else:
        print("Wandb logging disabled")

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Load the pipeline
    global_timings = {}
    model_defs = MODEL_DEFAULTS[args.model]
    _t_pipe_load_start = time.time()

    if args.model == "kontext":
        print("Loading FLUX Kontext pipeline...")
        if args.use_orig_pipeline:
            pipe = FluxKontextPipeline.from_pretrained(model_defs["pretrained_path"], torch_dtype=torch.bfloat16)
        else:
            pipe = LooseRoPEPipeline.from_pretrained(model_defs["pretrained_path"], torch_dtype=torch.bfloat16)
        if args.use_lora:
            pipe.load_lora_weights('/nfs/hf_home/hub/models--ilkerzgi--Overlay-Kontext-Dev-LoRA/snapshots/64be54220b7f223ce4d3536b1901aac39d898c5b/WVVtJFD90b8SsU6EzeGkO_adapter_model_comfy_patched.safetensors')
    elif args.model == "qwen":
        print("Loading Qwen-Image-Edit pipeline...")
        from looserope.qwen_looserope_pipeline import QwenLooseRoPEPipeline
        if args.use_orig_pipeline:
            from diffusers import QwenImageEditPipeline
            pipe = QwenImageEditPipeline.from_pretrained(model_defs["pretrained_path"])
        else:
            pipe = QwenLooseRoPEPipeline.from_pretrained(model_defs["pretrained_path"])
        pipe.to(torch.bfloat16)

    pipe.to("cuda")
    global_timings['Pipeline loading'] = time.time() - _t_pipe_load_start
    print(f"Pipeline loaded successfully. ({global_timings['Pipeline loading']:.2f}s)")

    # Initialize VLM if enabled in attention config
    if attention_config is not None and attention_config.enable_vlm_verdict:
        print(f"Initializing VLM for verdict (vlm_model_size={attention_config.vlm_model_size!r})...")
        _t_vlm_init_start = time.time()
        pipe._initialize_vlm(model_size=attention_config.vlm_model_size)
        global_timings['VLM model init'] = time.time() - _t_vlm_init_start
        if pipe.vlm_enabled:
            print("Loading VLM context examples...")
            _t_vlm_ctx_start = time.time()
            pipe._load_vlm_context_examples(
                context_folder=attention_config.vlm_context_folder,
                use_simplified_instruction=getattr(attention_config, "use_simplified_instruction", False),
            )
            global_timings['VLM context loading'] = time.time() - _t_vlm_ctx_start
            if pipe.vlm_enabled:
                print(f"VLM verdict enabled! Will run at timestep {attention_config.vlm_verdict_timestep}")
            else:
                print("VLM context examples failed to load, VLM verdict disabled")
        else:
            print("VLM initialization failed, VLM verdict disabled")

    # Build extra kwargs for the pipe call based on model type
    pipe_extra_kwargs = {}
    if args.model == "qwen":
        pipe_extra_kwargs["num_inference_steps"] = model_defs["num_inference_steps"]

    # Determine if input is a file or folder
    if os.path.isfile(args.input_path):
        # Single file processing
        original_image_path = resolve_original_path(args.original_path, os.path.dirname(args.input_path))
        if not can_prepare_or_run(args.input_path, original_image_path):
            print("Error: provide either original_path/original.png or crop_mask.npy next to the input image")
            return
        print(f"Processing single image: {args.input_path}")
        output_path = process_single_image(
            pipe,
            args.input_path,
            original_image_path,
            args.output_folder,
            args.prompt,
            args.guidance_scale,
            args.true_cfg_scale,
            args.negative_prompt,
            args.override,
            attention_config,
            args.seed,
            use_wandb,
            args.reconstruct_input,
            args.reconstruct_original,
            args.log_metrics,
            args.remove_crop_bg,
            global_timings=global_timings,
            pipe_extra_kwargs=pipe_extra_kwargs,
        )
        if output_path:
            print(f"Processed image saved to: {output_path}")
        else:
            print("Failed to process the image.")

    elif os.path.isdir(args.input_path):
        if is_sample_folder(args.input_path, input_filename=input_filename, original_path=args.original_path):
            sample_name = os.path.basename(os.path.normpath(args.input_path))
            sample_output = os.path.join(args.output_folder, sample_name)
            os.makedirs(sample_output, exist_ok=True)

            if attention_config is not None:
                attention_config.output_folder = sample_output
                attention_config.example_folder = args.input_path
                attention_config.reset_saliency_boost()
                inside_coords_path = os.path.join(args.input_path, "inside.json") if attention_config.inside_coords_file is None else attention_config.inside_coords_file
                outside_coords_path = os.path.join(args.input_path, "outside.json") if attention_config.outside_coords_file is None else attention_config.outside_coords_file
                attention_config.inside_coords = attention_config._load_coordinates_from_json(inside_coords_path)
                attention_config.outside_coords = attention_config._load_coordinates_from_json(outside_coords_path)

            input_img_path = sample_input_path(args.input_path, input_filename)
            original_img_path = resolve_original_path(args.original_path, args.input_path)
            print(f"Processing sample folder: {args.input_path}")
            output_path = process_single_image(
                pipe,
                input_img_path,
                original_img_path,
                sample_output,
                args.prompt,
                args.guidance_scale,
                args.true_cfg_scale,
                args.negative_prompt,
                args.override,
                attention_config,
                args.seed,
                use_wandb,
                args.reconstruct_input,
                args.reconstruct_original,
                args.log_metrics,
                args.remove_crop_bg,
                global_timings=global_timings,
                pipe_extra_kwargs=pipe_extra_kwargs,
            )
            if output_path:
                print(f"Processed sample saved to: {output_path}")
            else:
                print("Failed to process the sample.")

        else:
            # Check if this folder contains subfolders with image pairs
            subfolders_with_pairs = find_subfolders_with_image_pairs(args.input_path, input_filename=input_filename)

            if subfolders_with_pairs:
                # Process each subfolder
                total_folders = len(subfolders_with_pairs)
                print(f"Found {total_folders} subfolders with input plus original/mask")

                successful_count = 0
                for i, (subfolder_path, input_img_path, original_img_path) in enumerate(subfolders_with_pairs, 1):
                    subfolder_name = os.path.basename(subfolder_path)
                    print(f"\n{'='*60}")
                    print(f"Processing subfolder {i} of {total_folders}: {subfolder_name}")
                    print(f"{'='*60}")

                    # Create output folder for this subfolder
                    subfolder_output = os.path.join(args.output_folder, subfolder_name)
                    os.makedirs(subfolder_output, exist_ok=True)

                    # Update attention config for this subfolder
                    # (set_masks will be called inside process_single_image)
                    if attention_config is not None:
                        attention_config.output_folder = subfolder_output
                        attention_config.example_folder = subfolder_path

                        # Reset saliency boost to original value for each subfolder
                        attention_config.reset_saliency_boost()

                        # Reload coordinates from the new example_folder
                        inside_coords_path = os.path.join(subfolder_path, "inside.json") if attention_config.inside_coords_file is None else attention_config.inside_coords_file
                        outside_coords_path = os.path.join(subfolder_path, "outside.json") if attention_config.outside_coords_file is None else attention_config.outside_coords_file
                        attention_config.inside_coords = attention_config._load_coordinates_from_json(inside_coords_path)
                        attention_config.outside_coords = attention_config._load_coordinates_from_json(outside_coords_path)

                    output_path = process_single_image(
                        pipe,
                        input_img_path,
                        original_img_path,
                        subfolder_output,
                        args.prompt,
                        args.guidance_scale,
                        args.true_cfg_scale,
                        args.negative_prompt,
                        args.override,
                        attention_config,
                        args.seed,
                        use_wandb,
                        args.reconstruct_input,
                        args.reconstruct_original,
                        args.log_metrics,
                        args.remove_crop_bg,
                        global_timings=global_timings,
                        pipe_extra_kwargs=pipe_extra_kwargs,
                    )
                    if output_path:
                        print(f"✓ Saved: {os.path.basename(output_path)}")
                        successful_count += 1
                    else:
                        print(f"✗ Failed to process: {subfolder_name}")

                print(f"\n{'='*60}")
                print(f"Batch processing complete! Successfully processed {successful_count} of {total_folders} subfolders.")
                print(f"{'='*60}")

            else:
                # Batch processing for a flat folder. If originals are not provided,
                # each image must rely on crop_mask.npy in the input folder.
                image_files = find_image_files(args.input_path)

                if not image_files:
                    print(f"No image files (.png, .jpg, .jpeg) found in folder: {args.input_path}")
                    return

                total_images = len(image_files)
                print(f"Found {total_images} images to process")

                successful_count = 0
                for i, image_file in enumerate(image_files, 1):
                    input_filename = os.path.basename(image_file)
                    input_name_no_ext = os.path.splitext(input_filename)[0]

                    original_image_path = None
                    if args.original_path is not None:
                        # Look for corresponding original image with "_orig" suffix
                        original_filename = input_name_no_ext + "_orig.png"
                        original_image_path = os.path.join(args.original_path, original_filename)

                        if not os.path.exists(original_image_path):
                            print(f"Error: Original image not found for '{input_filename}'. Expected: '{original_image_path}'")
                            return
                    elif not has_crop_mask_for_image(image_file):
                        print(f"Error: crop_mask.npy not found for '{input_filename}', and no original_path was provided")
                        return

                    print(f"Processing image {i} of {total_images}: {image_file}")

                    output_path = process_single_image(
                        pipe,
                        image_file,
                        original_image_path,
                        args.output_folder,
                        args.prompt,
                        args.guidance_scale,
                        args.true_cfg_scale,
                        args.negative_prompt,
                        args.override,
                        attention_config,
                        args.seed,
                        use_wandb,
                        args.reconstruct_input,
                        args.reconstruct_original,
                        args.log_metrics,
                        args.remove_crop_bg,
                        global_timings=global_timings,
                        pipe_extra_kwargs=pipe_extra_kwargs,
                    )
                    if output_path:
                        print(f"Saved: {os.path.basename(output_path)}")
                        successful_count += 1
                    else:
                        print(f"Failed to process: {input_filename}")

                print(f"\nBatch processing complete! Successfully processed {successful_count} of {total_images} images.")

    else:
        print(f"Error: '{args.input_path}' is neither a file nor a directory.")

    # Finish wandb run if it was initialized
    if use_wandb and wandb.run is not None:
        wandb.finish()
        print("Wandb run completed")

if __name__ == "__main__":
    main()
