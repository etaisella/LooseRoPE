from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
from diffusers.models.transformers.transformer_flux import FluxAttention
from diffusers.models.transformers.transformer_flux import _get_qkv_projections
from diffusers.models.transformers.transformer_flux import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import dispatch_attention_fn
from diffusers.models.transformers.transformer_flux import FluxPosEmbed
import matplotlib.pyplot as plt
import torch
import math
import os
import numpy as np
import yaml
import json
from typing import Optional, Dict, Any, List
from scipy.ndimage import binary_dilation
import wandb
from PIL import Image

LAST_LAYER_NUM = 56
OUT_IMAGE_ATTN_OFFSET = 512
IN_IMAGE_ATTN_OFFSET = 512 + 64*64
ATTN_WIDTH = 64
ATTN_HEIGHT = 64
NUM_LAYERS = 57
NUM_STEPS = 28
LATENT_SIZE = 8704
SPECIAL_Q_BATCH_SIZE = 200
EPSILON = 1e-8

from .positional_embedding import set_custom_img_ids, LooseRoPEFluxPosEmbed

def exponential_interp(N, f0, f1, k=5.0):
    """
    Generate N exponentially interpolated values between f0 and f1,

    Parameters
    ----------
    N : int
        Number of points (including endpoints if desired)
    f0 : float
        Start value (at x=0)
    f1 : float
        End value (at x=1)
    k : float
        Steepness control parameter (k > 1 = steeper near the end)

    Returns
    -------
    x : np.ndarray
        Array of N points between 0 and 1
    f : np.ndarray
        Corresponding interpolated values
    """
    x = np.linspace(0, 1, N)
    f = f0 * (f1 / f0) ** (x ** k)
    return f

def tanh_interp(N, f0, f1, k=3.0):
    """
    Generate N tanh interpolated values between f0 and f1,
    """
    x = np.linspace(-1, 1, N)
    f = np.tanh(x * k) * 0.5 + 0.5
    f = f * (f1 - f0) + f0
    return f

class AttentionConfig:
    """Configuration class for attention processor settings."""

    def __init__(self, config_path: str = None, config_dict: Dict[str, Any] = None, example_folder: str = None, output_folder: str = None, use_wandb: bool = True):
        """Initialize attention configuration from YAML file or dictionary.

        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to file)
            example_folder: Parent folder of input image, used as fallback for coordinate files
            output_folder: Folder to save attention outputs
            use_wandb: Whether to log to wandb (default: True)
        """
        if config_path is not None:
            self.config = self._load_yaml_config(config_path)
        elif config_dict is not None:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        self.example_folder = example_folder
        self.output_folder = output_folder
        self.use_wandb = use_wandb

        # Set default values
        self.inside_coords_file = self.config.get('inside_coords_file')
        self.outside_coords_file = self.config.get('outside_coords_file')
        self.generate_random_points = self.config.get('generate_random_points', False)
        self.random_seed = self.config.get('random_seed', 42)
        self.num_random_points = self.config.get('num_random_points', 5)
        self.masking_step_range = self.config.get('masking_step_range', [0, NUM_STEPS])
        self.masking_layer_range = self.config.get('masking_layer_range', [0, NUM_LAYERS])
        self.dilation_steps_attn_mask = self.config.get('dilation_steps_attn_mask', 0)
        self.in_image_attn_mask_file = self.config.get('in_image_attn_mask')
        self.in_image_mask_val = self.config.get('in_image_mask_val', 1.0)
        self.use_smoothing = self.config.get('use_smoothing', False)
        self.post_softmax = self.config.get('post_softmax', False)
        self.text_emphasizing_step_range = self.config.get('text_emphasizing_step_range', [0, NUM_STEPS])
        self.text_emphasizing_layer_range = self.config.get('text_emphasizing_layer_range', [0, NUM_LAYERS])
        self.text_emphasizing_mask_val = self.config.get('text_emphasizing_mask_val', 1.0)
        self.text_emphasizing_mask_file = self.config.get('text_emphasizing_mask_file')
        self.dilation_steps_emphasizing_mask = self.config.get('dilation_steps_emphasizing_mask', 0)
        self.start_h = self.config.get('start_h', 0)
        self.start_w = self.config.get('start_w', 0)
        self.end_h = self.config.get('end_h', 63)
        self.end_w = self.config.get('end_w', 63)
        self.custom_pos_embed_step_range = self.config.get('custom_pos_embed_step_range', [0, 0])
        self.custom_pos_embed_layer_range = self.config.get('custom_pos_embed_layer_range', [0, 0])
        self.save_norm_maps = bool(self.config.get('save_norm_maps', False))
        self.save_attention_maps = bool(self.config.get('save_attention_maps', False))
        self.save_mask_previews = bool(self.config.get('save_mask_previews', False))
        self.use_custom_pos_embed = self.config.get('use_custom_pos_embed', False)
        self.per_query_mask_file = self.config.get('per_query_mask_file')
        self.per_query_dilation_steps_mask = self.config.get('per_query_dilation_steps_mask', 2)
        # X0 prediction configuration
        self.save_x0_predictions = self.config.get('save_x0_predictions', False)
        self.x0_prediction_steps = self.config.get('x0_prediction_steps', [])

        # VLM verdict configuration
        self.enable_vlm_verdict = self.config.get('enable_vlm_verdict', False)
        self.vlm_model_size = self.config.get('vlm_model_size', '4B')
        self.vlm_verdict_timestep = self.config.get('vlm_verdict_timestep', 2)
        self.vlm_context_folder = self.config.get('vlm_context_folder', None)
        self.vlm_max_tries = self.config.get('vlm_max_tries', 4)
        self.vlm_boost_neglect = self.config.get('vlm_boost_neglect', -0.08)
        self.vlm_boost_suppression = self.config.get('vlm_boost_suppression', 0.1)
        self.vlm_attn_low_neglect = self.config.get('vlm_attn_low_neglect', 0.0)
        self.vlm_attn_low_suppression = self.config.get('vlm_attn_low_suppression', 0.0)
        self.vlm_shrink_low_neglect = self.config.get('vlm_shrink_low_neglect', 0.0)
        self.vlm_shrink_low_suppression = self.config.get('vlm_shrink_low_suppression', 0.0)
        # Verdict-specific config file paths
        self.use_simplified_instruction = self.config.get('use_simplified_instruction', False)
        self.vlm_max_new_tokens = self.config.get('vlm_max_new_tokens', 1024)
        self.vlm_max_new_tokens_simplified = self.config.get('vlm_max_new_tokens_simplified', 512)

        # Attention factor configuration

        # Frequency modify configuration
        self.modify_freq_start = self.config.get('modify_freq_start', 0)
        self.modify_freq_end = self.config.get('modify_freq_end', 0)
        self.modify_freq_value = self.config.get('modify_freq_value', 1.0)

        # Continuous saliency configuration
        self.cont_mode = self.config.get('cont_mode', False)
        self.cont_freq_mode = self.config.get('cont_freq_mode', False)
        self.cont_N = self.config.get('cont_N', 3)
        self.cont_shrink_factor_low = self.config.get('cont_shrink_factor_low', 0.65)
        self.cont_shrink_factor_low_original = self.cont_shrink_factor_low  # Store original value
        self.cont_shrink_factor_high = self.config.get('cont_shrink_factor_high', 1.0)
        self.cont_in_img_attn_factor_low = self.config.get('cont_in_img_attn_factor_low', 1.0)
        self.cont_in_img_attn_factor_low_original = self.cont_in_img_attn_factor_low  # Store original value
        self.cont_in_img_attn_factor_high = self.config.get('cont_in_img_attn_factor_high', 1.325)
        self.cont_step_shrink = self.config.get('cont_step_shrink', 0.22)
        self.cont_cutoffs = self.config.get('cont_cutoffs', [10, 18])
        self.cont_saliency_boost = self.config.get('cont_saliency_boost', 1.0)
        self.cont_saliency_boost_original = self.cont_saliency_boost  # Store original value
        self.cont_annealing_factor = self.config.get('cont_annealing_factor', 1.0)
        # When True, multiply only negative (1 - factor) terms by cont_annealing_factor**(phase+1);
        # nonnegative diffs are unchanged (see cont_mode in-img attention scaling).
        self.anneal_below_zero = self.config.get('anneal_below_zero', False)
        self.cont_shrink_k = self.config.get('cont_shrink_k', 1.5)
        self.cont_in_img_attn_k = self.config.get('cont_in_img_attn_k', 3.5)

        # Attention locality recording configuration
        self.record_attention_locality = self.config.get('record_attention_locality', False)
        self.locality_step_range = self.config.get('locality_step_range', [0, 4])
        self.locality_layer_range = self.config.get('locality_layer_range', [0, 10])
        self.locality_gaussian_sigma = self.config.get('locality_gaussian_sigma', 3.0)

        # set up cont parameters
        if self.cont_mode:
            # this should not be a linear schedule
            #self.cont_shrink_factors = np.linspace(self.cont_shrink_factor_low, self.cont_shrink_factor_high, self.cont_N)
            #self.cont_shrink_factors = exponential_interp(self.cont_N, self.cont_shrink_factor_low, self.cont_shrink_factor_high, k=2.0)
            self.cont_shrink_factors = tanh_interp(self.cont_N, self.cont_shrink_factor_low, self.cont_shrink_factor_high, k=self.cont_shrink_k)
            #self.cont_in_img_attn_factors = np.linspace(self.cont_in_img_attn_factor_low, self.cont_in_img_attn_factor_high, self.cont_N)
            #self.cont_in_img_attn_factors = exponential_interp(self.cont_N, self.cont_in_img_attn_factor_low, self.cont_in_img_attn_factor_high, k=2.0)
            self.cont_in_img_attn_factors = tanh_interp(self.cont_N, self.cont_in_img_attn_factor_low, self.cont_in_img_attn_factor_high, k=self.cont_in_img_attn_k)

        # Validate range parameters
        self._validate_range_parameter(self.masking_step_range, 'masking_step_range')
        self._validate_range_parameter(self.masking_layer_range, 'masking_layer_range')
        self._validate_range_parameter(self.text_emphasizing_step_range, 'text_emphasizing_step_range')
        self._validate_range_parameter(self.text_emphasizing_layer_range, 'text_emphasizing_layer_range')
        self._validate_range_parameter(self.custom_pos_embed_step_range, 'custom_pos_embed_step_range')
        self._validate_range_parameter(self.custom_pos_embed_layer_range, 'custom_pos_embed_layer_range')

        # Text-related attributes (set later via set_text_info)
        self.text_offsets = None
        self.words_to_save = None

        # Load coordinates - use example_folder if not specified in config
        inside_coords_path = self.inside_coords_file
        outside_coords_path = self.outside_coords_file

        if inside_coords_path is None and self.example_folder is not None:
            inside_coords_path = os.path.join(self.example_folder, "inside.json")

        if outside_coords_path is None and self.example_folder is not None:
            outside_coords_path = os.path.join(self.example_folder, "outside.json")

        self.inside_coords = self._load_coordinates_from_json(inside_coords_path)
        self.outside_coords = self._load_coordinates_from_json(outside_coords_path)

        # Load special coordinates if available
        special_coords_path = None
        if self.example_folder is not None:
            special_coords_path = os.path.join(self.example_folder, "special.json")
        self.special_coords = self._load_coordinates_from_json(special_coords_path)

    def _set_unified_attn_mask(self, in_image_attn_mask, in_image_mask_val):
        unified_mask = torch.ones(1, LATENT_SIZE)
        in_image_attn_mask_w_val = np.ones_like(in_image_attn_mask).astype(np.float64)
        in_image_attn_mask_w_val[in_image_attn_mask != 0.0] = in_image_mask_val
        in_image_attn_mask_w_val = torch.Tensor(in_image_attn_mask_w_val).float().flatten()
        unified_mask[0, IN_IMAGE_ATTN_OFFSET:IN_IMAGE_ATTN_OFFSET + 64*64] = in_image_attn_mask_w_val
        unified_mask = unified_mask.to(torch.bfloat16)
        return unified_mask

    def _set_emphasizing_mask(self, text_emphasizing_mask, text_emphasizing_mask_val):
        # Determine the maximal index in self.text_offsets (which is a dict mapping words to list of indices)
        if self.text_offsets is not None and len(self.text_offsets) > 0:
            end_idx = max([max(idxs) for idxs in self.text_offsets.values() if len(idxs) > 0]) + 1
        else:
            end_idx = OUT_IMAGE_ATTN_OFFSET  # fallback if not set

        emphasizing_mask = torch.ones(LATENT_SIZE, LATENT_SIZE)
        text_emphasizing_mask_w_val = np.ones_like(text_emphasizing_mask).astype(np.float64)
        text_emphasizing_mask_w_val[text_emphasizing_mask != 0.0] = text_emphasizing_mask_val
        text_emphasizing_mask_w_val = torch.Tensor(text_emphasizing_mask_w_val).float().flatten()
        emphasizing_mask[:end_idx, IN_IMAGE_ATTN_OFFSET:IN_IMAGE_ATTN_OFFSET + 64*64] = text_emphasizing_mask_w_val
        emphasizing_mask = emphasizing_mask.to(torch.bfloat16)
        return emphasizing_mask

    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _validate_range_parameter(self, range_param: List[int], param_name: str) -> None:
        """Validate that a range parameter is properly formatted.

        Args:
            range_param: The range parameter to validate
            param_name: Name of the parameter for error messages
        """
        if not isinstance(range_param, list):
            raise ValueError(f"{param_name} must be a list, got {type(range_param)}")

        if len(range_param) != 2:
            raise ValueError(f"{param_name} must be a list of exactly 2 integers [start, end], got {len(range_param)} elements")

        if not all(isinstance(x, int) for x in range_param):
            raise ValueError(f"{param_name} must contain only integers, got {range_param}")

    def _load_attention_mask(self, mask_file_path: str, dilation_steps: int = 0) -> np.ndarray:
        """Load attention mask from numpy file or create default 64x64 ones matrix.

        Args:
            mask_file_path: Path to numpy file containing 64x64 binary mask or 'crop'

        Returns:
            64x64 numpy array (binary mask)
        """
        if mask_file_path == 'full':
            # Return 64x64 ones matrix as default
            mask = np.ones((64, 64))
        elif mask_file_path == 'crop':
            if self.example_folder is None:
                raise ValueError("example_folder must be set to load crop mask.")
            crop_mask_path = os.path.join(self.example_folder, "crop_mask.npy")
            if not os.path.exists(crop_mask_path):
                raise FileNotFoundError(f"Crop mask file not found: {crop_mask_path}")
            mask = np.load(crop_mask_path)
        elif mask_file_path == 'trimap':
            if self.example_folder is None:
                raise ValueError("example_folder must be set to load trimap.")
            trimap_path = os.path.join(self.example_folder, "trimap.npy")
            if not os.path.exists(trimap_path):
                raise FileNotFoundError(f"Trimap file not found: {trimap_path}")
            mask = np.load(trimap_path)
        else:
            if not os.path.exists(mask_file_path):
                raise FileNotFoundError(f"Attention mask file not found: {mask_file_path}")
            mask = np.load(mask_file_path)

        # Validate mask shape
        if mask.shape != (64, 64):
            raise ValueError(f"Attention mask must be 64x64, got shape {mask.shape}")

        # Dilate mask if dilation steps are provided
        if dilation_steps > 0:
            mask = binary_dilation(mask, iterations=dilation_steps)

        return mask.astype(np.uint8)

    def _load_saliency(self) -> np.ndarray:
        """Load saliency from the current example folder."""
        saliency_path = os.path.join(self.example_folder, "saliency.npy")
        if not os.path.exists(saliency_path):
            return None
        return np.load(saliency_path)

    def _load_coordinates_from_json(self, json_file_path: str) -> Optional[List[List[int]]]:
        """Load coordinates from a JSON file.

        Args:
            json_file_path: Path to JSON file containing coordinates

        Returns:
            List of coordinate pairs [[y, x], [y, x], ...] or None if file not provided or not found
        """
        if json_file_path is None:
            return None

        if not os.path.exists(json_file_path):
            return None

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

    def get_coordinates(self, mask_np: np.ndarray) -> tuple:
        """Get inside and outside coordinates, generating random ones if needed.

        Args:
            mask_np: Mask array where 1 indicates inside region, 0 indicates outside

        Returns:
            Tuple of (inside_coords, outside_coords)
        """
        inside_coords = self.inside_coords
        outside_coords = self.outside_coords

        # Generate random coordinates if not provided and flag is set
        if (inside_coords is None or outside_coords is None) and self.generate_random_points:
            rng = np.random.default_rng(self.random_seed)

            if inside_coords is None:
                mask_inside_coords = np.argwhere(mask_np == 1)
                if len(mask_inside_coords) >= self.num_random_points:
                    inside_coords = rng.choice(mask_inside_coords, size=self.num_random_points, replace=False).tolist()
                else:
                    inside_coords = mask_inside_coords.tolist()

            if outside_coords is None:
                mask_outside_coords = np.argwhere(mask_np == 0)
                if len(mask_outside_coords) >= self.num_random_points:
                    outside_coords = rng.choice(mask_outside_coords, size=self.num_random_points, replace=False).tolist()
                else:
                    outside_coords = mask_outside_coords.tolist()

        return inside_coords, outside_coords

    def should_apply_masking(self, curr_step: int, layer_num: int) -> bool:
        """Check if masking should be applied for the given step and layer."""
        step_start, step_end = self.masking_step_range
        layer_start, layer_end = self.masking_layer_range

        step_in_range = step_start <= curr_step <= step_end
        layer_in_range = layer_start <= layer_num <= layer_end

        return step_in_range and layer_in_range

    def should_use_custom_pos_embed(self, curr_step: int, layer_num: int) -> bool:
        """Check if custom pos embed should be used for the given step and layer."""
        step_start, step_end = self.custom_pos_embed_step_range
        layer_start, layer_end = self.custom_pos_embed_layer_range

        step_in_range = step_start <= curr_step <= step_end
        layer_in_range = layer_start <= layer_num <= layer_end

        return step_in_range and layer_in_range and self.use_custom_pos_embed

    def should_use_text_emphasizing(self, curr_step: int, layer_num: int) -> bool:
        """Check if text emphasizing should be used for the given step and layer."""
        step_start, step_end = self.text_emphasizing_step_range
        layer_start, layer_end = self.text_emphasizing_layer_range

        step_in_range = step_start <= curr_step <= step_end
        layer_in_range = layer_start <= layer_num <= layer_end

        return step_in_range and layer_in_range

    def should_record_attention_locality(self, curr_step: int, layer_num: int) -> bool:
        """Check if attention locality should be recorded for the given step and layer."""
        if not hasattr(self, '_locality_debug_printed'):
            self._locality_debug_printed = False

        if not self.record_attention_locality:
            if not self._locality_debug_printed:
                print(f"[LOCALITY DEBUG] record_attention_locality is False - feature DISABLED")
                self._locality_debug_printed = True
            return False

        step_start, step_end = self.locality_step_range
        layer_start, layer_end = self.locality_layer_range

        step_in_range = step_start <= curr_step < step_end
        layer_in_range = layer_start <= layer_num < layer_end

        result = step_in_range and layer_in_range

        if not self._locality_debug_printed:
            print(f"[LOCALITY DEBUG] record_attention_locality is True - feature ENABLED")
            print(f"[LOCALITY DEBUG]   step_range: [{step_start}, {step_end})")
            print(f"[LOCALITY DEBUG]   layer_range: [{layer_start}, {layer_end})")
            print(f"[LOCALITY DEBUG]   First check at step={curr_step}, layer={layer_num}: result={result}")
            self._locality_debug_printed = True

        return result

    def set_text_info(self, text_offsets: Dict[str, List[int]], words_to_save: List[str]) -> None:
        """Set text-related information for attention processing.

        Args:
            text_offsets: Dictionary mapping words to their token offset indices
            words_to_save: List of words to save attention maps for
        """
        self.text_offsets = text_offsets
        self.words_to_save = words_to_save

    def reset_saliency_boost(self) -> None:
        """Reset cont_saliency_boost to its original value."""
        if hasattr(self, 'cont_saliency_boost_original'):
            self.cont_saliency_boost = self.cont_saliency_boost_original

    def reset_attn_factor_low(self) -> None:
        """Reset cont_in_img_attn_factor_low to its original value."""
        if hasattr(self, 'cont_in_img_attn_factor_low_original'):
            self.cont_in_img_attn_factor_low = self.cont_in_img_attn_factor_low_original
            if self.cont_mode:
                self.cont_in_img_attn_factors = tanh_interp(self.cont_N, self.cont_in_img_attn_factor_low, self.cont_in_img_attn_factor_high, k=self.cont_in_img_attn_k)

    def reset_shrink_factor_low(self) -> None:
        """Reset cont_shrink_factor_low to its original value."""
        if hasattr(self, 'cont_shrink_factor_low_original'):
            self.cont_shrink_factor_low = self.cont_shrink_factor_low_original
            if self.cont_mode:
                self.cont_shrink_factors = tanh_interp(self.cont_N, self.cont_shrink_factor_low, self.cont_shrink_factor_high, k=self.cont_shrink_k)

    def update_cont_in_img_attn_factors(self) -> None:
        """Regenerate cont_in_img_attn_factors based on current low and high values."""
        if self.cont_mode:
            self.cont_in_img_attn_factors = tanh_interp(self.cont_N, self.cont_in_img_attn_factor_low, self.cont_in_img_attn_factor_high, k=self.cont_in_img_attn_k)

    def update_cont_shrink_factors(self) -> None:
        """Regenerate cont_shrink_factors based on current low and high values."""
        if self.cont_mode:
            self.cont_shrink_factors = tanh_interp(self.cont_N, self.cont_shrink_factor_low, self.cont_shrink_factor_high, k=self.cont_shrink_k)

    def reset_all_vlm_modified_params(self) -> None:
        """Reset all parameters that may have been modified by VLM verdict."""
        self.reset_saliency_boost()
        self.reset_attn_factor_low()
        self.reset_shrink_factor_low()

    def set_masks(self):
        # Validate mask value parameters
        if not isinstance(self.in_image_mask_val, (int, float)):
            raise ValueError(f"in_image_mask_val must be a number, got {type(self.in_image_mask_val)}")
        if not isinstance(self.text_emphasizing_mask_val, (int, float)):
            raise ValueError(f"text_emphasizing_mask_val must be a number, got {type(self.text_emphasizing_mask_val)}")

        # Load attention mask
        in_image_attn_mask = self._load_attention_mask(self.in_image_attn_mask_file, self.dilation_steps_attn_mask)
        if self.save_mask_previews and self.output_folder is not None:
            Image.fromarray((in_image_attn_mask * 255).astype(np.uint8)).save(os.path.join(self.output_folder, "in_image_attn_mask.png"))
        if self.use_wandb and wandb.run is not None:
            wandb.log({"in_image_attn_mask": wandb.Image((in_image_attn_mask * 255).astype(np.uint8))}, step=0)
        self.attn_mask = self._set_unified_attn_mask(in_image_attn_mask, self.in_image_mask_val)

        # Load emphasizing mask
        text_emphasizing_mask = self._load_attention_mask(self.text_emphasizing_mask_file, self.dilation_steps_emphasizing_mask)
        if self.save_mask_previews and self.output_folder is not None:
            Image.fromarray((text_emphasizing_mask * 255).astype(np.uint8)).save(os.path.join(self.output_folder, "text_emphasizing_mask.png"))
        if self.use_wandb and wandb.run is not None:
            wandb.log({"text_emphasizing_mask": wandb.Image((text_emphasizing_mask * 255).astype(np.uint8))}, step=0)
        self.text_emphasizing_mask = self._set_emphasizing_mask(text_emphasizing_mask, self.text_emphasizing_mask_val)

        # Load per query mask
        self.per_query_mask = self._load_attention_mask(self.per_query_mask_file, self.per_query_dilation_steps_mask)
        if self.save_mask_previews and self.output_folder is not None:
            Image.fromarray((self.per_query_mask * 255).astype(np.uint8)).save(os.path.join(self.output_folder, "per_query_mask.png"))
        if self.use_wandb and wandb.run is not None:
            wandb.log({"per_query_mask": wandb.Image((self.per_query_mask * 255).astype(np.uint8))}, step=0)

        # Load saliency
        self.saliency = self._load_saliency()
        if self.saliency is not None:
            if self.save_mask_previews and self.output_folder is not None:
                Image.fromarray((self.saliency * (255 // 3)).astype(np.uint8)).save(os.path.join(self.output_folder, "saliency.png"))
            if self.use_wandb and wandb.run is not None:
                wandb.log({"saliency": wandb.Image((self.saliency * (255 // 3)).astype(np.uint8))}, step=0)


class LooseRoPEFluxAttnProcessor(FluxAttnProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_folder = None
        self.curr_step = 0
        self.accumulated_attention_maps = {}
        self.locality_scores = []  # Store locality scores for analysis
        self.running_avg_correlation = 0.0  # Running average of mean correlations
        self.running_count = 0  # Count of measurements for running average

    def set_attention_config(self, save_folder, layer_num, attention_config):
        self.save_folder = save_folder
        self.layer_num = layer_num
        self.attention_config = attention_config
        self._set_custom_ids_and_pos_embed()

    def _set_custom_ids_and_pos_embed(self, annealing_factor=1.0):
        self._set_pos_embed(
            modify_freq_start=self.attention_config.modify_freq_start,
            modify_freq_end=self.attention_config.modify_freq_end,
            modify_freq_value=self.attention_config.modify_freq_value,
        )

        if not self.attention_config.cont_mode:
            return

        self.custom_rotary_embs = []
        shrink_factors = self.attention_config.cont_shrink_factors * annealing_factor
        shrink_factors = np.clip(shrink_factors, 0.0, 1.0)

        for i in range(self.attention_config.cont_N):
            if self.attention_config.cont_freq_mode:
                self._set_pos_embed(
                    modify_freq_start=self.attention_config.modify_freq_start,
                    modify_freq_end=self.attention_config.modify_freq_end,
                    modify_freq_value=shrink_factors[i],
                )
                img_ids = set_custom_img_ids(
                    shrink_factor_h=1.0,
                    shrink_factor_w=1.0,
                    start_h=self.attention_config.start_h,
                    start_w=self.attention_config.start_w,
                    end_h=self.attention_config.end_h,
                    end_w=self.attention_config.end_w,
                )
                pos_embed = self.special_pos_embed
            else:
                img_ids = set_custom_img_ids(
                    shrink_factor_h=shrink_factors[i],
                    shrink_factor_w=shrink_factors[i],
                    start_h=self.attention_config.start_h,
                    start_w=self.attention_config.start_w,
                    end_h=self.attention_config.end_h,
                    end_w=self.attention_config.end_w,
                )
                pos_embed = self.pos_embed

            in_img_ids = img_ids.clone()
            in_img_ids[..., 0] = 1.0
            text_ids = torch.zeros(OUT_IMAGE_ATTN_OFFSET, 3).to(img_ids.device)
            custom_ids = torch.cat([text_ids, img_ids, in_img_ids], dim=0)
            self.custom_rotary_embs.append(pos_embed(custom_ids))

    def _set_pos_embed(self, theta=10000, axes_dim=[16, 56, 56], modify_freq_start=0, modify_freq_end=0, modify_freq_value=1.0):
        #self.special_pos_embed = LooseRoPEFluxPosEmbed(
        #    theta=theta,
        #    axes_dim=axes_dim,
        #    modify_freq_start=modify_freq_start,
        #    modify_freq_end=modify_freq_end,
        #    modify_freq_value=modify_freq_value
        #)
        self.special_pos_embed = LooseRoPEFluxPosEmbed(
            theta=theta * modify_freq_value,
            axes_dim=axes_dim,
            modify_freq_start=modify_freq_start,
            modify_freq_end=modify_freq_end,
            modify_freq_value=1.0
        )
        self.pos_embed = FluxPosEmbed(
            theta=theta,
            axes_dim=axes_dim,
        )

    def _create_2d_gaussian(self, center_y: int, center_x: int, sigma: float) -> np.ndarray:
        """Create a 2D Gaussian centered at (center_y, center_x) with given sigma.

        Args:
            center_y: Y coordinate of the Gaussian center
            center_x: X coordinate of the Gaussian center
            sigma: Standard deviation of the Gaussian

        Returns:
            64x64 numpy array representing the Gaussian
        """
        y_coords = np.arange(ATTN_HEIGHT)
        x_coords = np.arange(ATTN_WIDTH)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

        # Calculate squared distance from center
        dist_sq = (yy - center_y) ** 2 + (xx - center_x) ** 2

        # Create Gaussian
        gaussian = np.exp(-dist_sq / (2 * sigma ** 2))

        # Normalize to sum to 1
        gaussian = gaussian / (gaussian.sum() + EPSILON)

        return gaussian

    def _calculate_attention_locality(self, attn_weight: torch.Tensor, curr_step: int) -> Dict[str, float]:
        """Calculate attention locality and inward/outward ratio for queries in the crop mask.

        For each query position in the crop mask:
        1. Calculate correlation between attention and a Gaussian (locality)
        2. Calculate ratio of attention to keys inside vs outside the mask (inward/outward ratio)

        Args:
            attn_weight: Attention weights tensor [batch, heads, queries, keys]
            curr_step: Current diffusion step

        Returns:
            Dictionary with locality and ratio statistics
        """
        if not hasattr(self, '_locality_calc_called'):
            print(f"[LOCALITY DEBUG] _calculate_attention_locality called FIRST TIME at step={curr_step}, layer={self.layer_num}")
            self._locality_calc_called = True

        # Get mask of query positions to analyze
        per_query_mask = self.attention_config.per_query_mask

        # Get positions where mask is nonzero
        mask_positions = np.argwhere(per_query_mask != 0)

        if len(mask_positions) == 0:
            return {}

        # Average over batch and heads
        attn_mean = attn_weight.mean(dim=(0, 1)).cpu().float().numpy()  # [queries, keys]

        # Create mask for inward keys (keys in the input image that correspond to crop mask)
        inward_key_mask = per_query_mask.flatten()  # Shape: (4096,) for 64x64
        inward_key_indices = np.where(inward_key_mask != 0)[0] + IN_IMAGE_ATTN_OFFSET
        outward_key_indices = np.where(inward_key_mask == 0)[0] + IN_IMAGE_ATTN_OFFSET

        num_inward_keys = len(inward_key_indices)
        num_outward_keys = len(outward_key_indices)

        locality_scores_list = []
        inward_outward_ratios_list = []
        sigma = self.attention_config.locality_gaussian_sigma

        for pos_idx, (y, x) in enumerate(mask_positions):
            # Get the query index in the full attention space
            query_idx = OUT_IMAGE_ATTN_OFFSET + y * ATTN_WIDTH + x

            # Extract attention from this query to the input image
            attn_to_input = attn_mean[query_idx, IN_IMAGE_ATTN_OFFSET:IN_IMAGE_ATTN_OFFSET + ATTN_HEIGHT * ATTN_WIDTH]

            # Reshape to 2D
            attn_to_input_2d = attn_to_input.reshape(ATTN_HEIGHT, ATTN_WIDTH)

            # === Locality calculation ===
            # Create Gaussian centered at the same position
            gaussian = self._create_2d_gaussian(y, x, sigma)

            # Flatten for correlation calculation
            attn_flat = attn_to_input_2d.flatten()
            gaussian_flat = gaussian.flatten()

            # Normalize both to zero mean for correlation
            attn_flat_normalized = attn_flat - attn_flat.mean()
            gaussian_flat_normalized = gaussian_flat - gaussian_flat.mean()

            # Calculate Pearson correlation
            numerator = (attn_flat_normalized * gaussian_flat_normalized).sum()
            denominator = np.sqrt((attn_flat_normalized ** 2).sum() * (gaussian_flat_normalized ** 2).sum())

            if denominator > EPSILON:
                correlation = numerator / denominator
            else:
                correlation = 0.0

            locality_scores_list.append(correlation)

            # === Inward/Outward ratio calculation ===
            # Calculate average attention to inward keys (normalized by number of keys)
            inward_attn_sum = attn_flat[inward_key_mask != 0].sum()
            inward_attn_avg = inward_attn_sum # / num_inward_keys if num_inward_keys > 0 else 0.0

            # Calculate average attention to outward keys (normalized by number of keys)
            outward_attn_sum = attn_flat[inward_key_mask == 0].sum()
            outward_attn_avg = outward_attn_sum # / num_outward_keys if num_outward_keys > 0 else 0.0

            # Debug: Print first few values (only once)
            if not hasattr(self, '_locality_ratio_debug_printed'):
                print(f"[LOCALITY DEBUG] First ratio calculation at step={curr_step}, layer={self.layer_num}, pos_idx={pos_idx}")
                print(f"[LOCALITY DEBUG]   attn_flat sum: {attn_flat.sum():.6f} (should be ~1.0 after softmax)")
                print(f"[LOCALITY DEBUG]   attn_flat min/max: {attn_flat.min():.6e} / {attn_flat.max():.6e}")
                print(f"[LOCALITY DEBUG]   inward_attn_sum: {inward_attn_sum:.6e}, num_inward_keys: {num_inward_keys}")
                print(f"[LOCALITY DEBUG]   outward_attn_sum: {outward_attn_sum:.6e}, num_outward_keys: {num_outward_keys}")
                print(f"[LOCALITY DEBUG]   inward_attn_avg: {inward_attn_avg:.6e}, outward_attn_avg: {outward_attn_avg:.6e}")
                self._locality_ratio_debug_printed = True

            # Calculate ratio (inward / outward)
            if outward_attn_avg > EPSILON:
                ratio = inward_attn_avg / outward_attn_avg
            else:
                ratio = float('inf') if inward_attn_avg > EPSILON else 0.0

            inward_outward_ratios_list.append(ratio)

        # Calculate statistics for locality
        locality_scores_array = np.array(locality_scores_list)
        mean_corr = float(locality_scores_array.mean())

        # Calculate statistics for inward/outward ratio (filter out inf values)
        inward_outward_ratios_array = np.array(inward_outward_ratios_list)
        finite_ratios = inward_outward_ratios_array[np.isfinite(inward_outward_ratios_array)]

        # Debug: Print ratio statistics (only once)
        if not hasattr(self, '_locality_ratio_stats_printed'):
            print(f"[LOCALITY DEBUG] First ratio statistics at step={curr_step}, layer={self.layer_num}")
            print(f"[LOCALITY DEBUG]   Total ratios: {len(inward_outward_ratios_array)}, Finite ratios: {len(finite_ratios)}")
            if len(inward_outward_ratios_array) > 0:
                print(f"[LOCALITY DEBUG]   Ratio values (first 10): {inward_outward_ratios_array[:10]}")
            if len(finite_ratios) > 0:
                print(f"[LOCALITY DEBUG]   Finite ratio range: [{finite_ratios.min():.4f}, {finite_ratios.max():.4f}]")
            self._locality_ratio_stats_printed = True

        if len(finite_ratios) > 0:
            mean_ratio = float(finite_ratios.mean())
            std_ratio = float(finite_ratios.std())
            min_ratio = float(finite_ratios.min())
            max_ratio = float(finite_ratios.max())
            median_ratio = float(np.median(finite_ratios))
        else:
            mean_ratio = std_ratio = min_ratio = max_ratio = median_ratio = 0.0

        # Update running average for correlation
        self.running_count += 1
        self.running_avg_correlation = (
            (self.running_avg_correlation * (self.running_count - 1) + mean_corr) / self.running_count
        )

        locality_stats = {
            'step': curr_step,
            'layer': self.layer_num,
            'mean_correlation': mean_corr,
            'std_correlation': float(locality_scores_array.std()),
            'min_correlation': float(locality_scores_array.min()),
            'max_correlation': float(locality_scores_array.max()),
            'median_correlation': float(np.median(locality_scores_array)),
            'num_queries': len(locality_scores_list),
            'running_avg': self.running_avg_correlation,
            'measurement_index': self.running_count,
            # Inward/Outward ratio statistics
            'mean_inward_outward_ratio': mean_ratio,
            'std_inward_outward_ratio': std_ratio,
            'min_inward_outward_ratio': min_ratio,
            'max_inward_outward_ratio': max_ratio,
            'median_inward_outward_ratio': median_ratio,
            'num_finite_ratios': len(finite_ratios),
        }

        # Store for later saving
        self.locality_scores.append(locality_stats)

        # Log to console
        print(f"[LOCALITY] Step {curr_step}, Layer {self.layer_num}: "
              f"corr_mean={locality_stats['mean_correlation']:.4f}, "
              f"corr_std={locality_stats['std_correlation']:.4f}, "
              f"ratio_mean={locality_stats['mean_inward_outward_ratio']:.4f}, "
              f"running_avg={locality_stats['running_avg']:.4f}")

        # Log to wandb if enabled
        if self.attention_config.use_wandb and wandb.run is not None:
            wandb.log({
                f"locality/step_{curr_step}_layer_{self.layer_num}/mean_correlation": locality_stats['mean_correlation'],
                f"locality/step_{curr_step}_layer_{self.layer_num}/std_correlation": locality_stats['std_correlation'],
                f"locality/step_{curr_step}_layer_{self.layer_num}/median_correlation": locality_stats['median_correlation'],
                f"locality/step_{curr_step}_layer_{self.layer_num}/mean_inward_outward_ratio": locality_stats['mean_inward_outward_ratio'],
                f"locality/step_{curr_step}_layer_{self.layer_num}/median_inward_outward_ratio": locality_stats['median_inward_outward_ratio'],
                f"locality/running_avg": locality_stats['running_avg'],
            })

        return locality_stats

    def calc_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        curr_step: int,
        layer_num: int,
        attn_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # set up custom pos embeds for trimap
        if image_rotary_emb is not None:
            if self.attention_config.cont_mode and self.attention_config.should_use_custom_pos_embed(self.curr_step, self.layer_num):
                custom_rotated_querys = []
                custom_rotated_keys = []
                for i in range(self.attention_config.cont_N):
                    #print(f"cont_shrink_factors[i]: {self.attention_config.cont_shrink_factors[i]}")
                    if self.attention_config.cont_shrink_factors[i] == 1.0:
                        #print("using bypass")
                        custom_rotated_querys.append(apply_rotary_emb(query, image_rotary_emb, sequence_dim=1).permute(0, 2, 1, 3))
                        custom_rotated_keys.append(apply_rotary_emb(key, image_rotary_emb, sequence_dim=1).permute(0, 2, 1, 3))
                    else:
                        custom_rotated_querys.append(apply_rotary_emb(query, self.custom_rotary_embs[i], sequence_dim=1).permute(0, 2, 1, 3))
                        custom_rotated_keys.append(apply_rotary_emb(key, self.custom_rotary_embs[i], sequence_dim=1).permute(0, 2, 1, 3))
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            else:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        ### normal attention stuff
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

        # save norm maps for keys and queries (something i did for debugging)
        if self.attention_config.save_norm_maps and self.save_folder is not None:
            in_img_queries = query.mean(dim=1).squeeze()[512 + 64*64:].reshape((64,64,query.shape[-1]))
            in_img_keys = key.mean(dim=1).squeeze()[512 + 64*64:].reshape((64,64,key.shape[-1]))
            in_img_queries = in_img_queries.cpu().float().numpy()
            in_img_keys = in_img_keys.cpu().float().numpy()
            np.savez_compressed(os.path.join(self.save_folder, f"s_{curr_step}_{layer_num}_in_img_queries.npz"), in_img_queries=in_img_queries)
            np.savez_compressed(os.path.join(self.save_folder, f"s_{curr_step}_{layer_num}_in_img_keys.npz"), in_img_keys=in_img_keys)

        ### normal attention stuff
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        self._save_special_coordinates_attention(attn_weight.mean(dim=1).detach(), curr_step, suffix="pre")

        # set crop mask indexes
        mask_idxs = torch.Tensor(self.attention_config.per_query_mask).flatten().nonzero().squeeze() + IN_IMAGE_ATTN_OFFSET

        # here is where we do actual trimap stuff
        if self.attention_config.cont_mode and self.attention_config.should_use_custom_pos_embed(self.curr_step, self.layer_num):
            # get current phase according to current step and cuttoffs
            cuttoffs = self.attention_config.cont_cutoffs
            phase = 0
            on_phase_change = False
            for i, cutoff in enumerate(cuttoffs):
                if curr_step > cutoff:
                    phase = i + 1
                if curr_step == cutoff:
                    on_phase_change = True

            if on_phase_change and curr_step != 0:
                self._set_custom_ids_and_pos_embed(annealing_factor=self.attention_config.cont_annealing_factor ** (phase + 1))

            # here you adjust the in img attention factor according to the phase (kinda hacky right now)
            #diffs_from_high = self.attention_config.cont_in_img_attn_factor_high - self.attention_config.cont_in_img_attn_factors
            #shrink_factors = self.attention_config.cont_step_shrink * (0.3 + 0.7 * ((diffs_from_high) / (diffs_from_high[0]))) * phase
            diffs_from_one = 1.0 - self.attention_config.cont_in_img_attn_factors
            if self.attention_config.anneal_below_zero:
                af = self.attention_config.cont_annealing_factor ** (phase + 1)
                diffs_from_one = np.where(diffs_from_one < 0, diffs_from_one * af, diffs_from_one)
            shrink_factors = diffs_from_one * (self.attention_config.cont_step_shrink ** phase)

            #shrink_factors = self.attention_config.cont_step_shrink * phase
            #cont_in_img_attn_factors = self.attention_config.cont_in_img_attn_factors - shrink_factors
            cont_in_img_attn_factors = 1.0 - shrink_factors

            # here we actually apply the process
            saliency = self.attention_config.saliency
            saliency = saliency * self.attention_config.cont_saliency_boost
            saliency = np.clip(saliency, 0.0, 1.0)

            per_query_mask = self.attention_config.per_query_mask

            # Make a copy of saliency
            filtered_saliency = np.copy(saliency)

            # Only keep pixels where per_query_mask is nonzero, set others to -1
            filtered_saliency[per_query_mask == 0] = -1
            for i in range(self.attention_config.cont_N):
                # Find indices in saliency map where values are in [i/N, (i+1)/N)
                epsilon = 1e-1 * (i == (self.attention_config.cont_N - 1))
                saliency_flat = torch.Tensor(filtered_saliency).flatten()
                lower = i * (1.0 / self.attention_config.cont_N)
                upper = (i + 1) * (1.0 / self.attention_config.cont_N)
                curr_idxs = ((saliency_flat >= lower) & (saliency_flat < (upper + epsilon))).nonzero().squeeze()
                if curr_idxs.dim() == 0:
                    continue
                curr_idxs = curr_idxs + OUT_IMAGE_ATTN_OFFSET
                indexed_query = custom_rotated_querys[i][:, :, curr_idxs, :]
                curr_attn = indexed_query @ custom_rotated_keys[i].transpose(-2, -1) * scale_factor
                curr_attn[:, :, :, mask_idxs] = curr_attn[:, :, :, mask_idxs] * cont_in_img_attn_factors[i]

                # Log attention for special coordinates if debug attention saving is enabled.
                if self.attention_config.save_attention_maps and self.save_folder is not None:
                    self._save_special_coordinates_cont_mode(curr_attn, curr_idxs, curr_step)

                attn_weight[:, :, curr_idxs, IN_IMAGE_ATTN_OFFSET:] = curr_attn[:, :, :, IN_IMAGE_ATTN_OFFSET:]

        attn_weight += attn_bias

        # apply masking
        if self.attention_config.should_apply_masking(curr_step, layer_num) and not self.attention_config.use_smoothing and not self.attention_config.post_softmax:
            attn_weight = attn_weight * self.attention_config.attn_mask.to(attn_weight.device).detach()
            #attn_weight[:, :, mask_idxs_output, mask_idxs_output] = attn_weight[:, :, mask_idxs_output, mask_idxs_output] * self.attention_config.in_image_mask_val


        # apply smoothing
        if self.attention_config.use_smoothing:
            orig_shape = attn_weight.shape
            attn_weight = attn_weight.reshape((-1, orig_shape[-1]))
            mask = self.attention_config.attn_mask.flatten() != 1.0
            attn_weight = self._smooth_subset_preserve_mass(
                attn_weight,
                mask,
                self.attention_config.in_image_mask_val
            )
            attn_weight = attn_weight.reshape(orig_shape)

        attn_weight = torch.softmax(attn_weight, dim=-1)

        # apply masking post softmax
        if self.attention_config.post_softmax:
            attn_weight = attn_weight * self.attention_config.attn_mask.to(attn_weight.device).detach()

        if self.attention_config.should_use_text_emphasizing(curr_step, layer_num):
            attn_weight = attn_weight * self.attention_config.text_emphasizing_mask.to(attn_weight.device).detach()

        # Record attention locality if enabled (AFTER softmax)
        if self.attention_config.should_record_attention_locality(curr_step, layer_num):
            self._calculate_attention_locality(attn_weight, curr_step)

        attention_weights_mean = attn_weight.mean(dim=1).detach()

        if self.attention_config.save_attention_maps and self.save_folder is not None:
            # accumulate attention weights across layers
            self._accumulate_attention_maps(attention_weights_mean, curr_step)
            self._save_special_coordinates_attention(attention_weights_mean, curr_step, suffix="post_all")

            # save and reset if this is the last layer
            if layer_num == LAST_LAYER_NUM:
                self._save_accumulated_attention_maps(curr_step)
                self._reset_accumulated_attention_maps()


        #print(f"attention_weights_mean.shape: {attention_weights_mean.shape}")
        #print(f"curr_step: {curr_step}, layer_num: {layer_num}, save_folder: {save_folder}")

        out = attn_weight @ value
        out = out.permute(0, 2, 1, 3)
        return out

    def _smooth_subset_preserve_mass(self, logits, mask, alpha):
        """
        logits: (B, N) float tensor
        mask:   (N,) bool tensor selecting the columns to smooth (same for all B)
        alpha:  in [0,1]; 0 => fully flat inside mask, 1 => unchanged
        Returns: adjusted logits (B, N) with same softmax mass on the masked subset per row.
        """
        if logits.dim() != 2:
            raise ValueError("logits must be 2D (B, N)")
        B, N = logits.shape
        mask = mask.to(device=logits.device, dtype=torch.bool)
        if mask.numel() != N:
            raise ValueError("mask length must match N")

        k = int(mask.sum().item())
        if k == 0:
            return logits.clone()

        with torch.no_grad():
            zS = logits[:, mask]                       # (B, k)
            m  = zS.mean(dim=1, keepdim=True)          # (B, 1)
            gS = m + alpha * (zS - m)                  # (B, k)

            # Offset to preserve total softmax mass on the masked subset
            b = torch.logsumexp(zS, dim=1, keepdim=True) - \
                torch.logsumexp(gS, dim=1, keepdim=True)  # (B, 1)

            out = logits.clone()
            out[:, mask] = gS + b                       # broadcast b across masked columns
        return out

    def _accumulate_attention_maps(self, attention_weights_mean, curr_step):
        """Accumulate attention maps across layers for each word/coordinate."""
        # Save textual attention
        if self.attention_config.words_to_save is not None:
            for word in self.attention_config.words_to_save:
                idxs = self.attention_config.text_offsets[word]
                attn_for_word = attention_weights_mean.squeeze()[idxs].cpu().float().numpy().max(axis=0)

                key = word
                if key not in self.accumulated_attention_maps:
                    self.accumulated_attention_maps[key] = []
                self.accumulated_attention_maps[key].append(attn_for_word)

        # Save specific attention points within crop mask
        if self.attention_config.inside_coords is not None:
            for inside_coord in self.attention_config.inside_coords:
                idxs = [OUT_IMAGE_ATTN_OFFSET + inside_coord[0] * ATTN_WIDTH + inside_coord[1]]
                attn_for_point = attention_weights_mean.squeeze()[idxs].cpu().float().numpy().max(axis=0)

                key = f"inside_{inside_coord[0]}_{inside_coord[1]}"
                if key not in self.accumulated_attention_maps:
                    self.accumulated_attention_maps[key] = []
                self.accumulated_attention_maps[key].append(attn_for_point)

                # accumulate complement attention
                comp_idxs = [IN_IMAGE_ATTN_OFFSET + inside_coord[0] * ATTN_WIDTH + inside_coord[1]]
                comp_attn_for_point = attention_weights_mean.squeeze()[comp_idxs].cpu().float().numpy().max(axis=0)

                comp_key = f"complement_inside_{inside_coord[0]}_{inside_coord[1]}"
                if comp_key not in self.accumulated_attention_maps:
                    self.accumulated_attention_maps[comp_key] = []
                self.accumulated_attention_maps[comp_key].append(comp_attn_for_point)

        # Save specific attention points outside crop mask
        if self.attention_config.outside_coords is not None:
            for outside_coord in self.attention_config.outside_coords:
                idxs = [OUT_IMAGE_ATTN_OFFSET + outside_coord[0] * ATTN_WIDTH + outside_coord[1]]
                attn_for_point = attention_weights_mean.squeeze()[idxs].cpu().float().numpy().max(axis=0)

                key = f"outside_{outside_coord[0]}_{outside_coord[1]}"
                if key not in self.accumulated_attention_maps:
                    self.accumulated_attention_maps[key] = []
                self.accumulated_attention_maps[key].append(attn_for_point)

                # accumulate complement attention
                comp_idxs = [IN_IMAGE_ATTN_OFFSET + outside_coord[0] * ATTN_WIDTH + outside_coord[1]]
                comp_attn_for_point = attention_weights_mean.squeeze()[comp_idxs].cpu().float().numpy().max(axis=0)

                comp_key = f"complement_outside_{outside_coord[0]}_{outside_coord[1]}"
                if comp_key not in self.accumulated_attention_maps:
                    self.accumulated_attention_maps[comp_key] = []
                self.accumulated_attention_maps[comp_key].append(comp_attn_for_point)

    def _save_special_coordinates_attention(self, attention_weights_mean, curr_step, suffix="pre"):
        # Save special coordinates attention only when curr_step < 4
        if curr_step < 4 and self.attention_config.special_coords is not None:
            for special_coord in self.attention_config.special_coords:
                idxs = [OUT_IMAGE_ATTN_OFFSET + special_coord[0] * ATTN_WIDTH + special_coord[1]]
                attn_for_point = attention_weights_mean.squeeze()[idxs].cpu().float().numpy().max(axis=0)
                attn_for_point = attn_for_point[IN_IMAGE_ATTN_OFFSET:].reshape(ATTN_HEIGHT, ATTN_WIDTH)
                # save as png with colormap jet
                filename = f"s_{curr_step}_{self.layer_num}_special_{special_coord[0]}_{special_coord[1]}_{suffix}"
                plt.imshow(attn_for_point, cmap='jet')
                plt.colorbar()
                plt.savefig(os.path.join(self.save_folder, filename + ".png"))
                plt.close()
                print(f"attn_for_point: {special_coord}, shape: {attn_for_point.shape}")
                # save as npy
                np.save(os.path.join(self.save_folder, filename + ".npy"), attn_for_point)

    def _save_special_coordinates_cont_mode(self, curr_attn, curr_idxs, curr_step):
        """Save special coordinates attention in continuous mode after custom RoPE application.

        Args:
            curr_attn: Current attention tensor after RoPE [batch, heads, queries, keys]
            curr_idxs: Current indices being processed (with OUT_IMAGE_ATTN_OFFSET already added)
            curr_step: Current diffusion step
        """
        if curr_step >= 4 or self.attention_config.special_coords is None:
            return

        print(f"[SPECIAL COORDS] Step {curr_step}, Layer {self.layer_num}: Checking for special coordinates...")

        for special_coord in self.attention_config.special_coords:
            # Convert special coord [y, x] to full latent space index
            special_idx_full = special_coord[0] * ATTN_WIDTH + special_coord[1] + OUT_IMAGE_ATTN_OFFSET

            # Check if this special coordinate is in curr_idxs
            if curr_idxs.dim() == 0:
                # Handle scalar case
                if curr_idxs.item() == special_idx_full:
                    pos_in_batch = 0
                    is_present = True
                else:
                    is_present = False
            else:
                # Handle tensor case
                matches = (curr_idxs == special_idx_full)
                is_present = matches.any().item()
                if is_present:
                    pos_in_batch = matches.nonzero(as_tuple=True)[0].item()

            if is_present:
                print(f"[SPECIAL COORDS] Found special coord {special_coord} at position {pos_in_batch} in current batch")

                # Extract attention for this specific query
                attn_for_special = curr_attn.mean(dim=1).squeeze()[pos_in_batch].cpu().float().numpy()
                attn_for_special_in_img = attn_for_special[IN_IMAGE_ATTN_OFFSET:].reshape(ATTN_HEIGHT, ATTN_WIDTH)

                # Save as png with colormap jet
                filename = f"s_{curr_step}_{self.layer_num}_special_{special_coord[0]}_{special_coord[1]}_post_rope"
                plt.imshow(attn_for_special_in_img, cmap='jet')
                plt.colorbar()
                plt.savefig(os.path.join(self.save_folder, filename + ".png"))
                plt.close()

                # Save as npy
                np.save(os.path.join(self.save_folder, filename + ".npy"), attn_for_special_in_img)

                print(f"[SPECIAL COORDS] Saved {filename} with shape {attn_for_special_in_img.shape}")

    def _save_accumulated_attention_maps(self, curr_step):
        """Save all accumulated attention maps in a single file for this step."""
        # Create a dictionary with all attention maps for this step
        step_attention_maps = {}
        for key, attention_list in self.accumulated_attention_maps.items():
            # Stack all layers to create [NUM_LAYERS, ORIGINAL_ATTENTION_DIMS]
            stacked_attention = np.stack(attention_list, axis=0)
            step_attention_maps[key] = stacked_attention

        # Save all attention maps for this step in a single file
        filename = f"s_{curr_step}_attention_maps.npz"
        np.savez_compressed(os.path.join(self.save_folder, filename), **step_attention_maps)

    def _reset_accumulated_attention_maps(self):
        """Reset the accumulated attention maps dictionary."""
        self.accumulated_attention_maps = {}

    def save_locality_scores(self):
        """Save all accumulated locality scores to JSON and CSV files."""
        print(f"[LOCALITY DEBUG] save_locality_scores called")
        print(f"[LOCALITY DEBUG] self.locality_scores length: {len(self.locality_scores)}")
        print(f"[LOCALITY DEBUG] self.save_folder: {self.save_folder}")

        if not self.locality_scores:
            print(f"[LOCALITY DEBUG] Early return: locality_scores is empty")
            return

        if self.save_folder is None:
            print(f"[LOCALITY DEBUG] Early return: save_folder is None")
            return

        # Save to parent folder of attention_maps (e.g., outputs/experiment_name/)
        output_folder = os.path.dirname(self.save_folder) if self.save_folder else "."
        os.makedirs(output_folder, exist_ok=True)

        # Save as JSON
        locality_file = os.path.join(output_folder, "attention_locality_scores.json")
        with open(locality_file, 'w') as f:
            json.dump(self.locality_scores, f, indent=2)

        print(f"[LOCALITY] Saved {len(self.locality_scores)} locality measurements to {locality_file}")

        # Save as CSV
        csv_file = os.path.join(output_folder, "attention_locality_scores.csv")
        with open(csv_file, 'w') as f:
            # Write header
            f.write("measurement_index,step,layer,mean_correlation,std_correlation,min_correlation,max_correlation,median_correlation,num_queries,running_avg,mean_inward_outward_ratio,std_inward_outward_ratio,min_inward_outward_ratio,max_inward_outward_ratio,median_inward_outward_ratio,num_finite_ratios\n")
            # Write data rows
            for score in self.locality_scores:
                f.write(f"{score['measurement_index']},{score['step']},{score['layer']},"
                       f"{score['mean_correlation']:.6f},{score['std_correlation']:.6f},"
                       f"{score['min_correlation']:.6f},{score['max_correlation']:.6f},"
                       f"{score['median_correlation']:.6f},{score['num_queries']},"
                       f"{score['running_avg']:.6f},"
                       f"{score['mean_inward_outward_ratio']:.6f},{score['std_inward_outward_ratio']:.6f},"
                       f"{score['min_inward_outward_ratio']:.6f},{score['max_inward_outward_ratio']:.6f},"
                       f"{score['median_inward_outward_ratio']:.6f},{score['num_finite_ratios']}\n")

        print(f"[LOCALITY] Saved CSV to {csv_file}")

        # Save running average as a separate CSV for easy plotting
        running_avg_csv = os.path.join(output_folder, "attention_locality_running_avg.csv")
        with open(running_avg_csv, 'w') as f:
            f.write("measurement_index,step,layer,running_avg\n")
            for score in self.locality_scores:
                f.write(f"{score['measurement_index']},{score['step']},{score['layer']},"
                       f"{score['running_avg']:.6f}\n")

        print(f"[LOCALITY] Saved running average CSV to {running_avg_csv}")

        # Save inward/outward ratios as a separate CSV for easy plotting
        ratios_csv = os.path.join(output_folder, "attention_inward_outward_ratios.csv")
        with open(ratios_csv, 'w') as f:
            f.write("measurement_index,step,layer,mean_inward_outward_ratio,median_inward_outward_ratio\n")
            for score in self.locality_scores:
                f.write(f"{score['measurement_index']},{score['step']},{score['layer']},"
                       f"{score['mean_inward_outward_ratio']:.6f},"
                       f"{score['median_inward_outward_ratio']:.6f}\n")

        print(f"[LOCALITY] Saved inward/outward ratios CSV to {ratios_csv}")

        # Also save a summary statistics file
        if len(self.locality_scores) > 0:
            # Calculate overall statistics
            all_means = [score['mean_correlation'] for score in self.locality_scores]
            summary = {
                'total_measurements': len(self.locality_scores),
                'overall_mean_correlation': float(np.mean(all_means)),
                'overall_std_correlation': float(np.std(all_means)),
                'overall_min_correlation': float(np.min(all_means)),
                'overall_max_correlation': float(np.max(all_means)),
                'per_step_summary': {}
            }

            # Group by step
            for score in self.locality_scores:
                step = score['step']
                if step not in summary['per_step_summary']:
                    summary['per_step_summary'][step] = {
                        'mean_correlations': [],
                        'num_layers': 0
                    }
                summary['per_step_summary'][step]['mean_correlations'].append(score['mean_correlation'])
                summary['per_step_summary'][step]['num_layers'] += 1

            # Calculate per-step averages
            for step, data in summary['per_step_summary'].items():
                data['avg_correlation'] = float(np.mean(data['mean_correlations']))
                del data['mean_correlations']  # Remove raw data

            summary_file = os.path.join(output_folder, "attention_locality_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            # Save summary as CSV too
            summary_csv_file = os.path.join(output_folder, "attention_locality_summary.csv")
            with open(summary_csv_file, 'w') as f:
                f.write("step,avg_correlation,num_layers\n")
                for step in sorted(summary['per_step_summary'].keys()):
                    data = summary['per_step_summary'][step]
                    f.write(f"{step},{data['avg_correlation']:.6f},{data['num_layers']}\n")
                # Add overall statistics at the end
                f.write(f"\n# Overall Statistics\n")
                f.write(f"# total_measurements,{summary['total_measurements']}\n")
                f.write(f"# overall_mean_correlation,{summary['overall_mean_correlation']:.6f}\n")
                f.write(f"# overall_std_correlation,{summary['overall_std_correlation']:.6f}\n")
                f.write(f"# overall_min_correlation,{summary['overall_min_correlation']:.6f}\n")
                f.write(f"# overall_max_correlation,{summary['overall_max_correlation']:.6f}\n")

            print(f"[LOCALITY] Overall mean correlation: {summary['overall_mean_correlation']:.4f}")
            print(f"[LOCALITY] Summary saved to {summary_file}")
            print(f"[LOCALITY] Summary CSV saved to {summary_csv_file}")

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Print config status at step 3 (first step after VLM verdict at step 2) for layer 0
        # if self.curr_step == 3 and self.layer_num == 0:
        #     print(f"\n{'='*80}")
        #     print(f"[ATTN PROCESSOR] Step {self.curr_step}, Layer {self.layer_num}")
        #     print(f"[ATTN PROCESSOR] Current config values:")
        #     print(f"[ATTN PROCESSOR]   neglect_mode: {self.attention_config.neglect_mode}")
        #     print(f"[ATTN PROCESSOR]   in_image_mask_val: {self.attention_config.in_image_mask_val}")
        #     print(f"[ATTN PROCESSOR]   text_emphasizing_mask_val: {self.attention_config.text_emphasizing_mask_val}")
        #     # Print continuous mode parameters if cont_mode is enabled
        #     if hasattr(self.attention_config, 'cont_mode') and self.attention_config.cont_mode:
        #         print(f"[ATTN PROCESSOR] Continuous mode parameters:")
        #         print(f"[ATTN PROCESSOR]   cont_shrink_factor_low: {self.attention_config.cont_shrink_factor_low}")
        #         print(f"[ATTN PROCESSOR]   cont_in_img_attn_factor_low: {self.attention_config.cont_in_img_attn_factor_low}")
        #         print(f"[ATTN PROCESSOR]   cont_in_img_attn_factor_high: {self.attention_config.cont_in_img_attn_factor_high}")
        #         print(f"[ATTN PROCESSOR]   cont_step_shrink: {self.attention_config.cont_step_shrink}")
        #         print(f"[ATTN PROCESSOR]   cont_saliency_boost: {self.attention_config.cont_saliency_boost}")
        #     print(f"{'='*80}\n")

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        hidden_states = self.calc_attention(
            query, key, value,
            self.curr_step,
            self.layer_num,
            attn_mask=attention_mask,
            image_rotary_emb=image_rotary_emb
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        self.curr_step += 1

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
