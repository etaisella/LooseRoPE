"""
Shared VLM (Vision-Language Model) verdict functionality for LooseRoPE pipelines.

This mixin provides VLM init, context loading, verdict logging, and config
update methods that work with both LooseRoPEPipeline and QwenLooseRoPEPipeline.
"""

import os
import re
import time
import copy
import tempfile
import shutil
import importlib
import torch
import logging

from PIL import Image

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None


def qwen_vl_hf_repo_id(model_size: str) -> str:
    """Map `vlm_model_size` from YAML to a Hugging Face repo id.

    Instruct builds use the ``-Instruct`` suffix. Thinking models (and similar)
    are published without it, e.g. ``Qwen/Qwen3-VL-8B-Thinking``.
    """
    ms = (model_size or "4B").strip()
    if "-Thinking" in ms:
        return f"Qwen/Qwen3-VL-{ms}"
    return f"Qwen/Qwen3-VL-{ms}-Instruct"


def vlm_exceeds_4b(model_size: str) -> bool:
    """True if `vlm_model_size` denotes more than 4B params (e.g. 8B, 8B-Thinking, 30B-A3B-Thinking)."""
    m = re.match(r"^(\d+)B", (model_size or "4B").strip(), re.IGNORECASE)
    return int(m.group(1)) > 4 if m else False


def vlm_should_swap_for_verdict(attention_config) -> bool:
    """YAML ``perform_offloading`` OR VLM >4B (swap diffusion off GPU for verdict)."""
    if attention_config is None:
        return False
    ms = getattr(attention_config, "vlm_model_size", "4B")
    return bool(getattr(attention_config, "perform_offloading", False)) or vlm_exceeds_4b(ms)


class LooseRoPEVLMMixin:
    """
    Mixin providing VLM verdict functionality for any LooseRoPE pipeline.

    Methods here are lifted from LooseRoPEPipeline so they can be shared
    with QwenLooseRoPEPipeline without code duplication.
    """

    def _initialize_vlm(self, model_size="4B"):
        print(f"\n{'='*80}")
        print(f"[VLM INIT] Starting VLM initialization with model size: {model_size}")
        print(f"{'='*80}\n")

        try:
            print("[VLM INIT] Importing transformers and qwen_vl_utils...")
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info  # noqa: F401
            print("[VLM INIT] Imports successful")

            _has_fa2 = False
            try:
                _has_fa2 = importlib.util.find_spec("flash_attn") is not None
            except Exception:
                _has_fa2 = False

            _attn_impl = "flash_attention_2" if _has_fa2 else "eager"
            print(f"[VLM INIT] Using attention implementation: {_attn_impl}")

            repo_id = qwen_vl_hf_repo_id(model_size)
            _devmap = "cpu" if vlm_exceeds_4b(model_size) else "auto"
            print(f"[VLM INIT] Loading VLM model: {repo_id} (device_map={_devmap})...")
            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                repo_id,
                torch_dtype=torch.bfloat16,
                attn_implementation=_attn_impl,
                device_map=_devmap,
            )
            print(f"[VLM INIT] VLM model loaded successfully")
            if vlm_exceeds_4b(model_size):
                torch.cuda.empty_cache()
                print("[VLM INIT] VLM >4B: kept on CPU until verdict (frees GPU for diffusion)")

            print(f"[VLM INIT] Loading VLM processor...")
            self.vlm_processor = AutoProcessor.from_pretrained(repo_id)
            print(f"[VLM INIT] VLM processor loaded successfully")

            self.vlm_enabled = True
            self.perform_offloading = False
            print(f"\n[VLM INIT] VLM initialization COMPLETE\n")

        except ImportError as e:
            print(f"[VLM INIT] Import error: {e}")
            logger.warning(f"VLM dependencies not available: {e}. VLM verdict will be disabled.")
            self.vlm_enabled = False
        except Exception as e:
            print(f"[VLM INIT] Error loading VLM: {e}")
            logger.error(f"Error loading VLM model: {e}. VLM verdict will be disabled.")
            self.vlm_enabled = False

    def _load_vlm_context_examples(self, context_folder=None, use_simplified_instruction=False):
        print(f"\n{'='*80}")
        print(f"[VLM CONTEXT] Loading VLM context examples...")
        print(f"{'='*80}\n")

        if not getattr(self, 'vlm_enabled', False):
            print(f"[VLM CONTEXT] VLM not enabled, skipping")
            return

        if use_simplified_instruction:
            print("[VLM CONTEXT] use_simplified_instruction: skipping few-shot example images")
            print(f"{'='*80}\n")
            return

        timestep = 2
        if hasattr(self, 'attention_config') and self.attention_config is not None:
            timestep = self.attention_config.vlm_verdict_timestep
        print(f"[VLM CONTEXT] Using timestep: {timestep}")

        try:
            if context_folder is None:
                context_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vlm_in_context_inputs")

            print(f"[VLM CONTEXT] Context folder: {context_folder}")

            names = [
                ("success", "vlm_context_success"),
                ("success2", "vlm_context_success2"),
                ("neglect", "vlm_context_neglect"),
                ("neglect2", "vlm_context_neglect2"),
                ("suppression", "vlm_context_suppression"),
                ("suppression2", "vlm_context_suppression2"),
            ]
            for prefix, attr_prefix in names:
                input_path = os.path.join(context_folder, f"{prefix}_input.png")
                x0_path = os.path.join(context_folder, f"{prefix}_x0_ts{timestep}.png")
                setattr(self, f"{attr_prefix}_input", Image.open(input_path).convert("RGB").resize((512, 512)))
                setattr(self, f"{attr_prefix}_x0", Image.open(x0_path).convert("RGB").resize((512, 512)))
                print(f"[VLM CONTEXT] Loaded {prefix} input + x0")

            print(f"\n[VLM CONTEXT] All context examples loaded (timestep={timestep})\n")

        except Exception as e:
            print(f"[VLM CONTEXT] Error loading context examples: {e}")
            logger.error(f"Error loading VLM context examples: {e}. VLM verdict will be disabled.")
            self.vlm_enabled = False

    def _log_vlm_verdict(self, verdict, vlm_try_number, timestep):
        if not hasattr(self, 'output_folder') or self.output_folder is None:
            return

        log_file_path = os.path.join(self.output_folder, "vlm_verdicts.txt")
        try:
            attn_config = self.attention_config
            current_saliency_boost = attn_config.cont_saliency_boost
            current_attn_factor_low = attn_config.cont_in_img_attn_factor_low
            current_shrink_factor_low = attn_config.cont_shrink_factor_low
            current_attn_factors = attn_config.cont_in_img_attn_factors if attn_config.cont_mode else None
            current_shrink_factors = attn_config.cont_shrink_factors if attn_config.cont_mode else None

            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
            attn_factors_str = ", ".join([f"{f:.4f}" for f in current_attn_factors]) if current_attn_factors is not None else "N/A"
            shrink_factors_str = ", ".join([f"{f:.4f}" for f in current_shrink_factors]) if current_shrink_factors is not None else "N/A"

            log_entry = f"""
{'='*80}
Timestamp: {timestamp_str}
VLM Try: {vlm_try_number}
Timestep: {timestep}
Classification: {verdict['classification']}
Reasoning: {verdict['reasoning']}
Current Saliency Boost: {current_saliency_boost:.4f}
Current Attn Factor Low: {current_attn_factor_low:.4f}
Current Attn Factors: [{attn_factors_str}]
Current Shrink Factor Low: {current_shrink_factor_low:.4f}
Current Shrink Factors: [{shrink_factors_str}]
{'='*80}

"""
            with open(log_file_path, 'a') as f:
                f.write(log_entry)

            print(f"[VLM LOG] Verdict logged to: {log_file_path}")

        except Exception as e:
            logger.error(f"Error logging VLM verdict: {e}")

    def _set_params_by_verdict(self, verdict):
        print(f"\n{'='*80}")
        print(f"VLM Verdict: {verdict['classification']}")
        print(f"Reasoning: {verdict['reasoning']}")
        print(f"{'='*80}\n")

        if wandb is not None and hasattr(self, 'attention_config') and self.attention_config.use_wandb:
            try:
                wandb.log({"vlm_classification": verdict['classification'], "vlm_reasoning": verdict['reasoning']}, step=2)
            except Exception:
                pass

        classification = verdict['classification'].lower()

        boost_addition = 0.0
        attn_low_addition = 0.0
        shrink_low_addition = 0.0
        if classification == 'success':
            return
        elif classification == 'neglect':
            boost_addition = getattr(self.attention_config, 'vlm_boost_neglect', -0.08)
            attn_low_addition = getattr(self.attention_config, 'vlm_attn_low_neglect', 0.0)
            shrink_low_addition = getattr(self.attention_config, 'vlm_shrink_low_neglect', 0.0)
        elif classification == 'suppression':
            boost_addition = getattr(self.attention_config, 'vlm_boost_suppression', 0.1)
            attn_low_addition = getattr(self.attention_config, 'vlm_attn_low_suppression', 0.0)
            shrink_low_addition = getattr(self.attention_config, 'vlm_shrink_low_suppression', 0.0)
        else:
            print(f"[CONFIG UPDATE] Unknown verdict '{verdict['classification']}', treating as success")
            verdict['classification'] = 'success'
            return

        layer_num = 0
        for block in self.transformer.transformer_blocks:
            if hasattr(block.attn, 'processor') and hasattr(block.attn.processor, 'attention_config'):
                if layer_num == 0:
                    proc = block.attn.processor
                    old_boost = proc.attention_config.cont_saliency_boost
                    old_attn_low = proc.attention_config.cont_in_img_attn_factor_low
                    old_shrink_low = proc.attention_config.cont_shrink_factor_low

                    proc.attention_config.cont_saliency_boost += boost_addition
                    proc.attention_config.cont_in_img_attn_factor_low += attn_low_addition
                    proc.attention_config.cont_shrink_factor_low += shrink_low_addition
                    proc.attention_config.update_cont_in_img_attn_factors()
                    proc.attention_config.update_cont_shrink_factors()

                    print(f"[CONFIG UPDATE] Saliency Boost: {old_boost:.4f} -> {proc.attention_config.cont_saliency_boost:.4f}")
                    print(f"[CONFIG UPDATE] Attn Factor Low: {old_attn_low:.4f} -> {proc.attention_config.cont_in_img_attn_factor_low:.4f}")
                    print(f"[CONFIG UPDATE] Shrink Factor Low: {old_shrink_low:.4f} -> {proc.attention_config.cont_shrink_factor_low:.4f}")
                block.attn.processor.curr_step = 0
            layer_num += 1

        if hasattr(self.transformer, 'single_transformer_blocks'):
            for block in self.transformer.single_transformer_blocks:
                if hasattr(block.attn, 'processor') and hasattr(block.attn.processor, 'attention_config'):
                    block.attn.processor.curr_step = 0
                layer_num += 1

        print(f"[CONFIG UPDATE] All {layer_num} processors updated\n")

    def _reset_all_vlm_modified_params(self):
        for block in self.transformer.transformer_blocks:
            if hasattr(block.attn, 'processor') and hasattr(block.attn.processor, 'attention_config'):
                block.attn.processor.attention_config.reset_all_vlm_modified_params()

        if hasattr(self.transformer, 'single_transformer_blocks'):
            for block in self.transformer.single_transformer_blocks:
                if hasattr(block.attn, 'processor') and hasattr(block.attn.processor, 'attention_config'):
                    block.attn.processor.attention_config.reset_all_vlm_modified_params()

        print(f"[CONFIG RESET] All VLM-modified parameters reset to original values")

    def _save_all_locality_scores(self):
        processor_with_scores = None
        for block in self.transformer.transformer_blocks:
            if hasattr(block.attn, 'processor') and hasattr(block.attn.processor, 'locality_scores'):
                if len(block.attn.processor.locality_scores) > 0:
                    processor_with_scores = block.attn.processor
                    break

        if processor_with_scores is None and hasattr(self.transformer, 'single_transformer_blocks'):
            for block in self.transformer.single_transformer_blocks:
                if hasattr(block.attn, 'processor') and hasattr(block.attn.processor, 'locality_scores'):
                    if len(block.attn.processor.locality_scores) > 0:
                        processor_with_scores = block.attn.processor
                        break

        if processor_with_scores is not None:
            processor_with_scores.save_locality_scores()
