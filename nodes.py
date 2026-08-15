import math
import os
import glob
from functools import wraps
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
import torch
import folder_paths
from comfy.utils import ProgressBar
try:
    import comfy.model_management as model_management
except ImportError:  # Allows lightweight tests outside a full ComfyUI install.
    model_management = None

from .inference import BiMVFIModel, EMAVFIModel, SGMVFIModel, GIMMVFIModel
from .speed_backend import SpeedVFIModel
from .ldf_backend import LDFVFIModel
from .external_sources import ensure_upstream_source
from .bim_vfi_arch import clear_backwarp_cache
from .ema_vfi_arch import clear_warp_cache as clear_ema_warp_cache
from .sgm_vfi_arch import clear_warp_cache as clear_sgm_warp_cache
from .gimm_vfi_arch import clear_gimm_caches

logger = logging.getLogger("Tween")


def _with_elapsed_seconds(label=None):
    """Append wall-clock node execution time without disturbing existing outputs."""
    def decorate(method):
        @wraps(method)
        def timed(self, *args, **kwargs):
            started = time.perf_counter()
            outputs = method(self, *args, **kwargs)
            elapsed_seconds = time.perf_counter() - started
            output_label = label or getattr(self, "MODEL_LABEL", self.__class__.__name__)
            logger.info(
                "%s: node completed in %.2f seconds", output_label, elapsed_seconds
            )
            return (*outputs, round(elapsed_seconds, 3))

        return timed

    return decorate


def _get_torch_device():
    """Honor ComfyUI's selected device instead of assuming the default CUDA GPU."""
    if model_management is not None:
        return torch.device(model_management.get_torch_device())
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _soft_empty_cache():
    if model_management is not None:
        model_management.soft_empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _throw_if_interrupted():
    if model_management is not None:
        model_management.throw_exception_if_processing_interrupted()


def _target_fps_enabled(source_fps, target_fps):
    if target_fps > 0 and source_fps <= 0:
        raise ValueError("source_fps must be greater than 0 when target_fps is enabled")
    return target_fps > 0 and source_fps > 0


def _interpolate_batch_with_offload(model, frames0, frames1, device, keep_device):
    """Run one pair batch and reliably return an optionally offloaded model to CPU."""
    if keep_device:
        return model.interpolate_batch(frames0, frames1, time_step=0.5)

    try:
        model.to(device)
        return model.interpolate_batch(frames0, frames1, time_step=0.5)
    finally:
        inference_failed = sys.exc_info()[0] is not None
        try:
            model.to("cpu")
        except Exception:
            # Preserve an inference exception if one is already active.
            if not inference_failed:
                raise
            logger.exception("Failed to offload VFI model after an inference error")


def _interpolate_multi_with_offload(model, frame0, frame1, num_intermediates,
                                    device, keep_device):
    if keep_device:
        return model.interpolate_multi(frame0, frame1, num_intermediates)

    try:
        model.to(device)
        return model.interpolate_multi(frame0, frame1, num_intermediates)
    finally:
        inference_failed = sys.exc_info()[0] is not None
        try:
            model.to("cpu")
        except Exception:
            if not inference_failed:
                raise
            logger.exception("Failed to offload VFI model after an inference error")




def _get_system_ram_gb():
    """Return total system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 ** 2)  # kB -> GB
    except (OSError, ValueError):
        pass
    return 16.0  # safe fallback


def _apply_vfi_settings(settings, batch_size, chunk_size, keep_device,
                        all_on_gpu, clear_cache_after_n_frames):
    """Override manual values with optimizer settings if provided."""
    if settings is None:
        return batch_size, chunk_size, keep_device, all_on_gpu, clear_cache_after_n_frames
    return (
        settings["batch_size"],
        settings["chunk_size"],
        settings["keep_device"],
        settings["all_on_gpu"],
        settings["clear_cache_after_n_frames"],
    )


def _clear_model_cache(model):
    """Clear warp caches based on model type."""
    if isinstance(model, BiMVFIModel):
        clear_backwarp_cache()
    elif isinstance(model, EMAVFIModel):
        clear_ema_warp_cache()
    elif isinstance(model, SGMVFIModel):
        clear_sgm_warp_cache()
    elif isinstance(model, GIMMVFIModel):
        clear_gimm_caches()
    elif isinstance(model, SpeedVFIModel):
        model.clear_cache()
    elif isinstance(model, LDFVFIModel):
        model.clear_cache()
    _soft_empty_cache()


def _compute_target_fps_params(source_fps, target_fps):
    """Compute oversampling parameters for target FPS mode.

    Returns (num_passes, mult) where mult = 2^num_passes is the power-of-2
    multiplier needed to oversample above the target ratio.
    """
    ratio = target_fps / source_fps
    if ratio <= 1.0:
        return 0, 1  # no interpolation needed (downsampling or same fps)
    num_passes = math.ceil(math.log2(ratio))
    mult = 2 ** num_passes
    if mult > 8:
        raise ValueError(
            f"Pairwise Tween nodes support target-FPS interpolation up to 8x; "
            f"{source_fps:g} -> {target_fps:g} FPS requires {mult}x. "
            "Use LDF-VFI for ratios up to 16x or interpolate in multiple stages."
        )
    return num_passes, mult


def _select_target_fps_frames(frames, source_fps, target_fps, mult, num_input):
    """Pick frames from oversampled [M,C,H,W] tensor to hit target FPS timing.

    For downsampling (mult=1, ratio<=1), selects from original input frames.
    For upsampling, selects the nearest oversampled frame for each target timestamp.
    """
    duration = (num_input - 1) / source_fps
    num_output = int(math.floor(duration * target_fps)) + 1
    oversampled_fps = source_fps * mult
    max_idx = frames.shape[0] - 1
    indices = [min(round(j / target_fps * oversampled_fps), max_idx) for j in range(num_output)]
    return frames[indices]


# Google Drive file ID for the pretrained BIM-VFI model
GDRIVE_FILE_ID = "18Wre7XyRtu_wtFRzcsit6oNfHiFRt9vC"
MODEL_FILENAME = "bim_vfi.pth"

# Google Drive folder ID for EMA-VFI pretrained models
EMA_GDRIVE_FOLDER_ID = "16jUa3HkQ85Z5lb5gce1yoaWkP-rdCd0o"
EMA_DEFAULT_MODEL = "ours_t.pkl"

# Register model folders with ComfyUI
MODEL_DIR = os.path.join(folder_paths.models_dir, "bim-vfi")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

EMA_MODEL_DIR = os.path.join(folder_paths.models_dir, "ema-vfi")
if not os.path.exists(EMA_MODEL_DIR):
    os.makedirs(EMA_MODEL_DIR, exist_ok=True)

# Google Drive folder ID for SGM-VFI pretrained models
SGM_GDRIVE_FOLDER_ID = "1S5O6W0a7XQDHgBtP9HnmoxYEzWBIzSJq"
SGM_DEFAULT_MODEL = "ours-1-2-points.pkl"

SGM_MODEL_DIR = os.path.join(folder_paths.models_dir, "sgm-vfi")
if not os.path.exists(SGM_MODEL_DIR):
    os.makedirs(SGM_MODEL_DIR, exist_ok=True)

# GIMM-VFI
GIMM_HF_REPO = "Kijai/GIMM-VFI_safetensors"
GIMM_AVAILABLE_MODELS = [
    "gimmvfi_r_arb_lpips_fp32.safetensors",
    "gimmvfi_f_arb_lpips_fp32.safetensors",
]

GIMM_MODEL_DIR = os.path.join(folder_paths.models_dir, "gimm-vfi")
if not os.path.exists(GIMM_MODEL_DIR):
    os.makedirs(GIMM_MODEL_DIR, exist_ok=True)

# SPEED
SPEED_HF_REPO = "zhZ524/SPEED"
SPEED_DEFAULT_MODEL = "speed.pt"
SPEED_MODEL_DIR = os.path.join(folder_paths.models_dir, "speed-vfi")
if not os.path.exists(SPEED_MODEL_DIR):
    os.makedirs(SPEED_MODEL_DIR, exist_ok=True)

# LDF-VFI
LDF_HF_REPO = "onecat-ai/LDF-VFI"
LDF_MODEL_FILES = (
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "Wan2.1_VAE_cond_v2.pth",
)
LDF_MODEL_DIR = os.path.join(folder_paths.models_dir, "ldf-vfi")
if not os.path.exists(LDF_MODEL_DIR):
    os.makedirs(LDF_MODEL_DIR, exist_ok=True)


def get_available_models():
    """List available checkpoint files in the bim-vfi model directory."""
    models = []
    if os.path.isdir(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            if f.endswith((".pth", ".pt", ".ckpt", ".safetensors")):
                models.append(f)
    if not models:
        models.append(MODEL_FILENAME)  # Will trigger auto-download
    return sorted(models)


def download_model_from_gdrive(file_id, dest_path):
    """Download a file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown is required to auto-download the BIM-VFI model. "
            "Install it with: pip install gdown"
        )
    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info(f"Downloading BIM-VFI model to {dest_path}...")
    gdown.download(url, dest_path, quiet=False)
    if not os.path.exists(dest_path):
        raise RuntimeError(f"Failed to download model to {dest_path}")
    logger.info("Download complete.")


class LoadBIMVFIModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (get_available_models(), {
                    "default": MODEL_FILENAME,
                    "tooltip": "Checkpoint file from models/bim-vfi/. Auto-downloads on first use if missing.",
                }),
                "auto_pyr_level": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use the official pyramid policy: below 1080p=5, 1080p=6, 4K=7. Disable to use manual pyr_level.",
                }),
                "pyr_level": ("INT", {
                    "default": 3, "min": 3, "max": 7, "step": 1,
                    "tooltip": "Manual pyramid levels for coarse-to-fine processing. Only used when auto_pyr_level is disabled. More levels = captures larger motion but slower.",
                }),
            },
            "optional": {
                "artifact_safe_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Disable BIM-VFI's RGB refinement residual. This can remove wrong-edge/halo artifacts when optical flow is misaligned by blur or large motion, but may reduce detail on easy shots.",
                }),
            },
        }

    RETURN_TYPES = ("BIM_VFI_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "video/BIM-VFI"

    def load_model(self, model_path, auto_pyr_level, pyr_level,
                   artifact_safe_mode=False):
        full_path = os.path.join(MODEL_DIR, model_path)

        if not os.path.exists(full_path):
            logger.info(f"Model not found at {full_path}, attempting download...")
            download_model_from_gdrive(GDRIVE_FILE_ID, full_path)

        wrapper = BiMVFIModel(
            checkpoint_path=full_path,
            pyr_level=pyr_level,
            auto_pyr_level=auto_pyr_level,
            artifact_safe_mode=artifact_safe_mode,
            device="cpu",
        )

        mode = "auto" if auto_pyr_level else f"manual ({pyr_level})"
        synthesis = "artifact-safe" if artifact_safe_mode else "official"
        logger.info(
            f"BIM-VFI model loaded (pyr_level={mode}, synthesis={synthesis})"
        )
        return (wrapper,)


class BIMVFIInterpolate:
    MODEL_LABEL = "BIM-VFI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input image batch. Output frame count: 2x=(2N-1), 4x=(4N-3), 8x=(8N-7).",
                }),
                "model": ("BIM_VFI_MODEL", {
                    "tooltip": "BIM-VFI model from the Load BIM-VFI Model node.",
                }),
                "multiplier": ([2, 4, 8], {
                    "default": 2,
                    "tooltip": "Frame rate multiplier. 2x=one interpolation pass, 4x=two recursive passes, 8x=three. Higher = more frames but longer processing.",
                }),
                "clear_cache_after_n_frames": ("INT", {
                    "default": 10, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Clear CUDA cache every N frame pairs to prevent VRAM buildup. Lower = less VRAM but slower.",
                }),
                "keep_device": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model on GPU between frame pairs. Faster but uses ~200MB VRAM constantly. Disable to free VRAM between pairs (slower due to CPU-GPU transfers).",
                }),
                "all_on_gpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Store all intermediate frames on GPU instead of CPU. Much faster (no transfers) but requires enough VRAM for all frames. Recommended for 48GB+ cards.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Number of frame pairs to process simultaneously. Higher = faster but uses more VRAM. Start with 1, increase until VRAM is full. Recommended: 1 for 8GB, 2-4 for 24GB, 4-16 for 48GB+.",
                }),
                "chunk_size": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Process input frames in chunks of this size (0=disabled). Bounds VRAM usage during processing but the full output is still assembled in RAM. To bound RAM, use the Segment Interpolate node instead. Result is identical to processing all at once.",
                }),
                "source_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Input frame rate. Required when target_fps > 0.",
                }),
                "target_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Target output FPS. When > 0, overrides multiplier and auto-computes a power-of-2 oversample up to 8x, then selects frames. 0 = use multiplier.",
                }),
            },
            "optional": {
                "settings": ("VFI_SETTINGS", {
                    "tooltip": "Auto-tuned settings from VFI Optimizer. Overrides batch_size, "
                               "chunk_size, keep_device, all_on_gpu, clear_cache_after_n_frames.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "oversampled", "elapsed_seconds")
    FUNCTION = "interpolate"
    CATEGORY = "video/BIM-VFI"

    def _interpolate_frames(self, frames, model, num_passes, batch_size,
                            device, storage_device, keep_device, all_on_gpu,
                            clear_cache_after_n_frames, pbar, step_ref):
        """Run all interpolation passes on a chunk of frames.

        Args:
            frames: [N, C, H, W] tensor on storage_device
            step_ref: list with single int, mutable counter for progress bar
        Returns:
            Interpolated frames as [M, C, H, W] tensor on storage_device
        """
        for pass_idx in range(num_passes):
            logger.info(f"{self.MODEL_LABEL}: pass {pass_idx + 1}/{num_passes}, {frames.shape[0]} -> {2 * frames.shape[0] - 1} frames")
            new_frames = []
            num_pairs = frames.shape[0] - 1
            pairs_since_clear = 0

            for i in range(0, num_pairs, batch_size):
                _throw_if_interrupted()
                batch_end = min(i + batch_size, num_pairs)
                actual_batch = batch_end - i

                frames0 = frames[i:batch_end]
                frames1 = frames[i + 1:batch_end + 1]

                mids = _interpolate_batch_with_offload(
                    model, frames0, frames1, device, keep_device
                )
                mids = mids.to(storage_device)

                for j in range(actual_batch):
                    new_frames.append(frames[i + j:i + j + 1])
                    new_frames.append(mids[j:j+1])

                step_ref[0] += actual_batch
                pbar.update_absolute(step_ref[0])

                pairs_since_clear += actual_batch
                if pairs_since_clear >= clear_cache_after_n_frames:
                    _clear_model_cache(model)
                    pairs_since_clear = 0

            new_frames.append(frames[-1:])
            frames = torch.cat(new_frames, dim=0)

            _clear_model_cache(model)

        return frames

    @staticmethod
    def _count_steps(num_frames, num_passes):
        """Count total interpolation steps for a given input frame count."""
        n = num_frames
        total = 0
        for _ in range(num_passes):
            total += n - 1
            n = 2 * n - 1
        return total

    @_with_elapsed_seconds()
    def interpolate(self, images, model, multiplier, clear_cache_after_n_frames,
                    keep_device, all_on_gpu, batch_size, chunk_size,
                    source_fps=0.0, target_fps=0.0, settings=None, seed=None):
        batch_size, chunk_size, keep_device, all_on_gpu, clear_cache_after_n_frames = \
            _apply_vfi_settings(settings, batch_size, chunk_size, keep_device,
                                all_on_gpu, clear_cache_after_n_frames)

        if seed is not None and hasattr(model, "set_seed"):
            model.set_seed(seed)
        if hasattr(model, "reset_seed"):
            model.reset_seed()

        if images.shape[0] < 2:
            return (images, images)

        # Target FPS mode: auto-compute multiplier from fps ratio
        use_target_fps = _target_fps_enabled(source_fps, target_fps)
        if use_target_fps:
            num_passes, mult = _compute_target_fps_params(source_fps, target_fps)
            if num_passes == 0:
                # Downsampling or same fps — select from input directly
                all_frames = images.permute(0, 3, 1, 2)
                result = _select_target_fps_frames(all_frames, source_fps, target_fps, mult, all_frames.shape[0])
                return (result.cpu().permute(0, 2, 3, 1), images)
        else:
            num_passes = {2: 1, 4: 2, 8: 3}[multiplier]
            mult = multiplier

        N = images.shape[0]
        expected = mult * (N - 1) + 1
        if use_target_fps:
            if num_passes == 0:
                expected = int(math.floor((N - 1) / source_fps * target_fps)) + 1
                logger.info(f"{self.MODEL_LABEL}: {N} frames, {source_fps}fps -> {target_fps}fps (downsampling), expected output: {expected} frames")
            else:
                expected_target = int(math.floor((N - 1) / source_fps * target_fps)) + 1
                logger.info(f"{self.MODEL_LABEL}: interpolating {N} frames, {source_fps}fps -> {target_fps}fps (oversample {mult}x, {num_passes} pass(es)), expected output: {expected_target} frames")
        else:
            logger.info(f"{self.MODEL_LABEL}: interpolating {N} frames, {mult}x ({num_passes} pass(es)), expected output: {expected} frames")

        device = _get_torch_device()

        if all_on_gpu:
            keep_device = True

        storage_device = device if all_on_gpu else torch.device("cpu")

        # Convert from ComfyUI [B, H, W, C] to model [B, C, H, W]
        all_frames = images.permute(0, 3, 1, 2).to(storage_device)
        total_input = all_frames.shape[0]

        # Build chunk boundaries (1-frame overlap between consecutive chunks)
        if chunk_size < 2 or chunk_size >= total_input:
            chunks = [(0, total_input)]
        else:
            chunks = []
            start = 0
            while start < total_input - 1:
                end = min(start + chunk_size, total_input)
                chunks.append((start, end))
                start = end - 1  # overlap by 1 frame
                if end == total_input:
                    break

        if len(chunks) > 1:
            logger.info(f"{self.MODEL_LABEL}: processing in {len(chunks)} chunk(s)")

        # Calculate total progress steps across all chunks
        total_steps = sum(self._count_steps(ce - cs, num_passes) for cs, ce in chunks)
        pbar = ProgressBar(total_steps)
        step_ref = [0]

        if keep_device:
            model.to(device)

        result_chunks = []
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_frames = all_frames[chunk_start:chunk_end].clone()

            chunk_result = self._interpolate_frames(
                chunk_frames, model, num_passes, batch_size,
                device, storage_device, keep_device, all_on_gpu,
                clear_cache_after_n_frames, pbar, step_ref,
            )

            # Skip first frame of subsequent chunks (duplicate of previous chunk's last frame)
            if chunk_idx > 0:
                chunk_result = chunk_result[1:]

            # Move completed chunk to CPU to bound memory when chunking
            if len(chunks) > 1:
                chunk_result = chunk_result.cpu()

            result_chunks.append(chunk_result)

        result = torch.cat(result_chunks, dim=0)

        # Convert oversampled to ComfyUI format for second output
        oversampled = result.cpu().permute(0, 2, 3, 1)

        # Target FPS: select frames from oversampled result
        if use_target_fps:
            result = _select_target_fps_frames(result, source_fps, target_fps, mult, total_input)

        # Convert back to ComfyUI [B, H, W, C], on CPU
        result = result.cpu().permute(0, 2, 3, 1)
        logger.info(f"{self.MODEL_LABEL}: done, {result.shape[0]} output frames")
        return (result, oversampled)


class BIMVFISegmentInterpolate(BIMVFIInterpolate):
    """Process a numbered segment of the input batch.

    Chain multiple instances with Save nodes between them to bound peak RAM.
    The model pass-through output forces sequential execution so each segment
    saves and frees from RAM before the next starts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = BIMVFIInterpolate.INPUT_TYPES()
        base["required"]["segment_index"] = ("INT", {
            "default": 0, "min": 0, "max": 10000, "step": 1,
            "tooltip": "Which segment to process (0-based). Bounds RAM by only producing this segment's output frames, "
                       "unlike chunk_size which bounds VRAM but still assembles the full output in RAM. "
                       "Chain the model output to the next Segment Interpolate to force sequential execution.",
        })
        base["required"]["segment_size"] = ("INT", {
            "default": 500, "min": 2, "max": 10000, "step": 1,
            "tooltip": "Number of input frames per segment. Adjacent segments overlap by 1 frame for seamless stitching. "
                       "Smaller = less peak RAM per segment. Save each segment's output to disk before the next runs.",
        })
        return base

    RETURN_TYPES = ("IMAGE", "BIM_VFI_MODEL", "FLOAT")
    RETURN_NAMES = ("images", "model", "elapsed_seconds")
    FUNCTION = "interpolate"
    CATEGORY = "video/BIM-VFI"

    @_with_elapsed_seconds()
    def interpolate(self, images, model, multiplier, clear_cache_after_n_frames,
                    keep_device, all_on_gpu, batch_size, chunk_size,
                    segment_index, segment_size,
                    source_fps=0.0, target_fps=0.0, settings=None, seed=None):
        batch_size, chunk_size, keep_device, all_on_gpu, clear_cache_after_n_frames = \
            _apply_vfi_settings(settings, batch_size, chunk_size, keep_device,
                                all_on_gpu, clear_cache_after_n_frames)

        if seed is not None and hasattr(model, "set_seed"):
            model.set_seed(seed)
        if hasattr(model, "reset_seed"):
            model.reset_seed()

        total_input = images.shape[0]
        use_target_fps = _target_fps_enabled(source_fps, target_fps)

        # Compute segment boundaries (1-frame overlap)
        start = segment_index * (segment_size - 1)
        end = min(start + segment_size, total_input)

        if start >= total_input - 1:
            # Past the end — return empty single frame + model
            return (images[:1], model)

        segment_images = images[start:end]
        logger.info(f"{self.MODEL_LABEL} segment {segment_index}: input frames [{start}:{end}] of {total_input}")

        if use_target_fps:
            num_passes, mult = _compute_target_fps_params(source_fps, target_fps)

            # Compute global output frame range for this segment
            seg_start_time = start / source_fps
            seg_end_time = (end - 1) / source_fps
            duration = (total_input - 1) / source_fps
            total_output = int(math.floor(duration * target_fps)) + 1

            if segment_index == 0:
                j_start = 0
            else:
                j_start = int(math.floor(seg_start_time * target_fps)) + 1
            j_end = min(int(math.floor(seg_end_time * target_fps)), total_output - 1)

            if j_start > j_end:
                raise ValueError(
                    "This segment contains no frames at the requested target FPS. "
                    "Increase segment_size or use the non-segment Interpolate node."
                )

            logger.info(f"{self.MODEL_LABEL} segment {segment_index}: target fps output j=[{j_start}..{j_end}]")

            if num_passes == 0:
                # Downsampling — select from segment input directly
                oversampled_fps = source_fps * mult
                all_seg = segment_images.permute(0, 3, 1, 2)
                out_frames = []
                for j in range(j_start, j_end + 1):
                    global_idx = min(round(j / target_fps * oversampled_fps), total_input - 1)
                    local_idx = global_idx - start
                    local_idx = max(0, min(local_idx, all_seg.shape[0] - 1))
                    out_frames.append(all_seg[local_idx:local_idx + 1])
                result = torch.cat(out_frames, dim=0).cpu().permute(0, 2, 3, 1)
                return (result, model)

            # Oversample segment using computed num_passes
            device = _get_torch_device()
            if all_on_gpu:
                keep_device = True
            storage_device = device if all_on_gpu else torch.device("cpu")
            seg_frames = segment_images.permute(0, 3, 1, 2).to(storage_device)

            total_steps = self._count_steps(seg_frames.shape[0], num_passes)
            pbar = ProgressBar(total_steps)
            step_ref = [0]
            if keep_device:
                model.to(device)

            oversampled = self._interpolate_frames(
                seg_frames, model, num_passes, batch_size,
                device, storage_device, keep_device, all_on_gpu,
                clear_cache_after_n_frames, pbar, step_ref,
            )
            oversampled_fps = source_fps * mult

            out_frames = []
            for j in range(j_start, j_end + 1):
                global_oversamp_idx = round(j / target_fps * oversampled_fps)
                local_idx = global_oversamp_idx - start * mult
                local_idx = max(0, min(local_idx, oversampled.shape[0] - 1))
                out_frames.append(oversampled[local_idx:local_idx + 1])
            result = torch.cat(out_frames, dim=0).cpu().permute(0, 2, 3, 1)
            return (result, model)

        # Standard multiplier mode
        is_continuation = segment_index > 0
        (result, _, _) = super().interpolate(
            segment_images, model, multiplier, clear_cache_after_n_frames,
            keep_device, all_on_gpu, batch_size, chunk_size,
            seed=seed,
        )

        if is_continuation:
            result = result[1:]  # skip duplicate boundary frame

        return (result, model)


class TweenConcatVideos:
    """Concatenate segment video files into a single video using ffmpeg.

    Connect the model output from the last Segment Interpolate node to ensure
    this runs only after all segments have been saved to disk.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("*", {
                    "tooltip": "Connect from the last Segment Interpolate's model output (any model type). "
                               "This ensures concatenation runs only after all segments are saved.",
                }),
                "output_directory": ("STRING", {
                    "default": "",
                    "tooltip": "Directory containing the segment video files. "
                               "Leave empty to use ComfyUI's default output directory. "
                               "Relative paths are resolved from the output directory.",
                }),
                "filename_prefix": ("STRING", {
                    "default": "segment",
                    "tooltip": "Filename prefix used when saving segments with VHS Video Combine. "
                               "Matches files like segment_00001.mp4, segment_00002.mp4, etc.",
                }),
                "output_filename": ("STRING", {
                    "default": "final_video.mp4",
                    "tooltip": "Name of the concatenated output file. Saved in the same directory.",
                }),
                "delete_segments": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Delete the individual segment files after successful concatenation. "
                               "Useful to avoid leftover files that would pollute the next run.",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "concat"
    CATEGORY = "video/Tween"

    @staticmethod
    def _find_ffmpeg():
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            try:
                from imageio_ffmpeg import get_ffmpeg_exe
                ffmpeg_path = get_ffmpeg_exe()
            except ImportError:
                pass
        if ffmpeg_path is None:
            raise RuntimeError(
                "ffmpeg not found. Install ffmpeg or pip install imageio-ffmpeg."
            )
        return ffmpeg_path

    def concat(self, model, output_directory, filename_prefix, output_filename, delete_segments):
        # Resolve output directory — empty or relative paths are relative to ComfyUI output
        comfy_output = folder_paths.get_output_directory()
        out_dir = output_directory.strip()
        if not out_dir:
            out_dir = comfy_output
        elif not os.path.isabs(out_dir):
            out_dir = os.path.join(comfy_output, out_dir)

        if not os.path.isdir(out_dir):
            raise ValueError(f"Output directory does not exist: {out_dir}")
        if (
            not output_filename
            or os.path.basename(output_filename) != output_filename
            or any(char in output_filename for char in "\r\n")
        ):
            raise ValueError("output_filename must be a plain filename without newlines")
        output_path = os.path.abspath(os.path.join(out_dir, output_filename))

        # Find segment files matching the prefix
        safe_prefix = glob.escape(filename_prefix)
        segments = []
        for ext in ("mp4", "webm", "mkv"):
            segments.extend(
                glob.glob(os.path.join(out_dir, f"{safe_prefix}_*.{ext}"))
            )
        segments = [
            segment for segment in segments
            if os.path.abspath(segment) != output_path
        ]
        if any("\n" in segment or "\r" in segment for segment in segments):
            raise ValueError("Segment filenames cannot contain newline characters")
        segments.sort(key=lambda path: [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", os.path.basename(path))
        ])

        if not segments:
            raise FileNotFoundError(
                f"No segment files found matching '{filename_prefix}_*' "
                f"in {out_dir}"
            )

        logger.info(f"Found {len(segments)} segment(s) to concatenate")

        # Write ffmpeg concat list to a temp file
        fd, concat_list_path = tempfile.mkstemp(suffix=".txt", prefix="bimvfi_concat_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("ffconcat version 1.0\n")
                for seg in segments:
                    # ffconcat escaping: \ -> \\, ' -> \'
                    escaped = os.path.abspath(seg).replace("\\", "\\\\").replace("'", "\\'")
                    f.write(f"file '{escaped}'\n")

            ffmpeg = self._find_ffmpeg()

            cmd = [
                ffmpeg,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",
                output_path,
            ]

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg concat failed (exit {result.returncode}):\n"
                    f"{result.stderr}"
                )

            logger.info(f"Concatenated video saved to {output_path}")

            if delete_segments:
                for seg in segments:
                    try:
                        os.remove(seg)
                    except OSError as e:
                        logger.warning(f"Failed to delete segment {seg}: {e}")
                logger.info(f"Deleted {len(segments)} segment file(s)")
        finally:
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)

        return {"result": (output_path,)}


class VFIOptimizer:
    """Benchmark the user's GPU with the actual model and resolution to compute
    optimal batch_size, chunk_size, keep_device, all_on_gpu, and
    clear_cache_after_n_frames.  Outputs a VFI_SETTINGS dict that can be
    connected to any Interpolate or Segment Interpolate node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input images — only the first 2 frames are used for calibration.",
                }),
                "model": ("*", {
                    "tooltip": "Any pairwise VFI model (BIM, EMA, SGM, GIMM, SPEED). LDF-VFI uses a separate sequence pipeline.",
                }),
                "min_free_vram_gb": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 48.0, "step": 0.5,
                    "tooltip": "VRAM to keep free for other tasks (ComfyUI, OS, etc). "
                               "Higher = safer but slower.",
                }),
            },
            "optional": {
                "force_batch_size": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Override auto-computed batch_size. 0 = auto.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "VFI_SETTINGS")
    RETURN_NAMES = ("images", "settings")
    FUNCTION = "optimize"
    CATEGORY = "video/Tween"

    @staticmethod
    def _conservative_defaults(images):
        """Return safe fallback settings with image passthrough."""
        return (images, {
            "batch_size": 1,
            "chunk_size": 0,
            "keep_device": True,
            "all_on_gpu": False,
            "clear_cache_after_n_frames": 5,
            "_info": {"source": "conservative_defaults"},
        })

    def optimize(self, images, model, min_free_vram_gb, force_batch_size=0):
        if isinstance(model, LDFVFIModel):
            logger.info("VFI Optimizer: LDF-VFI is sequence-native; returning conservative defaults")
            return self._conservative_defaults(images)
        device = _get_torch_device()
        if images.shape[0] < 2 or device.type != "cuda":
            logger.info("VFI Optimizer: <2 frames or selected device is not CUDA; returning conservative defaults")
            return self._conservative_defaults(images)

        # --- Static analysis: model VRAM ---
        model_params = getattr(model, "model", model)
        if hasattr(model_params, "parameters"):
            model_vram_bytes = sum(
                p.nelement() * p.element_size() for p in model_params.parameters()
            )
        else:
            model_vram_bytes = 0
        model_vram_mb = model_vram_bytes / (1024 ** 2)

        # --- Calibration: run 1 frame pair ---
        frame0 = images[0:1].permute(0, 3, 1, 2)  # [1, C, H, W]
        frame1 = images[1:2].permute(0, 3, 1, 2)

        try:
            _throw_if_interrupted()
            model.to(device)
            torch.cuda.reset_peak_memory_stats(device)
            mem_before = torch.cuda.memory_allocated(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                model.interpolate_batch(frame0, frame1, time_step=0.5)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            peak_mem = torch.cuda.max_memory_allocated(device)
            per_pair_vram_bytes = peak_mem - mem_before
        except Exception as e:
            interrupt_type = getattr(model_management, "InterruptProcessingException", ())
            if interrupt_type and isinstance(e, interrupt_type):
                raise
            logger.warning(f"VFI Optimizer: calibration failed ({e}), returning conservative defaults")
            return self._conservative_defaults(images)
        finally:
            try:
                model.to("cpu")
            except Exception:
                logger.exception("VFI Optimizer: model offload failed")
            try:
                _clear_model_cache(model)
            except Exception:
                logger.exception("VFI Optimizer: cache cleanup failed")

        per_pair_vram_mb = max(per_pair_vram_bytes / (1024 ** 2), 1.0)

        # --- Compute settings ---
        total_vram_mb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        min_free_mb = min_free_vram_gb * 1024
        available_mb = total_vram_mb - min_free_mb - model_vram_mb

        # batch_size
        if force_batch_size > 0:
            batch_size = force_batch_size
        else:
            batch_size = max(1, min(int(available_mb * 0.85 / per_pair_vram_mb), 64))

        # all_on_gpu: estimate output frames for 2x from actual input count
        H, W = images.shape[1], images.shape[2]
        N = images.shape[0]
        frame_mb = H * W * 3 * 4 / (1024 ** 2)  # float32 [C,H,W]
        estimated_output_frames = 2 * N - 1  # 2x is most common multiplier
        output_vram_mb = estimated_output_frames * frame_mb
        vram_after_model = total_vram_mb - min_free_mb - model_vram_mb
        all_on_gpu = output_vram_mb < vram_after_model * 0.5

        # keep_device: True unless VRAM is extremely tight
        keep_device = available_mb > model_vram_mb * 0.5

        # clear_cache_after_n_frames: scale with headroom ratio
        headroom_ratio = available_mb / max(per_pair_vram_mb * batch_size, 1.0)
        clear_cache = max(3, min(int(headroom_ratio * 5), 20))

        # chunk_size: based on system RAM
        system_ram_gb = _get_system_ram_gb()
        system_ram_mb = system_ram_gb * 1024
        # Check if full 2x output fits in 60% of RAM
        if output_vram_mb < system_ram_mb * 0.6:
            chunk_size = 0  # fits in RAM, no chunking needed
        else:
            # How many input frames can we afford per chunk?
            # Each input frame produces ~2 output frames at 2x, each frame_mb
            frames_per_mb = 1.0 / max(frame_mb * 2, 0.001)
            chunk_size = max(4, int(system_ram_mb * 0.4 * frames_per_mb))

        settings = {
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "keep_device": keep_device,
            "all_on_gpu": all_on_gpu,
            "clear_cache_after_n_frames": clear_cache,
            "_info": {
                "source": "VFI Optimizer",
                "model_vram_mb": round(model_vram_mb, 1),
                "per_pair_vram_mb": round(per_pair_vram_mb, 1),
                "calibration_time_ms": round(elapsed * 1000, 1),
                "total_vram_mb": round(total_vram_mb, 1),
                "available_mb": round(available_mb, 1),
                "system_ram_gb": round(system_ram_gb, 1),
                "resolution": f"{W}x{H}",
            },
        }

        logger.info(
            f"VFI Optimizer: batch_size={batch_size}, chunk_size={chunk_size}, "
            f"keep_device={keep_device}, all_on_gpu={all_on_gpu}, "
            f"clear_cache={clear_cache} | "
            f"model={model_vram_mb:.0f}MB, per_pair={per_pair_vram_mb:.0f}MB, "
            f"available={available_mb:.0f}MB, "
            f"calibration={elapsed*1000:.0f}ms, res={W}x{H}"
        )

        return (images, settings)


# ---------------------------------------------------------------------------
# EMA-VFI nodes
# ---------------------------------------------------------------------------

def get_available_ema_models():
    """List available checkpoint files in the ema-vfi model directory."""
    models = []
    if os.path.isdir(EMA_MODEL_DIR):
        for f in os.listdir(EMA_MODEL_DIR):
            if f.endswith((".pkl", ".pth", ".pt", ".ckpt", ".safetensors")):
                models.append(f)
    if not models:
        models.append(EMA_DEFAULT_MODEL)  # Will trigger auto-download
    return sorted(models)


def download_ema_model_from_gdrive(folder_id, dest_path):
    """Download EMA-VFI model from Google Drive folder using gdown."""
    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown is required to auto-download the EMA-VFI model. "
            "Install it with: pip install gdown"
        )
    filename = os.path.basename(dest_path)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    logger.info(f"Downloading {filename} from Google Drive folder to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    gdown.download_folder(url, output=os.path.dirname(dest_path), quiet=False, remaining_ok=True)
    if not os.path.exists(dest_path):
        raise RuntimeError(
            f"Failed to download {filename}. Please download manually from "
            f"https://drive.google.com/drive/folders/{folder_id} "
            f"and place it in {os.path.dirname(dest_path)}"
        )
    logger.info("Download complete.")


class LoadEMAVFIModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (get_available_ema_models(), {
                    "default": EMA_DEFAULT_MODEL,
                    "tooltip": "Checkpoint file from models/ema-vfi/. Auto-downloads on first use if missing. "
                               "Variant (large/small) and timestep support are auto-detected from filename.",
                }),
                "tta": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Test-time augmentation: flip input and average with unflipped result. "
                               "~2x slower but slightly better quality. Recommended for large model only.",
                }),
            }
        }

    RETURN_TYPES = ("EMA_VFI_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "video/EMA-VFI"

    def load_model(self, model_path, tta):
        full_path = os.path.join(EMA_MODEL_DIR, model_path)

        if not os.path.exists(full_path):
            logger.info(f"Model not found at {full_path}, attempting download...")
            download_ema_model_from_gdrive(EMA_GDRIVE_FOLDER_ID, full_path)

        wrapper = EMAVFIModel(
            checkpoint_path=full_path,
            variant="auto",
            tta=tta,
            device="cpu",
        )

        t_mode = "arbitrary" if wrapper.supports_arbitrary_t else "fixed (0.5)"
        logger.info(f"EMA-VFI model loaded (variant={wrapper.variant_name}, timestep={t_mode}, tta={tta})")
        return (wrapper,)


class EMAVFIInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input image batch. Output frame count: 2x=(2N-1), 4x=(4N-3), 8x=(8N-7).",
                }),
                "model": ("EMA_VFI_MODEL", {
                    "tooltip": "EMA-VFI model from the Load EMA-VFI Model node.",
                }),
                "multiplier": ([2, 4, 8], {
                    "default": 2,
                    "tooltip": "Frame rate multiplier. 2x=one interpolation pass, 4x=two recursive passes, 8x=three. Higher = more frames but longer processing.",
                }),
                "clear_cache_after_n_frames": ("INT", {
                    "default": 10, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Clear CUDA cache every N frame pairs to prevent VRAM buildup. Lower = less VRAM but slower.",
                }),
                "keep_device": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model on GPU between frame pairs. Faster but uses more VRAM constantly. Disable to free VRAM between pairs (slower due to CPU-GPU transfers).",
                }),
                "all_on_gpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Store all intermediate frames on GPU instead of CPU. Much faster (no transfers) but requires enough VRAM for all frames. Recommended for 48GB+ cards.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Number of frame pairs to process simultaneously. Higher = faster but uses more VRAM. Start with 1, increase until VRAM is full.",
                }),
                "chunk_size": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Process input frames in chunks of this size (0=disabled). Bounds VRAM usage during processing but the full output is still assembled in RAM. To bound RAM, use the Segment Interpolate node instead.",
                }),
                "source_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Input frame rate. Required when target_fps > 0.",
                }),
                "target_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Target output FPS. When > 0, overrides multiplier and auto-computes a power-of-2 oversample up to 8x, then selects frames. 0 = use multiplier.",
                }),
            },
            "optional": {
                "settings": ("VFI_SETTINGS", {
                    "tooltip": "Auto-tuned settings from VFI Optimizer. Overrides batch_size, "
                               "chunk_size, keep_device, all_on_gpu, clear_cache_after_n_frames.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "oversampled", "elapsed_seconds")
    FUNCTION = "interpolate"
    CATEGORY = "video/EMA-VFI"

    def _interpolate_frames(self, frames, model, num_passes, batch_size,
                            device, storage_device, keep_device, all_on_gpu,
                            clear_cache_after_n_frames, pbar, step_ref):
        """Run all interpolation passes on a chunk of frames."""
        for pass_idx in range(num_passes):
            logger.info(f"EMA-VFI: pass {pass_idx + 1}/{num_passes}, {frames.shape[0]} -> {2 * frames.shape[0] - 1} frames")
            new_frames = []
            num_pairs = frames.shape[0] - 1
            pairs_since_clear = 0

            for i in range(0, num_pairs, batch_size):
                _throw_if_interrupted()
                batch_end = min(i + batch_size, num_pairs)
                actual_batch = batch_end - i

                frames0 = frames[i:batch_end]
                frames1 = frames[i + 1:batch_end + 1]

                mids = _interpolate_batch_with_offload(
                    model, frames0, frames1, device, keep_device
                )
                mids = mids.to(storage_device)

                for j in range(actual_batch):
                    new_frames.append(frames[i + j:i + j + 1])
                    new_frames.append(mids[j:j+1])

                step_ref[0] += actual_batch
                pbar.update_absolute(step_ref[0])

                pairs_since_clear += actual_batch
                if pairs_since_clear >= clear_cache_after_n_frames:
                    clear_ema_warp_cache()
                    _soft_empty_cache()
                    pairs_since_clear = 0

            new_frames.append(frames[-1:])
            frames = torch.cat(new_frames, dim=0)

            clear_ema_warp_cache()
            _soft_empty_cache()

        return frames

    @staticmethod
    def _count_steps(num_frames, num_passes):
        """Count total interpolation steps for a given input frame count."""
        n = num_frames
        total = 0
        for _ in range(num_passes):
            total += n - 1
            n = 2 * n - 1
        return total

    @_with_elapsed_seconds("EMA-VFI")
    def interpolate(self, images, model, multiplier, clear_cache_after_n_frames,
                    keep_device, all_on_gpu, batch_size, chunk_size,
                    source_fps=0.0, target_fps=0.0, settings=None):
        batch_size, chunk_size, keep_device, all_on_gpu, clear_cache_after_n_frames = \
            _apply_vfi_settings(settings, batch_size, chunk_size, keep_device,
                                all_on_gpu, clear_cache_after_n_frames)

        if images.shape[0] < 2:
            return (images, images)

        # Target FPS mode: auto-compute multiplier from fps ratio
        use_target_fps = _target_fps_enabled(source_fps, target_fps)
        if use_target_fps:
            num_passes, mult = _compute_target_fps_params(source_fps, target_fps)
            if num_passes == 0:
                all_frames = images.permute(0, 3, 1, 2)
                result = _select_target_fps_frames(all_frames, source_fps, target_fps, mult, all_frames.shape[0])
                return (result.cpu().permute(0, 2, 3, 1), images)
        else:
            num_passes = {2: 1, 4: 2, 8: 3}[multiplier]
            mult = multiplier

        N = images.shape[0]
        expected = mult * (N - 1) + 1
        if use_target_fps:
            if num_passes == 0:
                expected = int(math.floor((N - 1) / source_fps * target_fps)) + 1
                logger.info(f"EMA-VFI: {N} frames, {source_fps}fps -> {target_fps}fps (downsampling), expected output: {expected} frames")
            else:
                expected_target = int(math.floor((N - 1) / source_fps * target_fps)) + 1
                logger.info(f"EMA-VFI: interpolating {N} frames, {source_fps}fps -> {target_fps}fps (oversample {mult}x, {num_passes} pass(es)), expected output: {expected_target} frames")
        else:
            logger.info(f"EMA-VFI: interpolating {N} frames, {mult}x ({num_passes} pass(es)), expected output: {expected} frames")

        device = _get_torch_device()

        if all_on_gpu:
            keep_device = True

        storage_device = device if all_on_gpu else torch.device("cpu")

        # Convert from ComfyUI [B, H, W, C] to model [B, C, H, W]
        all_frames = images.permute(0, 3, 1, 2).to(storage_device)
        total_input = all_frames.shape[0]

        # Build chunk boundaries (1-frame overlap between consecutive chunks)
        if chunk_size < 2 or chunk_size >= total_input:
            chunks = [(0, total_input)]
        else:
            chunks = []
            start = 0
            while start < total_input - 1:
                end = min(start + chunk_size, total_input)
                chunks.append((start, end))
                start = end - 1  # overlap by 1 frame
                if end == total_input:
                    break

        if len(chunks) > 1:
            logger.info(f"EMA-VFI: processing in {len(chunks)} chunk(s)")

        # Calculate total progress steps across all chunks
        total_steps = sum(self._count_steps(ce - cs, num_passes) for cs, ce in chunks)
        pbar = ProgressBar(total_steps)
        step_ref = [0]

        if keep_device:
            model.to(device)

        result_chunks = []
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_frames = all_frames[chunk_start:chunk_end].clone()

            chunk_result = self._interpolate_frames(
                chunk_frames, model, num_passes, batch_size,
                device, storage_device, keep_device, all_on_gpu,
                clear_cache_after_n_frames, pbar, step_ref,
            )

            # Skip first frame of subsequent chunks (duplicate of previous chunk's last frame)
            if chunk_idx > 0:
                chunk_result = chunk_result[1:]

            # Move completed chunk to CPU to bound memory when chunking
            if len(chunks) > 1:
                chunk_result = chunk_result.cpu()

            result_chunks.append(chunk_result)

        result = torch.cat(result_chunks, dim=0)

        # Convert oversampled to ComfyUI format for second output
        oversampled = result.cpu().permute(0, 2, 3, 1)

        # Target FPS: select frames from oversampled result
        if use_target_fps:
            result = _select_target_fps_frames(result, source_fps, target_fps, mult, total_input)

        # Convert back to ComfyUI [B, H, W, C], on CPU
        result = result.cpu().permute(0, 2, 3, 1)
        logger.info(f"EMA-VFI: done, {result.shape[0]} output frames")
        return (result, oversampled)


class EMAVFISegmentInterpolate(EMAVFIInterpolate):
    """Process a numbered segment of the input batch for EMA-VFI.

    Chain multiple instances with Save nodes between them to bound peak RAM.
    The model pass-through output forces sequential execution so each segment
    saves and frees from RAM before the next starts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = EMAVFIInterpolate.INPUT_TYPES()
        base["required"]["segment_index"] = ("INT", {
            "default": 0, "min": 0, "max": 10000, "step": 1,
            "tooltip": "Which segment to process (0-based). Bounds RAM by only producing this segment's output frames, "
                       "unlike chunk_size which bounds VRAM but still assembles the full output in RAM. "
                       "Chain the model output to the next Segment Interpolate to force sequential execution.",
        })
        base["required"]["segment_size"] = ("INT", {
            "default": 500, "min": 2, "max": 10000, "step": 1,
            "tooltip": "Number of input frames per segment. Adjacent segments overlap by 1 frame for seamless stitching. "
                       "Smaller = less peak RAM per segment. Save each segment's output to disk before the next runs.",
        })
        return base

    RETURN_TYPES = ("IMAGE", "EMA_VFI_MODEL", "FLOAT")
    RETURN_NAMES = ("images", "model", "elapsed_seconds")
    FUNCTION = "interpolate"
    CATEGORY = "video/EMA-VFI"

    @_with_elapsed_seconds("EMA-VFI segment")
    def interpolate(self, images, model, multiplier, clear_cache_after_n_frames,
                    keep_device, all_on_gpu, batch_size, chunk_size,
                    segment_index, segment_size,
                    source_fps=0.0, target_fps=0.0, settings=None):
        batch_size, chunk_size, keep_device, all_on_gpu, clear_cache_after_n_frames = \
            _apply_vfi_settings(settings, batch_size, chunk_size, keep_device,
                                all_on_gpu, clear_cache_after_n_frames)

        total_input = images.shape[0]
        use_target_fps = _target_fps_enabled(source_fps, target_fps)

        # Compute segment boundaries (1-frame overlap)
        start = segment_index * (segment_size - 1)
        end = min(start + segment_size, total_input)

        if start >= total_input - 1:
            return (images[:1], model)

        segment_images = images[start:end]
        logger.info(f"EMA-VFI segment {segment_index}: input frames [{start}:{end}] of {total_input}")

        if use_target_fps:
            num_passes, mult = _compute_target_fps_params(source_fps, target_fps)

            seg_start_time = start / source_fps
            seg_end_time = (end - 1) / source_fps
            duration = (total_input - 1) / source_fps
            total_output = int(math.floor(duration * target_fps)) + 1

            if segment_index == 0:
                j_start = 0
            else:
                j_start = int(math.floor(seg_start_time * target_fps)) + 1
            j_end = min(int(math.floor(seg_end_time * target_fps)), total_output - 1)

            if j_start > j_end:
                raise ValueError(
                    "This segment contains no frames at the requested target FPS. "
                    "Increase segment_size or use the non-segment Interpolate node."
                )

            logger.info(f"EMA-VFI segment {segment_index}: target fps output j=[{j_start}..{j_end}]")

            if num_passes == 0:
                oversampled_fps = source_fps * mult
                all_seg = segment_images.permute(0, 3, 1, 2)
                out_frames = []
                for j in range(j_start, j_end + 1):
                    global_idx = min(round(j / target_fps * oversampled_fps), total_input - 1)
                    local_idx = global_idx - start
                    local_idx = max(0, min(local_idx, all_seg.shape[0] - 1))
                    out_frames.append(all_seg[local_idx:local_idx + 1])
                result = torch.cat(out_frames, dim=0).cpu().permute(0, 2, 3, 1)
                return (result, model)

            # Oversample segment using computed num_passes
            device = _get_torch_device()
            if all_on_gpu:
                keep_device = True
            storage_device = device if all_on_gpu else torch.device("cpu")
            seg_frames = segment_images.permute(0, 3, 1, 2).to(storage_device)

            total_steps = self._count_steps(seg_frames.shape[0], num_passes)
            pbar = ProgressBar(total_steps)
            step_ref = [0]
            if keep_device:
                model.to(device)

            oversampled = self._interpolate_frames(
                seg_frames, model, num_passes, batch_size,
                device, storage_device, keep_device, all_on_gpu,
                clear_cache_after_n_frames, pbar, step_ref,
            )
            oversampled_fps = source_fps * mult

            out_frames = []
            for j in range(j_start, j_end + 1):
                global_oversamp_idx = round(j / target_fps * oversampled_fps)
                local_idx = global_oversamp_idx - start * mult
                local_idx = max(0, min(local_idx, oversampled.shape[0] - 1))
                out_frames.append(oversampled[local_idx:local_idx + 1])
            result = torch.cat(out_frames, dim=0).cpu().permute(0, 2, 3, 1)
            return (result, model)

        # Standard multiplier mode
        is_continuation = segment_index > 0
        (result, _, _) = super().interpolate(
            segment_images, model, multiplier, clear_cache_after_n_frames,
            keep_device, all_on_gpu, batch_size, chunk_size,
        )

        if is_continuation:
            result = result[1:]

        return (result, model)


# ---------------------------------------------------------------------------
# SGM-VFI nodes
# ---------------------------------------------------------------------------

def get_available_sgm_models():
    """List available checkpoint files in the sgm-vfi model directory."""
    models = []
    if os.path.isdir(SGM_MODEL_DIR):
        for f in os.listdir(SGM_MODEL_DIR):
            if f.endswith((".pkl", ".pth", ".pt", ".ckpt", ".safetensors")):
                models.append(f)
    if not models:
        models.append(SGM_DEFAULT_MODEL)  # Will trigger auto-download
    return sorted(models)


def download_sgm_model_from_gdrive(folder_id, dest_path):
    """Download SGM-VFI model from Google Drive folder using gdown."""
    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown is required to auto-download the SGM-VFI model. "
            "Install it with: pip install gdown"
        )
    filename = os.path.basename(dest_path)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    logger.info(f"Downloading {filename} from Google Drive folder to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    gdown.download_folder(url, output=os.path.dirname(dest_path), quiet=False, remaining_ok=True)
    if not os.path.exists(dest_path):
        raise RuntimeError(
            f"Failed to download {filename}. Please download manually from "
            f"https://drive.google.com/drive/folders/{folder_id} "
            f"and place it in {os.path.dirname(dest_path)}"
        )
    logger.info("Download complete.")


class LoadSGMVFIModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (get_available_sgm_models(), {
                    "default": SGM_DEFAULT_MODEL,
                    "tooltip": "Checkpoint file from models/sgm-vfi/. Auto-downloads on first use if missing. "
                               "Variant (base/small) is auto-detected from filename.",
                }),
                "tta": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Test-time augmentation: flip input and average with unflipped result. "
                               "~2x slower but slightly better quality.",
                }),
                "num_key_points": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Sparsity of global matching. 0.0 = global matching everywhere (slower, better for large motion). "
                               "Higher = sparser keypoints (faster). Default 0.5 is a good balance.",
                }),
            }
        }

    RETURN_TYPES = ("SGM_VFI_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "video/SGM-VFI"

    def load_model(self, model_path, tta, num_key_points):
        full_path = os.path.join(SGM_MODEL_DIR, model_path)

        if not os.path.exists(full_path):
            logger.info(f"Model not found at {full_path}, attempting download...")
            download_sgm_model_from_gdrive(SGM_GDRIVE_FOLDER_ID, full_path)

        wrapper = SGMVFIModel(
            checkpoint_path=full_path,
            variant="auto",
            num_key_points=num_key_points,
            tta=tta,
            device="cpu",
        )

        logger.info(f"SGM-VFI model loaded (variant={wrapper.variant_name}, num_key_points={num_key_points}, tta={tta})")
        return (wrapper,)


class SGMVFIInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input image batch. Output frame count: 2x=(2N-1), 4x=(4N-3), 8x=(8N-7).",
                }),
                "model": ("SGM_VFI_MODEL", {
                    "tooltip": "SGM-VFI model from the Load SGM-VFI Model node.",
                }),
                "multiplier": ([2, 4, 8], {
                    "default": 2,
                    "tooltip": "Frame rate multiplier. 2x=one interpolation pass, 4x=two recursive passes, 8x=three. Higher = more frames but longer processing.",
                }),
                "clear_cache_after_n_frames": ("INT", {
                    "default": 10, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Clear CUDA cache every N frame pairs to prevent VRAM buildup. Lower = less VRAM but slower.",
                }),
                "keep_device": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model on GPU between frame pairs. Faster but uses more VRAM constantly. Disable to free VRAM between pairs (slower due to CPU-GPU transfers).",
                }),
                "all_on_gpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Store all intermediate frames on GPU instead of CPU. Much faster (no transfers) but requires enough VRAM for all frames. Recommended for 48GB+ cards.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Number of frame pairs to process simultaneously. Higher = faster but uses more VRAM. Start with 1, increase until VRAM is full.",
                }),
                "chunk_size": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Process input frames in chunks of this size (0=disabled). Bounds VRAM usage during processing but the full output is still assembled in RAM. To bound RAM, use the Segment Interpolate node instead.",
                }),
                "source_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Input frame rate. Required when target_fps > 0.",
                }),
                "target_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Target output FPS. When > 0, overrides multiplier and auto-computes a power-of-2 oversample up to 8x, then selects frames. 0 = use multiplier.",
                }),
            },
            "optional": {
                "settings": ("VFI_SETTINGS", {
                    "tooltip": "Auto-tuned settings from VFI Optimizer. Overrides batch_size, "
                               "chunk_size, keep_device, all_on_gpu, clear_cache_after_n_frames.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "oversampled", "elapsed_seconds")
    FUNCTION = "interpolate"
    CATEGORY = "video/SGM-VFI"

    def _interpolate_frames(self, frames, model, num_passes, batch_size,
                            device, storage_device, keep_device, all_on_gpu,
                            clear_cache_after_n_frames, pbar, step_ref):
        """Run all interpolation passes on a chunk of frames."""
        for pass_idx in range(num_passes):
            logger.info(f"SGM-VFI: pass {pass_idx + 1}/{num_passes}, {frames.shape[0]} -> {2 * frames.shape[0] - 1} frames")
            new_frames = []
            num_pairs = frames.shape[0] - 1
            pairs_since_clear = 0

            for i in range(0, num_pairs, batch_size):
                _throw_if_interrupted()
                batch_end = min(i + batch_size, num_pairs)
                actual_batch = batch_end - i

                frames0 = frames[i:batch_end]
                frames1 = frames[i + 1:batch_end + 1]

                mids = _interpolate_batch_with_offload(
                    model, frames0, frames1, device, keep_device
                )
                mids = mids.to(storage_device)

                for j in range(actual_batch):
                    new_frames.append(frames[i + j:i + j + 1])
                    new_frames.append(mids[j:j+1])

                step_ref[0] += actual_batch
                pbar.update_absolute(step_ref[0])

                pairs_since_clear += actual_batch
                if pairs_since_clear >= clear_cache_after_n_frames:
                    clear_sgm_warp_cache()
                    _soft_empty_cache()
                    pairs_since_clear = 0

            new_frames.append(frames[-1:])
            frames = torch.cat(new_frames, dim=0)

            clear_sgm_warp_cache()
            _soft_empty_cache()

        return frames

    @staticmethod
    def _count_steps(num_frames, num_passes):
        """Count total interpolation steps for a given input frame count."""
        n = num_frames
        total = 0
        for _ in range(num_passes):
            total += n - 1
            n = 2 * n - 1
        return total

    @_with_elapsed_seconds("SGM-VFI")
    def interpolate(self, images, model, multiplier, clear_cache_after_n_frames,
                    keep_device, all_on_gpu, batch_size, chunk_size,
                    source_fps=0.0, target_fps=0.0, settings=None):
        batch_size, chunk_size, keep_device, all_on_gpu, clear_cache_after_n_frames = \
            _apply_vfi_settings(settings, batch_size, chunk_size, keep_device,
                                all_on_gpu, clear_cache_after_n_frames)

        if images.shape[0] < 2:
            return (images, images)

        # Target FPS mode: auto-compute multiplier from fps ratio
        use_target_fps = _target_fps_enabled(source_fps, target_fps)
        if use_target_fps:
            num_passes, mult = _compute_target_fps_params(source_fps, target_fps)
            if num_passes == 0:
                all_frames = images.permute(0, 3, 1, 2)
                result = _select_target_fps_frames(all_frames, source_fps, target_fps, mult, all_frames.shape[0])
                return (result.cpu().permute(0, 2, 3, 1), images)
        else:
            num_passes = {2: 1, 4: 2, 8: 3}[multiplier]
            mult = multiplier

        N = images.shape[0]
        expected = mult * (N - 1) + 1
        if use_target_fps:
            if num_passes == 0:
                expected = int(math.floor((N - 1) / source_fps * target_fps)) + 1
                logger.info(f"SGM-VFI: {N} frames, {source_fps}fps -> {target_fps}fps (downsampling), expected output: {expected} frames")
            else:
                expected_target = int(math.floor((N - 1) / source_fps * target_fps)) + 1
                logger.info(f"SGM-VFI: interpolating {N} frames, {source_fps}fps -> {target_fps}fps (oversample {mult}x, {num_passes} pass(es)), expected output: {expected_target} frames")
        else:
            logger.info(f"SGM-VFI: interpolating {N} frames, {mult}x ({num_passes} pass(es)), expected output: {expected} frames")

        device = _get_torch_device()

        if all_on_gpu:
            keep_device = True

        storage_device = device if all_on_gpu else torch.device("cpu")

        # Convert from ComfyUI [B, H, W, C] to model [B, C, H, W]
        all_frames = images.permute(0, 3, 1, 2).to(storage_device)
        total_input = all_frames.shape[0]

        # Build chunk boundaries (1-frame overlap between consecutive chunks)
        if chunk_size < 2 or chunk_size >= total_input:
            chunks = [(0, total_input)]
        else:
            chunks = []
            start = 0
            while start < total_input - 1:
                end = min(start + chunk_size, total_input)
                chunks.append((start, end))
                start = end - 1  # overlap by 1 frame
                if end == total_input:
                    break

        if len(chunks) > 1:
            logger.info(f"SGM-VFI: processing in {len(chunks)} chunk(s)")

        # Calculate total progress steps across all chunks
        total_steps = sum(self._count_steps(ce - cs, num_passes) for cs, ce in chunks)
        pbar = ProgressBar(total_steps)
        step_ref = [0]

        if keep_device:
            model.to(device)

        result_chunks = []
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_frames = all_frames[chunk_start:chunk_end].clone()

            chunk_result = self._interpolate_frames(
                chunk_frames, model, num_passes, batch_size,
                device, storage_device, keep_device, all_on_gpu,
                clear_cache_after_n_frames, pbar, step_ref,
            )

            # Skip first frame of subsequent chunks (duplicate of previous chunk's last frame)
            if chunk_idx > 0:
                chunk_result = chunk_result[1:]

            # Move completed chunk to CPU to bound memory when chunking
            if len(chunks) > 1:
                chunk_result = chunk_result.cpu()

            result_chunks.append(chunk_result)

        result = torch.cat(result_chunks, dim=0)

        # Convert oversampled to ComfyUI format for second output
        oversampled = result.cpu().permute(0, 2, 3, 1)

        # Target FPS: select frames from oversampled result
        if use_target_fps:
            result = _select_target_fps_frames(result, source_fps, target_fps, mult, total_input)

        # Convert back to ComfyUI [B, H, W, C], on CPU
        result = result.cpu().permute(0, 2, 3, 1)
        logger.info(f"SGM-VFI: done, {result.shape[0]} output frames")
        return (result, oversampled)


class SGMVFISegmentInterpolate(SGMVFIInterpolate):
    """Process a numbered segment of the input batch for SGM-VFI.

    Chain multiple instances with Save nodes between them to bound peak RAM.
    The model pass-through output forces sequential execution so each segment
    saves and frees from RAM before the next starts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = SGMVFIInterpolate.INPUT_TYPES()
        base["required"]["segment_index"] = ("INT", {
            "default": 0, "min": 0, "max": 10000, "step": 1,
            "tooltip": "Which segment to process (0-based). Bounds RAM by only producing this segment's output frames, "
                       "unlike chunk_size which bounds VRAM but still assembles the full output in RAM. "
                       "Chain the model output to the next Segment Interpolate to force sequential execution.",
        })
        base["required"]["segment_size"] = ("INT", {
            "default": 500, "min": 2, "max": 10000, "step": 1,
            "tooltip": "Number of input frames per segment. Adjacent segments overlap by 1 frame for seamless stitching. "
                       "Smaller = less peak RAM per segment. Save each segment's output to disk before the next runs.",
        })
        return base

    RETURN_TYPES = ("IMAGE", "SGM_VFI_MODEL", "FLOAT")
    RETURN_NAMES = ("images", "model", "elapsed_seconds")
    FUNCTION = "interpolate"
    CATEGORY = "video/SGM-VFI"

    @_with_elapsed_seconds("SGM-VFI segment")
    def interpolate(self, images, model, multiplier, clear_cache_after_n_frames,
                    keep_device, all_on_gpu, batch_size, chunk_size,
                    segment_index, segment_size,
                    source_fps=0.0, target_fps=0.0, settings=None):
        batch_size, chunk_size, keep_device, all_on_gpu, clear_cache_after_n_frames = \
            _apply_vfi_settings(settings, batch_size, chunk_size, keep_device,
                                all_on_gpu, clear_cache_after_n_frames)

        total_input = images.shape[0]
        use_target_fps = _target_fps_enabled(source_fps, target_fps)

        # Compute segment boundaries (1-frame overlap)
        start = segment_index * (segment_size - 1)
        end = min(start + segment_size, total_input)

        if start >= total_input - 1:
            return (images[:1], model)

        segment_images = images[start:end]
        logger.info(f"SGM-VFI segment {segment_index}: input frames [{start}:{end}] of {total_input}")

        if use_target_fps:
            num_passes, mult = _compute_target_fps_params(source_fps, target_fps)

            seg_start_time = start / source_fps
            seg_end_time = (end - 1) / source_fps
            duration = (total_input - 1) / source_fps
            total_output = int(math.floor(duration * target_fps)) + 1

            if segment_index == 0:
                j_start = 0
            else:
                j_start = int(math.floor(seg_start_time * target_fps)) + 1
            j_end = min(int(math.floor(seg_end_time * target_fps)), total_output - 1)

            if j_start > j_end:
                raise ValueError(
                    "This segment contains no frames at the requested target FPS. "
                    "Increase segment_size or use the non-segment Interpolate node."
                )

            logger.info(f"SGM-VFI segment {segment_index}: target fps output j=[{j_start}..{j_end}]")

            if num_passes == 0:
                oversampled_fps = source_fps * mult
                all_seg = segment_images.permute(0, 3, 1, 2)
                out_frames = []
                for j in range(j_start, j_end + 1):
                    global_idx = min(round(j / target_fps * oversampled_fps), total_input - 1)
                    local_idx = global_idx - start
                    local_idx = max(0, min(local_idx, all_seg.shape[0] - 1))
                    out_frames.append(all_seg[local_idx:local_idx + 1])
                result = torch.cat(out_frames, dim=0).cpu().permute(0, 2, 3, 1)
                return (result, model)

            # Oversample segment using computed num_passes
            device = _get_torch_device()
            if all_on_gpu:
                keep_device = True
            storage_device = device if all_on_gpu else torch.device("cpu")
            seg_frames = segment_images.permute(0, 3, 1, 2).to(storage_device)

            total_steps = self._count_steps(seg_frames.shape[0], num_passes)
            pbar = ProgressBar(total_steps)
            step_ref = [0]
            if keep_device:
                model.to(device)

            oversampled = self._interpolate_frames(
                seg_frames, model, num_passes, batch_size,
                device, storage_device, keep_device, all_on_gpu,
                clear_cache_after_n_frames, pbar, step_ref,
            )
            oversampled_fps = source_fps * mult

            out_frames = []
            for j in range(j_start, j_end + 1):
                global_oversamp_idx = round(j / target_fps * oversampled_fps)
                local_idx = global_oversamp_idx - start * mult
                local_idx = max(0, min(local_idx, oversampled.shape[0] - 1))
                out_frames.append(oversampled[local_idx:local_idx + 1])
            result = torch.cat(out_frames, dim=0).cpu().permute(0, 2, 3, 1)
            return (result, model)

        # Standard multiplier mode
        is_continuation = segment_index > 0
        (result, _, _) = super().interpolate(
            segment_images, model, multiplier, clear_cache_after_n_frames,
            keep_device, all_on_gpu, batch_size, chunk_size,
        )

        if is_continuation:
            result = result[1:]

        return (result, model)


# ---------------------------------------------------------------------------
# LDF-VFI nodes
# ---------------------------------------------------------------------------

def ensure_ldf_model_files():
    missing = [
        filename for filename in LDF_MODEL_FILES
        if not os.path.isfile(os.path.join(LDF_MODEL_DIR, filename))
    ]
    if not missing:
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download LDF-VFI. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    logger.warning(
        "Downloading LDF-VFI from %s (~6.4 GB total). This only happens once.",
        LDF_HF_REPO,
    )
    for filename in missing:
        logger.info("Downloading LDF-VFI file: %s", filename)
        downloaded = hf_hub_download(
            repo_id=LDF_HF_REPO,
            filename=filename,
            local_dir=LDF_MODEL_DIR,
        )
        if not os.path.isfile(downloaded):
            raise RuntimeError(f"Failed to download LDF-VFI file: {filename}")


class LoadLDFVFIModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["onecat-ai/LDF-VFI"], {
                    "default": "onecat-ai/LDF-VFI",
                    "tooltip": "Official LDF-VFI transformer + conditional VAE. Downloads ~6.4 GB on first use and needs about 20 GB VRAM.",
                }),
                "tile_size": ("INT", {
                    "default": 256, "min": 128, "max": 1024, "step": 8,
                    "tooltip": "Spatial VAE tile size. Larger tiles can improve throughput but use more VRAM.",
                }),
                "tile_overlap": ("INT", {
                    "default": 64, "min": 8, "max": 512, "step": 8,
                    "tooltip": "Spatial overlap blended between VAE tiles. Must be smaller than tile_size.",
                }),
                "vae_batch_size": ("INT", {
                    "default": 8, "min": 1, "max": 32, "step": 1,
                    "tooltip": "VAE temporal-tile batch size. Lower this first if VAE encoding or decoding runs out of VRAM.",
                }),
                "attention_type": ([
                    "slide_chunk_all_block_2x1x1",
                    "slide_chunk_all_block",
                    "slide_chunk_all",
                    "full",
                ], {
                    "default": "slide_chunk_all_block_2x1x1",
                    "tooltip": "Official quick-start sparse attention is recommended. Full attention is extremely memory-intensive.",
                }),
            }
        }

    RETURN_TYPES = ("LDF_VFI_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "video/LDF-VFI"

    def load_model(self, model, tile_size, tile_overlap, vae_batch_size, attention_type):
        del model  # The combo documents the fixed official checkpoint.
        if tile_overlap >= tile_size:
            raise ValueError("LDF-VFI tile_overlap must be smaller than tile_size")
        source_root = ensure_upstream_source("ldf", LDF_MODEL_DIR)
        ensure_ldf_model_files()
        wrapper = LDFVFIModel(
            model_root=LDF_MODEL_DIR,
            vae_path=os.path.join(LDF_MODEL_DIR, "Wan2.1_VAE_cond_v2.pth"),
            source_root=source_root,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            vae_batch_size=vae_batch_size,
            attention_type=attention_type,
        )
        logger.info(
            "LDF-VFI loaded on CPU (tile=%s, overlap=%s, VAE batch=%s)",
            tile_size, tile_overlap, vae_batch_size,
        )
        return (wrapper,)


class LDFVFIInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Ordered source sequence. LDF models it holistically with internal skip-concat chunks.",
                }),
                "model": ("LDF_VFI_MODEL", {
                    "tooltip": "LDF-VFI model from the Load LDF-VFI Model node.",
                }),
                "temporal_factor": ("INT", {
                    "default": 8, "min": 2, "max": 16, "step": 1,
                    "tooltip": "Native temporal upsampling factor. LDF supports every integer from 2x through 16x.",
                }),
                "sampling_steps": ("INT", {
                    "default": 16, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Diffusion steps per temporal chunk. Official quick start uses 16; fewer is faster but may reduce quality.",
                }),
                "t_shift": ("FLOAT", {
                    "default": 8.0, "min": 0.1, "max": 20.0, "step": 0.1,
                    "tooltip": "Diffusion timestep shift. Official quick start uses 8.",
                }),
                "t_cond": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Noise applied to autoregressive boundary latents during skip-concat sampling.",
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0x7FFFFFFF, "step": 1,
                    "tooltip": "Seed for VAE sampling and diffusion noise.",
                }),
                "offload_after": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Move the ~6.4 GB model stack back to CPU after generation to release VRAM.",
                }),
                "source_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Input FPS. Set with target_fps to produce the requested output cadence.",
                }),
                "target_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Optional exact output FPS. Chooses the smallest native integer factor up to 16x, then selects the nearest generated frames.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "generated_sequence", "elapsed_seconds")
    FUNCTION = "interpolate"
    CATEGORY = "video/LDF-VFI"

    @_with_elapsed_seconds("LDF-VFI")
    def interpolate(self, images, model, temporal_factor, sampling_steps,
                    t_shift, t_cond, seed, offload_after,
                    source_fps=0.0, target_fps=0.0):
        if images.shape[0] < 2:
            return (images, images)
        device = _get_torch_device()
        if device.type != "cuda":
            raise RuntimeError(
                f"LDF-VFI requires an NVIDIA CUDA GPU with BF16 support; "
                f"ComfyUI selected {device}"
            )

        use_target_fps = _target_fps_enabled(source_fps, target_fps)
        if use_target_fps:
            ratio = target_fps / source_fps
            if ratio <= 1:
                source = images.permute(0, 3, 1, 2)
                selected = _select_target_fps_frames(
                    source, source_fps, target_fps, 1, source.shape[0]
                ).permute(0, 2, 3, 1).cpu()
                return (selected, images)
            temporal_factor = math.ceil(ratio)
            if temporal_factor > 16:
                raise ValueError(
                    f"LDF-VFI supports at most 16x, but {source_fps} -> {target_fps} FPS needs {temporal_factor}x"
                )

        source = images.permute(0, 3, 1, 2).contiguous()
        blocks = model.sampling_block_count(source.shape[0], temporal_factor)
        pbar = ProgressBar(blocks * sampling_steps)
        progress_step = [0]

        def update_progress():
            _throw_if_interrupted()
            progress_step[0] += 1
            pbar.update_absolute(progress_step[0])

        logger.info(
            "LDF-VFI: %s source frames, %sx, %s sampling steps, %s diffusion blocks",
            source.shape[0], temporal_factor, sampling_steps, blocks,
        )
        try:
            model.to(device)
            generated = model.interpolate_sequence(
                source,
                temporal_factor=temporal_factor,
                num_steps=sampling_steps,
                t_shift=t_shift,
                t_cond=t_cond,
                seed=seed,
                progress_callback=update_progress,
            )
        finally:
            generation_failed = sys.exc_info()[0] is not None
            cleanup_error = None
            if offload_after:
                try:
                    model.to("cpu")
                except Exception as exc:
                    cleanup_error = exc
                    logger.exception("Failed to offload LDF-VFI after generation")
            try:
                _clear_model_cache(model)
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                logger.exception("Failed to clear LDF-VFI caches")
            if cleanup_error is not None and not generation_failed:
                raise RuntimeError("LDF-VFI cleanup failed") from cleanup_error

        generated_sequence = generated.permute(0, 2, 3, 1).cpu()
        if use_target_fps:
            generated = _select_target_fps_frames(
                generated, source_fps, target_fps,
                temporal_factor, source.shape[0],
            )
        result = generated.permute(0, 2, 3, 1).cpu()
        logger.info("LDF-VFI: done, %s output frames", result.shape[0])
        return (result, generated_sequence)


# ---------------------------------------------------------------------------
# SPEED nodes
# ---------------------------------------------------------------------------

def get_available_speed_models():
    models = []
    if os.path.isdir(SPEED_MODEL_DIR):
        for filename in os.listdir(SPEED_MODEL_DIR):
            if filename.endswith((".pt", ".pth", ".ckpt")):
                models.append(filename)
    if not models:
        models.append(SPEED_DEFAULT_MODEL)
    return sorted(models)


def download_speed_model(filename, dest_dir):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to auto-download SPEED. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    logger.info("Downloading %s from Hugging Face (%s)...", filename, SPEED_HF_REPO)
    downloaded = hf_hub_download(
        repo_id=SPEED_HF_REPO,
        filename=filename,
        local_dir=dest_dir,
    )
    if not os.path.isfile(downloaded):
        raise RuntimeError(f"Failed to download SPEED checkpoint to {downloaded}")
    return downloaded


class LoadSPEEDVFIModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (get_available_speed_models(), {
                    "default": SPEED_DEFAULT_MODEL,
                    "tooltip": "Checkpoint in models/speed-vfi/. The official ~447 MB checkpoint and pinned runtime source download on first use.",
                }),
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Inference precision. Auto uses BF16 on supported CUDA GPUs, otherwise FP16. FP32 is slower and uses more VRAM.",
                }),
            }
        }

    RETURN_TYPES = ("SPEED_VFI_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "video/SPEED"

    def load_model(self, model_path, precision):
        source_root = ensure_upstream_source("speed", SPEED_MODEL_DIR)
        full_path = os.path.join(SPEED_MODEL_DIR, model_path)
        if not os.path.isfile(full_path):
            full_path = download_speed_model(model_path, SPEED_MODEL_DIR)

        wrapper = SpeedVFIModel(
            checkpoint_path=full_path,
            source_root=source_root,
            precision=precision,
            device="cpu",
        )
        logger.info("SPEED loaded (precision=%s)", precision)
        return (wrapper,)


class SPEEDVFIInterpolate(BIMVFIInterpolate):
    MODEL_LABEL = "SPEED"
    CATEGORY = "video/SPEED"
    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "oversampled", "elapsed_seconds")

    @classmethod
    def INPUT_TYPES(cls):
        inputs = BIMVFIInterpolate.INPUT_TYPES()
        inputs["required"]["model"] = ("SPEED_VFI_MODEL", {
            "tooltip": "SPEED model from the Load SPEED Model node.",
        })
        inputs["required"]["multiplier"] = ([2, 4, 8], {
            "default": 2,
            "tooltip": "SPEED is midpoint-only: 4x and 8x use recursive midpoint passes.",
        })
        inputs["required"]["seed"] = ("INT", {
            "default": 0, "min": 0, "max": 0x7FFFFFFF, "step": 1,
            "tooltip": "SPEED starts from random pixel noise. The seed makes the same execution settings repeatable without reloading the model.",
        })
        inputs["required"]["batch_size"][1]["tooltip"] += (
            " SPEED is stochastic, so changing batch size can change how seeded noise is assigned to pairs."
        )
        inputs["required"]["chunk_size"][1]["tooltip"] = (
            "Process the source in overlapping chunks to bound VRAM. SPEED remains repeatable for the same "
            "seed and settings, but changing chunk or batch boundaries can change its stochastic result."
        )
        inputs["required"]["keep_device"][1]["tooltip"] = (
            "Keep the ~235M-parameter model on GPU between batches. Faster, but uses substantial VRAM."
        )
        return inputs

class SPEEDVFISegmentInterpolate(BIMVFISegmentInterpolate):
    MODEL_LABEL = "SPEED"
    RETURN_TYPES = ("IMAGE", "SPEED_VFI_MODEL", "FLOAT")
    RETURN_NAMES = ("images", "model", "elapsed_seconds")
    CATEGORY = "video/SPEED"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = SPEEDVFIInterpolate.INPUT_TYPES()
        inputs["required"]["segment_index"] = ("INT", {
            "default": 0, "min": 0, "max": 10000, "step": 1,
            "tooltip": "Zero-based segment to process. Adjacent segments overlap by one source frame.",
        })
        inputs["required"]["segment_size"] = ("INT", {
            "default": 500, "min": 2, "max": 10000, "step": 1,
            "tooltip": "Input frames per segment. Save each segment before processing the next to bound RAM.",
        })
        return inputs


# ---------------------------------------------------------------------------
# GIMM-VFI nodes
# ---------------------------------------------------------------------------

def get_available_gimm_models():
    """List available GIMM-VFI checkpoint files in the gimm-vfi model directory."""
    models = []
    if os.path.isdir(GIMM_MODEL_DIR):
        for f in os.listdir(GIMM_MODEL_DIR):
            if f.endswith((".safetensors", ".pth", ".pt", ".ckpt")):
                # Exclude flow estimator checkpoints from the model list
                if f.startswith(("raft-", "flowformer_")):
                    continue
                models.append(f)
    if not models:
        models = list(GIMM_AVAILABLE_MODELS)
    return sorted(models)


def download_gimm_model(filename, dest_dir):
    """Download a GIMM-VFI file from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required to auto-download GIMM-VFI models. "
            "Install it with: pip install huggingface_hub"
        )
    logger.info(f"Downloading {filename} from HuggingFace ({GIMM_HF_REPO})...")
    hf_hub_download(
        repo_id=GIMM_HF_REPO,
        filename=filename,
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
    )
    dest_path = os.path.join(dest_dir, filename)
    if not os.path.exists(dest_path):
        raise RuntimeError(f"Failed to download {filename} to {dest_path}")
    logger.info(f"Downloaded {filename}")


class LoadGIMMVFIModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (get_available_gimm_models(), {
                    "default": GIMM_AVAILABLE_MODELS[0],
                    "tooltip": "Checkpoint file from models/gimm-vfi/. Auto-downloads from HuggingFace on first use. "
                               "RAFT variant (~80MB) or FlowFormer variant (~123MB) auto-detected from filename.",
                }),
                "ds_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.125, "max": 1.0, "step": 0.125,
                    "tooltip": "Downscale factor for internal processing. 1.0 = full resolution. "
                               "Lower values reduce VRAM usage and speed up inference at the cost of quality. "
                               "Try 0.5 for 4K inputs.",
                }),
            }
        }

    RETURN_TYPES = ("GIMM_VFI_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "video/GIMM-VFI"

    def load_model(self, model_path, ds_factor):
        full_path = os.path.join(GIMM_MODEL_DIR, model_path)

        # Auto-download main model if missing
        if not os.path.exists(full_path):
            logger.info(f"Model not found at {full_path}, attempting download...")
            download_gimm_model(model_path, GIMM_MODEL_DIR)

        # Detect and download matching flow estimator
        if "gimmvfi_f" in model_path.lower():
            flow_filename = "flowformer_sintel_fp32.safetensors"
        else:
            flow_filename = "raft-things_fp32.safetensors"

        flow_path = os.path.join(GIMM_MODEL_DIR, flow_filename)
        if not os.path.exists(flow_path):
            logger.info(f"Flow estimator not found, downloading {flow_filename}...")
            download_gimm_model(flow_filename, GIMM_MODEL_DIR)

        wrapper = GIMMVFIModel(
            checkpoint_path=full_path,
            flow_checkpoint_path=flow_path,
            variant="auto",
            ds_factor=ds_factor,
            device="cpu",
        )

        logger.info(f"GIMM-VFI model loaded (variant={wrapper.variant_name}, ds_factor={ds_factor})")
        return (wrapper,)


class GIMMVFIInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input image batch. Output frame count: 2x=(2N-1), 4x=(4N-3), 8x=(8N-7).",
                }),
                "model": ("GIMM_VFI_MODEL", {
                    "tooltip": "GIMM-VFI model from the Load GIMM-VFI Model node.",
                }),
                "multiplier": ([2, 4, 8], {
                    "default": 2,
                    "tooltip": "Frame rate multiplier. In single-pass mode, all intermediate frames are generated "
                               "in one forward pass per pair. In recursive mode, uses 2x passes like other models.",
                }),
                "single_pass": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GIMM-VFI's single-pass arbitrary-timestep mode. Generates all intermediate frames "
                               "per pair in one forward pass (no recursive 2x passes). Disable to use the standard "
                               "recursive approach (same as BIM/EMA/SGM).",
                }),
                "clear_cache_after_n_frames": ("INT", {
                    "default": 10, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Clear CUDA cache every N frame pairs to prevent VRAM buildup. Lower = less VRAM but slower.",
                }),
                "keep_device": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model on GPU between frame pairs. Faster but uses more VRAM constantly. Disable to free VRAM between pairs (slower due to CPU-GPU transfers).",
                }),
                "all_on_gpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Store all intermediate frames on GPU instead of CPU. Much faster (no transfers) but requires enough VRAM for all frames. Recommended for 48GB+ cards.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Number of frame pairs to process simultaneously in recursive mode. Ignored in single-pass mode (pairs are processed one at a time since each generates multiple frames).",
                }),
                "chunk_size": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Process input frames in chunks of this size (0=disabled). Bounds VRAM usage during processing but the full output is still assembled in RAM. To bound RAM, use the Segment Interpolate node instead.",
                }),
                "source_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Input frame rate. Required when target_fps > 0.",
                }),
                "target_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Target output FPS. When > 0, overrides multiplier and auto-computes a power-of-2 oversample up to 8x, then selects frames. 0 = use multiplier.",
                }),
            },
            "optional": {
                "settings": ("VFI_SETTINGS", {
                    "tooltip": "Auto-tuned settings from VFI Optimizer. Overrides batch_size, "
                               "chunk_size, keep_device, all_on_gpu, clear_cache_after_n_frames.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "oversampled", "elapsed_seconds")
    FUNCTION = "interpolate"
    CATEGORY = "video/GIMM-VFI"

    def _interpolate_frames_single_pass(self, frames, model, multiplier,
                                        device, storage_device, keep_device, all_on_gpu,
                                        clear_cache_after_n_frames, pbar, step_ref):
        """Single-pass interpolation using GIMM-VFI's arbitrary timestep capability."""
        num_intermediates = multiplier - 1
        logger.info(f"GIMM-VFI: single-pass {multiplier}x, {frames.shape[0]} input frames, {num_intermediates} intermediates/pair")
        new_frames = []
        num_pairs = frames.shape[0] - 1
        pairs_since_clear = 0

        for i in range(num_pairs):
            _throw_if_interrupted()
            frame0 = frames[i:i+1]
            frame1 = frames[i+1:i+2]

            mids = _interpolate_multi_with_offload(
                model, frame0, frame1, num_intermediates, device, keep_device
            )
            mids = [m.to(storage_device) for m in mids]

            new_frames.append(frames[i:i+1])
            for m in mids:
                new_frames.append(m)

            step_ref[0] += 1
            pbar.update_absolute(step_ref[0])

            pairs_since_clear += 1
            if pairs_since_clear >= clear_cache_after_n_frames:
                clear_gimm_caches()
                _soft_empty_cache()
                pairs_since_clear = 0

        new_frames.append(frames[-1:])
        result = torch.cat(new_frames, dim=0)

        clear_gimm_caches()
        _soft_empty_cache()

        return result

    def _interpolate_frames(self, frames, model, num_passes, batch_size,
                            device, storage_device, keep_device, all_on_gpu,
                            clear_cache_after_n_frames, pbar, step_ref):
        """Recursive 2x interpolation (standard approach, same as other models)."""
        for pass_idx in range(num_passes):
            logger.info(f"GIMM-VFI: pass {pass_idx + 1}/{num_passes}, {frames.shape[0]} -> {2 * frames.shape[0] - 1} frames")
            new_frames = []
            num_pairs = frames.shape[0] - 1
            pairs_since_clear = 0

            for i in range(0, num_pairs, batch_size):
                _throw_if_interrupted()
                batch_end = min(i + batch_size, num_pairs)
                actual_batch = batch_end - i

                frames0 = frames[i:batch_end]
                frames1 = frames[i + 1:batch_end + 1]

                mids = _interpolate_batch_with_offload(
                    model, frames0, frames1, device, keep_device
                )
                mids = mids.to(storage_device)

                for j in range(actual_batch):
                    new_frames.append(frames[i + j:i + j + 1])
                    new_frames.append(mids[j:j+1])

                step_ref[0] += actual_batch
                pbar.update_absolute(step_ref[0])

                pairs_since_clear += actual_batch
                if pairs_since_clear >= clear_cache_after_n_frames:
                    clear_gimm_caches()
                    _soft_empty_cache()
                    pairs_since_clear = 0

            new_frames.append(frames[-1:])
            frames = torch.cat(new_frames, dim=0)

            clear_gimm_caches()
            _soft_empty_cache()

        return frames

    @staticmethod
    def _count_steps(num_frames, num_passes):
        """Count total interpolation steps for recursive mode."""
        n = num_frames
        total = 0
        for _ in range(num_passes):
            total += n - 1
            n = 2 * n - 1
        return total

    @_with_elapsed_seconds("GIMM-VFI")
    def interpolate(self, images, model, multiplier, single_pass,
                    clear_cache_after_n_frames, keep_device, all_on_gpu,
                    batch_size, chunk_size,
                    source_fps=0.0, target_fps=0.0, settings=None):
        batch_size, chunk_size, keep_device, all_on_gpu, clear_cache_after_n_frames = \
            _apply_vfi_settings(settings, batch_size, chunk_size, keep_device,
                                all_on_gpu, clear_cache_after_n_frames)

        if images.shape[0] < 2:
            return (images, images)

        # Target FPS mode: auto-compute multiplier from fps ratio
        use_target_fps = _target_fps_enabled(source_fps, target_fps)
        if use_target_fps:
            num_passes, mult = _compute_target_fps_params(source_fps, target_fps)
            if num_passes == 0:
                all_frames = images.permute(0, 3, 1, 2)
                result = _select_target_fps_frames(all_frames, source_fps, target_fps, mult, all_frames.shape[0])
                return (result.cpu().permute(0, 2, 3, 1), images)
            # Override multiplier for single_pass mode
            multiplier = mult
        else:
            mult = multiplier

        if not single_pass or use_target_fps:
            num_passes_recursive = (
                num_passes if use_target_fps
                else {2: 1, 4: 2, 8: 3}[multiplier]
            )

        N = images.shape[0]
        expected = mult * (N - 1) + 1
        if use_target_fps:
            if num_passes == 0:
                expected = int(math.floor((N - 1) / source_fps * target_fps)) + 1
                logger.info(f"GIMM-VFI: {N} frames, {source_fps}fps -> {target_fps}fps (downsampling), expected output: {expected} frames")
            else:
                expected_target = int(math.floor((N - 1) / source_fps * target_fps)) + 1
                logger.info(f"GIMM-VFI: interpolating {N} frames, {source_fps}fps -> {target_fps}fps (oversample {mult}x, {num_passes} pass(es)), expected output: {expected_target} frames")
        else:
            mode = f"{num_passes_recursive} recursive pass(es)" if not single_pass else "single-pass"
            logger.info(f"GIMM-VFI: interpolating {N} frames, {mult}x ({mode}), expected output: {expected} frames")

        device = _get_torch_device()

        if all_on_gpu:
            keep_device = True

        storage_device = device if all_on_gpu else torch.device("cpu")

        # Convert from ComfyUI [B, H, W, C] to model [B, C, H, W]
        all_frames = images.permute(0, 3, 1, 2).to(storage_device)
        total_input = all_frames.shape[0]

        # Build chunk boundaries (1-frame overlap between consecutive chunks)
        if chunk_size < 2 or chunk_size >= total_input:
            chunks = [(0, total_input)]
        else:
            chunks = []
            start = 0
            while start < total_input - 1:
                end = min(start + chunk_size, total_input)
                chunks.append((start, end))
                start = end - 1  # overlap by 1 frame
                if end == total_input:
                    break

        if len(chunks) > 1:
            logger.info(f"GIMM-VFI: processing in {len(chunks)} chunk(s)")

        # Calculate total progress steps across all chunks
        if single_pass:
            total_steps = sum(ce - cs - 1 for cs, ce in chunks)
        else:
            total_steps = sum(self._count_steps(ce - cs, num_passes_recursive) for cs, ce in chunks)
        pbar = ProgressBar(total_steps)
        step_ref = [0]

        if keep_device:
            model.to(device)

        result_chunks = []
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_frames = all_frames[chunk_start:chunk_end].clone()

            if single_pass:
                chunk_result = self._interpolate_frames_single_pass(
                    chunk_frames, model, multiplier,
                    device, storage_device, keep_device, all_on_gpu,
                    clear_cache_after_n_frames, pbar, step_ref,
                )
            else:
                chunk_result = self._interpolate_frames(
                    chunk_frames, model, num_passes_recursive, batch_size,
                    device, storage_device, keep_device, all_on_gpu,
                    clear_cache_after_n_frames, pbar, step_ref,
                )

            # Skip first frame of subsequent chunks (duplicate of previous chunk's last frame)
            if chunk_idx > 0:
                chunk_result = chunk_result[1:]

            # Move completed chunk to CPU to bound memory when chunking
            if len(chunks) > 1:
                chunk_result = chunk_result.cpu()

            result_chunks.append(chunk_result)

        result = torch.cat(result_chunks, dim=0)

        # Convert oversampled to ComfyUI format for second output
        oversampled = result.cpu().permute(0, 2, 3, 1)

        # Target FPS: select frames from oversampled result
        if use_target_fps:
            result = _select_target_fps_frames(result, source_fps, target_fps, mult, total_input)

        # Convert back to ComfyUI [B, H, W, C], on CPU
        result = result.cpu().permute(0, 2, 3, 1)
        logger.info(f"GIMM-VFI: done, {result.shape[0]} output frames")
        return (result, oversampled)


class GIMMVFISegmentInterpolate(GIMMVFIInterpolate):
    """Process a numbered segment of the input batch for GIMM-VFI.

    Chain multiple instances with Save nodes between them to bound peak RAM.
    The model pass-through output forces sequential execution so each segment
    saves and frees from RAM before the next starts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = GIMMVFIInterpolate.INPUT_TYPES()
        base["required"]["segment_index"] = ("INT", {
            "default": 0, "min": 0, "max": 10000, "step": 1,
            "tooltip": "Which segment to process (0-based). Bounds RAM by only producing this segment's output frames, "
                       "unlike chunk_size which bounds VRAM but still assembles the full output in RAM. "
                       "Chain the model output to the next Segment Interpolate to force sequential execution.",
        })
        base["required"]["segment_size"] = ("INT", {
            "default": 500, "min": 2, "max": 10000, "step": 1,
            "tooltip": "Number of input frames per segment. Adjacent segments overlap by 1 frame for seamless stitching. "
                       "Smaller = less peak RAM per segment. Save each segment's output to disk before the next runs.",
        })
        return base

    RETURN_TYPES = ("IMAGE", "GIMM_VFI_MODEL", "FLOAT")
    RETURN_NAMES = ("images", "model", "elapsed_seconds")
    FUNCTION = "interpolate"
    CATEGORY = "video/GIMM-VFI"

    @_with_elapsed_seconds("GIMM-VFI segment")
    def interpolate(self, images, model, multiplier, single_pass,
                    clear_cache_after_n_frames, keep_device, all_on_gpu,
                    batch_size, chunk_size, segment_index, segment_size,
                    source_fps=0.0, target_fps=0.0, settings=None):
        batch_size, chunk_size, keep_device, all_on_gpu, clear_cache_after_n_frames = \
            _apply_vfi_settings(settings, batch_size, chunk_size, keep_device,
                                all_on_gpu, clear_cache_after_n_frames)

        total_input = images.shape[0]
        use_target_fps = _target_fps_enabled(source_fps, target_fps)

        # Compute segment boundaries (1-frame overlap)
        start = segment_index * (segment_size - 1)
        end = min(start + segment_size, total_input)

        if start >= total_input - 1:
            return (images[:1], model)

        segment_images = images[start:end]
        logger.info(f"GIMM-VFI segment {segment_index}: input frames [{start}:{end}] of {total_input}")

        if use_target_fps:
            num_passes, mult = _compute_target_fps_params(source_fps, target_fps)

            seg_start_time = start / source_fps
            seg_end_time = (end - 1) / source_fps
            duration = (total_input - 1) / source_fps
            total_output = int(math.floor(duration * target_fps)) + 1

            if segment_index == 0:
                j_start = 0
            else:
                j_start = int(math.floor(seg_start_time * target_fps)) + 1
            j_end = min(int(math.floor(seg_end_time * target_fps)), total_output - 1)

            if j_start > j_end:
                raise ValueError(
                    "This segment contains no frames at the requested target FPS. "
                    "Increase segment_size or use the non-segment Interpolate node."
                )

            logger.info(f"GIMM-VFI segment {segment_index}: target fps output j=[{j_start}..{j_end}]")

            if num_passes == 0:
                oversampled_fps = source_fps * mult
                all_seg = segment_images.permute(0, 3, 1, 2)
                out_frames = []
                for j in range(j_start, j_end + 1):
                    global_idx = min(round(j / target_fps * oversampled_fps), total_input - 1)
                    local_idx = global_idx - start
                    local_idx = max(0, min(local_idx, all_seg.shape[0] - 1))
                    out_frames.append(all_seg[local_idx:local_idx + 1])
                result = torch.cat(out_frames, dim=0).cpu().permute(0, 2, 3, 1)
                return (result, model)

            # Oversample segment directly
            device = _get_torch_device()
            if all_on_gpu:
                keep_device = True
            storage_device = device if all_on_gpu else torch.device("cpu")
            seg_frames = segment_images.permute(0, 3, 1, 2).to(storage_device)

            if single_pass:
                total_steps = seg_frames.shape[0] - 1
            else:
                total_steps = self._count_steps(seg_frames.shape[0], num_passes)
            pbar = ProgressBar(total_steps)
            step_ref = [0]
            if keep_device:
                model.to(device)

            if single_pass:
                oversampled = self._interpolate_frames_single_pass(
                    seg_frames, model, mult,
                    device, storage_device, keep_device, all_on_gpu,
                    clear_cache_after_n_frames, pbar, step_ref,
                )
            else:
                oversampled = self._interpolate_frames(
                    seg_frames, model, num_passes, batch_size,
                    device, storage_device, keep_device, all_on_gpu,
                    clear_cache_after_n_frames, pbar, step_ref,
                )
            oversampled_fps = source_fps * mult

            out_frames = []
            for j in range(j_start, j_end + 1):
                global_oversamp_idx = round(j / target_fps * oversampled_fps)
                local_idx = global_oversamp_idx - start * mult
                local_idx = max(0, min(local_idx, oversampled.shape[0] - 1))
                out_frames.append(oversampled[local_idx:local_idx + 1])
            result = torch.cat(out_frames, dim=0).cpu().permute(0, 2, 3, 1)
            return (result, model)

        # Standard multiplier mode
        is_continuation = segment_index > 0
        (result, _, _) = super().interpolate(
            segment_images, model, multiplier, single_pass,
            clear_cache_after_n_frames, keep_device, all_on_gpu,
            batch_size, chunk_size,
        )

        if is_continuation:
            result = result[1:]

        return (result, model)
