"""Sequence-native ComfyUI adapter for the official Apache-2.0 LDF-VFI runtime."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
import sys
import threading
import types

from einops import rearrange, repeat
import torch
import torch.nn.functional as F


logger = logging.getLogger("Tween")
_LDF_NAMESPACE = "_tween_ldf_upstream"
_LDF_IMPORT_LOCK = threading.RLock()


def _cuda_bf16_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    with torch.cuda.device(device):
        return torch.cuda.is_bf16_supported()


def _namespace_package(name: str, path: Path):
    package = sys.modules.get(name)
    if package is not None:
        return package
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    sys.modules[name] = package
    return package


def load_ldf_runtime(source_root: str):
    """Load LDF under an isolated namespace without polluting ``training``."""
    root = Path(source_root).resolve()
    required = root / "training" / "models" / "precond.py"
    if not required.is_file():
        raise RuntimeError(f"Invalid LDF-VFI source directory: missing {required}")

    with _LDF_IMPORT_LOCK:
        cached = sys.modules.get(f"{_LDF_NAMESPACE}.models.precond")
        if cached is not None:
            transformer = importlib.import_module(f"{_LDF_NAMESPACE}.models.transformer_wan")
            return {
                "Precond": cached.Precond,
                "ConditionalVAE": cached.Wan2_1SpatialTiledConditionEncoder3Dv2,
                "MaskEncoder": cached.MaskSpatialTiledEncoder3D,
                "Transformer": transformer.WanTransformer3DModel,
            }

        training_root = _namespace_package(_LDF_NAMESPACE, root / "training")
        saved_training_modules = {
            name: module for name, module in tuple(sys.modules.items())
            if name == "training" or name.startswith("training.")
        }
        for name in saved_training_modules:
            sys.modules.pop(name, None)
        # One upstream transformer import is absolute (training.distributed.util).
        # Temporarily alias only while importing, then restore the host process.
        sys.modules["training"] = training_root
        try:
            precond = importlib.import_module(f"{_LDF_NAMESPACE}.models.precond")
            transformer = importlib.import_module(f"{_LDF_NAMESPACE}.models.transformer_wan")
        except (ImportError, AttributeError) as exc:
            for name in tuple(sys.modules):
                if name == _LDF_NAMESPACE or name.startswith(f"{_LDF_NAMESPACE}."):
                    sys.modules.pop(name, None)
            raise RuntimeError(
                "LDF-VFI requires PyTorch 2.5+ and diffusers 0.33+. "
                "Install Tween's current requirements and restart ComfyUI."
            ) from exc
        finally:
            for name in tuple(sys.modules):
                if name == "training" or name.startswith("training."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_training_modules)

    return {
        "Precond": precond.Precond,
        "ConditionalVAE": precond.Wan2_1SpatialTiledConditionEncoder3Dv2,
        "MaskEncoder": precond.MaskSpatialTiledEncoder3D,
        "Transformer": transformer.WanTransformer3DModel,
    }


def _upsample_nearest(frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Expand sparse source frames to every temporal position in a window."""
    kept_indices = torch.where(mask)[0]
    if kept_indices.numel() == 0:
        raise RuntimeError("LDF-VFI received a temporal window with no source frames")
    positions = torch.arange(mask.shape[0], device=mask.device)
    nearest = (positions[:, None] - kept_indices[None, :]).abs().argmin(dim=1)
    return frames[nearest]


class LDFVFIModel:
    """Long-sequence diffusion interpolation using LDF's skip-concat sampler."""

    TRAIN_FRAMES = 60
    TILE_TIME = 20
    CONDITION_TILES = 1

    def __init__(
        self,
        model_root: str,
        vae_path: str,
        source_root: str,
        tile_size: int = 256,
        tile_overlap: int = 64,
        vae_batch_size: int = 8,
        attention_type: str = "slide_chunk_all_block_2x1x1",
    ):
        if tile_size % 8 or tile_overlap % 8:
            raise ValueError("LDF-VFI tile size and overlap must be divisible by 8")
        if tile_overlap >= tile_size:
            raise ValueError("LDF-VFI tile overlap must be smaller than tile size")

        runtime = load_ldf_runtime(source_root)
        self.device = "cpu"
        self.dtype = torch.bfloat16
        self.vae_path = vae_path
        self._ConditionalVAE = runtime["ConditionalVAE"]
        self._MaskEncoder = runtime["MaskEncoder"]
        self._Precond = runtime["Precond"]

        logger.info("Loading LDF-VFI transformer from %s", model_root)
        try:
            transformer = runtime["Transformer"].from_pretrained(
                model_root,
                subfolder="transformer",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
        except TypeError:
            transformer = runtime["Transformer"].from_pretrained(
                model_root, subfolder="transformer", torch_dtype=self.dtype
            )
        transformer.set_attention_type(attention_type)
        transformer.requires_grad_(False).eval()

        stride = tile_size - tile_overlap
        tiled_kwargs = {
            "tile_sample_min_height": tile_size,
            "tile_sample_min_width": tile_size,
            "tile_sample_min_time": self.TILE_TIME,
            "tile_sample_stride_height": stride,
            "tile_sample_stride_width": stride,
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 4,
        }
        # The same conditional VAE can encode conditions and decode predictions;
        # sharing it avoids loading a second ~800 MB copy as the reference CLI does.
        self.vae = self._ConditionalVAE(vae_path, vae_batch_size, **tiled_kwargs)
        self.mask_encoder = self._MaskEncoder(**tiled_kwargs)
        self.model = self._Precond(
            transformer=transformer,
            vae=self.vae,
            lq_encoder=self.vae,
            msk_encoder=self.mask_encoder,
        )
        self.model.requires_grad_(False).eval()

    @property
    def transformer(self):
        return self.model.transformer

    def _move_auxiliary_models(self, device: torch.device) -> None:
        self.mask_encoder.mask_encoder.to(device=device, dtype=self.dtype)
        if self.vae.vae is None:
            if device.type != "cpu":
                self.vae.init(device)
            return
        self.vae.vae.to(device=device, dtype=self.dtype)
        if hasattr(self.vae, "mean") and hasattr(self.vae, "std"):
            self.vae.mean = self.vae.mean.to(device=device, dtype=self.dtype)
            self.vae.std = self.vae.std.to(device=device, dtype=self.dtype)
            self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]

    def to(self, device):
        target = torch.device(device)
        if target.type == "cuda" and not _cuda_bf16_supported(target):
            raise RuntimeError("LDF-VFI requires a CUDA GPU with BF16 support (Ampere or newer)")
        # from_pretrained(torch_dtype=...) keeps numerically sensitive modules
        # (time embedding, norms, scale/shift) in FP32. Passing dtype here would
        # flatten that mixed-precision policy and diffusers warns that results
        # can become inconsistent.
        self.transformer.to(device=target)
        self._move_auxiliary_models(target)
        self.device = str(target)
        return self

    def clear_cache(self) -> None:
        for name in (
            "_swin_attention_mask",
            "_sliding_chunk_attention_mask",
            "_sliding_window_attention_mask",
        ):
            method = getattr(self.transformer, name, None)
            cache_clear = getattr(method, "cache_clear", None)
            if cache_clear is not None:
                cache_clear()

    def _prepare_condition(self, frames, mask, device):
        if mask.shape[0] > self.TRAIN_FRAMES:
            raise ValueError("Internal LDF temporal window exceeds the training window")
        if mask.shape[0] < self.TRAIN_FRAMES:
            mask = F.pad(mask, (0, self.TRAIN_FRAMES - mask.shape[0]))
        dense = _upsample_nearest(frames, mask)
        dense = rearrange(dense, "t c h w -> 1 c t h w")
        dense = dense.to(device=device, dtype=self.dtype, non_blocking=True).mul(2).sub(1)
        dense_mask = repeat(
            mask, "t -> 1 1 t h w", h=dense.shape[-2], w=dense.shape[-1]
        ).to(device=device, dtype=self.dtype)
        condition = self.vae.encode(dense, for_train=True)
        encoded_mask = self.mask_encoder.encode(dense_mask, for_train=True)
        return dense, dense_mask, condition, encoded_mask

    def _time_schedule(self, num_steps: int, t_shift: float) -> torch.Tensor:
        schedule = torch.linspace(1.0, 0.0, steps=num_steps + 1)
        return t_shift * schedule / (1 + (t_shift - 1) * schedule)

    def _predict_step(self, latent, timestep, condition, encoded_mask):
        return self.model.predict_v(latent, timestep, condition, encoded_mask)

    def _sample_free(self, condition, encoded_mask, schedule, device, progress):
        latent = torch.randn_like(condition)
        for index in range(schedule.shape[0] - 1):
            progress()
            timestep = torch.full(
                condition.shape[:-4], float(schedule[index]), device=device, dtype=self.dtype
            )
            velocity = self._predict_step(latent, timestep, condition, encoded_mask)
            step_size = float(schedule[index + 1] - schedule[index])
            latent = latent + velocity * step_size
        return latent

    def _sample_between(self, previous, following, condition, encoded_mask,
                        schedule, t_cond, device, progress):
        previous_noisy = previous * (1 - t_cond) + torch.randn_like(previous) * t_cond
        following_noisy = following * (1 - t_cond) + torch.randn_like(following) * t_cond
        middle = torch.randn_like(condition[:, self.CONDITION_TILES:-self.CONDITION_TILES])
        previous_t = torch.full(
            previous.shape[:-4], t_cond, device=device, dtype=self.dtype
        )
        following_t = torch.full(
            following.shape[:-4], t_cond, device=device, dtype=self.dtype
        )
        for index in range(schedule.shape[0] - 1):
            progress()
            latent = torch.cat((previous_noisy, middle, following_noisy), dim=1)
            middle_t = torch.full(
                middle.shape[:-4], float(schedule[index]), device=device, dtype=self.dtype
            )
            timestep = torch.cat((previous_t, middle_t, following_t), dim=1)
            velocity = self._predict_step(
                latent, timestep, condition, encoded_mask
            )[:, self.CONDITION_TILES:-self.CONDITION_TILES]
            step_size = float(schedule[index + 1] - schedule[index])
            middle = middle + velocity * step_size
        return middle

    def _sample_tail(self, previous, condition, encoded_mask, schedule,
                     t_cond, device, progress):
        previous_noisy = previous * (1 - t_cond) + torch.randn_like(previous) * t_cond
        tail = torch.randn_like(condition[:, self.CONDITION_TILES:])
        previous_t = torch.full(
            previous.shape[:-4], t_cond, device=device, dtype=self.dtype
        )
        for index in range(schedule.shape[0] - 1):
            progress()
            latent = torch.cat((previous_noisy, tail), dim=1)
            tail_t = torch.full(
                tail.shape[:-4], float(schedule[index]), device=device, dtype=self.dtype
            )
            timestep = torch.cat((previous_t, tail_t), dim=1)
            velocity = self._predict_step(
                latent, timestep, condition, encoded_mask
            )[:, self.CONDITION_TILES:]
            step_size = float(schedule[index + 1] - schedule[index])
            tail = tail + velocity * step_size
        return tail

    def _decode(self, latent, dense, dense_mask, height, width):
        latent = rearrange(
            latent, "1 nt nh nw c t h w -> 1 nt c t (nh h) (nw w)"
        )
        if dense.ndim != 5 or dense_mask.ndim != 5:
            raise RuntimeError(
                "LDF-VFI decode conditions must use [batch, channels, time, height, width]"
            )

        temporal_tiles = latent.shape[1]
        if dense.shape[2] % temporal_tiles:
            raise RuntimeError(
                f"LDF-VFI condition length {dense.shape[2]} is not divisible by "
                f"the {temporal_tiles} decode tiles"
            )

        # Mirror the official generate.vae_decode adapter. The conditional VAE
        # consumes one condition and mask tile per latent temporal tile, not a
        # single continuous 5-D condition tensor.
        decode_height = latent.shape[-2] * self.vae.spatial_compression_ratio
        decode_width = latent.shape[-1] * self.vae.spatial_compression_ratio
        pad_height = decode_height - dense.shape[-2]
        pad_width = decode_width - dense.shape[-1]
        if pad_height < 0 or pad_width < 0:
            raise RuntimeError(
                "LDF-VFI decoded latent is smaller than its conditioning frames; "
                "check the VAE tile and overlap settings"
            )
        dense = F.pad(dense, (0, pad_width, 0, pad_height))
        dense = rearrange(
            dense, "b c (nt t) h w -> b nt c t h w", nt=temporal_tiles
        )

        dense_mask = dense_mask[..., 0, 0]
        dense_mask = repeat(
            dense_mask, "b c t -> b c t h w", h=decode_height, w=decode_width
        )
        dense_mask = rearrange(
            dense_mask, "b c (nt t) h w -> b nt c t h w", nt=temporal_tiles
        )

        prediction = self.vae.decode(
            latent, dense, dense_mask
        )[..., :height, :width]
        return rearrange(prediction, "1 c t h w -> t c h w").add(1).mul(0.5).clamp_(0, 1).float().cpu()

    @staticmethod
    def sampling_block_count(num_input_frames: int, temporal_factor: int) -> int:
        total_length = num_input_frames * temporal_factor
        t0 = 40
        stride = 20
        blocks = 1
        while t0 + stride <= total_length - 1:
            blocks += 2
            t0 += stride * 2
        if t0 < total_length:
            blocks += 1
        return blocks

    @torch.no_grad()
    def interpolate_sequence(
        self,
        frames: torch.Tensor,
        temporal_factor: int,
        num_steps: int = 16,
        t_shift: float = 8.0,
        t_cond: float = 0.1,
        seed: int = 42,
        progress_callback=None,
    ) -> torch.Tensor:
        if not 2 <= temporal_factor <= 16:
            raise ValueError("LDF-VFI temporal factor must be between 2 and 16")
        if num_steps < 1:
            raise ValueError("LDF-VFI num_steps must be at least 1")
        if t_shift <= 0:
            raise ValueError("LDF-VFI t_shift must be greater than 0")
        if not 0 <= t_cond <= 1:
            raise ValueError("LDF-VFI t_cond must be between 0 and 1")
        if frames.shape[0] < 2:
            return frames

        device = next(self.transformer.parameters()).device
        if device.type != "cuda":
            raise RuntimeError("Move LDF-VFI to a CUDA device before interpolation")
        source = frames.detach().float().cpu()
        height, width = source.shape[-2:]
        schedule = self._time_schedule(num_steps, t_shift)
        progress_callback = progress_callback or (lambda: None)

        cuda_index = device.index
        if cuda_index is None:
            cuda_index = torch.cuda.current_device()
        rng_context = torch.random.fork_rng(devices=[cuda_index])
        with rng_context:
            torch.manual_seed(int(seed))
            with torch.cuda.device(device):
                torch.cuda.manual_seed(int(seed))
            chunks = self._interpolate_sequence_impl(
                source, temporal_factor, schedule, t_cond, device, progress_callback
            )

        result = torch.cat(chunks, dim=0)
        expected = (frames.shape[0] - 1) * temporal_factor + 1
        return result[:expected]

    def _interpolate_sequence_impl(self, source, factor, schedule, t_cond,
                                   device, progress):
        total_mask = torch.zeros(source.shape[0] * factor, dtype=torch.bool)
        total_mask[::factor] = True
        n_tiles = self.TRAIN_FRAMES // self.TILE_TIME
        output_tiles = n_tiles - self.CONDITION_TILES
        outputs = []

        # First chunk.
        mask = total_mask[:self.TRAIN_FRAMES]
        input_end = int(mask.sum())
        dense, dense_mask, condition, encoded_mask = self._prepare_condition(
            source[:input_end], mask, device
        )
        latent = self._sample_free(condition, encoded_mask, schedule, device, progress)
        latent = latent[:, :output_tiles]
        previous = latent[:, -self.CONDITION_TILES:]
        decode_time = output_tiles * self.TILE_TIME
        outputs.append(self._decode(
            latent, dense[:, :, :decode_time], dense_mask[:, :, :decode_time],
            source.shape[-2], source.shape[-1],
        ))

        t0 = output_tiles * self.TILE_TIME
        stride = (n_tiles - self.CONDITION_TILES * 2) * self.TILE_TIME
        while t0 + stride <= total_mask.shape[0] - 1:
            # A future/skip chunk establishes the far-side condition.
            input_start = t0 + stride - self.CONDITION_TILES * self.TILE_TIME
            input_end_t = t0 + stride * 2 + self.CONDITION_TILES * self.TILE_TIME
            mask = total_mask[input_start:input_end_t]
            source_start = int(total_mask[:input_start].sum())
            source_end = int(total_mask[:input_end_t].sum())
            dense, dense_mask, condition, encoded_mask = self._prepare_condition(
                source[source_start:source_end], mask, device
            )
            skip = self._sample_free(condition, encoded_mask, schedule, device, progress)
            skip = skip[:, self.CONDITION_TILES:-self.CONDITION_TILES]
            following = skip[:, :self.CONDITION_TILES]
            previous_next = skip[:, -self.CONDITION_TILES:]
            start_time = self.CONDITION_TILES * self.TILE_TIME
            end_time = output_tiles * self.TILE_TIME
            decoded_skip = self._decode(
                skip, dense[:, :, start_time:end_time], dense_mask[:, :, start_time:end_time],
                source.shape[-2], source.shape[-1],
            )

            # Fill the gap between the preceding and skip chunks.
            input_start = t0 - self.CONDITION_TILES * self.TILE_TIME
            input_end_t = t0 + stride + self.CONDITION_TILES * self.TILE_TIME
            mask = total_mask[input_start:input_end_t]
            source_start = int(total_mask[:input_start].sum())
            source_end = int(total_mask[:input_end_t].sum())
            dense, dense_mask, condition, encoded_mask = self._prepare_condition(
                source[source_start:source_end], mask, device
            )
            middle = self._sample_between(
                previous, following, condition, encoded_mask,
                schedule, t_cond, device, progress,
            )
            decoded_middle = self._decode(
                middle, dense[:, :, start_time:end_time], dense_mask[:, :, start_time:end_time],
                source.shape[-2], source.shape[-1],
            )
            outputs.extend((decoded_middle, decoded_skip))
            previous = previous_next
            t0 += stride * 2

        # Remaining tail.
        if t0 < total_mask.shape[0]:
            input_start = t0 - self.CONDITION_TILES * self.TILE_TIME
            mask = total_mask[input_start:]
            source_start = int(total_mask[:input_start].sum())
            dense, dense_mask, condition, encoded_mask = self._prepare_condition(
                source[source_start:], mask, device
            )
            tail = self._sample_tail(
                previous, condition, encoded_mask, schedule,
                t_cond, device, progress,
            )
            start_time = self.CONDITION_TILES * self.TILE_TIME
            outputs.append(self._decode(
                tail, dense[:, :, start_time:], dense_mask[:, :, start_time:],
                source.shape[-2], source.shape[-1],
            ))

        return outputs
