"""ComfyUI inference adapter for the official SPEED runtime."""

from __future__ import annotations

from contextlib import nullcontext
import importlib
import logging
from pathlib import Path
import sys
import types

import torch


logger = logging.getLogger("Tween")
_SPEED_NAMESPACE = "_tween_speed_upstream"


def _cuda_bf16_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    with torch.cuda.device(device):
        return torch.cuda.is_bf16_supported()


def _namespace_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    sys.modules[name] = package


def load_speed_model_class(source_root: str):
    """Import SpeedDiT without adding the upstream repository to sys.path."""
    root = Path(source_root).resolve()
    model_file = root / "src" / "models" / "model.py"
    if not model_file.is_file():
        raise RuntimeError(f"Invalid SPEED source directory: missing {model_file}")

    _namespace_package(_SPEED_NAMESPACE, root)
    _namespace_package(f"{_SPEED_NAMESPACE}.src", root / "src")
    _namespace_package(f"{_SPEED_NAMESPACE}.src.models", root / "src" / "models")
    module = importlib.import_module(f"{_SPEED_NAMESPACE}.src.models.model")
    return module.SpeedDiT


class SpeedVFIModel:
    """Midpoint interpolation wrapper around the official SPEED SpeedDiT."""

    def __init__(self, checkpoint_path: str, source_root: str,
                 precision: str = "auto", device: str = "cpu"):
        SpeedDiT = load_speed_model_class(source_root)
        self.model = SpeedDiT(
            hidden_dim=768,
            head_dim=64,
            depths=(2, 6, 4),
            patch_sizes=(64, 32, 16),
        )
        self.precision = precision
        self.device = str(device)
        self._seed = 0
        self._generator = None
        self._generator_device = None
        self._load_checkpoint(checkpoint_path)
        self.model.requires_grad_(False).eval()
        self.to(device)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise TypeError(f"SPEED checkpoint does not contain a state dict: {checkpoint_path}")
        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)

    def _autocast_dtype(self, device: torch.device):
        if self.precision == "fp32" or device.type != "cuda":
            return None
        if self.precision == "fp16":
            return torch.float16
        if self.precision == "bf16":
            if not _cuda_bf16_supported(device):
                raise RuntimeError(
                    "SPEED BF16 precision requires a CUDA GPU with BF16 support; "
                    "select auto or fp16 on this GPU"
                )
            return torch.bfloat16
        if _cuda_bf16_supported(device):
            return torch.bfloat16
        return torch.float16

    def to(self, device):
        target = torch.device(device)
        self.device = str(target)
        # Match the official runtime: retain FP32 weights and use autocast for
        # CUDA inference. This also avoids mixed-dtype timestep embedding bugs.
        self.model.to(device=target, dtype=torch.float32)
        # Keep the generator alive across CPU offloading. Recreating it on every
        # pair batch would restart the noise stream whenever keep_device=False.
        # _get_generator replaces it automatically if inference changes device.
        return self

    def set_seed(self, seed: int) -> None:
        self._seed = int(seed)
        self._generator = None
        self._generator_device = None

    def reset_seed(self) -> None:
        self.set_seed(self._seed)

    def clear_cache(self) -> None:
        rope = getattr(self.model, "rope_embedder", None)
        cache = getattr(rope, "rope_cache", None)
        if cache is not None:
            cache.clear()

    def _get_generator(self, device: torch.device) -> torch.Generator:
        device_name = str(device)
        if self._generator is None or self._generator_device != device_name:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self._seed)
            self._generator_device = device_name
        return self._generator

    @torch.no_grad()
    def interpolate_batch(self, frames0, frames1, time_step=0.5):
        if abs(float(time_step) - 0.5) > 1e-6:
            raise ValueError("SPEED's released checkpoint supports midpoint interpolation only")

        device = next(self.model.parameters()).device
        frame0 = frames0.to(device=device, dtype=torch.float32, non_blocking=True).mul(2).sub(1)
        frame1 = frames1.to(device=device, dtype=torch.float32, non_blocking=True).mul(2).sub(1)
        cond_frames = torch.cat((frame0, frame1), dim=0)
        noisy_frames = torch.randn(
            frame0.shape,
            generator=self._get_generator(device),
            device=device,
            dtype=torch.float32,
        )
        timestep = torch.full(
            (frame0.shape[0],), 1000.0, device=device, dtype=torch.float32
        )
        autocast_dtype = self._autocast_dtype(device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with autocast:
            prediction = self.model(
                noisy_frames=noisy_frames,
                cond_frames=cond_frames,
                timestep=timestep,
            )
        return prediction.div(2).add(0.5).clamp_(0, 1).float()
