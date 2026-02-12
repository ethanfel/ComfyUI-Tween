import logging
from functools import partial

import torch
import torch.nn as nn

from .bim_vfi_arch import BiMVFI
from .ema_vfi_arch import feature_extractor as ema_feature_extractor
from .ema_vfi_arch import MultiScaleFlow as EMAMultiScaleFlow
from .utils.padder import InputPadder

logger = logging.getLogger("BIM-VFI")


class BiMVFIModel:
    """Clean inference wrapper around BiMVFI for ComfyUI integration."""

    def __init__(self, checkpoint_path, pyr_level=3, auto_pyr_level=True, device="cpu"):
        self.pyr_level = pyr_level
        self.auto_pyr_level = auto_pyr_level
        self.device = device

        self.model = BiMVFI(pyr_level=pyr_level, feat_channels=32)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.to(device)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip common prefixes (e.g. "module." from DDP or "model." from wrapper)
        cleaned = {}
        for k, v in state_dict.items():
            key = k
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("model."):
                key = key[len("model."):]
            cleaned[key] = v

        self.model.load_state_dict(cleaned)

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def _get_pyr_level(self, h):
        if self.auto_pyr_level:
            if h >= 2160:
                return 7
            elif h >= 1080:
                return 6
            elif h >= 540:
                return 5
            else:
                return 3
        return self.pyr_level

    @torch.no_grad()
    def interpolate_pair(self, frame0, frame1, time_step=0.5):
        """Interpolate a single frame between two input frames.

        Args:
            frame0: [1, C, H, W] tensor, float32, range [0, 1]
            frame1: [1, C, H, W] tensor, float32, range [0, 1]
            time_step: float in (0, 1), temporal position of interpolated frame

        Returns:
            Interpolated frame as [1, C, H, W] tensor, float32, clamped to [0, 1]
        """
        device = next(self.model.parameters()).device
        img0 = frame0.to(device)
        img1 = frame1.to(device)

        pyr_level = self._get_pyr_level(img0.shape[2])
        time_step_tensor = torch.tensor([time_step], device=device).view(1, 1, 1, 1)

        result_dict = self.model(
            img0=img0, img1=img1,
            time_step=time_step_tensor,
            pyr_level=pyr_level,
        )

        interp = result_dict["imgt_pred"]
        interp = torch.clamp(interp, 0, 1)
        return interp

    @torch.no_grad()
    def interpolate_batch(self, frames0, frames1, time_step=0.5):
        """Interpolate multiple frame pairs at once.

        Args:
            frames0: [B, C, H, W] tensor, float32, range [0, 1]
            frames1: [B, C, H, W] tensor, float32, range [0, 1]
            time_step: float in (0, 1), temporal position of interpolated frames

        Returns:
            Interpolated frames as [B, C, H, W] tensor, float32, clamped to [0, 1]
        """
        device = next(self.model.parameters()).device
        img0 = frames0.to(device)
        img1 = frames1.to(device)

        pyr_level = self._get_pyr_level(img0.shape[2])
        time_step_tensor = torch.tensor([time_step], device=device).view(1, 1, 1, 1)

        result_dict = self.model(
            img0=img0, img1=img1,
            time_step=time_step_tensor,
            pyr_level=pyr_level,
        )

        interp = result_dict["imgt_pred"]
        interp = torch.clamp(interp, 0, 1)
        return interp


# ---------------------------------------------------------------------------
# EMA-VFI model wrapper
# ---------------------------------------------------------------------------

def _ema_init_model_config(F=32, W=7, depth=[2, 2, 2, 4, 4]):
    """Build EMA-VFI model config dicts (backbone + multiscale)."""
    return {
        'embed_dims': [F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims': [0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'num_heads': [8*F//32, 16*F//32],
        'mlp_ratios': [4, 4],
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'depths': depth,
        'window_sizes': [W, W]
    }, {
        'embed_dims': [F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims': [0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'depths': depth,
        'num_heads': [8*F//32, 16*F//32],
        'window_sizes': [W, W],
        'scales': [4, 8, 16],
        'hidden_dims': [4*F, 4*F],
        'c': F
    }


def _ema_detect_variant(filename):
    """Auto-detect model variant and timestep support from filename.

    Returns (F, depth, supports_arbitrary_t).
    """
    name = filename.lower()
    is_small = "small" in name
    supports_t = "_t." in name or "_t_" in name or name.endswith("_t")

    if is_small:
        return 16, [2, 2, 2, 2, 2], supports_t
    else:
        return 32, [2, 2, 2, 4, 4], supports_t


class EMAVFIModel:
    """Clean inference wrapper around EMA-VFI for ComfyUI integration."""

    def __init__(self, checkpoint_path, variant="auto", tta=False, device="cpu"):
        import os
        filename = os.path.basename(checkpoint_path)

        if variant == "auto":
            F_dim, depth, self.supports_arbitrary_t = _ema_detect_variant(filename)
        elif variant == "small":
            F_dim, depth = 16, [2, 2, 2, 2, 2]
            self.supports_arbitrary_t = "_t." in filename.lower() or "_t_" in filename.lower()
        else:  # large
            F_dim, depth = 32, [2, 2, 2, 4, 4]
            self.supports_arbitrary_t = "_t." in filename.lower() or "_t_" in filename.lower()

        self.tta = tta
        self.device = device
        self.variant_name = "small" if F_dim == 16 else "large"

        backbone_cfg, multiscale_cfg = _ema_init_model_config(F=F_dim, depth=depth)
        backbone = ema_feature_extractor(**backbone_cfg)
        self.model = EMAMultiScaleFlow(backbone, **multiscale_cfg)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.to(device)

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint with module prefix stripping and buffer filtering."""
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle wrapped checkpoint formats
        if isinstance(state_dict, dict):
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        # Strip "module." prefix and filter out attn_mask/HW buffers
        cleaned = {}
        for k, v in state_dict.items():
            if "attn_mask" in k or k.endswith(".HW"):
                continue
            key = k
            if key.startswith("module."):
                key = key[len("module."):]
            cleaned[key] = v

        self.model.load_state_dict(cleaned)

    def to(self, device):
        """Move model to device (returns self for chaining)."""
        self.device = device
        self.model.to(device)
        return self

    @torch.no_grad()
    def _inference(self, img0, img1, timestep=0.5):
        """Run single inference pass. Inputs already padded, on device."""
        B = img0.shape[0]
        imgs = torch.cat((img0, img1), 1)

        if self.tta:
            imgs_ = imgs.flip(2).flip(3)
            input_batch = torch.cat((imgs, imgs_), 0)
            _, _, _, preds = self.model(input_batch, timestep=timestep)
            return (preds[:B] + preds[B:].flip(2).flip(3)) / 2.
        else:
            _, _, _, pred = self.model(imgs, timestep=timestep)
            return pred

    @torch.no_grad()
    def interpolate_pair(self, frame0, frame1, time_step=0.5):
        """Interpolate a single frame between two input frames.

        Args:
            frame0: [1, C, H, W] tensor, float32, range [0, 1]
            frame1: [1, C, H, W] tensor, float32, range [0, 1]
            time_step: float in (0, 1)

        Returns:
            Interpolated frame as [1, C, H, W] tensor, float32, clamped to [0, 1]
        """
        device = next(self.model.parameters()).device
        img0 = frame0.to(device)
        img1 = frame1.to(device)

        padder = InputPadder(img0.shape, divisor=32, mode='replicate', center=True)
        img0, img1 = padder.pad(img0, img1)

        pred = self._inference(img0, img1, timestep=time_step)
        pred = padder.unpad(pred)
        return torch.clamp(pred, 0, 1)

    @torch.no_grad()
    def interpolate_batch(self, frames0, frames1, time_step=0.5):
        """Interpolate multiple frame pairs at once.

        Args:
            frames0: [B, C, H, W] tensor, float32, range [0, 1]
            frames1: [B, C, H, W] tensor, float32, range [0, 1]
            time_step: float in (0, 1)

        Returns:
            Interpolated frames as [B, C, H, W] tensor, float32, clamped to [0, 1]
        """
        device = next(self.model.parameters()).device
        img0 = frames0.to(device)
        img1 = frames1.to(device)

        padder = InputPadder(img0.shape, divisor=32, mode='replicate', center=True)
        img0, img1 = padder.pad(img0, img1)

        pred = self._inference(img0, img1, timestep=time_step)
        pred = padder.unpad(pred)
        return torch.clamp(pred, 0, 1)
