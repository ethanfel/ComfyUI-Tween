import logging
from functools import partial

import torch
import torch.nn as nn

from .bim_vfi_arch import BiMVFI
from .ema_vfi_arch import feature_extractor as ema_feature_extractor
from .ema_vfi_arch import MultiScaleFlow as EMAMultiScaleFlow
from .sgm_vfi_arch import feature_extractor as sgm_feature_extractor
from .sgm_vfi_arch import MultiScaleFlow as SGMMultiScaleFlow
from .utils.padder import InputPadder

logger = logging.getLogger("Tween")


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


# ---------------------------------------------------------------------------
# SGM-VFI model wrapper
# ---------------------------------------------------------------------------

def _sgm_init_model_config(F=16, W=7, depth=[2, 2, 2, 4], num_key_points=0.5):
    """Build SGM-VFI model config dicts (backbone + multiscale)."""
    return {
        'embed_dims': [F, 2*F, 4*F, 8*F],
        'num_heads': [8*F//32],
        'mlp_ratios': [4],
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'depths': depth,
        'window_sizes': [W]
    }, {
        'embed_dims': [F, 2*F, 4*F, 8*F],
        'motion_dims': [0, 0, 0, 8*F//depth[-1]],
        'depths': depth,
        'scales': [8],
        'hidden_dims': [4*F],
        'c': F,
        'num_key_points': num_key_points,
    }


def _sgm_detect_variant(filename):
    """Auto-detect SGM-VFI model variant from filename.

    Returns (F, depth).
    Default is small (F=16) since the primary checkpoint (ours-1-2-points)
    is a small model. Only detect base when "base" is in the filename.
    """
    name = filename.lower()
    is_base = "base" in name
    if is_base:
        return 32, [2, 2, 2, 6]
    else:
        return 16, [2, 2, 2, 4]


class SGMVFIModel:
    """Clean inference wrapper around SGM-VFI for ComfyUI integration."""

    def __init__(self, checkpoint_path, variant="auto", num_key_points=0.5, tta=False, device="cpu"):
        import os
        filename = os.path.basename(checkpoint_path)

        if variant == "auto":
            F_dim, depth = _sgm_detect_variant(filename)
        elif variant == "small":
            F_dim, depth = 16, [2, 2, 2, 4]
        else:  # base
            F_dim, depth = 32, [2, 2, 2, 6]

        self.tta = tta
        self.device = device
        self.variant_name = "small" if F_dim == 16 else "base"

        backbone_cfg, multiscale_cfg = _sgm_init_model_config(
            F=F_dim, depth=depth, num_key_points=num_key_points)
        backbone = sgm_feature_extractor(**backbone_cfg)
        self.model = SGMMultiScaleFlow(backbone, **multiscale_cfg)
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

        self.model.load_state_dict(cleaned, strict=False)

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
            _, _, _, preds, _ = self.model(input_batch, timestep=timestep)
            return (preds[:B] + preds[B:].flip(2).flip(3)) / 2.
        else:
            _, _, _, pred, _ = self.model(imgs, timestep=timestep)
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


# ---------------------------------------------------------------------------
# GIMM-VFI model wrapper
# ---------------------------------------------------------------------------

class GIMMVFIModel:
    """Clean inference wrapper around GIMM-VFI for ComfyUI integration.

    Supports two modes:
    - interpolate_batch(): standard single-midpoint interface (compatible with
      recursive _interpolate_frames machinery used by other models)
    - interpolate_multi(): GIMM-VFI's unique single-pass mode, generates all
      N-1 intermediate frames between each pair in one forward pass
    """

    def __init__(self, checkpoint_path, flow_checkpoint_path, variant="auto",
                 ds_factor=1.0, device="cpu"):
        import os
        import yaml
        from omegaconf import OmegaConf
        from .gimm_vfi_arch import (
            GIMMVFI_R, GIMMVFI_F, GIMMVFIConfig,
            GIMM_RAFT, GIMM_FlowFormer, gimm_get_flowformer_cfg,
            GIMMInputPadder, GIMMRaftArgs, easydict_to_dict,
        )
        import comfy.utils

        self.ds_factor = ds_factor
        self.device = device
        self._InputPadder = GIMMInputPadder

        filename = os.path.basename(checkpoint_path).lower()

        # Detect variant from filename
        if variant == "auto":
            self.is_flowformer = "gimmvfi_f" in filename
        else:
            self.is_flowformer = (variant == "flowformer")

        self.variant_name = "flowformer" if self.is_flowformer else "raft"

        # Load config
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if self.is_flowformer:
            config_path = os.path.join(script_dir, "gimm_vfi_arch", "configs", "gimmvfi_f_arb.yaml")
        else:
            config_path = os.path.join(script_dir, "gimm_vfi_arch", "configs", "gimmvfi_r_arb.yaml")

        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = easydict_to_dict(config)
        config = OmegaConf.create(config)
        arch_defaults = GIMMVFIConfig.create(config.arch)
        config = OmegaConf.merge(arch_defaults, config.arch)

        # Build model + flow estimator
        dtype = torch.float32

        if self.is_flowformer:
            self.model = GIMMVFI_F(dtype, config)
            cfg = gimm_get_flowformer_cfg()
            flow_estimator = GIMM_FlowFormer(cfg.latentcostformer)
            flow_sd = comfy.utils.load_torch_file(flow_checkpoint_path)
            flow_estimator.load_state_dict(flow_sd, strict=True)
        else:
            self.model = GIMMVFI_R(dtype, config)
            raft_args = GIMMRaftArgs(small=False, mixed_precision=False, alternate_corr=False)
            flow_estimator = GIMM_RAFT(raft_args)
            flow_sd = comfy.utils.load_torch_file(flow_checkpoint_path)
            flow_estimator.load_state_dict(flow_sd, strict=True)

        # Load main model weights
        sd = comfy.utils.load_torch_file(checkpoint_path)
        self.model.load_state_dict(sd, strict=False)

        self.model.flow_estimator = flow_estimator
        self.model.eval()

    def to(self, device):
        """Move model to device (returns self for chaining)."""
        self.device = device if isinstance(device, str) else str(device)
        self.model.to(device)
        return self

    @torch.no_grad()
    def interpolate_batch(self, frames0, frames1, time_step=0.5):
        """Interpolate a single midpoint frame per pair (standard interface).

        Args:
            frames0: [B, C, H, W] tensor, float32, range [0, 1]
            frames1: [B, C, H, W] tensor, float32, range [0, 1]
            time_step: float in (0, 1)

        Returns:
            Interpolated frames as [B, C, H, W] tensor, float32, clamped to [0, 1]
        """
        device = next(self.model.parameters()).device
        results = []

        for i in range(frames0.shape[0]):
            I0 = frames0[i:i+1].to(device)
            I2 = frames1[i:i+1].to(device)

            padder = self._InputPadder(I0.shape, 32)
            I0_p, I2_p = padder.pad(I0, I2)

            xs = torch.cat((I0_p.unsqueeze(2), I2_p.unsqueeze(2)), dim=2)
            batch_size = xs.shape[0]
            s_shape = xs.shape[-2:]

            coord_inputs = [(
                self.model.sample_coord_input(
                    batch_size, s_shape, [time_step],
                    device=xs.device, upsample_ratio=self.ds_factor,
                ),
                None,
            )]
            timesteps = [
                time_step * torch.ones(xs.shape[0]).to(xs.device)
            ]

            all_outputs = self.model(xs, coord_inputs, t=timesteps, ds_factor=self.ds_factor)
            pred = padder.unpad(all_outputs["imgt_pred"][0])
            results.append(torch.clamp(pred, 0, 1))

        return torch.cat(results, dim=0)

    @torch.no_grad()
    def interpolate_multi(self, frame0, frame1, num_intermediates):
        """Generate all intermediate frames between a pair in one forward pass.

        This is GIMM-VFI's unique capability -- arbitrary timestep interpolation
        without recursive 2x passes.

        Args:
            frame0: [1, C, H, W] tensor, float32, range [0, 1]
            frame1: [1, C, H, W] tensor, float32, range [0, 1]
            num_intermediates: int, number of intermediate frames to generate

        Returns:
            List of [1, C, H, W] tensors, float32, clamped to [0, 1]
        """
        device = next(self.model.parameters()).device
        I0 = frame0.to(device)
        I2 = frame1.to(device)

        padder = self._InputPadder(I0.shape, 32)
        I0_p, I2_p = padder.pad(I0, I2)

        xs = torch.cat((I0_p.unsqueeze(2), I2_p.unsqueeze(2)), dim=2)
        batch_size = xs.shape[0]
        s_shape = xs.shape[-2:]
        interp_factor = num_intermediates + 1

        coord_inputs = [
            (
                self.model.sample_coord_input(
                    batch_size, s_shape,
                    [1.0 / interp_factor * i],
                    device=xs.device,
                    upsample_ratio=self.ds_factor,
                ),
                None,
            )
            for i in range(1, interp_factor)
        ]
        timesteps = [
            i * 1.0 / interp_factor * torch.ones(xs.shape[0]).to(xs.device)
            for i in range(1, interp_factor)
        ]

        all_outputs = self.model(xs, coord_inputs, t=timesteps, ds_factor=self.ds_factor)

        results = []
        for pred in all_outputs["imgt_pred"]:
            unpadded = padder.unpad(pred)
            results.append(torch.clamp(unpadded, 0, 1))

        return results
