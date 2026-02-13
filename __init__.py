import subprocess
import sys
import logging

logger = logging.getLogger("Tween")


def _auto_install_deps():
    """Auto-install missing dependencies on first load."""
    # gdown
    try:
        import gdown  # noqa: F401
    except ImportError:
        logger.info("[Tween] Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

    # timm (required for EMA-VFI's MotionFormer backbone)
    try:
        import timm  # noqa: F401
    except ImportError:
        logger.info("[Tween] Installing timm...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])

    # cupy
    try:
        import cupy  # noqa: F401
    except ImportError:
        try:
            import torch
            major = int(torch.version.cuda.split(".")[0])
            cupy_pkg = f"cupy-cuda{major}x"
            logger.info(f"[Tween] Installing {cupy_pkg} (CUDA {torch.version.cuda})...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", cupy_pkg])
        except Exception as e:
            logger.warning(f"[Tween] Could not auto-install cupy: {e}")

    # GIMM-VFI dependencies
    for pkg in ("omegaconf", "yacs", "easydict", "einops", "huggingface_hub"):
        try:
            __import__(pkg)
        except ImportError:
            logger.info(f"[Tween] Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


_auto_install_deps()

from .nodes import (
    LoadBIMVFIModel, BIMVFIInterpolate, BIMVFISegmentInterpolate, TweenConcatVideos,
    LoadEMAVFIModel, EMAVFIInterpolate, EMAVFISegmentInterpolate,
    LoadSGMVFIModel, SGMVFIInterpolate, SGMVFISegmentInterpolate,
    LoadGIMMVFIModel, GIMMVFIInterpolate, GIMMVFISegmentInterpolate,
)

NODE_CLASS_MAPPINGS = {
    "LoadBIMVFIModel": LoadBIMVFIModel,
    "BIMVFIInterpolate": BIMVFIInterpolate,
    "BIMVFISegmentInterpolate": BIMVFISegmentInterpolate,
    "TweenConcatVideos": TweenConcatVideos,
    "LoadEMAVFIModel": LoadEMAVFIModel,
    "EMAVFIInterpolate": EMAVFIInterpolate,
    "EMAVFISegmentInterpolate": EMAVFISegmentInterpolate,
    "LoadSGMVFIModel": LoadSGMVFIModel,
    "SGMVFIInterpolate": SGMVFIInterpolate,
    "SGMVFISegmentInterpolate": SGMVFISegmentInterpolate,
    "LoadGIMMVFIModel": LoadGIMMVFIModel,
    "GIMMVFIInterpolate": GIMMVFIInterpolate,
    "GIMMVFISegmentInterpolate": GIMMVFISegmentInterpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadBIMVFIModel": "Load BIM-VFI Model",
    "BIMVFIInterpolate": "BIM-VFI Interpolate",
    "BIMVFISegmentInterpolate": "BIM-VFI Segment Interpolate",
    "TweenConcatVideos": "Tween Concat Videos",
    "LoadEMAVFIModel": "Load EMA-VFI Model",
    "EMAVFIInterpolate": "EMA-VFI Interpolate",
    "EMAVFISegmentInterpolate": "EMA-VFI Segment Interpolate",
    "LoadSGMVFIModel": "Load SGM-VFI Model",
    "SGMVFIInterpolate": "SGM-VFI Interpolate",
    "SGMVFISegmentInterpolate": "SGM-VFI Segment Interpolate",
    "LoadGIMMVFIModel": "Load GIMM-VFI Model",
    "GIMMVFIInterpolate": "GIMM-VFI Interpolate",
    "GIMMVFISegmentInterpolate": "GIMM-VFI Segment Interpolate",
}
