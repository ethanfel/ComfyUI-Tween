import subprocess
import sys
import logging

logger = logging.getLogger("BIM-VFI")


def _auto_install_deps():
    """Auto-install missing dependencies on first load."""
    # gdown
    try:
        import gdown  # noqa: F401
    except ImportError:
        logger.info("[BIM-VFI] Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

    # timm (required for EMA-VFI's MotionFormer backbone)
    try:
        import timm  # noqa: F401
    except ImportError:
        logger.info("[BIM-VFI] Installing timm...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])

    # cupy
    try:
        import cupy  # noqa: F401
    except ImportError:
        try:
            import torch
            major = int(torch.version.cuda.split(".")[0])
            cupy_pkg = f"cupy-cuda{major}x"
            logger.info(f"[BIM-VFI] Installing {cupy_pkg} (CUDA {torch.version.cuda})...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", cupy_pkg])
        except Exception as e:
            logger.warning(f"[BIM-VFI] Could not auto-install cupy: {e}")


_auto_install_deps()

from .nodes import (
    LoadBIMVFIModel, BIMVFIInterpolate, BIMVFISegmentInterpolate, BIMVFIConcatVideos,
    LoadEMAVFIModel, EMAVFIInterpolate, EMAVFISegmentInterpolate,
    LoadSGMVFIModel, SGMVFIInterpolate, SGMVFISegmentInterpolate,
)

NODE_CLASS_MAPPINGS = {
    "LoadBIMVFIModel": LoadBIMVFIModel,
    "BIMVFIInterpolate": BIMVFIInterpolate,
    "BIMVFISegmentInterpolate": BIMVFISegmentInterpolate,
    "BIMVFIConcatVideos": BIMVFIConcatVideos,
    "LoadEMAVFIModel": LoadEMAVFIModel,
    "EMAVFIInterpolate": EMAVFIInterpolate,
    "EMAVFISegmentInterpolate": EMAVFISegmentInterpolate,
    "LoadSGMVFIModel": LoadSGMVFIModel,
    "SGMVFIInterpolate": SGMVFIInterpolate,
    "SGMVFISegmentInterpolate": SGMVFISegmentInterpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadBIMVFIModel": "Load BIM-VFI Model",
    "BIMVFIInterpolate": "BIM-VFI Interpolate",
    "BIMVFISegmentInterpolate": "BIM-VFI Segment Interpolate",
    "BIMVFIConcatVideos": "BIM-VFI Concat Videos",
    "LoadEMAVFIModel": "Load EMA-VFI Model",
    "EMAVFIInterpolate": "EMA-VFI Interpolate",
    "EMAVFISegmentInterpolate": "EMA-VFI Segment Interpolate",
    "LoadSGMVFIModel": "Load SGM-VFI Model",
    "SGMVFIInterpolate": "SGM-VFI Interpolate",
    "SGMVFISegmentInterpolate": "SGM-VFI Segment Interpolate",
}
