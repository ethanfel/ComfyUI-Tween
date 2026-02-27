import subprocess
import sys
import logging

logger = logging.getLogger("Tween")


def _ensure_cupy():
    """Try to auto-install cupy if missing, matching the PyTorch CUDA version."""
    try:
        import cupy  # noqa: F401
        return
    except ImportError:
        pass

    try:
        import torch
        if not torch.cuda.is_available() or not hasattr(torch.version, "cuda") or torch.version.cuda is None:
            logger.warning(
                "[Tween] CUDA not available — cupy cannot be installed. "
                "SGM-VFI and GIMM-VFI require CUDA."
            )
            return
        major = int(torch.version.cuda.split(".")[0])
        cupy_pkg = f"cupy-cuda{major}x"
        logger.info(f"[Tween] Installing {cupy_pkg} (CUDA {torch.version.cuda})...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", cupy_pkg])
    except Exception as e:
        logger.warning(
            f"[Tween] Could not auto-install cupy: {e}\n"
            f"[Tween] SGM-VFI and GIMM-VFI will not work without cupy. Install manually:\n"
            f"[Tween]   pip install cupy-cuda12x  # replace 12 with your CUDA major version"
        )


_ensure_cupy()

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
