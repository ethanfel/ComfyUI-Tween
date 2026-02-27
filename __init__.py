from .nodes import (
    LoadBIMVFIModel, BIMVFIInterpolate, BIMVFISegmentInterpolate, TweenConcatVideos,
    LoadEMAVFIModel, EMAVFIInterpolate, EMAVFISegmentInterpolate,
    LoadSGMVFIModel, SGMVFIInterpolate, SGMVFISegmentInterpolate,
    LoadGIMMVFIModel, GIMMVFIInterpolate, GIMMVFISegmentInterpolate,
    VFIOptimizer,
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
    "VFIOptimizer": VFIOptimizer,
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
    "VFIOptimizer": "VFI Optimizer",
}
