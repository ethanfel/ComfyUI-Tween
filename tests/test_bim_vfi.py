import importlib
from pathlib import Path
import sys
import types

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "_tween_bim_tests"


def _load_inference_module():
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(REPO_ROOT)]
    package.__package__ = PACKAGE_NAME
    sys.modules.setdefault(PACKAGE_NAME, package)
    return importlib.import_module(f"{PACKAGE_NAME}.inference")


def test_auto_pyramid_levels_match_official_video_inference():
    model_class = _load_inference_module().BiMVFIModel
    model = model_class.__new__(model_class)
    model.auto_pyr_level = True

    assert model._get_pyr_level(240) == 5
    assert model._get_pyr_level(539) == 5
    assert model._get_pyr_level(720) == 5
    assert model._get_pyr_level(1079) == 5
    assert model._get_pyr_level(1080) == 6
    assert model._get_pyr_level(2159) == 6
    assert model._get_pyr_level(2160) == 7


def test_manual_pyramid_level_remains_available():
    model_class = _load_inference_module().BiMVFIModel
    model = model_class.__new__(model_class)
    model.auto_pyr_level = False
    model.pyr_level = 3

    assert model._get_pyr_level(720) == 3


def test_artifact_safe_mode_removes_only_rgb_refinement_residual():
    module = _load_inference_module()
    official = module.BiMVFI(pyr_level=3, feat_channels=1)
    artifact_safe = module.BiMVFI(
        pyr_level=3,
        feat_channels=1,
        artifact_safe_mode=True,
    )

    assert official.sn.use_rgb_refine_residual is True
    assert artifact_safe.sn.use_rgb_refine_residual is False

    official_keys = official.state_dict().keys()
    artifact_safe_keys = artifact_safe.state_dict().keys()
    assert official_keys == artifact_safe_keys

    warped0 = torch.full((1, 3, 2, 2), 0.2)
    warped1 = torch.full((1, 3, 2, 2), 0.8)
    mask = torch.full((1, 1, 2, 2), 0.25)
    residual = torch.full((1, 3, 2, 2), 0.1)
    blend = warped0 * mask + warped1 * (1 - mask)

    assert torch.equal(
        official.sn.merge_warped_images(warped0, warped1, mask, residual),
        blend + residual,
    )
    assert torch.equal(
        artifact_safe.sn.merge_warped_images(warped0, warped1, mask, residual),
        blend,
    )


def test_demo_exposes_artifact_safe_mode_without_enabling_it():
    import json

    workflow_path = REPO_ROOT / "example_workflows" / "tween_speed_bim_model_lab.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    loader = next(node for node in workflow["nodes"] if node["type"] == "LoadBIMVFIModel")

    assert loader["widgets_values"] == ["bim_vfi.pth", True, 3, False]
