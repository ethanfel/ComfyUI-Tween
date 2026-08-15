import torch
import pytest

from ldf_backend import LDFVFIModel


class _RecordingConditionalVAE:
    spatial_compression_ratio = 8

    def __init__(self):
        self.decode_shapes = None

    def decode(self, latent, condition, mask):
        self.decode_shapes = (
            tuple(latent.shape),
            tuple(condition.shape),
            tuple(mask.shape),
        )
        # A small real result lets _decode finish without materializing the
        # full 720p tensors used for the shape-only inputs above.
        return torch.zeros(1, 3, 40, 1, 1)


class _ShapeOnlyLDF(LDFVFIModel):
    """Exercise sequence chunking without loading the multi-GB checkpoint."""

    def __init__(self):
        self.decode_tile_counts = []

    def _prepare_condition(self, frames, mask, device):
        assert frames.shape[0] == int(mask.sum())
        dense = torch.empty(1, 3, self.TRAIN_FRAMES, 1, 1)
        dense_mask = torch.empty(1, 1, self.TRAIN_FRAMES, 1, 1)
        latent = torch.empty(
            1, self.TRAIN_FRAMES // self.TILE_TIME, 1, 1, 1, 1, 1, 1
        )
        return dense, dense_mask, latent, latent

    def _sample_free(self, condition, encoded_mask, schedule, device, progress):
        return condition

    def _sample_between(
        self, previous, following, condition, encoded_mask,
        schedule, t_cond, device, progress,
    ):
        return condition[:, self.CONDITION_TILES:-self.CONDITION_TILES]

    def _sample_tail(
        self, previous, condition, encoded_mask, schedule,
        t_cond, device, progress,
    ):
        return condition[:, self.CONDITION_TILES:]

    def _decode(self, latent, dense, dense_mask, height, width):
        temporal_tiles = latent.shape[1]
        assert dense.shape[2] == temporal_tiles * self.TILE_TIME
        assert dense_mask.shape[2] == dense.shape[2]
        self.decode_tile_counts.append(temporal_tiles)
        return torch.empty(dense.shape[2], 3, height, width)


def test_decode_tiles_40_frame_720p_condition_for_conditional_vae():
    """Regression for the 5-D condition passed to upstream VAE.decode."""
    model = LDFVFIModel.__new__(LDFVFIModel)
    model.vae = _RecordingConditionalVAE()

    # Representative first LDF output block at 720p. Meta tensors exercise
    # exact shape transforms without allocating hundreds of MB in the test.
    latent = torch.empty((1, 2, 4, 7, 4, 5, 24, 24), device="meta")
    condition = torch.empty((1, 3, 40, 720, 1280), device="meta")
    mask = torch.empty((1, 1, 40, 720, 1280), device="meta")

    result = model._decode(latent, condition, mask, height=720, width=1280)

    assert model.vae.decode_shapes == (
        (1, 2, 4, 5, 96, 168),
        (1, 2, 3, 20, 768, 1344),
        (1, 2, 1, 20, 768, 1344),
    )
    assert result.shape == (40, 3, 1, 1)


def test_decode_rejects_condition_length_that_cannot_tile():
    model = LDFVFIModel.__new__(LDFVFIModel)
    model.vae = _RecordingConditionalVAE()
    latent = torch.empty((1, 2, 1, 1, 4, 5, 1, 1), device="meta")
    condition = torch.empty((1, 3, 39, 8, 8), device="meta")
    mask = torch.empty((1, 1, 39, 8, 8), device="meta")

    try:
        model._decode(latent, condition, mask, height=8, width=8)
    except RuntimeError as error:
        assert "not divisible" in str(error)
    else:
        raise AssertionError("Expected an invalid temporal tile error")


@pytest.mark.parametrize("temporal_factor", [2, 3, 8, 16])
def test_sequence_chunk_conditions_align_for_reported_391_frames(temporal_factor):
    model = _ShapeOnlyLDF()
    source = torch.empty(391, 3, 1, 1)
    chunks = model._interpolate_sequence_impl(
        source=source,
        factor=temporal_factor,
        schedule=torch.tensor([1.0, 0.0]),
        t_cond=0.1,
        device=torch.device("cpu"),
        progress=lambda: None,
    )

    expected_frames = (source.shape[0] - 1) * temporal_factor + 1
    assert sum(chunk.shape[0] for chunk in chunks) >= expected_frames
    assert len(chunks) == model.sampling_block_count(
        source.shape[0], temporal_factor
    )
