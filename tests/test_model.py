import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qildo_beats.audio import FRAME_RATE, MEL_BANDS  # noqa: E402
from qildo_beats.grid import Beat  # noqa: E402
from qildo_beats.model import (  # noqa: E402
    CHECKPOINT_VERSION,
    CONTEXT,
    build_model,
    left_context,
    load_predictor,
    select_device,
)
from train import frame_labels  # noqa: E402


def test_two_training_targets():
    labels = frame_labels([Beat(0.2, 3), Beat(0.5, 1)], 64)
    assert labels[:, 20].tolist() == [1, 0]
    assert labels[:, 50].tolist() == [1, 1]


@pytest.mark.parametrize("architecture", ["cnn", "residual", "hierarchical"])
def test_chunked_predictions_match_full_pass(tmp_path, architecture):
    torch.manual_seed(1)
    select_device("cpu")
    model = build_model(architecture, 4).eval()
    path = tmp_path / "test.pt"
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "frame_rate": FRAME_RATE,
            "mel_bands": MEL_BANDS,
            "channels": 4,
            "architecture": architecture,
            "state": model.state_dict(),
        },
        path,
    )
    mel = np.random.default_rng(4).random((3250, MEL_BANDS), dtype=np.float32)
    with torch.inference_mode():
        left = left_context(architecture)
        padded = np.pad(mel, ((left, CONTEXT), (0, 0)))
        expected = (
            model(torch.from_numpy(padded.T[None]))
            .sigmoid()[0, :, left:-CONTEXT]
            .numpy()
        )
        if architecture == "hierarchical":
            expected[1] *= expected[0]
    actual = load_predictor(path, "cpu")(mel)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
