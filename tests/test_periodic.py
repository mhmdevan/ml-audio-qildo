from pathlib import Path

from qildo_beats.audio import extract_features
from qildo_beats.grid import Beat
from qildo_beats.metrics import score_track
from qildo_beats.periodic import recover_periodic_grid
from tests.audio_cases import make_song


def test_recovers_beats_from_a_dropped_pulse_level(tmp_path: Path):
    path = tmp_path / "periodic.wav"
    reference = make_song(path, seconds=64)
    half_grid = [
        Beat(event.time, index % 4 + 1) for index, event in enumerate(reference[::2])
    ]
    predicted = recover_periodic_grid(extract_features(path), (4, 3), half_grid)
    scores = score_track(reference, predicted)
    assert scores["beat_f1"] > 0.97
    assert scores["downbeat_f1"] > 0.95
    assert scores["first_correct"]
