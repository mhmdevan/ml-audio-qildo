import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from qildo_beats.audio import extract_features
from qildo_beats.grid import read_grid
from qildo_beats.metrics import score_track
from qildo_beats.tracker import TrackerConfig, predict
from tests.audio_cases import make_song

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "meter,pickup,soft_intro,change_tempo",
    [
        (4, 0, False, False),
        (3, 0, False, False),
        (4, 2, False, False),
        (4, 0, True, False),
        (4, 0, False, True),
    ],
)
def test_tracking(tmp_path, meter, pickup, soft_intro, change_tempo):
    path = tmp_path / "song.wav"
    reference = make_song(
        path,
        meter=meter,
        pickup=pickup,
        soft_intro=soft_intro,
        change_tempo=change_tempo,
        seconds=20 if change_tempo else 12,
    )
    features = extract_features(path)
    assert np.isfinite(features.log_mel).all()
    result = predict(features, TrackerConfig())
    scores = score_track(reference, result)
    assert scores["beat_f1"] >= 0.95, result
    assert scores["downbeat_f1"] >= 0.90, result
    assert scores["first_correct"], result


def test_cli_silence_and_corrupt_input(tmp_path):
    audio, output = tmp_path / "silence.wav", tmp_path / "result.json"
    wavfile.write(audio, 44_100, np.zeros((33, 2), dtype=np.int16))
    command = [sys.executable, str(ROOT / "beats.py"), str(audio), str(output)]
    subprocess.run(command, check=True)
    assert read_grid(output) == []
    audio.write_bytes(b"broken")
    output.unlink()
    failed = subprocess.run(command, capture_output=True)
    assert failed.returncode != 0
    assert b"Traceback" not in failed.stderr
    assert not output.exists()


def test_half_tempo_does_not_win_on_long_tempo_change(tmp_path):
    path = tmp_path / "long.wav"
    reference = make_song(path, seconds=240, soft_intro=True, change_tempo=True)
    result = predict(extract_features(path), TrackerConfig())
    scores = score_track(reference, result)
    assert scores["beat_f1"] >= 0.97
    assert scores["tempo_error_bars"] == 0


def test_cli_writes_song_grid(tmp_path):
    audio, output = tmp_path / "song.wav", tmp_path / "result.json"
    reference = make_song(audio, pickup=2)
    subprocess.run(
        [sys.executable, str(ROOT / "beats.py"), str(audio), str(output)], check=True
    )
    assert score_track(reference, read_grid(output))["beat_f1"] == 1
    assert isinstance(json.loads(output.read_text()), list)
