import json

import numpy as np
import pytest

from qildo_beats.audio import extract_features
from qildo_beats.grid import Beat
from qildo_beats.local_tempo import tempo_curves, variable_path
from qildo_beats.ranker import GridRanker, checkpoint_digest, load_ranker
from qildo_beats.selection import FEATURE_NAMES, Hypothesis
from qildo_beats.tracker import TrackerConfig, predict
from scripts.prepare_aam import parse_beats
from tests.audio_cases import make_song


def test_local_tempo_retains_both_sides_of_a_sustained_change():
    first = np.arange(20, 1700, 60)
    second = np.arange(first[-1] + 35, 3300, 35)
    beats = np.concatenate((first, second))
    pulse = np.zeros(3350)
    pulse[beats] = 1
    curves = tempo_curves(pulse)
    paths = [
        variable_path(pulse, curve.periods, beats[0], beats[-1]) for curve in curves
    ]
    assert any(
        abs(len(path) - len(beats)) <= 2
        and np.median(np.diff(path[path < 1400])) == 60
        and np.median(np.diff(path[path > 2000])) == 35
        and np.isin(beats, path).mean() >= 0.96
        for path in paths
    )


def test_ranker_compares_grids_only_with_matching_checkpoint(tmp_path):
    checkpoint = tmp_path / "cnn.pt"
    checkpoint.write_bytes(b"weights")
    profile = tmp_path / "ranker.json"
    profile.write_text(
        json.dumps(
            {
                "version": 1,
                "checkpoint_sha256": checkpoint_digest(checkpoint),
                "features": FEATURE_NAMES,
                "weights": [1.0] * len(FEATURE_NAMES),
            }
        )
    )
    ranker = load_ranker(profile, checkpoint)
    a, b = [Beat(0.1, 1)], [Beat(0.2, 1)]
    empty = np.zeros(len(FEATURE_NAMES))
    assert ranker.best([Hypothesis(a, empty), Hypothesis(b, empty + 1)]) == b
    assert ranker.best([]) is None
    checkpoint.write_bytes(b"other weights")
    with pytest.raises(ValueError, match="исходный CNN-чекпойнт"):
        load_ranker(profile, checkpoint)


def test_aam_annotation_merges_overlapping_sections_and_checks_conflicts():
    data = b"@data\n0.50,2,2,0\n0.250001,2,1,0\n0.25,2,1,0\n1.0,2,3,0\n"
    assert parse_beats(data, 1) == [Beat(0.25, 1), Beat(0.5, 2)]
    with pytest.raises(ValueError, match="Несовместимые номера"):
        parse_beats(data + b"0.250002,2,2,0\n", 1)


def test_mixed_ranker_retains_acoustic_grid_when_model_has_no_anchors(tmp_path):
    audio = tmp_path / "fallback.wav"
    make_song(audio, seconds=12, soft_intro=True)
    features = extract_features(audio)
    config = TrackerConfig(decoder="joint")
    zeros = np.zeros((2, len(features.log_mel)))
    ranker = GridRanker(np.zeros(len(FEATURE_NAMES)), include_acoustic=True)
    expected = predict(features, config)
    assert len(expected) > 8
    assert predict(features, config, zeros, ranker) == expected
