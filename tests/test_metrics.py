import json

from qildo_beats.grid import Beat
from qildo_beats.metrics import score_track, summarize


def grid(step=0.5, length=48):
    return [Beat(i * step, i % 4 + 1) for i in range(length)]


def test_keeps_first_five_seconds_and_serializes_report():
    reference = grid()
    perfect = score_track(reference, reference)
    assert perfect["beat_f1"] == perfect["downbeat_f1"] == perfect["cmlt"] == 1
    missing = score_track(reference, reference[2:])
    assert missing["beat_f1"] < 1
    assert missing["first_correct"] is False
    assert missing["intro_matched"] == 6
    json.dumps({"summary": summarize([missing]), "rows": [missing]}, allow_nan=False)


def test_diagnoses_double_half_and_shifted_phase():
    reference = grid()
    doubled = score_track(reference, grid(0.25, 96))
    halved = score_track(reference, grid(1.0, 24))
    shifted = score_track(
        reference, [Beat(i.time, (i.beat - 2) % 4 + 1) for i in reference]
    )
    assert doubled["tempo_error_bars"] == doubled["stable_bars"] > 0
    assert halved["tempo_error_bars"] == halved["stable_bars"] > 0
    assert doubled["amlt"] > doubled["cmlt"]
    assert shifted["beat_f1"] == 1
    assert shifted["downbeat_f1"] == 0
    assert shifted["phase_error_bars"] == shifted["phase_eligible_bars"] > 0


def test_threshold_and_empty_grid():
    assert score_track([Beat(0, 1)], [Beat(0.069, 1)])["beat_f1"] == 1
    assert score_track([Beat(0, 1)], [Beat(0.071, 1)])["beat_f1"] == 0
    assert score_track([Beat(0, 1)], [])["beat_f1"] == 0
    silent = score_track([], [Beat(1, 1)])
    assert silent["beat_f1"] is None
    assert silent["early_false_clicks"] == 1
