"""Метрики MIR и диагностика ошибок метронома на полном треке."""

import warnings
from collections import Counter
from statistics import mean

import numpy as np
from mir_eval.beat import continuity, f_measure
from mir_eval.util import match_events

from .grid import Beat

TOLERANCE = 0.070
INTRO_BEATS = 8


def _runs(values: list[int], minimum: int) -> list[int]:
    indices = []
    length = previous = 0
    for end, value in enumerate([*values, 0]):
        if value and value == previous:
            length += 1
            continue
        if length >= minimum:
            indices.extend(range(end - length, end))
        length = int(value != 0)
        previous = value
    return indices


def _bar_errors(
    reference: list[Beat],
    times: np.ndarray,
    predicted: list[Beat],
    estimates: np.ndarray,
    matched: dict[int, int],
) -> dict:
    boundaries = [index for index, event in enumerate(reference) if event.beat == 1]
    tempo_runs = []
    phase_runs = []
    stable = 0
    phase_eligible = []
    estimated_intervals = np.diff(estimates)
    estimated_midpoints = (estimates[:-1] + estimates[1:]) / 2
    for bar, (first, stop) in enumerate(zip(boundaries, boundaries[1:])):
        length = stop - first
        intervals = np.diff(times[first : stop + 1])
        valid_bar = [event.beat for event in reference[first:stop]] == list(
            range(1, length + 1)
        )
        regular = np.max(np.abs(intervals / np.median(intervals) - 1)) <= 0.12
        tempo = 0
        if valid_bar and regular:
            stable += 1
            left, right = np.searchsorted(
                estimated_midpoints, [times[first], times[stop]]
            )
            if right > left:
                ratio = np.median(intervals) / np.median(
                    estimated_intervals[left:right]
                )
                tempo = (
                    2
                    if abs(ratio - 2) <= 0.25
                    else -2
                    if abs(ratio - 0.5) <= 0.15
                    else 0
                )
        tempo_runs.append(tempo)
        phase = 0
        indices = [matched.get(index) for index in range(first, stop)]
        left = np.searchsorted(estimates, times[first] - TOLERANCE)
        right = np.searchsorted(estimates, times[stop] - TOLERANCE)
        if (
            valid_bar
            and length >= 2
            and right - left == length
            and all(index is not None for index in indices)
        ):
            phase_eligible.append(bar)
            accents = [
                position
                for position, index in enumerate(indices)
                if predicted[index].beat == 1
            ]
            if len(accents) == 1:
                phase = 1 if accents[0] == 1 else -1 if accents[0] == length - 1 else 0
        phase_runs.append(phase)
    phase_errors = _runs(phase_runs, 2)
    return {
        "stable_bars": stable,
        "tempo_error_bars": len(_runs(tempo_runs, 2)),
        "phase_eligible_bars": len(phase_eligible),
        "phase_error_bars": len(phase_errors),
        "phase_eligible_indices": phase_eligible,
        "phase_error_indices": phase_errors,
    }


def score_track(reference: list[Beat], predicted: list[Beat]) -> dict:
    times = np.array([event.time for event in reference], dtype=float)
    estimates = np.array([event.time for event in predicted], dtype=float)
    truth_downbeats = times[
        np.array([event.beat == 1 for event in reference], dtype=bool)
    ]
    pred_downbeats = estimates[
        np.array([event.beat == 1 for event in predicted], dtype=bool)
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="(Reference|Estimated) beats are empty."
        )
        beat_f1 = f_measure(times, estimates, TOLERANCE) if len(times) else None
        downbeat_f1 = (
            f_measure(truth_downbeats, pred_downbeats, TOLERANCE)
            if len(truth_downbeats)
            else None
        )
        scores = continuity(times, estimates) if len(times) >= 2 else (None,) * 4
    cmlc, cmlt, amlc, amlt = (
        float(score) if score is not None else None for score in scores
    )
    matched = dict(match_events(times, estimates, TOLERANCE))
    first_error = (
        abs(estimates[0] - times[0]) if len(times) and len(estimates) else None
    )
    intro_total = min(INTRO_BEATS, len(times))
    matched_errors = [
        abs(times[ref] - estimates[pred]) for ref, pred in matched.items()
    ]
    return {
        "beat_f1": beat_f1,
        "downbeat_f1": downbeat_f1,
        "cmlc": cmlc,
        "cmlt": cmlt,
        "amlc": amlc,
        "amlt": amlt,
        "reference_beats": len(times),
        "predicted_beats": len(estimates),
        "first_correct": bool(first_error is not None and first_error <= TOLERANCE)
        if len(times)
        else None,
        "first_absolute_error_seconds": float(first_error)
        if first_error is not None
        else None,
        "intro_total": intro_total,
        "intro_matched": sum(int(index < intro_total) for index in matched),
        "early_false_clicks": int(np.sum(estimates < times[0] - TOLERANCE))
        if len(times)
        else len(estimates),
        "median_matched_error_ms": float(np.median(matched_errors) * 1000)
        if matched_errors
        else None,
        **_bar_errors(reference, times, predicted, estimates, matched),
    }


def summarize(rows: list[dict]) -> dict:
    averages = {}
    for key in (
        "beat_f1",
        "downbeat_f1",
        "cmlc",
        "cmlt",
        "amlc",
        "amlt",
        "median_matched_error_ms",
    ):
        values = [float(row[key]) for row in rows if row[key] is not None]
        averages[key] = mean(values) if values else None
    counts = Counter()
    for row in rows:
        for key in (
            "intro_total",
            "intro_matched",
            "early_false_clicks",
            "stable_bars",
            "tempo_error_bars",
            "phase_eligible_bars",
            "phase_error_bars",
        ):
            counts[key] += row[key]
        counts["first_eligible_tracks"] += row["first_correct"] is not None
        counts["first_error_tracks"] += row["first_correct"] is False
        counts["reference_empty_tracks"] += row["reference_beats"] == 0
    return {"tracks": len(rows), **averages, **counts}
