"""Гипотезы метрической сетки и оценка их акустической согласованности."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from .audio import FRAME_RATE, Features
from .decoder import activation_peaks, anchor_coverage, pulse_paths
from .grid import Beat
from .local_tempo import tempo_curves, variable_path
from .meter import align_bars
from .periodic import recover_periodic_grid

FEATURE_NAMES = (
    "path_score",
    "beat_coverage",
    "bar_score",
    "neural_beat_support",
    "neural_offbeat_support",
    "neural_peak_coverage",
    "onset_support",
    "onset_peak_coverage",
    "neural_downbeat_contrast",
    "acoustic_downbeat_contrast",
    "neural_downbeat_coverage",
    "interval_jitter",
    "interval_spread",
    "pulse_rate",
    "tempo_prior",
    "first_anchor_gap",
    "last_anchor_gap",
    "meter_change_rate",
    "downbeat_rate",
    "weak_beat_fraction",
    "beat_activation_spread",
    "first_downbeat",
)


@dataclass(frozen=True)
class Hypothesis:
    grid: list[Beat]
    features: np.ndarray


def _contrast(values: np.ndarray, marked: np.ndarray) -> float:
    if not marked.any() or marked.all():
        return 0.0
    return float(values[marked].mean() - values[~marked].mean())


def _pulse_options(pulse: np.ndarray, evidence: np.ndarray, anchors: np.ndarray):
    yield from pulse_paths(pulse, evidence, anchors)
    for curve in tempo_curves(evidence):
        frames = variable_path(pulse, curve.periods, int(anchors[0]), int(anchors[-1]))
        if len(frames) < 2:
            continue
        periods = np.diff(frames)
        deviation = np.log(periods / curve.periods[frames[1:]])
        boundary = abs(frames[0] - anchors[0]) + abs(frames[-1] - anchors[-1])
        period = float(np.median(periods))
        quality = (
            pulse[frames].mean() - 2 * np.mean(deviation**2) - 0.12 * boundary / period
        )
        prior = abs(np.log2(60 * FRAME_RATE / period / 115))
        yield frames, float(quality + 0.65 * curve.evidence - 0.12 * prior)


def _bar_options(
    features: Features,
    pulse: np.ndarray,
    accent: np.ndarray,
    anchors: np.ndarray,
    meters: tuple[int, ...],
    acoustic: list[Beat] | None,
):
    configurations = (
        [meters, *((meter,) for meter in meters)] if len(meters) > 1 else [meters]
    )
    evidence = 0.78 * pulse + 0.22 * accent
    for frames, path_score in _pulse_options(pulse, evidence, anchors):
        for configuration in configurations:
            refined, numbers, bar_score = align_bars(
                frames, pulse, accent, configuration, learned=True
            )
            yield refined, numbers, path_score, bar_score
    if acoustic is not None:
        periodic = recover_periodic_grid(features, meters, acoustic)
        for grid in (acoustic, periodic):
            if not grid or len(grid) < 2:
                continue
            frames = np.rint([event.time * FRAME_RATE for event in grid]).astype(int)
            numbers = np.array([event.beat for event in grid])
            intervals = np.diff(frames)
            period = np.median(intervals)
            path_score = float(
                pulse[frames].mean()
                - 2 * np.mean(np.log(intervals / period) ** 2)
                - 0.12
                * (abs(frames[0] - anchors[0]) + abs(frames[-1] - anchors[-1]))
                / period
            )
            condition = accent[frames] - 0.5 * pulse[frames]
            yield frames, numbers, path_score, 0.45 * _contrast(condition, numbers == 1)


def hypotheses(
    features: Features,
    activations: np.ndarray,
    meters: tuple[int, ...],
    acoustic: list[Beat] | None = None,
) -> list[Hypothesis]:
    pulse, accent = np.where(features.audible[None, :], activations, 0)
    anchors = activation_peaks(pulse, 0.22, 8)
    if len(anchors) < 2:
        return []
    onset_anchors = activation_peaks(features.onset, 0.27, 8)
    downbeat_anchors = activation_peaks(accent, 0.30, 12)
    beat_envelope = maximum_filter1d(pulse, size=11)
    onset_envelope = maximum_filter1d(features.onset, size=11)
    downbeat_envelope = maximum_filter1d(accent, size=13)
    acoustic_accents = maximum_filter1d(features.accent, size=9)
    options = []
    seen = set()
    for refined, numbers, path_score, bar_score in _bar_options(
        features, pulse, accent, anchors, meters, acoustic
    ):
        inside = refined / FRAME_RATE < features.duration
        refined, numbers = refined[inside], numbers[inside]
        if len(refined) < 2:
            continue
        key = (refined.tobytes(), numbers.tobytes())
        if key in seen:
            continue
        seen.add(key)
        intervals = np.diff(refined)
        median_interval = np.median(intervals)
        marked = numbers == 1
        midpoint = (refined[:-1] + refined[1:]) // 2
        bar_lengths = np.diff(np.flatnonzero(marked))
        conditional = downbeat_envelope[refined] - 0.5 * beat_envelope[refined]
        beat_values = beat_envelope[refined]
        measure = np.array(
            [
                path_score,
                anchor_coverage(refined, anchors, pulse[anchors]),
                bar_score,
                beat_values.mean(),
                beat_envelope[midpoint].mean(),
                anchor_coverage(refined, anchors, np.ones(len(anchors))),
                onset_envelope[refined].mean(),
                anchor_coverage(refined, onset_anchors, features.onset[onset_anchors])
                if len(onset_anchors)
                else 0,
                _contrast(conditional, marked),
                _contrast(acoustic_accents[refined], marked),
                anchor_coverage(
                    refined[marked], downbeat_anchors, accent[downbeat_anchors]
                )
                if marked.any() and len(downbeat_anchors)
                else 0,
                np.mean(np.abs(np.diff(np.log(intervals))))
                if len(intervals) > 1
                else 0,
                np.std(np.log(intervals)),
                np.log2(60 * FRAME_RATE / median_interval / 115),
                abs(np.log2(60 * FRAME_RATE / median_interval / 115)),
                abs(refined[0] - anchors[0]) / median_interval,
                abs(refined[-1] - anchors[-1]) / median_interval,
                np.mean(np.diff(bar_lengths) != 0) if len(bar_lengths) > 1 else 0,
                marked.mean(),
                np.mean(beat_values < 0.20),
                np.std(beat_values),
                float(marked[0]) * np.exp(-refined[0] / 25),
            ]
        )
        grid = [
            Beat(float(frame) / FRAME_RATE, int(number))
            for frame, number in zip(refined, numbers, strict=True)
        ]
        options.append(Hypothesis(grid, measure))
    return options
