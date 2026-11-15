"""Поиск пульса и фазы такта по всей песне."""

from collections.abc import Iterator

import numpy as np
from scipy import signal

from .audio import FRAME_RATE
from .grid import Beat

MIN_BPM = 45
MAX_BPM = 220
MIN_PERIOD = round(60 * FRAME_RATE / MAX_BPM)
MAX_PERIOD = round(60 * FRAME_RATE / MIN_BPM)


def activation_peaks(values: np.ndarray, minimum: float, distance: int) -> np.ndarray:
    # Дополняем края, чтобы атака в нулевом кадре могла стать кандидатом.
    peaks, _ = signal.find_peaks(np.pad(values, 1), height=minimum, distance=distance)
    return peaks - 1


def _periods(pulse: np.ndarray) -> list[tuple[int, float]]:
    centered = pulse - pulse.mean()
    correlation = signal.correlate(centered, centered, method="fft")[len(pulse) - 1 :]
    if correlation[0] <= 1e-9:
        return []
    correlation /= correlation[0]
    region = correlation[MIN_PERIOD : MAX_PERIOD + 1]
    if not len(region):
        return []
    peaks = activation_peaks(region, 0.025, 5)
    if not len(peaks):
        peaks = np.array([region.argmax()])
    strongest = sorted(peaks, key=lambda index: region[index], reverse=True)[:4]
    candidates = {int(index + MIN_PERIOD) for index in strongest}
    for period in tuple(candidates):
        for related in (round(period / 2), 2 * period):
            if MIN_PERIOD <= related <= MAX_PERIOD and related < len(correlation):
                candidates.add(related)
    return [(period, float(correlation[period])) for period in sorted(candidates)]


def _path(
    pulse: np.ndarray, period: int, first: int, last: int
) -> tuple[np.ndarray, float]:
    start = max(0, first - 3)
    stop = min(len(pulse), last + 4)
    evidence = pulse[start:stop]
    shortest = max(MIN_PERIOD, round(period * 0.72))
    longest = min(MAX_PERIOD, round(period * 1.38))
    steps = np.arange(shortest, longest + 1)
    penalties = 6.0 * np.log(steps / period) ** 2
    previous = np.full(len(evidence), -1, dtype=np.int32)
    scores = np.full(len(evidence), -np.inf)
    # Вознаграждение за переход и локальный штраф удерживают пульс через слабые атаки.
    for frame, activation in enumerate(evidence):
        if frame < period:
            scores[frame] = activation - 0.30 * frame / period
        available = min(frame - shortest + 1, len(steps))
        if available > 0:
            indices = frame - steps[:available]
            options = scores[indices] - penalties[:available]
            best = int(options.argmax())
            score = options[best] + activation + 0.08
            if score > scores[frame]:
                scores[frame] = score
                previous[frame] = indices[best]
    end = max(0, len(evidence) - period)
    frame = end + int(np.argmax(scores[end:]))
    path = []
    while frame >= 0:
        path.append(frame + start)
        frame = int(previous[frame])
    frames = np.asarray(path[::-1])
    regularity = (
        np.mean(np.log(np.diff(frames) / period) ** 2) if len(frames) > 1 else 0
    )
    boundary_gap = (abs(frames[0] - first) + abs(frames[-1] - last)) / period
    quality = float(pulse[frames].mean() - 2 * regularity - 0.12 * boundary_gap)
    return frames, quality


def track_pulse(pulse: np.ndarray, rhythm: np.ndarray) -> np.ndarray:
    anchors = activation_peaks(pulse, 0.22, 8)
    if len(anchors) < 2:
        return anchors
    return max(
        (
            (frames, score + 0.22 * anchor_coverage(frames, anchors, pulse[anchors]))
            for frames, score in pulse_paths(pulse, rhythm, anchors)
        ),
        key=lambda item: item[1],
        default=(anchors[:1], 0.0),
    )[0]


def anchor_coverage(
    frames: np.ndarray, anchors: np.ndarray, weights: np.ndarray
) -> float:
    right = np.searchsorted(frames, anchors).clip(0, len(frames) - 1)
    left = np.maximum(right - 1, 0)
    distance = np.minimum(abs(anchors - frames[left]), abs(anchors - frames[right]))
    return float(np.average(distance <= 6, weights=weights))


def pulse_paths(
    pulse: np.ndarray, rhythm: np.ndarray, anchors: np.ndarray
) -> Iterator[tuple[np.ndarray, float]]:
    for period, correlation in _periods(rhythm):
        frames, quality = _path(pulse, period, int(anchors[0]), int(anchors[-1]))
        if len(frames) < 2:
            continue
        tempo_prior = abs(np.log2((60 * FRAME_RATE / period) / 115))
        yield frames, quality + 0.65 * correlation - 0.12 * tempo_prior


def pick_pulse(pulse: np.ndarray, threshold: float) -> np.ndarray:
    return activation_peaks(pulse, threshold, MIN_PERIOD)


def number_beats(
    frames: np.ndarray, accent: np.ndarray, meters: tuple[int, ...]
) -> list[int]:
    if not len(frames):
        return []
    # Допускаем небольшой разнос атаки доли и низкочастотного акцента.
    values = np.array([accent[max(0, frame - 3) : frame + 4].max() for frame in frames])
    best = (meters[0], 0)
    best_score = 0.0
    for meter in meters:
        if len(values) < 2 * meter:
            continue
        for phase in range(meter):
            mask = np.arange(len(values)) % meter == phase
            on, off = values[mask], values[~mask]
            score = float(on.mean() - off.mean())
            if score > best_score + 1e-6:
                best_score, best = score, (meter, phase)
    meter, phase = best
    return [1 + (index - phase) % meter for index in range(len(frames))]


def make_grid(
    frames: np.ndarray, accent: np.ndarray, meters: tuple[int, ...], duration: float
) -> list[Beat]:
    numbers = number_beats(frames, accent, meters)
    return [
        Beat(float(frame) / FRAME_RATE, number)
        for frame, number in zip(frames, numbers, strict=True)
        if frame / FRAME_RATE < duration
    ]
