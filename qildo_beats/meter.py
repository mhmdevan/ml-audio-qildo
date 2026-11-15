"""Совместная оценка положения долей и тактов по гипотезам пульса."""

from functools import lru_cache

import numpy as np
from scipy.ndimage import maximum_filter1d

from .audio import FRAME_RATE
from .decoder import MIN_PERIOD, activation_peaks, anchor_coverage, pulse_paths
from .grid import Beat


# На последнем шаге сохраняем фазу и размер каждого найденного такта.
@lru_cache(maxsize=16)
def _states(meters: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = [(meter, position) for meter in meters for position in range(1, meter + 1)]
    sizes, positions = np.array(states).T
    predecessors = np.full((len(states), len(meters)), -1, dtype=int)
    costs = np.full_like(predecessors, np.inf, dtype=float)
    bar_ends = np.where(sizes == positions)[0]
    for index, (size, position) in enumerate(states):
        if position == 1:
            predecessors[index] = bar_ends
            costs[index] = np.where(sizes[bar_ends] == size, 0, 1.0)
        else:
            predecessors[index, 0] = index - 1
            costs[index, 0] = 0
    return positions, predecessors, costs


def align_bars(
    frames: np.ndarray,
    pulse: np.ndarray,
    accent: np.ndarray,
    meters: tuple[int, ...],
    *,
    learned: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    positions, predecessors, meter_cost = _states(meters)
    downbeats = positions == 1
    shifts = np.arange(-7, 8)
    time = np.clip(frames[:, None] + shifts, 0, len(pulse) - 1)
    pulse_evidence = pulse[time]
    # Downbeat-выход модели включает признаки самой атаки. Убираем их общий вклад,
    # чтобы сильная доля выигрывала за счёт относительной вероятности начала такта.
    if learned:
        conditional = np.clip(accent - 0.5 * pulse, -0.45, 0.55)
        conditional -= np.median(conditional[frames])
        # Выходы могут разойтись на несколько кадров при неточной разметке атаки.
        conditional = maximum_filter1d(conditional, size=5)
    else:
        conditional = maximum_filter1d(accent, size=5) - np.median(accent[frames])
    accents = conditional[time]
    contrast = np.where(downbeats, 1.0, -0.32)
    emission = (
        pulse_evidence[:, None, :]
        + 0.45 * contrast[None, :, None] * accents[:, None, :]
    )
    emission -= (0.025 * abs(shifts))[None, None, :]
    # Размеры с более частыми границами дают больше возможностей подогнать шумные акценты.
    emission[:, downbeats, :] -= 0.025
    # Начальная граница файла — слабая подсказка, пригодная и для музыки с затактом.
    first_bonus = (0.32 if learned else 0.12) * np.exp(-frames[0] / 25)
    scores = emission[0] + first_bonus * downbeats[:, None]
    origin = np.empty((len(frames), len(positions), len(shifts)), dtype=np.int32)
    for index in range(1, len(frames)):
        elapsed = time[index][None, :] - time[index - 1][:, None]
        expected = frames[index] - frames[index - 1]
        timing_cost = 3.0 * np.log(np.maximum(elapsed, 1) / expected) ** 2
        timing_cost[elapsed < MIN_PERIOD - 1] = np.inf
        transitions = (
            scores[predecessors, :, None]
            - meter_cost[:, :, None, None]
            - timing_cost[None, None]
        ).reshape(len(positions), -1, len(shifts))
        choices = np.argmax(transitions, axis=1)
        scores = np.take_along_axis(transitions, choices[:, None, :], axis=1)[:, 0, :]
        scores += emission[index]
        origin[index] = choices

    state, shift = np.unravel_index(scores.argmax(), scores.shape)
    quality = float(
        (scores[state, shift] - pulse_evidence.max(axis=1).sum()) / len(frames)
    )
    selected_frames = np.empty_like(frames)
    numbers = np.empty(len(frames), dtype=int)
    for index in range(len(frames) - 1, -1, -1):
        selected_frames[index] = time[index, shift]
        numbers[index] = positions[state]
        if index:
            prior, shift = divmod(int(origin[index, state, shift]), len(shifts))
            state = predecessors[state, prior]
    return selected_frames, numbers, quality


def track_meter(
    pulse: np.ndarray,
    accent: np.ndarray,
    meters: tuple[int, ...],
    duration: float,
    *,
    learned: bool,
) -> list[Beat]:
    anchors = activation_peaks(pulse, 0.22, 8)
    if len(anchors) < 2:
        return [
            Beat(float(frame) / FRAME_RATE, 1)
            for frame in anchors
            if frame / FRAME_RATE < duration
        ]
    rhythm = 0.78 * pulse + 0.22 * accent
    best_score = -np.inf
    best = (anchors[:1], [1])
    for frames, base_score in pulse_paths(pulse, rhythm, anchors):
        refined, numbers, bar_quality = align_bars(
            frames, pulse, accent, meters, learned=learned
        )
        score = base_score + 0.22 * anchor_coverage(refined, anchors, pulse[anchors])
        score += 1.25 * bar_quality
        if score > best_score:
            best = (refined, numbers)
            best_score = score
    return [
        Beat(float(frame) / FRAME_RATE, int(number))
        for frame, number in zip(*best, strict=True)
        if frame / FRAME_RATE < duration
    ]
