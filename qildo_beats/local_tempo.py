"""Траектория темпа по локальной автокорреляции и плавным переходам."""

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .audio import FRAME_RATE
from .decoder import MAX_PERIOD, MIN_PERIOD, _periods


@dataclass(frozen=True)
class TempoCurve:
    periods: np.ndarray
    evidence: float


def tempo_curves(pulse: np.ndarray) -> list[TempoCurve]:
    """Сохраняет соседние метрические уровни для дальнейшего выбора по всей песне."""
    proposals = _periods(pulse)
    if not proposals:
        return []
    hop = FRAME_RATE
    support = 4 * FRAME_RATE
    centers = np.arange(0, len(pulse) + hop, hop)
    periods = np.arange(MIN_PERIOD, MAX_PERIOD + 1)
    log_period = np.log2(periods)
    correlation = np.zeros((len(centers), len(periods)))
    for index, center in enumerate(centers):
        block = pulse[max(0, center - support) : center + support]
        block = block - block.mean()
        auto = signal.correlate(block, block, method="fft")[len(block) - 1 :]
        length = min(len(periods), max(0, len(auto) - MIN_PERIOD))
        if length and auto[0] > 1e-9:
            correlation[index, :length] = (
                auto[MIN_PERIOD : MIN_PERIOD + length] / auto[0]
            )

    step = np.abs(log_period[:, None] - log_period[None, :])
    # Постепенные отклонения дешевы. Устойчивая смена может окупить конечную цену перехода.
    transition_cost = np.minimum(10 * step**1.5, 1.4 + step)
    curves = []
    seen = set()
    for anchor, _ in sorted(proposals, key=lambda item: item[1], reverse=True)[:5]:
        penalty = 0.24 * np.abs(log_period - np.log2(anchor))
        emission = correlation - penalty
        scores = emission[0] - 2 * penalty
        previous = np.zeros_like(emission, dtype=np.int16)
        for index in range(1, len(centers)):
            transitions = scores[:, None] - transition_cost
            previous[index] = transitions.argmax(axis=0)
            scores = (
                emission[index] + transitions[previous[index], np.arange(len(periods))]
            )
        state = int(scores.argmax())
        states = np.empty(len(centers), dtype=int)
        for index in range(len(centers) - 1, -1, -1):
            states[index] = state
            state = previous[index, state]
        identity = states.tobytes()
        if identity not in seen:
            seen.add(identity)
            curves.append(
                TempoCurve(
                    np.interp(np.arange(len(pulse)), centers, periods[states]),
                    float(correlation[np.arange(len(centers)), states].mean()),
                )
            )
    return curves


def variable_path(
    pulse: np.ndarray, expected: np.ndarray, first: int, last: int
) -> np.ndarray:
    """Декодирует доли по локальному периоду с корректным переходом через границу темпа."""
    start, stop = max(0, first - 5), min(len(pulse), last + 6)
    target = expected[start:stop]
    evidence = pulse[start:stop]
    primitive = np.arange(MIN_PERIOD, MAX_PERIOD + 1)
    scores = np.full(len(evidence), -np.inf)
    previous = np.full(len(evidence), -1, dtype=np.int32)
    for frame, activation in enumerate(evidence):
        if frame < target[0]:
            scores[frame] = activation - 0.30 * frame / target[0]
        steps = primitive[: max(0, min(frame - MIN_PERIOD + 1, len(primitive)))]
        if not len(steps):
            continue
        indices = frame - steps
        # На границе темпа проверяем средний локальный период по обе стороны ребра.
        period = 0.5 * (target[frame] + target[indices])
        options = scores[indices] - 12.0 * np.log(steps / period) ** 2
        best = int(options.argmax())
        candidate = options[best] + activation + 0.08
        if candidate > scores[frame]:
            scores[frame] = candidate
            previous[frame] = indices[best]
    end = max(0, len(evidence) - int(target[-1]))
    frame = end + int(np.argmax(scores[end:]))
    path = []
    while frame >= 0:
        path.append(frame + start)
        frame = int(previous[frame])
    return np.asarray(path[::-1], dtype=int)
