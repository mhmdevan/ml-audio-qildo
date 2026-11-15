"""Ищет долгосрочный стабильный пульс по фазовому совпадению атак."""

import numpy as np
from scipy import ndimage

from .audio import FRAME_RATE, Features
from .decoder import MAX_BPM, MIN_BPM, activation_peaks
from .grid import Beat
from .meter import align_bars

PHASE_BINS = 72


def periodic_grid(features: Features, meters: tuple[int, ...]) -> list[Beat] | None:
    onset, accent = features.onset, features.accent
    anchors = activation_peaks(onset, 0.20, 6)
    if len(anchors) < 40 or anchors[-1] - anchors[0] < 20 * FRAME_RATE:
        return None
    weights = onset[anchors]
    spectrum_size = 1 << (len(onset) * 8 - 1).bit_length()
    spectrum = np.abs(np.fft.rfft(onset - onset.mean(), n=spectrum_size))
    candidates = activation_peaks(spectrum, 0, 3)
    peaks = candidates[np.argsort(spectrum[candidates])[-120:]]
    rates = peaks * (60 * FRAME_RATE / spectrum_size)
    rates = (rates[:, None] / [0.5, 1, 2, 3, 4]).ravel()
    rates = np.unique(np.round(rates[(rates >= MIN_BPM) & (rates <= MAX_BPM)], 2))
    nearby_onset = ndimage.maximum_filter1d(onset, size=15)
    nearby_accent = ndimage.maximum_filter1d(accent, size=7)
    options = {}
    for bpm in rates:
        period = 60 * FRAME_RATE / bpm
        phases = (anchors / period * PHASE_BINS) % PHASE_BINS
        histogram = np.bincount(
            phases.astype(int), weights=weights, minlength=PHASE_BINS
        )
        phase = ndimage.gaussian_filter1d(histogram, 1.5, mode="wrap").argmax()
        offset = phase * period / PHASE_BINS
        distances = np.abs((anchors - offset + period / 2) % period - period / 2)
        coverage = np.average(distances <= 6, weights=weights)
        first = np.ceil((anchors[0] - period / 2 - offset) / period)
        last = np.floor((anchors[-1] + 15 - offset) / period)
        frames = np.rint(offset + np.arange(first, last + 1) * period).astype(int)
        frames = frames[(frames >= 0) & (frames < len(onset))]
        if len(frames) < 8:
            continue
        support = nearby_onset[frames].mean()
        score = (
            0.75 * coverage
            + 0.18 * onset[frames].mean()
            + 0.12 * support
            - 0.105 * abs(np.log2(bpm / 112))
        )
        bucket = round(bpm)
        if bucket not in options or score > options[bucket][0]:
            options[bucket] = (score, frames, coverage, support)
    best_score = -np.inf
    best_grid = None
    for score, frames, coverage, support in sorted(
        options.values(), key=lambda item: item[0], reverse=True
    )[:8]:
        if coverage < 0.30 or support < 0.40:
            continue
        quarter_support = [
            section.mean() for section in np.array_split(nearby_onset[frames], 4)
        ]
        if min(quarter_support) < 0.22:
            continue
        refined, numbers, _ = align_bars(frames, onset, accent, meters, learned=False)
        marked = numbers == 1
        if not marked.any() or marked.all():
            continue
        values = nearby_accent[refined]
        contrast = values[marked].mean() - values[~marked].mean()
        if contrast < 0.16:
            continue
        score += 0.35 * contrast
        if score > best_score:
            best_score = score
            best_grid = [
                Beat(float(frame) / FRAME_RATE, int(number))
                for frame, number in zip(refined, numbers, strict=True)
                if frame / FRAME_RATE < features.duration
            ]
    return best_grid


def recover_periodic_grid(
    features: Features, meters: tuple[int, ...], acoustic: list[Beat]
) -> list[Beat] | None:
    if features.duration < 60:
        return None
    periodic = periodic_grid(features, meters)
    if periodic is None or len(periodic) <= 1.20 * len(acoustic):
        return None
    if acoustic and len(periodic) >= 2:
        period = float(np.median(np.diff([event.time for event in periodic])))
        gap = periodic[0].time - acoustic[0].time
        # Возвращаем раннюю долю только когда акустический трекер сам её нашёл.
        if abs(gap - period) <= 0.07:
            meter = max(event.beat for event in periodic[:16])
            previous = (periodic[0].beat - 2) % meter + 1
            return [Beat(acoustic[0].time, previous), *periodic]
    return periodic
