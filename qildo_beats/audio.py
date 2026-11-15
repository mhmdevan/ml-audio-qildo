"""Чтение WAV и признаки на единой шкале времени 100 Гц."""

import warnings
from dataclasses import dataclass
from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np
from scipy import fft, ndimage, signal
from scipy.io import wavfile

SAMPLE_RATE = 22_000
FRAME_RATE = 100
HOP = SAMPLE_RATE // FRAME_RATE
FFT_SIZE = 1024
MEL_BANDS = 64


@dataclass(frozen=True)
class Features:
    duration: float
    log_mel: np.ndarray
    onset: np.ndarray
    accent: np.ndarray
    audible: np.ndarray


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Chunk \(non-data\) not understood",
            category=wavfile.WavFileWarning,
        )
        warnings.filterwarnings(
            "error", message="Reached EOF", category=wavfile.WavFileWarning
        )
        try:
            sample_rate, raw = wavfile.read(path)
        except wavfile.WavFileWarning as error:
            raise ValueError(f"WAV оборван: {path}") from error
    if sample_rate <= 0 or raw.ndim not in (1, 2) or raw.shape[0] == 0:
        raise ValueError(f"Некорректный или пустой WAV: {path}")
    if raw.ndim == 2 and raw.shape[1] != 2:
        raise ValueError(f"Ожидается WAV с одним или двумя каналами: {path}")
    if raw.ndim == 1:
        raw = raw[:, None]
    samples = raw.astype(np.float32)
    if np.issubdtype(raw.dtype, np.integer):
        limits = np.iinfo(raw.dtype)
        if limits.min == 0:
            midpoint = (limits.max + 1) / 2
            samples = (samples - midpoint) / midpoint
        else:
            samples /= -limits.min
    if not np.isfinite(samples).all():
        raise ValueError(f"WAV содержит неконечные значения: {path}")
    return sample_rate, samples


@lru_cache(maxsize=1)
def _mel_filters() -> np.ndarray:
    frequencies = fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
    mel_edges = np.linspace(0, np.log1p(10_000 / 700), MEL_BANDS + 2)
    edges = 700 * np.expm1(mel_edges)
    left = (frequencies - edges[:-2, None]) / np.diff(edges)[:-1, None]
    right = (edges[2:, None] - frequencies) / np.diff(edges)[1:, None]
    filters = np.maximum(0, np.minimum(left, right)).astype(np.float32)
    return (filters / filters.sum(axis=1, keepdims=True)).T.copy()


def _local_scale(values: np.ndarray) -> np.ndarray:
    peaks = ndimage.maximum_filter1d(values, size=401, mode="nearest")
    floor = max(float(np.quantile(peaks, 0.8)) * 0.06, 1e-7)
    return np.clip(values / np.maximum(peaks, floor), 0, 1)


def extract_features(path: Path) -> Features:
    sample_rate, samples = read_wav(path)
    duration = len(samples) / sample_rate
    divisor = gcd(sample_rate, SAMPLE_RATE)
    if sample_rate != SAMPLE_RATE:
        samples = signal.resample_poly(
            samples, SAMPLE_RATE // divisor, sample_rate // divisor, axis=0
        )
    # Усреднение мощности каналов сохраняет атаки в противофазе.
    padded = np.pad(samples, ((FFT_SIZE // 2, FFT_SIZE // 2), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, FFT_SIZE, axis=0)[::HOP]
    windows = windows[: int(np.ceil(duration * FRAME_RATE))]
    taper = signal.windows.hann(FFT_SIZE, sym=False).astype(np.float32)
    log_mel = np.empty((len(windows), MEL_BANDS), dtype=np.float32)
    band_energy = np.empty((len(windows), 3), dtype=np.float32)
    chroma = np.empty((len(windows), 12), dtype=np.float32)
    energy = np.empty(len(windows), dtype=np.float32)
    freqs = fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
    bands = [
        np.flatnonzero((freqs >= lo) & (freqs < hi))
        for lo, hi in ((35, 210), (210, 2000), (2000, 10000))
    ]
    pitched = np.flatnonzero((freqs >= 110) & (freqs <= 2200))
    notes = np.rint(69 + 12 * np.log2(freqs[pitched] / 440)).astype(int) % 12
    for start in range(0, len(windows), 1024):
        block = slice(start, start + 1024)
        power = np.mean(np.abs(fft.rfft(windows[block] * taper, axis=-1)) ** 2, axis=1)
        energy[block] = power.sum(axis=1)
        mel = np.einsum("ij,jk->ik", power, _mel_filters(), optimize=False)
        log_mel[block] = np.log1p(mel)
        for index, bins in enumerate(bands):
            band_energy[block, index] = np.log1p(power[:, bins].mean(axis=1))
        for pitch in range(12):
            chroma[block, pitch] = np.sqrt(power[:, pitched[notes == pitch]]).sum(
                axis=1
            )
    chroma /= np.maximum(chroma.sum(axis=1, keepdims=True), 1e-7)
    # Разность на двух кадрах привязана к середине интервала, без сдвига шкалы.
    before = np.maximum(0, np.arange(len(windows)) - 1)
    after = np.minimum(len(windows) - 1, np.arange(len(windows)) + 1)
    flux = np.maximum(0, log_mel[after] - log_mel[before]).mean(axis=1)
    attacks = np.maximum(0, band_energy[after] - band_energy[before])
    onset = _local_scale(0.55 * flux + (attacks * [0.22, 0.18, 0.05]).sum(axis=1))
    harmony = np.abs(chroma[after] - chroma[before]).sum(axis=1)
    audible = ndimage.maximum_filter1d(energy, size=11) > max(
        float(energy.max()) * 1e-6, 1e-12
    )
    accent = (
        0.68 * _local_scale(attacks[:, 0]) + 0.22 * _local_scale(harmony) + 0.10 * onset
    )
    onset[~audible] = 0
    accent[~audible] = 0
    return Features(duration, log_mel, onset, accent, audible)
