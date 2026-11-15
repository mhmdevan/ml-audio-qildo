#!/usr/bin/env python3
"""Обучить CNN с нуля на манифесте Qildo."""

import argparse
import hashlib
import random
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from qildo_beats.audio import FRAME_RATE, MEL_BANDS, extract_features
from qildo_beats.dataset import Track, load_manifest
from qildo_beats.grid import Beat, validate_grid, write_json
from qildo_beats.model import (
    CHECKPOINT_VERSION,
    CONTEXT,
    MODEL_ARCHITECTURES,
    build_model,
    left_context,
    select_device,
)


def frame_labels(grid: list[Beat], frames: int) -> np.ndarray:
    labels = np.zeros((2, frames), dtype=np.float32)
    for beat in grid:
        center = min(frames - 1, round(beat.time * FRAME_RATE))
        first = max(0, center - 2)
        last = min(frames, center + 3)
        target = 1 - np.abs(np.arange(first, last) - center) / 3
        bands = slice(0, 2 if beat.beat == 1 else 1)
        labels[bands, first:last] = np.maximum(labels[bands, first:last], target)
    return labels


class BeatWindows(Dataset):
    def __init__(
        self,
        tracks: list[Track],
        core_frames: int,
        *,
        left_margin: int,
        augment: bool = False,
        tempo_warp: float = 0,
    ) -> None:
        self.features = []
        self.labels = []
        self.windows = []
        self.core_frames = core_frames
        self.left_margin = left_margin
        self.augment = augment
        self.tempo_warp = tempo_warp
        for track in tracks:
            features = extract_features(track.audio)
            grid = track.grid()
            validate_grid(grid, features.duration)
            index = len(self.features)
            self.features.append(features.log_mel.T.copy())
            self.labels.append(frame_labels(grid, len(features.log_mel)))
            self.windows.extend(
                (index, first) for first in range(0, len(features.log_mel), core_frames)
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        track, start = self.windows[index]
        features, labels = self.features[track], self.labels[track]
        first = max(0, start - self.left_margin)
        last = min(features.shape[1], start + self.core_frames + CONTEXT)
        offset = first - start + self.left_margin
        # Контекст используется для прогноза, а функция потерь видит только ядро окна.
        batch = np.zeros(
            (MEL_BANDS, self.core_frames + self.left_margin + CONTEXT), dtype=np.float32
        )
        batch[:, offset : offset + last - first] = features[:, first:last]
        if self.augment:
            gain, tilt = torch.empty(2).uniform_(-0.7, 0.7).tolist()
        size = min(self.core_frames, features.shape[1] - start)
        truth = np.zeros((2, self.core_frames), dtype=np.float32)
        truth[:, :size] = labels[:, start : start + size]
        if self.tempo_warp:
            speed = float(
                torch.empty(()).uniform_(1 - self.tempo_warp, 1 + self.tempo_warp)
            )
            # Одна шкала исходных кадров для входа и обеих целей.
            ticks = (np.arange(batch.shape[1]) - self.left_margin) * speed + start
            lo = np.floor(ticks).astype(int)
            fraction = (ticks - lo).astype(np.float32)

            def sample(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
                supported = (0 <= indices) & (indices < values.shape[1])
                return values[:, indices.clip(0, values.shape[1] - 1)] * supported

            batch = (
                sample(features, lo) * (1 - fraction)
                + sample(features, lo + 1) * fraction
            )
            core = slice(self.left_margin, self.left_margin + self.core_frames)
            low, alpha = lo[core], fraction[core]
            truth = sample(labels, low) * (1 - alpha) + sample(labels, low + 1) * alpha
            size = min(
                self.core_frames, int(np.ceil((features.shape[1] - start) / speed))
            )
        if self.augment:
            # log(1 + power) после усиления аудио или лёгкого спектрального наклона.
            scale = np.exp(
                gain + tilt * np.linspace(-1, 1, MEL_BANDS, dtype=np.float32)
            )
            batch = np.log1p(np.expm1(batch) * scale[:, None])
        return batch, truth, np.arange(self.core_frames) < size


def epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    left_margin: int,
    hierarchical: bool = False,
) -> float:
    model.train(optimizer is not None)
    total = 0.0
    frames = 0
    with torch.set_grad_enabled(optimizer is not None):
        for features, labels, valid in loader:
            features, labels, valid = (
                features.to(device),
                labels.to(device),
                valid.to(device),
            )
            logits = model(features)[:, :, left_margin : left_margin + labels.shape[-1]]
            if hierarchical:
                beat_target, accent_target = labels.unbind(dim=1)
                beat_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits[:, 0],
                    beat_target,
                    pos_weight=logits.new_tensor(12),
                    reduction="none",
                )
                conditional = accent_target / beat_target.clamp_min(1e-6)
                accent_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits[:, 1],
                    conditional,
                    pos_weight=logits.new_tensor(2),
                    reduction="none",
                )
                loss = (
                    (beat_loss + 4 * accent_loss * beat_target) * valid
                ).sum() / valid.sum()
            else:
                loss = (loss_fn(logits, labels) * valid[:, None]).sum() / (
                    2 * valid.sum()
                )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            count = int(valid.sum())
            total += float(loss.detach()) * count
            frames += count
    return total / frames


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    manifests = [args.manifest, *args.additional_manifest]
    tracks = [track for manifest in manifests for track in load_manifest(manifest)]
    if len({track.track_id for track in tracks}) != len(tracks):
        raise ValueError("Повторяющиеся id в объединённых манифестах")
    for field in ("audio", "group"):
        splits = {}
        for track in tracks:
            value = getattr(track, field)
            if splits.setdefault(value, track.split) != track.split:
                raise ValueError(f"Утечка между манифестами: {value}")
    train_tracks = [track for track in tracks if track.split == "train"]
    validation_tracks = [track for track in tracks if track.split == "validation"]
    if not train_tracks or not validation_tracks:
        raise ValueError("Нужны непустые выборки train и validation")
    print(
        f"Извлечение признаков: train={len(train_tracks)}, validation={len(validation_tracks)}",
        flush=True,
    )
    left_margin = left_context(args.architecture)
    train_set = BeatWindows(
        train_tracks,
        args.window,
        left_margin=left_margin,
        augment=args.augment,
        tempo_warp=args.tempo_warp,
    )
    validation_set = BeatWindows(
        validation_tracks, args.window, left_margin=left_margin
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)
    model = build_model(args.architecture, args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([[12.0], [35.0]], device=device), reduction="none"
    )
    history = []
    best = float("inf")
    settings = {
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "additional_manifests": {
            str(manifest): hashlib.sha256(manifest.read_bytes()).hexdigest()
            for manifest in args.additional_manifest
        },
        "architecture": args.architecture,
        "augment": args.augment,
        "tempo_warp": args.tempo_warp,
        "epochs": args.epochs,
        "seed": args.seed,
        "channels": args.channels,
        "window": args.window,
        "batch_size": args.batch_size,
        "device": args.device,
        "torch": str(torch.__version__),
    }
    for number in range(1, args.epochs + 1):
        training = epoch(
            model,
            train_loader,
            device,
            loss,
            optimizer,
            left_margin=left_margin,
            hierarchical=args.architecture == "hierarchical",
        )
        validation = epoch(
            model,
            validation_loader,
            device,
            loss,
            left_margin=left_margin,
            hierarchical=args.architecture == "hierarchical",
        )
        if not np.isfinite([training, validation]).all():
            raise RuntimeError("Функция потерь стала неконечной")
        history.append(
            {"epoch": number, "train_loss": training, "validation_loss": validation}
        )
        print(
            f"{number:02d} train={training:.5f} validation={validation:.5f}", flush=True
        )
        if validation < best:
            best = validation
            save_checkpoint(
                args.output,
                {
                    "version": CHECKPOINT_VERSION,
                    "frame_rate": FRAME_RATE,
                    "mel_bands": MEL_BANDS,
                    "channels": args.channels,
                    "architecture": args.architecture,
                    "epoch": number,
                    "state": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "settings": settings,
                },
            )
        write_json(
            args.output.with_suffix(".training.json"),
            {
                "settings": settings,
                "history": history,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path, help="Путь к выходному файлу весов .pt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument(
        "--window", type=int, default=1200, help="Длина ядра в кадрах по 10 мс"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--additional-manifest", type=Path, action="append", default=[])
    parser.add_argument("--architecture", choices=MODEL_ARCHITECTURES, default="cnn")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--tempo-warp", type=float, default=0)
    args = parser.parse_args()
    if (
        min(args.epochs, args.batch_size, args.channels, args.window) < 1
        or args.channels > 256
    ):
        parser.error("Требуются положительные параметры и не более 256 каналов")
    if args.output.suffix != ".pt":
        parser.error("Файл весов должен иметь расширение .pt")
    if not 0 <= args.tempo_warp <= 0.30:
        parser.error("Допустимый диапазон --tempo-warp от 0 до 0,30")
    try:
        train(args)
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"Ошибка обучения: {error}")


if __name__ == "__main__":
    main()
