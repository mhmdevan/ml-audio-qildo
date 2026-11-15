"""Небольшая временная CNN, обучаемая с нуля."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .audio import FRAME_RATE, MEL_BANDS

DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CONTEXT = sum(DILATIONS)
CHECKPOINT_VERSION = 1
MODEL_ARCHITECTURES = ("cnn", "residual", "hierarchical")


def left_context(architecture: str) -> int:
    return CONTEXT + int(architecture in ("residual", "hierarchical"))


class BeatCNN(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.input = nn.Conv1d(MEL_BANDS, channels, 1)
        self.temporal = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels, channels, 3, padding=dilation, dilation=dilation
                    ),
                    nn.GELU(),
                    nn.Conv1d(channels, channels, 1),
                )
                for dilation in DILATIONS
            ]
        )
        self.output = nn.Conv1d(channels, 2, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        hidden = self.input(mel)
        for layer in self.temporal:
            hidden = hidden + 0.3 * layer(hidden)
        return self.output(hidden)


class ResidualBeatCNN(nn.Module):
    """Спектральные атаки на нескольких масштабах и контекст соседних тактов."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.input = nn.Conv1d(2 * MEL_BANDS, channels, 1)
        self.temporal = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(channels, channels, 3, dilation=dilation, padding=dilation),
                nn.GELU(),
                nn.Conv1d(channels, channels, 1),
                nn.GELU(),
            )
            for dilation in DILATIONS
        )
        self.output = nn.Sequential(
            nn.Conv1d(channels * 3, channels, 1), nn.GELU(), nn.Conv1d(channels, 2, 1)
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        spectrum = mel * 0.25
        changes = nn.functional.pad(torch.diff(spectrum, dim=-1), (1, 0))
        hidden = self.input(torch.cat((spectrum, 2 * changes), dim=1))
        levels = []
        for index, layer in enumerate(self.temporal, 1):
            hidden = hidden + 0.4 * layer(hidden)
            if index in (3, 6, 9):
                levels.append(hidden)
        return self.output(torch.cat(levels, dim=1))


def build_model(architecture: str, channels: int) -> nn.Module:
    if architecture in ("residual", "hierarchical"):
        return ResidualBeatCNN(channels)
    if architecture == "cnn":
        return BeatCNN(channels)
    raise ValueError(f"Неизвестная архитектура модели: {architecture}")


def select_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA недоступна")
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    return torch.device(device)


def load_predictor(checkpoint: Path, device: str) -> Callable[[np.ndarray], np.ndarray]:
    target = select_device(device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or (
        saved.get("version"),
        saved.get("frame_rate"),
        saved.get("mel_bands"),
    ) != (CHECKPOINT_VERSION, FRAME_RATE, MEL_BANDS):
        raise ValueError(f"Несовместимый чекпойнт: {checkpoint}")
    channels = saved.get("channels")
    state = saved.get("state")
    if (
        type(channels) is not int
        or not 1 <= channels <= 256
        or not isinstance(state, dict)
    ):
        raise ValueError(
            f"Чекпойнт не содержит корректную конфигурацию CNN: {checkpoint}"
        )
    architecture = saved.get("architecture", "cnn")
    model = build_model(architecture, channels).to(target)
    model.load_state_dict(state)
    model.eval()

    @torch.inference_mode()
    def activate(log_mel: np.ndarray) -> np.ndarray:
        output = np.empty((2, len(log_mel)), dtype=np.float32)
        left_margin = left_context(architecture)
        padded = np.pad(log_mel, ((left_margin, CONTEXT), (0, 0)))
        for first in range(0, len(log_mel), 3000):
            end = min(first + 3000, len(log_mel))
            batch = torch.from_numpy(
                padded[first : end + left_margin + CONTEXT].T[None]
            ).to(target)
            logits = model(batch)[0, :, left_margin : left_margin + end - first]
            probabilities = logits.sigmoid()
            if architecture == "hierarchical":
                probabilities[1].mul_(probabilities[0])
            output[:, first:end] = probabilities.cpu().numpy()
        return output

    return activate
