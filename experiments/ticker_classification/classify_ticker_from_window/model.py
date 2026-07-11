import torch
from torch import nn


def _nearest_odd(x: float) -> int:
    return int(2 * round((x - 1) / 2) + 1)


def _kernel_schedule(n_blocks: int, kernel_start: int, kernel_end: int) -> list[int]:
    # quadratic ramp: stays near kernel_start for most of the depth, only grows toward
    # kernel_end in the last few blocks, so the receptive field doesn't saturate early
    return [_nearest_odd(kernel_start + (kernel_end - kernel_start) * (i / (n_blocks - 1)) ** 2) for i in range(n_blocks)]


class _ResConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout1d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class OHLCVClassifierCNN(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        *,
        hidden: int = 64,
        dropout: float = 0.3,
        n_blocks: int = 10,
        kernel_start: int = 3,
        kernel_end: int = 9,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=kernel_start, padding=kernel_start // 2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout1d(dropout),
        )
        kernel_sizes = _kernel_schedule(n_blocks - 1, kernel_start, kernel_end)
        self.blocks = nn.Sequential(*[_ResConvBlock(hidden, k, dropout) for k in kernel_sizes])
        self.last_conv = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.last_conv(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class MLPBaseline(nn.Module):
    def __init__(self, n_channels: int, window_length: int, n_classes: int, *, hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels * window_length, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
