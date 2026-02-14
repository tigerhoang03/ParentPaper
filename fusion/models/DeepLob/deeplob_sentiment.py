"""
DeepLOB with State-Space Augmentation: optional Sentiment (FinBERT) fusion.

Use this file to REPLACE or extend models/DeepLob/deeplob.py in LOBFrame when
you want to feed (LOB data + sentiment vector) into the same model.

Fusion point: we concatenate the sentiment vector with the 192-d LOB-derived
features *before* the LSTM, so the LSTM input is (192 + sentiment_dim).
Input layer (Conv2d) is unchanged; only the LSTM and forward() are extended.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn


# Default FinBERT (BERT base) hidden size
DEFAULT_SENTIMENT_DIM = 768


class DeepLOBSentiment(pl.LightningModule):
    """
    DeepLOB that optionally accepts a sentiment vector per sample (State-Space Augmentation).

    - If sentiment is None or not passed: behavior identical to original DeepLOB (LSTM input_size=192).
    - If sentiment is passed: we concat it with the 192-d LOB features before the LSTM,
      so the LSTM sees (192 + sentiment_dim). Sentiment is broadcast over the sequence length.
    """

    def __init__(self, lighten: bool, sentiment_dim: int = 0):
        super().__init__()
        self.name = "deeplob"
        if lighten:
            self.name += "-lighten"
        self.sentiment_dim = sentiment_dim
        self._lob_feature_dim = 192  # After conv + inception concat

        # Convolution blocks (unchanged from original DeepLOB — this is the "input layer" for LOB).
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        if lighten:
            conv3_kernel_size = 5
        else:
            conv3_kernel_size = 10

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, conv3_kernel_size)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # Inception modules (unchanged).
        self.inp1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(3, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(5, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # LSTM: input is 192 (LOB only) or 192 + sentiment_dim when using sentiment.
        lstm_input_size = self._lob_feature_dim + self.sentiment_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor, sentiment: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: LOB input (batch, 1, window_size, n_features), e.g. (B, 1, 100, 40).
            sentiment: Optional (batch, sentiment_dim). If provided, concatenated with
                       LOB features before the LSTM (state-space augmentation).

        Returns:
            logits: (batch, 3).
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        # (B, C, T, W) -> (B, T, C*W) for LSTM
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (x.shape[0], x.shape[1], self._lob_feature_dim))

        # State-space augmentation: concat sentiment with LOB features before LSTM
        if sentiment is not None:
            # sentiment: (B, sentiment_dim) -> (B, 1, sentiment_dim) -> (B, T, sentiment_dim)
            sent = sentiment.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat((x, sent), dim=2)

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        logits = self.fc1(x)

        return logits
