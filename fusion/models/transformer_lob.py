from __future__ import annotations

import torch
import torch.nn as nn


class DeepLOBTransformerSentiment(nn.Module):
    """
    Part 3 model:
      - DeepLOB-style CNN feature extractor (same as Part 2)
      - TransformerEncoder (instead of LSTM) for time modeling
      - Sentiment is concatenated at each timestep

    Inputs:
      lob: (B, 1, T, F)  e.g. (batch, 1, 100, 20)
      sentiment: (B, sentiment_dim)  where sentiment_dim is 1 (sent_score) or 3 (probs)

    Output:
      logits: (B, 3) for classes [down, flat, up]
    """

    def __init__(
        self,
        lighten: bool = True,
        sentiment_dim: int = 1,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sentiment_dim = sentiment_dim
        self._lob_feature_dim = 192
        k = 5 if lighten else 10

        # CNN feature extractor (matches run_part2.py structure)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, k)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        # Inception-style blocks
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (5, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, (1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )

        # Project (LOB feature + sentiment) -> Transformer dimension
        self.proj = nn.Linear(self._lob_feature_dim + self.sentiment_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, 3)

    def forward(self, lob: torch.Tensor, sentiment: torch.Tensor) -> torch.Tensor:
        x = self.conv1(lob)
        x = self.conv2(x)
        x = self.conv3(x)

        # concat inception outputs
        x = torch.cat((self.inp1(x), self.inp2(x), self.inp3(x)), dim=1)

        # reshape into sequence: (B, C, T, 1) -> (B, T, C)
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), x.size(2), -1)

        # add sentiment to each timestep
        sent = sentiment.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat((x, sent), dim=2)

        x = self.proj(x)         # (B, T, d_model)
        x = self.encoder(x)      # (B, T, d_model)

        # last timestep -> classifier
        return self.fc(x[:, -1, :])