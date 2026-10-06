import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SequenceAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        output_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        effective_dropout = dropout if num_layers > 1 else 0.0
        output_size = input_size if output_size is None else output_size

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        h_last = h_n[-1]
        z = self.to_latent(h_last)
        decoded_seed = self.from_latent(z)

        seq_len = x.size(1)
        # Repeat one latent-conditioned seed across time and let the decoder shape the sequence.
        repeated = decoded_seed.unsqueeze(1).repeat(1, seq_len, 1)

        decoded_seq, _ = self.decoder(repeated)
        return self.output_layer(decoded_seq)


class SequenceForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        forecast_horizon: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        effective_dropout = dropout if num_layers > 1 else 0.0
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, forecast_horizon * output_size),
        )

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        h_last = h_n[-1]
        # Predict the entire future block from the final encoded state.
        out = self.head(h_last)
        return out.view(-1, self.forecast_horizon, self.output_size)


class LSTMAutoencoder(SequenceAutoencoder):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(
            input_size=n_features,
            hidden_size=hidden_size,
            latent_size=latent_size,
            output_size=n_features,
            num_layers=num_layers,
            dropout=dropout,
        )


class TransformerSequenceAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        output_size: int | None = None,
        num_layers: int = 1,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_size % nhead != 0:
            raise ValueError("Transformer hidden_size must be divisible by transformer_nhead.")

        output_size = input_size if output_size is None else output_size
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position = SinusoidalPositionalEncoding(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        encoded_input = self.position(self.input_projection(x))
        encoded_seq = self.encoder(encoded_input)
        pooled = encoded_seq.mean(dim=1)
        z = self.to_latent(pooled)

        seq_len = x.size(1)
        decoded_seed = self.from_latent(z).unsqueeze(1).repeat(1, seq_len, 1)
        decoded_seq = self.decoder(self.position(decoded_seed))
        return self.output_layer(decoded_seq)


class ConvSequenceAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        output_size: int | None = None,
        kernel_size: int = 5,
        dropout: float = 0.0,
        temporal_bottleneck: bool = False,
    ):
        super().__init__()

        output_size = input_size if output_size is None else output_size
        padding = kernel_size // 2
        self.temporal_bottleneck = temporal_bottleneck

        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, latent_size, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )
        if temporal_bottleneck:
            # Squeeze the whole window into one latent vector like the LSTM/Transformer AEs.
            # Without this the per-timestep latent channels carry the input straight through
            # and the model reconstructs anomalies perfectly (near-zero error in suite runs).
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.to_latent = nn.Linear(latent_size, latent_size)
            self.from_latent = nn.Linear(latent_size, hidden_size)
            self.position = SinusoidalPositionalEncoding(hidden_size)
            self.decoder = nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_size, output_size, kernel_size=kernel_size, padding=padding),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Conv1d(latent_size, hidden_size, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_size, output_size, kernel_size=kernel_size, padding=padding),
            )

    def forward(self, x):
        # Conv1d expects channels-first, while the rest of the repo keeps [batch, time, features].
        x_cf = x.transpose(1, 2)
        z = self.encoder(x_cf)
        if self.temporal_bottleneck:
            pooled = self.pool(z).squeeze(-1)
            latent = self.to_latent(pooled)
            seed = self.from_latent(latent)
            # Broadcast one latent-conditioned seed across time; positional encoding lets the
            # translation-equivariant decoder produce time-varying output from it.
            repeated = seed.unsqueeze(1).repeat(1, x.size(1), 1)
            decoded = self.decoder(self.position(repeated).transpose(1, 2))
        else:
            decoded = self.decoder(z)
        return decoded.transpose(1, 2)


class ConvSequenceForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        forecast_horizon: int,
        output_size: int,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()

        padding = kernel_size // 2
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, forecast_horizon * output_size),
        )

    def forward(self, x):
        x_cf = x.transpose(1, 2)
        encoded = self.encoder(x_cf)
        out = self.head(encoded)
        return out.view(-1, self.forecast_horizon, self.output_size)
