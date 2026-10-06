from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from radon_lstm_ae_torch_pkg.model import (
    ConvSequenceAutoencoder,
    ConvSequenceForecaster,
    LSTMAutoencoder,
    SequenceAutoencoder,
    SequenceForecaster,
    TransformerSequenceAutoencoder,
)


@dataclass
class TrainingArtifacts:
    model: nn.Module
    history: dict[str, list[float]]


def make_autoencoder(args, input_size: int, output_size: int | None = None) -> nn.Module:
    if output_size is None or output_size == input_size:
        return LSTMAutoencoder(
            n_features=input_size,
            hidden_size=args.hidden_size,
            latent_size=args.latent_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    return SequenceAutoencoder(
        input_size=input_size,
        output_size=output_size,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )


def make_conv_autoencoder(args, input_size: int, output_size: int | None = None) -> nn.Module:
    return ConvSequenceAutoencoder(
        input_size=input_size,
        output_size=output_size,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        kernel_size=args.conv_kernel_size,
        dropout=args.dropout,
        temporal_bottleneck=getattr(args, "conv_temporal_bottleneck", False),
    )


def make_transformer_autoencoder(args, input_size: int, output_size: int | None = None) -> nn.Module:
    return TransformerSequenceAutoencoder(
        input_size=input_size,
        output_size=output_size,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        nhead=args.transformer_nhead,
        dim_feedforward=args.transformer_dim_feedforward,
        dropout=args.dropout,
    )


def make_forecaster(args, input_size: int, output_size: int) -> nn.Module:
    return SequenceForecaster(
        input_size=input_size,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        forecast_horizon=args.forecast_horizon,
        output_size=output_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )


def make_conv_forecaster(args, input_size: int, output_size: int) -> nn.Module:
    return ConvSequenceForecaster(
        input_size=input_size,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        forecast_horizon=args.forecast_horizon,
        output_size=output_size,
        kernel_size=args.conv_kernel_size,
        dropout=args.dropout,
    )


def make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

    return total_loss / max(1, total_items)


def compute_prediction_errors(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = make_dataloader(X, y, batch_size=batch_size, shuffle=False)
    errors = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            batch_err = torch.mean((pred - batch_y) ** 2, dim=tuple(range(1, batch_y.ndim)))
            errors.append(batch_err.cpu().numpy())

    return np.concatenate(errors)


def compute_reconstruction_errors(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    return compute_prediction_errors(model, X, X, batch_size=batch_size, device=device)


def train_supervised_model(args, model: nn.Module, prepared_data, device: torch.device) -> TrainingArtifacts:
    train_loader = make_dataloader(
        prepared_data.X_train,
        prepared_data.y_train,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = None
    if prepared_data.X_val is not None and prepared_data.y_val is not None and len(prepared_data.X_val) > 0:
        val_loader = make_dataloader(
            prepared_data.X_val,
            prepared_data.y_val,
            batch_size=args.batch_size,
            shuffle=False,
        )

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=getattr(args, "weight_decay", 0.0),
    )
    criterion = nn.MSELoss()
    grad_clip = getattr(args, "grad_clip", 0.0)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience = args.early_stopping_patience
    min_delta = args.early_stopping_min_delta
    min_epochs = args.min_epochs
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_items = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            batch_size = batch_x.size(0)
            running_loss += loss.item() * batch_size
            n_items += batch_size

        train_loss = running_loss / max(1, n_items)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)

            print(f"Epoch {epoch:03d}/{args.epochs} - train_loss={train_loss:.6f} - val_loss={val_loss:.6f}")

            if val_loss < (best_val_loss - min_delta):
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if epoch >= min_epochs and patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        else:
            print(f"Epoch {epoch:03d}/{args.epochs} - train_loss={train_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainingArtifacts(model=model, history=history)


def train_model(args, prepared_data, device: torch.device) -> TrainingArtifacts:
    model = make_autoencoder(args, input_size=prepared_data.X.shape[2])
    supervised_view = SimpleNamespace(
        X_train=prepared_data.X_train,
        y_train=prepared_data.X_train,
        X_val=prepared_data.X_val,
        y_val=prepared_data.X_val,
        X_train_total=prepared_data.X_train_total,
        y_train_total=prepared_data.X_train_total,
    )
    return train_supervised_model(args, model=model, prepared_data=supervised_view, device=device)


def train_conv_model(args, prepared_data, device: torch.device) -> TrainingArtifacts:
    model = make_conv_autoencoder(args, input_size=prepared_data.X.shape[2])
    supervised_view = SimpleNamespace(
        X_train=prepared_data.X_train,
        y_train=prepared_data.X_train,
        X_val=prepared_data.X_val,
        y_val=prepared_data.X_val,
        X_train_total=prepared_data.X_train_total,
        y_train_total=prepared_data.X_train_total,
    )
    return train_supervised_model(args, model=model, prepared_data=supervised_view, device=device)
