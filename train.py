"""Script to train model on seismic data."""

import logging
from collections.abc import Sequence

import torch
from absl import app, flags
from torch import optim
from torch.utils.data import DataLoader, random_split

from data_loader import LoadWaveData
from model import SeismicUNet

_DATA_PATH = flags.DEFINE_string("data_path", None, "Training dataset path.")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "Batch size value.")
_EPOCH = flags.DEFINE_integer("epoch", 100, "Number of epochs.")
_LERANING_RATE = flags.DEFINE_float(
    "learning_rate", 0.001, "Initial learning rate value."
)


def train(data_path: str, batch_size: int, epoch: int, learning_rate: float):
    """Start model training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LoadWaveData(root_dir=data_path)
    train_size = int(len(dataset) * 0.8)
    validation_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_data, validation_data = random_split(
        dataset, [train_size, validation_size], generator=generator
    )
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    validation_dataloader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=True
    )

    # load model, loss and optimizer
    model = SeismicUNet()
    model.to(device)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # early stopping params
    patience = 5
    min_delta = 1e-4
    epochs_no_improve = 0

    # start training
    best_val_loss = float("inf")
    train_loss_history = []
    val_loss_history = []
    for epoch_num in range(epoch):
        model.train()
        epoch_loss = []
        for data, velocity in train_dataloader:
            data = data.to(device)
            velocity = velocity.to(device)
            prediction = model(data)
            current_loss = criterion(prediction, velocity)

            # backward
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
            epoch_loss.append(current_loss.item())

        train_loss = sum(epoch_loss) / len(epoch_loss)
        train_loss_history.append(train_loss)

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for data, velocity in validation_dataloader:
                data = data.to(device)
                velocity = velocity.to(device)
                prediction = model(data)
                loss_value = criterion(prediction, velocity)
                val_losses.append(loss_value.item())

        val_loss = sum(val_losses) / len(val_losses)
        val_loss_history.append(val_loss)
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1

        logging.info(
            f"Epoch [{epoch_num + 1}/{epoch}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        if epochs_no_improve >= patience:
            logging.info(
                f"Early stopping triggered at epoch {epoch_num + 1}. "
                f"Best Val Loss: {best_val_loss:.6f}"
            )
            break

        # Use val_loss_history and train_loss_history for plotting


def main(argv: Sequence[str]):
    """Program starts here."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Start model training
    train(
        data_path=_DATA_PATH.value,
        batch_size=_BATCH_SIZE.value,
        epoch=_EPOCH.value,
        learning_rate=_LERANING_RATE.value,
    )


if __name__ == "__main__":
    app.run(main)
