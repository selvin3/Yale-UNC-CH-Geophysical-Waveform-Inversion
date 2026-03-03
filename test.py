"""Script to generate test submission file using train model."""

import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from absl import app, flags

from model import SeismicUNet

_DATA_PATH = flags.DEFINE_string("data_path", None, "Training dataset path.")
_MODEL_PATH = flags.DEFINE_string("model_path", None, "Trained model path.")


def test(data_path: str, model_path: str, batch_size: int = 256):
    """Start testing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SeismicUNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    file_names = sorted(os.listdir(data_path))
    rows = []

    for start in range(0, len(file_names), batch_size):
        batch_files = file_names[start : start + batch_size]

        batch_data = []
        batch_oids = []

        for file_name in batch_files:
            oid = file_name.replace(".npy", "").strip()
            data = np.load(os.path.join(data_path, file_name))
            data = np.moveaxis(data, -1, 0)  # (C, H, W)
            data = (data - data.mean()) / (data.std() + 1e-8)
            batch_data.append(data)
            batch_oids.append(oid)

        batch_data = torch.tensor(np.stack(batch_data), dtype=torch.float32).to(
            device
        )  # (B, C, H, W)

        with torch.no_grad():
            vel_batch = model(batch_data)  # (B, 1, H, W)

        vel_batch = vel_batch.squeeze(1)  # (B, H, W)
        vel_batch = vel_batch[:, :, 1::2]  # odd x columns

        B, H, W_odd = vel_batch.shape
        W = W_odd * 2

        for b in range(B):
            vel = vel_batch[b]
            oid = batch_oids[b]

            for y in range(H):
                row = {"oid_ypos": f"{oid}_y_{y}"}
                for i, xval in enumerate(range(1, W, 2)):
                    row[f"x_{xval}"] = vel[y, i].item()
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("submission.csv", index=False)


def main(argv: Sequence[str]):
    """Program starts here."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Test model on test dataset
    test(data_path=_DATA_PATH.value, model_path=_MODEL_PATH.value)


if __name__ == "__main__":
    app.run(main)
