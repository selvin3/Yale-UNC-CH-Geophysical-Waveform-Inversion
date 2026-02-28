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


def test(data_path: str, model_path: str):
    """Start testing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeismicUNet()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    # Predict results
    rows = []
    for file_name in sorted(os.listdir(data_path)):
        oid = file_name.replace(".npy", "").strip()

        data = np.load(os.path.join(data_path, file_name))
        data = np.moveaxis(data, -1, 0)
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            vel = model(data)

        vel = vel.squeeze(0)
        vel = vel[:, 1::2]

        H, W_odd = vel.shape
        W = W_odd * 2

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
