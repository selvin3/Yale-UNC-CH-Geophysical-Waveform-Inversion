"""Data loader file."""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class LoadWaveData(Dataset):
    """Data loader class."""

    def __init__(self, root_dir: str):
        """Initialize dataset metadata."""
        self.root_dir = root_dir
        self.inputs, self.targets = self._load_all_file_pairs()
        self.samples_per_file = 500

    def __len__(self) -> int:
        """Return total number of training samples."""
        return len(self.inputs) * self.samples_per_file

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        """Return one training sample."""
        file_index = index // self.samples_per_file
        local_index = index % self.samples_per_file
        data = np.load(self.inputs[file_index], mmap_mode="r")[
            local_index
        ]  # (source, timestamp, recievers)
        data = np.moveaxis(data, -1, 0)  # (recievers, source, timestamp)
        data = (data - data.mean()) / (data.std() + 1e-8)

        target = np.load(self.targets[file_index], mmap_mode="r")[local_index]

        return torch.tensor(data, dtype=torch.float32), torch.tensor(
            target, dtype=torch.float32
        )

    def _load_all_file_pairs(self) -> tuple[list, list]:
        inputs = []
        targets = []

        for family_name in os.listdir(self.root_dir):
            family_path = os.path.join(self.root_dir, family_name)

            # 1. vel/style families
            data_dir = os.path.join(family_path, "data")
            model_dir = os.path.join(family_path, "model")
            if os.path.isdir(data_dir) and os.path.isdir(model_dir):
                data_path = os.path.join(data_dir, "data1.npy")
                model_path = os.path.join(model_dir, "model1.npy")

                inputs.append(data_path)
                targets.append(model_path)

            # 1. fault families
            else:
                seis_files = sorted(
                    os.path.join(family_path, f)
                    for f in os.listdir(family_path)
                    if f.startswith("seis")
                )
                model_files = sorted(
                    os.path.join(family_path, f)
                    for f in os.listdir(family_path)
                    if f.startswith("vel")
                )

                inputs.extend(seis_files)
                targets.extend(model_files)

        return inputs, targets
