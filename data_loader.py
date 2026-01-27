"""Data loader file."""

from torch.utils.data import Dataset


class LoadWaveData(Dataset):
    """Data loader class."""

    def __init__(self, root_dir: str):
        """Initialize dataset metadata."""
        self.root_dir = root_dir
        self.file_pairs = []
        self.total_samples = 0
        self.cumulative_sizes = []

    def __len__(self) -> int:
        """Return total number of training samples."""
        return self.total_samples

    def __getitem__(self, index: int):
        """Return one training sample."""
        raise NotImplementedError

    def _global_to_local_index(self, index: int):
        """Convert a global dataset index into
        - file index
        - local index inside that file
        """
        raise NotImplementedError
