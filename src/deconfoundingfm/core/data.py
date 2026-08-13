"""Internal tensor dataset with stable original-row indices."""

import torch
from torch.utils.data import Dataset


class DatasetDict(Dataset):
    def __init__(self, X: torch.Tensor, A: torch.Tensor, Y: torch.Tensor):
        self.X = X.float()
        self.A = A.float()
        self.Y = Y.float()
        if not (len(self.X) == len(self.A) == len(self.Y)):
            raise ValueError("X, A, and Y must have the same number of rows.")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"X": self.X[idx], "A": self.A[idx], "Y": self.Y[idx], "idx": idx}
