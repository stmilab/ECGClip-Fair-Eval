import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import re


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        folder,
        metadata_csv,
        max_len=500,
        use_matched_file=False,
        matched_file_path="matched_patients.csv",
    ):
        self.folder = folder
        self.max_len = max_len
        self.metadata = pd.read_csv(metadata_csv)

        # Extract subject_id and hadm_id from filenames
        self.metadata["subject_id"] = self.metadata["stay"].apply(
            lambda x: int(re.match(r"^(\d+)_", x).group(1))
        )
        self.metadata["hadm_id"] = self.metadata["stay"].apply(
            lambda x: int(re.search(r"episode(\d+)", x).group(1))
        )

        # Optional filtering using matched patients
        if use_matched_file and os.path.exists(matched_file_path):
            matched_df = pd.read_csv(matched_file_path)
            self.metadata = pd.merge(
                self.metadata, matched_df, on=["subject_id", "hadm_id"]
            )

        # Infer numeric columns from first file
        first_file = os.path.join(self.folder, str(self.metadata.iloc[0]["stay"]))
        df = pd.read_csv(first_file)
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.folder, str(row["stay"]))
        df = pd.read_csv(file_path)

        df = df[self.numeric_cols].astype(np.float32)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method="ffill", inplace=True)
        df.fillna(0.0, inplace=True)
        signal = df.values
        label = int(row["y_true"])

        if signal.shape[0] < self.max_len:
            pad = np.zeros(
                (self.max_len - signal.shape[0], signal.shape[1]), dtype=np.float32
            )
            signal = np.vstack([signal, pad])
        else:
            signal = signal[: self.max_len]

        return torch.tensor(signal, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_embedding=False):

        x = self.embedding(x)

        x = self.encoder(x)

        x = x.mean(dim=1)
        if return_embedding:
            return x
        return self.classifier(x).squeeze(), x
