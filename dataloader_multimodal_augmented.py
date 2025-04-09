import os

import pandas as pd
import numpy as np
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import torchvision.transforms.functional as TF
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import preprocessing
from scipy.signal import butter, filtfilt, resample, iirnotch


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to the ECG signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data)
    return y


def notch_filter(data, notch_freq, fs, quality_factor=30):
    nyquist = 0.5 * fs
    w0 = notch_freq / nyquist
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, data)


def preprocess_ecg(ecg_signal, original_fs=500, target_fs=500, lowcut=0.5, highcut=150):
    """
    Preprocess an ECG signal:
    1. Filter the signal using a bandpass filter.
    2. Resample the signal to a target frequency.
    3. Normalize the signal.
    """
    # Step 1: Apply filters
    filtered_signal = butter_bandpass_filter(ecg_signal, lowcut, highcut, original_fs)

    filtered_signal = notch_filter(filtered_signal, lowcut, highcut, original_fs)

    # Step 2: Resample the signal
    num_samples = int(len(filtered_signal) * target_fs / original_fs)
    resampled_signal = resample(filtered_signal, num_samples)

    # Step 3: Normalize the signal
    normalized_signal = (resampled_signal - np.min(resampled_signal)) / (
        np.max(resampled_signal) - np.min(resampled_signal)
    )

    return normalized_signal


class ECGDataset(Dataset):
    def __init__(
        self,
        root="./",
        metadata_dir=f"/ssd-shared/mimiciv-ecg-echonotes/physionet.org/files/mimic-iv-ecg/1.0/processed/metadatav3_w_icd_augmented_report.csv",
        npy_file_dir="",
        split="Train",
        from_numpy=True,
        ecg_transform=None,
        range=[0, 1, 2],
    ):
        self.from_numpy = from_numpy
        self.range = range
        self.meta_data = (
            pd.read_csv(metadata_dir)
            # .sample(frac=1, random_state=42)
            # .reset_index(drop=True)
        )

        if self.from_numpy:
            self.npy_data = np.load(npy_file_dir)

            self.meta_data = self.meta_data[self.meta_data["data_warning"] == 0]
            self.meta_data = self.meta_data[self.meta_data["valid_bit"] == 1]
            self.meta_data = self.meta_data[
                self.meta_data["abnormality_label"] != "other"
            ]
            self.meta_data = self.meta_data.dropna(subset=["augmented_report_v2"])

            if split == "Train":
                self.meta_data = self.meta_data.iloc[: int(0.8 * len(self.meta_data))]
                self.npy_data = self.npy_data[self.meta_data.index]
                self.meta_data = self.meta_data.reset_index(drop=True)

            else:
                self.meta_data = self.meta_data.iloc[int(0.8 * len(self.meta_data)) :]
                self.npy_data = self.npy_data[self.meta_data.index]
                self.meta_data = self.meta_data.reset_index(drop=True)

        else:
            self.meta_data = self.meta_data[
                self.meta_data["data_warning"] == 0
            ].reset_index(drop=True)
            self.meta_data = self.meta_data[
                self.meta_data["valid_bit"] == 1
            ].reset_index(drop=True)

            self.meta_data = self.meta_data.iloc[:10000].reset_index(drop=True)
            print("self.meta_data", self.meta_data)
            print("self.meta_data", self.meta_data["abnormality_label"].value_counts())
            if split == "Train":
                self.meta_data = self.meta_data.iloc[
                    : int(0.8 * len(self.meta_data))
                ].reset_index(drop=True)


            else:
                self.meta_data = self.meta_data.iloc[
                    int(0.8 * len(self.meta_data)) :
                ].reset_index(drop=True)

        self.ecg1 = [None] * len(self.meta_data)
        self.gender = [None] * len(self.meta_data)
        self.age = [None] * len(self.meta_data)
        self.subject_id = [None] * len(self.meta_data)
        self.study_id = [None] * len(self.meta_data)
        self.abnormality_label = [None] * len(self.meta_data)
        self.root = root
        self.report = [None] * len(self.meta_data)
        self.get_data()

    def get_data(self):
        for idx in tqdm(range(len(self))):
            if self.from_numpy:
                self.ecg1[idx] = self.npy_data[idx]

            else:
                self.ecg1[idx] = preprocess_ecg(
                    np.load(
                        f"/ssd-shared/mimiciv-ecg-echonotes/physionet.org/files/mimic-iv-ecg/1.0/{self.meta_data.iloc[idx]['path']}_ECG_II.npy"
                    )
                )

            if not np.isfinite(self.ecg1[idx]).all():
                print("here1")
            self.subject_id[idx] = self.meta_data.iloc[idx]["subject_id"]
            self.study_id[idx] = self.meta_data.iloc[idx]["study_id"]
            self.report[idx] = self.meta_data.iloc[idx]["augmented_report_v2"]
            self.age[idx] = self.meta_data.iloc[idx]["age"]
            self.gender[idx] = self.meta_data.iloc[idx]["gender"]
            self.abnormality_label[idx] = self.meta_data.iloc[idx]["abnormality_label"]

    def __getitem__(self, index):

        ecg1 = signal.resample(self.ecg1[index], 1000).reshape(1, 1000)

        subject_id = self.subject_id[index]
        study_id = self.study_id[index]
        abnormality_label = self.abnormality_label[index]
        report = self.report[index]
        age = self.age[index]
        gender = self.gender[index]
        if self.abnormality_label[index] == "sinus_rhythm":
            abnormality_label = 3
        elif self.abnormality_label[index] == "sinus_bradycardia":
            abnormality_label = 0
        elif self.abnormality_label[index] == "sinus_tachycardia":
            abnormality_label = 1
        elif self.abnormality_label[index] == "atrial_fibrillation":
            abnormality_label = 2
        else:
            abnormality_label = 4

        return (ecg1, report, age, gender, subject_id, study_id)

    def collate(self, batch):
        (
            ecg1,
            report,
            age,
            gender,
            subject_id,
            study_id,
        ) = zip(*batch)

        ecg1 = torch.stack(ecg1, dim=0)
        return (ecg1, report, age, gender, subject_id, study_id)

    def __len__(self):
        return len(self.meta_data)
