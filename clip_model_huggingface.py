import pytorch_lightning as pl
import numpy as np
import pandas as pd
import shutil, time, os, requests, random, copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataloader_multimodal import ECGDataset
from torchvision import datasets, transforms, models
import torch.nn.functional as F

import matplotlib.pyplot as plt
from model import CLIPForECG, contrastive_loss
from transformers import AutoModel, AutoTokenizer, BertTokenizer

# %matplotlib inline

from sklearn.manifold import TSNE
import wandb

# from simclr_model import LARS, SimCLR_Loss, PreModel, SupConLoss
from utils import save_model, plot_features
import argparse

parser = argparse.ArgumentParser()


parser = argparse.ArgumentParser(description="MIMIC_IV_K")
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Choosing the Batch Size, default=512 (If out of memory, try a smaller batch_size)",
)

parser.add_argument(
    "--metod_type",
    default="SSLCon",
    help="Choosing the loss function between supervised contrastive and self-supervised contrastive",
)
parser.add_argument(
    "--use_wandb",
    type=int,
    default=0,
    help="Flag to use wandb or not",
)
parser.add_argument(
    "--lr",
    type=int,
    default=0.01,
    help="learning_rate",
)
parser.add_argument(
    "--exp_result_path",
    default="/home/grads/z/zhale/ECGClip/exp",
    help="path to save the model artifacts",
)


args = parser.parse_args()

EPOCHS = 4
PROJECTION_LAYERS = 3
EMBED_DIM = 512
TRANSFORMER_EMBED_DIM = 768
MAX_LEN = 64  # Maximum length of text


class Tokenizer:
    def __init__(self, tokenizer: BertTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, x: str) -> AutoTokenizer:
        return self.tokenizer(
            x, max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt"
        )

    def decode(self, x: Dict[str, torch.LongTensor]):
        return [
            self.tokenizer.decode(sentence[:sentence_len])
            for sentence, sentence_len in zip(
                x["input_ids"], target["attention_mask"].sum(axis=-1)
            )
        ]


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


def projection_layers(d_in: int, d_out: int, layers: int) -> nn.Module:
    layers = sum(
        [[Projection(d_in, d_in), nn.GELU()] for _ in range(layers - 1)], []
    ) + [Projection(d_in, d_out)]
    return nn.Sequential(*layers)


# class VisionEncoder(nn.Module):
#     def __init__(self, d_out: int) -> None:
#         super().__init__()
#         base = models.resnet34(pretrained=True)
#         d_in = base.fc.in_features
#         base.fc = nn.Identity()
#         self.base = base
#         self.projection = Projection(d_in, d_out)
#         for p in self.base.parameters():
#             p.requires_grad = False

#     def forward(self, x):
#         projected_vec = self.projection(self.base(x))
#         projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
#         return projected_vec / projection_len


class VisionEncoder(nn.Module):
    def __init__(
        self, d_in: int, d_out: int, n_projection_layers: int, in_channels, embed_dim
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
        )
        self.fc = nn.Linear(256, embed_dim)
        for p in self.cnn.parameters():
            p.requires_grad = True
        self.projection = projection_layers(d_in, d_out, n_projection_layers)

    def forward(self, x):
        # Input shape: (batch_size, in_channels, signal_length)
        projected_vec = self.projection(self.cnn(x))  # Shape: (batch_size, 256, 1)
        # x = x.squeeze(-1)  # Shape: (batch_size, 256)
        # x = self.fc(x)  # Shape: (batch_size, embed_dim)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class TextEncoder(nn.Module):
    def __init__(
        self, d_out: int, TEXT_MODEL="emilyalsentzer/Bio_ClinicalBERT"
    ) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained(TEXT_MODEL)
        self.projection = Projection(TRANSFORMER_EMBED_DIM, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(**x)[0]
        out = out[:, 0, :]  # get CLS token output
        projected_vec = self.projection(out)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0


def metrics(similarity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        TEXT_MODEL="emilyalsentzer/Bio_ClinicalBERT",
    ) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder(
            d_in=1,
            d_out=EMBED_DIM,
            n_projection_layers=PROJECTION_LAYERS,
            in_channels=1,
            embed_dim=EMBED_DIM,
        )
        self.caption_encoder = TextEncoder(EMBED_DIM)
        self.tokenizer = tokenizer = Tokenizer(
            AutoTokenizer.from_pretrained(TEXT_MODEL)
        )

        self.lr = lr

    def common_step(self, batch: Tuple[torch.Tensor, List[str]]) -> torch.Tensor:
        images, text = batch
        text_dev = {
            k: torch.tensor(v).to(self.device) for k, v in self.tokenizer(text).items()
        }

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text)
        similarity = caption_embed @ image_embed.T

        loss = clip_loss(similarity)
        img_acc, cap_acc = metrics(similarity)
        return loss, img_acc, cap_acc

    def training_step(
        self, batch: Tuple[torch.Tensor, List[str]], *args: list
    ) -> torch.Tensor:
        loss, img_acc, cap_acc = self.common_step(batch)
        self.log("training_loss", loss, on_step=True)
        self.log("training_img_acc", img_acc, on_step=True, prog_bar=True)
        self.log("training_cap_acc", cap_acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, List[str]], *args: list
    ) -> torch.Tensor:
        loss, img_acc, cap_acc = self.common_step(batch)
        self.log("validation_loss", loss, on_step=True)
        self.log("validation_img_acc", img_acc, on_step=True, prog_bar=True)
        self.log("validation_cap_acc", cap_acc, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        vision_params = {
            "params": self.vision_encoder.projection.parameters(),
            "lr": self.lr,
        }
        caption_params = {
            "params": self.caption_encoder.projection.parameters(),
            "lr": self.lr,
        }
        return torch.optim.Adam([vision_params, caption_params])


train_dataset = ECGDataset(
    split="Train",
    # npy_file_dir="/home/grads/z/zhale/MIMIC_ECG/MIMICIV_ECG_Processing/EDA/ecg_all.npy",
    from_numpy=False,
)
test_dataset = ECGDataset(
    split="Test",
    # npy_file_dir="/home/grads/z/zhale/MIMIC_ECG/MIMICIV_ECG_Processing/EDA/ecg_all.npy",
    from_numpy=False,
)

train_dl = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
valid_dl = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)


model = Model(1e-3)
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    # devices=torch.cuda.device_count(),
    devices=1,
    accelerator="gpu",
    gradient_clip_val=1.0,
    precision=16,
    num_sanity_val_steps=0,
)
print(trainer)
with torch.autocast("cuda"):
    trainer.fit(model, train_dl, valid_dl)  #
