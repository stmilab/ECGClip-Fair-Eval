import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# 1. CNN-Based Waveform Encoder
class CNNWaveformEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim):
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

    def forward(self, x):
        # Input shape: (batch_size, in_channels, signal_length)
        x = self.cnn(x)  # Shape: (batch_size, 256, 1)
        x = x.squeeze(-1)  # Shape: (batch_size, 256)
        x = self.fc(x)  # Shape: (batch_size, embed_dim)
        return x


# 1. ECG Encoder: 1D Vision Transformer
class PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv1d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # Input shape: (batch_size, in_channels, signal_length)
        x = self.projection(x)  # Shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        return x


class TransformerEncoder1D(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Attention block
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        # Feed-forward block
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x


class VisionTransformer1D(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding1D(in_channels, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 1000, embed_dim)
        )  # Adjust max sequence length as needed
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoder1D(embed_dim, num_heads, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # Create patches and add positional encoding
        x = self.patch_embedding(x)  # Shape: (batch_size, num_patches, embed_dim)
        batch_size, num_patches, _ = x.size()
        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # Shape: (batch_size, 1, embed_dim)
        x = torch.cat(
            (cls_tokens, x), dim=1
        )  # Shape: (batch_size, num_patches + 1, embed_dim)
        x = x + self.positional_encoding[:, : x.size(1), :]

        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)

        return x[:, 0]  # Return CLS token for classification


# 2. ClinicalBERT as Text Encoder
class TextEncoder(nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", embed_dim=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.model.config.hidden_size, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.model(**inputs)
        # embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        embeddings = embeddings[:, 0, :]
        embeddings = self.projection(embeddings)  # Project to shared embedding space
        return self.layer_norm(embeddings)


# 3. CLIP Model
class CLIPForECG(nn.Module):
    def __init__(self, ecg_embed_dim=512, text_embed_dim=512, shared_embed_dim=512):
        super().__init__()
        # ECG Encoder
        # self.ecg_encoder = VisionTransformer1D(
        #     in_channels=1,
        #     patch_size=32,
        #     embed_dim=ecg_embed_dim,
        #     num_heads=8,
        #     hidden_dim=1024,
        #     num_layers=6,
        #     dropout=0.1,
        # )
        # self.ecg_projection = nn.Linear(ecg_embed_dim, shared_embed_dim)

        # Waveform Encoder (CNN)
        self.ecg_encoder = CNNWaveformEncoder(in_channels=1, embed_dim=ecg_embed_dim)
        self.ecg_projection = nn.Linear(ecg_embed_dim, shared_embed_dim)
        self.layer_norm = nn.LayerNorm(shared_embed_dim)

        # Text Encoder
        self.text_encoder = TextEncoder(embed_dim=shared_embed_dim)

    def forward(self, ecg, text):
        # ECG embeddings
        ecg_embeddings = self.ecg_encoder(ecg)
        ecg_embeddings = self.layer_norm(self.ecg_projection(ecg_embeddings))
        # ecg_embeddings = F.normalize(ecg_embeddings, dim=-1)

        # Text embeddings
        text_embeddings = self.text_encoder(text)
        # text_embeddings = F.normalize(text_embeddings, dim=-1)

        return ecg_embeddings, text_embeddings


# 4. Contrastive Loss
def contrastive_loss(image_embeddings, text_embeddings, temperature=0.5):
    logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2
