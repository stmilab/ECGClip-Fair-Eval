# clip_train.py
import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import CLIPForECG, contrastive_loss

from datetime import datetime
from dataloader_multimodal import ECGDataset as ECGDatasetNoAug
from dataloader_multimodal_augmented import ECGDataset as ECGDatasetAug

def extract_embeddings_with_metadata(dataloader, model, device="cuda"):
    model.eval()
    model.to(device)

    results = []

    with torch.no_grad():
        for ecg, text, age, gender, subject_id, study_id in tqdm(dataloader):
            ecg = ecg.float().to(device)
            tokenized = model.text_encoder.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)

            ecg_embed = model.ecg_projection(model.ecg_encoder(ecg))
            text_embed = model.text_encoder.projection(
                model.text_encoder.model(**tokenized).last_hidden_state.mean(dim=1)
            )

            ecg_embed = F.normalize(ecg_embed, dim=-1).cpu().numpy()
            text_embed = F.normalize(text_embed, dim=-1).cpu().numpy()

            for i in range(len(ecg_embed)):
                results.append({
                    "ecg_embedding": ecg_embed[i],
                    "text_embedding": text_embed[i],
                    "combined": ecg_embed[i].tolist() + text_embed[i].tolist(),
                    "age": age[i],
                    "gender": gender[i],
                    "study_id": study_id[i],
                })

    return pd.DataFrame(results)

def train_clip(model, dataloader, optimizer, device, num_epochs=10, temperature=0.07):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for ecg, text, *_ in tqdm(dataloader):
            ecg = ecg.float().to(device)
            tokenized = model.text_encoder.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(device)

            ecg_embed = F.normalize(model.ecg_projection(model.ecg_encoder(ecg)), dim=-1)
            text_embed = F.normalize(model.text_encoder.projection(
                model.text_encoder.model(**tokenized).last_hidden_state.mean(dim=1)), dim=-1)

            loss = contrastive_loss(ecg_embed, text_embed, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Avg Contrastive Loss: {total_loss / len(dataloader):.4f}")

# ===== Main =====
def main(augmented=True, batch_size=64, epochs=10, save_path="./clip_outputs", model_path=None):
    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    if augmented:
        dataset = ECGDatasetAug(split="Train", from_numpy=False)
        test_dataset = ECGDatasetAug(split="Test", from_numpy=False)
    else:
        dataset = ECGDatasetNoAug(split="Train", from_numpy=False)
        test_dataset = ECGDatasetNoAug(split="Test", from_numpy=False)

    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    model = CLIPForECG()
    if model_path:
        print(f"Loading pretrained model from {model_path}")
        model.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("\n===> Starting CLIP training...")
    train_clip(model, train_loader, optimizer, device, num_epochs=epochs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f"{save_path}/clip_model_{'aug' if augmented else 'noaug'}_{timestamp}.pt")
    print(f"\nModel saved to {save_path}/clip_model_*.pt")

    print("\n===> Extracting embeddings for downstream...")
    embeddings_df = extract_embeddings_with_metadata(test_loader, model, device)
    embeddings_df.to_pickle(f"{save_path}/embeddings_{'aug' if augmented else 'noaug'}_{timestamp}.pkl")
    print(f"Embeddings saved to {save_path}/embeddings_*.pkl")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmented", action="store_true", help="Use augmented notes")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="./clip_outputs")
    parser.add_argument("--model_path", type=str, default=None, help="Pretrained CLIP checkpoint (optional)")

    args = parser.parse_args()
    main(args.augmented, args.batch_size, args.epochs, args.save_path, args.model_path)
