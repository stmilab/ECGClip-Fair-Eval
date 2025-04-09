**ECGClip-Fair-Eval** is a research framework for studying fairness-aware multimodal pretraining using electrocardiogram (ECG) waveforms and rhythm descriptions. The project extends CLIP-style contrastive learning to align ECG signals with text and investigates how incorporating protected demographic attributes (age and gender) during pretraining impacts downstream classification performance and fairness.

---

## 🧠 Overview

This repository implements a pipeline that:

1. Pretrains a multimodal contrastive encoder on ECG signals and textual rhythm interpretations.
2. Incorporates age and gender into training via augmented natural language descriptions.
3. Extracts frozen embeddings from the ECG encoder.
4. Evaluates those embeddings using Logistic Regression, MLP, and XGBoost classifiers on a binary rhythm classification task.
5. Analyzes performance gaps across demographic subgroups.

---

## 📂 Repository Structure

```
├── dataloader_multimodal.py           # Original ECG+text dataloader
├── dataloader_multimodal_augmented.py # Augmented ECG+demographics+text dataloader
├── clip_train.py                      # Pretraining script
├── clip_prob.py                       # Evaluation script for downstream probing
├── visualizations/                    # Optional: notebooks/plots for performance analysis
├── utils.py                           # Helper utilities for saving, logging, plotting
├── clip_outputs/                       # Folder to save the outputs
```

---

## 📊 Downstream Evaluation

After pretraining:
- Embeddings are extracted using the frozen ECG encoder.
- A probing classifier (Logistic Regression / MLP / XGBoost) is trained to predict whether an ECG shows sinus rhythm (normal) or any other abnormal rhythm.
- Metrics: F1-score and AUROC
- Evaluation is performed:
  - Overall
  - By gender (Male / Female)
  - By age group (<60 / >=60)


---

## ⚙️ Requirements

- Python >= 3.7
- PyTorch
- scikit-learn
- xgboost
- pandas, numpy
- transformers (for BERT encoding)
- matplotlib / seaborn (for visualizations)

---

## 🧪 Experiments

We compare two pretraining settings:
- **Non-Augmented:** Only rhythm description used in text input
- **Augmented:** Age and gender added to the clinical note

Each setup is evaluated using linear probing, and results are saved for further visualization and analysis.

---

