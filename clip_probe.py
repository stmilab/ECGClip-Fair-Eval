# clip_probe.py
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import argparse


def map_label(label):
    return 0 if "sinus" in str(label).lower() else 1


def run_all_classifiers(df, feature_col="combined", label_col="label", group_name="Overall"):
    X = np.stack(df[feature_col].values)
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "MLPClassifier": MLPClassifier(hidden_layer_sizes=(256,), max_iter=500),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    results = []

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        for metric in ["precision", "recall", "f1-score"]:
            for cls in ["0", "1"]:
                results.append({
                    "Group": group_name,
                    "Classifier": name,
                    "Class": cls,
                    "Metric": metric,
                    "Value": report[cls][metric]
                })
        results.append({
            "Group": group_name,
            "Classifier": name,
            "Class": "overall",
            "Metric": "accuracy",
            "Value": report["accuracy"]
        })

    return results


def run_by_groups(df):
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 60, 120], labels=["<30", "30–60", ">60"])

    all_results = []
    all_results.extend(run_all_classifiers(df, group_name="Overall"))

    for gender in df["gender"].unique():
        gender_df = df[df["gender"] == gender]
        all_results.extend(run_all_classifiers(gender_df, group_name=f"Gender: {gender}"))

    for age_group in df["age_group"].unique():
        age_df = df[df["age_group"] == age_group]
        all_results.extend(run_all_classifiers(age_df, group_name=f"Age Group: {age_group}"))

    return pd.DataFrame(all_results)


def load_raw_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    df = pd.DataFrame(data)

    # Make sure embeddings are NumPy arrays
    for col in ["ecg_embedding", "text_embedding", "combined"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.array(x) if not isinstance(x, np.ndarray) else x)

    return df


def main(embedding_pkl, metadata_csv, output_csv):
    print(f"📦 Loading embeddings from: {embedding_pkl}")
    df = load_raw_pickle(embedding_pkl)

    print("📄 Loading metadata and merging...")
    meta = pd.read_csv(metadata_csv)[["study_id", "abnormality_label"]]

    # 🔧 FIX TYPE MISMATCH HERE
    df["study_id"] = df["study_id"].astype(str)
    meta["study_id"] = meta["study_id"].astype(str)

    df = df.merge(meta, on="study_id", how="left")

    print("🔖 Assigning binary labels (sinus rhythm = 0, else = 1)...")
    df["label"] = df["abnormality_label"].apply(map_label)

    print("🚀 Running full classifier evaluation...")
    results_df = run_by_groups(df)

    print(f"💾 Saving results to: {output_csv}")
    results_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, required=True, help="Path to .pkl embeddings (raw pickle)")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata CSV with abnormality_label")
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv", help="Where to save CSV results")

    args = parser.parse_args()
    main(args.embeddings, args.metadata, args.output_csv)
