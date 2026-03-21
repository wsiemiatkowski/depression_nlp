import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "data/02_training_data/depression_nlp.tsv"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_prepare_data(label_encoder_path):
    depression_df = pd.read_csv(DATASET_PATH, sep="\t")

    label_encoder = LabelEncoder()
    depression_df = depression_df.copy()
    depression_df["encoded_labels"] = label_encoder.fit_transform(
        depression_df["class"]
    )

    os.makedirs(os.path.dirname(label_encoder_path), exist_ok=True)
    joblib.dump(label_encoder, label_encoder_path)

    X_train, X_test, y_train, y_test = train_test_split(
        depression_df["text"],
        depression_df["encoded_labels"],
        test_size=0.2,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test, label_encoder


def tokenize_dataset(tokenizer, queries, labels):
    tokenized = tokenizer(
        queries.tolist(),
        padding=True,
        truncation=True,
        max_length=128,
    )
    tokenized["labels"] = labels.tolist()
    return Dataset.from_dict(tokenized)


def save_metrics(
    metrics_dir, decoded_true_labels, decoded_predictions, label_encoder, model_name
):
    os.makedirs(metrics_dir, exist_ok=True)

    report = classification_report(
        decoded_true_labels, decoded_predictions, target_names=label_encoder.classes_
    )
    micro_f1 = f1_score(decoded_true_labels, decoded_predictions, average="micro")
    macro_f1 = f1_score(decoded_true_labels, decoded_predictions, average="macro")

    with open(f"{metrics_dir}/metrics.txt", "w") as f:
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write(f"Micro F1 Score: {micro_f1}\n")
        f.write(f"Macro F1 Score: {macro_f1}\n\n")

    conf_matrix = confusion_matrix(
        decoded_true_labels, decoded_predictions, labels=label_encoder.classes_
    )
    conf_matrix_df = pd.DataFrame(
        conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.tight_layout()
    plt.savefig(f"{metrics_dir}/confusion_matrix.png")
    plt.close()
