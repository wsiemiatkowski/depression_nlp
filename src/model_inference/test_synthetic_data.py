import joblib
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# Config
SYNTHETIC_DATA_PATH = "data/04_synthetic_data/synthetic_judged_data.tsv"

MODELS = {
    "albert": "fine_tuned_models/albert/fine_tuned_albert",
    "tinybert": "fine_tuned_models/mobilebert/fine_tuned_mobilebert",
    "mobilebert": "fine_tuned_models/tinybert/fine_tuned_tinybert",
}

PICKLES = {
    "albert": "fine_tuned_models/albert/label_encoder.pkl",
    "tinybert": "fine_tuned_models/mobilebert/label_encoder.pkl",
    "mobilebert": "fine_tuned_models/tinybert/label_encoder.pkl",
}

METRICS_DIR = Path("data/03_metrics/synthetic_eval")
METRICS_DIR.mkdir(parents=True, exist_ok=True)
BATCH_SIZE = 16
MAX_LENGTH = 128
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}


def predict(model, dataloader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)


def plot_confusion_matrix(conf_matrix, labels, model_name, output_path):
    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix_df,
        annot=True,
        fmt="d",
        cmap="RdPu",
        linewidths=0.5,
        linecolor="white",
    )
    plt.title(f"{model_name} Confusion Matrix — Synthetic Data")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def evaluate_model(model_name, model_path, pkl_path, texts, true_labels):
    print(f"\nEvaluating {model_name}...")

    # Load label encoder
    with open(pkl_path, "rb") as f:
        label_encoder = joblib.load(f)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)

    # Encode true labels
    encoded_true = label_encoder.transform(true_labels)

    # Build dataloader
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Predict
    predictions = predict(model, dataloader)

    # Decode
    decoded_preds = label_encoder.inverse_transform(predictions)
    decoded_true = label_encoder.inverse_transform(encoded_true)

    # Confusion matrix
    conf_matrix = confusion_matrix(
        decoded_true, decoded_preds, labels=label_encoder.classes_
    )
    plot_confusion_matrix(
        conf_matrix,
        label_encoder.classes_,
        model_name,
        METRICS_DIR / f"{model_name}_synthetic_confusion_matrix.png",
    )

    return decoded_true, decoded_preds


def main():
    df = pd.read_csv(SYNTHETIC_DATA_PATH, sep="\t")
    texts = df["text"].tolist()
    true_labels = df["label"].tolist()

    for model_name, model_path in MODELS.items():
        evaluate_model(model_name, model_path, PICKLES[model_name], texts, true_labels)


if __name__ == "__main__":
    main()
