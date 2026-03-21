import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AlbertTokenizer,
    AlbertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}


def tokenize_data(queries, labels):
    tokenized = tokenizer(
        queries.tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    tokenized["labels"] = torch.tensor(labels.tolist())
    return tokenized


# Paths to files
dataset_file = "data/01_raw/Suicide_Detection.csv"
intermediate_file = "data/02_intermediate/albert/Suicide_detection_intermediate.csv"
metrics_directory = "data/03_metrics/albert"
metrics_file = f"{metrics_directory}/metrics.txt"
confusion_matrix_file = f"{metrics_directory}/confusion_matrix.png"

# Create dataframe for training
main_df = pd.read_csv(dataset_file)
depression_df = main_df.sample(6000)

# Save unused rows to an intermediate file for future inferencing
non_overlapping_df = main_df[~main_df.isin(depression_df).all(axis=1)]
non_overlapping_df.to_csv(intermediate_file)

# Detect device for full M1 Mac support - mostly needed for inferencing and metrics
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Encode textual labels to integers
label_encoder = LabelEncoder()
depression_df["encoded_labels"] = label_encoder.fit_transform(depression_df["class"])

# Save the label encoder for later inference
joblib.dump(label_encoder, "DEMO/demo_albert/label_encoder.pkl")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    depression_df["text"],
    depression_df["encoded_labels"],
    test_size=0.2,
    random_state=42,
)

# Load ALBERT v2 small tokenizer and model
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)

# Prepare data for training
train_data = tokenize_data(X_train, y_train)
test_data = tokenize_data(X_test, y_test)

train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)

# Training arguments
training_args = TrainingArguments(
    output_dir="DEMO/demo_albert/results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("DEMO/demo_albert/fine_tuned_albert")
tokenizer.save_pretrained("DEMO/demo_albert/fine_tuned_albert")

# Evaluate the model
model.eval()

# Tokenize test data for evaluation
test_inputs = tokenizer(
    X_test.tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt"
).to(device)

model.to(device)

with torch.no_grad():
    outputs = model(**test_inputs)
    predictions = (
        torch.argmax(outputs.logits, dim=1).cpu().tolist()
    )

decoded_predictions = label_encoder.inverse_transform(predictions)
decoded_true_labels = label_encoder.inverse_transform(y_test.tolist())

# Metrics
classification_report = classification_report(
    decoded_true_labels, decoded_predictions, target_names=label_encoder.classes_
)
micro_f1 = f1_score(decoded_true_labels, decoded_predictions, average="micro")
macro_f1 = f1_score(decoded_true_labels, decoded_predictions, average="macro")

with open(metrics_file, "w") as f:
    f.write("Classification Report:\n")
    f.write(classification_report + "\n\n")
    f.write(f"Micro F1 Score: {micro_f1}\n")
    f.write(f"Macro F1 Score: {macro_f1}\n\n")

# Confusion matrix
conf_matrix = confusion_matrix(
    decoded_true_labels, decoded_predictions, labels=label_encoder.classes_
)
conf_matrix_df = pd.DataFrame(
    conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_
)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Labels")
plt.xlabel("Predicted Labels")
plt.tight_layout()
plt.savefig(confusion_matrix_file)
plt.close()