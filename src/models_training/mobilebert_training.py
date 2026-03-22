import torch

from transformers import (
    MobileBertForSequenceClassification,
    MobileBertTokenizer,
    Trainer,
    TrainingArguments,
)

from src.utils.training_utils import (
    get_device,
    load_and_prepare_data,
    save_metrics,
    tokenize_dataset,
)

MODEL_NAME = "google/mobilebert-uncased"
MODEL_DIR = "fine_tuned_models/mobilebert"
METRICS_DIR = "data/03_metrics/mobilebert"


X_train, X_test, y_train, y_test, label_encoder = load_and_prepare_data(
    label_encoder_path=f"{MODEL_DIR}/label_encoder.pkl",
)

# Load model and tokenizer
tokenizer = MobileBertTokenizer.from_pretrained(MODEL_NAME)
model = MobileBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Prepare dataset and trainer
train_dataset = tokenize_dataset(tokenizer, X_train, y_train)
test_dataset = tokenize_dataset(tokenizer, X_test, y_test)

training_args = TrainingArguments(
    output_dir=f"{MODEL_DIR}/results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# Save model and evaluate results
model.save_pretrained(f"{MODEL_DIR}/fine_tuned_mobilebert")
tokenizer.save_pretrained(f"{MODEL_DIR}/fine_tuned_mobilebert")

device = get_device()
model.eval()
model.to(device)

test_inputs = tokenizer(
    X_test.tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**test_inputs)
    predictions = torch.argmax(outputs.logits, dim=1).cpu().tolist()

decoded_predictions = label_encoder.inverse_transform(predictions)
decoded_true_labels = label_encoder.inverse_transform(y_test.tolist())

save_metrics(
    METRICS_DIR, decoded_true_labels, decoded_predictions, label_encoder, MODEL_NAME
)
