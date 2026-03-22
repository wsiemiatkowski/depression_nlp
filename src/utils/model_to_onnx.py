import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = {
    "albert": "fine_tuned_models/albert/fine_tuned_albert",
    "tinybert": "fine_tuned_models/tinybert/fine_tuned_tinybert",
    "mobilebert": "fine_tuned_models/mobilebert/fine_tuned_mobilebert",
}

MAX_LENGTH = 128


def convert_to_onnx(model_name: str, model_path: str, output_dir: str):
    print(f"Converting {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    dummy_input = tokenizer(
        "example text",
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
    )

    output_path = Path(output_dir) / "model.onnx"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
    )
    print(f"Saved to {output_path}")


def main():
    for model_name, model_path in MODELS.items():
        convert_to_onnx(model_name, model_path, f"mobile_models/{model_name}_onnx")


if __name__ == "__main__":
    main()
