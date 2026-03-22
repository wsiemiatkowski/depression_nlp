import onnxruntime as ort
import numpy as np
import joblib
from transformers import AutoTokenizer

models = {
    "albert": (
        "mobile_models/albert_onnx/model.onnx",
        "fine_tuned_models/albert/fine_tuned_albert",
        "fine_tuned_models/albert/label_encoder.pkl",
    ),
    "tinybert": (
        "mobile_models/tinybert_onnx/model.onnx",
        "fine_tuned_models/tinybert/fine_tuned_tinybert",
        "fine_tuned_models/tinybert/label_encoder.pkl",
    ),
    "mobilebert": (
        "mobile_models/mobilebert_onnx/model.onnx",
        "fine_tuned_models/mobilebert/fine_tuned_mobilebert",
        "fine_tuned_models/mobilebert/label_encoder.pkl",
    ),
}

for model_name, (onnx_path, tokenizer_path, encoder_path) in models.items():
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    label_encoder = joblib.load(encoder_path)
    session = ort.InferenceSession(onnx_path)

    inputs = tokenizer(
        "I feel hopeless",
        return_tensors="np",
        padding="max_length",
        max_length=128,
        truncation=True,
    )

    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        },
    )

    predicted_class = label_encoder.inverse_transform([np.argmax(outputs[0])])[0]
    print(f"{model_name}: {predicted_class}")
