import dspy
import random

import pandas as pd
from tqdm import tqdm

generator = dspy.LM(
    "ollama_chat/gemma3:27b",
    api_base="http://localhost:11434",
    temperature=0.9,
    max_tokens=200,
)
dspy.configure(lm=generator)


class SyntheticSuicidalPost(dspy.Signature):
    """Generate a realistic and unique social media post for NLP research on mental health detection.
    Each post should be different in tone, length, writing style, and context."""

    label: str = dspy.InputField(desc="Either 'suicide' or 'non-suicide'")
    variation_hint: str = dspy.InputField(
        desc="A hint to vary the output e.g. 'teenager, late night, short post'"
    )
    post: str = dspy.OutputField(
        desc="A unique realistic social media post matching the label"
    )


class DataGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.Predict(SyntheticSuicidalPost)

    def forward(self, label: str, variation_hint: str):  # add variation_hint here
        return self.generate(label=label, variation_hint=variation_hint)


def run_generator(file_name):
    """Generate synthetic data for NLP research on mental health detection."""
    generator_module = DataGenerator()
    synthetic_data = []

    tones = [
        "short and vague",
        "long and detailed",
        "angry",
        "hopeless",
        "casual",
        "crying for help",
    ]
    contexts = [
        "teenager",
        "adult",
        "late night",
        "after argument",
        "chronic illness",
        "job loss",
    ]

    labels = ["suicide"] * 100 + ["non-suicide"] * 100

    for i, label in enumerate(tqdm(labels, desc="Generating data")):
        hint = f"{random.choice(contexts)}, {random.choice(tones)}"
        result = generator_module(label=label, variation_hint=hint)

        if result.post:
            synthetic_data.append({"text": result.post, "label": label})
        else:
            tqdm.write(f"Skipped empty output at step {i}, empty response")

    df = pd.DataFrame(synthetic_data)
    df = df.drop_duplicates()
    df.to_csv(file_name, sep="\t", index=False)


if __name__ == "__main__":
    output_file = "data/04_synthetic_data/synthetic_data.tsv"
    run_generator(output_file)
