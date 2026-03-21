import pandas as pd

# Load the dataset
df = pd.read_csv("data/01_raw/Suicide_Detection.csv", index_col=0)

# Analyze dataset
print(f"Total examples: {len(df)}")
print(f"{df["class"].value_counts()}")

null_classes = df["class"].isnull().any()
null_text = df["text"].isnull().any()

print(f"Null classes:\n{null_classes}")
print(f"Null text:\n{null_text}")

# Create sub dataset with 6000 examples
suicide = df[df["class"] == "suicide"].sample(n=7500, random_state=42)
non_suicide = df[df["class"] == "non-suicide"].sample(n=7500, random_state=42)

df_sample = (
    pd.concat([suicide, non_suicide])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

df_sample.to_csv("data/02_training_data/depression_nlp.tsv", sep="\t", index=False)

print(f"\nSaved {len(df_sample)} examples to depression_nlp.tsv")
print(df_sample["class"].value_counts())
