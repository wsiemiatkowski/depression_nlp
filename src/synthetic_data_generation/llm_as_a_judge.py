import dspy
import pandas as pd

judge_lm = dspy.LM("ollama_chat/gpt-oss:20b", api_base="http://localhost:11434")
dspy.configure(lm=judge_lm)


class JudgeSignature(dspy.Signature):
    """Evaluate whether a social media post genuinely reflects the given mental health label.
    Be strict - flag anything that feels synthetic, stereotyped, or unrealistic."""

    post: str = dspy.InputField(desc="The social media post to evaluate")
    label: str = dspy.InputField(desc="The intended label: 'suicide' or 'non-suicide'")
    is_valid: bool = dspy.OutputField(
        desc="True if post is realistic and matches label"
    )
    reasoning: str = dspy.OutputField(desc="Brief explanation of the decision")
    quality_score: float = dspy.OutputField(desc="Quality score from 0.0 to 1.0")


class LLMJudge(dspy.Module):
    def __init__(self):
        self.judge = dspy.ChainOfThought(JudgeSignature)

    def forward(self, post: str, label: str):
        return self.judge(post=post, label=label)


def run_judge(input_path: str, output_path: str, quality_threshold: float = 0.8):
    df = pd.read_csv(input_path, sep="\t")
    judge = LLMJudge()

    results = []
    for _, row in df.iterrows():
        verdict = judge(post=row["text"], label=row["label"])
        results.append(
            {
                "text": row["text"],
                "label": row["label"],
                "is_valid": verdict.is_valid,
                "quality_score": verdict.quality_score,
                "reasoning": verdict.reasoning,
            }
        )

    results_df = pd.DataFrame(results)

    approved_df = results_df[
        (results_df["is_valid"] == True)
        & (results_df["quality_score"] >= quality_threshold)
    ][["text", "label"]]
    approved_df.to_csv(output_path, sep="\t", index=False)

    print(f"\nTotal: {len(results_df)}")
    print(f"Approved: {len(approved_df)} ({len(approved_df)/len(results_df):.1%})")
    print(f"Rejected: {len(results_df) - len(approved_df)}")


if __name__ == "__main__":
    run_judge(
        input_path="data/04_synthetic_data/synthetic_data.tsv",
        output_path="data/04_synthetic_data/synthetic_data_judged.tsv",
    )
