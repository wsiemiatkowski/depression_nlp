import os
from pathlib import Path

models = {
    "albert": (
        "fine_tuned_models/albert/fine_tuned_albert",
        "mobile_models/albert_onnx/model.onnx",
    ),
    "tinybert": (
        "fine_tuned_models/tinybert/fine_tuned_tinybert",
        "mobile_models/tinybert_onnx/model.onnx",
    ),
    "mobilebert": (
        "fine_tuned_models/mobilebert/fine_tuned_mobilebert",
        "mobile_models/mobilebert_onnx/model.onnx",
    ),
}


def get_dir_size(path: str) -> int:
    """Get total size of all files in a directory."""
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def bytes_to_mb(size: int) -> float:
    """Convert bytes to MB"""
    return size / (1024 * 1024)


print(f"{'Model':<12} {'Original (MB)':>15} {'ONNX (MB)':>12} {'Reduction':>12}")
print("-" * 55)

output_lines = []
output_lines.append(
    f"{'Model':<12} {'Original (MB)':>15} {'ONNX (MB)':>12} {'Reduction':>12}"
)
output_lines.append("-" * 55)

for model_name, (original_path, onnx_path) in models.items():
    original_size = get_dir_size(original_path)
    onnx_size = os.path.getsize(onnx_path)
    reduction = (1 - onnx_size / original_size) * 100
    output_lines.append(
        f"{model_name:<12} {bytes_to_mb(original_size):>15.1f} {bytes_to_mb(onnx_size):>12.1f} {reduction:>11.1f}%"
    )

output = "\n".join(output_lines)
print(output)

with open("mobile_models/model_size_comparison.txt", "w") as f:
    f.write(output)
