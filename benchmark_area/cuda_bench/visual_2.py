import os
import pandas as pd
import matplotlib.pyplot as plt

# Visualize triton_benchmarking.csv where each non-x column is named:
#   "methodName_branchingFactor"  (e.g., "two_4", "three_64")

csv_path = "./reports/triton_benchmarking.csv"
out_dir = "."
out_path = os.path.join(out_dir, "triton_benchmark_runtimes_vs_n.png")

x_col = "n"

df = pd.read_csv(csv_path)

if x_col not in df.columns:
    raise ValueError(f"Expected x column '{x_col}' in {csv_path}. Found: {list(df.columns)}")

value_cols = [c for c in df.columns if c != x_col]
if not value_cols:
    raise ValueError(f"No benchmark columns found in {csv_path} (expected columns like 'method_bf').")


def parse_method_and_bf(col_name: str) -> tuple[str, int]:
    # Split on the last underscore so method names may contain underscores.
    if "_" not in col_name:
        raise ValueError(
            f"Column '{col_name}' does not match expected pattern 'method_branchingFactor'."
        )
    method, bf_str = col_name.rsplit("_", 1)
    try:
        bf = int(bf_str)
    except ValueError as e:
        raise ValueError(
            f"Column '{col_name}' has non-integer branching factor '{bf_str}'."
        ) from e
    return method, bf


# Long-form: n, method, branching_factor, mean_ms
records: list[dict] = []
for col in value_cols:
    method, bf = parse_method_and_bf(col)
    records.append(
        pd.DataFrame(
            {
                x_col: df[x_col].astype(float),
                "method": method,
                "branching_factor": bf,
                "mean_ms": df[col].astype(float),
            }
        )
    )

long_df = pd.concat(records, ignore_index=True)

# For each (method, n), keep the branching factor with the minimum runtime.
idx = long_df.groupby(["method", x_col], sort=True)["mean_ms"].idxmin()
best_long = long_df.loc[idx].copy().sort_values(["method", x_col])

os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(8, 4.5))
for method, g in best_long.groupby("method"):
    g = g.sort_values(x_col)
    plt.plot(g[x_col], g["mean_ms"], marker="o", linewidth=2, label=method)

    # Annotate each point with the branching factor that achieved the best time.
    for x, y, bf in zip(g[x_col], g["mean_ms"], g["branching_factor"], strict=False):
        plt.annotate(
            f"b={int(bf)}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            alpha=0.9,
        )

plt.title("Triton benchmark runtimes vs n")
plt.xlabel(x_col)
plt.ylabel("Mean time (ms)")
plt.grid(True, alpha=0.3)
plt.legend(title="Method", frameon=False)
plt.tight_layout()

plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()

print("Saved:", out_path)
