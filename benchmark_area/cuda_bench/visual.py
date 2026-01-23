import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "./benchmark_results.csv"
out_dir = "."
out_path = os.path.join(out_dir, "benchmark_runtimes_vs_num_keys.png")

df = pd.read_csv(csv_path)

required_cols = {"num_keys", "branching_factor"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(
        f"Expected columns {sorted(required_cols)} in {csv_path}, missing: {sorted(missing)}"
    )

method_cols = [c for c in df.columns if c not in {"num_keys", "branching_factor"}]
if not method_cols:
    raise ValueError(f"No method columns found in {csv_path}.")


def best_by_num_keys(method: str) -> pd.DataFrame:
    # For each num_keys, select the row (branching_factor) with the minimum time.
    idx = df.groupby("num_keys", sort=True)[method].idxmin()
    best = df.loc[idx, ["num_keys", "branching_factor", method]].copy()
    best = best.sort_values("num_keys")
    best = best.rename(columns={method: "mean_ms"})
    best["method"] = method
    return best


best_long = pd.concat([best_by_num_keys(m) for m in method_cols], ignore_index=True)

os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(8, 4.5))
for method, g in best_long.groupby("method"):
    g = g.sort_values("num_keys")
    plt.plot(g["num_keys"], g["mean_ms"], marker="o", linewidth=2, label=method)

    # Annotate each point with the branching factor that achieved the best time.
    for x, y, bf in zip(g["num_keys"], g["mean_ms"], g["branching_factor"], strict=False):
        plt.annotate(
            f"b={int(bf)}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            alpha=0.9,
        )

plt.title("Benchmark runtimes vs num_keys")
plt.xlabel("num_keys")
plt.ylabel("Mean time (ms)")
plt.grid(True, alpha=0.3)
plt.legend(title="Method", frameon=False)
plt.tight_layout()

plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()

print("Saved:", out_path)
