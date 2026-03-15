from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATH
# ============================================================

CSV_PATH = r"C:\Users\revit\מסמכים\Data Science\M.Sc. Data Science\First Year\First semester\Data Streaming Algorithms\Final Project\kmeans5\results_final\Kmeans_sweep_raw.csv"

OUTPUT_DIR = os.path.dirname(CSV_PATH)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Loaded rows:", len(df))


# ============================================================
# AGGREGATE RESULTS
# ============================================================

agg = df.groupby("algorithm").agg({
    "runtime_sec": "mean",
    "memory": "mean",
    "cost_sse": "mean",
    "ari": "mean",
    "nmi": "mean"
}).reset_index()

agg = agg.sort_values("ari", ascending=False)

# save table
table_path = os.path.join(OUTPUT_DIR, "Table1_Aggregated_Results.csv")
agg.to_csv(table_path, index=False)

print("Saved table:", table_path)


# ============================================================
# FIGURE 1 — Quality vs Memory
# ============================================================

plt.figure()

plt.scatter(agg["memory"], agg["ari"])

for i, row in agg.iterrows():
    plt.text(row["memory"], row["ari"], row["algorithm"])

plt.xlabel("Memory Usage")
plt.ylabel("Clustering Quality (ARI)")
plt.title("Quality vs Memory")

fig1 = os.path.join(OUTPUT_DIR, "Figure1_Quality_vs_Memory.png")
plt.savefig(fig1, bbox_inches="tight")
plt.close()

print("Saved:", fig1)


# ============================================================
# FIGURE 2 — Quality vs Runtime
# ============================================================

plt.figure()

plt.scatter(agg["runtime_sec"], agg["ari"])

for i, row in agg.iterrows():
    plt.text(row["runtime_sec"], row["ari"], row["algorithm"])

plt.xlabel("Runtime (seconds)")
plt.ylabel("Clustering Quality (ARI)")
plt.title("Quality vs Runtime")

fig2 = os.path.join(OUTPUT_DIR, "Figure2_Quality_vs_Runtime.png")
plt.savefig(fig2, bbox_inches="tight")
plt.close()

print("Saved:", fig2)


# ============================================================
# FIGURE 3 — Runtime vs Memory
# ============================================================

plt.figure()

plt.scatter(agg["memory"], agg["runtime_sec"])

for i, row in agg.iterrows():
    plt.text(row["memory"], row["runtime_sec"], row["algorithm"])

plt.xlabel("Memory Usage")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs Memory")

fig3 = os.path.join(OUTPUT_DIR, "Figure3_Runtime_vs_Memory.png")
plt.savefig(fig3, bbox_inches="tight")
plt.close()

print("Saved:", fig3)

print("\nDone. All results saved to:")
print(OUTPUT_DIR)