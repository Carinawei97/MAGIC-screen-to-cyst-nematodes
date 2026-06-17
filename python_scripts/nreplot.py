import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load CSV
df = pd.read_csv("/Users/pro/Desktop/analysis/correlation_matrix_2.csv", index_col=0)

# Output directory
output_dir = "/Users/pro/Desktop/analysis/"
os.makedirs(output_dir, exist_ok=True)

# --- Plot 1: Full Square Heatmap ---
plt.figure(figsize=(10, 8))
ax1 = sns.heatmap(
    df,
    cmap="vlag",
    vmin=-1, vmax=1,
    square=True,
    linewidths=0,
    cbar_kws={"shrink": 0.75}
)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "square_heatmap_2.pdf"), format="pdf")
plt.savefig(os.path.join(output_dir, "square_heatmap_2.png"), format="png", dpi=300)
plt.close()

# --- Plot 2: Focus on 'Nematode size 13dpi' ---
target = 'Nematode size 13dpi'
focused_df = df[[target]].copy()  # Select only the column of interest
focused_df = focused_df.loc[[target] + [i for i in focused_df.index if i != target]]  # Reorder so target is at top

plt.figure(figsize=(6, len(focused_df) * 0.5))  # Dynamic height based on number of variables
ax2 = sns.heatmap(
    focused_df,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    vmin=-1, vmax=1,
    cbar_kws={"shrink": 0.5}
)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
plt.title(f"Correlations with '{target}'")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "focused_heatmap_13dpi.pdf"), format="pdf")
plt.savefig(os.path.join(output_dir, "focused_heatmap_13dpi.png"), format="png", dpi=300)
plt.close()

print("✅ Full and focused heatmaps saved.")