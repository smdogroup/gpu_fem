import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("qorder-times.csv")
df['fill_type'] = df['fill_type'].str.strip()

# Set style and color palette
custom_palette = ["#264653", "#287271", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
sns.set(style="whitegrid", context="paper", font_scale=1.8)

# Plot
plt.figure(figsize=(8, 6))
sns.lineplot(
    data=df,
    x='p_factor',
    y='runtime(s)',
    hue='fill_type',
    marker='o',
    linewidth=2.5,
    markersize=9,
    palette=custom_palette
)

# Axis labels and formatting
plt.xlabel('qordering p_factor', fontsize=16, fontweight='bold')
plt.ylabel('Runtime (s)', fontsize=16, fontweight='bold')
plt.legend(title='ILU Fill', fontsize=13, title_fontsize=14)
# plt.grid(True)
plt.xscale('log')
plt.tight_layout()
plt.grid(False)

# Save as SVG with high resolution
plt.savefig("qorder_runtime.svg", format='svg', dpi=400)
plt.show()
