import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("barchart-times.csv", skipinitialspace=True)

# Remove bad/irrelevant data
df = df[~df['fill_type'].isin(['ILU(0)', 'ILU(1)'])]
df = df[~((df['solver'] == 'LU') & (df['ordering'].isin(['qorder', 'RCM'])))]

# Mark which runs succeeded
df['success'] = df['resid_norm'] <= 1e0

# Define all possible x and hue categories
x_order = df['fill_type'].unique().tolist()
hue_order = df['ordering'].unique().tolist()

# Plot setup
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
sns.set_context("talk", font_scale=1.2)  # Increase general font size

# Bar plot for successful runs, forcing x/hue order
sns.barplot(
    data=df[df['success']],
    x='fill_type', y='runtime(s)',
    hue='ordering',
    dodge=True, palette="viridis",
    order=x_order, hue_order=hue_order
)

# Get width of each dodge offset
n_hue = len(hue_order)
width = 0.8 / n_hue  # Seaborn default total width = 0.8

# Add X markers for failed runs
for _, row in df[~df['success']].iterrows():
    try:
        xpos = x_order.index(row['fill_type'])
        offset_idx = hue_order.index(row['ordering'])
        xloc = xpos - 0.4 + (offset_idx + 0.5) * width
        plt.text(
            x=xloc, y=0.05, s='X',
            color='red', ha='center', va='bottom',
            fontsize=14, weight='bold'
        )
    except ValueError:
        continue

# Axis labels and legend formatting
plt.ylabel('Runtime (s)', fontsize=16, weight='bold')
plt.xlabel('Fill Type', fontsize=16, weight='bold')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(title='Ordering', fontsize=13, title_fontsize=14)

plt.tight_layout()
plt.savefig("cylinder_runtimes.svg", dpi=400)
