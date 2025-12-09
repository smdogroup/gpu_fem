import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import niceplots

# --- Load CSV data ---
# case = 'plate'
# case = 'cylinder'

# csv_file = 'out/_plate.csv'  # change to your CSV filename
# csv_file = 'out/cyl_cfi4_vs_mitc4.csv'
csv_file = 'out/plate_cfi4_vs_mitc4.csv'

df = pd.read_csv(csv_file)


plt.style.use(niceplots.get_style())
ms = 7 # marker size
# ms = 8

# fs = 24
fs = 20
# fs = 22

# text_fs = 12
text_fs = 14

plt.rcParams.update({
    'font.size': fs,
    'axes.titlesize': fs,
    'axes.labelsize': fs,
    'xtick.labelsize': fs,
    'ytick.labelsize': fs,
    'legend.fontsize': fs,
    'figure.titlesize': fs,
})

size = (8,6.5) # figsize

# --- Plot 2: NDOF vs linear runtime, mask by element type ---
# plt.figure(figsize=size)
fig, ax = plt.subplots(figsize=(8, 6.5))

six_colors1 = ["#d95a72", "#eb9a79", "#f3d27d", "#3cc7a1", "#3b90b3", "#1a4c60", "tab:gray"]
colors = six_colors1


elem_order = ['CFI4', 'MITC4'][::-1]


i = 0
for elem_type in elem_order:
    if elem_type in df['elem'].unique():
        group = df[df['elem'] == elem_type]
        plt.plot(1.0/group['SR'],
                 group['lin_time'],
                 marker='o', #linestyle='--' if elem_type in ['HR4', 'LFI16'] else '-',
                 #color=colors[i],
                 label=elem_type)
        i += 1

# for elem_type, group in df.groupby('elem_type'):
#     plt.plot(group['NDOF'], group['lin_runtime(s)'], marker='o', linestyle='-', label=elem_type)
plt.xlabel("Normalized Thickness t/" + r"$L_{ref}$")
plt.ylabel('Linear runtime (s)')
# plt.title('Linear runtime vs NDOF by element type')
plt.xscale('log')  # log scale can help visualize wide range of NDOF
# plt.yscale('log')  # optional: runtime varies widely
plt.legend()
plt.grid(True, which='major', ls='--')
plt.tight_layout()
# plt.show()
plt.margins(x=0.05, y=0.05)

xticks = ax.get_xticks()
xlabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in xticks]
ax.set_xticklabels(xlabels)
# yticks = ax.get_yticks()
# ylabels = [f"$10^{{{int(np.log10(y))}}}$" if y > 0 else "" for y in yticks]
# ax.set_yticklabels(ylabels)

geom = "cylinder" if "cyl" in csv_file else "plate"
plt.savefig(f"out/{geom}_elem_compare.png", dpi=400)
plt.close('all')
