import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import niceplots

# ==========================
# Settings
# ==========================
csv_file = "out/cylinder-A100-times.csv"
NXE_TARGET = 128
combine_dupes = "min"   # "min" / "mean" / "max" for duplicate solver rows (CP appears twice)
y_log = False           # True -> log y like old figure; False -> linear runtime

# ==========================
# Load + filter
# ==========================
df = pd.read_csv(csv_file)
df.columns = [c.strip() for c in df.columns]
df["solver"] = df["solver"].astype(str).str.strip()

df = df[df["nxe"] == NXE_TARGET].copy()
if df.empty:
    raise RuntimeError(f"No rows found with nxe == {NXE_TARGET} in {csv_file}")

df["t/R"] = pd.to_numeric(df["t/R"], errors="coerce")
df["lin_runtime(s)"] = pd.to_numeric(df["lin_runtime(s)"], errors="coerce")
df = df.dropna(subset=["t/R", "lin_runtime(s)", "solver"])

# ==========================
# Solver name mapping (match your desired labels)
# ==========================
solver_label_map = {
    "LU": "LU",
    "ILU0": "ILU0",
    "ILU1": "ILU1",
    "ILU2": "ILU2",
    "CP": "CP8",
    "GSMC": "GSMC",
    "MCGS": "GSMC",
    "ASW": "ASW",
    # "JACOBI": "Jacobi",
    "DJ": "Jacobi",
}
df["solver_plot"] = df["solver"].map(lambda s: solver_label_map.get(s.strip(), s.strip()))

# ==========================
# Combine duplicates per (t/R, solver_plot)
# ==========================
aggfunc = {"min": "min", "mean": "mean", "max": "max"}[combine_dupes]
dfp = (
    df.groupby(["t/R", "solver_plot"], as_index=False)["lin_runtime(s)"]
      .agg(aggfunc)
      .rename(columns={"lin_runtime(s)": "runtime"})
)

# ==========================
# Plot styling
# ==========================
plt.style.use(niceplots.get_style())

fs = 18
text_fs = 14
plt.rcParams.update({
    "font.size": fs,
    "axes.titlesize": fs,
    "axes.labelsize": fs,
    "xtick.labelsize": fs,
    "ytick.labelsize": fs,
    "legend.fontsize": fs,
    "figure.titlesize": fs,
})

fig, ax = plt.subplots(figsize=(8.5, 7))

# This is your palette list (ordered)
six_colors1 = ["#d95a72", "#eb9a79", "#f3d27d", "#3cc7a1", "#3b90b3", "#1a4c60", "k"]

# --------------------------
# ORDER + COLORS (positional!)
# You said:
#   ASW is the first color, CP8 second, etc.
#   and black is ILU0
# --------------------------
plot_order = ["ASW", "CP8", "GSMC", "Jacobi", "ILU2", "ILU1", "ILU0", "LU"]

# Assign colors *by index in plot_order*
# first six use six_colors1[0..5], ILU0 is black, LU is gray
plot_colors = {
    "ASW":    six_colors1[0],
    "CP8":    six_colors1[1],
    "GSMC":   six_colors1[2],
    "Jacobi": six_colors1[3],
    "ILU2":   six_colors1[4],
    "ILU1":   six_colors1[5],
    "ILU0":   "k",          # <-- as requested
    "LU":     "tab:gray",
}

# line style tweaks (like your old script: GSMC dashed)
linestyle = {"GSMC": "--"}

# ==========================
# Helper for callouts (initial & final)
# ==========================
def callouts(x, y, color, left=True, y_mult=1.04):
    if len(x) == 0:
        return
    if left:
        x_pt, y_pt = x[0], y[0]
        x_off = 0.7
    else:
        x_pt, y_pt = x[-1], y[-1]
        x_off = 1.2
    if np.isnan(y_pt):
        return
    ax.text(
        x_pt * x_off,
        y_pt * y_mult,
        f"{y_pt:.3f}",
        color=color,
        fontsize=text_fs,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

# ==========================
# Plot each solver (in order)
# ==========================
for name in plot_order:
    sub = dfp[dfp["solver_plot"] == name].sort_values("t/R")
    if sub.empty:
        continue

    x = sub["t/R"].to_numpy()
    y = sub["runtime"].to_numpy()

    ax.plot(
        x, y,
        "o-" if name != "GSMC" else "o--",
        linewidth=3,
        markersize=8,
        color=plot_colors.get(name, None),
        label=name,
        linestyle=linestyle.get(name, "-"),
    )

    callouts(x, y, plot_colors.get(name, "k"), left=True,  y_mult=1.04)
    callouts(x, y, plot_colors.get(name, "k"), left=False, y_mult=1.04)

# ==========================
# Axes / labels
# ==========================
handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], loc="lower left")
ax.legend(handles[::-1], labels[::-1], loc="upper right")

ax.set_xlabel("Normalized Thickness, t/R")
ax.set_xscale("log")
ax.set_ylabel("Linear runtime (s)")

if y_log:
    ax.set_yscale("log")

ax.grid(True, which="major", ls="--", alpha=0.5)
ax.set_xmargin(0.025)
ax.set_ymargin(0.05)

xticks = ax.get_xticks()
ax.set_xticklabels([f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in xticks])

plt.tight_layout()
plt.savefig("out/1_cylinder_A100_lin_runtime_nxe128.png", dpi=400)
plt.close("all")

print("Wrote: out/1_cylinder_A100_lin_runtime_nxe128.png")
print("Legend order:", plot_order)
print("Plotted solvers:", sorted(dfp["solver_plot"].unique().tolist()))
