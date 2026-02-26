import os, json, glob
import numpy as np
import matplotlib.pyplot as plt
import niceplots

def load_all_norm_hist(cache_dir=".mg_cache/norm_hist"):
    files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    runs = []
    for fp in files:
        try:
            d = np.load(fp, allow_pickle=False)
            norm_hist = d["norm_hist"].astype(float)
            meta = json.loads(str(d["meta_json"]))
            runs.append({"file": fp, "norm_hist": norm_hist, "meta": meta})
        except Exception as e:
            print(f"SKIP {fp}: {e}")
    return runs


def make_label(meta, fields):
    parts = []
    for f in fields:
        if f in meta:
            parts.append(f"{f}={meta[f]}")
    return ", ".join(parts)


def filter_runs(runs, must_match=None):
    """
    must_match: dict like {"load":"axial", "nxe":32, "solve":"kmg"}
    """
    if not must_match:
        return runs
    out = []
    for r in runs:
        ok = True
        for k, v in must_match.items():
            if r["meta"].get(k) != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out

def plot_runs(
    runs,
    title=None,
    label_fields=("elem", "nsmooth"),
    y_floor=1e-16,
    save_png=None,
):
    if not runs:
        print("No runs to plot.")
        return

    plt.style.use(niceplots.get_style())

    fig, ax = plt.subplots(figsize=(8, 6))

    ymin_global = 1e100
    ymax_global = 0.0

    for r in runs:
        nh = r["norm_hist"]
        it = np.arange(len(nh))
        y = np.maximum(nh, y_floor)

        ymin_global = min(ymin_global, np.min(y))
        ymax_global = max(ymax_global, np.max(y))

        lbl = ", ".join(
            f"{f}={r['meta'][f]}" for f in label_fields if f in r["meta"]
        )

        ax.semilogy(it, y, label=lbl)

    ax.set_xlabel("PCG iteration")
    ax.set_ylabel("Relative ||r||")

    if title is not None:
        ax.set_title(title)

    # ---- Add reference horizontal lines ----
    ax.axhline(1.0, linestyle="--", linewidth=1.0, color="gray", alpha=0.5)
    ax.axhline(1e-6, linestyle="--", linewidth=1.0, color="gray", alpha=0.5)

    # ---- Remove minor ticks and grid ----
    ax.minorticks_off()
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(False, which="minor")

    # ---- Manually construct log ticks as 10^{p} ----
    import math

    pmax = math.ceil(math.log10(ymax_global))
    pmin = math.floor(math.log10(ymin_global))

    powers = list(range(pmin, pmax + 1))
    yticks = [10.0 ** p for p in powers]

    ax.set_yticks(yticks)
    ax.set_yticklabels([rf"$10^{{{p}}}$" for p in powers])

    ax.set_ylim(10.0 ** pmin, 10.0 ** pmax)

    # ---- Legend on top ----
    ax.legend(
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()

    if save_png:
        fig.savefig(save_png, dpi=200, bbox_inches="tight")
        print(f"Saved {save_png}")
    else:
        plt.show()

if __name__ == "__main__":
    runs = load_all_norm_hist(".mg_cache/norm_hist")

    # change limiting plot settings
    # -------------------

    # load, nsmooth = 'axial', 1
    load, nsmooth = 'axial', 2
    # load, nsmooth = 'shear', 2

    # ---- Example filters you can tweak ----
    # Only compare a fixed nxe/thick/load/solve:

    runs = filter_runs(runs, must_match={
        # "solve": "kmg",
        "load": load,
        # "nxe": 32,
        # "thick": 1e-3,
        # "clamped": True,
        "nsmooth" : nsmooth
        # "nsmooth" : 2
    })

    # Optional: sort runs so legends are stable (e.g., by elem then omega)
    runs.sort(key=lambda r: (
        r["meta"].get("elem", ""),
        r["meta"].get("smoother", ""),
        r["meta"].get("coupled", 0),
        r["meta"].get("omega", 0.0),
        r["meta"].get("nsmooth", 0),
        r["meta"].get("nprolong", 0),
    ))

    plot_runs(
        runs,
        # title="Convergence comparison (cached norm_hist)",
        # label_fields=("elem", "smoother", "coupled", "omega", "nsmooth", "nprolong", "line_search"),
        label_fields=("elem", "nsmooth"),
        save_png=f"sphere_hist_{load}{nsmooth}.png",
        # save_png="cyl_hist_compare2.png",
    )