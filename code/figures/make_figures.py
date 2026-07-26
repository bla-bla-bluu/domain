"""
Generate results figures for main.tex from the Table III / Table IV numbers.
Categorical color mapping is fixed per entity across all figures (never reassigned
per-chart): Baseline=blue, CycleGAN v1=aqua, CycleGAN v2=yellow, Oracle=green
(palette slots 1-4 of the validated default categorical theme). Hatch patterns and
distinct markers/linestyles are layered on top of color so the figures remain
legible in grayscale print.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT = "/home/deepak/domain"

COLOR = {"Baseline": "#2a78d6", "v1": "#1baf7a", "v2": "#eda100", "SelfTrain": "#7b3294", "Oracle": "#008300"}
HATCH = {"Baseline": "", "v1": "///", "v2": "xx", "SelfTrain": "oo", "Oracle": ".."}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "svg.fonttype": "none",
})

# ---------------------------------------------------------------------------
# Figure A: Recall, mAP50, and SAHI's delta-mAP50 by model
# ---------------------------------------------------------------------------
models = ["Baseline", "v1", "v2", "SelfTrain", "Oracle"]
label = {"Baseline": "Baseline", "v1": "CycleGAN v1", "v2": "CycleGAN v2",
         "SelfTrain": "Self-Train", "Oracle": "Oracle"}

recall_std = {"Baseline": 40.9, "v1": 11.7, "v2": 26.8, "SelfTrain": 86.0, "Oracle": 96.2}
recall_std_ci = {"Baseline": (34.2, 47.4), "v1": None, "v2": (21.6, 32.0),
                  "SelfTrain": (81.8, 89.8), "Oracle": (94.1, 98.1)}
recall_sahi = {"Baseline": 63.0, "v1": 32.3, "v2": 28.5, "SelfTrain": 71.5, "Oracle": 91.7}
recall_sahi_ci = {"Baseline": (58.1, 67.8), "v1": None, "v2": (23.1, 34.0),
                   "SelfTrain": (66.4, 76.2), "Oracle": (88.9, 94.3)}

map50_std = {"Baseline": 35.3, "v1": 18.7, "v2": 33.7, "SelfTrain": 86.5, "Oracle": 96.9}
map50_sahi = {"Baseline": 79.3, "v1": None, "v2": 43.6, "SelfTrain": 88.7, "Oracle": 94.7}

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))

def grouped_bars(ax, std_vals, sahi_vals, std_ci, sahi_ci, ylabel, title):
    x = np.arange(len(models))
    w = 0.36
    for i, m in enumerate(models):
        # standard bar
        ax.bar(x[i] - w/2, std_vals[m], width=w, color=COLOR[m], hatch=HATCH[m],
               edgecolor=INK, linewidth=0.6, alpha=0.55)
        # sahi bar
        v = sahi_vals.get(m)
        if v is not None:
            ax.bar(x[i] + w/2, v, width=w, color=COLOR[m], hatch=HATCH[m],
                   edgecolor=INK, linewidth=0.6, alpha=1.0)
        # CI error bars
        if std_ci.get(m):
            lo, hi = std_ci[m]
            ax.errorbar(x[i] - w/2, std_vals[m], yerr=[[std_vals[m]-lo], [hi-std_vals[m]]],
                        fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5)
        if sahi_ci.get(m) and v is not None:
            lo, hi = sahi_ci[m]
            ax.errorbar(x[i] + w/2, v, yerr=[[v-lo], [hi-v]],
                        fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5)
        # direct labels (offset above whichever is higher: bar top or CI whisker)
        std_top = std_ci[m][1] if std_ci.get(m) else std_vals[m]
        ax.text(x[i]-w/2, std_top+3.2, f"{std_vals[m]:.0f}", ha="center", fontsize=6.5, color=MUTED)
        if v is not None:
            sahi_top = sahi_ci[m][1] if sahi_ci.get(m) else v
            ax.text(x[i]+w/2, sahi_top+3.2, f"{v:.0f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([label[m] for m in models], fontsize=7.5, rotation=12)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 108)
    ax.spines[["top", "right"]].set_visible(False)

grouped_bars(axes[0], recall_std, recall_sahi, recall_std_ci, recall_sahi_ci,
             "Recall (%)", "(a) Recall: Standard vs. SAHI")
grouped_bars(axes[1], map50_std, map50_sahi, {}, {},
             "mAP50 (%)", "(b) mAP50: Standard vs. SAHI")

# Panel (c): delta mAP50 from SAHI (diverging)
ax = axes[2]
deltas, cols = [], []
dmodels = [m for m in models if map50_sahi.get(m) is not None]
for m in dmodels:
    d = map50_sahi[m] - map50_std[m]
    deltas.append(d)
    cols.append("#2a78d6" if d >= 0 else "#e34948")
y = np.arange(len(dmodels))
ax.barh(y, deltas, color=cols, edgecolor=INK, linewidth=0.6, height=0.55)
ax.axvline(0, color=INK, linewidth=0.8)
for yi, d in zip(y, deltas):
    ax.text(d + (1.5 if d >= 0 else -1.5), yi, f"{d:+.1f}", va="center",
            ha="left" if d >= 0 else "right", fontsize=7.5, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([label[m] for m in dmodels], fontsize=7.5)
ax.set_xlabel("$\\Delta$ mAP50, SAHI $-$ Standard (pp)", fontsize=8.5)
ax.set_title("(c) SAHI helps scale-limited\nmodels, hurts the oracle", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-15, 55)

# shared legend: standard (light) vs sahi (solid) swatch
handles = [
    plt.Rectangle((0,0),1,1, facecolor="none", edgecolor=INK, alpha=0.55, hatch="", linewidth=0.6, label="Standard (light)"),
    plt.Rectangle((0,0),1,1, facecolor=INK, edgecolor=INK, label="+ SAHI (solid)"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(f"{OUT}/results_comparison.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/results_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote results_comparison.pdf/.png")

# ---------------------------------------------------------------------------
# Figure B: SAHI tile-size sweep (Recall vs S), per model
# ---------------------------------------------------------------------------
S = [640, 480, 320, 256, 160]
sweep = {
    "Baseline":  [40.9, 40.0, 63.0, 69.4, 32.8],
    "v2":        [26.8, 31.3, 28.5, 21.3, 15.5],
    "SelfTrain": [86.0, 73.0, 71.5, 71.9, 75.1],
    "Oracle":    [96.2, 91.1, 91.7, 88.9, 94.3],
}
marker = {"Baseline": "o", "v2": "s", "SelfTrain": "D", "Oracle": "^"}
linestyle = {"Baseline": "-", "v2": "--", "SelfTrain": "-.", "Oracle": ":"}

fig2, ax = plt.subplots(figsize=(5.2, 3.6))
x = np.arange(len(S))
for m in ["Baseline", "v2", "SelfTrain", "Oracle"]:
    ax.plot(x, sweep[m], color=COLOR[m], marker=marker[m], linestyle=linestyle[m],
            linewidth=1.8, markersize=6, markeredgecolor=INK, markeredgewidth=0.5,
            label=label[m])
    # mark the per-model peak
    peak_i = int(np.argmax(sweep[m]))
    ax.annotate(f"{sweep[m][peak_i]:.1f}", (x[peak_i], sweep[m][peak_i]),
                textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5, color=COLOR[m])

ax.set_xticks(x)
ax.set_xticklabels([f"{s}\n(std)" if s == 640 else str(s) for s in S], fontsize=8)
ax.set_xlabel("SAHI tile size $S$ (px)", fontsize=9)
ax.set_ylabel("Recall (%)", fontsize=9)
ax.set_title("Optimal tile size is model-dependent", fontsize=10)
ax.set_ylim(0, 105)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig2.tight_layout()
fig2.savefig(f"{OUT}/tile_size_sweep.pdf", bbox_inches="tight")
fig2.savefig(f"{OUT}/tile_size_sweep.png", dpi=200, bbox_inches="tight")
plt.close(fig2)
print("wrote tile_size_sweep.pdf/.png")
