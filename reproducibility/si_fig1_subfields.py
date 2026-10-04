"""
Supplementary figure: consensus at the field scale vs the subfield scale,
one panel per measure, styled as Fig. 2 (reproducibility/fig2.py) with the
A/B/C panel labels of Fig. 1.

  A  venues in 12 consideration sets (Fig. 1A measure at N=12)
  B  mean consideration-set overlap  (Fig. 1B measure)
  C  top-five agreement              (Fig. 2 measure)

x = value computed over the whole field, y = size-weighted mean over the
field's subfields, both over the same respondents (see si_table1_subfields.py).
The dashed line is y = x. Fields that cannot be subdivided (History,
Medicine) are not plotted.

Input   reproducibility/output/results/subfield_consensus.csv   (from si_table1_subfields.py)
Output  reproducibility/output/figures/si_fig1.pdf   (SI Fig. S1)

Usage   cd <repo_root>
        .venv/bin/python reproducibility/si_table1_subfields.py   # first
        .venv/bin/python reproducibility/si_fig1_subfields.py
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

SRC = common.RESULTS / "subfield_consensus.csv"
if not SRC.exists():
    sys.exit(f"Missing {SRC}. Run si_table1_subfields.py first.")
d = pd.read_csv(SRC).set_index("field")

# ---------------------------------------------------------------------------
# Inlined utils/field_colors.py (as in fig2.py)
# ---------------------------------------------------------------------------
fields_to_use = [
    "Biology", "Business", "Sociology", "Computer science",
    "Economics", "Engineering", "History", "Mathematics",
    "Medicine", "Philosophy", "Physics", "Psychology", "Chemistry",
]
n_fields = len(fields_to_use)
field_color_map = {
    field: plt.get_cmap("tab20")(i / n_fields)
    for i, field in enumerate(fields_to_use)
}
abbrev = {f: f for f in fields_to_use}
abbrev["Computer science"] = "Comp. Sci."
abbrev["Mathematics"] = "Math"

plotted = [f for f in fields_to_use if bool(d.loc[f, "ok"])]   # drops History, Medicine

# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
# side: 1 = label to the right of the point, 0 = to the left, 2 = centred above
# tweak: vertical nudge in data units, per panel, hand-tuned as in fig2.py
PANELS = [
    dict(key="acc", label="A",
         xlabel="Field: Distinct Venues in 12 Consideration Sets",
         ylabel="Subfield: Distinct Venues in 12 Consideration Sets",
         pct=False, lim=(40, 150), ticks=np.arange(40, 141, 20),
         side={"Biology": 1, "Business": 1, "Chemistry": 1, "Computer science": 2,
               "Economics": 0, "Engineering": 0, "Mathematics": 1,
               "Philosophy": 0, "Physics": 1, "Psychology": 0, "Sociology": 0},
         tweak={"Economics": -3, "Philosophy": 2, "Computer science": 2,
                "Sociology": -6, "Biology": -3, "Mathematics": -4, "Psychology": 2}),
    dict(key="ovl", label="B",
         xlabel="Field: Mean Respondent Overlap",
         ylabel="Subfield: Mean Respondent Overlap",
         pct=True, lim=(0, 65), ticks=np.arange(0, 61, 10),
         side={"Biology": 1, "Business": 0, "Chemistry": 1, "Computer science": 1,
               "Economics": 1, "Engineering": 1, "Mathematics": 0,
               "Philosophy": 0, "Physics": 0, "Psychology": 1, "Sociology": 0},
         tweak={"Philosophy": -3.5, "Computer science": -1.5, "Sociology": -1}),
    dict(key="top5", label="C",
         xlabel="Field: Top-5 Agreement",
         ylabel="Subfield: Top-5 Agreement",
         pct=True, lim=(0, 80), ticks=np.arange(0, 81, 10),
         side={"Biology": 1, "Business": 0, "Chemistry": 0, "Computer science": 2,
               "Economics": 0, "Engineering": 1, "Mathematics": 1,
               "Philosophy": 1, "Physics": 0, "Psychology": 1, "Sociology": 0},
         tweak={"Computer science": 2.5, "Mathematics": -3.5, "Engineering": -4.5,
                "Psychology": -2, "Biology": -1.5, "Philosophy": -1.5, "Sociology": -1}),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

for ax, p in zip(axes, PANELS):
    lo, hi = p["lim"]
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.6", lw=1, zorder=1)
    for field in plotted:
        x, y = d.loc[field, f"{p['key']}_field"], d.loc[field, f"{p['key']}_sub"]
        ax.scatter(x, y, color=field_color_map[field], s=50, zorder=10,
                   edgecolors="black", linewidth=1)
        span = hi - lo
        off_x, off_y = 0.012 * span, 0.008 * span
        if p["side"][field] == 1:
            lx, ha = x + off_x, "left"
        elif p["side"][field] == 0:
            lx, ha = x - off_x, "right"
        else:
            lx, ha = x, "center"
        ax.text(lx, y + off_y + p["tweak"].get(field, 0), abbrev[field],
                color=field_color_map[field], fontsize=10,
                ha=ha, va="baseline", fontweight="bold", zorder=3)

    ax.set_xlabel(p["xlabel"])
    ax.set_ylabel(p["ylabel"])
    if p["pct"]:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(p["ticks"]); ax.set_yticks(p["ticks"])
    ax.set_xlim(p["lim"]); ax.set_ylim(p["lim"])
    ax.set_aspect("equal")
    ax.text(-0.12, 1.02, p["label"], transform=ax.transAxes,
            fontweight="bold", fontsize=14)

plt.tight_layout(w_pad=2.0)

out_path = common.FIGURES / "si_fig1.pdf"
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
if len(sys.argv) > 1:          # optional PNG preview: si_fig1_subfields.py <png path>
    plt.savefig(sys.argv[1], dpi=130, bbox_inches="tight")
    print(f"Saved: {sys.argv[1]}")
