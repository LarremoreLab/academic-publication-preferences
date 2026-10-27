"""SI Fig. S2: Fig. 2 with 95% bootstrap intervals (si_fig2.pdf).

Field-level prediction accuracy (horizontal axis) and top-5 agreement (vertical axis) as in Fig. 2,
each with its 95% percentile interval from the respondent bootstrap of bootstrap_consensus.py:
1,000 replicates in which respondents were resampled with replacement within each field, keeping
all of a respondent's comparisons and rankings together; field rankings were refit in every
replicate, and prediction accuracy excludes all copies of the scored respondent from the field
ranking it is scored against. Points show the full-sample values (identical to Fig. 2).

Input   reproducibility/cache/bootstrap_consensus.csv
        The cache is committed with the repository. To regenerate it run
        `.venv/bin/python reproducibility/bootstrap_consensus.py` (about 13 minutes on 16 cores).
Output  reproducibility/output/figures/si_fig2.pdf

Usage   cd <repo_root>
        .venv/bin/python reproducibility/si_fig2_bootstrap.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

CACHE_FILE = common.CACHE / "bootstrap_consensus.csv"
if not CACHE_FILE.exists():
    sys.exit(f"Missing {CACHE_FILE}. Run `.venv/bin/python reproducibility/bootstrap_consensus.py` "
             "to generate the bootstrap cache (about 13 minutes on 16 cores).")

ci = pd.read_csv(CACHE_FILE).set_index(["field", "measure"])
field_color_map = common.field_colors()
fields = common.FIELDS
k = 5

# ---------------------------------------------------------------------------
# Label placement, as in fig2.py
# ---------------------------------------------------------------------------
abbrev = {
    "Biology": "Biology",
    "Business": "Business",
    "Chemistry": "Chemistry",
    "Computer science": "Computer Science",
    "Economics": "Economics",
    "Engineering": "Engineering",
    "History": "History",
    "Mathematics": "Mathematics",
    "Medicine": "Medicine",
    "Philosophy": "Philosophy",
    "Physics": "Physics",
    "Psychology": "Psychology",
    "Sociology": "Sociology",
}

position_side = {
    "Biology": 1, "Business": 1, "Chemistry": 0, "Computer science": 1,
    "Economics": 0, "Engineering": 1, "History": 0, "Mathematics": 1,
    "Medicine": 0, "Philosophy": 1, "Physics": 0, "Psychology": 0, "Sociology": 0,
}

tweaks = {field: 0 for field in fields}
tweaks["Psychology"] = 1.5
tweaks["Sociology"] = -3
tweaks["Business"] = -3
tweaks["Engineering"] = -4
tweaks["Computer science"] = -4

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
point_positions = {f: (ci.loc[(f, "prediction_accuracy"), "point"], ci.loc[(f, "top5_agreement"), "point"])
                   for f in fields}
offset_x, offset_y = 0.2, 0.3
fig, ax = plt.subplots()
for f, (x, y) in point_positions.items():
    xl, xh = ci.loc[(f, "prediction_accuracy"), ["lo", "hi"]]
    yl, yh = ci.loc[(f, "top5_agreement"), ["lo", "hi"]]
    # (the max(0, .) only matters for test runs with very few replicates, whose percentile interval
    # need not contain the full-sample point)
    ax.errorbar(x, y, xerr=[[max(0, x - xl)], [max(0, xh - x)]], yerr=[[max(0, y - yl)], [max(0, yh - y)]], fmt="none",
                ecolor=field_color_map[f], elinewidth=1, alpha=0.7, zorder=5)
    ax.scatter(x, y, color=field_color_map[f], s=50, zorder=10, edgecolors="black", linewidth=1)
for f, (x, y) in point_positions.items():
    lx, ha = (x + offset_x, "left") if position_side[f] == 1 else (x - offset_x, "right")
    ax.text(lx, y + offset_y + tweaks[f], abbrev[f], color=field_color_map[f], fontsize=10,
            ha=ha, va="baseline", fontweight="bold", zorder=3)
ax.set_xlabel("Average Prediction Accuracy on Random Venue Pairs")
ax.set_ylabel("Avg. Percentage of Individual Top-%d Venues in Field Top-%d" % (k, k))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_yticks(np.arange(0, 81, 10))
ax.set_xticks(np.arange(60, 81, 5))
ax.set_xlim([60, 82])
ax.set_ylim([0, 81])
plt.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = common.FIGURES / "si_fig2.pdf"
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
