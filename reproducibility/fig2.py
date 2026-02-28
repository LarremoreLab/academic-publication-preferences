"""
Reproducibility script for Figure 2 (fig2.pdf).
Source notebook: 10_preference_focused_analysis/top_vs_full.ipynb

Inputs:
  - public_data/comparisons.csv
  - reproducibility/derived/individual_rankings.csv    (from compute_rankings.py)
  - reproducibility/derived/field_rankings_per_user.csv (from compute_rankings.py)

Output:
  - reproducibility/fig2.pdf

Usage:
  cd <repo_root>
  .venv/bin/python reproducibility/compute_rankings.py  # only needed once
  .venv/bin/python reproducibility/fig2.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
# (SpringRank is not imported here — rankings are pre-computed by compute_rankings.py)
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
PUBLIC = HERE.parent / "public_data"
DERIVED = HERE / "derived"

for f in ["individual_rankings.csv", "field_rankings_per_user.csv"]:
    if not (DERIVED / f).exists():
        sys.exit(f"Missing {DERIVED / f}. Run compute_rankings.py first.")

# ---------------------------------------------------------------------------
# Inlined utils/field_colors.py
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

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
comp_df = pd.read_csv(PUBLIC / "comparisons.csv")
individual_rankings = pd.read_csv(DERIVED / "individual_rankings.csv")
field_rankings_df = pd.read_csv(DERIVED / "field_rankings_per_user.csv")

# Reconstruct field_rankings dict: user_db_id -> {venue_db_id: score}
individual_field_rankings = {
    user: dict(zip(grp["venue_db_id"], grp["score"]))
    for user, grp in field_rankings_df.groupby("user_db_id")
}

# ---------------------------------------------------------------------------
# Compute field_top_k_overlaps
# ---------------------------------------------------------------------------
k = 5
field_top_k_counts = {f: defaultdict(int) for f in fields_to_use}
field_top_counts = {f: 0 for f in fields_to_use}

for user, user_comps in comp_df.groupby("user_db_id"):
    user_field = user_comps["field"].values[0]
    user_ranks = individual_rankings[individual_rankings["user_db_id"] == user]
    user_top_k = user_ranks.loc[user_ranks["ordinal_rank"] < k, "venue_db_id"].values
    for v in user_top_k:
        field_top_k_counts[user_field][v] += 1
        field_top_counts[user_field] += 1

field_top_k_overlaps = {
    field: sum(sorted(counts.values(), reverse=True)[:k]) / field_top_counts[field]
    for field, counts in field_top_k_counts.items()
    if field_top_counts[field] > 0
}

# ---------------------------------------------------------------------------
# Prediction accuracy from leave-one-out field rankings
# ---------------------------------------------------------------------------


def reveal_upsets(user_rankings, individual_comps, only_random_comps=True):
    all_upsets = []
    for user, u_comps in individual_comps.groupby("user_db_id"):
        u_ranks = user_rankings.get(user, {})
        user_field = u_comps["field"].values[0]
        u_times_compared = defaultdict(int)
        correct = preds = obs = 0
        for pref, other, tie in u_comps[
            ["pref_venue_db_id", "other_venue_db_id", "is_tie"]
        ].values:
            u_times_compared[pref] += 1
            u_times_compared[other] += 1
            if only_random_comps and (
                u_times_compared[pref] > 1 or u_times_compared[other] > 1
            ):
                break
            if tie is False:
                obs += 1
                if pref in u_ranks and other in u_ranks:
                    preds += 1
                    if u_ranks[pref] > u_ranks[other]:
                        correct += 1
        all_upsets.append((user, user_field, correct, preds, obs))
    df = pd.DataFrame(all_upsets, columns=["user", "field", "correct", "preds", "obs"])
    df["frac_correct"] = df["correct"] / df["preds"]
    df["frac_preds"] = df["preds"] / df["obs"]
    return df


individual_field_order = reveal_upsets(individual_field_rankings, comp_df)

# ---------------------------------------------------------------------------
# Plot
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

tweaks = {field: 0 for field in fields_to_use}
tweaks["Psychology"] = 1.5
tweaks["Sociology"] = -3
tweaks["Business"] = -3
tweaks["Engineering"] = -4
tweaks["Computer science"] = -4

point_positions = {
    field: (
        100 * stats["frac_correct"].mean(),
        100 * field_top_k_overlaps[field],
    )
    for field, stats in individual_field_order.groupby("field")
}

offset_x, offset_y = 0.2, 0.3
adjusted_positions = {}
for field, (x, y) in point_positions.items():
    if position_side[field] == 1:
        adjusted_positions[field] = (x + offset_x, y + offset_y, "left")
    else:
        adjusted_positions[field] = (x - offset_x, y + offset_y, "right")

fig, ax = plt.subplots()
for field, (x, y) in point_positions.items():
    ax.scatter(x, y, color=field_color_map[field], s=50, zorder=10,
               edgecolors="black", linewidth=1)

ax.set_xlabel("Average Prediction Accuracy on Random Venue Pairs")
ax.set_ylabel("Avg. Percentage of Individual Top-%d Venues in Field Top-%d" % (k, k))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for field, (label_x, label_y, ha) in adjusted_positions.items():
    ax.text(label_x, label_y + tweaks[field], abbrev[field],
            color=field_color_map[field], fontsize=10,
            ha=ha, va="baseline", fontweight="bold", zorder=3)

ax.set_yticks(np.arange(0, 81, 10))
ax.set_xticks(np.arange(65, 81, 5))
ax.set_xlim([65, 81])
ax.set_ylim([0, 81])
plt.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = HERE / "fig2.pdf"
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
