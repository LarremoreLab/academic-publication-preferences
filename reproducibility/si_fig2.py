"""Generate the JIF vs. consensus preference rankings SI figure (per-field colors).

Source: 06_jif/03_most_selected_jif_porc_differences.ipynb
        (cell 3fc0646f — the final per-field colored version)

Target: _LATEX SOURCE/si_figs/field_pref_jif_rank_diffs_field.pdf

Produces:
  - reproducibility/si_fig2.pdf

Data:
  - public_data/venues.csv              → venue name / id lookup
  - public_data/field_jif_rankings.csv  → ordinal porc/jif ranks + frac_selected

Run from the project root:
  .venv/bin/python reproducibility/si_fig2.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
ROOT = HERE.parent
PUBLIC = ROOT / "public_data"

sys.path.insert(0, str(ROOT))
from utils.field_colors import field_color_map          # noqa: E402
from utils.short_venue_names import short_venue_names   # noqa: E402

# ── Load data ─────────────────────────────────────────────────────────────────
venues_df = pd.read_csv(PUBLIC / "venues.csv")
venue_names = dict(zip(venues_df["venue_db_id"], venues_df["name"]))
venue_ids   = {v: k for k, v in venue_names.items()}

field_rankings = pd.read_csv(PUBLIC / "field_jif_rankings.csv")

# ── Constants (from notebook setup cells) ────────────────────────────────────
top_venues = [
    "Nature",
    "Science",
    "PNAS",
    "The American Economic Review",
    "The Accounting Review",
    "American Sociological Review",
    "Journal of Personality and Social Psychology",
    "Journal of the American Chemical Society",
    "Physical Review Letters",
    "Annals of Mathematics",
    # "USENIX Security Symposium (UsenixSec)",  # commented out in source
    "The American Historical Review",
    "The Philosophical Review",
]

venue_field_map = {
    "The American Economic Review":               "Economics",
    "The Accounting Review":                      "Business",
    "American Sociological Review":               "Sociology",
    "Journal of Personality and Social Psychology": "Psychology",
    "Journal of the American Chemical Society":   "Chemistry",
    "Physical Review Letters":                    "Physics",
    "Annals of Mathematics":                      "Mathematics",
    "USENIX Security Symposium (UsenixSec)":      "Computer science",
    "The American Historical Review":             "History",
    "The Philosophical Review":                   "Philosophy",
    "Nature":                                     "Biology, Engineering,\nMedicine",
}

non_highlighted_fields  = {"Economics", "Business", "Mathematics",
                            "Computer Science", "History", "Philosophy"}
interdisciplinary_venues = {"Nature", "Science", "PNAS"}

# ── Derived quantities ────────────────────────────────────────────────────────
NATURE_ID = 1647092  # Nature's venue_db_id
field_venue_fracs = (
    field_rankings.set_index(["field", "venue_db_id"])["frac_selected"].to_dict()
)
field_order = [
    fld for fld, _ in sorted(
        [(fld, field_venue_fracs.get((fld, NATURE_ID), 0.0))
         for fld in field_color_map],
        key=lambda x: -x[1],
    )
]

# ── Plot ─────────────────────────────────────────────────────────────────────
ROW_BG_COLOR = "#f7f7f7"

fig, ax = plt.subplots(figsize=(3.4, 5))

# Step 1: Y-axis mapping
ypos_map = {venue_name: i for i, venue_name in enumerate(top_venues[::-1])}

# Step 2: Gridlines behind everything
plt.axvline(0, color="k", alpha=0.5, zorder=-20, linestyle="-")
for x in [-20, -10, 10, 20, 30, 40]:
    plt.axvline(x, color="k", alpha=0.3, zorder=-20, linestyle="-", lw=0.8)

# Step 3: Alternating row background shading
x_data_min, x_data_max = -22, 32
for i, venue in enumerate(top_venues[::-1]):
    y = ypos_map[venue]
    if i % 2 == 0:
        rect = patches.Rectangle(
            (x_data_min, y - 0.5),
            x_data_max - x_data_min,
            1.0,
            color=ROW_BG_COLOR,
            zorder=-100,
            clip_on=False,
        )
        ax.add_patch(rect)

# Step 4: Plot points
rng = np.random.default_rng(seed=42)
jitter_cache    = {}
highlighted_venues = set()
fields_seen     = {}

for field in field_order:
    field_color = field_color_map[field]
    field_df    = field_rankings.loc[field_rankings["field"] == field].dropna()
    field_venues = set(field_df["venue_db_id"].values)

    for venue_name in top_venues[::-1]:
        if venue_name not in venue_ids:
            continue
        venue_id = venue_ids[venue_name]
        if venue_id not in field_venues:
            continue

        x      = ypos_map[venue_name]
        f_pref = field_df.loc[field_df["venue_db_id"] == venue_id, "subset_ordinal_porc"].values[0]
        f_jif  = field_df.loc[field_df["venue_db_id"] == venue_id, "subset_ordinal_jif"].values[0]
        pref_diff = f_jif - f_pref

        key = (pref_diff, x)
        if key not in jitter_cache:
            jitter_cache[key] = (rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2))
        jitter, v_jitter = jitter_cache[key]

        is_interdisciplinary = venue_name in interdisciplinary_venues
        is_accent = (
            venue_field_map.get(venue_name) == field
            and venue_name not in highlighted_venues
        )

        if is_interdisciplinary and field in non_highlighted_fields:
            ax.scatter(
                pref_diff + jitter, x + v_jitter,
                facecolor=field_color, edgecolor=field_color,
                s=20, alpha=0.8, zorder=-1,
            )
            fields_seen.setdefault(field, set()).add("o")
        elif is_accent or (is_interdisciplinary and field not in non_highlighted_fields):
            ax.scatter(
                pref_diff + jitter, x + v_jitter,
                color=field_color, marker="x",
                linewidths=0.75, s=20, zorder=5,
            )
            highlighted_venues.add(venue_name)
            fields_seen.setdefault(field, set()).add("x")
        else:
            ax.scatter(
                pref_diff + jitter, x + v_jitter,
                facecolor=field_color, edgecolor=field_color,
                s=20, alpha=0.8, zorder=-1,
            )
            fields_seen.setdefault(field, set()).add("o")

# Step 5: Y-axis ticks
sns.despine(left=True, right=False)
yticks = [ypos_map[v] for v in top_venues[::-1]]
ax.set_yticks(yticks)
ax.tick_params(axis="y", length=0)

ytick_labels = []
for v in top_venues[::-1]:
    short_name = short_venue_names.get(v, v)
    fld = venue_field_map.get(v, "")
    label = f"{short_name}\n({fld})" if fld else short_name
    ytick_labels.append(label)

ax.set_yticklabels(ytick_labels, fontsize="x-small")

# Step 6: X-axis ticks
plt.xticks(
    ticks=range(-20, 31, 10),
    labels=[
        "↑" + str(tick) if tick > 0
        else ("↓" + str(tick)[1:] if tick != 0 else str(tick))
        for tick in range(-20, 31, 10)
    ],
    fontsize="small",
)

# Step 7: Axes config
plt.xlabel("Difference between\nConsensus Ranking and JIF Ranking", fontsize="x-small")
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.spines["left"].set_visible(True)
ax.spines["right"].set_visible(False)
plt.xlim(left=x_data_min, right=x_data_max)
plt.tight_layout()
sns.despine(left=True, right=True)

# Step 8: Legend (on the right, outside plot area)
legend_handles = []
for field in field_order:
    if field not in field_color_map:
        continue
    color = field_color_map[field]
    marker_style = "x" if "x" in fields_seen.get(field, set()) else "o"
    handle = Line2D(
        [0], [0],
        marker=marker_style,
        color=color,
        linestyle="None",
        markersize=5,
        markeredgewidth=0.75,
        markerfacecolor=color if marker_style == "o" else "none",
        alpha=0.8 if marker_style == "o" else 1.0,
        label=field,
    )
    legend_handles.append(handle)

plt.subplots_adjust(right=0.72)
ax.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(1.5, 0.8),
    fontsize="x-small",
    frameon=False,
    handletextpad=0.4,
    columnspacing=0.8,
    handlelength=1.2,
    borderaxespad=0.2,
    labelspacing=0.3,
)

out = HERE / "si_fig2.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
