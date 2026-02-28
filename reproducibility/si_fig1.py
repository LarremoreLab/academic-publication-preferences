"""Generate the full-regression SI figure (field-normed aspiration/preference ranks).

Source: 04_aspiration_preference_regressions/03_aspiration_preferences_dankatfork.ipynb
        (function plot_full_regression_results, cells 17–18)

Target: _LATEX SOURCE/si_figs/full_regression_aspiration_rank_field_normed.pdf

Produces:
  - reproducibility/si_fig1.pdf  (2-row × 4-col forest plot)
      Row A: Top preference rank (top_ranked_venue)
      Row B: Top aspiration rank (top_init)
      Columns: Prestige / Gender / Associate / Full

Notes:
  - plot_order uses FULL field names ("Computer science", "Mathematics") to match
    the data.  The notebook's plot_order_umbrella mistakenly used abbreviated names
    ("Comp. Sci.", "Math"), which caused those two fields to sort as NaN and land at
    the bottom of the plot.  The commented-out line in the notebook has the correct
    full names and was used to produce the published PDF.
  - Prestige predictor: academia_prestige_bin_10 is used DIRECTLY (not inverted).
    bin=1 = most prestigious (MIT, Harvard), bin=10 = least prestigious, so the
    coefficient is negative (higher bin = less prestigious = lower venue rank).
    The xlim (-0.051, 0.025) was designed for this negative-coefficient direction.
    The notebook's run_one_regression call uses academia_prestige_bin_10_inv, but
    the published PDF was produced with the non-inverted version.
  - Font settings: import of custom_plotting sets Arial; notebook cell 10 sets
    axes/tick label sizes to 12pt.  These must be replicated explicitly here.
  - Gender coefficient is NOT negated here (unlike the main Fig 4 forest plot panels).
  - Colors: myblue for preference row, myyellow for aspiration row.

Usage:
  .venv/bin/python reproducibility/si_fig1.py

Run compute_rankings.py first if derived/ files are missing.
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import blended_transform_factory
from statsmodels.formula.api import ols

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE   = Path(__file__).parent
PUBLIC = HERE.parent / "public_data"
DERIVED = HERE / "derived"

_required = [
    DERIVED / "individual_rankings.csv",
    DERIVED / "field_consensus_rankings.csv",
]
_missing = [p for p in _required if not p.exists()]
if _missing:
    print("Missing required derived files:")
    for p in _missing:
        print(f"  {p}")
    print("\nPlease run first:\n  .venv/bin/python reproducibility/compute_rankings.py")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Colors (from custom_plotting.py)
# ---------------------------------------------------------------------------
myblue   = "#1F3C88"
myyellow = "#E4B700"

# ---------------------------------------------------------------------------
# Load and assemble data  (same logic as fig4.py)
# ---------------------------------------------------------------------------
respondents = pd.read_csv(PUBLIC / "respondents.csv")
field_cons  = pd.read_csv(DERIVED / "field_consensus_rankings.csv")
indiv       = pd.read_csv(DERIVED / "individual_rankings.csv")

# Field-normed score: 0=lowest, 1=highest within each field's consensus ranking
field_cons["normed"] = field_cons.groupby("field")["score"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)
field_normed_lookup = field_cons.set_index(["field", "venue_db_id"])["normed"]


def lookup_field_normed(field_series, venue_series):
    keys = list(zip(field_series, venue_series))
    return pd.Series(
        [field_normed_lookup.get(k, np.nan) for k in keys],
        index=field_series.index,
    )


# Top-ranked preference venue per user (ordinal_rank == 0 → best)
top_venue = (
    indiv[indiv["ordinal_rank"] == 0]
    .set_index("user_db_id")["venue_db_id"]
    .rename("top_ranked_venue_id")
)

df = respondents.copy()
df["gender"] = df["gender"].map({"gm": 0.0, "gf": 1.0})

PROFESSOR_STAGES = {"Assistant Professor", "Associate Professor", "Full Professor"}
df = df[df["career_stage"].isin(PROFESSOR_STAGES)].copy()

df["top_init_field_normed"]          = lookup_field_normed(df["field"], df["top_init_venue_db_id"])
df = df.join(top_venue, on="user_db_id")
df["top_ranked_venue_field_normed"]  = lookup_field_normed(df["field"], df["top_ranked_venue_id"])

df = df[
    (df["venue_bug"] == False) &
    (df["gender"].notna()) &
    (df["academia_prestige_bin_10"].notna())
].copy()

career_dummies = pd.get_dummies(df["career_stage"])[["Associate Professor", "Full Professor"]]
career_dummies.columns = ["associate_prof_dummy", "full_prof_dummy"]
df = df.join(career_dummies)

# Use academia_prestige_bin_10 directly (NOT inverted).
# bin=1 = most prestigious (MIT, Harvard); bin=10 = least prestigious.
# Coefficient will be negative: higher bin (less prestigious) → lower venue rank.
# The xlim (-0.051, 0.025) was designed for this direction.
PRESTIGE_VAR = "academia_prestige_bin_10"

outcome = "field_normed"

# ---------------------------------------------------------------------------
# Font / style settings
# ---------------------------------------------------------------------------
# custom_plotting import sets font.family=Arial.
# Notebook cell 10 sets axes/tick label sizes = 12.
# Older seaborn (≤ 0.11) auto-applied its "notebook" context on import,
# setting font.size = 12.  Modern seaborn does not, so we set it explicitly.
# This base size drives fontsize="medium" titles, un-sized annotations, etc.
plt.rcParams["font.family"]     = "Arial"
plt.rcParams["font.size"]       = 16
plt.rcParams["axes.labelsize"]  = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# ---------------------------------------------------------------------------
# Regression helper
# ---------------------------------------------------------------------------
def run_one_regression(data, venue_type, outcome_type, level_type, prestige_level):
    model = ols(
        f"{venue_type}_{outcome_type} ~ associate_prof_dummy + full_prof_dummy"
        f" + {prestige_level} + gender",
        data=data,
    ).fit()
    coefficients = model.params[1:]
    p_values     = model.pvalues[1:]
    conf_int     = model.conf_int()[1:]
    conf_int.columns = ["ci_lower", "ci_upper"]
    results = coefficients.to_frame(name="coefficient")
    results["p-value"]                = p_values
    results[["ci_lower", "ci_upper"]] = conf_int
    results["n"]     = model.nobs
    results["model"] = f"{level_type}_{venue_type}_{outcome_type}"
    results.index    = ["Associate", "Full", "Prestige", "Gender"]
    return results


# ---------------------------------------------------------------------------
# Compute regression results
# ---------------------------------------------------------------------------
res = pd.DataFrame()
for venue in ("top_init", "top_ranked_venue"):
    add = run_one_regression(df, venue, outcome, "Academia", PRESTIGE_VAR)
    res = pd.concat([res, add])
    for field in sorted(df["field"].unique()):
        subset = df[df["field"] == field]
        add = run_one_regression(subset, venue, outcome, field, PRESTIGE_VAR)
        res = pd.concat([res, add])

# ---------------------------------------------------------------------------
# Forest-plot helper
# ---------------------------------------------------------------------------
# Full field names must be used here — abbreviated names ("Comp. Sci.", "Math")
# cause "Computer science" and "Mathematics" to receive NaN sort keys and appear
# at the wrong position.  The published figure used the full-name ordering.
_PLOT_ORDER = [
    "Academia", "Biology", "Business", "Chemistry", "Computer science",
    "Economics", "Engineering", "History", "Mathematics", "Medicine",
    "Philosophy", "Physics", "Psychology", "Sociology",
][::-1]   # reversed so y=0 is Sociology (bottom) and Academia is at top


def plot_full_regression_results(axes, result_df, venue_type, outcome_type,
                                  custom_color, panel_label):
    whitespace = []

    for i, rank_index in enumerate(["Prestige", "Gender", "Associate", "Full"]):
        to_plot = result_df[
            result_df["model"].str.contains(venue_type) &
            result_df["model"].str.contains(outcome_type) &
            (result_df.index == rank_index)
        ].copy()
        to_plot["name_to_plot"] = to_plot["model"].str.replace(
            rf"_{venue_type}_{outcome_type}", "", regex=True
        )
        to_plot["sort"] = pd.Categorical(
            to_plot["name_to_plot"], categories=_PLOT_ORDER, ordered=True
        )
        to_plot = to_plot.sort_values("sort")

        colors     = ["white" if p > 0.05 else custom_color for p in to_plot["p-value"]]
        edgecolors = ["grey"  if p > 0.05 else custom_color for p in to_plot["p-value"]]

        ypos = 0
        for j in range(len(to_plot)):
            axes[i].scatter(to_plot["coefficient"].iloc[j], ypos,
                            marker="o", facecolors=colors[j],
                            edgecolors=edgecolors[j], zorder=2)
            axes[i].hlines(ypos,
                           to_plot["ci_lower"].iloc[j],
                           to_plot["ci_upper"].iloc[j],
                           color=edgecolors[j], zorder=0)
            ypos += 1
            if "Biology" in to_plot["name_to_plot"].iloc[j]:
                whitespace.append(ypos)
                ypos += 1

        axes[i].axvline(x=0, color="darkgrey", zorder=0)
        axes[i].set_yticks([])
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        if i == 0:
            axes[i].set_xlim(-0.051, 0.025)

    axes[0].set_title("Prestige",   fontsize="medium")
    axes[1].set_title("Gender",     fontsize="medium")
    axes[2].set_title("Associate",  fontsize="medium")
    axes[3].set_title("Full",       fontsize="medium")

    # y-axis labels (field names + N) on the left of the first panel
    num_people = to_plot["n"].astype(int).tolist()
    ax_labels  = [
        re.sub(rf"_{venue_type}_{outcome_type}", "", name)
        for name in to_plot["model"]
    ]
    trans = blended_transform_factory(axes[0].transAxes, axes[0].transData)

    ypos_label = 0
    for num, label in zip(num_people, ax_labels):
        kw = {"weight": "bold"} if "Academia" in label else {}
        axes[0].annotate(num,   (-0.1, ypos_label), ha="right", va="center",
                         xycoords=trans, **kw)
        axes[0].annotate(label, (-0.7, ypos_label), ha="right", va="center",
                         xycoords=trans, **kw)
        ypos_label += 1
        if "Biology" in label:
            ypos_label += 1

    axes[0].annotate("$N$",       (-0.2, 17),   ha="center", va="center",
                     xycoords=trans, weight="bold")
    axes[0].annotate(panel_label, (-0.7, 18.5), ha="right",  va="top",
                     xycoords=trans, fontsize="large", weight="bold")

    # White gap lines between fields and Academia
    for ws in whitespace[:2]:
        con = ConnectionPatch(
            xyA=(0, ws), xyB=(0, ws),
            coordsA=trans, coordsB="data",
            axesA=axes[0], axesB=axes[-1],
            color="white", zorder=50, lw=10,
        )
        axes[-1].add_artist(con)

    legend_elements = [
        Line2D([0], [0], marker="o", color=custom_color,
               label=r"Significant ($\alpha = 0.05$)", markersize=7, linewidth=2.5),
        Line2D([0], [0], marker="o", color="grey", markerfacecolor="white",
               label="Not significant", markersize=7, linewidth=2.5),
    ]
    axes[0].legend(handles=legend_elements, bbox_to_anchor=(4.9, 1.25),
                   frameon=False, fontsize=11)
    return axes


# ---------------------------------------------------------------------------
# Build and save figure
# ---------------------------------------------------------------------------
fig_si, axs_si = plt.subplots(
    nrows=2, ncols=4, figsize=(12, 10.5),
    sharey=True, sharex="col",
    gridspec_kw={"hspace": 0.35},
)

plot_full_regression_results(axs_si[0], res, "top_ranked_venue", outcome,
                              myblue,   "A: Top Preference")
plot_full_regression_results(axs_si[1], res, "top_init",         outcome,
                              myyellow, "B: Top Aspiration")

fig_si.subplots_adjust(left=0.31, bottom=0.05)

out = HERE / "si_fig1.pdf"
plt.savefig(out)
print(f"Saved: {out}")
