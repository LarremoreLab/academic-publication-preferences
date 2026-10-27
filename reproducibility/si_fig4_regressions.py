"""SI Fig. S4: full Eq. S1 estimates (prestige, gender, associate, full) for both outcomes.

Each row of the figure is one regression of Eq. S1,

    y_i = b0 + b1 Associate_i + b2 Full_i + b3 Prestige_i + b4 Gender_i + e_i,

where y is the field-normalized rank of the respondent's top preference (panel row A) or top
aspiration (panel row B). "Academia" is the pooled model over all respondents with field fixed
effects; the other rows are estimated within each field. Error bars are 95% confidence intervals
with standard errors clustered by institution (t reference distribution). Colored markers are
significant at alpha = 0.05 and open gray markers are not; as in Fig. 4, preference is dark blue
and aspiration light blue. The per-field Associate and Full
coefficients are drawn in gray without significance coloring because the reference category,
assistant professors, has fewer than ten respondents in most fields. Gender coefficients are
plotted as men minus women, as in Fig. 4E-F. Fig. 4C-F shows the prestige and gender columns.

Sample: the Eq. S1 sample (common.eqs1_sample; N = 2,196 professors at 258 institutions).

Inputs:  public_data/respondents.csv and reproducibility/derived/ (through common.eqs1_sample)
Outputs: output/figures/si_fig4.pdf

Usage (from the project root, after compute_rankings.py):
    .venv/bin/python reproducibility/si_fig4_regressions.py
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def eqs1_coefficients(df):
    """Eq. S1 for both outcomes at every level: pooled over all respondents with field fixed
    effects ("Academia") and separately within each field. Standard errors are clustered by
    institution. One row per (outcome, level, coefficient). Same as fig4.eqs1_coefficients."""
    rows = []
    for outcome in common.OUTCOME_COLS:
        levels = [("Academia", df, True)] + [(f, df[df["field"] == f], False) for f in common.FIELDS]
        for level, data, field_fe in levels:
            m = common.fit_clustered(common.eqs1_formula(outcome, field_fe=field_fe), data)
            ci = m.conf_int()
            for term, label in common.COEF_LABELS.items():
                rows.append({"outcome": outcome, "level": level, "coefficient": label,
                             "estimate": m.params[term], "se": m.bse[term],
                             "ci_lower": ci.loc[term, 0], "ci_upper": ci.loc[term, 1],
                             "p": m.pvalues[term], "n_obs": int(m.nobs),
                             "n_clusters": m.model.data.frame[common.CLUSTER].nunique()})
    return pd.DataFrame(rows)


df = common.eqs1_sample()
coefs = eqs1_coefficients(df)

# Forest-plot layout: one row per coefficient (index = coefficient name),
# model = "<level>_<outcome column>", e.g. "Biology_top_init_field_normed".
outcome = "field_normed"
res = pd.DataFrame({"coefficient": coefs["estimate"].values, "p-value": coefs["p"].values,
                    "ci_lower": coefs["ci_lower"].values, "ci_upper": coefs["ci_upper"].values,
                    "n": coefs["n_obs"].values,
                    "model": (coefs["level"] + "_" + coefs["outcome"].map(common.OUTCOME_COLS)).values},
                   index=coefs["coefficient"].values)

# ---------------------------------------------------------------------------
# Colors and font settings
# ---------------------------------------------------------------------------
C_LIGHT = np.array([107, 171, 250]) / 255   # light blue (aspirations), as in fig4.py
C_DARK  = C_LIGHT * 0.55                     # dark blue  (preferences)

plt.rcParams["font.family"]     = "Arial"
plt.rcParams["font.size"]       = 16
plt.rcParams["axes.labelsize"]  = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# ---------------------------------------------------------------------------
# Forest-plot helper
# ---------------------------------------------------------------------------
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

        # Negate gender coefficients to show men − women (as in fig4.py)
        if rank_index == "Gender":
            to_plot["coefficient"] = -to_plot["coefficient"]
            to_plot["ci_lower"], to_plot["ci_upper"] = (
                -to_plot["ci_upper"].copy(),
                -to_plot["ci_lower"].copy(),
            )

        colors     = ["white" if p > 0.05 else custom_color for p in to_plot["p-value"]]
        edgecolors = ["grey"  if p > 0.05 else custom_color for p in to_plot["p-value"]]
        # Per-field career-stage coefficients in neutral gray, no significance coloring
        # (most fields have <10 Assistant Professors, the reference group).
        # The Academia row keeps its coloring.
        if rank_index in ("Associate", "Full"):
            for j, name in enumerate(to_plot["name_to_plot"]):
                if name != "Academia":
                    colors[j], edgecolors[j] = "grey", "grey"

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
            axes[i].set_xlim(-0.025, 0.051)  # 10 = most prestigious

    axes[0].set_title("Prestige",   fontsize="medium")
    axes[1].set_title("Gender (men − women)", fontsize="medium")
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
                              C_DARK,   "A: Top Preference")
plot_full_regression_results(axs_si[1], res, "top_init",         outcome,
                              C_LIGHT,  "B: Top Aspiration")

fig_si.subplots_adjust(left=0.31, bottom=0.05)

out = common.FIGURES / "si_fig4.pdf"
plt.savefig(out)
print(f"Saved: {out}")
