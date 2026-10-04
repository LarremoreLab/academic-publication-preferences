"""Fig. 4: top preferences and top aspirations by institutional prestige and gender.

Panels A and B show group means of the field-normalized rank (0 = lowest, 1 = highest venue in
the field's consensus ranking) of each respondent's top preference and top aspiration, by
prestige decile (A; 10 = most prestigious) and by gender (B). Error bars are +/-1 standard error
clustered by institution (an intercept-only OLS within each group). The asterisks in B give the
significance of a regression of the outcome on gender with institution-clustered standard errors
(*** p < 0.001, ** p < 0.01, * p < 0.05).

Panels C-F show the prestige (C, D) and gender (E, F) coefficients of Eq. S1,

    y_i = b0 + b1 Associate_i + b2 Full_i + b3 Prestige_i + b4 Gender_i + e_i,

estimated by OLS within each field and, for the pooled "All Resp." row, over all respondents
with field fixed effects. Error bars are 95% confidence intervals with standard errors clustered
by institution (t reference distribution); filled markers are significant at alpha = 0.05.
Gender coefficients are plotted as men minus women.

Sample: the Eq. S1 sample (common.eqs1_sample; N = 2,196 professors at 258 institutions).

Inputs:  public_data/respondents.csv and reproducibility/derived/ (through common.eqs1_sample)
Outputs: output/figures/fig4.pdf
         output/results/eqs1_coefficients.csv  every Eq. S1 coefficient: outcome, level
             ("Academia" = pooled with field fixed effects, or one of the 13 fields), coefficient,
             estimate, clustered se, 95% CI, p, n_obs, n_clusters
         output/results/fig4_group_means.csv   the panel A/B group means with clustered SE, and the
             women - men and decile 10 - decile 1 comparisons (estimate, se, CI, p), per outcome

Usage (from the project root, after compute_rankings.py):
    .venv/bin/python reproducibility/fig4.py
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


# ---------------------------------------------------------------------------
# Eq. S1 and the panel A/B group statistics
# ---------------------------------------------------------------------------
def eqs1_coefficients(df):
    """Eq. S1 for both outcomes at every level: pooled over all respondents with field fixed
    effects ("Academia") and separately within each field. Standard errors are clustered by
    institution. One row per (outcome, level, coefficient)."""
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


def group_statistics(df):
    """Panel A/B statistics for each outcome, on the rows where the outcome is observed:
    group means with institution-clustered SE (y ~ 1 within each prestige decile and gender),
    women - men (y ~ women) and decile 10 - decile 1 (y ~ top, deciles 1 and 10 only)."""
    rows = []
    for outcome, ycol in common.OUTCOME_COLS.items():
        d = df[df[ycol].notna()].assign(y=df[ycol])
        groups = [(f"decile_{k}", d[d[common.PRESTIGE] == k]) for k in range(1, 11)]
        groups += [("men", d[d["gender"] == 0]), ("women", d[d["gender"] == 1])]
        for name, g in groups:
            rows.append(_term_row(outcome, "group_mean", name, common.fit_clustered("y ~ 1", g), "Intercept"))
        m = common.fit_clustered("y ~ women", d.assign(women=(d["gender"] == 1).astype(float)))
        rows.append(_term_row(outcome, "comparison", "women_minus_men", m, "women"))
        dd = d[d[common.PRESTIGE].isin([1, 10])]
        m = common.fit_clustered("y ~ top", dd.assign(top=(dd[common.PRESTIGE] == 10).astype(float)))
        rows.append(_term_row(outcome, "comparison", "decile10_minus_decile1", m, "top"))
    return pd.DataFrame(rows)


def _term_row(outcome, kind, quantity, m, term):
    lo, hi = m.conf_int().loc[term]
    return {"outcome": outcome, "kind": kind, "quantity": quantity, "estimate": m.params[term],
            "se": m.bse[term], "ci_lower": lo, "ci_upper": hi, "p": m.pvalues[term],
            "n_obs": int(m.nobs), "n_clusters": m.model.data.frame[common.CLUSTER].nunique()}


df = common.eqs1_sample()
coefs = eqs1_coefficients(df)
stats = group_statistics(df)
coefs.to_csv(common.RESULTS / "eqs1_coefficients.csv", index=False)
stats.to_csv(common.RESULTS / "fig4_group_means.csv", index=False)
print(f"Eq. S1 sample: {len(df)} respondents, {df[common.CLUSTER].nunique()} institutions")
print(f"Saved: {common.RESULTS / 'eqs1_coefficients.csv'}")
print(f"Saved: {common.RESULTS / 'fig4_group_means.csv'}")

# ---------------------------------------------------------------------------
# Plot data. Forest plots: one row per coefficient, model = "<level>_<outcome column>".
# Panels A/B: means and clustered SEs by decile (index 1..10) and by gender (0 = men, 1 = women).
# ---------------------------------------------------------------------------
outcome = "field_normed"
res = pd.DataFrame({"coefficient": coefs["estimate"].values, "p-value": coefs["p"].values,
                    "ci_lower": coefs["ci_lower"].values, "ci_upper": coefs["ci_upper"].values,
                    "n": coefs["n_obs"].values,
                    "model": (coefs["level"] + "_" + coefs["outcome"].map(common.OUTCOME_COLS)).values},
                   index=coefs["coefficient"].values)

_means = stats[stats["kind"] == "group_mean"]
_OUT = {f"top_init_{outcome}": "top_aspiration", f"top_ranked_venue_{outcome}": "top_preference"}


def _table(groups):
    cols = {}
    for col, oname in _OUT.items():
        g = _means[_means["outcome"] == oname].set_index("quantity")
        cols[(col, "mean")] = [g.loc[q, "estimate"] for q in groups]
        cols[(col, "sem")] = [g.loc[q, "se"] for q in groups]
        cols[(col, "count")] = [g.loc[q, "n_obs"] for q in groups]
    return pd.DataFrame(cols)


prestige = _table([f"decile_{k}" for k in range(1, 11)])
prestige.index += 1
gender = _table(["men", "women"])

_cmp = stats[stats["quantity"] == "women_minus_men"].set_index("outcome")["p"]


def _stars(p):
    if p < 0.001:   return "***"
    elif p < 0.01:  return "**"
    elif p < 0.05:  return "*"
    else:           return "n.s."


stars_pref = _stars(_cmp["top_preference"])
stars_asp = _stars(_cmp["top_aspiration"])

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C_LIGHT = np.array([107, 171, 250]) / 255   # light blue (aspirations)
C_DARK  = C_LIGHT * 0.55                     # dark blue  (preferences)

# ---------------------------------------------------------------------------
# Forest plot configuration
# ---------------------------------------------------------------------------
# "All Resp." in plot_order causes "Academia" (from model names) to sort as NaN
# → placed at the end → highest y position (top of plot). Intentional.
plot_order = [
    "All Resp.", "Biology", "Business", "Chemistry", "Computer science",
    "Economics", "Engineering", "History", "Mathematics", "Medicine",
    "Philosophy", "Physics", "Psychology", "Sociology",
][::-1]

names_abbrev = {
    "Academia":        "All Resp.",
    "Biology":         "Biology",
    "Business":        "Business",
    "Chemistry":       "Chemistry",
    "Computer science":"Comp. Sci.",
    "Economics":       "Economics",
    "Engineering":     "Engineering",
    "History":         "History",
    "Mathematics":     "Math",
    "Medicine":        "Medicine",
    "Philosophy":      "Philosophy",
    "Physics":         "Physics",
    "Psychology":      "Psychology",
    "Sociology":       "Sociology",
}

xlim_prestige = (-0.05, 0.05)
xlim_gender   = (-0.27, 0.27)

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
plt.rcParams["font.size"]       = 12
plt.rcParams["font.family"]     = "Arial"
plt.rcParams["axes.labelsize"]  = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
label_fontsize = 12
ms = 8
ls = "--"

fig = plt.figure(figsize=(13, 7))
gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 1.3], hspace=0.35)
gs_top    = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],
                                             width_ratios=[2, 1.3], wspace=0.3)
gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], wspace=0.3)

ax_A  = fig.add_subplot(gs_top[0, 0])
ax_B  = fig.add_subplot(gs_top[0, 1])
axes_bottom = [fig.add_subplot(gs_bottom[0, i]) for i in range(4)]

# ---------------------------------------------------------------------------
# Panel A: prestige scatter
# ---------------------------------------------------------------------------
x1, y1, yerr1 = prestige.index, prestige[f"top_init_{outcome}"]["mean"],         prestige[f"top_init_{outcome}"]["sem"]
x2, y2, yerr2 = prestige.index, prestige[f"top_ranked_venue_{outcome}"]["mean"],  prestige[f"top_ranked_venue_{outcome}"]["sem"]

ax_A.errorbar(x1, y1, yerr1, ls=ls, marker="o", markersize=ms, markeredgecolor="k", color=C_LIGHT)
ax_A.errorbar(x2, y2, yerr2, ls=ls, marker="o", markersize=ms, markeredgecolor="k", color=C_DARK)
ax_A.text(0.97, 0.02, "increasing prestige →",
          transform=ax_A.transAxes, ha="right", va="bottom", fontsize=label_fontsize)
ax_A.text(1.1, y1.iloc[2] + 0.02, "Top aspirations",
          ha="left", va="bottom", color=C_LIGHT, fontsize=label_fontsize)
ax_A.text(1.1, y2.iloc[2] + 0.02, "Top preferences",
          ha="left", va="bottom", color=C_DARK, fontsize=label_fontsize)
ax_A.set_xticks(x1)
ax_A.set_ylim([0.55, 0.9])
ax_A.set_xlabel("Prestige decile")
ax_A.set_ylabel("Field rank of\nselected venue")

# ---------------------------------------------------------------------------
# Panel B: gender scatter with significance brackets
# ---------------------------------------------------------------------------
men_pref  = gender.loc[0, (f"top_ranked_venue_{outcome}", "mean")]
women_pref = gender.loc[1, (f"top_ranked_venue_{outcome}", "mean")]
men_asp   = gender.loc[0, (f"top_init_{outcome}", "mean")]
women_asp  = gender.loc[1, (f"top_init_{outcome}", "mean")]
men_pref_err   = gender.loc[0, (f"top_ranked_venue_{outcome}", "sem")]
women_pref_err = gender.loc[1, (f"top_ranked_venue_{outcome}", "sem")]
men_asp_err    = gender.loc[0, (f"top_init_{outcome}", "sem")]
women_asp_err  = gender.loc[1, (f"top_init_{outcome}", "sem")]

y_b    = np.array([men_pref, women_pref, men_asp, women_asp])
yerr_b = np.array([men_pref_err, women_pref_err, men_asp_err, women_asp_err])
colors_b  = [C_DARK, C_DARK, C_LIGHT, C_LIGHT]
markers_b = ["s", "v", "s", "v"]
labs_b    = ["Men", "Women", "Men", "Women"]
x_b = np.arange(len(y_b))

for i in range(len(x_b)):
    ax_B.errorbar(x_b[i], y_b[i], yerr=yerr_b[i],
                  marker=markers_b[i], markersize=ms,
                  color=colors_b[i], markeredgecolor="k")
    ax_B.text(x_b[i], y_b[i] - 0.02, labs_b[i],
              ha="center", va="top", fontsize=label_fontsize)

yl = ax_A.get_ylim()
y_bar = yl[1] - 0.03
for x_lo, x_hi, stars in [(0, 1, stars_pref), (2, 3, stars_asp)]:
    ax_B.plot([x_lo, x_hi], [y_bar, y_bar], "k-", linewidth=1)
    ax_B.plot([x_lo, x_lo], [y_bar - 0.01, y_bar], "k-", linewidth=1)
    ax_B.plot([x_hi, x_hi], [y_bar - 0.01, y_bar], "k-", linewidth=1)
    ax_B.text((x_lo + x_hi) / 2, y_bar + 0.005, stars,
              ha="center", va="bottom", fontsize=12)

ax_B.set_xlim([-0.5, 3.5])
ax_B.set_xticks([0.5, 2.5])
ax_B.set_xticklabels(["Top preferences", "Top aspirations"])
ax_B.set_ylim(yl)
ax_B.set_ylabel("Field rank of\nselected venue")

# ---------------------------------------------------------------------------
# Panels C–F: forest plots
# ---------------------------------------------------------------------------
whitespace = []

for a, rank_index in enumerate(["Prestige", "Gender"]):
    for b, venue_type in enumerate([f"top_ranked_venue_{outcome}", f"top_init_{outcome}"]):
        i = 2 * a + b
        ax = axes_bottom[i]

        to_plot = res[
            res["model"].str.contains(venue_type.replace(f"_{outcome}", "")) &
            (res.index == rank_index)
        ].copy()
        to_plot["name_to_plot"] = to_plot["model"].str.replace(
            rf"_{venue_type}", "", regex=True
        )
        to_plot["sort"] = pd.Categorical(
            to_plot["name_to_plot"], categories=plot_order, ordered=True
        )
        to_plot = to_plot.sort_values("sort")

        # Negate gender coefficients to show men − women
        if rank_index == "Gender":
            to_plot["coefficient"] = -to_plot["coefficient"]
            to_plot["ci_lower"], to_plot["ci_upper"] = (
                -to_plot["ci_upper"].copy(),
                -to_plot["ci_lower"].copy(),
            )

        color = C_DARK if "top_ranked_venue" in venue_type else C_LIGHT
        colors     = ["white" if p > 0.05 else color for p in to_plot["p-value"]]
        edgecolors = ["grey"  if p > 0.05 else color for p in to_plot["p-value"]]

        ypos = 0
        for j in range(len(to_plot)):
            name = to_plot["name_to_plot"].iloc[j]
            edge = "k" if name == "All Resp." else edgecolors[j]
            sz   = ms ** 2 if name == "All Resp." else plt.rcParams["lines.markersize"] ** 2
            ax.scatter(to_plot["coefficient"].iloc[j], ypos,
                       marker="o", facecolors=colors[j], edgecolors=edge, s=sz, zorder=2)
            ax.hlines(ypos, to_plot["ci_lower"].iloc[j], to_plot["ci_upper"].iloc[j],
                      color=edgecolors[j], zorder=0)
            ypos += 1
            if "Biology" in name:
                if i == 0:           # record gap position only once
                    whitespace.append(ypos)
                ypos += 1

        ax.axvline(x=0, color="black", linewidth=1.5, zorder=0)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(xlim_prestige if rank_index == "Prestige" else xlim_gender)
        ax.grid(axis="x", alpha=0.5)

# y-axis labels (field names + N) on the left of panel C only
num_people = to_plot["n"].astype(int).tolist()
ax_labels  = [re.sub(rf"_{venue_type}", "", name) for name in to_plot["model"]]

pad = -0.4
ypos_label = 0
for num, label in zip(num_people, ax_labels):
    kw = {"weight": "bold"} if "Academia" in label else {}
    n_str = f"N={num}" if "Academia" in label else str(num)
    axes_bottom[0].text(
        -0.05, ypos_label, n_str,
        transform=blended_transform_factory(axes_bottom[0].transAxes, axes_bottom[0].transData),
        fontsize=label_fontsize, ha="right", va="center", **kw,
    )
    axes_bottom[0].text(
        pad, ypos_label, names_abbrev[label],
        transform=blended_transform_factory(axes_bottom[0].transAxes, axes_bottom[0].transData),
        fontsize=label_fontsize, ha="right", va="center", **kw,
    )
    ypos_label += 1
    if "Biology" in label:
        ypos_label += 1

# x-axis labels
axes_bottom[0].set_xlabel("Top preference rank\nper prestige decile",  fontsize=label_fontsize)
axes_bottom[1].set_xlabel("Top aspiration rank\nper prestige decile",   fontsize=label_fontsize)
axes_bottom[2].set_xlabel("Top preference rank diff.\nmen - women",     fontsize=label_fontsize)
axes_bottom[3].set_xlabel("Top aspiration rank diff.\nmen - women",     fontsize=label_fontsize)

# x-axis ticks
axes_bottom[0].set_xticks(np.arange(-0.04, 0.041, 0.02))
axes_bottom[1].set_xticks(np.arange(-0.04, 0.041, 0.02))
axes_bottom[2].set_xticks(np.arange(-0.2, 0.21, 0.1))
axes_bottom[3].set_xticks(np.arange(-0.2, 0.21, 0.1))

# Panel labels
for idx, ax in enumerate(axes_bottom):
    ax.text(0.01, 1.0, "CDEF"[idx],
            transform=ax.transAxes, fontsize=16, ha="left", va="top", weight="bold")
ax_A.text(-0.15, 1.03, "A", transform=ax_A.transAxes,
          fontsize=16, ha="left", va="top", weight="bold")
ax_B.text(-0.3,  1.03, "B", transform=ax_B.transAxes,
          fontsize=16, ha="left", va="top", weight="bold")

sns.despine()
# Remove left spine from panels D, E, F (must follow sns.despine which restores it)
for ax in axes_bottom[1:]:
    ax.spines["left"].set_visible(False)

out = common.FIGURES / "fig4.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
