"""Generate preference realization figure (Fig 5).

Source: 05_publication_histories/04_plot_rankings DBL.ipynb  (cell 4, the 2-panel version)

Produces:
  - reproducibility/fig5.pdf  (main figure, panels A and B)

Data:
  - public_data/publication_counts.csv
      Per-user counts of how many of each respondent's top-5 venues
      (by personal preference and by field consensus) appear in their
      actual publication record.

Usage:
  .venv/bin/python reproducibility/fig5.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).parent.parent
PUBLIC = ROOT / "public_data"
OUT = Path(__file__).parent

# ── field color map (mirrors utils/field_colors.py) ───────────────────────────
_FIELDS_ORDERED = [
    "Biology", "Business", "Sociology", "Computer science",
    "Economics", "Engineering", "History", "Mathematics",
    "Medicine", "Philosophy", "Physics", "Psychology", "Chemistry",
]
_N = len(_FIELDS_ORDERED)
field_color_map = {
    field: plt.get_cmap("tab20")(i / _N)
    for i, field in enumerate(_FIELDS_ORDERED)
}

# ── x-axis label abbreviations ────────────────────────────────────────────────
names_abbrev = {
    "Academia":        "All Resp.",
    "Biology":         "Biology",
    "Business":        "Business",
    "Chemistry":       "Chemistry",
    "Computer science": "Comp. Sci.",
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


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data():
    path = PUBLIC / "publication_counts.csv"
    if not path.exists():
        sys.exit(
            "ERROR: public_data/publication_counts.csv not found.\n"
            "Re-run assemble_public_data.py to produce it."
        )
    return pd.read_csv(path)


def compute_means_sems(df):
    """Mean and SEM of field_count and indiv_count by field × prestige decile."""
    df = df.copy()
    df["prestige"] = 11 - df["institution_bin_10"]

    def _stats(col):
        mean = (
            df.groupby("field")[col]
            .apply(lambda x: df.loc[x.index].groupby("prestige")[col].mean())
            .unstack(fill_value=np.nan)
        )
        mean.loc["Academia"] = df.groupby("prestige")[col].mean()
        sem = (
            df.groupby("field")[col]
            .apply(lambda x: df.loc[x.index].groupby("prestige")[col].sem())
            .unstack(fill_value=np.nan)
        )
        sem.loc["Academia"] = df.groupby("prestige")[col].sem()
        cols = [i for i in range(1, 11) if i in mean.columns]
        return mean[cols], sem[cols]

    fc_mean, fc_sem = _stats("field_count")
    ic_mean, ic_sem = _stats("indiv_count")
    return fc_mean, fc_sem, ic_mean, ic_sem


def fit_regressions(fc_mean, ic_mean):
    """Linear regression per field; predictions at prestige=10 and slopes."""
    fields = fc_mean.index.tolist()
    # 'All Academia' was old name; filter it out defensively
    fields_to_plot = sorted([f for f in fields if f != "All Academia"])

    fc_pred10, ic_pred10 = {}, {}
    fc_pred10_se, ic_pred10_se = {}, {}

    x_all = fc_mean.columns.values

    for field in fields_to_plot:
        # field count
        y = fc_mean.loc[field].values
        mask = ~np.isnan(y)
        xf, yf = x_all[mask], y[mask]
        slope_f, intercept_f, _, _, se_f = stats.linregress(xf, yf)
        fc_pred10[field] = (slope_f * 10 + intercept_f) / 5
        n, xm = len(xf), np.mean(xf)
        sxx = np.sum((xf - xm) ** 2)
        mse = np.sum((yf - (slope_f * xf + intercept_f)) ** 2) / (n - 2)
        fc_pred10_se[field] = np.sqrt(mse * (1 / n + (10 - xm) ** 2 / sxx)) / 5

        # individual count
        y = ic_mean.loc[field].values
        mask = ~np.isnan(y)
        xi, yi = x_all[mask], y[mask]
        slope_i, intercept_i, _, _, se_i = stats.linregress(xi, yi)
        ic_pred10[field] = (slope_i * 10 + intercept_i) / 5
        mse_i = np.sum((yi - (slope_i * xi + intercept_i)) ** 2) / (n - 2)
        ic_pred10_se[field] = np.sqrt(mse_i * (1 / n + (10 - xm) ** 2 / sxx)) / 5

    return fields_to_plot, fc_pred10, ic_pred10, fc_pred10_se, ic_pred10_se


def significance_tests(fields_to_plot, fc_pred10, ic_pred10, fc_pred10_se, ic_pred10_se):
    """BH-corrected significance tests of (indiv − field) at prestige=10."""
    fields_no_acad = [f for f in fields_to_plot if f != "Academia"]
    p_values = []
    for field in fields_no_acad:
        diff = ic_pred10[field] - fc_pred10[field]
        se_diff = np.sqrt(fc_pred10_se[field] ** 2 + ic_pred10_se[field] ** 2)
        t_stat = diff / se_diff
        p_values.append(2 * (1 - stats.norm.cdf(abs(t_stat))))
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
    return {
        field: {
            "significant_05": reject[i],
            "significant_01": pvals_corrected[i] < 0.01,
        }
        for i, field in enumerate(fields_no_acad)
    }


def make_figure(fc_mean, fc_sem, ic_mean, ic_sem,
                fields_to_plot, fc_pred10, ic_pred10,
                fc_pred10_se, ic_pred10_se, corrected):
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["font.size"] = 12
    plt.rcParams["font.family"] = "Arial"

    fig, axs = plt.subplots(ncols=2, figsize=(13, 4.5))

    # ── Panel A: Academia-wide line plot ──────────────────────────────────────
    ax = axs[0]
    cols = ic_mean.columns.tolist()

    # respondent's top 5 (filled markers)
    for i, col in enumerate(cols):
        marker = "D" if i == len(cols) - 1 else "o"
        ax.plot(col, ic_mean.loc["Academia", col] / 5,
                marker=marker, linewidth=1 if i == 0 else 0, markersize=8,
                color="k", markerfacecolor="k", markeredgecolor="w",
                label="respondent's top 5" if i == 0 else "")
    ax.plot(cols, ic_mean.loc["Academia"] / 5, linewidth=1, color="k", zorder=0)

    # field's top 5 (open markers)
    for i, col in enumerate(cols):
        marker = "D" if i == len(cols) - 1 else "o"
        ax.plot(col, fc_mean.loc["Academia", col] / 5,
                marker=marker, linewidth=1 if i == 0 else 0, markersize=8,
                color="k", markerfacecolor="w", markeredgecolor="k",
                label="field's top 5" if i == 0 else "")
    ax.plot(cols, fc_mean.loc["Academia"] / 5, linewidth=1, color="k", zorder=0)

    ax.legend(fontsize=12, frameon=False)
    ax.set_xlabel("Prestige decile", fontsize=12)
    ax.set_ylabel("Fraction of top-5 venues\nin publication record", fontsize=12)
    ax.set_xticks(range(1, 11))
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0 of 5", "1 of 5", "2 of 5", "3 of 5", "4 of 5", "5 of 5"])
    ax.text(0.97, 0.02, "increasing prestige →",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12)

    # ── Panel B: per-field predictions at prestige=10 ─────────────────────────
    ax = axs[1]
    fields_panel_b = [f for f in fields_to_plot if f != "Academia"]
    x_positions = np.arange(len(fields_panel_b))

    max_y = max(
        max(fc_pred10[f] + 1.96 * fc_pred10_se[f] for f in fields_panel_b),
        max(ic_pred10[f] + 1.96 * ic_pred10_se[f] for f in fields_panel_b),
    )
    y_bar = max_y + 0.03

    for i, field in enumerate(fields_panel_b):
        color = field_color_map.get(field, "gray")

        fc_ci = 1.96 * fc_pred10_se[field]
        ax.errorbar(i - 0.15, fc_pred10[field],
                    yerr=fc_ci, fmt="D", markersize=8,
                    color=color, markerfacecolor="w", markeredgecolor=color,
                    markeredgewidth=1.5, capsize=3, elinewidth=1)

        ic_ci = 1.96 * ic_pred10_se[field]
        ax.errorbar(i + 0.15, ic_pred10[field],
                    yerr=ic_ci, fmt="D", markersize=8,
                    color=color, markerfacecolor=color, markeredgecolor="w",
                    markeredgewidth=1.5, capsize=3, elinewidth=1)

        if corrected[field]["significant_05"]:
            sig_marker = "**" if corrected[field]["significant_01"] else "*"
            ax.plot([i - 0.15, i + 0.15], [y_bar, y_bar], "k-", linewidth=1)
            ax.plot([i - 0.15, i - 0.15], [y_bar - 0.015, y_bar], "k-", linewidth=1)
            ax.plot([i + 0.15, i + 0.15], [y_bar - 0.015, y_bar], "k-", linewidth=1)
            ax.text(i, y_bar + 0.005, sig_marker,
                    ha="center", va="bottom", fontsize=14, fontweight="bold")
        else:
            ax.text(i, y_bar, "n.s.", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([names_abbrev[f] for f in fields_panel_b],
                       rotation=45, ha="right")
    ax.set_ylabel("Fraction of top-5 venues\nin publication record", fontsize=12)
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0 of 5", "1 of 5", "2 of 5", "3 of 5", "4 of 5", "5 of 5"])

    # panel labels
    axs[0].text(-0.15, 1.03, "A", transform=axs[0].transAxes,
                fontsize=16, ha="left", va="top", weight="bold")
    axs[1].text(-0.13, 1.03, "B", transform=axs[1].transAxes,
                fontsize=16, ha="left", va="top", weight="bold")

    for ax in axs:
        ax.grid(axis="y")
    sns.despine()
    plt.tight_layout()
    return fig


def main():
    df = load_data()
    fc_mean, fc_sem, ic_mean, ic_sem = compute_means_sems(df)
    fields_to_plot, fc_pred10, ic_pred10, fc_pred10_se, ic_pred10_se = fit_regressions(
        fc_mean, ic_mean
    )
    corrected = significance_tests(
        fields_to_plot, fc_pred10, ic_pred10, fc_pred10_se, ic_pred10_se
    )
    fig = make_figure(
        fc_mean, fc_sem, ic_mean, ic_sem,
        fields_to_plot, fc_pred10, ic_pred10,
        fc_pred10_se, ic_pred10_se, corrected,
    )
    out_path = OUT / "fig5.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
