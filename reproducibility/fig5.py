"""Fig. 5: alignment between stated preferences and publication records, by institutional prestige.

Paper items produced
  Fig. 5 (output/figures/fig5.pdf)
    A  Observed mean fraction of the five reference venues found in respondents' publication
       records, by prestige decile (10 = most prestigious), for the respondent's own top 5
       (solid markers) and the field's top 5 (open markers). Lines are predictions from the
       pooled model, averaged over respondents' fields.
    B  Per-field model predictions at decile 10 with 95% confidence intervals. A bracket and
       asterisk mark fields where the own-minus-field difference at decile 10 is significant
       after Benjamini-Hochberg adjustment across the 13 fields at alpha = 0.01; "n.s." otherwise.
  SI Table S2 is built from output/results/fig5_models.csv by si_table2_fig5_models.py.

Inputs
  public_data/publication_counts.csv, through common.fig5_long(): respondents in the final
  sample (publication history, prestige decile and institution; N = 1,872). indiv_count is the
  number of the respondent's own top-5 venues ever published in; field_count is the same for
  the field's top 5 (public_data/field_top5.csv: the five venues appearing most often in
  respondents' individual top-5 lists, as in Fig. 2).

Model
  Each respondent contributes two observations, y = count / 5 for ref in {own, field}:
    per field:  y ~ C(ref) * prestige
    pooled:     y ~ C(ref) * (prestige + C(field))
  Ordinary least squares with standard errors clustered by institution and a t reference
  distribution (common.fit_clustered). Every reported quantity is a linear contrast of the
  coefficients, evaluated with the fitted model's t_test so the clustered covariance carries
  through:
    slope(ref)   change in y per one-decile increase in prestige for that reference group
    pred10(ref)  predicted y at prestige decile 10 (pooled model: averaged over respondents' fields)
    diff10       pred10(own) - pred10(field)
  The 13 per-field diff10 p-values are Benjamini-Hochberg adjusted (alpha = 0.01); the pooled
  (Academia) diff10 is reported outside that family.

Outputs
  output/figures/fig5.pdf
  output/results/fig5_models.csv        slope, pred10 and diff10 per field and pooled, with
                                        clustered SEs, 95% CIs, p-values and BH-adjusted p-values
  output/results/fig5_lines.csv         pooled-model predictions by decile and ref (panel A lines)
  output/results/fig5_decile_means.csv  observed decile means and SEMs (panel A markers)

Usage (from the repository root): .venv/bin/python reproducibility/fig5.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from patsy import build_design_matrices
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ALPHA_BH = 0.01
POOLED = "Academia"

NAMES_ABBREV = {
    "Biology": "Biology", "Business": "Business", "Chemistry": "Chemistry",
    "Computer science": "Comp. Sci.", "Economics": "Economics", "Engineering": "Engineering",
    "History": "History", "Mathematics": "Math", "Medicine": "Medicine",
    "Philosophy": "Philosophy", "Physics": "Physics", "Psychology": "Psychology",
    "Sociology": "Sociology",
}


# ── contrasts ─────────────────────────────────────────────────────────────────

def design_row(m, resp, ref, prestige):
    """Mean design-matrix row of model `m` over respondents `resp` with ref and prestige fixed.

    For the per-field model every respondent gives the same row; for the pooled model the mean
    averages the field intercepts over the respondents' own fields."""
    nd = resp.assign(ref=ref, prestige=float(prestige))
    X = np.asarray(build_design_matrices([m.model.data.design_info], nd)[0])
    return X.mean(axis=0)


def contrast(m, L):
    """L . beta with its clustered standard error, 95% CI and t-test p-value."""
    t = m.t_test(L.reshape(1, -1))
    lo, hi = np.asarray(t.conf_int(alpha=0.05)).ravel()
    return {"estimate": float(np.ravel(t.effect)[0]), "se": float(np.ravel(t.sd)[0]),
            "ci_lower": lo, "ci_upper": hi, "p": float(np.ravel(t.pvalue)[0]), "df": float(t.df_denom)}


def quantities(m, d, level):
    """slope(own), pred10(own), slope(field), pred10(field) and diff10 for one fitted model."""
    resp = d[d["ref"] == "own"][["field", "prestige", common.CLUSTER]]  # one row per respondent
    top = resp["prestige"] == 10
    base = {"level": level, "n_respondents": len(resp), "n_clusters": resp[common.CLUSTER].nunique(),
            "n_decile10": int(top.sum()), "n_clusters_decile10": resp.loc[top, common.CLUSTER].nunique()}
    L = {}
    for ref in ("own", "field"):
        L[("slope", ref)] = design_row(m, resp, ref, 1) - design_row(m, resp, ref, 0)
        L[("pred10", ref)] = design_row(m, resp, ref, 10)
    L[("diff10", "own-field")] = L[("pred10", "own")] - L[("pred10", "field")]
    return [{**base, "quantity": q, "ref": r, **contrast(m, vec)} for (q, r), vec in L.items()]


def fit_models(long):
    """Per-field and pooled models; returns (results table, pooled predictions by decile)."""
    rows = []
    for field in common.FIELDS:
        d = long[long["field"] == field]
        rows += quantities(common.fit_clustered("y ~ C(ref) * prestige", d), d, field)
    m_pooled = common.fit_clustered("y ~ C(ref) * (prestige + C(field))", long)
    rows += quantities(m_pooled, long, POOLED)
    res = pd.DataFrame(rows)

    # Check: per-field slopes and predictions equal those of a separate line fit per reference group
    for field in common.FIELDS:
        d = long[long["field"] == field]
        for ref in ("own", "field"):
            b = np.polyfit(d.loc[d.ref == ref, "prestige"], d.loc[d.ref == ref, "y"], 1)
            r = res[(res.level == field) & (res.ref == ref)].set_index("quantity")["estimate"]
            assert np.isclose(r["slope"], b[0]) and np.isclose(r["pred10"], b[0] * 10 + b[1])

    # Benjamini-Hochberg across the 13 per-field diff10 tests; the pooled diff10 is outside the family
    is_fam = (res["quantity"] == "diff10") & (res["level"] != POOLED)
    assert is_fam.sum() == len(common.FIELDS)
    reject, p_bh, _, _ = multipletests(res.loc[is_fam, "p"], alpha=ALPHA_BH, method="fdr_bh")
    res["p_bh"] = np.nan
    res.loc[is_fam, "p_bh"] = p_bh
    res["sig_bh_01"] = pd.NA
    res.loc[is_fam, "sig_bh_01"] = reject

    # Panel A lines: pooled-model prediction at each decile, averaged over respondents' fields
    resp = long[long["ref"] == "own"][["field", "prestige", common.CLUSTER]]
    lines = pd.DataFrame([{"ref": ref, "prestige": p, **contrast(m_pooled, design_row(m_pooled, resp, ref, p))}
                          for ref in ("own", "field") for p in range(1, 11)])
    return res, lines


def decile_means(long):
    """Observed mean, SEM and n of y by prestige decile and reference group (panel A markers)."""
    g = long.groupby(["prestige", "ref"])["y"].agg(["mean", "sem", "count"]).reset_index()
    g["prestige"] = g["prestige"].astype(int)
    return g.rename(columns={"count": "n"}).sort_values(["ref", "prestige"], ascending=[False, True])


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(res, lines, means):
    field_color_map = common.field_colors()
    pred = res[res["quantity"] == "pred10"].set_index(["level", "ref"])
    diff = res[res["quantity"] == "diff10"].set_index("level")
    dec = means.pivot(index="prestige", columns="ref", values="mean")

    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["font.size"] = 12
    plt.rcParams["font.family"] = "Arial"

    fig, axs = plt.subplots(ncols=2, figsize=(13, 4.5))

    # ── Panel A: observed decile means (markers) and pooled-model predictions (lines) ──
    ax = axs[0]
    cols = dec.index.tolist()
    for ref, face, edge, label in [
        ("own", "k", "w", "respondent's top 5"),
        ("field", "w", "k", "field's top 5"),
    ]:
        for i, col in enumerate(cols):
            marker = "D" if i == len(cols) - 1 else "o"
            ax.plot(col, dec.loc[col, ref],
                    marker=marker, linewidth=0, markersize=8,
                    color="k", markerfacecolor=face, markeredgecolor=edge,
                    label=label if i == 0 else "")
        ln = lines[lines["ref"] == ref]
        ax.plot(ln["prestige"], ln["estimate"], linewidth=1, color="k", zorder=0)

    ax.legend(fontsize=12, frameon=False)
    ax.set_xlabel("Prestige decile", fontsize=12)
    ax.set_ylabel("Fraction of top-5 venues\nin publication record", fontsize=12)
    ax.set_xticks(range(1, 11))
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0 of 5", "1 of 5", "2 of 5", "3 of 5", "4 of 5", "5 of 5"])
    ax.text(0.97, 0.02, "increasing prestige →",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12)

    # ── Panel B: per-field predictions at prestige decile 10 ──────────────────
    ax = axs[1]
    fields_panel_b = sorted(f for f in res["level"].unique() if f != POOLED)
    x_positions = np.arange(len(fields_panel_b))

    max_y = max(pred.loc[(f, r), "ci_upper"] for f in fields_panel_b for r in ("own", "field"))
    y_bar = max_y + 0.03

    for i, field in enumerate(fields_panel_b):
        color = field_color_map.get(field, "gray")
        for ref, dx, face, edge in [("field", -0.15, "w", color), ("own", 0.15, color, "w")]:
            p = pred.loc[(field, ref)]
            ax.errorbar(i + dx, p["estimate"],
                        yerr=[[p["estimate"] - p["ci_lower"]], [p["ci_upper"] - p["estimate"]]],
                        fmt="D", markersize=8,
                        color=color, markerfacecolor=face, markeredgecolor=edge,
                        markeredgewidth=1.5, capsize=3, elinewidth=1)

        if bool(diff.loc[field, "sig_bh_01"]):
            ax.plot([i - 0.15, i + 0.15], [y_bar, y_bar], "k-", linewidth=1)
            ax.plot([i - 0.15, i - 0.15], [y_bar - 0.015, y_bar], "k-", linewidth=1)
            ax.plot([i + 0.15, i + 0.15], [y_bar - 0.015, y_bar], "k-", linewidth=1)
            ax.text(i, y_bar + 0.005, "*",
                    ha="center", va="bottom", fontsize=14, fontweight="bold")
        else:
            ax.text(i, y_bar, "n.s.", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([NAMES_ABBREV[f] for f in fields_panel_b], rotation=45, ha="right")
    ax.set_ylabel("Fraction of top-5 venues\nin publication record", fontsize=12)
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0 of 5", "1 of 5", "2 of 5", "3 of 5", "4 of 5", "5 of 5"])

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
    long = common.fig5_long()
    assert set(long["field"]) == set(common.FIELDS)
    res, lines = fit_models(long)
    means = decile_means(long)

    res.to_csv(common.RESULTS / "fig5_models.csv", index=False)
    lines.to_csv(common.RESULTS / "fig5_lines.csv", index=False)
    means.to_csv(common.RESULTS / "fig5_decile_means.csv", index=False)

    fig = make_figure(res, lines, means)
    out = common.FIGURES / "fig5.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")

    pd.set_option("display.width", 200)
    show = res.copy()
    for c in ["estimate", "ci_lower", "ci_upper"]:
        show[c] = show[c].round(3)
    print(show[["level", "quantity", "ref", "estimate", "ci_lower", "ci_upper", "p", "p_bh", "sig_bh_01",
                "n_respondents", "n_clusters"]].to_string(index=False))
    d10 = res[(res.level == POOLED) & (res.quantity == "diff10")].iloc[0]
    n_sig = int(res.loc[res.quantity == "diff10", "sig_bh_01"].eq(True).sum())
    print(f"\nPooled own - field difference at decile 10: {5 * d10.estimate:.2f} venues "
          f"(95% CI {5 * d10.ci_lower:.2f} to {5 * d10.ci_upper:.2f}, p = {d10.p:.2g}); "
          f"{n_sig} of {len(common.FIELDS)} fields significant after BH at alpha = {ALPHA_BH}")


if __name__ == "__main__":
    main()
