"""SI Fig. S5: permutation test of the pooled prestige and gender coefficients.

Asks whether the pooled associations of top preference and top aspiration with prestige and
gender could arise from field composition alone, i.e. from differences across fields in how
prestige and gender are distributed. The statistic is the coefficient on the permuted variable
in the pooled Eq. S1 OLS model WITHOUT field fixed effects (so that field composition is the
only channel the permutation can act through),

    y_i = b0 + b1 Associate_i + b2 Full_i + b3 Prestige_i + b4 Gender_i + e_i,

on the fixed Eq. S1 estimation sample. In each of 10,000 permutations per panel, gender is
permuted among respondents within each field; prestige, an institution-level variable, is
permuted among the institutions within each field, all respondents at an institution moving
together. This preserves each field's aggregate composition and removes any within-field
association. Equal-tailed p-value with N permutations:

    p = min(1, 2 min(A, B)),  A = (1 + #{b_perm >= b_obs}) / (N + 1),  B = (1 + #{b_perm <= b_obs}) / (N + 1),

whose floor 2 / 10,001 is reported as "p < 2e-4". Histograms show the null distributions; the
black line is the null median and the red line the observed coefficient. Seeds are fixed.

Inputs:  public_data/respondents.csv and reproducibility/derived/ (through common.eqs1_sample)
Outputs: output/figures/si_fig5.pdf
         output/results/permutation_test.csv  per panel: observed coefficient, null median, mean and
             2.5% / 97.5% quantiles, p, seed, sample size
Runtime: well under a minute (40,000 small least-squares fits).

Usage (from the project root, after compute_rankings.py):
    .venv/bin/python reproducibility/si_fig5_permutation.py
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

N_PERM = 10_000
SEED = 20260929
OUTCOMES = list(common.OUTCOME_COLS.values())      # outcome columns: top preference, top aspiration
OUTCOME_NAME = {v: k for k, v in common.OUTCOME_COLS.items()}
COVS = ["associate_prof_dummy", "full_prof_dummy", common.PRESTIGE, "gender"]

t_start = time.time()
df = common.eqs1_sample()
df = df.dropna(subset=OUTCOMES + COVS + [common.CLUSTER, "field"]).copy()   # fixed sample
X0 = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in COVS])
col = {"Gender": 1 + COVS.index("gender"), "Prestige": 1 + COVS.index(common.PRESTIGE)}


def coef(X, y, j):
    return np.linalg.lstsq(X, y, rcond=None)[0][j]


# row indices by field, and (field, institution) blocks for prestige
field_rows = {f: np.flatnonzero(df["field"].to_numpy() == f) for f in df["field"].unique()}
inst = df[common.CLUSTER].to_numpy()
pv = df[common.PRESTIGE].to_numpy(float)
blocks = {}
for f, rows in field_rows.items():
    ids, inv = np.unique(inst[rows], return_inverse=True)
    vals = np.array([pv[rows][inv == k][0] for k in range(len(ids))])   # one prestige value per institution
    blocks[f] = (rows, inv, vals)


def permute_prestige(rng):
    out = pv.copy()
    for rows, inv, vals in blocks.values():
        out[rows] = rng.permutation(vals)[inv]
    return out


g = df["gender"].to_numpy(float)


def permute_gender(rng):
    out = g.copy()
    for rows in field_rows.values():
        out[rows] = rng.permutation(g[rows])
    return out


rows_out, null = [], {}
for k, (label, permute) in enumerate([("Gender", permute_gender), ("Prestige", permute_prestige)]):
    for m, outcome in enumerate(OUTCOMES):
        y = df[outcome].to_numpy(float)
        j = col[label]
        obs = coef(X0, y, j)
        ref = smf.ols(f"{outcome} ~ " + " + ".join(COVS), data=df).fit().params.iloc[j]
        assert np.isclose(obs, ref), (obs, ref)
        rng = np.random.default_rng([SEED, k, m])
        X = X0.copy()
        perms = np.empty(N_PERM)
        for i in range(N_PERM):
            X[:, j] = permute(rng)
            perms[i] = coef(X, y, j)
        A = (1 + (perms >= obs).sum()) / (N_PERM + 1)
        B = (1 + (perms <= obs).sum()) / (N_PERM + 1)
        p = min(1.0, 2 * min(A, B))
        at_floor = bool(np.isclose(p, 2 / (N_PERM + 1)))
        null[(label, outcome)] = (perms, obs)
        rows_out.append({"outcome": OUTCOME_NAME[outcome], "quantity": f"{label.lower()}_coefficient",
                         "estimate": obs, "null_median": np.median(perms), "null_mean": perms.mean(),
                         "null_q025": np.quantile(perms, 0.025), "null_q975": np.quantile(perms, 0.975),
                         "p": p, "p_at_floor": at_floor,
                         "p_report": "p < 2e-4" if at_floor else f"{p:.4f}",
                         "n_perm": N_PERM, "seed": f"[{SEED}, {k}, {m}]",
                         "n_obs": len(df), "n_clusters": df[common.CLUSTER].nunique(),
                         "shuffle": "within field, across institutions" if label == "Prestige"
                         else "within field, across respondents"})

res = pd.DataFrame(rows_out)
res.to_csv(common.RESULTS / "permutation_test.csv", index=False)
pd.set_option("display.width", 200)
print(res[["outcome", "quantity", "estimate", "null_median", "p_report", "n_obs", "n_clusters"]].to_string(index=False))
print(f"Saved: {common.RESULTS / 'permutation_test.csv'}")

# ── Plot ─────────────────────────────────────────────────────────────────────
sub_titles = {
    ("Gender",   "top_ranked_venue_field_normed"): "Preference x Gender",
    ("Gender",   "top_init_field_normed"):          "Aspiration x Gender",
    ("Prestige", "top_ranked_venue_field_normed"): "Preference x Prestige",
    ("Prestige", "top_init_field_normed"):          "Aspiration x Prestige",
}
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
for ax, key in zip(axes.flatten(), null.keys()):
    perms, obs = null[key]
    ax.hist(perms, bins=30, label="simulations with shuffled labels",
            alpha=0.8, color="tab:blue", edgecolor="w")
    ax.axvline(np.median(perms), c="k", label="median simulated coefficient")
    ax.axvline(obs, c="r", label="estimated coefficient")
    lo, hi = min(perms.min(), obs), max(perms.max(), obs)
    ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))  # keep the observed line off the frame
    if key == ("Prestige", "top_init_field_normed"):
        ax.set_xticks(ax.get_xticks()[1::2])
        # the legend goes below the panels so that it does not cover the bars
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
    ax.set_title(sub_titles[key])
for ax_row in axes:
    ax_row[0].set_ylabel("Number of simulations")
fig.text(0.5, 0.07, "Estimated Coefficient", ha="center", va="center")
out = common.FIGURES / "si_fig5.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
print(f"Runtime: {time.time() - t_start:.0f} s")
