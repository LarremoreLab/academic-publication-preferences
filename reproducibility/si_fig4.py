"""Generate the regression robustness check SI figure (permutation test).

Source: 04_aspiration_preference_regressions/04_robustness_check.ipynb
        (cells c86373e1, f8649fd3, 6226b0cf, cd010d51)

Target: _LATEX SOURCE/si_figs/aspiration_preferences_regression_robustness_check.pdf

Produces:
  - reproducibility/si_fig4.pdf

Data (same assembly as si_fig1.py):
  - public_data/respondents.csv
  - reproducibility/derived/field_consensus_rankings.csv
  - reproducibility/derived/individual_rankings.csv

Method:
  For each of 4 (label × outcome) combinations, run 10,000 within-field
  permutations of the label ('gender' or 'academia_prestige_bin_10'), refit
  the OLS model, and compare the permuted coefficient distribution to the
  actual estimated coefficient.

  Runtime: ~3 minutes on a modern laptop.

Run from the project root:
  .venv/bin/python reproducibility/si_fig4.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols

HERE    = Path(__file__).parent
ROOT    = HERE.parent
PUBLIC  = ROOT / "public_data"
DERIVED = HERE / "derived"

# ── Check required files ──────────────────────────────────────────────────────
_required = [
    PUBLIC  / "respondents.csv",
    DERIVED / "field_consensus_rankings.csv",
    DERIVED / "individual_rankings.csv",
]
_missing = [p for p in _required if not p.exists()]
if _missing:
    print("Missing required files:")
    for p in _missing:
        print(f"  {p}")
    print("\nPlease run first:\n  .venv/bin/python reproducibility/compute_rankings.py")
    sys.exit(1)

# ── Load and assemble data (same pipeline as si_fig1.py) ─────────────────────
respondents = pd.read_csv(PUBLIC  / "respondents.csv")
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

df["top_init_field_normed"]         = lookup_field_normed(df["field"], df["top_init_venue_db_id"])
df = df.join(top_venue, on="user_db_id")
df["top_ranked_venue_field_normed"] = lookup_field_normed(df["field"], df["top_ranked_venue_id"])

# Apply notebook filters: venue_bug==False, gender.notna()
# (academia_prestige_bin_10 NaN rows are dropped by OLS internally)
df = df[(df["venue_bug"] == False) & (df["gender"].notna())].copy()

# Career stage dummy variables (baseline: Assistant Professor)
career_dummies = pd.get_dummies(df["career_stage"])[["Associate Professor", "Full Professor"]]
career_dummies.columns = ["associate_prof_dummy", "full_prof_dummy"]
df = df.join(career_dummies)

print(f"Dataset: {len(df)} respondents after filters")

# ── Regression helper (from notebook cell f8649fd3) ───────────────────────────
def run_one_regression(data, outcome_type, level_type, prestige_level):
    model = ols(
        f"{outcome_type}~associate_prof_dummy+full_prof_dummy+{prestige_level}+gender",
        data=data,
    ).fit()
    coefficients         = model.params[1:]
    p_values             = model.pvalues[1:]
    confidence_intervals = model.conf_int()[1:]
    confidence_intervals.columns = ["ci_lower", "ci_upper"]
    n_participants = model.nobs

    results_df = coefficients.to_frame(name="coefficient")
    results_df["p-value"]               = p_values
    results_df[["ci_lower", "ci_upper"]] = confidence_intervals
    results_df["n"]     = n_participants
    results_df["model"] = f"{level_type}_{outcome_type}"
    results_df.index = ["Associate", "Full", "Prestige", "Gender"]
    return results_df


# ── Permutation test (from notebook cell 6226b0cf) ───────────────────────────
N_PERM = 10_000
PRESTIGE_VAR = "academia_prestige_bin_10"

combos = [
    ("gender",       "Gender"),
    (PRESTIGE_VAR,   "Prestige"),
]
outcomes = ["top_ranked_venue_field_normed", "top_init_field_normed"]

shuffled_fits = {}
actual_fits   = {}

for label, fit_label in combos:
    for outcome in outcomes:
        key = (fit_label, outcome)

        # Actual coefficient
        fit_df = run_one_regression(
            data=df, outcome_type=outcome,
            level_type="Academia", prestige_level=PRESTIGE_VAR,
        )
        actual_fits[key] = fit_df.loc[fit_label, "coefficient"]

        # Permuted coefficients
        print(f"  Permuting {key}  ({N_PERM:,} iterations)…", flush=True)
        perms = []
        for _ in range(N_PERM):
            perm_df = df.copy()
            perm_df[label] = df.groupby("field")[label].transform(np.random.permutation)
            fit_perm = run_one_regression(
                data=perm_df, outcome_type=outcome,
                level_type="Academia", prestige_level=PRESTIGE_VAR,
            )
            perms.append(fit_perm.loc[fit_label, "coefficient"])
        shuffled_fits[key] = perms

# ── Plot (from notebook cell cd010d51) ───────────────────────────────────────
sub_titles = {
    ("Gender",   "top_ranked_venue_field_normed"): "Preference x Gender",
    ("Gender",   "top_init_field_normed"):          "Aspiration x Gender",
    ("Prestige", "top_ranked_venue_field_normed"): "Preference x Prestige",
    ("Prestige", "top_init_field_normed"):          "Aspiration x Prestige",
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)

for ax, key in zip(axes.flatten(), shuffled_fits.keys()):
    ax.hist(
        shuffled_fits[key], bins=30,
        label="simulations with shuffled labels",
        alpha=0.8, color="tab:blue", edgecolor="w",
    )
    ax.axvline(np.mean(shuffled_fits[key]), c="k",
               label="average simulated coefficient")
    ax.axvline(actual_fits[key], c="r",
               label="estimated coefficient")

    if key == ("Prestige", "top_init_field_normed"):
        ax.set_xticks(ax.get_xticks()[1::2])
        ax.legend(framealpha=1, loc="lower left")

    ax.set_title(sub_titles[key])

for ax_row in axes:
    ax_row[0].set_ylabel("Number of simulations")

fig.text(0.5, 0.07, "Estimated Coefficient", ha="center", va="center")

out = HERE / "si_fig4.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"\nSaved: {out}")
