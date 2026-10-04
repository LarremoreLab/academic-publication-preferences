"""Shared loaders and model helpers for the reproducibility scripts.

Everything reads only public_data/ and reproducibility/derived/ (the latter is produced by
compute_rankings.py). Conventions used throughout the paper:

  prestige_decile   AARC hiring-network prestige, Academia-wide deciles of equal 2023 faculty
                    headcount, 10 = most prestigious (public_data/respondents.csv)
  institution_id    anonymous institution index; standard errors are clustered on it
  Eq. S1 sample     professors (assistant / associate / full) with a reported gender of man or
                    woman, not affected by the aspiration-menu error (venue_bug), and with a
                    prestige decile; N = 2,196
  Fig. 5 sample     respondents with a publication history, a prestige decile and an institution
                    (in_final_sample in public_data/publication_counts.csv); N = 1,872

Scripts write figures to reproducibility/output/figures/, LaTeX tables to
reproducibility/output/tables/ and machine-readable results to reproducibility/output/results/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public_data"
DERIVED = ROOT / "reproducibility" / "derived"
OUTPUT = ROOT / "reproducibility" / "output"
FIGURES, TABLES, RESULTS = OUTPUT / "figures", OUTPUT / "tables", OUTPUT / "results"
CACHE = ROOT / "reproducibility" / "cache"
for _d in (FIGURES, TABLES, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

FIELDS = ["Biology", "Business", "Chemistry", "Computer science", "Economics", "Engineering", "History",
          "Mathematics", "Medicine", "Philosophy", "Physics", "Psychology", "Sociology"]
PROFESSOR_STAGES = ["Assistant Professor", "Associate Professor", "Full Professor"]
CLUSTER = "institution_id"
PRESTIGE = "prestige_decile"

# Eq. S1 outcomes: field-normalized rank of the top preference (from the individual ranking) and of
# the top aspiration (the first venue named in the survey)
OUTCOME_COLS = {"top_preference": "top_ranked_venue_field_normed", "top_aspiration": "top_init_field_normed"}
COEF_LABELS = {"associate_prof_dummy": "Associate", "full_prof_dummy": "Full", PRESTIGE: "Prestige", "gender": "Gender"}


def require_derived():
    missing = [f for f in ("individual_rankings.csv", "field_rankings_per_user.csv",
                           "field_consensus_rankings.csv", "global_rankings_per_user.csv")
               if not (DERIVED / f).exists()]
    if missing:
        raise SystemExit(f"Missing {missing} in {DERIVED}. Run reproducibility/compute_rankings.py first.")


def load_respondents():
    return pd.read_csv(PUBLIC / "respondents.csv")


def field_normed_scores():
    """(field, venue_db_id) -> field-normalized consensus score (0 = lowest, 1 = highest in the field)."""
    fc = pd.read_csv(DERIVED / "field_consensus_rankings.csv")
    fc["normed"] = fc.groupby("field")["score"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    return fc.set_index(["field", "venue_db_id"])["normed"]


def eqs1_sample():
    """Respondent-level data for Eq. S1 (one row per respondent, indexed by user_db_id).

    Columns: field, career_stage, gender (0 = man, 1 = woman), associate_prof_dummy, full_prof_dummy,
    prestige_decile, institution_id, top_init_field_normed, top_ranked_venue_field_normed.
    Outcomes may be missing for a few respondents; model fits drop those rows.
    """
    require_derived()
    resp = load_respondents()
    normed = field_normed_scores()
    indiv = pd.read_csv(DERIVED / "individual_rankings.csv")
    top_venue = indiv[indiv["ordinal_rank"] == 0].set_index("user_db_id")["venue_db_id"].rename("top_ranked_venue_id")

    def lookup(fields, venues):
        return pd.Series([normed.get(k, np.nan) for k in zip(fields, venues)], index=fields.index)

    df = resp.copy()
    df["gender"] = df["gender"].map({"gm": 0.0, "gf": 1.0})
    df = df[df["career_stage"].isin(PROFESSOR_STAGES)].copy()
    df["top_init_field_normed"] = lookup(df["field"], df["top_init_venue_db_id"])
    df = df.join(top_venue, on="user_db_id")
    df["top_ranked_venue_field_normed"] = lookup(df["field"], df["top_ranked_venue_id"])
    df = df[(df["venue_bug"] == False) & df["gender"].notna() & df[PRESTIGE].notna()].copy()
    dummies = pd.get_dummies(df["career_stage"])[["Associate Professor", "Full Professor"]].astype(float)
    dummies.columns = ["associate_prof_dummy", "full_prof_dummy"]
    df = df.join(dummies)
    df[PRESTIGE] = df[PRESTIGE].astype(float)
    df[CLUSTER] = df[CLUSTER].astype(float)
    df.index = df["user_db_id"].values
    return df


def eqs1_formula(outcome, field_fe=False):
    y = OUTCOME_COLS.get(outcome, outcome)
    return f"{y} ~ associate_prof_dummy + full_prof_dummy + {PRESTIGE} + gender" + (" + C(field)" if field_fe else "")


def fit_clustered(formula, data, cluster=CLUSTER):
    """OLS with standard errors clustered on `cluster` and a t reference distribution.
    Rows with a missing value in any formula variable or in the cluster variable are dropped."""
    cols = _formula_columns(formula, data) + [cluster]
    d = data.dropna(subset=cols)
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[cluster]}, use_t=True)


def fit_conventional(formula, data, cluster=CLUSTER):
    """Same rows as fit_clustered, conventional (iid) standard errors."""
    cols = _formula_columns(formula, data) + [cluster]
    return smf.ols(formula, data=data.dropna(subset=cols)).fit()


def _formula_columns(formula, data):
    import re
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)) - {"C"}
    return [c for c in data.columns if c in names]


def fig5_long():
    """Stacked Fig. 5 data: two rows per respondent in the final sample, y = fraction of the five
    reference venues (own top 5 / field top 5) found in the publication record."""
    pc = pd.read_csv(PUBLIC / "publication_counts.csv")
    fin = pc[pc["in_final_sample"]].copy()
    fin["prestige"] = fin[PRESTIGE].astype(float)
    fin[CLUSTER] = fin[CLUSTER].astype(float)
    keep = ["user_db_id", "field", "prestige", CLUSTER]
    long = pd.concat([fin[keep].assign(ref="own", y=fin["indiv_count"] / 5),
                      fin[keep].assign(ref="field", y=fin["field_count"] / 5)], ignore_index=True)
    assert long[["y", "prestige", CLUSTER]].notna().all().all()
    return long


def field_colors():
    import sys
    sys.path.insert(0, str(ROOT))
    from utils.field_colors import field_color_map  # noqa: E402
    return field_color_map
