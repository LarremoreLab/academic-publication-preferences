"""Numbers quoted in the SI section on statistical inference: clustering by institution, field
fixed effects, a multilevel check, functional form (fractional logit), fitted-value ranges, and
the self-consistency of the pairwise comparisons.

All models are at the specification used throughout the paper: prestige deciles (10 = most
prestigious), OLS with standard errors clustered by institution and t reference distributions,
field fixed effects in pooled models. Eq. S1 is

    y_i = b0 + b1 Associate_i + b2 Full_i + b3 Prestige_i + b4 Gender_i + e_i

for the field-normalized rank of the top preference and of the top aspiration, estimated pooled
over all respondents with field fixed effects and within each field (common.eqs1_sample,
N = 2,196). The Fig. 5 models regress the fraction of five reference venues (the respondent's
own top 5, or the field's top 5) found in the publication record on prestige, stacked over the
two reference sets, y ~ C(ref) * prestige per field and y ~ C(ref) * (prestige + C(field))
pooled (common.fig5_long, N = 1,872 respondents).

Sections (the `section` column of the output CSV):
  clustering        conventional vs institution-clustered SE for every Eq. S1 coefficient (ratio,
                    both p-values, significance changes at alpha = 0.05)
  fixed_effects     pooled Eq. S1 with and without field fixed effects
  mixed_model       linear mixed model with a random intercept for institution on the field-fixed-
                    effects specification (REML, statsmodels MixedLM, default optimizer settings).
                    Convergence is reported as returned by the optimizer; the fit is not tuned.
  icc               residual intraclass correlation by institution of the field-FE OLS (one-way
                    ANOVA estimator for unequal group sizes), 95% percentile interval from a cluster
                    bootstrap over institutions (1,000 replicates), design effect 1 + (n0 - 1) ICC
                    at the ANOVA mean cluster size n0
  fractional_logit  every Eq. S1 and Fig. 5 regression refit as a binomial GLM with a logit link and
                    institution-clustered SE; average marginal effects (delta method, normal-based CI)
                    compared with the OLS coefficients and contrasts
  fitted_values     range of the OLS fitted values (Eq. S1, Fig. 5) and of the OLS predictions and
                    confidence bounds plotted in Fig. 5
  self_consistency  share of strict pairwise comparisons that disagree with the respondent's own
                    inferred ranking (overall and by field), and share of respondents with none

Inputs:  public_data/respondents.csv, publication_counts.csv, comparisons.csv; reproducibility/derived/
Outputs: output/results/inference_checks.csv  (long format: section, outcome, level, term, quantity,
         value, note) and a printed summary
Runtime: under a minute.

Usage (from the project root, after compute_rankings.py):
    .venv/bin/python reproducibility/si_inference_numbers.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy import stats
from scipy.special import expit
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

CL, P = common.CLUSTER, common.PRESTIGE
OUTCOMES = list(common.OUTCOME_COLS)                 # ["top_preference", "top_aspiration"]
EQS1_VARS = ["associate_prof_dummy", "full_prof_dummy", P, "gender"]
TERMS = list(common.COEF_LABELS.items())             # [(column, label), ...]
POOLED, POOLED_NO_FE, MIXED = "Academia", "Academia (no field FE)", "Academia (mixed model)"

rows = []


def add(section, quantity, value, outcome="", level="", term="", note=""):
    rows.append({"section": section, "outcome": outcome, "level": level, "term": term,
                 "quantity": quantity, "value": float(value), "note": note})


def header(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def estimation_frame(data, outcome):
    """Rows of the Eq. S1 estimation sample for `outcome` (no missing model variable)."""
    return data.dropna(subset=[common.OUTCOME_COLS[outcome]] + EQS1_VARS + [CL, "field"])


def eqs1_levels(df):
    """(outcome, level, data, field_fe) for the pooled model with field FE and the 13 fields."""
    for outcome in OUTCOMES:
        yield outcome, POOLED, df, True
        for f in common.FIELDS:
            yield outcome, f, df[df["field"] == f], False


def term_stats(m, term):
    lo, hi = m.conf_int().loc[term]
    return {"estimate": m.params[term], "se": m.bse[term], "ci_lower": lo, "ci_upper": hi, "p": m.pvalues[term]}


df = common.eqs1_sample()
pd.set_option("display.width", 200)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Clustering by institution: conventional vs clustered SE, every Eq. S1 coefficient
# ═══════════════════════════════════════════════════════════════════════════════
header("1. Clustering by institution (Eq. S1: pooled with field FE + 13 fields, both outcomes)")
ratio_rows = []
for outcome, level, data, fe in list(eqs1_levels(df)) + [(o, POOLED_NO_FE, df, False) for o in OUTCOMES]:
    formula = common.eqs1_formula(outcome, field_fe=fe)
    mc, mo = common.fit_clustered(formula, data), common.fit_conventional(formula, data)
    for term, label in TERMS:
        ratio = mc.bse[term] / mo.bse[term]
        changed = (mo.pvalues[term] < 0.05) != (mc.pvalues[term] < 0.05)
        ratio_rows.append({"outcome": outcome, "level": level, "term": label, "ratio": ratio,
                           "p_conventional": mo.pvalues[term], "p_clustered": mc.pvalues[term], "changed": changed})
        add("clustering", "se_ratio_clustered_over_conventional", ratio, outcome, level, label)
        add("clustering", "p_conventional", mo.pvalues[term], outcome, level, label)
        add("clustering", "p_clustered", mc.pvalues[term], outcome, level, label)
ratios = pd.DataFrame(ratio_rows)
main = ratios[ratios["level"] != POOLED_NO_FE]          # the 28 regressions of the paper
for label in ["Prestige", "Gender", "Associate", "Full"]:
    s = main[main["term"] == label]
    changes = s[s["changed"]]
    note = "; ".join(f"{r.outcome} {r.level}: p {r.p_conventional:.3f} (conventional) -> {r.p_clustered:.3f} (clustered)"
                     for r in changes.itertuples())
    add("clustering", "se_ratio_median", s["ratio"].median(), term=label, note=f"{len(s)} coefficients")
    add("clustering", "se_ratio_min", s["ratio"].min(), term=label)
    add("clustering", "se_ratio_max", s["ratio"].max(), term=label)
    add("clustering", "n_significance_changes", len(changes), term=label, note=note)
    print(f"  {label:9s} SE ratio clustered/conventional: median {s['ratio'].median():.3f}, "
          f"range {s['ratio'].min():.2f}-{s['ratio'].max():.2f} ({len(s)} coefficients); "
          f"significance changes at 0.05: {len(changes)}" + (f"  [{note}]" if note else ""))
pg = main[main["term"].isin(["Prestige", "Gender"])]
add("clustering", "n_prestige_gender_coefficients", len(pg))
add("clustering", "n_prestige_gender_significance_changes", pg["changed"].sum())
print(f"  Prestige and gender together: {len(pg)} coefficients, {pg['changed'].sum()} significance change(s)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Field fixed effects: pooled Eq. S1 with and without C(field)
# ═══════════════════════════════════════════════════════════════════════════════
header("2. Pooled Eq. S1 with and without field fixed effects (clustered SE)")
fe_fits = {}
for outcome in OUTCOMES:
    for level, fe in [(POOLED_NO_FE, False), (POOLED, True)]:
        m = common.fit_clustered(common.eqs1_formula(outcome, field_fe=fe), df)
        fe_fits[(outcome, level)] = m
        add("fixed_effects", "n_obs", m.nobs, outcome, level)
        add("fixed_effects", "n_clusters", m.model.data.frame[CL].nunique(), outcome, level)
        for term, label in TERMS:
            for k, v in term_stats(m, term).items():
                add("fixed_effects", k, v, outcome, level, label)
            s = term_stats(m, term)
            print(f"  {outcome:15s} {level:24s} {label:9s} {s['estimate']:+.4f}  SE {s['se']:.4f}  "
                  f"CI ({s['ci_lower']:+.4f}, {s['ci_upper']:+.4f})  p {s['p']:.2e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Multilevel check: random intercept for institution (REML)
# ═══════════════════════════════════════════════════════════════════════════════
header("3. Linear mixed model, random intercept for institution, field FE in the fixed part (REML)")
for outcome in OUTCOMES:
    d = estimation_frame(df, outcome)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mm = smf.mixedlm(common.eqs1_formula(outcome, field_fe=True), d, groups=d[CL]).fit()
    msgs = "; ".join(f"{w.category.__name__}: {w.message}" for w in caught)
    var_inst, var_resid = float(mm.cov_re.iloc[0, 0]), float(mm.scale)
    for term, label in TERMS:
        for k, v in term_stats(mm, term).items():
            add("mixed_model", k, v, outcome, MIXED, label, note="REML; model-based SE; normal reference distribution")
    add("mixed_model", "institution_variance", var_inst, outcome, MIXED)
    add("mixed_model", "residual_variance", var_resid, outcome, MIXED)
    add("mixed_model", "implied_icc", var_inst / (var_inst + var_resid), outcome, MIXED)
    add("mixed_model", "converged", mm.converged, outcome, MIXED, note=msgs or "no warnings")
    fe = fe_fits[(outcome, POOLED)]
    max_diff = max(abs(mm.params[t] - fe.params[t]) for t, _ in TERMS)
    add("mixed_model", "max_abs_difference_from_field_fe_ols", max_diff, outcome, MIXED)
    print(f"  {outcome:15s} converged: {mm.converged}; institution variance {var_inst:.5f}, residual variance "
          f"{var_resid:.4f}, implied ICC {var_inst / (var_inst + var_resid):.3f}; "
          f"max |mixed - field-FE OLS| over the four coefficients {max_diff:.4f}")
    for term, label in TERMS[2:]:
        s = term_stats(mm, term)
        print(f"      {label:9s} {s['estimate']:+.4f}  SE {s['se']:.4f}  p {s['p']:.2e}")
    if msgs:
        print(f"      warnings: {msgs}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Intraclass correlation by institution (field-FE OLS residuals) with a cluster bootstrap
# ═══════════════════════════════════════════════════════════════════════════════
header("4. Residual intraclass correlation by institution (field-FE OLS), cluster bootstrap 95% CI")
ICC_SEED, N_BOOT = 20261001, 1000


def anova_icc(resid, groups):
    """One-way ANOVA ICC for unequal group sizes: (MSB - MSW) / (MSB + (n0 - 1) MSW), with
    n0 = (N - sum n_i^2 / N) / (k - 1). Returns (icc, n0, k, N); the estimate can be negative."""
    d = pd.DataFrame({"r": resid, "g": groups})
    g = d.groupby("g")["r"]
    n_i, m_i = g.size().to_numpy(float), g.mean().to_numpy()
    N, k = n_i.sum(), len(n_i)
    ssb = (n_i * (m_i - d["r"].mean()) ** 2).sum()
    ssw = ((d["r"] - g.transform("mean")) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw), n0, k, int(N)


def fe_residuals(d, ycol):
    """Residuals of y ~ Eq. S1 covariates + field dummies, by least squares (numpy, for speed)."""
    fields = sorted(d["field"].unique())
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in EQS1_VARS]
                        + [(d["field"] == f).to_numpy(float) for f in fields[1:]])
    y = d[ycol].to_numpy(float)
    with np.errstate(all="ignore"):   # some BLAS builds emit spurious floating-point warnings here
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


for i, outcome in enumerate(OUTCOMES):
    ycol = common.OUTCOME_COLS[outcome]
    d = estimation_frame(df, outcome).reset_index(drop=True)
    ref = common.fit_conventional(common.eqs1_formula(outcome, field_fe=True), d)
    r = fe_residuals(d, ycol)
    assert np.allclose(r, ref.resid.to_numpy(), atol=1e-10)
    icc, n0, k, N = anova_icc(r, d[CL].to_numpy())

    insts = d[CL].unique()
    by_inst = {j: np.flatnonzero(d[CL].to_numpy() == j) for j in insts}
    rng = np.random.default_rng([ICC_SEED, i])
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        draw = rng.choice(insts, size=len(insts), replace=True)   # each draw is its own cluster
        idx = np.concatenate([by_inst[j] for j in draw])
        grp = np.concatenate([np.full(len(by_inst[j]), c) for c, j in enumerate(draw)])
        boot[b] = anova_icc(fe_residuals(d.iloc[idx].reset_index(drop=True), ycol), grp)[0]
    assert np.isfinite(boot).all()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    deff = 1 + (n0 - 1) * icc
    singles = int((d.groupby(CL).size() == 1).sum())
    add("icc", "icc", icc, outcome, POOLED)
    add("icc", "ci_lower", lo, outcome, POOLED, note=f"cluster bootstrap, {N_BOOT} replicates, seed [{ICC_SEED}, {i}]")
    add("icc", "ci_upper", hi, outcome, POOLED)
    add("icc", "bootstrap_median", np.median(boot), outcome, POOLED)
    add("icc", "share_bootstrap_negative", (boot < 0).mean(), outcome, POOLED)
    add("icc", "design_effect", deff, outcome, POOLED, note="1 + (n0 - 1) ICC")
    add("icc", "n0_anova_mean_cluster_size", n0, outcome, POOLED)
    add("icc", "mean_cluster_size", N / k, outcome, POOLED)
    add("icc", "n_institutions", k, outcome, POOLED)
    add("icc", "n_singleton_institutions", singles, outcome, POOLED)
    print(f"  {outcome:15s} ICC {icc:.4f} (95% CI {lo:.4f} to {hi:.4f}); design effect {deff:.3f} at n0 = {n0:.2f} "
          f"(mean cluster size {N / k:.2f}); {k} institutions, {singles} with one respondent")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Fractional logit vs OLS: Eq. S1 and Fig. 5
# ═══════════════════════════════════════════════════════════════════════════════
header("5. Fractional logit (binomial GLM, logit link, clustered SE): average marginal effects vs OLS")
glm_warnings = []


def glm(formula, d, label):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = smf.glm(formula, d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d[CL]})
    for w in caught:
        if "integer" not in str(w.message).lower():   # the non-integer-outcome notice is expected
            glm_warnings.append(f"{label}: {w.category.__name__}: {w.message}")
    if not m.converged:
        glm_warnings.append(f"{label}: not converged")
    return m


def design(m, nd):
    """Design rows of fitted model `m` for new data `nd`."""
    return np.asarray(build_design_matrices([m.model.data.design_info], nd)[0])


def delta(est, grad, V):
    """Estimate with delta-method SE from covariance V, normal-based 95% CI and p."""
    se = float(np.sqrt(grad @ V @ grad))
    z = est / se
    return {"estimate": est, "se": se, "ci_lower": est - stats.norm.ppf(0.975) * se,
            "ci_upper": est + stats.norm.ppf(0.975) * se, "p": 2 * stats.norm.sf(abs(z))}


def glm_mean(m, nd):
    """Average predicted probability over the rows of nd, and its gradient in the coefficients."""
    X = design(m, nd)
    mu = expit(X @ m.params.to_numpy())
    return mu.mean(), ((mu * (1 - mu))[:, None] * X).mean(axis=0)


def glm_slope(m, nd, var):
    """Average marginal effect of continuous `var`: mean g'(eta) (D beta), D = X(var + 1) - X(var)."""
    X = design(m, nd)
    D = design(m, nd.assign(**{var: nd[var] + 1.0})) - X
    b = m.params.to_numpy()
    mu = expit(X @ b)
    Db = D @ b
    g1, g2 = mu * (1 - mu), mu * (1 - mu) * (1 - 2 * mu)
    return float((g1 * Db).mean()), ((g2 * Db)[:, None] * X + g1[:, None] * D).mean(axis=0)


def glm_binary(m, nd, var):
    """Average marginal effect of a 0/1 variable: mean mu(var = 1) - mean mu(var = 0)."""
    m1, g1 = glm_mean(m, nd.assign(**{var: 1.0}))
    m0, g0 = glm_mean(m, nd.assign(**{var: 0.0}))
    return m1 - m0, g1 - g0


def ols_contrast(m, L):
    """OLS linear combination L'beta with clustered t inference."""
    t = m.t_test(L.reshape(1, -1))
    lo, hi = np.asarray(t.conf_int(alpha=0.05)).ravel()
    return {"estimate": float(np.ravel(t.effect)[0]), "se": float(np.ravel(t.sd)[0]),
            "ci_lower": lo, "ci_upper": hi, "p": float(np.ravel(t.pvalue)[0])}


def fitted_range(fitted):
    f = np.asarray(fitted)
    return {"min": f.min(), "max": f.max(), "share_below_0": (f < 0).mean(), "share_above_1": (f > 1).mean(), "n": len(f)}


# ── 5a. Eq. S1 ────────────────────────────────────────────────────────────────
eq_rows, eq_fitted = [], {}
pooled_pref = None
for outcome, level, data, fe in eqs1_levels(df):
    d = estimation_frame(data, outcome)
    formula = common.eqs1_formula(outcome, field_fe=fe)
    mo = common.fit_clustered(formula, d)
    mg = glm(formula, d, f"Eq. S1 {outcome} {level}")
    V = mg.cov_params().to_numpy()
    eq_fitted[(outcome, level)] = fitted_range(mo.fittedvalues)
    for term, label in TERMS:
        o = term_stats(mo, term)
        est, grad = glm_slope(mg, d, term) if term == P else glm_binary(mg, d, term)
        a = delta(est, grad, V)
        eq_rows.append({"outcome": outcome, "level": level, "term": label,
                        **{f"ols_{k}": v for k, v in o.items()}, **{f"ame_{k}": v for k, v in a.items()},
                        "ratio": a["estimate"] / o["estimate"],
                        "sign_change": np.sign(a["estimate"]) != np.sign(o["estimate"]),
                        "sig_change": (a["p"] < 0.05) != (o["p"] < 0.05)})
        for k in ["estimate", "p"]:
            add("fractional_logit", f"ols_{k}", o[k], outcome, level, label)
        for k, v in a.items():
            add("fractional_logit", f"ame_{k}", v, outcome, level, label)
        add("fractional_logit", "ame_over_ols", a["estimate"] / o["estimate"], outcome, level, label)
    if (outcome, level) == ("top_preference", POOLED):
        pooled_pref = (formula, d, mg)
eq = pd.DataFrame(eq_rows)

# check of the delta-method code against statsmodels' get_margeff on the pooled preference model
formula, d, mg = pooled_pref
lg = smf.logit(formula, d).fit(cov_type="cluster", cov_kwds={"groups": d[CL]}, disp=0)
mfx = lg.get_margeff(at="overall", dummy=True).summary_frame()
mine = eq[(eq["outcome"] == "top_preference") & (eq["level"] == POOLED)].set_index("term")
for term, label in TERMS:
    assert np.isclose(mine.loc[label, "ame_estimate"], mfx.loc[term, "dy/dx"], atol=1e-8)
    assert np.isclose(mine.loc[label, "ame_se"], mfx.loc[term, "Std. Err."], atol=1e-8)
print("  AME code agrees with statsmodels get_margeff on the pooled preference model (atol 1e-8)")

for label in ["Prestige", "Gender"]:
    s = eq[eq["term"] == label]
    changes = s[s["sign_change"] | s["sig_change"]]
    note = "; ".join(f"{r.outcome} {r.level}: OLS p {r.ols_p:.3f}, AME p {r.ame_p:.3f}" for r in changes.itertuples())
    add("fractional_logit", "eqs1_ame_over_ols_median", s["ratio"].median(), term=label, note=f"{len(s)} coefficients")
    add("fractional_logit", "eqs1_ame_over_ols_min", s["ratio"].min(), term=label)
    add("fractional_logit", "eqs1_ame_over_ols_max", s["ratio"].max(), term=label)
    add("fractional_logit", "eqs1_n_sign_or_significance_changes", len(changes), term=label, note=note)
    print(f"  Eq. S1 {label:9s} AME/OLS: median {s['ratio'].median():.3f}, range {s['ratio'].min():.3f}-{s['ratio'].max():.3f} "
          f"({len(s)} coefficients); sign or significance changes: {len(changes)}" + (f"  [{note}]" if note else ""))
pg = eq[eq["term"].isin(["Prestige", "Gender"])]
add("fractional_logit", "eqs1_n_prestige_gender_coefficients", len(pg))
add("fractional_logit", "eqs1_n_prestige_gender_sign_or_significance_changes", (pg["sign_change"] | pg["sig_change"]).sum())
career = eq[eq["term"].isin(["Associate", "Full"])]
cc = career[career["sign_change"] | career["sig_change"]]
add("fractional_logit", "eqs1_n_career_stage_sign_or_significance_changes", len(cc),
    note="; ".join(f"{r.outcome} {r.level} {r.term}: OLS {r.ols_estimate:+.4f} p {r.ols_p:.3f}, AME {r.ame_estimate:+.4f} p {r.ame_p:.3f}"
                   for r in cc.itertuples()))
print("  Pooled (field FE) OLS vs AME:")
for r in eq[eq["level"] == POOLED].itertuples():
    print(f"      {r.outcome:15s} {r.term:9s} OLS {r.ols_estimate:+.4f} ({r.ols_ci_lower:+.4f}, {r.ols_ci_upper:+.4f}) p {r.ols_p:.1e}"
          f"   AME {r.ame_estimate:+.4f} ({r.ame_ci_lower:+.4f}, {r.ame_ci_upper:+.4f}) p {r.ame_p:.1e}")

# ── 5b. Fig. 5 ────────────────────────────────────────────────────────────────
long = common.fig5_long()
share_zero = (long.loc[long["ref"] == "field", "y"] == 0).mean()
add("fractional_logit", "fig5_share_field_reference_outcomes_zero", share_zero, note="stacked Fig. 5 data, field top-5 rows")
f5_rows, f5_fitted = [], {}
lines = None
for level in [POOLED] + common.FIELDS:
    d = long if level == POOLED else long[long["field"] == level]
    formula = "y ~ C(ref) * (prestige + C(field))" if level == POOLED else "y ~ C(ref) * prestige"
    mo = common.fit_clustered(formula, d)
    mg = glm(formula, d, f"Fig. 5 {level}")
    V = mg.cov_params().to_numpy()
    f5_fitted[level] = fitted_range(mo.fittedvalues)
    resp = d[d["ref"] == "own"][["field", "prestige", CL]]     # one row per respondent
    L, q = {}, {}
    for ref in ("own", "field"):
        nd = resp.assign(ref=ref)
        L[("slope", ref)] = design(mo, nd.assign(prestige=1.0)).mean(0) - design(mo, nd.assign(prestige=0.0)).mean(0)
        L[("pred10", ref)] = design(mo, nd.assign(prestige=10.0)).mean(0)
        q[("slope", ref)] = glm_slope(mg, nd, "prestige")
        q[("pred10", ref)] = glm_mean(mg, nd.assign(prestige=10.0))
    L[("diff10", "own-field")] = L[("pred10", "own")] - L[("pred10", "field")]
    q[("diff10", "own-field")] = (q[("pred10", "own")][0] - q[("pred10", "field")][0],
                                  q[("pred10", "own")][1] - q[("pred10", "field")][1])
    for key in L:
        o, a = ols_contrast(mo, L[key]), delta(*q[key], V)
        f5_rows.append({"level": level, "quantity": key[0], "ref": key[1], "n_respondents": len(resp),
                        **{f"ols_{k}": v for k, v in o.items()}, **{f"fl_{k}": v for k, v in a.items()}})
    if level == POOLED:   # Fig. 5A lines: average OLS predictions at each decile
        lines = np.array([design(mo, resp.assign(ref=ref, prestige=float(p))).mean(0) @ mo.params.to_numpy()
                          for ref in ("own", "field") for p in range(1, 11)])
f5 = pd.DataFrame(f5_rows)
fam = (f5["quantity"] == "diff10") & (f5["level"] != POOLED)   # the 13 per-field differences, BH at alpha = 0.01
f5["ols_p_bh"], f5["fl_p_bh"] = np.nan, np.nan
f5.loc[fam, "ols_p_bh"] = multipletests(f5.loc[fam, "ols_p"], alpha=0.01, method="fdr_bh")[1]
f5.loc[fam, "fl_p_bh"] = multipletests(f5.loc[fam, "fl_p"], alpha=0.01, method="fdr_bh")[1]
f5["sig_ols"] = np.where(fam, f5["ols_p_bh"] < 0.01, f5["ols_p"] < 0.05)
f5["sig_fl"] = np.where(fam, f5["fl_p_bh"] < 0.01, f5["fl_p"] < 0.05)
f5["sign_change"] = np.sign(f5["fl_estimate"]) != np.sign(f5["ols_estimate"])
for r in f5.itertuples():
    term = f"{r.quantity} ({r.ref})"
    for k in ["estimate", "ci_lower", "ci_upper", "p"]:
        add("fractional_logit", f"fig5_ols_{k}", getattr(r, f"ols_{k}"), level=r.level, term=term)
    for k in ["estimate", "se", "ci_lower", "ci_upper", "p"]:
        add("fractional_logit", f"fig5_fl_{k}", getattr(r, f"fl_{k}"), level=r.level, term=term)
    if r.quantity == "diff10" and r.level != POOLED:
        add("fractional_logit", "fig5_ols_p_bh", r.ols_p_bh, level=r.level, term=term)
        add("fractional_logit", "fig5_fl_p_bh", r.fl_p_bh, level=r.level, term=term)

bh = f5[fam]
bh_changes = bh[bh["sig_ols"] != bh["sig_fl"]]
add("fractional_logit", "fig5_n_bh_significance_changes", len(bh_changes),
    note="; ".join(f"{r.level}: OLS BH p {r.ols_p_bh:.3f}, FL BH p {r.fl_p_bh:.3f}" for r in bh_changes.itertuples()))
add("fractional_logit", "fig5_n_sign_changes", f5["sign_change"].sum())
slopes = f5[f5["quantity"] == "slope"]
sc = slopes[slopes["sig_ols"] != slopes["sig_fl"]]
add("fractional_logit", "fig5_n_slope_significance_changes", len(sc),
    note="; ".join(f"{r.level} {r.ref}: OLS p {r.ols_p:.3f}, FL p {r.fl_p:.3f}" for r in sc.itertuples()))
for r in sc.itertuples():
    add("fractional_logit", "fig5_slope_significance_change_ols_p", r.ols_p, level=r.level, term=f"slope ({r.ref})")
    add("fractional_logit", "fig5_slope_significance_change_fl_p", r.fl_p, level=r.level, term=f"slope ({r.ref})")
for qty, lab in [("slope", "fig5_slope_fl_over_ols"), ("diff10", "fig5_diff10_fl_over_ols")]:
    s = f5[f5["quantity"] == qty]
    ratio = s["fl_estimate"] / s["ols_estimate"]
    add("fractional_logit", f"{lab}_median", ratio.median(), note=f"{len(s)} contrasts")
    add("fractional_logit", f"{lab}_min", ratio.min())
    add("fractional_logit", f"{lab}_max", ratio.max())
pooled_diff = f5[(f5["level"] == POOLED) & (f5["quantity"] == "diff10")].iloc[0]
for k in ["estimate", "ci_lower", "ci_upper"]:
    add("fractional_logit", f"fig5_pooled_own_minus_field_at_decile10_venues_fl_{k}", 5 * pooled_diff[f"fl_{k}"],
        level=POOLED, note="fraction of five venues x 5")
    add("fractional_logit", f"fig5_pooled_own_minus_field_at_decile10_venues_ols_{k}", 5 * pooled_diff[f"ols_{k}"],
        level=POOLED, note="fraction of five venues x 5")
print(f"  Fig. 5: {share_zero * 100:.0f}% of field-reference outcomes are exactly 0")
print(f"  Fig. 5B own - field differences (13 fields, BH at 0.01): {len(bh_changes)} significance change(s), "
      f"{f5['sign_change'].sum()} sign change(s) among all contrasts")
print(f"  Fig. 5 prestige slopes with a significance change at 0.05: {len(sc)}"
      + "".join(f"\n      {r.level} ({r.ref}): OLS {r.ols_estimate:+.4f} p {r.ols_p:.3f}; FL {r.fl_estimate:+.4f} p {r.fl_p:.3f}"
                for r in sc.itertuples()))
print(f"  Pooled own - field at decile 10: OLS {5 * pooled_diff['ols_estimate']:.2f} venues "
      f"({5 * pooled_diff['ols_ci_lower']:.2f}, {5 * pooled_diff['ols_ci_upper']:.2f}); "
      f"FL {5 * pooled_diff['fl_estimate']:.2f} venues ({5 * pooled_diff['fl_ci_lower']:.2f}, {5 * pooled_diff['fl_ci_upper']:.2f})")
print(f"  GLM warnings / non-convergence: {glm_warnings or 'none'}")
add("fractional_logit", "n_glm_warnings", len(glm_warnings), note="; ".join(glm_warnings))

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Fitted-value ranges (OLS)
# ═══════════════════════════════════════════════════════════════════════════════
header("6. OLS fitted values and plotted predictions")
eqf = pd.DataFrame(eq_fitted).T
lo_model, hi_model = eqf["min"].idxmin(), eqf["max"].idxmax()
add("fitted_values", "eqs1_fitted_min", eqf["min"].min(), note=f"{lo_model[0]} {lo_model[1]}")
add("fitted_values", "eqs1_fitted_max", eqf["max"].max(), note=f"{hi_model[0]} {hi_model[1]}")
add("fitted_values", "eqs1_n_models_with_fitted_outside_unit_interval",
    ((eqf["share_below_0"] > 0) | (eqf["share_above_1"] > 0)).sum(), note=f"{len(eqf)} models")
for (outcome, level), r in eq_fitted.items():
    add("fitted_values", "fitted_min", r["min"], outcome, level)
    add("fitted_values", "fitted_max", r["max"], outcome, level)
print(f"  Eq. S1 ({len(eqf)} models): fitted values in [{eqf['min'].min():.3f}, {eqf['max'].max():.3f}] "
      f"({lo_model[1]} {lo_model[0]}; {hi_model[1]} {hi_model[0]}); "
      f"{((eqf['share_below_0'] > 0) | (eqf['share_above_1'] > 0)).sum()} model(s) with fitted values outside [0, 1]")
f5f = pd.DataFrame(f5_fitted).T
below = f5f[f5f["share_below_0"] > 0]
for level, r in f5_fitted.items():
    add("fitted_values", "fitted_min", r["min"], "fig5", level)
    add("fitted_values", "fitted_max", r["max"], "fig5", level)
    add("fitted_values", "fitted_share_below_0", r["share_below_0"], "fig5", level)
add("fitted_values", "fig5_n_models_with_fitted_below_0", len(below),
    note="; ".join(f"{lvl}: {r['share_below_0'] * 100:.1f}% below 0, min {r['min']:.3f}" for lvl, r in below.iterrows()))
add("fitted_values", "fig5_n_models_with_fitted_above_1", (f5f["share_above_1"] > 0).sum())
print(f"  Fig. 5 ({len(f5f)} models): pooled fitted values in [{f5f.loc[POOLED, 'min']:.3f}, {f5f.loc[POOLED, 'max']:.3f}], "
      f"{f5f.loc[POOLED, 'share_below_0'] * 100:.1f}% below 0; models with any fitted value below 0: "
      + ", ".join(f"{lvl} ({r['share_below_0'] * 100:.1f}%, min {r['min']:.3f})" for lvl, r in below.iterrows())
      + f"; above 1: {(f5f['share_above_1'] > 0).sum()}")
points = f5[(f5["quantity"] == "pred10") & (f5["level"] != POOLED)]
plotted = {"fig5a_lines": (lines.min(), lines.max()),
           "fig5b_points": (points["ols_estimate"].min(), points["ols_estimate"].max()),
           "fig5b_ci_bounds": (points["ols_ci_lower"].min(), points["ols_ci_upper"].max())}
for name, (lo, hi) in plotted.items():
    add("fitted_values", f"{name}_min", lo)
    add("fitted_values", f"{name}_max", hi)
all_lo, all_hi = min(v[0] for v in plotted.values()), max(v[1] for v in plotted.values())
add("fitted_values", "fig5_plotted_min", all_lo, note="Fig. 5A lines, Fig. 5B points and 95% CI bounds")
add("fitted_values", "fig5_plotted_max", all_hi)
print("  Plotted in Fig. 5: " + "; ".join(f"{k} [{v[0]:.3f}, {v[1]:.3f}]" for k, v in plotted.items())
      + f"; all [{all_lo:.3f}, {all_hi:.3f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Self-consistency of the pairwise comparisons
# ═══════════════════════════════════════════════════════════════════════════════
header("7. Self-consistency: strict comparisons that disagree with the respondent's own inferred ranking")
comp = pd.read_csv(common.PUBLIC / "comparisons.csv")
rank = pd.read_csv(common.DERIVED / "individual_rankings.csv").set_index(["user_db_id", "venue_db_id"])["normed_rank"]
c = comp[comp["is_tie"] == False].copy()   # noqa: E712
c["rank_pref"] = rank.reindex(pd.MultiIndex.from_arrays([c["user_db_id"], c["pref_venue_db_id"]])).to_numpy()
c["rank_other"] = rank.reindex(pd.MultiIndex.from_arrays([c["user_db_id"], c["other_venue_db_id"]])).to_numpy()
c = c.dropna(subset=["rank_pref", "rank_other"])
c["consistent"] = c["rank_pref"] > c["rank_other"]
pct_upset = 100 * (1 - c["consistent"].mean())
per_user = c.groupby("user_db_id")["consistent"].agg(["sum", "size"])
pct_without = 100 * (per_user["sum"] == per_user["size"]).mean()
by_field = 100 * (1 - c.groupby("field")["consistent"].mean())
add("self_consistency", "pct_comparisons_inconsistent", pct_upset, note=f"{len(c)} strict comparisons, {len(per_user)} respondents")
add("self_consistency", "pct_respondents_fully_consistent", pct_without)
add("self_consistency", "pct_comparisons_inconsistent_field_min", by_field.min(), level=by_field.idxmin())
add("self_consistency", "pct_comparisons_inconsistent_field_max", by_field.max(), level=by_field.idxmax())
for f, v in by_field.items():
    add("self_consistency", "pct_comparisons_inconsistent", v, level=f)
print(f"  {pct_upset:.1f}% of {len(c)} strict comparisons are inconsistent with the respondent's own ranking "
      f"(range {by_field.min():.1f}% in {by_field.idxmin()} to {by_field.max():.1f}% in {by_field.idxmax()}); "
      f"{pct_without:.1f}% of {len(per_user)} respondents have no inconsistency")

out = pd.DataFrame(rows)
out.to_csv(common.RESULTS / "inference_checks.csv", index=False)
print(f"\nSaved: {common.RESULTS / 'inference_checks.csv'} ({len(out)} rows)")
