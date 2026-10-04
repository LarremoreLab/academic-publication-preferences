"""SI section "Robustness to Alternative Measures of Institutional Prestige".

The main text measures institutional prestige with faculty hiring networks (prestige_decile,
10 = most prestigious). This script reproduces the two checks of that choice reported in the SI.

1. Carnegie Classification 2021 (public_data/respondents.csv, column carnegie_2021: R1, R2 or
   Other, where Other collects doctoral/professional universities, special-focus institutions
   such as medical schools and research institutes, and all remaining classes). How much does the
   Carnegie tier vary in the Eq. S1 and Fig. 5 samples, and how much of the hiring-network
   prestige variation does it capture? Reported: respondents and institutions per tier, the tier
   shares by field, the tier-by-decile cross-tabulation, the Spearman correlation between tier
   (R1 = 3, R2 = 2, Other = 1) and prestige decile, the share of prestige-decile variance lying
   within tiers (1 - eta^2 from a one-way decomposition), and the standard deviation and range of
   prestige deciles within R1 vs in the whole sample.

2. Times Higher Education World University Rankings 2024 (column the_2024_decile: THE-ranked
   institutions ordered by rank and cut into ten groups of equal 2023 faculty headcount, 10 =
   highest; NaN for institutions THE does not rank). Reported: coverage of the two samples, where
   the unranked respondents sit on the hiring-network scale, the Spearman correlation between the
   two decile measures among covered respondents, and the paper's prestige models refit three
   ways, so that the effect of the measure can be separated from the effect of the restricted
   sample:
     hiring (full sample)    hiring-network deciles, all respondents (the main text)
     hiring (THE-covered)    hiring-network deciles, respondents at THE-ranked institutions only
     THE (THE-covered)       THE deciles, the same respondents
   Models (OLS, standard errors clustered by institution, t reference distribution):
     Eq. S1, pooled    y ~ associate + full + prestige + gender + C(field), both outcomes
     Eq. S1, by field  y ~ associate + full + prestige + gender
     Fig. 4A contrast  decile 10 vs decile 1: y ~ 1[decile = 10] on respondents in those deciles
     Fig. 5            the per-field and pooled models of fig5.py (fig5.fit_models), including
                       the Benjamini-Hochberg adjustment of the 13 per-field own-minus-field
                       contrasts at decile 10 (alpha = 0.01)

Inputs
  public_data/respondents.csv (carnegie_2021, the_2024_decile, institution_id), through
  common.eqs1_sample() (Eq. S1 sample, N = 2,196) and common.fig5_long() (Fig. 5 sample,
  N = 1,872; the two columns are institution attributes and are attached by institution_id).
  reproducibility/derived/ (run compute_rankings.py first).

Output
  output/results/prestige_measures.csv   one row per quantity. Columns: section (carnegie,
                                         carnegie_by_field, carnegie_by_decile, the_coverage,
                                         eqs1, fig4_contrast, fig5), sample (Eq. S1 / Fig. 5),
                                         measure (one of the three variants above), level
                                         (Academia or a field), outcome (top_preference /
                                         top_aspiration), group (Carnegie tier, model term, or
                                         Fig. 5 reference group), decile, quantity, value, and
                                         se, ci_lower, ci_upper, p, p_bh, n, n_inst where they
                                         apply. Fig. 5 estimates are fractions of five venues;
                                         the rows quantity = diff10_venues multiply by five.

Quantities in the SI that depend on institution names or national totals (the number of exact
and hand-resolved name matches, the 146 R1 and 133 R2 universities nationally, and the number of
distinct IPEDS identifiers, which differs from the number of distinct institution_id values when
a branch campus reports under its parent institution) cannot be computed from the anonymised
public data and are not reproduced here.

Usage (from the repository root): .venv/bin/python reproducibility/si_prestige_measures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import fig5  # noqa: E402  (model code shared with Fig. 5; importing it draws nothing)

CARNEGIE, THE = "carnegie_2021", "the_2024_decile"
TIERS = ["R1", "R2", "Other"]
TIER_RANK = {"R1": 3, "R2": 2, "Other": 1}
HIRING_FULL, HIRING_THE, THE_THE = "hiring (full sample)", "hiring (THE-covered)", "THE (THE-covered)"
POOLED = "Academia"
CL = common.CLUSTER
COLUMNS = ["section", "sample", "measure", "level", "outcome", "group", "decile", "quantity",
           "value", "se", "ci_lower", "ci_upper", "p", "p_bh", "n", "n_inst"]

rows = []


def add(section, quantity, value, **kw):
    rows.append({"section": section, "quantity": quantity, "value": value, **kw})


def coef(m, term):
    lo, hi = m.conf_int().loc[term]
    return {"se": m.bse[term], "ci_lower": lo, "ci_upper": hi, "p": m.pvalues[term], "n": int(m.nobs)}


# ── data ──────────────────────────────────────────────────────────────────────

def institution_attributes():
    """institution_id -> Carnegie tier and THE decile. Both are institution attributes, so they
    are constant across the respondents of an institution."""
    r = common.load_respondents().dropna(subset=[CL])
    g = r.groupby(CL)[[CARNEGIE, THE]]
    assert (g.nunique(dropna=False) <= 1).all().all()
    return g.first()


def load_samples():
    attrs = institution_attributes()
    eq = common.eqs1_sample()
    assert eq[CARNEGIE].isin(TIERS).all()
    assert (eq[CL].map(attrs[CARNEGIE]) == eq[CARNEGIE]).all()
    assert eq[CL].map(attrs[THE]).equals(eq[THE])

    long = common.fig5_long()
    long[CARNEGIE] = long[CL].map(attrs[CARNEGIE])
    long[THE] = long[CL].map(attrs[THE])
    assert long[CARNEGIE].isin(TIERS).all()
    f5 = long[long["ref"] == "own"].rename(columns={"prestige": common.PRESTIGE})  # one row per respondent
    return eq, f5, long


# ── Carnegie ──────────────────────────────────────────────────────────────────

def carnegie(sample, d):
    dec, tier = d[common.PRESTIGE], d[CARNEGIE]
    for g in TIERS:
        x = d[tier == g]
        add("carnegie", "respondents", len(x), sample=sample, group=g)
        add("carnegie", "share_respondents", len(x) / len(d), sample=sample, group=g)
        add("carnegie", "institutions", x[CL].nunique(), sample=sample, group=g)
        add("carnegie", "mean_prestige_decile", x[common.PRESTIGE].mean(), sample=sample, group=g, n=len(x))
    add("carnegie", "spearman_tier_vs_prestige_decile", spearmanr(tier.map(TIER_RANK), dec).correlation,
        sample=sample, n=len(d))
    # one-way decomposition of the prestige-decile variance into between- and within-tier parts
    grand = dec.mean()
    ss_between = sum(len(dec[tier == g]) * (dec[tier == g].mean() - grand) ** 2 for g in TIERS)
    eta_sq = ss_between / ((dec - grand) ** 2).sum()
    add("carnegie", "eta_squared_prestige_decile_on_tier", eta_sq, sample=sample, n=len(d))
    add("carnegie", "share_prestige_decile_variance_within_tiers", 1 - eta_sq, sample=sample, n=len(d))
    add("carnegie", "sd_prestige_decile", dec.std(ddof=1), sample=sample, group="all", n=len(d))
    r1 = dec[tier == "R1"]
    add("carnegie", "sd_prestige_decile", r1.std(ddof=1), sample=sample, group="R1", n=len(r1))
    add("carnegie", "min_prestige_decile", r1.min(), sample=sample, group="R1")
    add("carnegie", "max_prestige_decile", r1.max(), sample=sample, group="R1")
    for f in common.FIELDS:
        x = d[d["field"] == f]
        for g in TIERS:
            add("carnegie_by_field", "share_respondents", (x[CARNEGIE] == g).mean(), sample=sample, level=f,
                group=g, n=len(x))
    for k in range(1, 11):
        for g in TIERS:
            add("carnegie_by_decile", "respondents", int(((dec == k) & (tier == g)).sum()), sample=sample,
                group=g, decile=k)


# ── THE: coverage ─────────────────────────────────────────────────────────────

def the_coverage(sample, d):
    cov = d[THE].notna()
    add("the_coverage", "respondents", len(d), sample=sample, n_inst=d[CL].nunique())
    add("the_coverage", "covered_respondents", int(cov.sum()), sample=sample, n_inst=d.loc[cov, CL].nunique())
    add("the_coverage", "covered_share", cov.mean(), sample=sample)
    unr = d[~cov]
    add("the_coverage", "unranked_respondents", len(unr), sample=sample, n_inst=unr[CL].nunique())
    add("the_coverage", "unranked_mean_hiring_decile", unr[common.PRESTIGE].mean(), sample=sample, n=len(unr))
    low = unr[unr[common.PRESTIGE] <= 5]
    add("the_coverage", "unranked_in_hiring_deciles_1_to_5", len(low), sample=sample, n_inst=low[CL].nunique())
    add("the_coverage", "spearman_the_vs_hiring_decile", spearmanr(d.loc[cov, THE], d.loc[cov, common.PRESTIGE]).correlation,
        sample=sample, n=int(cov.sum()))


# ── THE: Eq. S1 and the Fig. 4A contrast ──────────────────────────────────────

def eqs1_variants(eq):
    the_s = eq[eq[THE].notna()]
    return [(HIRING_FULL, eq, common.PRESTIGE), (HIRING_THE, the_s, common.PRESTIGE), (THE_THE, the_s, THE)]


def eqs1_models(eq):
    for outcome, y in common.OUTCOME_COLS.items():
        for measure, d0, pv in eqs1_variants(eq):
            base = f"{y} ~ associate_prof_dummy + full_prof_dummy + {pv} + gender"
            d = d0.dropna(subset=[y, pv, "gender", CL])
            levels = [(POOLED, d, base + " + C(field)")] + [(f, d[d["field"] == f], base) for f in common.FIELDS]
            for level, dd, formula in levels:
                m = common.fit_clustered(formula, dd)
                for term, label in [(pv, "prestige"), ("gender", "gender")]:
                    add("eqs1", "coefficient", m.params[term], sample="Eq. S1", measure=measure, level=level,
                        outcome=outcome, group=label, n_inst=dd[CL].nunique(), **coef(m, term))
                # association per standard deviation of the prestige measure in the fitted sample
                add("eqs1", "coefficient_per_sd", m.params[pv] * dd[pv].std(ddof=1), sample="Eq. S1",
                    measure=measure, level=level, outcome=outcome, group="prestige", n=int(m.nobs))
            # Fig. 4A: respondents in the top vs the bottom decile of the measure
            ext = d[d[pv].isin([1, 10])].assign(top=lambda x: (x[pv] == 10).astype(float))
            m = common.fit_clustered(f"{y} ~ top", ext)
            for k, flag in [(10, 1.0), (1, 0.0)]:
                x = ext[ext["top"] == flag]
                add("fig4_contrast", "mean_outcome", x[y].mean(), sample="Eq. S1", measure=measure, outcome=outcome,
                    decile=k, n=len(x), n_inst=x[CL].nunique())
            add("fig4_contrast", "difference_decile_10_vs_1", m.params["top"], sample="Eq. S1", measure=measure,
                outcome=outcome, n_inst=ext[CL].nunique(), **coef(m, "top"))


# ── THE: Fig. 5 ───────────────────────────────────────────────────────────────

def fig5_variants(long):
    cov = long[THE].notna()
    return [(HIRING_FULL, long), (HIRING_THE, long[cov]), (THE_THE, long[cov].assign(prestige=long.loc[cov, THE]))]


def fig5_models(long):
    results = {}
    for measure, d in fig5_variants(long):
        res, _ = fig5.fit_models(d.reset_index(drop=True))
        results[measure] = res
        for r in res.itertuples():
            add("fig5", r.quantity, r.estimate, sample="Fig. 5", measure=measure, level=r.level, group=r.ref,
                se=r.se, ci_lower=r.ci_lower, ci_upper=r.ci_upper, p=r.p, p_bh=r.p_bh, n=r.n_respondents,
                n_inst=r.n_clusters)
        d10 = res[(res["level"] == POOLED) & (res["quantity"] == "diff10")].iloc[0]
        add("fig5", "diff10_venues", 5 * d10.estimate, sample="Fig. 5", measure=measure, level=POOLED, group="own-field",
            se=5 * d10.se, ci_lower=5 * d10.ci_lower, ci_upper=5 * d10.ci_upper, p=d10.p, n=d10.n_respondents,
            n_inst=d10.n_clusters)
    # the full-sample variant is the paper's Fig. 5 model; check against fig5.py's output when available
    ref = common.RESULTS / "fig5_models.csv"
    if ref.exists():
        a, b = results[HIRING_FULL], pd.read_csv(ref)
        key = ["level", "quantity", "ref"]
        merged = a.merge(b, on=key, suffixes=("", "_fig5"))
        assert len(merged) == len(a) and np.allclose(merged["estimate"], merged["estimate_fig5"], atol=1e-12)
        print(f"Check: full-sample Fig. 5 estimates equal {ref.name}.")
    return results


# ── summary ───────────────────────────────────────────────────────────────────

def ci(r, d=4):
    return f"{r.value:.{d}f} ({r.ci_lower:.{d}f}, {r.ci_upper:.{d}f}), p = {r.p:.2g}"


def summarize(out, fig5_results):
    pd.set_option("display.width", 200)
    print("\n=== Carnegie Classification 2021 ===")
    c = out[out.section == "carnegie"]
    for sample in ["Eq. S1", "Fig. 5"]:
        s = c[c["sample"] == sample]
        t = s[s.group.isin(TIERS) & s.quantity.isin(["respondents", "share_respondents", "institutions", "mean_prestige_decile"])]
        print(f"\n{sample} (N = {int(t[t.quantity == 'respondents'].value.sum())})")
        print(t.pivot(index="group", columns="quantity", values="value").reindex(TIERS)
              .rename(columns={"share_respondents": "share"})[["respondents", "share", "institutions", "mean_prestige_decile"]]
              .round(3).to_string())
        one = s.set_index(["quantity", "group"])["value"]
        print(f"Spearman(tier, prestige decile) = {one[('spearman_tier_vs_prestige_decile', np.nan)]:.2f}; "
              f"eta^2 = {one[('eta_squared_prestige_decile_on_tier', np.nan)]:.2f} "
              f"({100 * one[('share_prestige_decile_variance_within_tiers', np.nan)]:.0f}% of decile variance within tiers); "
              f"SD of decile {one[('sd_prestige_decile', 'R1')]:.2f} within R1 (deciles "
              f"{int(one[('min_prestige_decile', 'R1')])}-{int(one[('max_prestige_decile', 'R1')])}) vs "
              f"{one[('sd_prestige_decile', 'all')]:.2f} overall")
    bf = out[(out.section == "carnegie_by_field") & (out["sample"] == "Eq. S1")]
    bf = bf.pivot(index="level", columns="group", values="value")[TIERS].join(
        bf[bf.group == "R1"].set_index("level")["n"].astype(int)).sort_values("R1")
    print("\nEq. S1 tier shares by field:")
    print(bf.round(3).to_string())

    print("\n=== THE World University Rankings 2024 ===")
    cv = out[out.section == "the_coverage"].set_index(["sample", "quantity"])
    for sample in ["Eq. S1", "Fig. 5"]:
        r = cv.loc[sample]
        print(f"{sample}: THE ranks the institutions of {int(r.loc['covered_respondents', 'value'])} of "
              f"{int(r.loc['respondents', 'value'])} respondents ({100 * r.loc['covered_share', 'value']:.1f}%; "
              f"{int(r.loc['covered_respondents', 'n_inst'])} of {int(r.loc['respondents', 'n_inst'])} institutions). "
              f"Unranked: {int(r.loc['unranked_respondents', 'value'])} respondents at "
              f"{int(r.loc['unranked_respondents', 'n_inst'])} institutions, mean hiring decile "
              f"{r.loc['unranked_mean_hiring_decile', 'value']:.1f}, {int(r.loc['unranked_in_hiring_deciles_1_to_5', 'value'])} "
              f"in deciles 1-5. Spearman(THE, hiring decile) among covered = {r.loc['spearman_the_vs_hiring_decile', 'value']:.2f}.")

    e = out[(out.section == "eqs1") & (out.level == POOLED)]
    print("\nEq. S1 pooled (field fixed effects, clustered SEs): estimate (95% CI), p")
    for outcome in common.OUTCOME_COLS:
        for term in ["prestige", "gender"]:
            print(f"  {outcome:15s} {term:9s}")
            for measure in [HIRING_FULL, HIRING_THE, THE_THE]:
                r = e[(e.outcome == outcome) & (e.group == term) & (e.measure == measure) & (e.quantity == "coefficient")].iloc[0]
                extra = ""
                if term == "prestige":
                    sd = e[(e.outcome == outcome) & (e.measure == measure) & (e.quantity == "coefficient_per_sd")].iloc[0]
                    extra = f"; per SD of the measure {sd.value:.3f}"
                print(f"    {measure:22s} N = {int(r.n):5d}  {ci(r)}{extra}")

    f = out[(out.section == "eqs1") & (out.level != POOLED) & (out.quantity == "coefficient")]
    print("\nEq. S1 by field: fields with p < 0.05 in the expected direction (prestige > 0, gender < 0)")
    for outcome in common.OUTCOME_COLS:
        for term, sign in [("prestige", 1), ("gender", -1)]:
            counts = []
            for measure in [HIRING_FULL, HIRING_THE, THE_THE]:
                x = f[(f.outcome == outcome) & (f.group == term) & (f.measure == measure)]
                counts.append(f"{measure}: {int(((x.p < 0.05) & (np.sign(x.value) == sign)).sum())}")
            print(f"  {outcome:15s} {term:9s} " + "; ".join(counts))
    a = f[f.measure == HIRING_THE].set_index(["outcome", "level", "group"])
    b = f[f.measure == THE_THE].set_index(["outcome", "level", "group"])
    flip = (np.sign(a.value) != np.sign(b.value))
    sig_flip = flip & ((a.p < 0.05) | (b.p < 0.05))
    print(f"  Coefficients whose sign differs between THE and hiring deciles on the THE-covered sample: "
          f"{int(flip.sum())} ({', '.join(f'{o} {l} {g}' for o, l, g in flip[flip].index)}); "
          f"significant (p < 0.05) under either measure: {int(sig_flip.sum())}")

    f4 = out[out.section == "fig4_contrast"]
    print("\nFig. 4A contrast, decile 10 vs decile 1 of the measure: mean (n, institutions)")
    for outcome in common.OUTCOME_COLS:
        print(f"  {outcome}")
        for measure in [HIRING_FULL, HIRING_THE, THE_THE]:
            x = f4[(f4.outcome == outcome) & (f4.measure == measure)].set_index("quantity")
            m10, m1 = x[x.decile == 10].loc["mean_outcome"], x[x.decile == 1].loc["mean_outcome"]
            dif = x.loc["difference_decile_10_vs_1"]
            print(f"    {measure:22s} {m10.value:.2f} ({int(m10.n)}, {int(m10.n_inst)}) vs {m1.value:.2f} "
                  f"({int(m1.n)}, {int(m1.n_inst)}), difference {dif.value:.3f} "
                  f"({dif.ci_lower:.3f}, {dif.ci_upper:.3f}), p = {dif.p:.2g}")

    print("\nFig. 5: pooled own - field difference at decile 10 (venues) and fields not significant after BH")
    for measure in [HIRING_FULL, HIRING_THE, THE_THE]:
        res = fig5_results[measure]
        d10 = out[(out.section == "fig5") & (out.measure == measure) & (out.quantity == "diff10_venues")].iloc[0]
        fam = res[(res.quantity == "diff10") & (res.level != POOLED)]
        ns = sorted(fam.loc[~fam.sig_bh_01.astype(bool), "level"])
        slopes = res[(res.quantity == "slope") & (res.level == POOLED)].set_index("ref")["estimate"]
        print(f"  {measure:22s} N = {int(d10.n):4d}: {d10.value:.2f} venues ({d10.ci_lower:.2f}, {d10.ci_upper:.2f}), "
              f"p = {d10.p:.2g}; pooled slopes own {slopes['own']:.3f} / field {slopes['field']:.3f}; "
              f"n.s. fields ({len(ns)}): {', '.join(ns)}")


def main():
    eq, f5, long = load_samples()
    for sample, d in [("Eq. S1", eq), ("Fig. 5", f5)]:
        carnegie(sample, d)
        the_coverage(sample, d)
    eqs1_models(eq)
    fig5_results = fig5_models(long)

    out = pd.DataFrame(rows).reindex(columns=COLUMNS)
    path = common.RESULTS / "prestige_measures.csv"
    out.to_csv(path, index=False)
    summarize(out, fig5_results)
    print(f"\nSaved {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
