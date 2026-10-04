#!/usr/bin/env python3
"""Every number quoted in the paper, as LaTeX macros.

Writes reproducibility/output/numbers.tex: one \\newcommand per macro that the paper's LaTeX uses
(main text, supplementary methods, tables and figure captions), grouped by topic. The copy the paper
was compiled with is paper_numbers/numbers.tex; `--check` compares the two macro by macro.

How the numbers flow
    public_data/*.csv  ->  compute_rankings.py          ->  reproducibility/derived/*.csv   (SpringRank)
                       ->  fig*.py, si_*.py             ->  reproducibility/output/results/*.csv
                       ->  bootstrap_consensus.py       ->  reproducibility/cache/*.csv      (committed)
                       ->  this script                  ->  reproducibility/output/numbers.tex

Wherever another script already produces a quantity (a regression coefficient, a bootstrap interval,
a robustness check) this script reads that script's results file instead of refitting the model, so
that every number has a single source. Quantities that no other script produces are computed here
from public_data/ and reproducibility/derived/: the survey sample description, the top-aspiration
breakdown by field, JIF coverage, ranking consistency, the Sociology subfield example, the shares of
respondents who had already published in their aspiration venues, and the sample side of SI Table S11.

Constants
    A small set of values cannot be recomputed from the files in this repository. They are typed in
    the last section of the output ("Values from sources not in this repository"), each with a
    comment naming its source:
      * survey email records: the sample frame, consent, start-page and completion counts per field
        (SI Table S6), and the response-rate analysis by institutional prestige, gender and rank
        (SI Table S7). These records identify individuals and are not shared; the rates derived from
        the counts are computed here.
      * the AARC faculty census (via the scieco package): the mean and standard deviation of the
        respondents' institutional prestige percentiles (SI Table S11), which need the raw
        institution ranks. The population side of SI Table S11 is shipped as aggregates in
        public_data/aarc_field_margins.csv and is read from there.
      * the complete survey database, before restricting to the 13 fields with at least 100
        respondents: total participants, venues and pairwise comparisons.
      * the null-model simulation that randomizes responses while preserving the survey's adaptive
        pairing algorithm (average rescaled rank of the higher-ranked venue in a simulated ranking
        violation), which needs the survey software.
      * institution-name crosswalks to the Carnegie Classification 2021 and the THE World University
        Rankings 2024 (numbers of exact and manual name matches, national totals, R1 coverage). Only
        the resulting per-respondent tier and decile are in public_data/respondents.csv.

Usage (from the repository root, after run_all.py or the individual scripts)
    .venv/bin/python reproducibility/compute_numbers.py
    .venv/bin/python reproducibility/compute_numbers.py --check             # compare with paper_numbers/numbers.tex
    .venv/bin/python reproducibility/compute_numbers.py --write-paper-copy  # also refresh paper_numbers/numbers.tex

Inputs that other scripts must have produced (the script names the producer when a file is missing):
    output/results/  eqs1_coefficients.csv, fig4_group_means.csv     (fig4.py)
                     fig5_models.csv                                 (fig5.py)
                     permutation_test.csv                            (si_fig5_permutation.py)
                     inference_checks.csv                            (si_inference_numbers.py)
                     prestige_measures.csv                           (si_prestige_measures.py)
                     jif_accuracy_summary.csv                        (si_fig6_jif_accuracy.py)
                     alt_metrics_accuracy.csv                        (si_table3_alt_metrics.py)
                     jif_rank_tables.csv                             (si_tables4_5_jif_ranks.py)
                     subfield_consensus.csv, subfield_correlations.csv            (si_table1_subfields.py)
                     subfield_consensus_catchall.csv, subfield_correlations_catchall.csv
                                                     (si_table1_subfields.py --include-catch-alls)
                     subgroup_summary.csv, raked_consensus.csv, raking_diagnostics.csv,
                     field_weighted_models.csv                       (si_tables8_10_composition.py)
    cache/           bootstrap_point_estimates.csv, bootstrap_consensus.csv, bootstrap_correlations.csv,
                     bootstrap_field_order_stability.csv, bootstrap_top5_stability.csv
                                                                     (bootstrap_consensus.py; committed)
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

OUT_FILE = common.OUTPUT / "numbers.tex"
PAPER_COPY = common.ROOT / "paper_numbers" / "numbers.tex"
FIELDS = common.FIELDS
MACRO = {f: "".join(f.title().split()) for f in FIELDS}   # "Computer science" -> "ComputerScience"
SHORT = {f: f.split()[0] for f in FIELDS}                  # "Computer science" -> "Computer" (aspiration macros)
OUTCOMES = [("top_preference", "Pref"), ("top_aspiration", "Asp")]

PRODUCER = {
    "eqs1_coefficients.csv": "fig4.py", "fig4_group_means.csv": "fig4.py", "fig5_models.csv": "fig5.py",
    "permutation_test.csv": "si_fig5_permutation.py", "inference_checks.csv": "si_inference_numbers.py",
    "prestige_measures.csv": "si_prestige_measures.py", "jif_accuracy_summary.csv": "si_fig6_jif_accuracy.py",
    "alt_metrics_accuracy.csv": "si_table3_alt_metrics.py", "jif_rank_tables.csv": "si_tables4_5_jif_ranks.py",
    "subfield_consensus.csv": "si_table1_subfields.py", "subfield_correlations.csv": "si_table1_subfields.py",
    "subfield_consensus_catchall.csv": "si_table1_subfields.py --include-catch-alls",
    "subfield_correlations_catchall.csv": "si_table1_subfields.py --include-catch-alls",
    "subgroup_summary.csv": "si_tables8_10_composition.py", "raked_consensus.csv": "si_tables8_10_composition.py",
    "raking_diagnostics.csv": "si_tables8_10_composition.py", "field_weighted_models.csv": "si_tables8_10_composition.py",
    "bootstrap_point_estimates.csv": "bootstrap_consensus.py", "bootstrap_consensus.csv": "bootstrap_consensus.py",
    "bootstrap_correlations.csv": "bootstrap_consensus.py", "bootstrap_field_order_stability.csv": "bootstrap_consensus.py",
    "bootstrap_top5_stability.csv": "bootstrap_consensus.py",
}


# ── formatting ────────────────────────────────────────────────────────────────────────────────────
def fmt(x, d=1, comma=False):
    """Fixed number of decimals (0.70 keeps its zero); comma=True adds thousands separators."""
    if isinstance(x, str):
        return x
    if comma:
        return f"{x:,.{d}f}" if d else f"{int(round(x)):,d}"
    return f"{x:.{d}f}"


def num(x, d=2):
    """Shortest form of x rounded to d decimals, as Python prints it (24.40 -> 24.4, 80 -> 80.0)."""
    return str(round(float(x), d))


def count(n):
    """Integer with thousands separators."""
    return f"{int(n):,}"


def pfmt(p):
    """p-value for math mode: 0.29, 0.021, 0.0018 or 1.8\\times10^{-6}."""
    if p >= 0.01:
        return f"{p:.3f}".rstrip("0").rstrip(".") if p < 0.1 else f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.4f}"
    e = int(np.floor(np.log10(p)))
    return f"{p / 10 ** e:.1f}\\times10^{{{e}}}"


def p3(p):
    """p-value for the sample-composition tables: three decimals or $<$0.001."""
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def neg(s):
    """Typeset minus signs in numbers: -0.12 -> $-$0.12 (sample-composition macros)."""
    return re.sub(r"(?<![A-Za-z$])-(?=\d)", "$-$", s)


# ── registry ──────────────────────────────────────────────────────────────────────────────────────
ENTRIES = []   # ("section", title) | ("note", text) | ("macro", name, value, comment)
NAMES = set()


def section(title):
    ENTRIES.append(("section", title))


def note(text):
    ENTRIES.append(("note", text))


def add(name, value, comment=None):
    if name in NAMES:
        raise ValueError(f"macro {name} defined twice")
    NAMES.add(name)
    ENTRIES.append(("macro", name, str(value), comment))


def addcr(name, value, comment=None):
    """Sample-composition macros (\\CR...) carry typeset minus signs."""
    add(name, neg(str(value)), comment)


def result(name):
    folder = common.CACHE if name.startswith("bootstrap_") else common.RESULTS
    path = folder / name
    if not path.exists():
        raise SystemExit(f"Missing {path}.\nRun  .venv/bin/python reproducibility/{PRODUCER[name]}  first.")
    return pd.read_csv(path)


# ── public inputs ─────────────────────────────────────────────────────────────────────────────────
common.require_derived()
resp = common.load_respondents()
venues = pd.read_csv(common.PUBLIC / "venues.csv")
venue_name = dict(zip(venues.venue_db_id, venues.name))


def venue_id(name):
    return int(venues.loc[venues.name == name, "venue_db_id"].iloc[0])


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Survey sample")
eq = result("eqs1_coefficients.csv")
eq_all = eq[eq.level == "Academia"].iloc[0]
f5 = result("fig5_models.csv")
f5_all = f5[f5.level == "Academia"].iloc[0]
add("ParticipantsInRelevantFields", count(len(resp)), "respondents in the 13 fields with at least 100 respondents")
add("ParticipantsWithAspirations", count(eq_all.n_obs), "Eq. S1 sample (see common.eqs1_sample)")
add("RevEqSOneN", count(eq_all.n_obs))
add("RevEqSOneClusters", fmt(eq_all.n_clusters, 0), "institutions in the Eq. S1 sample")
add("RevFigFiveN", count(f5_all.n_respondents), "Fig. 5 sample: publication record, prestige decile and institution")
add("RevFigFiveClusters", fmt(f5_all.n_clusters, 0))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Top aspirations by field (SI): respondents unaffected by the aspiration-menu error")
nobug = resp[resp.venue_bug == False]  # noqa: E712
for field in FIELDS:
    g = nobug[nobug.field == field]
    n, F = len(g), SHORT[field]
    add(f"{F}ParticipantsNoBug", n)
    add(f"{F}TopUnique", g.top_init_venue_db_id.nunique())
    add(f"{F}MidUnique", g.mid_init_venue_db_id.nunique())
    add(f"{F}LowUnique", g.low_init_venue_db_id.nunique())
    shares = [(venue_name.get(v, str(v)), round(100 * c / n, 1)) for v, c in g.top_init_venue_db_id.value_counts().items()]
    for (name, pct), letter in zip(sorted(shares, key=lambda x: -x[1])[:5], "ABCDE"):
        add(f"{F}TopName{letter}", name.replace("&", r"\&"))
        add(f"{F}TopPercent{letter}", pct)

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Share of each field's venues with a 2-year JIF (SI)")
fj = pd.read_csv(common.PUBLIC / "field_jif_rankings.csv")
jif_share = fj.groupby("field").jif.count() / fj.groupby("field").venue_db_id.count()
for field in FIELDS:
    add(f"{MACRO[field]}PercentJIF", num(100 * jif_share[field], 2))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Fig. 1A: expected number of distinct venues in 100 consideration sets")
pe = result("bootstrap_point_estimates.csv").set_index("field")
for field in ["Physics", "Economics", "Chemistry", "Business", "Computer science", "Engineering"]:
    add(f"RevAccHundred{MACRO[field]}", fmt(pe.loc[field, "accumulation_at_100"], 0))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Fig. 2: prediction accuracy and top-5 agreement, with respondent-bootstrap intervals (SI Fig. S2)")
for field in ["Computer science", "Economics"]:
    add(f"{MACRO[field]}LOOPredAccuracy", num(pe.loc[field, "prediction_accuracy"], 1))
for field in ["Computer science", "Economics"]:
    add(f"{MACRO[field]}TopFiveAgreement", num(pe.loc[field, "top5_agreement"], 1))
bc = result("bootstrap_correlations.csv")
add("TopFiveLOOPredCorrR", num(bc.point.iloc[0], 2), "Pearson r over the 13 fields")
ci = result("bootstrap_consensus.csv")
for measure, tag in [("prediction_accuracy", "PredAcc"), ("top5_agreement", "TopFive")]:
    for field in ["Computer science", "Economics"]:
        r = ci[(ci.measure == measure) & (ci.field == field)].iloc[0]
        add(f"RevCI{tag}{MACRO[field]}Lo", fmt(r.lo, 1))
        add(f"RevCI{tag}{MACRO[field]}Hi", fmt(r.hi, 1))
add("RevBootReplicates", count(int(re.search(r"(\d+) replicates", ci.method.iloc[0]).group(1))))
add("RevTopFiveLOOPredCorrCILo", fmt(bc.lo.iloc[0], 2))
add("RevTopFiveLOOPredCorrCIHi", fmt(bc.hi.iloc[0], 2))
add("RevTopThreeOverlapCorrCILo", fmt(bc.lo.iloc[1], 2))
add("RevTopThreeOverlapCorrCIHi", fmt(bc.hi.iloc[1], 2))
fo = result("bootstrap_field_order_stability.csv")


def same_order(measure, higher, lower):
    r = fo[(fo.measure == measure) & (fo.higher_in_full_sample == higher) & (fo.lower_in_full_sample == lower)]
    return float(r.share_same_order.iloc[0])


add("RevStabPredAccHistoryOverCS", fmt(100 * same_order("prediction_accuracy", "History", "Computer science"), 0))
add("RevStabPredAccEconOverBiology", fmt(100 * same_order("prediction_accuracy", "Economics", "Biology"), 0))
high_low = [same_order("prediction_accuracy", h, l) for h in ["Economics", "Chemistry", "Biology"]
            for l in ["Engineering", "History", "Computer science"]]
add("RevStabHighVsLowPredAccMin", fmt(100 * min(high_low), 0), "minimum over the 9 high-end vs low-end field pairs")
acc_pairs = fo[fo.measure == "prediction_accuracy"]
add("RevPairsBelowNinetyFivePredAcc", int((acc_pairs.share_same_order < 0.95).sum()), f"of {len(acc_pairs)} field pairs")
t5 = result("bootstrap_top5_stability.csv")
add("RevTopFiveStabCSJACM", fmt(100 * t5[(t5.field == "Computer science") & (t5.name == "Journal of the ACM")].share_of_replicates.iloc[0], 0))
stable = [f for f in FIELDS if (t5[(t5.field == f) & t5.in_full_sample_top5].share_of_replicates == 1).all()]
add("RevTopFiveFullyStableFields", ", ".join(stable), "full-sample top 5 appears in every replicate")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Ranking consistency: non-tie comparisons that contradict the respondent's inferred ranking")
comp = pd.read_csv(common.PUBLIC / "comparisons.csv")
ind = pd.read_csv(common.DERIVED / "individual_rankings.csv")
normed_rank = ind.set_index(["user_db_id", "venue_db_id"])["normed_rank"]
c = comp[comp.is_tie == False].copy()  # noqa: E712
c["rank_pref"] = normed_rank.reindex(pd.MultiIndex.from_arrays([c.user_db_id, c.pref_venue_db_id])).to_numpy()
c["rank_other"] = normed_rank.reindex(pd.MultiIndex.from_arrays([c.user_db_id, c.other_venue_db_id])).to_numpy()
pred = c.dropna(subset=["rank_pref", "rank_other"]).copy()
pred["consistent"] = pred.rank_pref > pred.rank_other
per_user = pred.groupby("user_db_id").consistent.agg(["sum", "count"]).reindex(ind.user_db_id.unique(), fill_value=0)
add("PercentWithoutUpset", num(100 * (per_user["sum"] == per_user["count"]).mean(), 1), "respondents with no contradiction")
add("PercentOfCompsUpset", num(100 * (1 - pred.consistent.mean()), 1))
upset_by_field = 100 * (1 - pred.groupby("field").consistent.mean())
add("MinPercentOfCompsUpsetByField", num(upset_by_field.min(), 1), upset_by_field.idxmin())
add("MaxPercentOfCompsUpsetByField", num(upset_by_field.max(), 1), upset_by_field.idxmax())
add("ObsAvgUpsetRank", num(pred.loc[~pred.consistent, "rank_other"].mean(), 2),
    "mean rescaled rank (1 = most preferred) of the higher-ranked venue in a contradicted pair")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Subfields (SI Table S1, SI Fig. S1), with catch-all answers excluded and included")
ms = result("subfield_consensus.csv").set_index("field")
for field, parts in {"Business": ["Field", "Sub", "Gain"], "Computer science": ["Field", "Sub", "Gain"],
                     "Philosophy": ["Gain"], "Sociology": ["Gain"], "Psychology": ["Field"], "Biology": ["Field"],
                     "Economics": ["Field", "Sub"], "Chemistry": ["Field", "Sub"]}.items():
    for part in parts:
        add(f"RevSub{MACRO[field]}TopFive{part}", fmt(ms.loc[field, f"top5_{part.lower()}"], 1))
mc = result("subfield_correlations.csv").set_index("comparison")
for key, tag in {"Venues in 12 sets: field vs subfield": "Acc", "Mean overlap: field vs subfield": "Overlap",
                 "Top-five agreement: field vs subfield": "TopFive",
                 "subfield: Venues in 12 sets vs Mean overlap": "AccOverlap",
                 "subfield: Venues in 12 sets vs Top-five agreement": "AccTopFive",
                 "subfield: Mean overlap vs Top-five agreement": "OverlapTopFive"}.items():
    r = mc.loc[key]
    add(f"RevSubCorr{tag}Pearson", fmt(r.pearson_r, 2))
    if tag in ("Acc", "Overlap", "TopFive"):
        add(f"RevSubCorr{tag}Spearman", fmt(r.spearman_rho, 2))
    else:
        add(f"RevSubCorr{tag}PearsonP", pfmt(r.pearson_p))
cat = result("subfield_consensus_catchall.csv").set_index("field")
for field in ["Business", "Computer science", "Economics", "Chemistry", "Sociology", "Philosophy", "Medicine"]:
    add(f"RevSubCatchAll{MACRO[field]}TopFiveGain", fmt(cat.loc[field, "top5_gain"], 1))
cat_corr = result("subfield_correlations_catchall.csv").set_index("comparison")
add("RevSubCatchAllCorrTopFiveSpearman", fmt(cat_corr.loc["Top-five agreement: field vs subfield", "spearman_rho"], 2))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Sociology subfields: American Anthropologist and the New England Journal of Medicine (SI)")
selections = pd.read_csv(common.PUBLIC / "venue_selections.csv")
chosen = selections[selections.would_publish]
soc = resp[resp.field == "Sociology"].copy()
soc["subfield"] = soc.subfield.fillna("(none)")
soc = soc[soc.user_db_id.isin(set(chosen.user_db_id))]          # respondents with a consideration set
soc_ids = {k: set(g.user_db_id) for k, g in soc.groupby("subfield")}


def soc_share(subfield, venue):
    ids = soc_ids[subfield]
    k = chosen[(chosen.venue_db_id == venue_id(venue)) & chosen.user_db_id.isin(ids)].user_db_id.nunique()
    return 100 * k / len(ids)


add("RevSocAnthroN", len(soc_ids["Anthropology"]))
add("RevSocAnthroAmAnthPct", fmt(soc_share("Anthropology", "American Anthropologist"), 0))
add("RevSocOtherAmAnthPct", fmt(soc_share("Other", "American Anthropologist"), 1))
for subfield, tag in [("Demography", "Demography"), ("Social science", "SocialScience"), ("Criminology", "Criminology"), ("Other", "Other")]:
    add(f"RevSoc{tag}NEJMPct", fmt(soc_share(subfield, "The New England Journal of Medicine"), 0))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Eq. S1 and Fig. 4: institutional prestige, gender and career stage")
fe = eq[eq.level == "Academia"].set_index(["outcome", "coefficient"])   # pooled, field fixed effects, clustered SEs
for oc, ot in OUTCOMES:
    for coef in ["Prestige", "Gender"]:
        r = fe.loc[(oc, coef)]
        add(f"RevFE{ot}{coef}", fmt(r.estimate, 4))
        add(f"RevFE{ot}{coef}P", pfmt(r.p))
        if (ot, coef) == ("Pref", "Prestige"):
            add("RevFEPrefPrestigeNineDecilesPoints", fmt(900 * r.estimate, 1), "decile 10 minus decile 1 implied by the slope, in percentile points")
perm = result("permutation_test.csv")
perm_i = perm.set_index(["outcome", "quantity"])
for oc, ot in OUTCOMES:   # pooled Eq. S1 without field fixed effects (the observed coefficients of the permutation test)
    add(f"RevPooled{ot}Prestige", fmt(perm_i.loc[(oc, "prestige_coefficient"), "estimate"], 4))
    add(f"RevPooled{ot}Gender", fmt(perm_i.loc[(oc, "gender_coefficient"), "estimate"], 4))
for oc, ot in OUTCOMES:
    for coef in ["Associate", "Full"]:
        r = fe.loc[(oc, coef)]
        add(f"RevFE{ot}{coef}", fmt(r.estimate, 3))
        add(f"RevFE{ot}{coef}P", pfmt(r.p))
g4 = result("fig4_group_means.csv")
gmean = g4[g4.kind == "group_mean"].set_index(["outcome", "quantity"]).estimate
contrast = g4[g4.kind == "comparison"].set_index(["outcome", "quantity"])
add("FigFourMenPrefMean", fmt(gmean[("top_preference", "men")], 2))
add("FigFourWomenPrefMean", fmt(gmean[("top_preference", "women")], 2))
add("FigFourMenAspMean", fmt(gmean[("top_aspiration", "men")], 2))
add("FigFourWomenAspMean", fmt(gmean[("top_aspiration", "women")], 2))
add("FigFourPrefMostPrestigiousMean", fmt(gmean[("top_preference", "decile_10")], 2))
add("FigFourPrefLeastPrestigiousMean", fmt(gmean[("top_preference", "decile_1")], 2))
add("FigFourPrestigePrefPValue", f"{contrast.loc[('top_preference', 'decile10_minus_decile1'), 'p']:.3f}")
add("FigFourAspMostPrestigiousMean", fmt(gmean[("top_aspiration", "decile_10")], 2))
add("FigFourAspLeastPrestigiousMean", fmt(gmean[("top_aspiration", "decile_1")], 2))
r = contrast.loc[("top_preference", "decile10_minus_decile1")]
add("RevFigFourPrefDecileDiff", fmt(r.estimate, 3))
add("RevFigFourPrefDecileDiffCILo", fmt(r.ci_lower, 3))
add("RevFigFourPrefDecileDiffCIHi", fmt(r.ci_upper, 3))
add("RevFigFourPrefGenderP", pfmt(contrast.loc[("top_preference", "women_minus_men"), "p"]))
add("RevFigFourAspDecileP", pfmt(contrast.loc[("top_aspiration", "decile10_minus_decile1"), "p"]))
add("RevFigFourAspGenderP", pfmt(contrast.loc[("top_aspiration", "women_minus_men"), "p"]))
add("RevGapDecileOne", fmt(gmean[("top_preference", "decile_1")] - gmean[("top_aspiration", "decile_1")], 2),
    "preference minus aspiration rank, least prestigious decile")
add("RevGapDecileTen", fmt(gmean[("top_preference", "decile_10")] - gmean[("top_aspiration", "decile_10")], 2),
    "most prestigious decile")
# mean top-preference and top-aspiration rank over the Eq. S1 sample; paired difference with
# institution-clustered standard errors
s = common.eqs1_sample().dropna(subset=["top_ranked_venue_field_normed", "top_init_field_normed", common.CLUSTER]).copy()
s["gap"] = s.top_ranked_venue_field_normed - s.top_init_field_normed
gap = common.fit_clustered("gap ~ 1", s)
add("OverallPrefMean", fmt(s.top_ranked_venue_field_normed.mean(), 2), f"Eq. S1 sample, N = {len(s):,}")
add("OverallAspMean", fmt(s.top_init_field_normed.mean(), 2))
add("RevOverallGap", fmt(gap.params["Intercept"], 2))
add("RevOverallGapP", pfmt(gap.pvalues["Intercept"]))
# per-field Eq. S1 fits (SI Fig. S4)
per_field = eq[eq.level != "Academia"]
for coef in ["Prestige", "Gender"]:
    sig = per_field[(per_field.outcome == "top_preference") & (per_field.coefficient == coef) & (per_field.p < 0.05)]
    add(f"RevPref{coef}SigFields", len(sig), "fields with p < 0.05 for the top preference")
    if coef == "Prestige":
        add("RevPrefPrestigeSigPos", int((sig.estimate > 0).sum()))
    add(f"RevPref{coef}NSFields", len(FIELDS) - len(sig))
phys = per_field[(per_field.outcome == "top_preference") & (per_field.coefficient == "Prestige") & (per_field.level == "Physics")].iloc[0]
add("RevPhysicsPrefPrestige", fmt(phys.estimate, 4))
add("RevPhysicsPrefPrestigeP", pfmt(phys.p))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Permutation test of the pooled Eq. S1 coefficients (SI Fig. S5)")
for _, r in perm.iterrows():
    ot = dict(OUTCOMES)[r.outcome]
    lab = "Gender" if r.quantity.startswith("gender") else "Prestige"
    floor = bool(r.p_at_floor) or r.p <= 2e-4
    add(f"RevPerm{ot}{lab}P", "<2\\times10^{-4}" if floor else pfmt(r.p))
    if (ot, lab) == ("Asp", "Gender"):
        add("RevPermAspGenderNullMedian", fmt(r.null_median, 4))
        add("RevPermAspGenderObs", fmt(r.estimate, 4))
add("RevPermN", count(perm.n_perm.iloc[0]))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Clustering by institution: intraclass correlation and mixed model (SI)")
ic = result("inference_checks.csv")


def check(sec, quantity, outcome=None, level=None, term=None):
    m = ic[(ic.section == sec) & (ic.quantity == quantity)]
    if outcome is not None:
        m = m[m.outcome == outcome]
    if level is not None:
        m = m[m.level == level]
    if term is not None:
        m = m[m.term == term]
    if len(m) != 1:
        raise ValueError(f"inference_checks.csv: {len(m)} rows for {(sec, quantity, outcome, level, term)}")
    return float(m.value.iloc[0])


for oc, ot in OUTCOMES:
    add(f"RevICC{ot}", fmt(check("icc", "icc", oc), 3))
    add(f"RevICC{ot}CILo", fmt(check("icc", "ci_lower", oc), 3))
    add(f"RevICC{ot}CIHi", fmt(check("icc", "ci_upper", oc), 3))
    add(f"RevICC{ot}Deff", fmt(check("icc", "design_effect", oc), 2))
    add(f"RevMixed{ot}InstVar", fmt(check("mixed_model", "institution_variance", oc), 4))
    add(f"RevMixed{ot}ResidVar", fmt(check("mixed_model", "residual_variance", oc), 3))
add("RevICCMeanClusterSize", fmt(check("icc", "n0_anova_mean_cluster_size", "top_preference"), 1))
add("RevICCSingletonInst", fmt(check("icc", "n_singleton_institutions", "top_preference"), 0))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Fig. 5: own top-5 vs field top-5 venues in the publication record (SI Table S2)")
pooled = f5[f5.level == "Academia"].set_index(["quantity", "ref"])
d10 = pooled.loc[("diff10", "own-field")]
add("FigFiveOverallDiffAtPrestigeTen", fmt(5 * d10.estimate, 2), "venues out of five, own minus field at decile 10")
add("FigFiveOverallDiffPPAtPrestigeTen", fmt(100 * d10.estimate, 1), "percentage points")
add("RevFigFiveOverallDiffCILo", fmt(5 * d10.ci_lower, 2))
add("RevFigFiveOverallDiffCIHi", fmt(5 * d10.ci_upper, 2))
for field in ["Biology", "Mathematics"]:
    own = f5[(f5.level == field) & (f5.quantity == "pred10") & (f5.ref == "own")].estimate.iloc[0]
    fld = f5[(f5.level == field) & (f5.quantity == "pred10") & (f5.ref == "field")].estimate.iloc[0]
    add(f"FigFive{field}IndivPredAtTen", fmt(5 * own, 1), "predicted venues out of five at decile 10")
    add(f"FigFive{field}FieldPredAtTen", fmt(5 * fld, 2 if 5 * fld < 1 else 1))
diffs = f5[(f5.quantity == "diff10") & (f5.level != "Academia")]
not_sig = sorted(diffs[~diffs.sig_bh_01.astype(bool)].level)
add("RevFigFiveNSFields", ", ".join(not_sig), "own minus field at decile 10 not significant at BH 0.01")
add("RevFigFiveSigFieldsCount", len(FIELDS) - len(not_sig))
slopes = f5[(f5.quantity == "slope") & (f5.level != "Academia")]
add("RevFigFiveOwnSlopeSigFields", int(((slopes.ref == "own") & (slopes.p < 0.05) & (slopes.estimate > 0)).sum()))
add("RevFigFiveFieldSlopeSigFields", int(((slopes.ref == "field") & (slopes.p < 0.05) & (slopes.estimate > 0)).sum()))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Functional form (fractional logit) and fitted-value checks (SI)")
for coef in ["Prestige", "Gender"]:
    for quantity, tag in [("eqs1_ame_over_ols_median", "Median"), ("eqs1_ame_over_ols_min", "Min"), ("eqs1_ame_over_ols_max", "Max")]:
        add(f"RevFLRatio{coef}{tag}", fmt(check("fractional_logit", quantity, term=coef), 2), "average marginal effect / OLS coefficient")
add("RevFLTermsChecked", fmt(check("fractional_logit", "eqs1_n_prestige_gender_coefficients"), 0))
add("RevFLMedicineOwnSlopeOLSP", pfmt(check("fractional_logit", "fig5_slope_significance_change_ols_p", level="Medicine")))
add("RevFLMedicineOwnSlopeFLP", pfmt(check("fractional_logit", "fig5_slope_significance_change_fl_p", level="Medicine")))
add("RevFLOverallDiffVenues", fmt(check("fractional_logit", "fig5_pooled_own_minus_field_at_decile10_venues_fl_estimate", level="Academia"), 2))
add("RevFLOverallDiffCILo", fmt(check("fractional_logit", "fig5_pooled_own_minus_field_at_decile10_venues_fl_ci_lower", level="Academia"), 2))
add("RevFLOverallDiffCIHi", fmt(check("fractional_logit", "fig5_pooled_own_minus_field_at_decile10_venues_fl_ci_upper", level="Academia"), 2))
add("RevFittedEqSOneMin", fmt(check("fitted_values", "eqs1_fitted_min"), 2))
add("RevFittedEqSOneMax", fmt(check("fitted_values", "eqs1_fitted_max"), 2))
add("RevFittedFigFivePooledShareBelowZero", fmt(100 * check("fitted_values", "fitted_share_below_0", "fig5", "Academia"), 1))
add("RevFittedFigFivePooledMin", fmt(check("fitted_values", "fitted_min", "fig5", "Academia"), 2))
add("RevFittedFigFivePlottedMin", fmt(check("fitted_values", "fig5_plotted_min"), 2))
add("RevFittedFigFivePlottedMax", fmt(check("fitted_values", "fig5_plotted_max"), 2))
add("RevFigFiveShareFieldZero", fmt(100 * check("fractional_logit", "fig5_share_field_reference_outcomes_zero"), 0),
    "share of field-top-5 outcomes exactly zero")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Venues already published in: aspirations and consideration sets (SI)")
pubs = pd.read_csv(common.PUBLIC / "publication_counts.csv").merge(resp[["user_db_id", "venue_bug"]], on="user_db_id", how="left")
asp_cols = ["published_in_top_aspiration", "published_in_mid_aspiration", "published_in_low_aspiration"]
add("RevPubAspN", count(len(pubs)), "respondents with a publication record (public_data/publication_counts.csv)")
add("RevPubAnyAspirationPct", fmt(100 * pubs[asp_cols].any(axis=1).mean(), 1))
add("RevPubTopAspirationPct", fmt(100 * pubs.published_in_top_aspiration.mean(), 1))
add("RevPubMidAspirationPct", fmt(100 * pubs.published_in_mid_aspiration.mean(), 1))
add("RevPubLowAspirationPct", fmt(100 * pubs.published_in_low_aspiration.mean(), 1))
add("RevPubTopAspirationNoBugPct", fmt(100 * pubs.loc[pubs.venue_bug == False, "published_in_top_aspiration"].mean(), 1))  # noqa: E712
cons = pubs[pubs.n_consideration_set > 0]
add("RevPubConsSize", fmt(cons.n_consideration_set.mean(), 1))
add("RevPubConsPub", fmt(cons.n_consideration_published.mean(), 1))
add("RevPubConsPubPct", fmt(100 * (cons.n_consideration_published / cons.n_consideration_set).mean(), 1), "mean over respondents")
add("RevPubConsZeroN", int((cons.n_consideration_published == 0).sum()))
add("RevPubConsZeroPct", fmt(100 * (cons.n_consideration_published == 0).mean(), 1))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Flagship venues: consensus rank vs JIF rank in the home field (SI Tables S4-S5, SI Fig. S7)")
jt = result("jif_rank_tables.csv")
flag = jt[(jt.table == "S5") & jt.flagship_field.astype(bool)]
add("FlagshipMaxDiff", num(flag["diff"].max(), 1), "JIF rank minus consensus rank; positive = JIF undervalues the venue")
add("FlagshipMinPosDiff", num(flag.loc[flag["diff"] > 0, "diff"].min(), 1))
add("RevFlagshipMedianDiff", fmt(flag["diff"].median(), 1))
add("RevFlagshipN", len(flag))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("JIF prediction accuracy and alternative citation metrics (SI Fig. S6, SI Table S3)")
jn = result("jif_accuracy_summary.csv").set_index("key")["value"]
add("RevJIFVenueCoveragePct", fmt(float(jn.venues_with_jif_pct), 0))
add("RevJIFComparisonCoveragePct", fmt(float(jn.comparisons_available_pct), 0))
add("RevJIFAccAll", fmt(float(jn.jif_acc_all), 0))
add("RevConsensusAccOnJIFComparisons", fmt(float(jn.consensus_acc_all), 0), "field consensus rankings on the same comparisons")
add("RevJIFMostPredictiveField", str(jn.most_predictive_field))
add("RevJIFMostPredictiveAcc", fmt(float(jn.most_predictive_acc), 0))
add("RevJIFLeastPredictiveField", str(jn.least_predictive_field))
add("RevJIFLeastPredictiveAcc", fmt(float(jn.least_predictive_acc), 0))
add("RevJIFCoverageMaxField", str(jn.coverage_max_field))
add("RevJIFCoverageMaxPct", fmt(float(jn.coverage_max_pct), 0))
add("RevJIFCoverageMinField", str(jn.coverage_min_field).replace("Computer science", "Computer Science"))
add("RevJIFCoverageMinPct", fmt(float(jn.coverage_min_pct), 0))
add("RevJIFAdvantageMinPP", fmt(float(jn.advantage_min_pp), 0), str(jn.advantage_min_field))
add("RevJIFAdvantageMaxPP", fmt(float(jn.advantage_max_pp), 0), str(jn.advantage_max_field))
am = result("alt_metrics_accuracy.csv").set_index("field")
adv = am.drop(index="All fields")[[col for col in am.columns if col.startswith("adv_")]]
add("RevAltMetricCombos", adv.size, "field x metric combinations")
add("RevAltMetricSurveyWins", int((adv > 0).sum().sum()), "combinations in which the consensus ranking beats the metric")
add("RevAltMetricCSAISAdv", fmt(adv.loc["Computer science", "adv_ais"], 1))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Alternative prestige measure: THE World University Rankings 2024 (SI)")
pm = result("prestige_measures.csv")
the = pm[pm.section == "the_coverage"]


def the_row(sample, quantity):
    return the[(the["sample"] == sample) & (the.quantity == quantity)].iloc[0]


add("RevTHECoverageEqSOnePct", fmt(100 * the_row("Eq. S1", "covered_share").value, 1), "Eq. S1 respondents at a THE-ranked institution")
add("RevTHECoverageEqSOneN", count(the_row("Eq. S1", "covered_respondents").value))
add("RevTHECoverageFigFivePct", fmt(100 * the_row("Fig. 5", "covered_share").value, 1))
unranked = the_row("Eq. S1", "unranked_respondents")
add("RevTHEUnrankedEqSOneN", fmt(unranked.value, 0))
add("RevTHEUnrankedEqSOneInst", fmt(unranked.n_inst, 0))
add("RevTHEUnrankedMeanAARCDecile", fmt(the_row("Eq. S1", "unranked_mean_hiring_decile").value, 1))
add("RevTHEUnrankedInLowerHalf", fmt(the_row("Eq. S1", "unranked_in_hiring_deciles_1_to_5").value, 0))
add("RevTHESpearmanWithAARC", fmt(the_row("Eq. S1", "spearman_the_vs_hiring_decile").value, 2), "respondent-weighted, THE-covered Eq. S1 sample")
e1 = pm[(pm.section == "eqs1") & (pm.level == "Academia") & (pm.quantity == "coefficient")]


def the_coef(measure, outcome, group):
    return e1[(e1.measure == measure) & (e1.outcome == outcome) & (e1.group == group)].iloc[0]


for oc, ot in OUTCOMES:
    for measure, tag in [("THE (THE-covered)", "THE"), ("hiring (THE-covered)", "AARCOnTHESample")]:
        r = the_coef(measure, oc, "prestige")
        add(f"Rev{tag}{ot}Prestige", fmt(r.value, 4))
        if tag == "THE":
            add(f"Rev{tag}{ot}PrestigeCILo", fmt(r.ci_lower, 4))
            add(f"Rev{tag}{ot}PrestigeCIHi", fmt(r.ci_upper, 4))
            add(f"Rev{tag}{ot}PrestigeP", pfmt(r.p))
        add(f"Rev{tag}{ot}Gender", fmt(the_coef(measure, oc, "gender").value, 3))
f4the = pm[(pm.section == "fig4_contrast") & (pm.measure == "THE (THE-covered)")]
for oc, ot in OUTCOMES:
    rows = f4the[f4the.outcome == oc]
    add(f"RevTHE{ot}DecileTenMean", fmt(rows[rows.decile == 10].value.iloc[0], 2))
    add(f"RevTHE{ot}DecileOneMean", fmt(rows[rows.decile == 1].value.iloc[0], 2))
    add(f"RevTHE{ot}DecileP", pfmt(rows[rows.quantity == "difference_decile_10_vs_1"].p.iloc[0]))
f5the = pm[(pm.section == "fig5") & (pm.measure == "THE (THE-covered)")]
dv = f5the[(f5the.level == "Academia") & (f5the.quantity == "diff10_venues")].iloc[0]
add("RevTHEFigFiveDiffVenues", fmt(dv.value, 2))
add("RevTHEFigFiveDiffCILo", fmt(dv.ci_lower, 2))
add("RevTHEFigFiveDiffCIHi", fmt(dv.ci_upper, 2))
ns_the = sorted(f5the[(f5the.level != "Academia") & (f5the.quantity == "diff10") & (f5the.p_bh >= 0.01)].level)
add("RevTHEFigFiveNSFields", ", ".join(ns_the))
add("RevTHEFigFiveNSFieldsCount", len(ns_the))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Alternative prestige measure: Carnegie Classification 2021 (SI)")
cb = pm[(pm.section == "carnegie") & (pm["sample"] == "Eq. S1")]


def carnegie(group, quantity):
    m = cb[cb.group.isna()] if group is None else cb[cb.group == group]
    return float(m[m.quantity == quantity].value.iloc[0])


r1, r2 = carnegie("R1", "share_respondents"), carnegie("R2", "share_respondents")
add("RevCarnegieROnePct", fmt(100 * r1, 1), "Eq. S1 respondents at R1 institutions")
add("RevCarnegieRTwoPct", fmt(100 * r2, 1))
add("RevCarnegieRTwoInst", fmt(carnegie("R2", "institutions"), 0))
add("RevCarnegieOtherPct", fmt(100 * (1 - r1 - r2), 1))
add("RevCarnegieWithinPct", fmt(100 * carnegie(None, "share_prestige_decile_variance_within_tiers"), 0),
    "share of prestige-decile variance within Carnegie tiers")
add("RevCarnegieSpearmanTierDecile", fmt(carnegie(None, "spearman_tier_vs_prestige_decile"), 2))
add("RevCarnegieDecileSDROne", fmt(carnegie("R1", "sd_prestige_decile"), 2))
add("RevCarnegieDecileSDAll", fmt(carnegie("all", "sd_prestige_decile"), 2))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Sample composition: subgroups, raking and field weights (SI Tables S8-S10)")
S = result("subgroup_summary.csv")
sp = S[S.measure.isin(["acc", "ovl", "top5"])].spearman_field_order_vs_all
addcr("CRSubSpearmanMin", fmt(sp.min(), 2), "Spearman correlation of the field order within a subgroup with the all-respondent order")
addcr("CRSubSpearmanMax", fmt(sp.max(), 2))
addcr("CRSubNNull", 200, "random same-size subsets per field and subgroup (N_NULL in si_tables8_10_composition.py)")
rank_rows = S[S.measure == "rank_rho"]
addcr("CRSubRankOutside", int(rank_rows.fields_below_null.sum() + rank_rows.fields_above_null.sum()), "subgroup rankings outside the null band")
addcr("CRSubRankTotal", int(rank_rows.n_fields.sum()))
si = S.set_index(["measure", "subgroup"])
for sg, tag in [("women", "Women"), ("prestige_bottom", "Bottom")]:
    addcr(f"CRSub{tag}OvlDiff", fmt(si.loc[("ovl", sg), "mean_value_minus_null"], 1))
    addcr(f"CRSub{tag}TopFiveDiff", fmt(si.loc[("top5", sg), "mean_value_minus_null"], 1))
    addcr(f"CRSub{tag}RankRho", fmt(si.loc[("rank_rho", sg), "mean_value"], 2))
    addcr(f"CRSub{tag}RankRhoNull", fmt(si.loc[("rank_rho", sg), "mean_null"], 2))
addcr("CRSubMenTopFiveDiff", fmt(si.loc[("top5", "men"), "mean_value_minus_null"], 1))
addcr("CRSubTopTopFiveDiff", fmt(si.loc[("top5", "prestige_top"), "mean_value_minus_null"], 1))
addcr("CRSubBottomNFields", int(si.loc[("ovl", "prestige_bottom"), "n_fields"]), "fields with enough bottom-tercile respondents")
A = result("raking_diagnostics.csv").set_index("field")
addcr("CRRakeDeffMin", fmt(A.deff.min(), 2), "design effect of the raking weights")
addcr("CRRakeDeffMax", fmt(A.deff.max(), 2))
addcr("CRRakeCapBinds", int(A.cap_binds.sum()), "weights at the cap of 5")
R = result("raked_consensus.csv")


def raked(measure):
    g = R[R.measure == measure]
    return {"lower": int((g["diff"] < 0).sum()), "higher": int((g["diff"] > 0).sum()),
            "max_abs": g["diff"].abs().max(), "field_max": g.loc[g["diff"].abs().idxmax(), "field"],
            "mean_abs": g["diff"].abs().mean(), "spearman": spearmanr(g.unweighted, g.weighted)[0],
            "range": g.unweighted.max() - g.unweighted.min()}


t5r = raked("top5")
addcr("CRRakeTopFiveLower", t5r["lower"], "fields whose raked top-5 agreement is below the unweighted value")
addcr("CRRakeTopFiveMaxAbs", fmt(t5r["max_abs"], 1))
addcr("CRRakeTopFiveMaxField", t5r["field_max"])
addcr("CRRakeTopFiveMeanAbs", fmt(t5r["mean_abs"], 1))
addcr("CRRakeTopFiveSpearman", fmt(t5r["spearman"], 2))
addcr("CRRakeTopFiveRange", fmt(t5r["range"], 0), "range of the unweighted values across fields")
ovl = raked("ovl")
addcr("CRRakeOvlLower", ovl["lower"])
addcr("CRRakeOvlMaxAbs", fmt(ovl["max_abs"], 1))
addcr("CRRakeOvlMaxField", ovl["field_max"])
addcr("CRRakeOvlMeanAbs", fmt(ovl["mean_abs"], 1))
addcr("CRRakeOvlSpearman", fmt(ovl["spearman"], 2))
acc = raked("acc100")
addcr("CRRakeAccHigher", acc["higher"], "fields whose raked venue accumulation is above the unweighted value")
addcr("CRRakeAccMaxAbs", fmt(acc["max_abs"], 1))
addcr("CRRakeAccMaxField", acc["field_max"])
addcr("CRRakeAccMeanAbs", fmt(acc["mean_abs"], 1))
addcr("CRRakeAccSpearman", fmt(acc["spearman"], 2))
addcr("CRRakeAccRange", fmt(acc["range"], 0))
addcr("CRRakeTopThreeSpearman", fmt(raked("top3")["spearman"], 2))
FW = result("field_weighted_models.csv").set_index(["analysis", "outcome", "term", "scheme"])


def weighted(outcome, term, scheme, analysis="eqS1_pooled_fieldFE"):
    return FW.loc[(analysis, outcome, term, scheme)]


u, w = weighted("top_aspiration", "Prestige", "unweighted"), weighted("top_aspiration", "Prestige", "field_weighted")
addcr("CRFwAspPrestigeU", fmt(u.estimate, 4), "U = unweighted, W = fields weighted to their AARC share")
addcr("CRFwAspPrestigeW", fmt(w.estimate, 4))
addcr("CRFwAspPrestigePctChange", f"{100 * (abs(w.estimate) - abs(u.estimate)) / abs(u.estimate):+.0f}", "change in magnitude, percent")
u, w = weighted("top_aspiration", "Gender", "unweighted"), weighted("top_aspiration", "Gender", "field_weighted")
addcr("CRFwAspGenderU", fmt(u.estimate, 4))
addcr("CRFwAspGenderUP", p3(u.p))
addcr("CRFwAspGenderW", fmt(w.estimate, 4))
addcr("CRFwAspGenderWP", p3(w.p))
addcr("CRFwAspWmMWP", p3(weighted("top_aspiration", "women_minus_men", "field_weighted", "fig4_comparison").p))
u, w = (weighted("top_aspiration", "decile10_minus_decile1", sc, "fig4_comparison") for sc in ("unweighted", "field_weighted"))
addcr("CRFwAspTenMinusOneU", fmt(u.estimate, 3))
addcr("CRFwAspTenMinusOneW", fmt(w.estimate, 3))
addcr("CRFwAspTenMinusOneWP", p3(w.p))
u, w = weighted("top_preference", "Prestige", "unweighted"), weighted("top_preference", "Prestige", "field_weighted")
addcr("CRFwPrefPrestigeU", fmt(u.estimate, 4))
addcr("CRFwPrefPrestigeW", fmt(w.estimate, 4))
addcr("CRFwPrefPrestigeWP", p3(w.p))
u, w = weighted("top_preference", "Gender", "unweighted"), weighted("top_preference", "Gender", "field_weighted")
addcr("CRFwPrefGenderU", fmt(u.estimate, 4))
addcr("CRFwPrefGenderW", fmt(w.estimate, 4))
addcr("CRFwPrefGenderWP", p3(w.p))
u, w = (weighted("top_preference", "decile10_minus_decile1", sc, "fig4_comparison") for sc in ("unweighted", "field_weighted"))
addcr("CRFwPrefTenMinusOneU", fmt(u.estimate, 3))
addcr("CRFwPrefTenMinusOneW", fmt(w.estimate, 3))
addcr("CRFwPrefTenMinusOneWP", p3(w.p))
addcr("CRFwFiveSlopeOwnW", fmt(weighted("own", "slope", "field_weighted", "fig5_pooled").estimate, 3))
addcr("CRFwFiveSlopeFieldW", fmt(weighted("field", "slope", "field_weighted", "fig5_pooled").estimate, 3))
addcr("CRFwFiveDiffU", fmt(weighted("own-field", "diff10", "unweighted", "fig5_pooled").estimate, 3))
addcr("CRFwFiveDiffW", fmt(weighted("own-field", "diff10", "field_weighted", "fig5_pooled").estimate, 3))
addcr("CRFwFiveDiffVenuesW", fmt(5 * weighted("own-field", "diff10", "field_weighted", "fig5_pooled").estimate, 2), "venues out of five")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Sample vs faculty population (SI Table S11): respondents (PorC) and AARC 2023 aggregates")
S11 = [("Academia", "Academia")] + [(f, MACRO[f]) for f in FIELDS]
for field, F in S11:
    sub = resp if field == "Academia" else resp[resp.field == field]
    gender = sub.gender.value_counts(normalize=True)
    stage = sub.career_stage.value_counts(normalize=True)
    add(f"{F}MenPercentPorC", num(100 * gender.get("gm", 0), 2))
    add(f"{F}WomenPercentPorC", num(100 * gender.get("gf", 0), 2))
    add(f"{F}UnknownPercentPorC", num(100 * gender.get("-", 0), 2))
    add(f"{F}AssistantProfessorPercentPorC", num(100 * stage.get("Assistant Professor", 0), 2))
    add(f"{F}AssociateProfessorPercentPorC", num(100 * stage.get("Associate Professor", 0), 2))
    add(f"{F}FullProfessorPercentPorC", num(100 * stage.get("Full Professor", 0), 2))
    add(f"{F}CountPorC", count(len(sub)))
aarc = pd.read_csv(common.PUBLIC / "aarc_field_margins.csv").set_index("field")
for field, F in S11:
    r = aarc.loc[field]
    add(f"{F}AssistantProfessorPercentAARC", num(r.pct_assistant, 2))
    add(f"{F}AssociateProfessorPercentAARC", num(r.pct_associate, 2))
    add(f"{F}CountAARC", count(r.aarc_n_2023))
    add(f"{F}FullProfessorPercentAARC", num(r.pct_full, 2))
    add(f"{F}MeanPrestigeAARC", fmt(r.prestige_percentile_mean, 2), "100 - prestige percentile, higher = more prestigious")
    add(f"{F}MenPercentAARC", num(r.pct_men, 2))
    add(f"{F}SdPrestigeAARC", fmt(r.prestige_percentile_sd, 2))
    add(f"{F}UnknownPercentAARC", num(r.pct_gender_unknown, 2))
    add(f"{F}WomenPercentAARC", num(r.pct_women, 2))

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
section("Values from sources not in this repository")
note("Each constant below names its source. Rates and shares that follow from the constants are derived here.")

note("-- complete survey database, all fields (public_data/ holds only the 13 fields with >= 100 respondents) --")
SRC_DB = "complete survey database (all fields)"
add("TotalParticipants", "3,620", SRC_DB)
add("TotalVenues", "8,044", SRC_DB + ": distinct venues selected by any participant")
add("TotalVenueComparisons", "163,002", SRC_DB + ": pairwise comparisons by all participants")

note("-- survey email records: sample frame and response funnel (SI Table S6); not shareable --")
SRC_EMAIL = "survey email records"
N_FRAME, N_STARTED, N_COMPLETED, N_NOT_IN_FRAME = 71390, 4448, 3462, 140
add("NSurveyed", count(N_FRAME), SRC_EMAIL + ": faculty emailed")
add("NSurveyedStarted", count(N_STARTED), SRC_EMAIL + ": frame members who started the survey")
add("NSurveyedCompleted", count(N_COMPLETED), SRC_EMAIL + ": frame members who completed the main portion")
add("PercSurveyedStart", num(100 * N_STARTED / N_FRAME, 2), "derived: started / emailed")
add("PercSurveyedCompleted", num(100 * N_COMPLETED / N_FRAME, 2), "derived: completed / emailed")
add("NStudied", count(len(resp)), "respondents in the 13 analysis fields (public_data/respondents.csv)")
add("NStudiedNotSurveyed", count(N_NOT_IN_FRAME), SRC_EMAIL + ": analysed respondents who were not in the email frame")
add("PercStudiedNotSurveyed", num(100 * N_NOT_IN_FRAME / len(resp), 2), "derived")
FUNNEL = {   # emailed, consented, completed the start page, completed the survey (AARC field labels)
    "Business": (8783, 539, 433, 418), "Psychology": (6376, 431, 359, 352), "Biology": (4385, 366, 326, 324),
    "PhysicsAstronomy": (5805, 379, 313, 306), "Sociology": (2418, 257, 226, 221), "Mathematics": (4691, 249, 194, 189),
    "ComputerScience": (5579, 274, 193, 187), "History": (4264, 240, 179, 175), "Chemistry": (3277, 203, 178, 175),
    "Economics": (3451, 195, 164, 162), "Biochemistry": (2715, 172, 150, 146), "Anthropology": (1951, 175, 120, 118),
    "MechanicalEngineering": (3576, 143, 116, 115), "CivilEngineering": (2519, 143, 113, 109), "Philosophy": (1929, 128, 98, 96),
}
for F, (emailed, consented, started, completed) in FUNNEL.items():
    add(f"{F}EmailedCount", emailed, SRC_EMAIL)
    add(f"{F}ConsentedCount", consented, SRC_EMAIL)
    add(f"{F}StartPageCount", started, SRC_EMAIL)
    add(f"{F}ResponseCount", completed, SRC_EMAIL)
    add(f"{F}ResponseRate", num(100 * completed / emailed, 2), "derived: completed / emailed")
    add(f"{F}ConditionalCompletionRate", num(100 * completed / started, 2), "derived: completed / start page")

note("-- survey email records linked to the AARC census: response by institutional prestige, gender and rank (SI Table S7) --")
SRC_LINK = "survey email records linked to the AARC census"
N_LINKED = 66371
addcr("CRRespNLinked", count(N_LINKED), SRC_LINK + ": frame members linked to a prestige decile")
addcr("CRRespPctLinked", fmt(100 * N_LINKED / N_FRAME, 0), "derived: linked / emailed")
addcr("CRRespNCompLinked", "3,265", SRC_LINK + ": linked frame members who completed the survey")
for name, value, what in [
        ("CRRespRateMin", "3.7", "lowest completion rate across prestige deciles, %"),
        ("CRRespRateMax", "5.5", "highest completion rate across prestige deciles, %"),
        ("CRRespRateDecOne", "3.8", "completion rate in decile 1, %"), ("CRRespRateDecTen", "3.7", "completion rate in decile 10, %"),
        ("CRRespFrameShareDecOne", "2.5", "share of the linked frame in decile 1, %"),
        ("CRRespFrameShareDecTen", "13.9", "share of the linked frame in decile 10, %"),
        ("CRRespSlope", "-0.04", "OLS slope of completion (pp) per prestige decile"), ("CRRespSlopeLo", "-0.15", "95% CI"),
        ("CRRespSlopeHi", "0.06", "95% CI"), ("CRRespSlopeP", "0.400", "p-value"),
        ("CRRespSlopeFE", "-0.03", "slope with AARC field fixed effects"), ("CRRespSlopeFEP", "0.523", "p-value"),
        ("CRRespSlopePersonFE", "-0.03", "slope, person-linked frame, field fixed effects"), ("CRRespSlopePersonFEP", "0.579", "p-value"),
        ("CRRespWomen", "5.1", "completion rate of women, %"), ("CRRespMen", "5.5", "completion rate of men, %"),
        ("CRRespAsst", "5.6", "completion rate of assistant professors, %"), ("CRRespAssoc", "5.2", "associate professors, %"),
        ("CRRespFull", "5.0", "full professors, %"),
        ("CRRespWomenDiff", "-0.49", "women minus men, pp, model with field fixed effects"), ("CRRespWomenDiffP", "0.049", "p-value"),
        ("CRRespFullDiff", "-0.84", "full minus assistant professors, pp"), ("CRRespFullDiffP", "0.006", "p-value")]:
    addcr(name, value, SRC_LINK + ": " + what)
note("-- mean institutional prestige (100 - AARC percentile) at each stage from population to analysed sample --")
addcr("CRStageAARCTwentyThree", fmt(aarc.loc["Academia", "prestige_percentile_mean"], 1), "AARC 2023 faculty (public_data/aarc_field_margins.csv)")
addcr("CRStageAARCTwenty", "70.8", "AARC faculty census 2020 (scieco)")
addcr("CRStageFrame", "75.3", SRC_LINK + ": email frame")
addcr("CRStageResp", "75.5", SRC_LINK + ": frame members who completed the survey")

note("-- AARC faculty census via the scieco package: prestige percentiles of the respondents' institutions (SI Table S11) --")
SRC_AARC = "AARC census (scieco): mean/SD of 100 - prestige percentile over respondents with a ranked institution; needs raw ranks"
PRESTIGE_PORC = {   # mean, standard deviation
    "Academia": (75.09, 20.63), "Biology": (72.45, 20.50), "Business": (74.94, 19.40), "Chemistry": (73.43, 20.59),
    "ComputerScience": (78.98, 18.09), "Economics": (75.34, 20.82), "Engineering": (75.57, 20.81), "History": (78.17, 18.29),
    "Mathematics": (75.48, 20.54), "Medicine": (77.23, 20.58), "Philosophy": (79.60, 18.74), "Physics": (77.47, 19.33),
    "Psychology": (70.27, 24.06), "Sociology": (79.64, 17.53),
}
for F, (mean, sd) in PRESTIGE_PORC.items():
    add(f"{F}MeanPrestigePorC", fmt(mean, 2), SRC_AARC)
    add(f"{F}SdPrestigePorC", fmt(sd, 2), SRC_AARC)
addcr("CRStageSample", fmt(PRESTIGE_PORC["Academia"][0], 1), "derived from AcademiaMeanPrestigePorC")

note("-- null-model simulation of the survey's adaptive pairing algorithm (needs the survey software) --")
add("SimAvgUpsetRank", "0.6", "mean over 10 simulated datasets of the rescaled rank of the higher-ranked venue in a simulated ranking violation")

note("-- institution-name crosswalks (institution names are not shared; only the resulting tier/decile are in respondents.csv) --")
add("RevTHEExactMatches", 133, "crosswalk to the THE World University Rankings 2024: exact name matches after normalization")
add("RevTHEManualMatches", 28, "crosswalk to the THE World University Rankings 2024: matches resolved by hand")
add("RevCarnegieExactMatches", 216, "crosswalk to the Carnegie Classification 2021: exact name matches after normalization")
add("RevCarnegieManualMatches", 43, "crosswalk to the Carnegie Classification 2021: matches resolved by hand")
add("RevCarnegieROneNational", 146, "Carnegie Classification 2021: R1 institutions nationally (basic classification 15)")
add("RevCarnegieRTwoNational", 133, "Carnegie Classification 2021: R2 institutions nationally (basic classification 16)")
add("RevCarnegieROneCovered", 144, "Carnegie crosswalk: distinct R1 institutions (Carnegie unit ids) with Eq. S1 respondents")


# ── output ────────────────────────────────────────────────────────────────────────────────────────
def render():
    lines = ["% numbers.tex -- generated by reproducibility/compute_numbers.py; do not edit by hand.",
             "% One \\newcommand per number quoted in the paper, grouped by topic. The last section holds the",
             "% values that come from sources outside this repository, each with its source."]
    for e in ENTRIES:
        if e[0] == "section":
            lines += ["", f"% ===== {e[1]} ====="]
        elif e[0] == "note":
            lines.append(f"% {e[1]}")
        else:
            _, name, value, comment = e
            lines.append(f"\\newcommand{{\\{name}}}{{{value}}}" + (f"  % {comment}" if comment else ""))
    return "\n".join(lines) + "\n"


MACRO_RE = re.compile(r"^\\(?:re)?newcommand\{\\(\w+)\}\{")


def read_macros(path):
    """{name: value} for every \\newcommand in a macro file (values may contain braces)."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = MACRO_RE.match(line)
        if not m:
            continue
        depth, value = 1, []
        for ch in line[m.end():]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            value.append(ch)
        out[m.group(1)] = "".join(value)
    return out


def main():
    ap = argparse.ArgumentParser(description="Regenerate every number quoted in the paper as LaTeX macros.")
    ap.add_argument("--check", action="store_true", help="compare the generated file with paper_numbers/numbers.tex (exit 1 on any difference)")
    ap.add_argument("--write-paper-copy", action="store_true", help="also write the generated file to paper_numbers/numbers.tex")
    args = ap.parse_args()

    text = render()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(text)
    print(f"Wrote {len(NAMES)} macros -> {OUT_FILE}")
    if args.write_paper_copy:
        PAPER_COPY.parent.mkdir(parents=True, exist_ok=True)
        PAPER_COPY.write_text(text)
        print(f"Wrote {PAPER_COPY}")
    if args.check:
        if not PAPER_COPY.exists():
            sys.exit(f"{PAPER_COPY} does not exist; nothing to check against.")
        mine, paper = read_macros(OUT_FILE), read_macros(PAPER_COPY)
        diffs = [(k, paper.get(k), mine.get(k)) for k in sorted(set(mine) | set(paper)) if mine.get(k) != paper.get(k)]
        for name, old, new in diffs:
            print(f"  {name}: paper_numbers = {old!r}, regenerated = {new!r}")
        if diffs:
            sys.exit(f"{len(diffs)} macro(s) differ from {PAPER_COPY}")
        print(f"All {len(mine)} macros match {PAPER_COPY}")


if __name__ == "__main__":
    main()
