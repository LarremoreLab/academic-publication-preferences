"""
Are low-consensus fields really bundles of high-consensus subfields?

For every field we measure consensus twice: once at the FIELD scale and once
at the scale of the respondent's own SUBFIELD, using the same subfield
divisions throughout. The difference between the two ("gain", signed so that
positive means the subfield is more consensual) is the quantity of interest.

  gain >> 0   the field is a bundle of internally-agreeing subfields
  gain ~= 0   subfields look like the field
              -- with HIGH field consensus this is a unified field
              -- with LOW  field consensus this is a diffuse field
  gain <  0   subdividing actively hurts; the field already describes
              everyone better than any local subfield does

Three consensus measures, each the paper's field-level measure applied at two
scales. Field and subfield values are always computed over the SAME
respondents: those who belong to a subfield large enough to score.

  TOP-5 AGREEMENT   (Fig. 2)  percentage of a respondent's individual top-5
                    that appears in the group's top-5. See below.
  ACCUMULATION      (Fig. 1A) expected number of distinct venues in N_ACC
                    consideration sets drawn from the group without
                    replacement. Fig. 1 uses N=100 over the whole field; the
                    smallest scorable subfield has 12 respondents, so here
                    N_ACC = 12 at both scales. Computed exactly by rarefaction,
                    sum_v [1 - C(n-k_v, N)/C(n, N)] with k_v the number of
                    members considering venue v, rather than by Monte Carlo as
                    in fig1.py (the two agree to ~1 venue). Fewer venues =
                    more consensus, so gain = field - subfield. Averaged over
                    subfields weighted by size.
  OVERLAP           (Fig. 1B) for each venue in a respondent's consideration
                    set, the fraction of the OTHER members of the group who
                    also consider it; averaged over all respondent-venue
                    pairs pooled, exactly as fig1.py does (so respondents with
                    larger consideration sets count more). Leave-one-out by
                    construction. gain = subfield - field. The whole-field
                    value (ovl_field_full) reproduces the Fig. 1B diagonal.

The top-5 measure is the paper's Fig. 2 top-5 agreement (reproducibility/fig2.py),
applied at two scales. A group's top-5 is a VOTE COUNT: every member's
individual top-5 casts five votes and the group's top-5 are the five most-voted
venues. No group-level SpringRank is fit, so there is no regularisation choice
to make at the subfield scale. The one difference from Fig. 2 is that the
group top-5 is computed leave-one-out (the respondent's own votes are removed
before they are scored); the non-leave-one-out field values are printed as a
check and match Fig. 2 (numbers.tex \<Field>TopFiveAgreement) to 0.1 pp. The
one visible difference is Computer science, 16.3 here vs 16.4 there: one CS
respondent ranked only four venues, and Fig. 2 divides by votes cast (1094)
where this script divides by 5 per respondent (1095).

Subfields are the ones respondents selected during the survey. Catch-all
answers ("Other" etc.) are not treated as subfields. A field with fewer than
MIN_UNITS subfields of at least MIN_GROUP respondents is reported as
non-decomposable: its subfield value is set equal to its field value and its
gain to zero. That is a statement about the labels, not about the field.

With --include-catch-alls the catch-all answers count as subfields whenever
they reach MIN_GROUP respondents (every other rule unchanged). This is the
sensitivity check quoted in the SI: a catch-all mixes specialties and behaves
like a random subsample of its field, so counting it pulls the gains toward
zero. In that mode the results go to subfield_consensus_catchall.csv and
subfield_correlations_catchall.csv and no LaTeX table is written.

Definitions
  individual top-5   a respondent's five highest-ranked venues (individual
                     SpringRank, alpha=0; reproducibility/derived/
                     individual_rankings.csv)
  group top-5        the five venues appearing most often in the individual
                     top-5s of that group's members, computed LEAVE-ONE-OUT so
                     a respondent never helps define the set they are scored
                     against. Ties in the vote count are broken by venue id so
                     the result is deterministic.
  agreement          |individual top-5 & group top-5| / 5, as a percentage
  averaging          over respondents, so each field's value is weighted by
                     subfield size (a size-weighted mean of subfield means).
                     Field and subfield values are averaged over the SAME
                     respondents: those in a subfield large enough to score.

Inputs   public_data/respondents.csv       (field, subfield)
         public_data/venue_selections.csv  (consideration sets)
         reproducibility/derived/individual_rankings.csv
Outputs  reproducibility/output/tables/si_table1.tex          (SI Table S1)
         reproducibility/output/results/subfield_consensus.csv
         reproducibility/output/results/subfield_correlations.csv (field-vs-subfield and
                                                  between-measure correlations
                                                  over the decomposable fields)
         si_fig1_subfields.py reads subfield_consensus.csv to draw SI Fig. S1.
         With --include-catch-alls: subfield_consensus_catchall.csv and
         subfield_correlations_catchall.csv (read by compute_numbers.py).

Usage    cd <repo_root>
         .venv/bin/python reproducibility/compute_rankings.py  # only needed once
         .venv/bin/python reproducibility/si_table1_subfields.py
         .venv/bin/python reproducibility/si_table1_subfields.py --include-catch-alls
"""
import argparse
from collections import Counter
from math import comb
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MIN_GROUP = 12          # smallest subfield we will score
MIN_UNITS = 2           # fewer than this -> non-decomposable
TOP_K = 5
N_ACC = 12              # consideration sets drawn for the accumulation measure
CATCH_ALL_ANSWERS = {"Other", "Social science", "General", "Miscellaneous"}

parser = argparse.ArgumentParser(description="SI Table S1: consensus at the field vs subfield scale.")
parser.add_argument("--include-catch-alls", action="store_true",
                    help="treat the catch-all answers (%s) as subfields when they reach MIN_GROUP respondents; "
                         "writes subfield_consensus_catchall.csv and subfield_correlations_catchall.csv, no LaTeX table"
                         % ", ".join(sorted(CATCH_ALL_ANSWERS)))
ARGS = parser.parse_args()
CATCH_ALL = set() if ARGS.include_catch_alls else CATCH_ALL_ANSWERS
SUFFIX = "_catchall" if ARGS.include_catch_alls else ""
FIG1_REF = "Fig.~1"     # replace with \ref{...} when merging
FIG2_REF = "Fig.~2"

for p in [ROOT / "public_data/respondents.csv",
          ROOT / "public_data/venue_selections.csv",
          ROOT / "reproducibility/derived/individual_rankings.csv"]:
    if not p.exists():
        sys.exit(f"Missing required input: {p}")

resp = pd.read_csv(ROOT / "public_data/respondents.csv")
ind = pd.read_csv(ROOT / "reproducibility/derived/individual_rankings.csv")
sel = pd.read_csv(ROOT / "public_data/venue_selections.csv")
sel = sel[sel.would_publish == True]

# individual top-5: the five lowest ordinal ranks (0..4), as in fig2.py
own_top5 = {u: set(g.loc[g.ordinal_rank < TOP_K, "venue_db_id"])
            for u, g in ind.groupby("user_db_id")}
# consideration sets
cons = {u: set(g.venue_db_id) for u, g in sel.groupby("user_db_id")}


def accumulation(users, N=N_ACC):
    """Expected # distinct venues in N consideration sets drawn w/o replacement."""
    n = len(users)
    if n < N:
        return np.nan
    k = Counter(v for u in users for v in cons.get(u, ()))
    return sum(1 - comb(n - kv, N) / comb(n, N) for kv in k.values())


def overlap(u, users, counts):
    """For each venue u considers, the share of the OTHER members considering it.
    Returns one value per venue; fig1.py pools these across respondents before
    averaging, so respondents with larger consideration sets count more."""
    return [100 * (counts[v] - 1) / (len(users) - 1) for v in cons.get(u, ())]


def votes(users):
    """Vote count over the individual top-5s of a set of respondents."""
    return Counter(v for u in users for v in own_top5.get(u, ()))


def group_top5(counts, exclude_venues):
    """Most-voted venues in a group, with the focal respondent's votes removed."""
    c = Counter(counts)
    for v in exclude_venues:
        c[v] -= 1
    ranked = sorted(((v, n) for v, n in c.items() if n > 0), key=lambda x: (-x[1], x[0]))
    return {v for v, _ in ranked[:TOP_K]}


def agreement(u, gt5):
    return len(own_top5[u] & gt5) / TOP_K


rows = []
for field, fu in resp.groupby("field"):
    members = [u for u in fu.user_db_id if u in own_top5]
    field_counts = votes(members)
    n_field = fu.user_db_id.nunique()

    # non-leave-one-out field agreement: must reproduce Fig. 2
    full_top5 = group_top5(field_counts, ())
    fig2 = 100 * np.mean([agreement(u, full_top5) for u in members])

    groups = {}
    for k, g in fu.groupby("subfield"):
        ids = [u for u in g.user_db_id if u in own_top5]
        if len(ids) >= MIN_GROUP and k not in CATCH_ALL:
            groups[k] = ids
    cov = 100 * sum(len(v) for v in groups.values()) / n_field

    # whole-field reference values (all members), for the record
    all_counts = Counter(v for u in members for v in cons.get(u, ()))
    field_full = {
        "top5_field_full": 100 * np.mean(
            [agreement(u, group_top5(field_counts, own_top5[u])) for u in members]),
        "acc_field_full": accumulation(members),
        "ovl_field_full": np.mean([x for u in members for x in overlap(u, members, all_counts)]),
    }

    if len(groups) < MIN_UNITS:
        # non-decomposable: too few subfields large enough to score
        rows.append({"field": field, "N": n_field, "fig2_field_agreement": fig2,
                     "units": len(groups), "cov": cov, "ok": False,
                     "top5_field": field_full["top5_field_full"],
                     "top5_sub": field_full["top5_field_full"], "top5_gain": 0.0,
                     "acc_field": field_full["acc_field_full"],
                     "acc_sub": field_full["acc_field_full"], "acc_gain": 0.0,
                     "ovl_field": field_full["ovl_field_full"],
                     "ovl_sub": field_full["ovl_field_full"], "ovl_gain": 0.0,
                     **field_full})
        continue

    covered = [u for ids in groups.values() for u in ids]
    weights = np.array([len(ids) for ids in groups.values()], dtype=float)

    # ── top-5 agreement (per respondent, LOO group top-5) ────────────────────
    gcounts = {k: votes(ids) for k, ids in groups.items()}
    t_field, t_sub = [], []
    for k, ids in groups.items():
        for u in ids:
            own = own_top5[u]
            t_field.append(agreement(u, group_top5(field_counts, own)))
            t_sub.append(agreement(u, group_top5(gcounts[k], own)))

    # ── accumulation at N_ACC (group property; size-weighted mean over subfields)
    acc_field = accumulation(covered)
    acc_sub = float(np.average([accumulation(ids) for ids in groups.values()],
                               weights=weights))

    # ── overlap (pooled over respondent-venue pairs as in fig1.py; LOO) ──────
    cov_counts = Counter(v for u in covered for v in cons.get(u, ()))
    o_field, o_sub = [], []
    for k, ids in groups.items():
        sub_counts = Counter(v for u in ids for v in cons.get(u, ()))
        for u in ids:
            o_field += overlap(u, covered, cov_counts)
            o_sub += overlap(u, ids, sub_counts)

    rows.append({"field": field, "N": n_field, "fig2_field_agreement": fig2,
                 "units": len(groups), "cov": cov, "ok": True,
                 "top5_field": 100 * np.mean(t_field), "top5_sub": 100 * np.mean(t_sub),
                 "top5_gain": 100 * (np.mean(t_sub) - np.mean(t_field)),
                 "acc_field": acc_field, "acc_sub": acc_sub,
                 "acc_gain": acc_field - acc_sub,          # fewer venues = more consensus
                 "ovl_field": np.mean(o_field), "ovl_sub": np.mean(o_sub),
                 "ovl_gain": np.mean(o_sub) - np.mean(o_field),
                 **field_full})

d = pd.DataFrame(rows).sort_values("top5_gain", ascending=False)
d.to_csv(common.RESULTS / f"subfield_consensus{SUFFIX}.csv", index=False)

pd.set_option("display.width", 250)
show = ["field", "N", "units", "cov",
        "acc_field", "acc_sub", "acc_gain",
        "ovl_field", "ovl_sub", "ovl_gain",
        "top5_field", "top5_sub", "top5_gain"]
print("Consensus at the field vs subfield scale (weighted by subfield size)\n")
print(d[show].to_string(index=False, float_format=lambda x: f"{x:.1f}"))
nd = [r.field for _, r in d.iterrows() if not r.ok]
print("\nnon-decomposable:", ", ".join(nd) or "none")
print(f"\naccumulation uses N_ACC = {N_ACC} (must not exceed MIN_GROUP = {MIN_GROUP})")
print("\nfig2_field_agreement is the non-leave-one-out, whole-field top-5 value and "
      "should match numbers.tex \\<Field>TopFiveAgreement.")

# ── Correlations (decomposable fields only) ──────────────────────────────────
# (a) field-scale vs subfield-scale value of each measure: how much does the
#     ordering of fields change when we move to the subfield scale?
# (b) mutual correlations of the three subfield-scale values: do the measures
#     agree about which fields have consensual subfields? Accumulation is
#     reversed (fewer venues = more consensus) so its correlations with the
#     other two are expected to be negative.
from scipy.stats import pearsonr, spearmanr

dd = d[d.ok]
MEASURES = [("acc", "Venues in %d sets" % N_ACC), ("ovl", "Mean overlap"),
            ("top5", "Top-five agreement")]
corr_rows = []
for m, name in MEASURES:
    x, y = dd[f"{m}_field"], dd[f"{m}_sub"]
    corr_rows.append({"comparison": f"{name}: field vs subfield", "n": len(dd),
                      "spearman_rho": spearmanr(x, y)[0], "spearman_p": spearmanr(x, y)[1],
                      "pearson_r": pearsonr(x, y)[0], "pearson_p": pearsonr(x, y)[1]})
for i in range(3):
    for j in range(i + 1, 3):
        (m1, n1), (m2, n2) = MEASURES[i], MEASURES[j]
        x, y = dd[f"{m1}_sub"], dd[f"{m2}_sub"]
        corr_rows.append({"comparison": f"subfield: {n1} vs {n2}", "n": len(dd),
                          "spearman_rho": spearmanr(x, y)[0], "spearman_p": spearmanr(x, y)[1],
                          "pearson_r": pearsonr(x, y)[0], "pearson_p": pearsonr(x, y)[1]})
corr = pd.DataFrame(corr_rows)
corr.to_csv(common.RESULTS / f"subfield_correlations{SUFFIX}.csv", index=False)
print(f"\nCorrelations over the {len(dd)} decomposable fields\n")
print(corr.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

if ARGS.include_catch_alls:
    print(f"\nwrote {common.RESULTS/'subfield_consensus_catchall.csv'} and "
          f"{common.RESULTS/'subfield_correlations_catchall.csv'} (catch-all answers counted as subfields; no LaTeX table)")
    sys.exit(0)

# ── LaTeX ────────────────────────────────────────────────────────────────────
L = ["% Requires: \\usepackage{booktabs}",
     "\\begin{table}[t]", "\\centering", "\\small",
     "\\setlength{\\tabcolsep}{3.5pt}",
     "\\begin{tabular}{l r r rrr rrr rrr}", "\\toprule",
     " & & & \\multicolumn{3}{c}{Venues in " + str(N_ACC) + " sets} "
     "& \\multicolumn{3}{c}{Mean overlap (\\%)} "
     "& \\multicolumn{3}{c}{Top-five agreement (\\%)} \\\\",
     "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}",
     "Field & $N$ & subfields & field & subfield & gain "
     "& field & subfield & gain & field & subfield & gain \\\\", "\\midrule"]
for _, r in d.iterrows():
    cells = [r.field.replace("Computer science", "Computer sci."), f"{int(r.N)}",
             f"{int(r.units)}" if r.ok else "--$^{\\dagger}$"]
    for m in ["acc", "ovl", "top5"]:
        cells += [f"{r[f'{m}_field']:.1f}", f"{r[f'{m}_sub']:.1f}",
                  f"{r[f'{m}_gain']:+.1f}" if r.ok else "0.0"]
    L.append(" & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}",
      "\\caption{\\textbf{Consensus at the field scale compared with consensus at "
      "the scale of a respondent's own subfield.} Each of the paper's three "
      "consensus measures is computed twice for every field: over the field, and "
      "over the respondent's own subfield, with the gain signed so that a positive "
      "value means the subfield is more consensual than the field. "
      "\\emph{Venues in " + str(N_ACC) + " sets}: the "
      "expected number of distinct venues in " + str(N_ACC) + " consideration sets "
      "drawn from the group, as in " + FIG1_REF + "A but with $N=" + str(N_ACC)
      + "$ rather than $100$ so that every subfield can be evaluated; fewer venues "
      "means more consensus, so the gain is field minus subfield. \\emph{Mean "
      "overlap}: for each venue a respondent would consider, the percentage of the "
      "group's other members who would also consider it, averaged over all such "
      "venues of all respondents, as in " + FIG1_REF + "B. "
      "\\emph{Top-five agreement}: the percentage of a respondent's five most "
      "preferred venues that also appear in the group's top five, where a group's "
      "top five are the five venues appearing most often in the individual top "
      "fives of its members, as in " + FIG2_REF + ", computed without the "
      "respondent being scored. Subfields are those respondents "
      "selected during the survey; field and subfield values are computed over the "
      "same respondents, those who belong to a subfield of at least "
      + str(MIN_GROUP) + " respondents, and are averaged over respondents so that "
      "larger subfields count proportionally more. $^{\\dagger}$Fewer than two "
      "such subfields are available, so the field cannot be subdivided; its values "
      "are computed over all its respondents and its gains are zero.}",
      "\\label{tab:subfields}", "\\end{table}"]
(common.TABLES / "si_table1.tex").write_text("\n".join(L))
print(f"\nwrote {common.RESULTS/'subfield_consensus.csv'}, {common.RESULTS/'subfield_correlations.csv'} "
      f"and {common.TABLES/'si_table1.tex'}")
