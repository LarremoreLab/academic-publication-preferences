"""SI Fig. S6: how well the 2-year Journal Impact Factor predicts pairwise venue preferences,
field by field, compared with the field consensus ranking.

For every non-tie pairwise comparison in which both venues carry a 2-year JIF (JCR 2023) and
the two JIFs differ, two predictions are scored against the respondent's actual choice:
  * JIF            the venue with the higher 2-year JIF is preferred;
  * field consensus the venue scored higher by the respondent's field consensus ranking,
                   computed with that respondent's own responses held out (leave-one-out
                   SpringRank, reproducibility/derived/field_rankings_per_user.csv).
Comparisons in which both venues have the same JIF are excluded because a tied metric makes
no prediction. The "Academia" row pools all fields' comparisons (still scored with the field
leave-one-out rankings). Numeric annotations give the difference in accuracy, in percentage
points, between the field consensus ranking and the JIF.

Inputs
  public_data/comparisons.csv                           pairwise comparisons (is_tie, field)
  public_data/venues.csv                                jcr_jif, the 2-year JIF per venue
  reproducibility/derived/field_rankings_per_user.csv   leave-one-out field SpringRank scores

Outputs
  output/figures/si_fig6.pdf
  output/results/jif_accuracy_by_field.csv   per field (and Academia): n_comparisons (all
      comparisons incl. ties), n_with_jif (non-tie comparisons with both JIFs and both consensus
      scores), n_scored (after excluding equal-JIF pairs), field_acc, jif_acc (%), difference
      (pp), coverage (n_with_jif / n_comparisons, %; the main text's "comparisons for which both
      venues' 2-yr JIFs exist"), scored_pct (n_scored / n_comparisons, %)
  output/results/jif_accuracy_summary.csv    headline numbers quoted in the main text: share of
      venues with a JIF, comparison coverage, overall JIF and consensus accuracy and their gap,
      most/least predictive field, fields with the largest/smallest coverage, and the range of
      the consensus ranking's advantage across fields

Run from the project root:  .venv/bin/python reproducibility/si_fig6_jif_accuracy.py
(compute_rankings.py must have been run first).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

common.require_derived()
comp_all = pd.read_csv(common.PUBLIC / "comparisons.csv")
venues = pd.read_csv(common.PUBLIC / "venues.csv")
fr = pd.read_csv(common.DERIVED / "field_rankings_per_user.csv")

# ── Score each non-tie comparison with the JIF and the leave-one-out field consensus ─────────
jif = dict(zip(venues["venue_db_id"], venues["jcr_jif"]))
comp = comp_all[comp_all["is_tie"] == False].copy()  # noqa: E712
for side, col in (("pref_venue_db_id", "p"), ("other_venue_db_id", "o")):
    scores = fr.rename(columns={"venue_db_id": side, "score": f"f_{col}"})
    comp = comp.merge(scores, on=["user_db_id", side], how="left")
    comp[f"j_{col}"] = comp[side].map(jif)

with_jif = comp[comp["f_p"].notna() & comp["f_o"].notna() & comp["j_p"].notna() & comp["j_o"].notna()]
n_jif_ties = int((with_jif["j_p"] == with_jif["j_o"]).sum())
d = with_jif[with_jif["j_p"] != with_jif["j_o"]].copy()
d["field_acc"] = d["f_p"] > d["f_o"]
d["jif_acc"] = d["j_p"] > d["j_o"]

# ── Per-field summary and the pooled Academia row ─────────────────────────────────────────────
summary = 100 * d.groupby("field")[["field_acc", "jif_acc"]].mean()
summary["difference"] = summary["field_acc"] - summary["jif_acc"]
summary["n_comparisons"] = comp_all.groupby("field").size()
summary["n_with_jif"] = with_jif.groupby("field").size()
summary["n_scored"] = d.groupby("field").size()
acad_field, acad_jif = 100 * d["field_acc"].mean(), 100 * d["jif_acc"].mean()
summary.loc["Academia"] = [acad_field, acad_jif, acad_field - acad_jif, len(comp_all), len(with_jif), len(d)]
summary["coverage"] = 100 * summary["n_with_jif"] / summary["n_comparisons"]
summary["scored_pct"] = 100 * summary["n_scored"] / summary["n_comparisons"]
summary = summary[["n_comparisons", "n_with_jif", "n_scored", "field_acc", "jif_acc", "difference",
                   "coverage", "scored_pct"]]
summary[["n_comparisons", "n_with_jif", "n_scored"]] = summary[["n_comparisons", "n_with_jif", "n_scored"]].astype(int)
summary.rename_axis("field").to_csv(common.RESULTS / "jif_accuracy_by_field.csv")

# ── Plot ─────────────────────────────────────────────────────────────────────────────────────
plt.rcParams["font.size"] = 13
plt.rcParams["legend.fontsize"] = 13
s = summary.drop(index="Academia").sort_values("field_acc")
fig, ax = plt.subplots(figsize=(8, 6))
fields, ypos = s.index, range(len(s))
ax.scatter(s["field_acc"], ypos, color="k", label="Field consensus\nranking", s=100)
ax.scatter(s["jif_acc"], ypos, color="white", alpha=1, s=100, marker="s")
ax.scatter(s["jif_acc"], ypos, color="grey", label="JIF Ranking", alpha=0.7, s=100)

# Academia row at the top
x1, x0, y = acad_field, acad_jif, len(fields) + 0.5
ax.scatter(x1, y, color="k", s=100)
ax.scatter(x0, y, color="white", s=100, alpha=1)
ax.scatter(x0, y, color="grey", s=100, alpha=0.7)
ax.text((x1 + x0) / 2, y, f"{abs(x1 - x0):.2f}", color="black", ha="center", va="bottom")
ax.plot([x1, x0], [y, y], color="black", linestyle="-", alpha=0.5, zorder=-1)

# Per-field connecting lines and difference labels
for i, f in enumerate(fields):
    fa, ja = s.loc[f, "field_acc"], s.loc[f, "jif_acc"]
    ax.plot([fa, ja], [i, i], color="black", linestyle="-", alpha=0.5, zorder=-1)
    ax.text((fa + ja) / 2, i, f"{abs(fa - ja):.2f}", color="black", ha="center", va="bottom")

ax.set_yticks(list(ypos) + [len(fields) + 0.5])
ax.set_yticklabels(list(fields) + ["Academia"])
ax.set_ylabel("Field")
ax.set_xlabel("% of Pairwise Comparisons Correctly Predicted")
ax.legend(loc="lower right", frameon=False)
# White patch covering the gap between the field rows and the Academia row
plt.text(52.5, len(fields) - 0.25, "   ", fontsize=8, va="center", ha="left", color="black",
         backgroundcolor="white", zorder=10)
plt.tight_layout()
sns.despine()
out = common.FIGURES / "si_fig6.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")

# ── Headline numbers for the main text ───────────────────────────────────────────────────────
f = summary.drop(index="Academia")
numbers = {
    "venues_with_jif_pct": 100 * venues["jcr_jif"].notna().mean(),
    "comparisons_available_pct": 100 * len(with_jif) / len(comp_all),
    "comparisons_scored": len(d),
    "jif_ties_excluded": n_jif_ties,
    "jif_acc_all": acad_jif,
    "consensus_acc_all": acad_field,
    "gap_all": acad_field - acad_jif,
    "most_predictive_field": f["jif_acc"].idxmax(), "most_predictive_acc": f["jif_acc"].max(),
    "least_predictive_field": f["jif_acc"].idxmin(), "least_predictive_acc": f["jif_acc"].min(),
    "coverage_max_field": f["coverage"].idxmax(), "coverage_max_pct": f["coverage"].max(),
    "coverage_min_field": f["coverage"].idxmin(), "coverage_min_pct": f["coverage"].min(),
    "advantage_min_field": f["difference"].idxmin(), "advantage_min_pp": f["difference"].min(),
    "advantage_max_field": f["difference"].idxmax(), "advantage_max_pp": f["difference"].max(),
}
pd.Series(numbers, name="value").rename_axis("key").to_csv(common.RESULTS / "jif_accuracy_summary.csv")

print(f"Comparisons: {len(comp_all):,} total; {len(comp):,} non-tie; {len(with_jif):,} with both JIFs and both "
      f"consensus scores ({numbers['comparisons_available_pct']:.1f}% of all); {n_jif_ties:,} equal-JIF pairs "
      f"excluded; {len(d):,} scored.")
print(f"Venues with a 2-yr JIF: {venues['jcr_jif'].notna().sum()} of {len(venues)} ({numbers['venues_with_jif_pct']:.1f}%)")
print(f"Academia: JIF {acad_jif:.2f}%, field consensus {acad_field:.2f}%, gap {acad_field - acad_jif:.2f} pp")
print(f"JIF most predictive in {numbers['most_predictive_field']} ({numbers['most_predictive_acc']:.1f}%), least in "
      f"{numbers['least_predictive_field']} ({numbers['least_predictive_acc']:.1f}%)")
print(f"Coverage: max {numbers['coverage_max_field']} ({numbers['coverage_max_pct']:.1f}%), min "
      f"{numbers['coverage_min_field']} ({numbers['coverage_min_pct']:.1f}%)")
print(f"Consensus advantage: {numbers['advantage_min_pp']:.2f} pp ({numbers['advantage_min_field']}) to "
      f"{numbers['advantage_max_pp']:.2f} pp ({numbers['advantage_max_field']})")
print(summary.round(2).to_string())
