"""SI Table S3: the consensus ranking's advantage over the 2-year JIF generalizes to other
citation metrics.

For each field, four citation metrics are scored on the respondents' non-tie pairwise venue
comparisons: the venue with the higher metric value is predicted to be preferred. Each metric
is scored on every comparison for which both venues carry that metric and the two values
differ (a tied metric makes no prediction), and the leave-one-out field consensus ranking
(SpringRank with the respondent's own responses held out) is re-scored on the identical
subset, so that each "advantage" is a paired contrast on the same comparisons. Comparisons
in which the two venues' consensus scores are exactly equal are left out as well.

Metrics (public_data/venues.csv)
  jif_2yr       jcr_jif                2-year Journal Impact Factor (JCR 2023)
  jif_5yr       jcr_jif_5yr            5-year Journal Impact Factor (JCR 2023)
  ais           jcr_article_influence  Article Influence Score (JCR 2023)
  oa_citedness  2yr_mean_citedness     OpenAlex 2-year mean citedness (also covers conferences)

Inputs
  public_data/comparisons.csv, public_data/venues.csv,
  reproducibility/derived/field_rankings_per_user.csv

Outputs
  output/tables/si_table3.tex             Table S3 (label tab:alt_metrics); a grey cell marks an
                                          advantage <= 0, i.e. a metric matching or beating the
                                          survey ranking
  output/results/alt_metrics_accuracy.csv per field and pooled over all fields: for each metric
                                          the accuracy (acc_*, %), the survey ranking's advantage
                                          (adv_*, percentage points) and the number of
                                          comparisons scored (n_*)

Run from the project root:  .venv/bin/python reproducibility/si_table3_alt_metrics.py
(compute_rankings.py must have been run first).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

METRICS = [("jif_2yr", "jcr_jif", "2-yr JIF"),
           ("jif_5yr", "jcr_jif_5yr", "5-yr JIF"),
           ("ais", "jcr_article_influence", "AIS"),
           ("oa_citedness", "2yr_mean_citedness", "OA cited.")]

common.require_derived()
comp = pd.read_csv(common.PUBLIC / "comparisons.csv")
venues = pd.read_csv(common.PUBLIC / "venues.csv").set_index("venue_db_id")
fr = pd.read_csv(common.DERIVED / "field_rankings_per_user.csv")

# ── Prediction flags: True/False where a ranking makes a prediction, NaN where it makes none ──
comp = comp[comp["is_tie"] == False].copy()  # noqa: E712
for side, col in (("pref_venue_db_id", "p"), ("other_venue_db_id", "o")):
    scores = fr.rename(columns={"venue_db_id": side, "score": f"f_{col}"})
    comp = comp.merge(scores, on=["user_db_id", side], how="left")


def prediction(a, b):
    """True if the preferred venue has the higher value, False if lower, NaN if missing or tied."""
    pred = pd.Series(pd.NA, index=a.index, dtype="boolean")
    ok = a.notna() & b.notna() & (a != b)
    pred[ok] = a[ok] > b[ok]
    return pred


flags = pd.DataFrame({"field": comp["field"], "consensus": prediction(comp["f_p"], comp["f_o"])})
for key, col, _ in METRICS:
    flags[key] = prediction(comp["pref_venue_db_id"].map(venues[col]), comp["other_venue_db_id"].map(venues[col]))


def score(sub):
    rec = {}
    for key, _, _ in METRICS:
        m = sub[sub[key].notna() & sub["consensus"].notna()]
        rec[f"acc_{key}"] = 100 * m[key].mean()
        rec[f"adv_{key}"] = 100 * (m["consensus"].mean() - m[key].mean())
        rec[f"n_{key}"] = len(m)
    return rec


rows = [{"field": f, **score(flags[flags["field"] == f])} for f in common.FIELDS]
rows.append({"field": "All fields", **score(flags)})
d = pd.DataFrame(rows)
d.to_csv(common.RESULTS / "alt_metrics_accuracy.csv", index=False)

adv_cols = [f"adv_{k}" for k, _, _ in METRICS]
n_wins = int((d.loc[d["field"] != "All fields", adv_cols] > 0).sum().sum())
n_cells = int(d.loc[d["field"] != "All fields", adv_cols].notna().sum().sum())
print(d.round(1).to_string(index=False))
print(f"\nSurvey ranking ahead in {n_wins} of {n_cells} field x metric combinations; exceptions: "
      + (", ".join(f"{r.field} / {c[4:]} ({r[c]:+.1f} pp)" for _, r in d.iterrows() for c in adv_cols
                   if r.field != "All fields" and r[c] <= 0) or "none"))

# ── LaTeX ─────────────────────────────────────────────────────────────────────────────────────
SHADE = "\\cellcolor{gray!15}"
L = ["% Requires \\usepackage{booktabs,colortbl,xcolor}",
     "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
     "\\begin{tabular}{l rrrr rrrr}", "\\toprule",
     " & \\multicolumn{4}{c}{Metric accuracy (\\%)} & \\multicolumn{4}{c}{Survey ranking advantage (pp)} \\\\",
     "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
     "Field & " + " & ".join([lab for _, _, lab in METRICS] * 2) + " \\\\", "\\midrule"]
for _, r in d.iterrows():
    pooled = r["field"] == "All fields"
    if pooled:
        L.append("\\midrule")
    cells = ["\\textbf{All fields}" if pooled else r["field"].replace("Computer science", "Computer sci.")]
    cells += [f"{r[f'acc_{k}']:.1f}" for k, _, _ in METRICS]
    cells += [(SHADE if r[f"adv_{k}"] <= 0 else "") + f"{r[f'adv_{k}']:+.1f}" for k, _, _ in METRICS]
    L.append(" & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}",
      "\\caption{\\textbf{Consensus rankings' advantage over JIF generalizes to other metrics.} "
      "The main text reports that rankings predict choices better than the 2-year JIF, raising the "
      "question of whether this result is particular to the 2-year JIF. Here, we generalize to three "
      "other metrics: the 5-year JIF, the Article Influence Score (AIS) which looks at 5y of citations "
      "and weights their influence by the prestige of the venue, and Open Alex 2-year mean citedness "
      "(OA cited.) which has superior coverage of conference proceedings. Columns show the accuracy "
      "with which each citation metric predicts respondents' pairwise venue preferences (left), and "
      "the percentage-point advantage of the leave-one-out survey consensus ranking evaluated on that "
      "metric's own comparison subset (right). Each metric is scored on every non-tie comparison for "
      "which both venues carry that metric, and the survey ranking is re-scored on the identical "
      "subset, thus making each advantage a paired contrast. The grey shaded cell marks the sole "
      "instance in which a citation metric matches or exceeds the survey ranking.}",
      "\\label{tab:alt_metrics}", "\\end{table}"]
out = common.TABLES / "si_table3.tex"
out.write_text("\n".join(L) + "\n")
print(f"Saved: {out}")
