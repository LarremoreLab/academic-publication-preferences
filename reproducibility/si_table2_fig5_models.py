"""SI Table S2: regression results behind Fig. 5 (label stab:top5_regression_results).

For Academia (pooled) and each of the 13 fields, the table reports, for both reference groups
(respondent's own top 5 and the field's top 5): the prestige slope, the predicted fraction of
top-5 venues in the publication record at prestige decile 10, and the own-minus-field
difference at decile 10, each with a 95% confidence interval from standard errors clustered
by institution. N is the number of respondents in the Fig. 5 sample and Inst. the number of
institutions. Rows are shaded grey/white alternating per field.

Input   output/results/fig5_models.csv  (written by fig5.py; run that script first)
Output  output/tables/si_table2.tex     (requires booktabs, multirow and colortbl/xcolor with
                                         \\rowcolor; the \\multirow{-2} cell is placed in the
                                         second row of each field so the row colour does not
                                         paint over it)

Usage (from the repository root): .venv/bin/python reproducibility/si_table2_fig5_models.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

POOLED = "Academia"
REF_LABELS = {"own": "Respondent's Top 5", "field": "Field's Top 5"}


def cell(r, level, quantity, ref):
    x = r.loc[(level, quantity, ref)]
    return f"{x.estimate:.3f} ({x.ci_lower:.3f}, {x.ci_upper:.3f})"


def make_table(res):
    r = res.set_index(["level", "quantity", "ref"])
    levels = [POOLED] + sorted(l for l in res["level"].unique() if l != POOLED)
    L = [r"\begin{table}[htbp]", r"\footnotesize", r"\centering",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrlccc}", r"\toprule",
         r"\textbf{Field} & \textbf{N} & \textbf{Inst.} & \textbf{Reference group} & \textbf{Slope} & "
         r"\textbf{Prestige = 10} & \textbf{Own $-$ field at 10} \\",
         r" & & & & \textbf{(95\% CI)} & \textbf{(95\% CI)} & \textbf{(95\% CI)} \\",
         r"\midrule"]
    for i, level in enumerate(levels):
        d = r.loc[(level, "diff10", "own-field")]
        shade = [r"\rowcolor{gray!20}"] if i % 2 == 0 else []
        L += shade + [f"{level} & {int(d.n_respondents)} & {int(d.n_clusters)} & {REF_LABELS['own']} & "
                      f"{cell(r, level, 'slope', 'own')} & {cell(r, level, 'pred10', 'own')} & \\\\"]
        L += shade + [f" & & & {REF_LABELS['field']} & {cell(r, level, 'slope', 'field')} & "
                      f"{cell(r, level, 'pred10', 'field')} & "
                      f"\\multirow{{-2}}{{*}}{{{cell(r, level, 'diff10', 'own-field')}}} \\\\"]
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Linear regression results predicting the fraction of top-5 venues in "
          r"publication records as a function of institutional prestige decile.} \normalfont{"
          r"Each respondent contributes two observations: the fraction of their own top-5 venues "
          r"(from their individual ranking), and the fraction of their field's top-5 venues (the "
          r"five venues appearing most often in respondents' individual top-5 lists, as in Fig.~2), "
          r"found in their publication record. For each field, we fit an ordinary least squares "
          r"regression of this fraction on reference group, prestige decile, and their interaction. "
          r"For Academia, the model additionally includes field intercepts for each reference group, "
          r"so the Academia slopes are within-field slopes (the average association between prestige "
          r"and top-5 fraction within fields), and the Academia prediction at prestige decile 10 "
          r"averages over respondents' fields. Slopes give the change in the fraction per "
          r"one-decile increase in prestige; ``Prestige = 10'' gives the predicted fraction at the "
          r"highest prestige decile. The last column is the difference between the two reference "
          r"groups at decile 10; its significance (Benjamini--Hochberg adjusted across the 13 "
          r"fields, $\alpha=0.01$) is indicated in Fig.~5B. Confidence intervals use standard errors "
          r"clustered by institution and $t$ reference distributions. "
          r"$N$ is the number of respondents with publication data, institutional prestige, and "
          r"an institution identifier; Inst.\ is the number of institutions. These results "
          r"correspond to Fig.~5 in the main text.}}",
          r"\label{stab:top5_regression_results}",
          r"\end{table}"]
    return "\n".join(L) + "\n"


def main():
    src = common.RESULTS / "fig5_models.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run reproducibility/fig5.py first.")
    out = common.TABLES / "si_table2.tex"
    out.write_text(make_table(pd.read_csv(src)))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
