"""Run every reproducibility script in order.

Usage (from the repository root):
    .venv/bin/python reproducibility/run_all.py                  # everything except the bootstrap
    .venv/bin/python reproducibility/run_all.py --skip-rankings  # reuse reproducibility/derived/
    .venv/bin/python reproducibility/run_all.py --bootstrap      # also rerun the respondent bootstrap

Stages
  1. compute_rankings.py          SpringRank rankings -> reproducibility/derived/   (~10-15 min)
  2. main-text figures            fig1.py ... fig5.py
  3. supplementary figures        si_fig1_subfields.py ... si_fig7_jif_ranks.py
  4. supplementary tables         si_table1_subfields.py ... si_tables8_10_composition.py
  5. supplementary numbers        si_inference_numbers.py, si_prestige_measures.py
  6. compute_numbers.py           every number quoted in the paper -> reproducibility/output/numbers.tex

The respondent bootstrap behind SI Fig. S2 (bootstrap_consensus.py) refits 1,000 x 13 leave-one-out
SpringRank models and takes ~15 min on 16 cores (hours on a laptop); its summary is committed in
reproducibility/cache/, so it is skipped unless --bootstrap is given. The permutation test behind
SI Fig. S5 (a few seconds) always runs.

All outputs are written under reproducibility/output/ (figures/, tables/, results/, numbers.tex).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

RANKING_SCRIPT = HERE / "compute_rankings.py"
BOOTSTRAP_SCRIPT = HERE / "bootstrap_consensus.py"

SCRIPTS = [
    # main-text figures
    ("Fig. 1", "fig1.py"),
    ("Fig. 2", "fig2.py"),
    ("Fig. 3 (13 HTML network panels)", "fig3.py"),
    ("Fig. 4", "fig4.py"),
    ("Fig. 5", "fig5.py"),
    # supplementary tables that other scripts read
    ("SI Table S1 (field vs subfield consensus)", "si_table1_subfields.py"),
    ("SI subfield sensitivity (catch-all answers as subfields)", "si_table1_subfields.py --include-catch-alls"),
    ("SI Table S2 (Fig. 5 regression table)", "si_table2_fig5_models.py"),
    ("SI Table S3 (alternative citation metrics)", "si_table3_alt_metrics.py"),
    ("SI Tables S4, S5 (preference vs JIF ranks)", "si_tables4_5_jif_ranks.py"),
    ("SI Tables S8-S10 (sample composition)", "si_tables8_10_composition.py"),
    # supplementary figures
    ("SI Fig. S1 (field vs subfield consensus)", "si_fig1_subfields.py"),
    ("SI Fig. S2 (bootstrap intervals; from cache)", "si_fig2_bootstrap.py"),
    ("SI Fig. S4 (full Eq. S1 estimates)", "si_fig4_regressions.py"),
    ("SI Fig. S5 (permutation test)", "si_fig5_permutation.py"),
    ("SI Fig. S6 (JIF vs consensus accuracy)", "si_fig6_jif_accuracy.py"),
    ("SI Fig. S7 (JIF vs consensus ranks)", "si_fig7_jif_ranks.py"),
    # supplementary numbers
    ("SI inference checks", "si_inference_numbers.py"),
    ("SI prestige-measure robustness", "si_prestige_measures.py"),
    # every number in the paper
    ("numbers.tex", "compute_numbers.py"),
]

REQUIRED_PUBLIC = ["respondents.csv", "comparisons.csv", "venue_selections.csv", "venues.csv",
                   "publication_counts.csv", "field_jif_rankings.csv", "field_top5.csv", "aarc_field_margins.csv"]
DERIVED_FILES = ["individual_rankings.csv", "field_rankings_per_user.csv",
                 "field_consensus_rankings.csv", "global_rankings_per_user.csv"]


def run(script: Path, label: str, extra=()) -> bool:
    print(f"\n{'=' * 60}\n  {label}\n  {script.relative_to(ROOT)}\n{'=' * 60}")
    t0 = time.time()
    result = subprocess.run([PYTHON, str(script), *extra], cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n[FAILED] {label} (exit code {result.returncode})")
        return False
    print(f"\n[OK] {label} in {time.time() - t0:.1f}s")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-rankings", action="store_true", help="do not rerun compute_rankings.py")
    ap.add_argument("--bootstrap", action="store_true", help="rerun bootstrap_consensus.py (slow)")
    ap.add_argument("--replicates", type=int, default=1000, help="bootstrap replicates (with --bootstrap)")
    args = ap.parse_args()

    missing = [f for f in REQUIRED_PUBLIC if not (ROOT / "public_data" / f).exists()]
    if missing:
        sys.exit(f"Missing public data files: {missing}")

    t_start = time.time()
    derived_ok = all((HERE / "derived" / f).exists() for f in DERIVED_FILES)
    if args.skip_rankings and derived_ok:
        print("Skipping compute_rankings.py (reproducibility/derived/ present)")
    elif not run(RANKING_SCRIPT, "Compute SpringRank rankings"):
        sys.exit(1)

    if args.bootstrap and not run(BOOTSTRAP_SCRIPT, "Respondent bootstrap", ["--replicates", str(args.replicates)]):
        sys.exit(1)

    failed = []
    for label, entry in SCRIPTS:
        name, *extra = entry.split()
        if not run(HERE / name, label, extra):
            failed.append(label)
    print(f"\n{'=' * 60}\n  Total time: {(time.time() - t_start) / 60:.1f} min")
    if failed:
        print("  Failed:", ", ".join(failed))
        sys.exit(1)
    print(f"  All outputs are in {HERE / 'output'}\n{'=' * 60}")


if __name__ == "__main__":
    main()
