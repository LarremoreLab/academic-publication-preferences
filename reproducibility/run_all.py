"""Run all reproducibility scripts in order.

Usage (from the project root):
    .venv/bin/python reproducibility/run_all.py

This script:
  1. Runs compute_rankings.py to produce reproducibility/derived/ from public_data/.
     Skip with --skip-rankings if derived/ already exists and is up to date.
  2. Runs all figure scripts (fig1–fig5, si_fig1–si_fig4) in order.

Expected total runtime: ~15–25 minutes.
  - compute_rankings.py : ~10–15 min (SpringRank, leave-one-out × 3,508 users)
  - fig1–fig5, si_fig1–si_fig3: < 1 min each
  - si_fig4: ~3 min (10,000 permutations × 4 combinations)

Outputs: one PDF per script, written to reproducibility/.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent

# ── Resolve Python interpreter ────────────────────────────────────────────────
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# ── Script execution order ────────────────────────────────────────────────────
RANKING_SCRIPT = HERE / "compute_rankings.py"

FIGURE_SCRIPTS = [
    HERE / "fig1.py",
    HERE / "fig2.py",
    HERE / "fig3.py",
    HERE / "fig4.py",
    HERE / "fig5.py",
    HERE / "si_fig1.py",
    HERE / "si_fig2.py",
    HERE / "si_fig3.py",
    HERE / "si_fig4.py",   # ~3 min
]


def run(script: Path, label: str) -> bool:
    """Run a script, print timing, return True on success."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {script.relative_to(ROOT)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([PYTHON, str(script)], cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAILED] {label} (exit code {result.returncode})")
        return False
    print(f"\n[OK] {label} completed in {elapsed:.1f}s")
    return True


def check_public_data():
    """Verify that public_data/ exists and has expected files."""
    pub = ROOT / "public_data"
    required = [
        "respondents.csv",
        "comparisons.csv",
        "venue_selections.csv",
        "venues.csv",
        "publication_counts.csv",
        "field_jif_rankings.csv",
    ]
    missing = [f for f in required if not (pub / f).exists()]
    if missing:
        print("ERROR: The following public_data/ files are missing:")
        for f in missing:
            print(f"  public_data/{f}")
        print("\nMake sure you have cloned the full repository, including public_data/.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--skip-rankings", action="store_true",
        help="Skip compute_rankings.py (use if derived/ already exists)",
    )
    args = parser.parse_args()

    print(f"Python: {PYTHON}")
    print(f"Root:   {ROOT}")

    # Check public_data/ is populated
    check_public_data()

    overall_start = time.time()
    failures = []

    # ── Step 1: Rankings ──────────────────────────────────────────────────────
    if args.skip_rankings:
        print("\n[Skipping compute_rankings.py  (--skip-rankings)]")
        derived = HERE / "derived"
        needed = ["individual_rankings.csv", "field_rankings_per_user.csv",
                  "field_consensus_rankings.csv", "global_rankings_per_user.csv"]
        missing = [f for f in needed if not (derived / f).exists()]
        if missing:
            print("ERROR: --skip-rankings was set but these derived files are missing:")
            for f in missing:
                print(f"  reproducibility/derived/{f}")
            print("Run without --skip-rankings to generate them.")
            sys.exit(1)
    else:
        ok = run(RANKING_SCRIPT, "compute_rankings.py  (SpringRank, ~10-15 min)")
        if not ok:
            print("\nAborting: compute_rankings.py failed.")
            sys.exit(1)

    # ── Step 2: Figures ───────────────────────────────────────────────────────
    for script in FIGURE_SCRIPTS:
        label = script.name
        if label == "si_fig4.py":
            label += "  (~3 min, permutation test)"
        ok = run(script, label)
        if not ok:
            failures.append(script.name)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  Finished in {total/60:.1f} min")
    if failures:
        print(f"  FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        n = len(FIGURE_SCRIPTS)
        print(f"  All {n} figure scripts completed successfully.")
        print(f"  PDFs written to: reproducibility/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
