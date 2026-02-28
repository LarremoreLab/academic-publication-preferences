"""
Compute SpringRank rankings from raw comparison data.

Inputs (from public_data/):
  - comparisons.csv

Outputs (written to reproducibility/derived/):
  - individual_rankings.csv
      user_db_id, venue_db_id, unscaled_rank, ordinal_rank, normed_rank
      SpringRank(alpha=0) run independently per user on their own comparisons.

  - field_rankings_per_user.csv
      user_db_id, venue_db_id, score
      SpringRank(alpha=20) leave-one-out per field: for each user, fit on all
      field comparisons *excluding* that user, then retain only venues that
      user actually compared. This is a field-level consensus ranking from
      each user's peers.

  - field_consensus_rankings.csv
      field, venue_db_id, score
      SpringRank(alpha=20) fit on all comparisons in a field, scores rescaled
      via get_rescaled_ranks(0.75). One row per (field, venue) pair. Used for
      vertical node positioning in the network visualizations (Fig 3).

  - global_rankings_per_user.csv
      user_db_id, venue_db_id, score
      SpringRank(alpha=20) leave-one-out across ALL fields: for each user, fit
      on all comparisons (from all fields) *excluding* that user's own, then
      retain only venues that user actually compared. This is a cross-field
      "academia-wide" consensus ranking, used as a baseline in SI Figure 3.

Tie handling:
  Indifferent responses (is_tie=True) are counted as half-wins for each venue:
  A[i,j] += 0.5 and A[j,i] += 0.5. Indifferent responses constitute 8.1% of
  all comparisons.

Usage:
  cd <repo_root>
  .venv/bin/python reproducibility/compute_rankings.py

Run this once before running any figure scripts that depend on rankings.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from springrank.springrank import SpringRank

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
PUBLIC = HERE.parent / "public_data"
DERIVED = HERE / "derived"
DERIVED.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
comp_df = pd.read_csv(PUBLIC / "comparisons.csv")


# ---------------------------------------------------------------------------
# Helper: build adjacency matrix and fit SpringRank
# ---------------------------------------------------------------------------
def build_and_fit(comparisons, alpha):
    knowns = set(
        comparisons[["pref_venue_db_id", "other_venue_db_id"]].values.flatten()
    )
    a_pos = {v: i for i, v in enumerate(knowns)}
    a_matrix = np.zeros((len(knowns), len(knowns)))
    for winner, loser, is_tie in comparisons[
        ["pref_venue_db_id", "other_venue_db_id", "is_tie"]
    ].values:
        if is_tie is False:
            a_matrix[a_pos[winner], a_pos[loser]] += 1
        else:
            # Indifferent responses count as a half-win for each venue
            a_matrix[a_pos[winner], a_pos[loser]] += 0.5
            a_matrix[a_pos[loser], a_pos[winner]] += 0.5
    model = SpringRank(alpha=alpha)
    model.fit(a_matrix)
    return a_pos, model


# ---------------------------------------------------------------------------
# Individual rankings: SpringRank(alpha=0) per user
# ---------------------------------------------------------------------------
print("Computing individual rankings (SpringRank alpha=0)...")
rows = []
for user_db_id, comparisons in comp_df.groupby("user_db_id"):
    a_pos, model = build_and_fit(comparisons, alpha=0)
    ranks = model.ranks
    max_r, min_r = ranks.max(), ranks.min()
    norm = (lambda x: (x - min_r) / (max_r - min_r)) if max_r != min_r else (lambda x: 0.0)
    ranking = sorted(
        [(v, ranks[i]) for v, i in a_pos.items()], key=lambda x: -x[1]
    )
    for ordinal, (venue_db_id, unscaled) in enumerate(ranking):
        rows.append((user_db_id, venue_db_id, unscaled, ordinal, norm(unscaled)))

individual_rankings = pd.DataFrame(
    rows,
    columns=["user_db_id", "venue_db_id", "unscaled_rank", "ordinal_rank", "normed_rank"],
)
out = DERIVED / "individual_rankings.csv"
individual_rankings.to_csv(out, index=False)
print(f"  Saved: {out}  ({len(individual_rankings):,} rows)")

# ---------------------------------------------------------------------------
# Field rankings per user: SpringRank(alpha=20), leave-one-out per field
# ---------------------------------------------------------------------------
print("Computing field rankings per user (SpringRank alpha=20, leave-one-out)...")
rows = []
for field, field_comps in comp_df.groupby("field"):
    users = field_comps["user_db_id"].unique()
    for user in users:
        leave_one_out = field_comps.loc[field_comps["user_db_id"] != user]
        user_venues = set(
            field_comps.loc[
                field_comps["user_db_id"] == user,
                ["pref_venue_db_id", "other_venue_db_id"],
            ].values.flatten()
        )
        a_pos, model = build_and_fit(leave_one_out, alpha=20)
        for venue_db_id, i in a_pos.items():
            if venue_db_id in user_venues:
                rows.append((user, venue_db_id, model.ranks[i]))

field_rankings = pd.DataFrame(rows, columns=["user_db_id", "venue_db_id", "score"])
out = DERIVED / "field_rankings_per_user.csv"
field_rankings.to_csv(out, index=False)
print(f"  Saved: {out}  ({len(field_rankings):,} rows)")

# ---------------------------------------------------------------------------
# Field consensus rankings: SpringRank(alpha=20), rescaled, one per field
# Used for vertical node positioning in network visualizations (Fig 3).
# ---------------------------------------------------------------------------
print("Computing field consensus rankings (SpringRank alpha=20, rescaled)...")
rows = []
for field, field_comps in comp_df.groupby("field"):
    a_pos, model = build_and_fit(field_comps, alpha=20)
    rescaled = model.get_rescaled_ranks(0.75)
    for venue_db_id, i in a_pos.items():
        rows.append((field, venue_db_id, rescaled[i]))

field_consensus = pd.DataFrame(rows, columns=["field", "venue_db_id", "score"])
out = DERIVED / "field_consensus_rankings.csv"
field_consensus.to_csv(out, index=False)
print(f"  Saved: {out}  ({len(field_consensus):,} rows)")

# ---------------------------------------------------------------------------
# Global rankings per user: SpringRank(alpha=20), leave-one-out, all fields
# ---------------------------------------------------------------------------
# Same logic as field_rankings_per_user but pooling comparisons across all
# fields. The resulting score for each user reflects where their compared
# venues sit in an academia-wide consensus built from everyone else.
#
# Used in si_fig3.py for the "Academia" row (cross-field accuracy baseline).
print("Computing global rankings per user (SpringRank alpha=20, leave-one-out, all fields)...")
rows = []
all_users = comp_df["user_db_id"].unique()
for user in all_users:
    leave_one_out = comp_df.loc[comp_df["user_db_id"] != user]
    user_venues = set(
        comp_df.loc[
            comp_df["user_db_id"] == user,
            ["pref_venue_db_id", "other_venue_db_id"],
        ].values.flatten()
    )
    a_pos, model = build_and_fit(leave_one_out, alpha=20)
    for venue_db_id, i in a_pos.items():
        if venue_db_id in user_venues:
            rows.append((user, venue_db_id, model.ranks[i]))

global_rankings = pd.DataFrame(rows, columns=["user_db_id", "venue_db_id", "score"])
out = DERIVED / "global_rankings_per_user.csv"
global_rankings.to_csv(out, index=False)
print(f"  Saved: {out}  ({len(global_rankings):,} rows)")

print("Done.")
