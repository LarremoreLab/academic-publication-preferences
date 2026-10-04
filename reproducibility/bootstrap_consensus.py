"""Respondent bootstrap of the field-level consensus measures.

Produces the 95% intervals drawn in SI Fig. S2 (Fig. 2 with bootstrap intervals, drawn by
si_fig2_bootstrap.py from the cache written here) and the bootstrap numbers quoted in the paper
and its SI: per-field intervals for prediction accuracy, top-5 agreement and consideration-set
overlap; intervals for the two correlations between consensus measures; the stability of the
field ordering under resampling; the stability of the field top-5 lists; and the exact venue
accumulation values of Fig. 1A with half-sample bands.

Measures (the full-sample definitions are in the scripts named)
  prediction accuracy     fig2.py / compute_rankings.py: a field SpringRank (alpha = 20) fit on the
                          comparisons of everyone else in the field (leave-one-out), scored on the
                          respondent's own strict comparisons up to the first repeated venue
  top-5 agreement         fig2.py: share of all individual top-5 votes in a field that go to the
                          field's five most-voted venues
  overlap                 fig1.py (Fig. 1B diagonal): for each venue a respondent would consider,
                          the share of the field's other respondents who would also consider it,
                          pooled over all respondent-venue pairs
  top-3 selection shares  fig1.py (Fig. 1C): share of respondents selecting each of the field's
                          three most-selected venues, in whole percent
  venue accumulation      fig1.py (Fig. 1A): expected number of distinct venues in the consideration
                          sets of x respondents drawn without replacement. Computed exactly by
                          rarefaction, sum_v [1 - C(n - k_v, x) / C(n, x)] with k_v the number of
                          respondents considering venue v, instead of averaging random orderings.

Resampling
  Respondent bootstrap. In every replicate each field's respondents are drawn with replacement
  (n_f draws for a field of n_f respondents). A drawn respondent brings all their comparisons, their
  individual ranking and their consideration set, and every measure is recomputed. Prediction
  accuracy refits the leave-one-out field ranking for every distinct drawn respondent, leaving out
  all copies of that respondent, and averages over drawn copies. Top-5 votes count once per copy.
  Overlap excludes pairs of copies of the same respondent. Intervals are 95% percentile intervals.
  Replicate r is generated from numpy.random.SeedSequence(seed).spawn(N)[r], so the result does
  not depend on the number of workers.

  Half-samples for accumulation. A with-replacement bootstrap duplicates consideration sets, so
  accumulation is resampled by drawing floor(n/2) respondents without replacement (seed + 1). The
  band at x <= floor(n/2) is the 2.5th-97.5th percentile of the exact curves of the half-samples;
  there is no band beyond floor(n/2). The field ordering of accumulation is assessed at
  x = min over fields of floor(n/2), pairing half-sample b of one field with half-sample b of another.

Inputs   public_data/comparisons.csv, public_data/venue_selections.csv, public_data/respondents.csv,
         public_data/venues.csv, reproducibility/derived/individual_rankings.csv (compute_rankings.py)

Outputs  reproducibility/cache/ (committed, so the figure can be drawn without rerunning)
  bootstrap_consensus.csv              field x measure: point, lo, hi, boot_median, method, n, x
  bootstrap_point_estimates.csv        full-sample value of every measure by field
  bootstrap_replicates.csv             every replicate's field-level values
  bootstrap_field_order_stability.csv  field pair x measure: share of replicates keeping the
                                       full-sample order
  bootstrap_top5_stability.csv         field x venue: share of replicates with the venue in the
                                       field's top 5
  bootstrap_correlations.csv           point and interval of the two correlations between consensus
                                       measures quoted in the paper
  accumulation_bands.csv               field x x: exact accumulation curve with its half-sample band

Usage    cd <repo_root>
         .venv/bin/python reproducibility/compute_rankings.py   # only needed once
         .venv/bin/python reproducibility/bootstrap_consensus.py [--replicates 1000] [--workers W]
                                                                 [--seed 20260929] [--half-samples 1000]
         One replicate is about 2,200 SpringRank fits (8.5 s on one core); 1,000 replicates take
         about 13 minutes on 16 cores.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"      # one BLAS thread per process; parallelism is over replicates

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.special import gammaln  # noqa: E402
from springrank.springrank import SpringRank  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FIELDS = common.FIELDS
K = 5                            # top-k of Fig. 2
ALPHA = 20                       # field SpringRank regularisation, as in compute_rankings.py
DEFAULT_SEED = 20260929
MEASURES = {"prediction_accuracy": "Prediction accuracy (%)",
            "top5_agreement": "Top-5 agreement (%)",
            "overlap": "Within-field overlap (%)"}


# ── data ──────────────────────────────────────────────────────────────────────────────────────
class FieldData:
    """One field's respondents with their comparisons, individual top 5 and consideration sets."""

    def __init__(self, comp, ind, sel, users):
        self.users = list(users)                                   # respondents.csv order
        c = comp[comp.user_db_id.isin(self.users)]
        groups = {u: g for u, g in c.groupby("user_db_id", sort=False)}
        self.comp_tuples = {u: [(int(p), int(o), bool(t)) for p, o, t in
                                g[["pref_venue_db_id", "other_venue_db_id", "is_tie"]].values]
                            for u, g in groups.items()}
        self.comp_arrays = {u: (g.pref_venue_db_id.to_numpy(), g.other_venue_db_id.to_numpy(),
                                g.is_tie.to_numpy(bool)) for u, g in groups.items()}
        self.top5 = {u: set(g.loc[g.ordinal_rank < K, "venue_db_id"])
                     for u, g in ind[ind.user_db_id.isin(self.users)].groupby("user_db_id")}
        w = sel[(sel.would_publish == True) & sel.user_db_id.isin(self.users)]  # noqa: E712
        self.cons = {u: list(g.venue_db_id) for u, g in w.groupby("user_db_id", sort=False)}
        self.order = list(c.user_db_id.drop_duplicates())          # comparisons.csv order


def load_fields():
    comp = pd.read_csv(common.PUBLIC / "comparisons.csv")
    ind = pd.read_csv(common.DERIVED / "individual_rankings.csv")
    sel = pd.read_csv(common.PUBLIC / "venue_selections.csv")
    resp = pd.read_csv(common.PUBLIC / "respondents.csv", usecols=["user_db_id", "field"])
    assert sorted(resp.field.unique()) == FIELDS
    return {f: FieldData(comp, ind, sel, resp.loc[resp.field == f, "user_db_id"]) for f in FIELDS}


# ── field SpringRank (same matrix and node order as compute_rankings.build_and_fit) ───────────
def fit_field(pref, other, tie, weight):
    """Leave-one-out field ranking from comparison rows with integer weights (bootstrap
    multiplicities). Strict preferences add `weight` to A[pref, other]; ties add half to each
    direction. Returns {venue: SpringRank score}."""
    seq = np.empty(2 * len(pref), dtype=object)
    seq[0::2], seq[1::2] = pref.tolist(), other.tolist()
    knowns = set(seq.tolist())
    a_pos = {v: i for i, v in enumerate(knowns)}
    n = len(knowns)
    w = np.asarray(weight, float)
    pi = np.fromiter((a_pos[v] for v in pref.tolist()), int, len(pref))
    oi = np.fromiter((a_pos[v] for v in other.tolist()), int, len(pref))
    A = np.zeros((n, n))
    strict = ~tie
    np.add.at(A, (pi[strict], oi[strict]), w[strict])
    np.add.at(A, (pi[tie], oi[tie]), 0.5 * w[tie])
    np.add.at(A, (oi[tie], pi[tie]), 0.5 * w[tie])
    model = SpringRank(alpha=ALPHA)
    model.fit(A)
    return {v: model.ranks[i] for v, i in a_pos.items()}


# ── the measures, for a multiset of drawn respondents (counts = {user: multiplicity}) ─────────
def user_accuracy(u_comps, ranks):
    """fig2.py reveal_upsets for one respondent: share of their strict comparisons, up to the first
    repeated venue, that the ranking predicts correctly."""
    seen = defaultdict(int)
    correct = preds = 0
    for pref, other, tie in u_comps:
        seen[pref] += 1
        seen[other] += 1
        if seen[pref] > 1 or seen[other] > 1:
            break
        if not tie:
            if pref in ranks and other in ranks:
                preds += 1
                if ranks[pref] > ranks[other]:
                    correct += 1
    return correct / preds if preds else np.nan


def draw_arrays(fd, counts):
    """Comparison rows of the drawn respondents (in comparisons.csv order), each row weighted by its
    respondent's multiplicity. Returns pref, other, tie, weight, owner."""
    us = [u for u in fd.order if counts.get(u, 0) > 0]
    P = np.concatenate([fd.comp_arrays[u][0] for u in us])
    O = np.concatenate([fd.comp_arrays[u][1] for u in us])
    T = np.concatenate([fd.comp_arrays[u][2] for u in us])
    W = np.concatenate([np.full(len(fd.comp_arrays[u][0]), counts[u], float) for u in us])
    own = np.concatenate([np.full(len(fd.comp_arrays[u][0]), u) for u in us])
    return P, O, T, W, own


def prediction_accuracy(fd, counts):
    """Mean leave-one-out accuracy over drawn copies; every copy of a respondent is left out of the
    field ranking that respondent is scored against."""
    P, O, T, W, own = draw_arrays(fd, counts)
    accs, wts = [], []
    for u, cnt in counts.items():
        if cnt == 0 or u not in fd.comp_tuples:
            continue
        keep = own != u
        ranks = fit_field(P[keep], O[keep], T[keep], W[keep])
        a = user_accuracy(fd.comp_tuples[u], ranks)
        if not np.isnan(a):
            accs.append(a)
            wts.append(cnt)
    return np.average(accs, weights=wts)


def top5_agreement(fd, counts):
    """fig2.py field_top_k_overlaps: share of all individual top-5 votes that go to the field's five
    most-voted venues. Also returns the vote counts."""
    votes = defaultdict(int)
    total = 0
    for u, cnt in counts.items():
        for v in fd.top5.get(u, ()):
            votes[v] += cnt
            total += cnt
    top_counts = sorted(votes.values(), reverse=True)[:K]
    return sum(top_counts) / total, votes


def field_top5(votes):
    """The five most-voted venues; ties broken by venue id."""
    return [int(v) for v, _ in sorted(votes.items(), key=lambda x: (-x[1], x[0]))[:K]]


def overlap(fd, counts):
    """fig1.py Fig. 1B diagonal: mean over (drawn copy, considered venue) of the share of the OTHER
    drawn respondents (excluding every copy of the same respondent) who also consider the venue, %."""
    n = sum(c for u, c in counts.items() if u in fd.cons)
    vc = defaultdict(int)
    for u, c in counts.items():
        for v in fd.cons.get(u, ()):
            vc[v] += c
    num = den = 0.0
    for u, c in counts.items():
        if c == 0 or u not in fd.cons:
            continue
        vals = [(vc[v] - c) / (n - c) for v in fd.cons[u]]
        num += c * sum(vals)
        den += c * len(vals)
    return 100 * num / den


def top3_selection(fd, counts):
    """fig1.py Fig. 1C: percentage of respondents selecting each of the field's three most-selected
    venues, rounded to whole percent."""
    n = sum(c for u, c in counts.items() if u in fd.cons)
    vc = defaultdict(int)
    for u, c in counts.items():
        for v in set(fd.cons.get(u, ())):
            vc[v] += c
    top = sorted(vc.values(), reverse=True)[:3]
    return [int(round(100 * t / n)) for t in top]


def accumulation_curve(sets, x_max):
    """Exact rarefaction curve for x = 0..x_max: expected number of distinct venues in the
    consideration sets of x respondents drawn without replacement from `sets`."""
    n = len(sets)
    k = np.array(list(Counter(v for s in sets for v in set(s)).values()), float)
    ku, kc = np.unique(k, return_counts=True)
    m = np.arange(x_max + 1, dtype=float)[:, None]
    a = n - ku[None, :]
    with np.errstate(invalid="ignore"):   # log C(n - k, m) - log C(n, m); zero when n - k < m
        log_ratio = (gammaln(a + 1) - gammaln(a - m + 1)) - (gammaln(n + 1) - gammaln(n - m + 1))
    ratio = np.where(a >= m, np.exp(log_ratio), 0.0)
    return ((1 - ratio) * kc[None, :]).sum(axis=1)


def half_sample_bands(FD, n_half, seed):
    """Exact accumulation curve of every field (x = 0..n) with its 95% half-sample band up to
    floor(n/2). Returns the band table and, per field, the full curve and the half-sample curves."""
    rng = np.random.default_rng(seed)
    rows, full_curves, half_curves = [], {}, {}
    for f in FIELDS:                      # fixed field order: the draws are deterministic
        fd = FD[f]
        sets = [fd.cons[u] for u in fd.users if u in fd.cons]
        n, h = len(sets), len(sets) // 2
        full = accumulation_curve(sets, n)
        assert abs(full[-1] - len({v for s in sets for v in s})) < 1e-6
        H = np.empty((n_half, h + 1))
        for b in range(n_half):
            idx = rng.choice(n, size=h, replace=False)
            H[b] = accumulation_curve([sets[i] for i in idx], h)
        lo, hi = np.percentile(H, [2.5, 97.5], axis=0)
        full_curves[f], half_curves[f] = full, H
        rows += [(f, x, full[x], lo[x], hi[x], n, h) for x in range(h + 1)]
    bands = pd.DataFrame(rows, columns=["field", "x", "point_exact", "lo", "hi", "n", "half"])
    return bands, full_curves, half_curves


# ── one bootstrap replicate (runs in a worker process) ────────────────────────────────────────
_FD = None


def _init_worker():
    global _FD
    _FD = load_fields()


def one_replicate(args):
    r, seed_seq = args
    rng = np.random.default_rng(seed_seq)
    rows = []
    for f in FIELDS:                      # fixed field order: the draws are deterministic
        fd = _FD[f]
        draws = rng.choice(len(fd.users), size=len(fd.users), replace=True)
        counts = {}
        for i in draws:
            u = fd.users[i]
            counts[u] = counts.get(u, 0) + 1
        share, votes = top5_agreement(fd, counts)
        t3 = top3_selection(fd, counts)
        rows.append({"replicate": r, "field": f,
                     "prediction_accuracy": 100 * prediction_accuracy(fd, counts),
                     "top5_agreement": 100 * share, "overlap": overlap(fd, counts),
                     "top3_1": t3[0], "top3_2": t3[1], "top3_3": t3[2],
                     "n_distinct": len(counts),
                     "top5_venues": ";".join(map(str, field_top5(votes)))})
    return rows


def pooled_top3_spearman(top3, ov):
    """Spearman correlation over 39 points: each field's three top-3 shares against its overlap."""
    xs = [v for f in FIELDS for v in top3[f]]
    ys = [round(ov[f], 2) for f in FIELDS for _ in range(3)]
    return stats.spearmanr(xs, ys)[0]


# ── main ──────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicates", type=int, default=1000, help="bootstrap replicates (default 1000)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="worker processes (default: all cores)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"bootstrap seed (default {DEFAULT_SEED}); "
                                                                  "half-samples use seed + 1")
    ap.add_argument("--half-samples", type=int, default=1000, help="half-samples per field (default 1000)")
    args = ap.parse_args()
    R, n_half = args.replicates, args.half_samples
    common.require_derived()
    common.CACHE.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    FD = load_fields()
    names = pd.read_csv(common.PUBLIC / "venues.csv").set_index("venue_db_id")["name"]

    # full-sample values of every measure (every respondent drawn once)
    pt_rows, full_top5 = [], {}
    for f in FIELDS:
        fd, counts = FD[f], {u: 1 for u in FD[f].users}
        share, votes = top5_agreement(fd, counts)
        full_top5[f] = field_top5(votes)
        t3 = top3_selection(fd, counts)
        pt_rows.append({"field": f, "n_respondents": len(fd.users),
                        "prediction_accuracy": 100 * prediction_accuracy(fd, counts),
                        "top5_agreement": 100 * share, "overlap": overlap(fd, counts),
                        "top3_1": t3[0], "top3_2": t3[1], "top3_3": t3[2]})
    pt = pd.DataFrame(pt_rows).set_index("field")
    print(f"full-sample values: {time.time() - t_start:.0f} s")

    # exact accumulation curves with half-sample bands
    t0 = time.time()
    bands, full_curves, half_curves = half_sample_bands(FD, n_half, args.seed + 1)
    bands.to_csv(common.CACHE / "accumulation_bands.csv", index=False)
    pt["n_consideration"] = [len(full_curves[f]) - 1 for f in FIELDS]
    pt["accumulation_at_100"] = [full_curves[f][100] if len(full_curves[f]) > 100 else np.nan for f in FIELDS]
    pt["unique_venues"] = [full_curves[f][-1] for f in FIELDS]
    pt.to_csv(common.CACHE / "bootstrap_point_estimates.csv")
    print(f"{n_half} half-samples per field: {time.time() - t0:.0f} s")

    # the bootstrap
    t0 = time.time()
    seeds = np.random.SeedSequence(args.seed).spawn(R)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        results = list(ex.map(one_replicate, [(r, seeds[r]) for r in range(R)], chunksize=1))
    t_boot = time.time() - t0
    reps = pd.DataFrame([row for rows in results for row in rows])
    reps.to_csv(common.CACHE / "bootstrap_replicates.csv", index=False)
    print(f"{R} replicates on {args.workers} workers: {t_boot:.0f} s ({t_boot / R:.1f} s per replicate)")
    boot = {m: reps.pivot(index="replicate", columns="field", values=m)[FIELDS] for m in MEASURES}

    # intervals
    rows = []
    for m in MEASURES:
        for f in FIELDS:
            lo, hi = np.percentile(boot[m][f], [2.5, 97.5])
            rows.append({"field": f, "measure": m, "point": pt.loc[f, m], "lo": lo, "hi": hi,
                         "boot_median": boot[m][f].median(),
                         "method": f"respondent bootstrap, {R} replicates, 95% percentile",
                         "n": int(pt.loc[f, "n_respondents"]), "x": np.nan})
    x_cmp = int(bands.half.min())           # largest x with a band in every field
    for f in FIELDS:
        a = bands[bands.field == f]
        n, h = int(a.n.iloc[0]), int(a.half.iloc[0])
        for x in (100, x_cmp):
            r = a[a.x == x]
            if len(r):
                rows.append({"field": f, "measure": f"accumulation_at_{x}", "point": r.point_exact.iloc[0],
                             "lo": r.lo.iloc[0], "hi": r.hi.iloc[0], "boot_median": np.nan,
                             "method": f"half-sample (floor(n/2) = {h}), {n_half}, 95% percentile; "
                                       "point = exact expected curve", "n": n, "x": x})
            else:
                rows.append({"field": f, "measure": f"accumulation_at_{x}", "point": np.nan, "lo": np.nan,
                             "hi": np.nan, "boot_median": np.nan,
                             "method": f"not defined: x = {x} > floor(n/2) = {h}", "n": n, "x": x})
        rows.append({"field": f, "measure": "accumulation_endpoint", "point": pt.loc[f, "unique_venues"],
                     "lo": np.nan, "hi": np.nan, "boot_median": np.nan,
                     "method": "no interval: evaluated at x = n > floor(n/2) (Fig. 1A circle and label)",
                     "n": n, "x": n})
    ci = pd.DataFrame(rows)
    ci.to_csv(common.CACHE / "bootstrap_consensus.csv", index=False)

    # field-order stability: share of replicates in which a field pair keeps its full-sample order
    acc_x = pd.DataFrame({f: half_curves[f][:, x_cmp] for f in FIELDS})
    acc_x_pt = bands[bands.x == x_cmp].set_index("field").point_exact.to_dict()
    series = {**{m: (boot[m], pt[m].to_dict()) for m in MEASURES}, f"accumulation_at_{x_cmp}": (acc_x, acc_x_pt)}
    ord_rows = []
    for m, (bd, p) in series.items():
        for a, b in combinations(FIELDS, 2):
            hi_f, lo_f = (a, b) if p[a] > p[b] else (b, a)
            ord_rows.append({"measure": m, "higher_in_full_sample": hi_f, "lower_in_full_sample": lo_f,
                             "point_higher": p[hi_f], "point_lower": p[lo_f],
                             "share_same_order": (bd[hi_f] > bd[lo_f]).mean()})
    order = pd.DataFrame(ord_rows)
    order.to_csv(common.CACHE / "bootstrap_field_order_stability.csv", index=False)

    # top-5 stability: share of replicates in which each venue is in the field's top 5
    t5_rows = []
    for f in FIELDS:
        lists = reps.loc[reps.field == f, "top5_venues"]
        cnt = pd.Series([int(v) for s in lists for v in s.split(";")]).value_counts() / R
        for v, share in cnt.items():
            t5_rows.append({"field": f, "venue_db_id": v, "name": names.get(v), "share_of_replicates": share,
                            "in_full_sample_top5": v in full_top5[f]})
    top5 = pd.DataFrame(t5_rows).sort_values(["field", "share_of_replicates"], ascending=[True, False])
    top5.to_csv(common.CACHE / "bootstrap_top5_stability.csv", index=False)

    # the two correlations between consensus measures quoted in the paper
    r_t5_acc = [stats.pearsonr(boot["top5_agreement"].iloc[i], boot["prediction_accuracy"].iloc[i])[0]
                for i in range(R)]
    top3_by_rep = {r: {g.field: [g.top3_1, g.top3_2, g.top3_3] for g in grp.itertuples()}
                   for r, grp in reps.groupby("replicate")}
    r_top3 = [pooled_top3_spearman(top3_by_rep[i], boot["overlap"].iloc[i]) for i in range(R)]
    pt_top3 = {f: [pt.loc[f, "top3_1"], pt.loc[f, "top3_2"], pt.loc[f, "top3_3"]] for f in FIELDS}
    method = f"respondent bootstrap, {R} replicates, 95% percentile"
    corr = pd.DataFrame([
        {"correlation": "Pearson r, top-5 agreement vs prediction accuracy (13 fields; Fig. 2)",
         "point": stats.pearsonr(pt.top5_agreement, pt.prediction_accuracy)[0],
         "lo": np.percentile(r_t5_acc, 2.5), "hi": np.percentile(r_t5_acc, 97.5), "method": method},
        {"correlation": "Spearman r, top-3 selection shares vs mean overlap (39 pooled points, 3 per field; Fig. 1)",
         "point": pooled_top3_spearman(pt_top3, pt.overlap),
         "lo": np.percentile(r_top3, 2.5), "hi": np.percentile(r_top3, 97.5), "method": method},
    ])
    corr.to_csv(common.CACHE / "bootstrap_correlations.csv", index=False)

    # printout
    pd.set_option("display.width", 220)
    show = ci[ci.measure.isin(MEASURES)].pivot(index="field", columns="measure", values=["point", "lo", "hi"])
    print("\nfull-sample values with 95% bootstrap intervals\n")
    print(show.round(1).to_string())
    print("\naccumulation at 100 respondents (exact; half-sample band):\n")
    print(ci[ci.measure == "accumulation_at_100"].set_index("field")[["point", "lo", "hi"]].round(1).to_string())
    print(f"\nfield pairs whose order holds in fewer than 95% of replicates (of {len(order) // len(series)}):")
    print(order.groupby("measure").share_same_order.apply(lambda s: int((s < 0.95).sum())).to_string())
    print()
    print(corr[["correlation", "point", "lo", "hi"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote the cache to {common.CACHE}; total {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
