"""SI Tables S8-S10: robustness of the results to the composition of the sample.

Respondents differ from the AARC faculty population in field size, gender, career stage and
institutional prestige (SI Table S11). Three checks ask whether that matters for the paper's results
(SI section "Robustness to Sample Composition").

  Table S8   stab:subgroup_consensus
      Field consensus recomputed within respondent subgroups -- men, women, the top and bottom
      prestige tercile, associate professors, full professors -- and compared with 200 random
      subsets of the same field and the same size. Measures: venue accumulation at N = 12
      consideration sets (the paper's Table S8 labels this column "Accuracy"; fewer venues = more
      consensus), mean consideration-set overlap (Fig. 1B), leave-one-out top-5 agreement (Fig. 2),
      and the Spearman correlation between the SpringRank field ranking fit to the subgroup's
      comparisons alone and the all-respondent ranking. A subgroup is "outside the null band" in a
      field when its value lies below the 2.5th or above the 97.5th percentile of the random subsets.
  Table S9   stab:raked_consensus
      Field consensus after raking each field's respondents (iterative proportional fitting) to the
      AARC 2023 share of women among faculty of known gender and to equal shares across the ten
      prestige deciles, with weights normalized to mean 1 and capped at 5. Five measures, unweighted
      (the published value) and raked: expected distinct venues in 100 consideration sets (Fig. 1A;
      raked value by Monte Carlo, 3,000 draws), overlap (Fig. 1B), share selecting the field's three
      most-selected venues (Fig. 1C), top-5 agreement (Fig. 2), and the share whose top aspiration is
      the field's modal top aspiration.
  Table S10  stab:field_weighted
      The pooled ("Academia") models of Figs. 4 and 5 re-estimated with each respondent weighted by
      AARC field share / sample field share (normalized to mean 1): Eq. S1 with field fixed effects
      for both outcomes, the women - men and decile 10 - decile 1 contrasts of Fig. 4A/B, and the
      respondent-level Fig. 5 model (prestige slopes, predictions at decile 10, own - field gap).
      OLS (unweighted) or WLS (weighted), standard errors clustered by institution, t reference
      distribution. Fig. 5 quantities are averaged over respondents with the same weights.

Inputs   public_data/respondents.csv, venue_selections.csv, comparisons.csv, publication_counts.csv,
         aarc_field_margins.csv; reproducibility/derived/ (compute_rankings.py).
         reproducibility/output/results/subfield_consensus.csv (si_table1_subfields.py) is used, when
         present, to check that the whole-field consensus values computed here equal SI Table S1's.
Outputs  reproducibility/output/tables/si_table8.tex, si_table9.tex, si_table10.tex
         reproducibility/output/results/
             subgroup_consensus.csv     field x subgroup x measure: value, null percentile, null mean
                                        and central 95% band (Table S8 inputs)
             subgroup_summary.csv       Table S8 rows
             raked_consensus.csv        field x measure: unweighted and raked values
             raking_diagnostics.csv     per field: weight range, design effect, achieved margins
             field_weighted_models.csv  Table S10 rows
             field_weights.csv          AARC share, sample share and weight per field
         The numbers quoted in the SI text are printed at the end.

Usage    cd <repo_root>
         .venv/bin/python reproducibility/compute_rankings.py          # once
         .venv/bin/python reproducibility/si_table1_subfields.py       # optional, enables a check
         .venv/bin/python reproducibility/si_tables8_10_composition.py  # ~1 min, 13 worker processes
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"  # one BLAS thread per worker process (set before numpy is imported)

import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from math import comb  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from patsy import build_design_matrices  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from springrank.springrank import SpringRank  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TOP_K = 5                       # individual / group top-5
# Table S8
N_NULL, SEED_SUBGROUPS = 200, 20261002
N_ACC_SUBGROUP, TOP_RANK = 12, 10
SUBGROUPS = {"men": lambda d: d.gender == "gm",
             "women": lambda d: d.gender == "gf",
             "prestige_top": lambda d: d.prestige_tercile == 3,
             "prestige_bottom": lambda d: d.prestige_tercile == 1,
             "associate": lambda d: d.career_stage == "Associate Professor",
             "full": lambda d: d.career_stage == "Full Professor"}
SUBGROUP_LABELS = {"men": "Men", "women": "Women", "prestige_top": "Top prestige tercile",
                   "prestige_bottom": "Bottom prestige tercile", "associate": "Associate professors",
                   "full": "Full professors"}
# Table S9
N_ACC_RAKED, N_MC, SEED_RAKING, W_CAP = 100, 3000, 20261003, 5.0
RAKED_MEASURES = ["acc100", "ovl", "top3", "top5", "flagship"]

_G = {}   # data shared by the subgroup workers (filled by _init)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared inputs
# ═══════════════════════════════════════════════════════════════════════════════
def _init():
    """Respondents, consideration sets, individual top-5s and pairwise comparisons."""
    if _G:
        return
    resp = common.load_respondents()
    ind = pd.read_csv(common.DERIVED / "individual_rankings.csv")
    sel = pd.read_csv(common.PUBLIC / "venue_selections.csv")
    sel = sel[sel.would_publish == True]  # noqa: E712
    comp = pd.read_csv(common.PUBLIC / "comparisons.csv")
    _G.update(resp=resp,
              own_top5={u: set(g.loc[g.ordinal_rank < TOP_K, "venue_db_id"]) for u, g in ind.groupby("user_db_id")},
              cons={u: set(g.venue_db_id) for u, g in sel.groupby("user_db_id")},
              comp=comp)


def aarc_margins():
    """AARC 2023 per field: faculty count, share of the 13 fields, share of men among known gender."""
    m = pd.read_csv(common.PUBLIC / "aarc_field_margins.csv").set_index("field").loc[common.FIELDS]
    out = pd.DataFrame({"N": m.aarc_n_2023.astype(int)})
    out["share"] = out.N / out.N.sum()
    out["men_share_known"] = m.pct_men / (m.pct_men + m.pct_women)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Table S8: consensus within subgroups vs random subsets of the same size
# ═══════════════════════════════════════════════════════════════════════════════
# The three consensus measures are those of si_table1_subfields.py (whole-field values checked below).
def accumulation(users, N=N_ACC_SUBGROUP):
    """Expected number of distinct venues in N consideration sets drawn without replacement."""
    cons = _G["cons"]
    n = len(users)
    if n < N:
        return np.nan
    k = Counter(v for u in users for v in cons.get(u, ()))
    return sum(1 - comb(n - kv, N) / comb(n, N) for kv in k.values())


def overlap_pooled(users):
    """Mean over respondent-venue pairs of the % of the OTHER members who also consider the venue."""
    cons = _G["cons"]
    counts = Counter(v for u in users for v in cons.get(u, ()))
    vals = [100 * (counts[v] - 1) / (len(users) - 1) for u in users for v in cons.get(u, ())]
    return float(np.mean(vals)) if vals else np.nan


def top5_loo(users):
    """Mean % of a respondent's top 5 found in the group's vote-count top 5 (leave-one-out)."""
    own_top5 = _G["own_top5"]
    members = [u for u in users if u in own_top5]
    votes = Counter(v for u in members for v in own_top5.get(u, ()))
    out = []
    for u in members:
        c = Counter(votes)
        for v in own_top5[u]:
            c[v] -= 1
        ranked = sorted(((v, n) for v, n in c.items() if n > 0), key=lambda x: (-x[1], x[0]))
        g5 = {v for v, _ in ranked[:TOP_K]}
        out.append(len(own_top5[u] & g5) / TOP_K)
    return 100 * float(np.mean(out)) if out else np.nan


def fit_field(pref, other, tie, alpha=20):
    """SpringRank field ranking from comparison rows, as in compute_rankings.build_and_fit
    (ties count half a win for each venue). Returns {venue: score}."""
    seq = np.empty(2 * len(pref), dtype=object)
    seq[0::2], seq[1::2] = pref.tolist(), other.tolist()
    knowns = set(seq.tolist())
    a_pos = {v: i for i, v in enumerate(knowns)}
    n = len(knowns)
    pi = np.fromiter((a_pos[v] for v in pref.tolist()), int, len(pref))
    oi = np.fromiter((a_pos[v] for v in other.tolist()), int, len(pref))
    A = np.zeros((n, n))
    strict = ~tie
    np.add.at(A, (pi[strict], oi[strict]), 1.0)
    np.add.at(A, (pi[tie], oi[tie]), 0.5)
    np.add.at(A, (oi[tie], pi[tie]), 0.5)
    model = SpringRank(alpha=alpha)
    model.fit(A)
    return {v: model.ranks[i] for v, i in a_pos.items()}


def ranking(users):
    comp = _G["comp"]
    c = comp[comp.user_db_id.isin(set(users))]
    if c.empty:
        return {}
    return fit_field(c.pref_venue_db_id.to_numpy(), c.other_venue_db_id.to_numpy(), c.is_tie.to_numpy(bool))


def rank_sim(sub, full):
    """Spearman correlation over shared venues and overlap of the two top-10 lists."""
    shared = [v for v in sub if v in full]
    if len(shared) < 5:
        return np.nan, np.nan
    rho = spearmanr([sub[v] for v in shared], [full[v] for v in shared])[0]
    t_full = sorted(full, key=lambda v: -full[v])[:TOP_RANK]
    t_sub = sorted(sub, key=lambda v: -sub[v])[:TOP_RANK]
    return rho, len(set(t_full) & set(t_sub))


def subgroup_stats(users, full_rank):
    rho, t10 = rank_sim(ranking(users), full_rank)
    return {"acc": accumulation(users), "ovl": overlap_pooled(users), "top5": top5_loo(users),
            "rank_rho": rho, "rank_top10": t10}


def one_field(args):
    """All subgroups of one field, each against N_NULL random subsets of the field of the same size."""
    field, seed = args
    rng = np.random.default_rng(seed)
    fd = _G["resp"][_G["resp"].field == field]
    members = list(fd.user_db_id)
    full_rank = ranking(members)
    rows = [{"field": field, "subgroup": "all", "n": len(members), "measure": m, "value": v,
             "null_pct": np.nan, "null_mean": np.nan, "null_lo": np.nan, "null_hi": np.nan}
            for m, v in {"acc": accumulation(members), "ovl": overlap_pooled(members),
                         "top5": top5_loo(members), "rank_rho": 1.0, "rank_top10": float(TOP_RANK)}.items()]
    for sg, mask in SUBGROUPS.items():
        ids = list(fd.loc[mask(fd), "user_db_id"])
        n = len(ids)
        if n < N_ACC_SUBGROUP:
            rows += [{"field": field, "subgroup": sg, "n": n, "measure": m, "value": np.nan}
                     for m in ["acc", "ovl", "top5", "rank_rho", "rank_top10"]]
            continue
        val = subgroup_stats(ids, full_rank)
        null = {m: [] for m in val}
        for _ in range(N_NULL):
            s = subgroup_stats(list(rng.choice(members, size=n, replace=False)), full_rank)
            for m in s:
                null[m].append(s[m])
        for m, v in val.items():
            nl = np.array([x for x in null[m] if not np.isnan(x)])
            rows.append({"field": field, "subgroup": sg, "n": n, "measure": m, "value": v,
                         "null_pct": 100 * np.mean(nl <= v) if len(nl) and not np.isnan(v) else np.nan,
                         "null_mean": nl.mean(), "null_lo": np.percentile(nl, 2.5), "null_hi": np.percentile(nl, 97.5)})
    return rows


def check_whole_field_values():
    """Whole-field values of the three measures must equal SI Table S1's (si_table1_subfields.py)."""
    path = common.RESULTS / "subfield_consensus.csv"
    if not path.exists():
        print("  (subfield_consensus.csv not found; run si_table1_subfields.py to enable the Table S1 check)")
        return
    ref = pd.read_csv(path).set_index("field")
    resp = _G["resp"]
    for f in common.FIELDS:
        u = list(resp.loc[resp.field == f, "user_db_id"])
        assert abs(accumulation(u) - ref.loc[f, "acc_field_full"]) < 1e-9, f
        assert abs(overlap_pooled(u) - ref.loc[f, "ovl_field_full"]) < 1e-9, f
        assert abs(top5_loo(u) - ref.loc[f, "top5_field_full"]) < 1e-9, f
    print("  whole-field accumulation, overlap and top-5 values equal SI Table S1's for all 13 fields")


def run_subgroups():
    _init()
    check_whole_field_values()
    seeds = np.random.SeedSequence(SEED_SUBGROUPS).spawn(len(common.FIELDS))
    workers = min(len(common.FIELDS), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        rows = [r for rr in ex.map(one_field, zip(common.FIELDS, seeds)) for r in rr]
    long = pd.DataFrame(rows)
    long.to_csv(common.RESULTS / "subgroup_consensus.csv", index=False)

    summ = []
    for m in ["acc", "ovl", "top5", "rank_rho", "rank_top10"]:
        base = long[(long.measure == m) & (long.subgroup == "all")].set_index("field").value
        for sg in SUBGROUPS:
            s = long[(long.measure == m) & (long.subgroup == sg)].set_index("field")
            ok = s.value.notna()
            rec = {"measure": m, "subgroup": sg, "n_fields": int(ok.sum()), "median_n": float(s.n[ok].median()),
                   "fields_below_null": int((s.null_pct < 2.5).sum()), "fields_above_null": int((s.null_pct > 97.5).sum()),
                   "mean_value": float(s.value[ok].mean()), "mean_null": float(s.null_mean[ok].mean())}
            if m in ("acc", "ovl", "top5"):
                rec["spearman_field_order_vs_all"] = spearmanr(base[ok.index[ok]], s.value[ok])[0]
                rec["mean_value_minus_null"] = float((s.value[ok] - s.null_mean[ok]).mean())
            summ.append(rec)
    summary = pd.DataFrame(summ)
    summary.to_csv(common.RESULTS / "subgroup_summary.csv", index=False)
    return long, summary


# ═══════════════════════════════════════════════════════════════════════════════
# Table S9: raking each field to AARC gender shares and uniform prestige deciles
# ═══════════════════════════════════════════════════════════════════════════════
def rake(d, margins, iters=200, cap=W_CAP):
    """Iterative proportional fitting of weights to the marginal targets {column: {level: total}};
    weights are capped at `cap` times their mean after each cycle and returned with mean 1."""
    w = np.ones(len(d))
    for _ in range(iters):
        for col, tgt in margins.items():
            g = d[col].values
            cur = pd.Series(w).groupby(g).sum()
            for lvl, t in tgt.items():
                idx = g == lvl
                if idx.any() and cur.get(lvl, 0) > 0:
                    w[idx] *= t / cur[lvl]
        w = np.minimum(w, cap * w.mean())
    return w / w.mean()


def raking_weights(resp, marg):
    """Per-field raking to AARC gender shares (known gender) and uniform prestige deciles.
    Respondents of unknown gender or without a prestige decile keep their own share."""
    out = []
    for f, fd in resp.groupby("field"):
        d = fd.copy()
        n = len(d)
        d["g"] = d.gender.where(d.gender.isin(["gm", "gf"]), "unk")
        d["p"] = d[common.PRESTIGE].fillna(-1).astype(int).astype(str)
        nk = (d.g != "unk").sum()
        ms = marg.loc[f, "men_share_known"]
        npb = (d.p != "-1").sum()
        tg = {"gm": nk * ms, "gf": nk * (1 - ms), "unk": n - nk}
        tp = {**{str(k): npb / 10 for k in range(1, 11)}, "-1": n - npb}
        d["weight"] = rake(d, {"g": tg, "p": tp})
        out.append(d[["user_db_id", "field", "g", "p", "weight", "venue_bug", "top_init_venue_db_id"]])
    return pd.concat(out)


def acc100(users, w=None, rng=None):
    """Distinct venues in N_ACC_RAKED consideration sets: exact rarefaction when unweighted, otherwise
    Monte Carlo (N_MC draws of N_ACC_RAKED respondents without replacement, in proportion to w)."""
    cons = _G["cons"]
    if w is None:
        n = len(users)
        k = Counter(v for u in users for v in cons.get(u, ()))
        return sum(1 - comb(n - kv, N_ACC_RAKED) / comb(n, N_ACC_RAKED) for kv in k.values())
    users, p = np.asarray(users), np.asarray(w) / np.sum(w)
    return np.mean([len(set().union(*(cons.get(u, set()) for u in rng.choice(users, N_ACC_RAKED, replace=False, p=p))))
                    for _ in range(N_MC)])


def ovl_weighted(users, w):
    """Fig. 1B overlap with weighted shares of the other respondents; pairs weighted by w."""
    cons = _G["cons"]
    w = np.asarray(w, float)
    wv = defaultdict(float)
    for u, wu in zip(users, w):
        for v in cons.get(u, ()):
            wv[v] += wu
    num = den = 0.0
    for u, wu in zip(users, w):
        for v in cons.get(u, ()):
            num += wu * 100 * (wv[v] - wu) / (w.sum() - wu)
            den += wu
    return num / den


def top3_weighted(users, w, top):
    """Mean weighted % of respondents selecting each of the field's three most-selected venues."""
    cons = _G["cons"]
    w = np.asarray(w, float)
    return 100 * np.mean([sum(wu for u, wu in zip(users, w) if v in cons.get(u, ())) / w.sum() for v in top])


def top5_weighted(users, w):
    """Fig. 2 top-5 agreement: weighted share of individual top-5 votes going to the five most-voted venues."""
    own_top5 = _G["own_top5"]
    pairs = [(u, wu) for u, wu in zip(users, w) if u in own_top5]
    votes = defaultdict(float)
    for u, wu in pairs:
        for v in own_top5[u]:
            votes[v] += wu
    g5 = {v for v, _ in sorted(votes.items(), key=lambda x: (-x[1], x[0]))[:TOP_K]}
    return 100 * sum(wu * len(own_top5[u] & g5) for u, wu in pairs) / sum(wu * len(own_top5[u]) for u, wu in pairs)


def flagship_weighted(d, w, venue):
    w = np.asarray(w, float)
    return 100 * w[(d.top_init_venue_db_id == venue).values].sum() / w.sum()


def check_unweighted_values(results):
    """Unweighted top-5 agreement must equal Fig. 2's (public_data/field_top5.csv: votes for the
    field's five most-voted venues over all top-5 votes cast) and unweighted overlap must equal the
    Fig. 1B diagonal (via SI Table S1's CSV, when present)."""
    r = results.pivot(index="field", columns="measure", values="unweighted")
    own_top5, resp = _G["own_top5"], _G["resp"]
    top5 = pd.read_csv(common.PUBLIC / "field_top5.csv").groupby("field").votes.sum()
    for f in common.FIELDS:
        cast = sum(len(own_top5[u]) for u in resp.loc[resp.field == f, "user_db_id"] if u in own_top5)
        assert abs(r.loc[f, "top5"] - 100 * top5[f] / cast) < 1e-9, f
    msg = "  unweighted top-5 agreement equals Fig. 2's (field_top5.csv)"
    path = common.RESULTS / "subfield_consensus.csv"
    if path.exists():
        ref = pd.read_csv(path).set_index("field")
        for f in common.FIELDS:
            assert abs(r.loc[f, "ovl"] - ref.loc[f, "ovl_field_full"]) < 1e-9, f
        msg += "; unweighted overlap equals the Fig. 1B diagonal (SI Table S1)"
    print(msg)


def run_raking(marg):
    _init()
    cons = _G["cons"]
    W = raking_weights(_G["resp"], marg)
    rng = np.random.default_rng(SEED_RAKING)
    rows, diag = [], []
    for f, d in W.groupby("field"):
        users, w1, w2 = list(d.user_db_id), np.ones(len(d)), d.weight.to_numpy()
        counts = Counter(v for u in users for v in cons.get(u, ()))
        top = [v for v, _ in counts.most_common(3)]
        db = d[d.venue_bug == False]  # noqa: E712  (the menu error affected only aspirations)
        flag = db.top_init_venue_db_id.mode().iloc[0]
        vals = {"acc100": (acc100(users), acc100(users, w2, rng)),
                "ovl": (ovl_weighted(users, w1), ovl_weighted(users, w2)),
                "top3": (top3_weighted(users, w1, top), top3_weighted(users, w2, top)),
                "top5": (top5_weighted(users, w1), top5_weighted(users, w2)),
                "flagship": (flagship_weighted(db, np.ones(len(db)), flag), flagship_weighted(db, db.weight.to_numpy(), flag))}
        for m, (a, b) in vals.items():
            rows.append({"field": f, "n": len(users), "measure": m, "unweighted": a, "weighted": b, "diff": b - a})
        k = d[d.g != "unk"]
        pb = d[d.p != "-1"]
        diag.append({"field": f, "n": len(d), "w_min": d.weight.min(), "w_max": d.weight.max(),
                     "deff": 1 + d.weight.var(ddof=0), "ess_over_n": 1 / (1 + d.weight.var(ddof=0)),
                     "women_share_unweighted": (k.g == "gf").mean(),
                     "women_share_weighted": k.loc[k.g == "gf", "weight"].sum() / k.weight.sum(),
                     "women_share_target": 1 - marg.loc[f, "men_share_known"],
                     "top_decile_share_unweighted": (pb.p == "10").mean(),
                     "top_decile_share_weighted": pb.loc[pb.p == "10", "weight"].sum() / pb.weight.sum(),
                     "cap_binds": int((d.weight >= W_CAP * 0.999).sum())})
    results = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(diag)
    check_unweighted_values(results)
    results.to_csv(common.RESULTS / "raked_consensus.csv", index=False)
    diagnostics.to_csv(common.RESULTS / "raking_diagnostics.csv", index=False)
    summary = pd.DataFrame([{"measure": m, "spearman_field_order": spearmanr(g.unweighted, g.weighted)[0],
                             "mean_abs_diff": g["diff"].abs().mean(), "max_abs_diff": g["diff"].abs().max(),
                             "field_max": g.loc[g["diff"].abs().idxmax(), "field"],
                             "n_lower": int((g["diff"] < 0).sum()), "n_higher": int((g["diff"] > 0).sum()),
                             "range_between_fields": g.unweighted.max() - g.unweighted.min()}
                            for m, g in results.groupby("measure")]).set_index("measure")
    return results, diagnostics, summary


# ═══════════════════════════════════════════════════════════════════════════════
# Table S10: pooled models weighted by AARC field share
# ═══════════════════════════════════════════════════════════════════════════════
def field_weights(field_series, marg):
    """AARC field share / sample field share per respondent, normalized to mean 1."""
    s = field_series.value_counts(normalize=True)
    w = field_series.map(marg["share"] / s)
    return w / w.mean()


def fit_weighted(formula, data, weights, cluster=common.CLUSTER):
    """WLS counterpart of common.fit_clustered: SEs clustered on `cluster`, t reference distribution."""
    return smf.wls(formula, data=data, weights=weights).fit(cov_type="cluster", cov_kwds={"groups": data[cluster]}, use_t=True)


def fit(formula, data, weights=None):
    return common.fit_clustered(formula, data) if weights is None else fit_weighted(formula, data, weights)


def term_row(m, t):
    lo, hi = m.conf_int().loc[t]
    return {"estimate": m.params[t], "se": m.bse[t], "ci_lower": lo, "ci_upper": hi, "p": m.pvalues[t]}


def run_field_weights(marg):
    rows = []
    # ── Eq. S1 with field fixed effects and the Fig. 4A/B contrasts ──────────────
    df = common.eqs1_sample()
    df["w"] = field_weights(df.field, marg)
    for outcome in ["top_aspiration", "top_preference"]:
        y = common.OUTCOME_COLS[outcome]
        d = df.dropna(subset=[y])
        f = common.eqs1_formula(outcome, field_fe=True)
        for scheme, w in [("unweighted", None), ("field_weighted", d.w)]:
            m = fit(f, d, w)
            for t, lab in [("associate_prof_dummy", "Associate"), ("full_prof_dummy", "Full"),
                           (common.PRESTIGE, "Prestige"), ("gender", "Gender")]:
                rows.append({"analysis": "eqS1_pooled_fieldFE", "outcome": outcome, "term": lab, "scheme": scheme,
                             **term_row(m, t), "n": int(m.nobs), "n_clusters": d[common.CLUSTER].nunique()})
            dg = d.assign(women=(d.gender == 1).astype(float))
            m = fit(f"{y} ~ women", dg, None if w is None else dg.w)
            rows.append({"analysis": "fig4_comparison", "outcome": outcome, "term": "women_minus_men", "scheme": scheme,
                         **term_row(m, "women"), "n": int(m.nobs), "n_clusters": dg[common.CLUSTER].nunique()})
            dd = d[d[common.PRESTIGE].isin([1, 10])].assign(top=lambda x: (x[common.PRESTIGE] == 10).astype(float))
            m = fit(f"{y} ~ top", dd, None if w is None else dd.w)
            rows.append({"analysis": "fig4_comparison", "outcome": outcome, "term": "decile10_minus_decile1", "scheme": scheme,
                         **term_row(m, "top"), "n": int(m.nobs), "n_clusters": dd[common.CLUSTER].nunique()})

    # ── Fig. 5: stacked respondent-level model y ~ C(ref) * (prestige + C(field)) ──
    long = common.fig5_long()
    resp5 = long[long.ref == "own"].copy()
    resp5["w"] = field_weights(resp5.field, marg)
    long["w"] = long.user_db_id.map(resp5.set_index("user_db_id").w)

    def X(m, nd):
        return np.asarray(build_design_matrices([m.model.data.design_info], nd)[0])

    for scheme in ["unweighted", "field_weighted"]:
        m = fit("y ~ C(ref) * (prestige + C(field))", long, None if scheme == "unweighted" else long.w)
        aw = None if scheme == "unweighted" else resp5.w.to_numpy()
        avg = lambda M: np.average(M, axis=0, weights=aw)  # noqa: E731
        L = {}
        for ref in ("own", "field"):
            nd = resp5.assign(ref=ref)
            L[("slope", ref)] = avg(X(m, nd.assign(prestige=1.0))) - avg(X(m, nd.assign(prestige=0.0)))
            L[("pred10", ref)] = avg(X(m, nd.assign(prestige=10.0)))
        L[("diff10", "own-field")] = L[("pred10", "own")] - L[("pred10", "field")]
        for (q, ref), vec in L.items():
            t = m.t_test(vec.reshape(1, -1))
            lo, hi = np.asarray(t.conf_int(alpha=0.05)).ravel()
            rows.append({"analysis": "fig5_pooled", "outcome": ref, "term": q, "scheme": scheme,
                         "estimate": float(np.ravel(t.effect)[0]), "se": float(np.ravel(t.sd)[0]),
                         "ci_lower": lo, "ci_upper": hi, "p": float(np.ravel(t.pvalue)[0]),
                         "n": len(resp5), "n_clusters": resp5[common.CLUSTER].nunique()})
    models = pd.DataFrame(rows)
    models.to_csv(common.RESULTS / "field_weighted_models.csv", index=False)

    fw = pd.DataFrame({"aarc_n_2023": marg.N, "aarc_share": marg.share,
                       "fig4_n": df.field.value_counts(), "fig4_weight": df.groupby("field").w.first(),
                       "fig5_n": resp5.field.value_counts(), "fig5_weight": resp5.groupby("field").w.first()}).loc[common.FIELDS]
    fw.index.name = "field"
    fw.to_csv(common.RESULTS / "field_weights.csv")
    return models, fw


# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX tables and the numbers quoted in the SI text
# ═══════════════════════════════════════════════════════════════════════════════
def p_fmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def neg(x):
    """Typeset minus signs in numbers: -0.12 -> $-$0.12."""
    return re.sub(r"(?<![A-Za-z$])-(?=\d)", "$-$", x)


def est_ci(r, k=3):
    return neg(f"{r.estimate:.{k}f} [{r.ci_lower:.{k}f}, {r.ci_upper:.{k}f}]")


def write_table8(summary, n_assistant_range):
    S = summary
    rows = []
    for sg, lab in SUBGROUP_LABELS.items():
        s = S[S.subgroup == sg].set_index("measure")
        cell = lambda m, k=1: neg(f"{round(s.loc[m, 'mean_value_minus_null'], k) + 0.0:+.{k}f} "  # noqa: E731
                                  f"({int(s.loc[m, 'fields_below_null'])}/{int(s.loc[m, 'fields_above_null'])})")
        rk = s.loc["rank_rho"]
        rho_min = s[["spearman_field_order_vs_all"]].dropna().spearman_field_order_vs_all.min()
        rows.append(f"{lab} & {int(s.loc['ovl', 'n_fields'])} & {s.loc['ovl', 'median_n']:.0f} & {cell('acc')} & {cell('ovl')} & "
                    f"{cell('top5')} & {rk.mean_value:.2f} / {rk.mean_null:.2f} ({int(rk.fields_below_null)}/{int(rk.fields_above_null)}) & "
                    f"{rho_min:.2f} \\\\")
    lo, hi = n_assistant_range
    # The paper's Table S8 labels the accumulation column "Accuracy".
    tex = r"""\begin{table}[h]
\centering\small
\begin{tabular}{lrrccccc}
\toprule
Subgroup & Fields & Median $n$ & Accuracy & Overlap & Top-5 & Ranking $\rho$ & $\rho_{\min}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Field consensus within respondent subgroups, compared with random subsets of the same size.} For each field and subgroup we recomputed venue accumulation at $N=""" + str(N_ACC_SUBGROUP) + r"""$ consideration sets (Accuracy; fewer venues = more consensus), mean consideration-set overlap (Overlap, Fig.~1B), leave-one-out top-5 agreement (Top-5, Fig.~2), and the SpringRank field ranking fit to the subgroup's comparisons alone. Each value is compared with """ + str(N_NULL) + r""" random subsets of the field's respondents of the same size. Accuracy, Overlap and Top-5 columns give the mean over fields of (subgroup value $-$ mean of the null), with the number of fields in which the subgroup falls below / above the central 95\% of the null in parentheses. Ranking: mean Spearman correlation between the subgroup's ranking and the all-respondent ranking / the same for random subsets (fields below / above the null band). $\rho_{\min}$: the smallest Spearman correlation, across the three consensus measures, between the subgroup's ordering of the fields and the all-respondent ordering. Assistant professors are too few per field (""" + f"{lo}--{hi}" + r""") to analyze.}
\label{stab:subgroup_consensus}
\end{table}
"""
    (common.TABLES / "si_table8.tex").write_text(tex)


def write_table9(results, diagnostics, summary):
    A = diagnostics.set_index("field")
    pv = results.pivot(index="field", columns="measure", values=["unweighted", "weighted"])
    rows = []
    for f in common.FIELDS:
        a = A.loc[f]
        c = lambda m, k=1: f"{pv.loc[f, ('unweighted', m)]:.{k}f} & {pv.loc[f, ('weighted', m)]:.{k}f}"  # noqa: E731
        rows.append(f"{f} & {int(a.n)} & {a.deff:.2f} & {100 * a.women_share_unweighted:.0f} $\\to$ {100 * a.women_share_weighted:.0f} & "
                    f"{c('acc100', 0)} & {c('ovl')} & {c('top3')} & {c('top5')} & {c('flagship')} \\\\")
    sp = {m: f"{summary.loc[m, 'spearman_field_order']:.2f}" for m in RAKED_MEASURES}
    tex = r"""\begin{table}[h]
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrcrrrrrrrrrr}
\toprule
 & & & & \multicolumn{2}{c}{Acc.$_{100}$} & \multicolumn{2}{c}{Overlap} & \multicolumn{2}{c}{Top-3} & \multicolumn{2}{c}{Top-5} & \multicolumn{2}{c}{Flagship} \\
\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}
Field & $n$ & Deff & \% Women & u & w & u & w & u & w & u & w & u & w \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
\multicolumn{4}{l}{Spearman, field order u vs w} & \multicolumn{2}{c}{""" + sp["acc100"] + r"""} & \multicolumn{2}{c}{""" + sp["ovl"] + r"""} & \multicolumn{2}{c}{""" + sp["top3"] + r"""} & \multicolumn{2}{c}{""" + sp["top5"] + r"""} & \multicolumn{2}{c}{""" + sp["flagship"] + r"""} \\
\bottomrule
\end{tabular}
\caption{\textbf{Field-level consensus measures after raking each field's respondents to AARC gender shares and to uniform institutional prestige.} Within each field, weights are obtained by iterative proportional fitting to the AARC 2023 share of women among faculty of known gender and to equal shares across the prestige deciles (which have equal headcount by construction), normalized to mean 1 and capped at """ + f"{W_CAP:.0f}" + r""". Respondents of unknown gender or without a prestige decile keep their own share. Deff: design effect $1+\mathrm{CV}^2(w)$. Women: \% women among respondents of known gender, unweighted $\to$ raked. For each measure the unweighted value (u) is the published value (Figs.~1--2) and w is the raked value. Acc.$_{100}$: expected distinct venues in """ + str(N_ACC_RAKED) + r""" consideration sets (Fig.~1A; raked value by Monte Carlo sampling in proportion to the weights, """ + f"{N_MC:,}" + r""" draws). Overlap: Fig.~1B. Top-3: mean \% selecting the field's three most-selected venues (Fig.~1C). Top-5: Fig.~2. Flagship: \% whose top aspiration is the field's modal top aspiration.}
\label{stab:raked_consensus}
\end{table}
"""
    (common.TABLES / "si_table9.tex").write_text(tex)


def write_table10(models, fw):
    k4 = models[models.analysis != "fig5_pooled"].set_index(["analysis", "outcome", "term", "scheme"])
    k5 = models[models.analysis == "fig5_pooled"].set_index(["term", "outcome", "scheme"])
    deff = lambda w, n: 1 + np.average((w - np.average(w, weights=n)) ** 2, weights=n)  # noqa: E731
    deff4, deff5 = deff(fw.fig4_weight, fw.fig4_n), deff(fw.fig5_weight, fw.fig5_n)
    n4, n5 = int(fw.fig4_n.sum()), int(fw.fig5_n.sum())
    OL = {"top_aspiration": "Top aspiration", "top_preference": "Top preference"}
    TL = {"Prestige": "Prestige decile", "Gender": "Woman", "Associate": "Associate prof.", "Full": "Full prof.",
          "women_minus_men": "Women $-$ men (Fig.~4A/B)", "decile10_minus_decile1": "Decile 10 $-$ decile 1 (Fig.~4A/B)"}
    rows = []
    for o in ["top_aspiration", "top_preference"]:
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{OL[o]}}}, Eq.~S1 with field fixed effects}} \\\\")
        for a, t in [("eqS1_pooled_fieldFE", x) for x in ["Prestige", "Gender", "Associate", "Full"]] + \
                    [("fig4_comparison", x) for x in ["women_minus_men", "decile10_minus_decile1"]]:
            u, wv = k4.loc[(a, o, t, "unweighted")], k4.loc[(a, o, t, "field_weighted")]
            k = 4 if a == "eqS1_pooled_fieldFE" else 3
            rows.append(f"\\quad {TL[t]} & {est_ci(u, k)} & {p_fmt(u.p)} & {est_ci(wv, k)} & {p_fmt(wv.p)} \\\\")
    rows.append(r"\multicolumn{5}{l}{\emph{Fig.~5, pooled respondent-level model} (share of five venues)} \\")
    for (q, ref), lab in {("slope", "own"): "Prestige slope, own top 5", ("slope", "field"): "Prestige slope, field top 5",
                          ("pred10", "own"): "Predicted at decile 10, own top 5", ("pred10", "field"): "Predicted at decile 10, field top 5",
                          ("diff10", "own-field"): "Own $-$ field at decile 10"}.items():
        u, wv = k5.loc[(q, ref, "unweighted")], k5.loc[(q, ref, "field_weighted")]
        rows.append(f"\\quad {lab} & {est_ci(u)} & {p_fmt(u.p)} & {est_ci(wv)} & {p_fmt(wv.p)} \\\\")
    tex = r"""\begin{table}[h]
\centering\footnotesize
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{Unweighted (published)} & \multicolumn{2}{c}{Field-weighted} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
 & Estimate [95\% CI] & $p$ & Estimate [95\% CI] & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Pooled (``Academia'') estimates with each field weighted by its share of AARC faculty.}
Weights are the field's AARC 2023 share divided by its share of the analysis sample, normalised to mean 1 (range """ + f"{fw.fig4_weight.min():.2f}, {fw.fig4_weight.idxmin()}, to {fw.fig4_weight.max():.2f}, {fw.fig4_weight.idxmax()}" + r"""; design effect """ + f"{deff4:.2f}" + r""" for Fig.~4, effective sample size """ + f"{n4 / deff4:,.0f} of {n4:,}; {deff5:.2f}" + r""" for Fig.~5, """ + f"{n5 / deff5:,.0f} of {n5:,}" + r"""). Estimates with 95\% confidence intervals and $p$-values from OLS (unweighted) or WLS (field-weighted) with standard errors clustered by institution and $t$ reference distributions. Prestige is the AARC 2023 decile (10 = most prestigious); outcomes are field-normalised ranks (higher = higher-ranked venue). Fig.~5 quantities are averaged over respondents, weighted by the field weights in the weighted model.}
\label{stab:field_weighted}
\end{table}
"""
    (common.TABLES / "si_table10.tex").write_text(tex)
    return deff4, deff5


def print_text_numbers(sub_long, sub_summary, rake_diag, rake_summary, models, fw, deff4, deff5):
    """The numbers quoted in the SI text of 'Robustness to Sample Composition' (Tables S8-S10)."""
    S = sub_summary.set_index(["measure", "subgroup"])
    sp = sub_summary[sub_summary.measure.isin(["acc", "ovl", "top5"])].spearman_field_order_vs_all
    rr = sub_summary[sub_summary.measure == "rank_rho"]
    w = sub_long[sub_long.measure == "top5"].pivot(index="field", columns="subgroup", values="value")
    print("\nNumbers quoted in the SI text")
    print("Subgroups (Table S8)")
    print(f"  random subsets per subgroup and field: {N_NULL}")
    print(f"  Spearman, subgroup field ordering vs all-respondent ordering (3 measures x 6 subgroups): {sp.min():.2f} to {sp.max():.2f}")
    print(f"  subgroup-field rankings outside the central 95% of the null: "
          f"{int(rr.fields_below_null.sum() + rr.fields_above_null.sum())} of {int(rr.n_fields.sum())}")
    for sg in ["women", "prestige_bottom", "men", "prestige_top"]:
        print(f"  {SUBGROUP_LABELS[sg]:24s} top-5 minus null {S.loc[('top5', sg), 'mean_value_minus_null']:+.1f}, "
              f"overlap minus null {S.loc[('ovl', sg), 'mean_value_minus_null']:+.1f}, "
              f"ranking rho {S.loc[('rank_rho', sg), 'mean_value']:.2f} vs null {S.loc[('rank_rho', sg), 'mean_null']:.2f}")
    print(f"  fields with >= {N_ACC_SUBGROUP} bottom-tercile respondents: {int(S.loc[('ovl', 'prestige_bottom'), 'n_fields'])} of 13")
    print(f"  fields where women's top-5 agreement is below men's: {int(((w.women - w.men) < 0).sum())}; "
          f"bottom tercile below top tercile: {int(((w.prestige_bottom - w.prestige_top).dropna() < 0).sum())}")

    print("Raking (Table S9)")
    print(f"  design effects: {rake_diag.deff.min():.2f} to {rake_diag.deff.max():.2f}; weight cap binds for "
          f"{int(rake_diag.cap_binds.sum())} respondents")
    for m, lab in [("top5", "top-5 agreement"), ("ovl", "overlap"), ("acc100", "venues in 100 sets"),
                   ("top3", "top-3 share"), ("flagship", "flagship share")]:
        r = rake_summary.loc[m]
        print(f"  {lab:20s} lower in {int(r.n_lower)} / higher in {int(r.n_higher)} fields; mean |change| {r.mean_abs_diff:.1f}, "
              f"max {r.max_abs_diff:.1f} ({r.field_max}); Spearman u vs w {r.spearman_field_order:.2f}; "
              f"between-field range {r.range_between_fields:.0f}")

    print("Field weights (Table S10)")
    print(f"  weights {fw.fig4_weight.min():.2f} ({fw.fig4_weight.idxmin()}) to {fw.fig4_weight.max():.2f} ({fw.fig4_weight.idxmax()}); "
          f"design effect {deff4:.2f} (ESS {fw.fig4_n.sum() / deff4:,.0f} of {fw.fig4_n.sum():,}) for Fig. 4, "
          f"{deff5:.2f} (ESS {fw.fig5_n.sum() / deff5:,.0f} of {fw.fig5_n.sum():,}) for Fig. 5")
    k = models.set_index(["analysis", "outcome", "term", "scheme"])
    for o in ["top_aspiration", "top_preference"]:
        for a, t in [("eqS1_pooled_fieldFE", x) for x in ["Prestige", "Gender", "Associate", "Full"]] + \
                    [("fig4_comparison", x) for x in ["women_minus_men", "decile10_minus_decile1"]]:
            u, wv = k.loc[(a, o, t, "unweighted")], k.loc[(a, o, t, "field_weighted")]
            d = 4 if a == "eqS1_pooled_fieldFE" else 3
            extra = f"; magnitude change {100 * (abs(wv.estimate) - abs(u.estimate)) / abs(u.estimate):+.0f}%" if d == 4 else ""
            print(f"  {o:15s} {t:24s} {u.estimate:.{d}f} (p {p_fmt(u.p)}) -> {wv.estimate:.{d}f} (p {p_fmt(wv.p)}){extra}")
    for q, ref, lab in [("slope", "own", "prestige slope, own top 5"), ("slope", "field", "prestige slope, field top 5"),
                        ("pred10", "own", "predicted at decile 10, own"), ("pred10", "field", "predicted at decile 10, field"),
                        ("diff10", "own-field", "own - field at decile 10")]:
        u, wv = k.loc[("fig5_pooled", ref, q, "unweighted")], k.loc[("fig5_pooled", ref, q, "field_weighted")]
        print(f"  Fig. 5 {lab:30s} {u.estimate:.3f} -> {wv.estimate:.3f} (p {p_fmt(wv.p)})")
    u, wv = k.loc[("fig5_pooled", "own-field", "diff10", "unweighted")], k.loc[("fig5_pooled", "own-field", "diff10", "field_weighted")]
    print(f"  Fig. 5 own - field gap in venues (x5): {5 * u.estimate:.2f} -> {5 * wv.estimate:.2f}")


def main():
    pd.set_option("display.width", 220)
    marg = aarc_margins()

    t0 = time.time()
    print("Table S8: consensus within subgroups vs random subsets of the same size")
    sub_long, sub_summary = run_subgroups()
    # The caption's count of assistant professors per field is, as in the paper, taken over the Eq. S1
    # sample (professors of known gender with a prestige decile, unaffected by the aspiration-menu
    # error). Among all respondents of a field, the sample Table S8 uses, the range is wider.
    asst = lambda d: d[d.career_stage == "Assistant Professor"].groupby("field").size().reindex(common.FIELDS).fillna(0).astype(int)  # noqa: E731
    n_asst_eqs1, n_asst_all = asst(common.eqs1_sample()), asst(_G["resp"])
    write_table8(sub_summary, (int(n_asst_eqs1.min()), int(n_asst_eqs1.max())))
    print(sub_summary.round(3).to_string(index=False))
    print(f"  assistant professors per field: {n_asst_eqs1.min()}-{n_asst_eqs1.max()} in the Eq. S1 sample (caption), "
          f"{n_asst_all.min()}-{n_asst_all.max()} among all respondents")
    print(f"  ({time.time() - t0:.0f} s)")

    t0 = time.time()
    print("\nTable S9: raking to AARC gender shares and uniform prestige deciles")
    rake_results, rake_diag, rake_summary = run_raking(marg)
    write_table9(rake_results, rake_diag, rake_summary)
    print(rake_diag.round(3).to_string(index=False))
    print(rake_results.pivot(index="field", columns="measure", values="diff").round(2).to_string())
    print(rake_summary.round(3).to_string())
    print(f"  ({time.time() - t0:.0f} s)")

    t0 = time.time()
    print("\nTable S10: pooled models weighted by AARC field share")
    models, fw = run_field_weights(marg)
    deff4, deff5 = write_table10(models, fw)
    print(fw.round(3).to_string())
    print(models.round(4).to_string(index=False))
    print(f"  ({time.time() - t0:.0f} s)")

    print_text_numbers(sub_long, sub_summary, rake_diag, rake_summary, models, fw, deff4, deff5)
    print(f"\nwrote {common.TABLES / 'si_table8.tex'}, si_table9.tex, si_table10.tex and the CSVs in {common.RESULTS}")


if __name__ == "__main__":
    main()
