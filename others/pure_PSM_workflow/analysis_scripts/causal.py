"""Close the causal link: is pure PSM's attribution instability located in the collinear columns?

If a greedy tree is picking arbitrarily among collinear columns, then within a group of mutually
collinear columns the GROUP total gain should be stable across folds while the split among its
MEMBERS should not. That signature is specific to the redundancy mechanism; it cannot be
produced by ordinary sampling noise, which would destabilise group totals too.

Also re-checks the cross-fold stability result under a second importance measure.
"""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this archived analysis with shared paths; import performs no calculation."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print("Archived analysis: interpret results with the original report and its limitations.")
    import numpy as np, pandas as pd, xgboost as xgb
    from scipy import stats
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix

    F = str(config.final_dir); EMB = f"{F}/feature_embedding/Kinase_test_MCA"
    PSMM = str(config.psm_model_dir)
    OUT = str(config.output_dir)
    BANDS = ["10_50", "20_60", "30_70", "40_80", "50_90", "60_100"]

    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy(float)
    nz = np.where((psm.mean(0) > 0) & (psm.mean(0) < 1))[0]
    Xc = psm[:, nz].T
    Xc = (Xc - Xc.mean(1, keepdims=True)) / (Xc.std(1, keepdims=True) + 1e-12)
    R = np.abs(Xc @ Xc.T / psm.shape[0]); np.fill_diagonal(R, 0)

    for THR in (0.999, 0.95):
        A = csr_matrix(R >= THR)
        ncomp, lab = connected_components(A, directed=False)
        sizes = np.bincount(lab)
        multi = np.where(sizes >= 2)[0]
        print(f"collinearity threshold |r| >= {THR}: {len(multi)} groups with >=2 members, "
              f"covering {int(sizes[multi].sum())} of {len(nz)} non-constant columns")
        if THR == 0.999:
            GROUP = np.full(1955, -1); GROUP[nz] = lab
            MULTI = set(multi.tolist())
    print()

    def gains(path, nk):
        b = xgb.Booster(); load_model(b, path, config)
        out = {}
        for imp in ("total_gain", "weight"):
            g = b.get_score(importance_type=imp); v = np.zeros(417 + nk)
            for k, val in g.items(): v[int(str(k).lstrip("f"))] = val
            out[imp] = v[417:]
        del b
        return out

    print("=" * 88)
    print("[A] within perfectly collinear groups (|r|>=0.999): is the GROUP stable but the")
    print("    split among MEMBERS not?   CV = std/mean across the 5 folds")
    print("=" * 88)
    print(f"  {'band':<9}{'group CV':>11}{'member CV':>12}{'ratio':>9}{'groups':>9}{'members':>9}")
    rows = []
    for band in BANDS:
        G = np.array([gains(f"{PSMM}/KinSPEC_MCA_training_band_{band}_fold{f}_pure_PSM_model_v4.json", 1955)["total_gain"]
                      for f in range(1, 6)])
        gcv, mcv, ng, nm = [], [], 0, 0
        for gid in MULTI:
            cols = np.where(GROUP == gid)[0]
            tot = G[:, cols].sum(1)
            if tot.mean() <= 0: continue
            gcv.append(tot.std(ddof=1) / tot.mean()); ng += 1
            for c in cols:
                if G[:, c].mean() > 0:
                    mcv.append(G[:, c].std(ddof=1) / G[:, c].mean()); nm += 1
        rows.append((band, np.mean(gcv), np.mean(mcv), np.mean(mcv) / np.mean(gcv), ng, nm))
        print(f"  {band:<9}{np.mean(gcv):>11.3f}{np.mean(mcv):>12.3f}{np.mean(mcv)/np.mean(gcv):>9.2f}{ng:>9}{nm:>9}")
    d = pd.DataFrame(rows, columns=["band", "group_cv", "member_cv", "ratio", "n_groups", "n_members"])
    t, p = stats.ttest_rel(d.member_cv, d.group_cv)
    print(f"  {'MEAN':<9}{d.group_cv.mean():>11.3f}{d.member_cv.mean():>12.3f}{d.ratio.mean():>9.2f}")
    print(f"  member CV > group CV in {int((d.member_cv>d.group_cv).sum())}/6 bands, paired p={p:.2e}")
    print("\n  -> the information a collinear group carries is used consistently across folds;")
    print("     which member of the group receives the credit is not. That is the mechanism.")

    print("\n" + "=" * 88)
    print("[B] does a column's importance instability track its collinearity?")
    print("=" * 88)
    maxr = np.zeros(1955); maxr[nz] = R.max(1)
    G = np.array([gains(f"{PSMM}/KinSPEC_MCA_training_band_10_50_fold{f}_pure_PSM_model_v4.json", 1955)["total_gain"]
                  for f in range(1, 6)])
    used = (G.mean(0) > 0) & (np.isin(np.arange(1955), nz))
    cv = np.where(used, G.std(0, ddof=1) / np.where(G.mean(0) > 0, G.mean(0), 1), np.nan)
    # control for mean gain: compare within deciles of mean gain
    mg = G.mean(0)
    dec = pd.qcut(pd.Series(mg[used]), 10, labels=False, duplicates="drop")
    sub = pd.DataFrame({"cv": cv[used], "maxr": maxr[used], "dec": dec})
    r_all = stats.spearmanr(sub.cv, sub.maxr).statistic
    r_within = sub.groupby("dec").apply(lambda g: stats.spearmanr(g.cv, g.maxr).statistic, include_groups=False)
    print(f"  Spearman(gain CV across folds, max|r| with another column) = {r_all:+.3f}")
    print(f"  same, computed within deciles of mean gain (removes the size effect): "
          f"{np.nanmean(r_within):+.3f}  (decile range {np.nanmin(r_within):+.2f} to {np.nanmax(r_within):+.2f})")
    print(f"  columns with max|r|>0.999: mean CV {sub[sub.maxr>0.999].cv.mean():.3f}   "
          f"columns with max|r|<0.5: mean CV {sub[sub.maxr<0.5].cv.mean():.3f}")

    print("\n" + "=" * 88)
    print("[C] robustness: does the position-level stability result hold under split COUNT?")
    print("=" * 88)
    C = pd.read_csv(f"{EMB}/MCA_200D_category_contributions.csv", index_col=0)
    cat_res = np.array([int(s.split("_")[0][3:]) - 1 for s in C.index]); Cm = C.to_numpy()
    col_res = np.arange(1955) // 23
    def to_pos(w, grp):
        o = np.zeros(85); np.add.at(o, grp, w); return o / o.sum()
    print(f"  {'importance':<14}{'band':<9}{'PSM top10':>11}{'MCA top10':>11}")
    agg = {"total_gain": [], "weight": []}
    for imp in ("total_gain", "weight"):
        for band in BANDS:
            P = [to_pos(gains(f"{PSMM}/KinSPEC_MCA_training_band_{band}_fold{f}_pure_PSM_model_v4.json", 1955)[imp], col_res)
                 for f in range(1, 6)]
            M = [to_pos(Cm @ gains(f"{F}/model/KinSPEC_MCA_training_band_{band}_fold{f}_model_v4.json", 200)[imp], cat_res)
                 for f in range(1, 6)]
            ov = lambda Z: np.mean([len(set(np.argsort(Z[a])[::-1][:10]) & set(np.argsort(Z[b])[::-1][:10])) / 10
                                    for a in range(5) for b in range(a + 1, 5)])
            agg[imp].append((ov(P), ov(M)))
            print(f"  {imp:<14}{band:<9}{ov(P):>11.2f}{ov(M):>11.2f}")
        a = np.array(agg[imp])
        t, p = stats.ttest_rel(a[:, 1], a[:, 0])
        print(f"  {imp:<14}{'MEAN':<9}{a[:,0].mean():>11.2f}{a[:,1].mean():>11.2f}   "
              f"MCA higher {(a[:,1]>a[:,0]).sum()}/6, p={p:.4f}\n")


if __name__ == "__main__":
    single_main("causal")
