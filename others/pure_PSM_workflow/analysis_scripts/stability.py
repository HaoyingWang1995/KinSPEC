"""Redundancy has a concrete cost that does not show up in AUROC: attribution instability.

When the same fact is encoded by several perfectly correlated columns, a greedy tree picks
among them arbitrarily, so the credit assigned to a pocket position is split unpredictably
between runs. MCA has no such freedom -- its axes are unique up to sign.

Both representations are mapped onto the SAME 425 variables (85 residues x 5 property systems)
and the same 85 positions, then compared for cross-fold reproducibility.
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
    import numpy as np, pandas as pd, xgboost as xgb, json
    from scipy import stats

    F = str(config.final_dir); EMB = f"{F}/feature_embedding/Kinase_test_MCA"
    PSMM = str(config.psm_model_dir)
    OUT = str(config.output_dir)
    BANDS = ["10_50", "20_60", "30_70", "40_80", "50_90", "60_100"]

    st = np.load(f"{OUT}/strata.npz", allow_pickle=True)
    var_of, vres, vsys = st["var_of"], st["res"], st["sys"]
    lk = pd.read_csv(f"{EMB}/MCA_200D_category_lookup.csv").dropna(subset=["bit_index_0_based"])
    lk["bit"] = lk.bit_index_0_based.astype(int)
    bit2sys = dict(zip(lk[lk.residue == 1].bit, lk[lk.residue == 1].system))
    vidx = {(r, s): i for i, (r, s) in enumerate(zip(vres, vsys))}
    col_var = np.array([vidx[(1 + c // 23, bit2sys[c % 23])] for c in range(1955)])
    col_res = np.array([c // 23 for c in range(1955)])

    C = pd.read_csv(f"{EMB}/MCA_200D_category_contributions.csv", index_col=0)
    G = pd.read_csv(f"{EMB}/MCA_200D_category_coordinates.csv", index_col=0)
    cat_var = np.array([vidx[(int(s.split("_")[0][3:]), s.split("_")[1])] for s in C.index])
    cat_res = np.array([int(s.split("_")[0][3:]) - 1 for s in C.index])
    Cm = C.to_numpy()

    def agg(w, groups, n):
        o = np.zeros(n); np.add.at(o, groups, w); return o / o.sum()

    rows = []
    for band in BANDS:
        prof = {"PSM": [], "MCA": []}
        for fold in range(1, 6):
            b = xgb.Booster(); load_model(b, f"{PSMM}/KinSPEC_MCA_training_band_{band}_fold{fold}_pure_PSM_model_v4.json", config)
            g = b.get_score(importance_type="total_gain"); v = np.zeros(2372)
            for kk, vv in g.items(): v[int(kk)] = vv
            prof["PSM"].append((agg(v[417:], col_var, 425), agg(v[417:], col_res, 85)))
            del b
            b = xgb.Booster(); load_model(b, f"{F}/model/KinSPEC_MCA_training_band_{band}_fold{fold}_model_v4.json", config)
            g = b.get_score(importance_type="total_gain"); v = np.zeros(617)
            for kk, vv in g.items(): v[int(kk)] = vv
            w = Cm @ v[417:]
            prof["MCA"].append((agg(w, cat_var, 425), agg(w, cat_res, 85)))
            del b
        for tag in ("PSM", "MCA"):
            for lvl, i in (("variable(425)", 0), ("position(85)", 1)):
                P = np.array([p[i] for p in prof[tag]])
                sp, co = [], []
                for a in range(5):
                    for c in range(a + 1, 5):
                        sp.append(stats.spearmanr(P[a], P[c]).statistic)
                        co.append(np.dot(P[a], P[c]) / (np.linalg.norm(P[a]) * np.linalg.norm(P[c])))
                # top-10 position overlap
                ov = []
                if i == 1:
                    for a in range(5):
                        for c in range(a + 1, 5):
                            ov.append(len(set(np.argsort(P[a])[::-1][:10]) & set(np.argsort(P[c])[::-1][:10])) / 10)
                rows.append({"band": band, "rep": tag, "level": lvl,
                             "spearman": np.mean(sp), "cosine": np.mean(co),
                             "top10_overlap": np.mean(ov) if ov else np.nan})
    d = pd.DataFrame(rows); d.to_csv(f"{OUT}/stability.csv", index=False)

    print("=" * 90)
    print("Cross-fold reproducibility of where the model says the pocket matters")
    print("(5 folds per band, 10 fold-pairs; both mapped onto the same variables/positions)")
    print("=" * 90)
    for lvl in ("variable(425)", "position(85)"):
        print(f"\n  aggregated to {lvl}")
        print(f"    {'band':<9}{'PSM rho':>9}{'MCA rho':>9}{'PSM cos':>9}{'MCA cos':>9}", end="")
        print(f"{'PSM top10':>11}{'MCA top10':>11}" if lvl.startswith("position") else "")
        for band in BANDS:
            p = d[(d.band == band) & (d.rep == "PSM") & (d.level == lvl)].iloc[0]
            m = d[(d.band == band) & (d.rep == "MCA") & (d.level == lvl)].iloc[0]
            line = f"    {band:<9}{p.spearman:>9.3f}{m.spearman:>9.3f}{p.cosine:>9.3f}{m.cosine:>9.3f}"
            if lvl.startswith("position"): line += f"{p.top10_overlap:>11.2f}{m.top10_overlap:>11.2f}"
            print(line)
        P = d[(d.rep == "PSM") & (d.level == lvl)]; M = d[(d.rep == "MCA") & (d.level == lvl)]
        t, pv = stats.ttest_rel(M.spearman.values, P.spearman.values)
        print(f"    {'MEAN':<9}{P.spearman.mean():>9.3f}{M.spearman.mean():>9.3f}"
              f"{P.cosine.mean():>9.3f}{M.cosine.mean():>9.3f}", end="")
        if lvl.startswith("position"): print(f"{P.top10_overlap.mean():>11.2f}{M.top10_overlap.mean():>11.2f}")
        else: print()
        print(f"    paired over the 6 bands: MCA higher {(M.spearman.values>P.spearman.values).sum()}/6, p={pv:.4f}")

    # how much of PSM's instability is redundancy-driven?
    print("\n" + "=" * 90)
    print("Is PSM's instability driven by redundant columns?")
    print("=" * 90)
    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy()
    nz = (psm.mean(0) > 0) & (psm.mean(0) < 1)
    Cn = psm[:, nz].T.astype(float)
    Cn = (Cn - Cn.mean(1, keepdims=True)) / (Cn.std(1, keepdims=True) + 1e-12)
    R = np.abs((Cn @ Cn.T) / psm.shape[0])
    np.fill_diagonal(R, 0)
    maxr = np.zeros(1955); maxr[np.where(nz)[0]] = R.max(1)
    b = xgb.Booster(); load_model(b, f"{PSMM}/KinSPEC_MCA_training_band_10_50_fold1_pure_PSM_model_v4.json", config)
    g = b.get_score(importance_type="total_gain"); v = np.zeros(2372)
    for kk, vv in g.items(): v[int(kk)] = vv
    kin = v[417:]
    grp = np.digitize(maxr, [0.5, 0.8, 0.95, 0.999])
    labs = ["max|r|<0.5", "0.5-0.8", "0.8-0.95", "0.95-1.0", "=1.0 (exact duplicate)"]
    print(f"  {'group':<26}{'columns':>9}{'gain share':>12}")
    for i, l in enumerate(labs):
        m = (grp == i) & nz
        print(f"  {l:<26}{m.sum():>9}{kin[m].sum()/kin[nz].sum()*100:>11.1f}%")
    print(f"\n  columns whose fact is also carried by another column at |r|>0.95: "
          f"{int(((maxr>0.95)&nz).sum())} of {int(nz.sum())}, "
          f"holding {kin[(maxr>0.95)&nz].sum()/kin[nz].sum()*100:.1f}% of the kinase gain")


if __name__ == "__main__":
    single_main("stability")
