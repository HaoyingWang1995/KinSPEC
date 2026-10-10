"""Experiment 3: project each MCA axis's XGBoost gain back onto the 1955 raw PSM columns
via the MCA category contributions, so pure PSM and MCA can be compared in one coordinate system."""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np, pandas as pd, xgboost as xgb, json, os
    from scipy import stats

    FINAL = str(config.final_dir); EMB = f"{FINAL}/feature_embedding/Kinase_test_MCA"
    OUT = str(config.output_dir)
    BANDS = ["10_50", "20_60", "30_70", "40_80", "50_90", "60_100"]
    LAB = ["rare<1%", "uncommon1-5%", "common5-20%", "promisc>20%"]

    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy(np.float32)
    prev = psm.mean(0); strat = np.digitize(prev, [.01, .05, .20])
    meta = json.load(open(f"{OUT}/aa23.json"))
    coarse = np.array(meta["spec"])[np.tile(np.arange(23), 85)]

    C = pd.read_csv(f"{EMB}/MCA_200D_category_contributions.csv", index_col=0)
    lk = pd.read_csv(f"{EMB}/MCA_200D_category_lookup.csv")
    lk["row"] = lk.category_name.str.replace("_", "", regex=False).map(
        {s.replace("_", ""): i for i, s in enumerate(C.index)})
    lk = lk.dropna(subset=["row"])
    has = lk.dropna(subset=["bit_index_0_based"]).copy(); has["bit"] = has.bit_index_0_based.astype(int)
    rowbit = np.full(len(C), -1); rowbit[has.row.astype(int).values] = has.bit.values
    isgap = np.array([s.endswith("__Gap") for s in C.index])
    Cm = C.to_numpy()
    print(f"categories mapped to a PSM column: {(rowbit>=0).sum()};  Gap categories: {isgap.sum()}")

    res = []
    for band in BANDS:
        for fold in range(1, 6):
            b = xgb.Booster(); load_model(b, f"{FINAL}/model/KinSPEC_MCA_training_band_{band}_fold{fold}_model_v4.json", config)
            g = b.get_score(importance_type="total_gain"); v = np.zeros(617)
            for k, val in g.items(): v[int(k)] = val
            w = Cm @ v[417:]; w = w / w.sum()
            col = np.zeros(1955); m = rowbit >= 0
            np.add.at(col, rowbit[m], w[m]); col = col / col.sum()
            r = {"band": band, "fold": fold, "gap_share": float(w[isgap].sum()),
                 "coarse>=7": col[coarse >= 7].sum(), "fine<=2": col[coarse <= 2].sum(),
                 "sp_prev": stats.spearmanr(col, prev).statistic}
            for i, l in enumerate(LAB): r[l] = col[strat == i].sum()
            res.append(r); del b
    d = pd.DataFrame(res); d.to_csv(f"{OUT}/mech_mca_backproj.csv", index=False)

    print("\nMCA effective attention on the 1955 PSM columns (mean over 30 models):")
    for i, l in enumerate(LAB):
        print(f"  {l:<14} {d[l].mean()*100:5.1f}%   [cols {(strat==i).mean()*100:5.1f}%]  "
              f"enrichment {d[l].mean()/(strat==i).mean():.2f}x")
    print(f"  coarse>=7/20   {d['coarse>=7'].mean()*100:5.1f}%   [cols {(coarse>=7).mean()*100:5.1f}%]  "
          f"enrichment {d['coarse>=7'].mean()/(coarse>=7).mean():.2f}x")
    print(f"  fine  <=2/20   {d['fine<=2'].mean()*100:5.1f}%   [cols {(coarse<=2).mean()*100:5.1f}%]  "
          f"enrichment {d['fine<=2'].mean()/(coarse<=2).mean():.2f}x")
    print(f"  Spearman(attention, prevalence) = {d.sp_prev.mean():+.3f}")
    print(f"  share routed to Gap categories  = {d.gap_share.mean()*100:.1f}%")


if __name__ == "__main__":
    single_main("backproj")
