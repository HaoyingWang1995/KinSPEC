"""Is the higher cross-fold agreement of MCA a property of the REPRESENTATION,
or of the back-projection operator used to map it onto the 85 KLIFS positions?

PSM  : 1955 column gains -> 85 bins, each column feeds exactly one position.
MCA  : 200 axis gains -> (fixed 2113x200 contribution matrix) -> 2113 categories -> 85 bins.
       Every axis spreads over many positions with FIXED weights, identical in every fold.

Null: replace the model's importance vector with a random one drawn from the same marginal
distribution (a permutation of the real gains), push it through the SAME aggregation, and
measure fold-pair agreement. That is the floor the operator imposes on its own.
If MCA's null floor is already ~0.95, the measurement says nothing about the model.
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

    F = str(config.final_dir); EMB = f"{F}/feature_embedding/Kinase_test_MCA"
    PSMM = str(config.psm_model_dir)
    BANDS = ["10_50", "20_60", "30_70", "40_80", "50_90", "60_100"]
    rng = np.random.default_rng(42)

    C = pd.read_csv(f"{EMB}/MCA_200D_category_contributions.csv", index_col=0)
    cat_res = np.array([int(s.split("_")[0][3:]) - 1 for s in C.index]); Cm = C.to_numpy()
    col_res = np.arange(1955) // 23

    def to_pos(w, grp):
        o = np.zeros(85); np.add.at(o, grp, w)
        return o / o.sum() if o.sum() > 0 else o

    def gain(path, nk):
        b = xgb.Booster(); load_model(b, path, config)
        g = b.get_score(importance_type="total_gain"); v = np.zeros(417 + nk)
        for k, val in g.items(): v[int(str(k).lstrip("f"))] = val
        del b
        return v[417:]

    def agree(P):
        sp, ov = [], []
        for a in range(len(P)):
            for b in range(a + 1, len(P)):
                sp.append(stats.spearmanr(P[a], P[b]).statistic)
                ov.append(len(set(np.argsort(P[a])[::-1][:10]) & set(np.argsort(P[b])[::-1][:10])) / 10)
        return np.mean(sp), np.mean(ov)

    print("=" * 90)
    print("How much of each axis's / column's gain reaches how many of the 85 positions?")
    print("=" * 90)
    # spread of one MCA axis over positions
    S = np.zeros((200, 85))
    for k in range(200):
        S[k] = to_pos(Cm[:, k], cat_res)
    eff = np.exp(-(np.where(S > 0, S * np.log(np.where(S > 0, S, 1)), 0)).sum(1))
    print(f"  one MCA axis reaches, in effective terms, {eff.mean():.1f} of 85 positions "
          f"(min {eff.min():.1f}, max {eff.max():.1f})")
    print(f"  one PSM column reaches exactly 1 of 85 positions")
    print(f"  -> the MCA route averages each fold's profile over ~{eff.mean():.0f} positions per axis;")
    print(f"     any two folds are therefore pushed toward the same profile by construction.")

    print("\n" + "=" * 90)
    print("NULL CONTROL: random importance vectors through the same aggregation")
    print("(gains permuted within each fold -> same marginal distribution, no model information)")
    print("=" * 90)
    print(f"  {'band':<9}{'':<4}{'PSM real':>10}{'PSM null':>10}{'':<4}{'MCA real':>10}{'MCA null':>10}")
    rows = []
    for band in BANDS:
        gp = [gain(f"{PSMM}/KinSPEC_MCA_training_band_{band}_fold{f}_pure_PSM_model_v4.json", 1955)
              for f in range(1, 6)]
        gm = [gain(f"{F}/model/KinSPEC_MCA_training_band_{band}_fold{f}_model_v4.json", 200)
              for f in range(1, 6)]
        Pr = [to_pos(g, col_res) for g in gp]
        Mr = [to_pos(Cm @ g, cat_res) for g in gm]
        # null: permute the gain vector within each fold
        Pn = [to_pos(rng.permutation(g), col_res) for g in gp]
        Mn = [to_pos(Cm @ rng.permutation(g), cat_res) for g in gm]
        (sp_r, ov_pr), (sp_n, ov_pn) = agree(Pr), agree(Pn)
        (sm_r, ov_mr), (sm_n, ov_mn) = agree(Mr), agree(Mn)
        rows.append((band, ov_pr, ov_pn, ov_mr, ov_mn, sp_r, sp_n, sm_r, sm_n))
        print(f"  {band:<9}{'top10':<4}{ov_pr:>10.2f}{ov_pn:>10.2f}{'':<4}{ov_mr:>10.2f}{ov_mn:>10.2f}")
    d = pd.DataFrame(rows, columns=["band", "P_real", "P_null", "M_real", "M_null",
                                    "Psp_real", "Psp_null", "Msp_real", "Msp_null"])
    print(f"  {'MEAN':<9}{'top10':<4}{d.P_real.mean():>10.2f}{d.P_null.mean():>10.2f}"
          f"{'':<4}{d.M_real.mean():>10.2f}{d.M_null.mean():>10.2f}")
    print(f"  {'':<9}{'rho':<4}{d.Psp_real.mean():>10.3f}{d.Psp_null.mean():>10.3f}"
          f"{'':<4}{d.Msp_real.mean():>10.3f}{d.Msp_null.mean():>10.3f}")
    print()
    print(f"  signal above the operator's own floor (real - null, top-10 overlap):")
    print(f"    pure PSM  {d.P_real.mean()-d.P_null.mean():+.3f}")
    print(f"    MCA       {d.M_real.mean()-d.M_null.mean():+.3f}")
    print(f"  headroom available (1 - null):  pure PSM {1-d.P_null.mean():.3f}   MCA {1-d.M_null.mean():.3f}")
    print(f"  fraction of available headroom used:")
    print(f"    pure PSM  {(d.P_real.mean()-d.P_null.mean())/(1-d.P_null.mean())*100:.1f}%")
    print(f"    MCA       {(d.M_real.mean()-d.M_null.mean())/(1-d.M_null.mean())*100:.1f}%")


if __name__ == "__main__":
    single_main("null_check")
