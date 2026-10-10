"""Where does MCA's compound-ranking advantage actually live?

Decompose the VSDS signal (band 10-50, the similarity-controlled setting):
  pooled AUROC        = between-target calibration + within-target compound ranking
  within-target AUROC = compound ranking with target-level offsets removed
and compare each with the kinase block intact vs scrambled across the 482 kinases.
"""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np, pandas as pd, xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from scipy import stats

    F = str(config.final_dir); EMB = f"{F}/feature_embedding/Kinase_test_MCA"
    PSMM = str(config.psm_model_dir)
    OUT = str(config.output_dir)
    rng = np.random.default_rng(42)

    v = pd.read_csv(f"{F}/dataset/VSDS_truedecoy_integrated_data.csv",
                    usecols=["uniprot_id", "KLIFS", "molecule_label"])
    y = v.molecule_label.to_numpy().astype(int); tgt = v.uniprot_id.to_numpy()
    mol = pd.read_csv(f"{F}/dataset/VSDS_truedecoy_LSAE_latent_top95_features_v4.csv",
                      skiprows=1, header=None, dtype=np.float32).to_numpy()
    ref = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    lut = pd.Series(np.arange(len(ref)), index=ref.KLIFS.astype(str).str.strip()).groupby(level=0).last()
    pos = v.KLIFS.astype(str).str.strip().map(lut).to_numpy(np.intp)
    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy(np.float32)
    mca = pd.read_csv(f"{EMB}/MCA_klifs_482_200D_features.csv", skiprows=1, header=None).to_numpy(np.float32)
    FN = lambda n: [str(i) for i in range(n)]
    T = np.unique(tgt); has = [t for t in T if y[tgt == t].sum() > 0 and (y[tgt == t] == 0).sum() > 0]
    print(f"VSDS {len(v)} rows, {y.sum()} actives, {len(has)} usable targets\n")

    def metrics(s):
        pooled = roc_auc_score(y, s)
        within = np.mean([roc_auc_score(y[tgt == t], s[tgt == t]) for t in has])
        # rank-within-target, then pool: isolates calibration
        r = pd.Series(s).groupby(pd.Series(tgt)).rank(pct=True).to_numpy()
        calib_removed = roc_auc_score(y, r)
        return pooled, within, calib_removed

    rows = []
    for fold in range(1, 6):
        for tag, mp, K, nf in (("PSM", f"{PSMM}/KinSPEC_MCA_training_band_10_50_fold{fold}_pure_PSM_model_v4.json", psm, 2372),
                               ("MCA", f"{F}/model/KinSPEC_MCA_training_band_10_50_fold{fold}_model_v4.json", mca, 617)):
            b = xgb.Booster(); load_model(b, mp, config)
            if b.feature_names is None: b.feature_names = FN(nf)
            for cond, Kc in (("intact", K), ("kinase scrambled", K[rng.permutation(482)])):
                X = np.hstack([mol, Kc[pos]])
                s = b.predict(xgb.DMatrix(X, feature_names=FN(nf)))
                p, w, c = metrics(s)
                rows.append({"fold": fold, "rep": tag, "cond": cond,
                             "pooled": p, "within_target": w, "calib_removed": c})
            del b
        print(f"  fold {fold} done", flush=True)
    d = pd.DataFrame(rows); d.to_csv(f"{OUT}/decomp.csv", index=False)

    print("\n" + "=" * 92)
    print("VSDS band 10-50: where the kinase representation acts")
    print("=" * 92)
    print(f"{'condition':<18}{'metric':<16}{'pure PSM':>11}{'MCA':>11}{'diff':>10}{'MCA wins':>10}{'p':>9}")
    for cond in ("intact", "kinase scrambled"):
        for m in ("pooled", "within_target", "calib_removed"):
            a = d[(d.rep == "PSM") & (d.cond == cond)].sort_values("fold")[m].values
            b_ = d[(d.rep == "MCA") & (d.cond == cond)].sort_values("fold")[m].values
            t, p = stats.ttest_rel(b_, a)
            print(f"{cond:<18}{m:<16}{a.mean():11.4f}{b_.mean():11.4f}{b_.mean()-a.mean():+10.4f}"
                  f"{(b_>a).sum():>7}/5{p:9.4f}")
        print()
    print("  pooled        : all 49,706 rows ranked together (calibration across targets matters)")
    print("  within_target : AUROC computed per target then averaged (calibration removed)")
    print("  calib_removed : scores replaced by within-target percentile rank, then pooled")
    print()
    print("  contribution of the kinase block (intact - scrambled):")
    for m in ("pooled", "within_target"):
        for rep in ("PSM", "MCA"):
            a = d[(d.rep == rep) & (d.cond == "intact")].sort_values("fold")[m].values
            b_ = d[(d.rep == rep) & (d.cond == "kinase scrambled")].sort_values("fold")[m].values
            print(f"    {m:<16}{rep}  {a.mean()-b_.mean():+.4f}")


if __name__ == "__main__":
    single_main("decomp")
