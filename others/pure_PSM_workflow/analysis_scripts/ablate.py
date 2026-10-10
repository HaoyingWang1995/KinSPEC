"""Causal check on the gain attribution: permute kinase features by prevalence stratum."""

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
    FINAL = str(config.final_dir); EMB = f"{FINAL}/feature_embedding/Kinase_test_MCA"
    PSMM = str(config.psm_model_dir); OUT = str(config.output_dir)
    rng = np.random.default_rng(42)

    v = pd.read_csv(f"{FINAL}/dataset/VSDS_truedecoy_integrated_data.csv", usecols=["KLIFS", "molecule_label"])
    y = v.molecule_label.to_numpy().astype(int)
    mol = pd.read_csv(f"{FINAL}/dataset/VSDS_truedecoy_LSAE_latent_top95_features_v4.csv",
                      skiprows=1, header=None, dtype=np.float32).to_numpy()
    ref = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    lut = pd.Series(np.arange(len(ref)), index=ref.KLIFS.astype(str).str.strip()).groupby(level=0).last()
    pos = v.KLIFS.astype(str).str.strip().map(lut).to_numpy(np.intp)
    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy(np.float32)
    mca = pd.read_csv(f"{EMB}/MCA_klifs_482_200D_features.csv", skiprows=1, header=None).to_numpy(np.float32)
    prev = psm.mean(0); strat = np.digitize(prev, [.01, .05, .20])
    FN = lambda n: [str(i) for i in range(n)]

    def auc(booster, K):
        X = np.hstack([mol, K[pos]])
        return roc_auc_score(y, booster.predict(xgb.DMatrix(X, feature_names=FN(X.shape[1]))))

    print("Permuting kinase features ACROSS the 482 kinases (breaks the pocket-label link),")
    print("then re-scoring VSDS.  Reported: AUROC, and the fraction of the kinase")
    print("contribution that survives, relative to a fully permuted kinase block.\n")
    print(f"{'band':<8}{'fold':>4} | {'PSM full':>9}{'-promisc':>9}{'-rare':>8}{'-all kin':>9} | {'MCA full':>9}{'-all kin':>9}")
    rows = []
    for band in ["10_50", "30_70", "60_100"]:
        for fold in range(1, 6):
            bp = xgb.Booster(); load_model(bp, f"{PSMM}/KinSPEC_MCA_training_band_{band}_fold{fold}_pure_PSM_model_v4.json", config)
            bm = xgb.Booster(); load_model(bm, f"{FINAL}/model/KinSPEC_MCA_training_band_{band}_fold{fold}_model_v4.json", config)
            perm = rng.permutation(482)
            p_full = auc(bp, psm)
            p_all = auc(bp, psm[perm])
            Kp = psm.copy(); Kp[:, strat == 3] = psm[perm][:, strat == 3]; p_pro = auc(bp, Kp)
            Kr = psm.copy(); Kr[:, strat == 0] = psm[perm][:, strat == 0]; p_rar = auc(bp, Kr)
            m_full = auc(bm, mca); m_all = auc(bm, mca[perm])
            rows.append(dict(band=band, fold=fold, p_full=p_full, p_pro=p_pro, p_rar=p_rar,
                             p_all=p_all, m_full=m_full, m_all=m_all))
            print(f"{band:<8}{fold:>4} | {p_full:9.4f}{p_pro:9.4f}{p_rar:8.4f}{p_all:9.4f} | {m_full:9.4f}{m_all:9.4f}")
    d = pd.DataFrame(rows); d.to_csv(f"{OUT}/ablate.csv", index=False)
    print()
    print(f"  mean over {len(d)} models:")
    lost_pro = (d.p_full - d.p_pro) / (d.p_full - d.p_all)
    lost_rar = (d.p_full - d.p_rar) / (d.p_full - d.p_all)
    print(f"    PSM: permuting the promiscuous third of columns destroys "
          f"{lost_pro.mean()*100:.1f}% of the whole kinase contribution")
    print(f"    PSM: permuting the rare fifth of columns destroys "
          f"{lost_rar.mean()*100:.1f}%")
    print(f"    kinase block is worth  PSM {(d.p_full-d.p_all).mean():+.4f} AUROC   "
          f"MCA {(d.m_full-d.m_all).mean():+.4f} AUROC")
    print(f"    kinase-blind floor:    PSM {d.p_all.mean():.4f}   MCA {d.m_all.mean():.4f}")


if __name__ == "__main__":
    single_main("ablate")
