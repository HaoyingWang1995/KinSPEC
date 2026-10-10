"""Which representation actually produced Karaman_pure_PSM_v4_result?
Recompute per-molecule per-fold AUROC under BOTH representations and match."""

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
    F = str(config.final_dir); EMB = f"{F}/feature_embedding/Kinase_test_MCA"
    PSMM = str(config.psm_model_dir)

    k = pd.read_csv(f"{F}/dataset/Karaman_Kinase_selectivity_filtered_withseq.csv")
    print("Karaman columns:", list(k.columns))
    mol = pd.read_csv(f"{F}/dataset/Karaman_10_50_filtered_SAE_latent_top95_features_from_merge_v4.csv",
                      skiprows=1, header=None, dtype=np.float32).to_numpy()
    print("karaman rows", len(k), " mol latent", mol.shape)

    ref = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    col = "KLIFS" if "KLIFS" in k.columns else None
    if col is None:
        seq2 = dict(zip(ref.UniProt_ID.astype(str), ref.KLIFS.astype(str)))
        k["KLIFS"] = k.UniProt_ID.astype(str).map(seq2)
    lut = pd.Series(np.arange(len(ref)), index=ref.KLIFS.astype(str).str.strip()).groupby(level=0).last()
    pos = k.KLIFS.astype(str).str.strip().map(lut)
    print("unmapped kinases:", int(pos.isna().sum()))
    keep = pos.notna().to_numpy(); pos = pos[keep].to_numpy(np.intp)
    k = k[keep].reset_index(drop=True); mol = mol[keep]

    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy(np.float32)
    mca = pd.read_csv(f"{EMB}/MCA_klifs_482_200D_features.csv", skiprows=1, header=None).to_numpy(np.float32)
    FN = lambda n: [str(i) for i in range(n)]
    X = {"PSM": np.hstack([mol, psm[pos]]), "MCA": np.hstack([mol, mca[pos]])}
    D = {t: xgb.DMatrix(v, feature_names=FN(v.shape[1])) for t, v in X.items()}

    stored_p = pd.read_csv(str(config.project_dir) + '/Karaman_pure_PSM_v4_result/'
                           "kinspec_mac_karaman_per_molecule_all_folds.csv")
    stored_m = pd.read_csv(str(config.final_dir) + '/model_training/Karaman_MAC_v4_result/'
                           "kinspec_mac_karaman_per_molecule_all_folds.csv")
    lab = k["label"].to_numpy()
    out = {}
    for tag, mp in (("PSM", f"{PSMM}/KinSPEC_MCA_training_band_10_50_fold%d_pure_PSM_model_v4.json"),
                    ("MCA", f"{F}/model/KinSPEC_MCA_training_band_10_50_fold%d_model_v4.json")):
        recs = []
        for fold in range(1, 6):
            b = xgb.Booster(); load_model(b, mp % fold, config)
            if b.feature_names is None: b.feature_names = D[tag].feature_names
            s = b.predict(D[tag])
            for mn, g in pd.DataFrame({"m": k.molecule_name, "s": s, "y": lab}).groupby("m"):
                if g.y.nunique() < 2: continue
                recs.append({"molecule_name": mn, "fold": f"fold{fold}",
                             f"AUROC_{tag}": roc_auc_score(g.y, g.s)})
        out[tag] = pd.DataFrame(recs)
        print(f"  {tag}: {len(recs)} molecule x fold records")

    cmp = stored_p[["molecule_name", "fold", "AUROC"]].rename(columns={"AUROC": "stored_pure_PSM"}).merge(
          stored_m[["molecule_name", "fold", "AUROC"]].rename(columns={"AUROC": "stored_MCA"}),
          on=["molecule_name", "fold"]).merge(out["PSM"], on=["molecule_name", "fold"]).merge(
          out["MCA"], on=["molecule_name", "fold"])
    print(f"\nmatched {len(cmp)} molecule x fold rows")
    for stored in ("stored_pure_PSM", "stored_MCA"):
        for recomp in ("AUROC_PSM", "AUROC_MCA"):
            d = np.abs(cmp[stored] - cmp[recomp])
            print(f"  {stored:<17} vs recomputed {recomp:<10}  "
                  f"max|diff| {d.max():.2e}   mean|diff| {d.mean():.2e}   "
                  f"{'*** MATCH ***' if d.max() < 1e-6 else ''}")

    cmp.to_csv(config.output_dir / "karaman_verification.csv", index=False)


if __name__ == "__main__":
    single_main("karaman_verify")
