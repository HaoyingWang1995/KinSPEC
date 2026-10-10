"""Paired VSDS evaluation: pure PSM (1955 bit) vs KLIFS-MCA (200D), matched mol LSAE + folds."""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np, pandas as pd, xgboost as xgb, os, json
    from sklearn.metrics import roc_auc_score, average_precision_score

    FINAL = str(config.final_dir)
    PSMM  = str(config.psm_model_dir)
    MCAM  = f"{FINAL}/model"
    OUT   = str(config.output_dir)
    BANDS = ["10_50", "20_60", "30_70", "40_80", "50_90", "60_100"]
    EFCUT = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]

    def bedroc(y, s, alpha):
        y = np.asarray(y); s = np.asarray(s); N = len(y); na = int((y == 1).sum())
        if na == 0: return 0.0
        ranks = np.arange(1, N + 1)[y[np.argsort(s)[::-1]] == 1]
        x = (ranks - 0.5) / N
        C = (1.0 - np.exp(-alpha)) / alpha
        rie = np.mean(np.exp(-alpha * x)) / C
        rmax = np.mean(np.exp(-alpha * ((np.arange(1, na + 1) - 0.5) / N))) / C
        rmin = np.mean(np.exp(-alpha * ((np.arange(N - na + 1, N + 1) - 0.5) / N))) / C
        return (rie - rmin) / (rmax - rmin)

    def ef(y, s, frac):
        N = len(y); na = int((y == 1).sum())
        if na == 0: return np.nan
        k = max(1, int(round(N * frac)))
        top = y[np.argsort(s)[::-1][:k]]
        return (top.sum() / k) / (na / N)

    def allmetrics(y, s):
        m = {"AUROC": roc_auc_score(y, s), "AUPRC": average_precision_score(y, s),
             "BEDROC20": bedroc(y, s, 20.0), "BEDROC80.5": bedroc(y, s, 80.5)}
        for f in EFCUT: m[f"EF_{f*100:g}%"] = ef(y, s, f)
        return m

    print("loading VSDS ...", flush=True)
    vsds = pd.read_csv(f"{FINAL}/dataset/VSDS_truedecoy_integrated_data.csv",
                       usecols=["uniprot_id", "KLIFS", "molecule_label"])
    mol = pd.read_csv(f"{FINAL}/dataset/VSDS_truedecoy_LSAE_latent_top95_features_v4.csv",
                      skiprows=1, header=None, dtype=np.float32).to_numpy()
    y = vsds.molecule_label.to_numpy().astype(int)
    tgt = vsds.uniprot_id.to_numpy()
    print(f"  VSDS {len(vsds)} rows, actives {y.sum()}, targets {len(np.unique(tgt))}, mol {mol.shape}", flush=True)

    ref = pd.read_csv(f"{FINAL}/feature_embedding/Kinase_test_MCA/human_482_typical_kinase_1955bit.csv")
    lookup = pd.Series(np.arange(len(ref)), index=ref.KLIFS.astype(str).str.strip()).groupby(level=0).last()
    pos = vsds.KLIFS.astype(str).str.strip().map(lookup).to_numpy(dtype=np.intp)
    assert not np.isnan(pos.astype(float)).any()

    psm = pd.read_csv(f"{FINAL}/feature_embedding/Kinase_test_MCA/human_482_typical_kinase_1955bit_onehot.csv",
                      header=None).to_numpy(dtype=np.float32)
    mca = pd.read_csv(f"{FINAL}/feature_embedding/Kinase_test_MCA/MCA_klifs_482_200D_features.csv",
                      skiprows=1, header=None).to_numpy(dtype=np.float32)
    print(f"  PSM {psm.shape}  MCA {mca.shape}", flush=True)

    X_psm = np.hstack([mol, psm[pos]]);  print(f"  X_psm {X_psm.shape}", flush=True)
    X_mca = np.hstack([mol, mca[pos]]);  print(f"  X_mca {X_mca.shape}", flush=True)
    D_psm = xgb.DMatrix(X_psm, feature_names=[str(i) for i in range(X_psm.shape[1])])
    D_mca = xgb.DMatrix(X_mca, feature_names=[str(i) for i in range(X_mca.shape[1])])

    rows = []
    for band in BANDS:
        for fold in range(1, 6):
            for tag, path, D in (
                ("PSM", f"{PSMM}/KinSPEC_MCA_training_band_{band}_fold{fold}_pure_PSM_model_v4.json", D_psm),
                ("MCA", f"{MCAM}/KinSPEC_MCA_training_band_{band}_fold{fold}_model_v4.json", D_mca)):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing band model: {path}")
                b = xgb.Booster(); load_model(b, path, config)
                if b.feature_names is None: b.feature_names = D.feature_names
                s = b.predict(D)
                r = {"band": band, "fold": fold, "rep": tag, "scope": "pooled", **allmetrics(y, s)}
                rows.append(r)
                # macro over targets
                per = [allmetrics(y[tgt == t], s[tgt == t]) for t in np.unique(tgt) if (y[tgt == t] == 1).sum() > 0]
                rows.append({"band": band, "fold": fold, "rep": tag, "scope": "macro_target",
                             **{k: float(np.mean([p[k] for p in per])) for k in per[0]}})
                np.save(f"{OUT}/scores_{tag}_{band}_f{fold}.npy", s.astype(np.float32))
                del b, s
            print(f"  done band {band} fold {fold}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/vsds_perfold_PSM_vs_MCA.csv", index=False)
    print("\nwrote", f"{OUT}/vsds_perfold_PSM_vs_MCA.csv", flush=True)


if __name__ == "__main__":
    single_main("vsds_eval")
