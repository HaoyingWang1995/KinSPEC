"""Decisive control: is MCA's compound-ranking gain specific to MCA, or does ANY 200-d
kinase block produce it by relieving column-subsample dilution of the 417 molecular features?

Identical data, folds, seed and hyperparameters; only the kinase block changes.
  PSM-1955   raw one-hot                (baseline)
  MCA-200    the categorical embedding  (baseline)
  PCA-200    linear reduction of PSM    (dimension-matched, information-preserving)
  RAND-200   Gaussian random projection (dimension-matched, NOT tuned to the data)
  PSM-1955 @ colsample rescaled so the molecular:kinase candidate ratio per tree
             matches MCA's -- isolates the dilution mechanism with the encoding unchanged.
"""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np, pandas as pd, xgboost as xgb, json, os
    from analysis_scripts.training_checks import save_checkpoint
    from analysis_scripts.pca_features import fitted_features
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.decomposition import PCA

    F = str(config.final_dir); EMB = f"{F}/feature_embedding/Kinase_test_MCA"
    SP = str(config.split_dir) + '/band_0.10_0.50'
    SD = f"{F}/dataset/spectrum_data/band_0.10_0.50"
    OUT = str(config.output_dir)

    HP = {"learning_rate": 0.013558017143562114, "max_depth": 10, "subsample": 0.8775537271396525,
          "colsample_bytree": 0.5557089020337987, "lambda": 24.927473547707798,
          "alpha": 0.8681806294618319, "gamma": 2.7323787690988164, "min_child_weight": 10}
    BASE = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
            "device": config.device, "scale_pos_weight": 14,"seed":0,"nthread":config.nthread}

    print("loading training data ...", flush=True)
    tr = pd.read_csv(f"{SP}/train.csv", usecols=["KLIFS", "unified_bio_label"])
    molt = pd.read_csv(f"{SD}/train_LSAE_latent_top95_features_v4.csv", skiprows=1, header=None,
                       dtype=np.float32).to_numpy()
    y = tr.unified_bio_label.to_numpy().astype(int)
    ref = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    lut = pd.Series(np.arange(len(ref)), index=ref.KLIFS.astype(str).str.strip()).groupby(level=0).last()
    post = tr.KLIFS.astype(str).str.strip().map(lut).to_numpy(np.intp)
    print(f"  train {molt.shape}, actives {y.sum()}", flush=True)

    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy(np.float32)
    mca = pd.read_csv(f"{EMB}/MCA_klifs_482_200D_features.csv", skiprows=1, header=None).to_numpy(np.float32)
    rng = np.random.default_rng(0)
    pca, _, pca_variance = fitted_features(psm)
    np.save(config.output_dir / "control_pca_200.npy", pca)
    rp = (psm @ rng.normal(0, 1 / np.sqrt(200), (1955, 200))).astype(np.float32)
    print(f"  PCA-200 keeps {pca_variance*100:.1f}%"
          f" of the raw PSM variance", flush=True)

    # VSDS test set
    v = pd.read_csv(f"{F}/dataset/VSDS_truedecoy_integrated_data.csv",
                    usecols=["uniprot_id", "KLIFS", "molecule_label"])
    molv = pd.read_csv(f"{F}/dataset/VSDS_truedecoy_LSAE_latent_top95_features_v4.csv",
                       skiprows=1, header=None, dtype=np.float32).to_numpy()
    yv = v.molecule_label.to_numpy().astype(int); tgt = v.uniprot_id.to_numpy()
    posv = v.KLIFS.astype(str).str.strip().map(lut).to_numpy(np.intp)
    hasT = [t for t in np.unique(tgt) if yv[tgt == t].sum() > 0 and (yv[tgt == t] == 0).sum() > 0]

    REPS = {"PSM-1955": (psm, HP["colsample_bytree"]),
            "MCA-200": (mca, HP["colsample_bytree"]),
            "PCA-200": (pca, HP["colsample_bytree"]),
            "RAND-200": (rp, HP["colsample_bytree"]),
            # match MCA's mol:kinase candidate ratio (232:111) while keeping the 1955 one-hot:
            # need c*1955 / (c*417) fixed -> instead cap kinase candidates by shrinking colsample
            # to the value that gives ~111 kinase candidates, then compensate is impossible per-block,
            # so we use colsample_bytree=1.0 as the dilution-free reference for BOTH 1955 and 200.
            "PSM-1955 cs=1.0": (psm, 1.0),
            "MCA-200 cs=1.0": (mca, 1.0)}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(y)), y))
    rows = []
    for name, (K, cs) in REPS.items():
        Xtr = np.hstack([molt, K[post]])
        Xte = np.hstack([molv, K[posv]])
        Dte = xgb.DMatrix(Xte)
        params = {**BASE, **HP, "colsample_bytree": cs}
        for fi, (ti, vi) in enumerate(splits, 1):
            print(f"Training {name}, fold {fi}/5 ...", flush=True)
            dtr = xgb.DMatrix(Xtr[ti], label=y[ti]); dva = xgb.DMatrix(Xtr[vi], label=y[vi])
            m = xgb.train(params, dtr, num_boost_round=20000, early_stopping_rounds=50,
                          evals=[(dva, "eval")], verbose_eval=False)
            save_checkpoint(m, config, "control", name, fi, params)
            s = m.predict(Dte, iteration_range=(0, m.best_iteration + 1))
            pooled = roc_auc_score(yv, s)
            within = float(np.mean([roc_auc_score(yv[tgt == t], s[tgt == t]) for t in hasT]))
            rows.append({"rep": name, "colsample": cs, "fold": fi, "n_feat": Xtr.shape[1],
                         "best_iter": m.best_iteration, "pooled": pooled, "within_target": within})
            print(f"  {name:<18} fold {fi}  iters {m.best_iteration:>5}  "
                  f"pooled {pooled:.4f}  within {within:.4f}", flush=True)
            del m, dtr, dva
        del Xtr, Xte, Dte
    pd.DataFrame(rows).to_csv(f"{OUT}/control.csv", index=False)
    print("\nwrote control.csv", flush=True)


if __name__ == "__main__":
    single_main("control")
