"""Is the attribution stability specific to MCA, or does any 200-d reduction give it?
Train PSM-1955 / MCA-200 / PCA-200 on the same 5 folds with the SAME hyperparameters,
back-project each model's gain onto the same 85 KLIFS positions, measure cross-fold agreement.
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
    from analysis_scripts.training_checks import save_checkpoint
    from analysis_scripts.pca_features import fitted_features
    from sklearn.model_selection import StratifiedKFold
    from sklearn.decomposition import PCA
    from scipy import stats

    F = str(config.final_dir); EMB = f"{F}/feature_embedding/Kinase_test_MCA"
    SP = str(config.split_dir) + '/band_0.10_0.50'
    SD = f"{F}/dataset/spectrum_data/band_0.10_0.50"
    OUT = str(config.output_dir)
    HP = {"learning_rate": 0.013558017143562114, "max_depth": 10, "subsample": 0.8775537271396525,
          "colsample_bytree": 0.5557089020337987, "lambda": 24.927473547707798,
          "alpha": 0.8681806294618319, "gamma": 2.7323787690988164, "min_child_weight": 10}
    BASE = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
            "device": config.device, "scale_pos_weight": 14, "seed": 0, "nthread": config.nthread}

    tr = pd.read_csv(f"{SP}/train.csv", usecols=["KLIFS", "unified_bio_label"])
    molt = pd.read_csv(f"{SD}/train_LSAE_latent_top95_features_v4.csv", skiprows=1, header=None,
                       dtype=np.float32).to_numpy()
    y = tr.unified_bio_label.to_numpy().astype(int)
    ref = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    lut = pd.Series(np.arange(len(ref)), index=ref.KLIFS.astype(str).str.strip()).groupby(level=0).last()
    post = tr.KLIFS.astype(str).str.strip().map(lut).to_numpy(np.intp)
    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy(np.float32)
    mca = pd.read_csv(f"{EMB}/MCA_klifs_482_200D_features.csv", skiprows=1, header=None).to_numpy(np.float32)
    pca, pca_components, _ = fitted_features(psm, transform=True)
    np.save(config.output_dir / "stability2_pca_200.npy", pca)
    np.save(config.output_dir / "stability2_pca_components.npy", pca_components)

    col_res = np.arange(1955) // 23                      # PSM column -> KLIFS position
    C = pd.read_csv(f"{EMB}/MCA_200D_category_contributions.csv", index_col=0)
    cat_res = np.array([int(s.split("_")[0][3:]) - 1 for s in C.index])
    Cm = C.to_numpy()
    # PCA: |loading| gives how much each raw column feeds each component
    L = np.abs(pca_components)                            # 200 x 1955

    def to_pos(w, groups, n=85):
        o = np.zeros(n); np.add.at(o, groups, w); return o / o.sum()

    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    prof = {"PSM-1955": [], "MCA-200": [], "PCA-200": []}
    for name, K in (("PSM-1955", psm), ("MCA-200", mca), ("PCA-200", pca)):
        Xtr = np.hstack([molt, K[post]])
        for fi, (ti, vi) in enumerate(splits, 1):
            print(f"Training {name}, fold {fi}/5 ...", flush=True)
            m = xgb.train({**BASE, **HP}, xgb.DMatrix(Xtr[ti], label=y[ti]), num_boost_round=20000,
                          early_stopping_rounds=50,
                          evals=[(xgb.DMatrix(Xtr[vi], label=y[vi]), "e")], verbose_eval=False)
            save_checkpoint(m, config, "stability2", name, fi, {**BASE, **HP})
            g = m.get_score(importance_type="total_gain"); v = np.zeros(Xtr.shape[1])
            for kk, vv in g.items(): v[int(str(kk).lstrip("f"))] = vv
            kin = v[417:]
            if name == "PSM-1955":   prof[name].append(to_pos(kin, col_res))
            elif name == "MCA-200":  prof[name].append(to_pos(Cm @ kin, cat_res))
            else:                    prof[name].append(to_pos(L.T @ kin, col_res))
            print(f"  {name:<10} fold {fi} done (iters {m.best_iteration})", flush=True)
            del m
        del Xtr

    print("\n" + "=" * 80)
    print("Cross-fold agreement of the 85-position attribution profile (10 fold-pairs)")
    print("same folds, same hyperparameters, same coordinate system")
    print("=" * 80)
    print(f"{'representation':<16}{'Spearman':>10}{'cosine':>10}{'top-10 overlap':>16}")
    res = {}
    for name in prof:
        P = np.array(prof[name]); sp, co, ov = [], [], []
        for a in range(5):
            for c in range(a + 1, 5):
                sp.append(stats.spearmanr(P[a], P[c]).statistic)
                co.append(np.dot(P[a], P[c]) / (np.linalg.norm(P[a]) * np.linalg.norm(P[c])))
                ov.append(len(set(np.argsort(P[a])[::-1][:10]) & set(np.argsort(P[c])[::-1][:10])) / 10)
        res[name] = (np.array(sp), np.array(co), np.array(ov))
        print(f"{name:<16}{np.mean(sp):>10.3f}{np.mean(co):>10.3f}{np.mean(ov):>16.2f}")
    print()
    for a, b in (("MCA-200", "PSM-1955"), ("MCA-200", "PCA-200"), ("PCA-200", "PSM-1955")):
        t, p = stats.ttest_rel(res[a][2], res[b][2])
        print(f"  top-10 overlap  {a} vs {b}: {np.mean(res[a][2])-np.mean(res[b][2]):+.3f}  p={p:.4f}")
    np.savez(f"{OUT}/stability2.npz", **{k: np.array(v) for k, v in prof.items()})


if __name__ == "__main__":
    single_main("stability2")
