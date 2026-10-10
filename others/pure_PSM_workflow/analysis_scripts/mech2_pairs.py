"""The adversarial-subset analogue, on the kinase side.

If a model's kinase-side evidence comes from variables that most kinases share, then two
kinases that AGREE on those variables are indistinguishable to it.  For a molecule that is
active on one and inactive on the other, such a model must fail.

Triple = (molecule, kinase A active, kinase B inactive).
Pairwise accuracy = P(score(A) > score(B)).  Chance = 0.5.
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
    OUT = str(config.output_dir)

    st = np.load(f"{OUT}/strata.npz", allow_pickle=True)
    var_of, ent = st["var_of"], st["ent"]
    Z = np.load(f"{OUT}/Z.npy"); G = np.load(f"{OUT}/G.npy"); lam = np.load(f"{OUT}/lam.npy")
    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy(np.float32)
    mca = ((Z @ G) / (425 * np.sqrt(lam))).astype(np.float32)

    # active level index per (kinase, variable): lets us compare two kinases variable by variable
    Q = 425
    lev = np.zeros((482, Q), np.int32)
    for v in range(Q):
        c = np.where(var_of == v)[0]
        lev[:, v] = c[np.argmax(Z[:, c], axis=1)]
    order = np.argsort(ent)
    SHARED = order[:85]        # the 85 most shared (lowest entropy) variables — the coarse end
    print(f"SHARED stratum: {len(SHARED)} variables, mean entropy {ent[SHARED].mean():.3f}, "
          f"modal frequency {np.mean([np.bincount(lev[:,v]).max()/482 for v in SHARED]):.3f}")

    k = pd.read_csv(f"{F}/dataset/Karaman_Kinase_selectivity_filtered_withseq.csv")
    mol = pd.read_csv(f"{F}/dataset/Karaman_10_50_filtered_SAE_latent_top95_features_from_merge_v4.csv",
                      skiprows=1, header=None, dtype=np.float32).to_numpy()
    ref = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    lut = pd.Series(np.arange(len(ref)), index=ref.KLIFS.astype(str).str.strip()).groupby(level=0).last()
    pos = k.UniProt_ID.astype(str).map(dict(zip(ref.UniProt_ID.astype(str), ref.KLIFS.astype(str)))).map(lut)
    ok = pos.notna().to_numpy()
    k, mol, pos = k[ok].reset_index(drop=True), mol[ok], pos[ok].to_numpy(np.intp)
    lab = k["label"].to_numpy(); mn = k.molecule_name.to_numpy()
    FN = lambda n: [str(i) for i in range(n)]

    # ---- build triples ----
    agree = (lev[:, SHARED][:, None, :] == lev[:, SHARED][None, :, :]).mean(2)   # 482x482
    tri = []
    for m, g in pd.DataFrame({"m": mn, "p": pos, "y": lab, "i": np.arange(len(k))}).groupby("m"):
        A = g[g.y == 1]; B = g[g.y == 0]
        if len(A) == 0 or len(B) == 0: continue
        for _, a in A.iterrows():
            for _, b in B.iterrows():
                tri.append((m, a.i, b.i, agree[a.p, b.p]))
    tri = pd.DataFrame(tri, columns=["m", "ia", "ib", "shared_agree"])
    print(f"triples (molecule, active kinase, inactive kinase): {len(tri):,} "
          f"from {tri.m.nunique()} molecules")
    q = tri.shared_agree.quantile([.25, .5, .75, .9]).to_dict()
    print(f"  agreement on the 85 most-shared variables: median {q[0.5]:.3f}, "
          f"p75 {q[0.75]:.3f}, p90 {q[0.9]:.3f}")
    ADV = tri.shared_agree >= q[0.9]          # shortcut cannot separate these
    CTL = tri.shared_agree <= q[0.25]         # shortcut can separate these
    print(f"  adversarial subset (top decile of agreement): {ADV.sum():,} triples")
    print(f"  control subset     (bottom quartile)        : {CTL.sum():,} triples")

    rows = []
    for fold in range(1, 6):
        for tag, mp, K, nf in (("PSM", f"{PSMM}/KinSPEC_MCA_training_band_10_50_fold{fold}_pure_PSM_model_v4.json", psm, 2372),
                               ("MCA", f"{F}/model/KinSPEC_MCA_training_band_10_50_fold{fold}_model_v4.json", mca, 617)):
            b = xgb.Booster(); load_model(b, mp, config)
            if b.feature_names is None: b.feature_names = FN(nf)
            X = np.hstack([mol, K[pos]])
            s = b.predict(xgb.DMatrix(X, feature_names=FN(nf)))
            w = (s[tri.ia.values] > s[tri.ib.values]).astype(float) + \
                0.5 * (s[tri.ia.values] == s[tri.ib.values])
            rows.append({"fold": fold, "rep": tag, "all": w.mean(),
                         "adversarial": w[ADV.values].mean(), "control": w[CTL.values].mean()})
            del b
    d = pd.DataFrame(rows); d.to_csv(f"{OUT}/mech2_pairs.csv", index=False)

    print("\n" + "=" * 84)
    print("Pairwise accuracy: does the model rank the ACTIVE kinase above the INACTIVE one?")
    print("=" * 84)
    print(f"{'subset':<16}{'pure PSM':>11}{'MCA':>11}{'diff':>10}{'MCA wins':>11}{'p':>9}")
    for c in ("all", "control", "adversarial"):
        x = d[d.rep == "PSM"].sort_values("fold")[c].values
        y = d[d.rep == "MCA"].sort_values("fold")[c].values
        t, p = stats.ttest_rel(y, x)
        print(f"{c:<16}{x.mean():11.4f}{y.mean():11.4f}{y.mean()-x.mean():+10.4f}{(y>x).sum():>8}/5{p:9.4f}")
    print("\n  adversarial = the two kinases agree on >=90th percentile of the 85 most-shared")
    print("  variables, so a model relying on those variables cannot tell them apart (chance 0.5).")


if __name__ == "__main__":
    single_main("mech2_pairs")
