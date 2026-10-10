"""Adversarial subset defined by the features the pure-PSM model ITSELF relies on
(mirrors Morgan exp 7, which used the promiscuous positions that model relied on)."""

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
    Z = np.load(f"{OUT}/Z.npy"); G = np.load(f"{OUT}/G.npy"); lam = np.load(f"{OUT}/lam.npy")
    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy(np.float32)
    mca = ((Z @ G) / (425 * np.sqrt(lam))).astype(np.float32)

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

    tri = []
    for m, g in pd.DataFrame({"m": mn, "p": pos, "y": lab, "i": np.arange(len(k))}).groupby("m"):
        A, B = g[g.y == 1], g[g.y == 0]
        if len(A) == 0 or len(B) == 0: continue
        for _, a in A.iterrows():
            for _, b in B.iterrows(): tri.append((m, a.i, b.i, a.p, b.p))
    tri = pd.DataFrame(tri, columns=["m", "ia", "ib", "pa", "pb"])
    print(f"triples: {len(tri):,} from {tri.m.nunique()} molecules\n")

    rows = []
    for fold in range(1, 6):
        bp = xgb.Booster(); load_model(bp, f"{PSMM}/KinSPEC_MCA_training_band_10_50_fold{fold}_pure_PSM_model_v4.json", config)
        if bp.feature_names is None: bp.feature_names = FN(2372)
        g = bp.get_score(importance_type="total_gain"); v = np.zeros(2372)
        for kk, vv in g.items(): v[int(kk)] = vv
        kin = v[417:]
        for TOP in (20, 50, 100):
            top = np.argsort(kin)[::-1][:TOP]
            # a pair is "confusable for this model" when it agrees on every top-gain column
            ag = (psm[:, top][tri.pa.values] == psm[:, top][tri.pb.values]).mean(1)
            adv = ag >= 1.0
            ctl = ag <= np.quantile(ag, .25)
            for tag, path, K, nf in (("PSM", None, psm, 2372),
                                     ("MCA", f"{F}/model/KinSPEC_MCA_training_band_10_50_fold{fold}_model_v4.json", mca, 617)):
                b = bp if tag == "PSM" else xgb.Booster()
                if tag == "MCA":
                    load_model(b, path, config)
                    if b.feature_names is None: b.feature_names = FN(nf)
                X = np.hstack([mol, K[pos]])
                s = b.predict(xgb.DMatrix(X, feature_names=FN(nf)))
                w = (s[tri.ia.values] > s[tri.ib.values]).astype(float) + \
                    0.5 * (s[tri.ia.values] == s[tri.ib.values])
                rows.append({"fold": fold, "TOP": TOP, "rep": tag, "n_adv": int(adv.sum()),
                             "all": w.mean(), "adversarial": w[adv].mean(), "control": w[ctl].mean()})
        print(f"  fold {fold} done", flush=True)
    d = pd.DataFrame(rows); d.to_csv(f"{OUT}/mech2_pairs2.csv", index=False)

    print("\n" + "=" * 90)
    print("Pairwise accuracy on kinase pairs that AGREE on every top-gain column of pure PSM")
    print("=" * 90)
    for TOP in (20, 50, 100):
        s = d[d.TOP == TOP]
        print(f"\n  confusable w.r.t. pure PSM's top-{TOP} gain columns "
              f"(n={int(s.n_adv.mean()):,} of {len(tri):,} triples, {s.n_adv.mean()/len(tri)*100:.1f}%)")
        print(f"    {'subset':<14}{'pure PSM':>11}{'MCA':>11}{'diff':>10}{'MCA wins':>11}{'p':>9}")
        for c in ("all", "control", "adversarial"):
            x = s[s.rep == "PSM"].sort_values("fold")[c].values
            y = s[s.rep == "MCA"].sort_values("fold")[c].values
            t, p = stats.ttest_rel(y, x)
            print(f"    {c:<14}{x.mean():11.4f}{y.mean():11.4f}{y.mean()-x.mean():+10.4f}"
                  f"{(y>x).sum():>8}/5{p:9.4f}")


if __name__ == "__main__":
    single_main("mech2_pairs2")
