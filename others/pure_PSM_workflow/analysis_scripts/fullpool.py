"""Full training pool models: pure PSM vs KLIFS-MCA.
Both tuned by Optuna over an identical search space (200 trials, BEDROC a=80.5 objective),
so this is the fair protocol-level comparison."""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np, pandas as pd, xgboost as xgb, json
    from sklearn.metrics import roc_auc_score, average_precision_score

    F = str(config.final_dir); EMB = f"{F}/feature_embedding/Kinase_test_MCA"
    PSMM = str(config.psm_model_dir) + '/KinSPEC_full_training_pool_pure_PSM_v4.json'
    MCAM = f"{F}/model/KinSPEC_full_training_pool_MCA_200D_v4.json"
    OUT = str(config.output_dir)

    for p, lab in ((PSMM, "pure PSM"), (MCAM, "MCA")):
        m = json.load(open(p))["learner"]
        print(f"{lab:<10} num_feature {m['learner_model_param']['num_feature']:>5}  "
              f"trees {len(m['gradient_booster']['model']['trees'])}")

    def bedroc(y, s, a):
        y = np.asarray(y); N = len(y); na = int(y.sum())
        r = np.arange(1, N + 1)[y[np.argsort(s)[::-1]] == 1]; x = (r - .5) / N
        C = (1 - np.exp(-a)) / a
        rie = np.mean(np.exp(-a * x)) / C
        rmax = np.mean(np.exp(-a * ((np.arange(1, na + 1) - .5) / N))) / C
        rmin = np.mean(np.exp(-a * ((np.arange(N - na + 1, N + 1) - .5) / N))) / C
        return (rie - rmin) / (rmax - rmin)
    def ef(y, s, f):
        N = len(y); na = int(y.sum()); k = max(1, int(round(N * f)))
        return (y[np.argsort(s)[::-1][:k]].sum() / k) / (na / N)

    # ---------- VSDS ----------
    v = pd.read_csv(f"{F}/dataset/VSDS_truedecoy_integrated_data.csv",
                    usecols=["uniprot_id", "KLIFS", "molecule_label"])
    molv = pd.read_csv(f"{F}/dataset/VSDS_truedecoy_LSAE_latent_top95_features_v4.csv",
                       skiprows=1, header=None, dtype=np.float32).to_numpy()
    yv = v.molecule_label.to_numpy().astype(int); tgt = v.uniprot_id.to_numpy()
    ref = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    lut = pd.Series(np.arange(len(ref)), index=ref.KLIFS.astype(str).str.strip()).groupby(level=0).last()
    posv = v.KLIFS.astype(str).str.strip().map(lut).to_numpy(np.intp)
    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy(np.float32)
    mca = pd.read_csv(f"{EMB}/MCA_klifs_482_200D_features.csv", skiprows=1, header=None).to_numpy(np.float32)
    FN = lambda n: [str(i) for i in range(n)]
    hasT = [t for t in np.unique(tgt) if yv[tgt == t].sum() > 0 and (yv[tgt == t] == 0).sum() > 0]
    rng = np.random.default_rng(42)

    print(f"\nVSDS {len(v)} rows, {yv.sum()} actives, {len(hasT)} targets")
    print("=" * 94)
    print("VSDS, full training pool models (molecular similarity NOT restricted -- same for both)")
    print("=" * 94)
    print(f"{'condition':<18}{'metric':<16}{'pure PSM':>11}{'MCA':>11}{'diff':>10}")
    res = {}
    scores = {}
    for cond in ("intact", "kinase scrambled"):
        perm = rng.permutation(482)
        for tag, mp, K, nf in (("PSM", PSMM, psm, 2372), ("MCA", MCAM, mca, 617)):
            b = xgb.Booster(); load_model(b, mp, config)
            if b.feature_names is None: b.feature_names = FN(nf)
            Kc = K if cond == "intact" else K[perm]
            X = np.hstack([molv, Kc[posv]])
            s = b.predict(xgb.DMatrix(X, feature_names=FN(nf)))
            if cond == "intact": scores[tag] = s
            within = float(np.mean([roc_auc_score(yv[tgt == t], s[tgt == t]) for t in hasT]))
            res[(cond, tag)] = {"AUROC": roc_auc_score(yv, s), "AUPRC": average_precision_score(yv, s),
                                "BEDROC20": bedroc(yv, s, 20.), "BEDROC80.5": bedroc(yv, s, 80.5),
                                "EF_0.5%": ef(yv, s, .005), "EF_1%": ef(yv, s, .01),
                                "EF_5%": ef(yv, s, .05), "within_target": within}
            del b
        for m in ("AUROC", "AUPRC", "BEDROC20", "BEDROC80.5", "EF_0.5%", "EF_1%", "EF_5%", "within_target"):
            a, c = res[(cond, "PSM")][m], res[(cond, "MCA")][m]
            print(f"{cond:<18}{m:<16}{a:>11.4f}{c:>11.4f}{c-a:+10.4f}")
        print()
    print("  kinase block contribution (intact - scrambled):")
    for m in ("AUROC", "within_target"):
        for tag in ("PSM", "MCA"):
            print(f"    {m:<14}{tag}  {res[('intact',tag)][m]-res[('kinase scrambled',tag)][m]:+.4f}")
    np.save(f"{OUT}/fullpool_scores_PSM.npy", scores["PSM"]); np.save(f"{OUT}/fullpool_scores_MCA.npy", scores["MCA"])

    # ---------- Karaman selectivity ----------
    k = pd.read_csv(f"{F}/dataset/Karaman_Kinase_selectivity_filtered_withseq.csv")
    molk = pd.read_csv(f"{F}/dataset/Karaman_10_50_filtered_SAE_latent_top95_features_from_merge_v4.csv",
                       skiprows=1, header=None, dtype=np.float32).to_numpy()
    pk = k.UniProt_ID.astype(str).map(dict(zip(ref.UniProt_ID.astype(str), ref.KLIFS.astype(str)))).map(lut)
    ok = pk.notna().to_numpy(); k, molk, pk = k[ok].reset_index(drop=True), molk[ok], pk[ok].to_numpy(np.intp)
    lab = k["label"].to_numpy(); mn = k.molecule_name.to_numpy()
    print("\n" + "=" * 94)
    print("Karaman kinase selectivity, full training pool models")
    print("=" * 94)
    out = {}
    for tag, mp, K, nf in (("PSM", PSMM, psm, 2372), ("MCA", MCAM, mca, 617)):
        b = xgb.Booster(); load_model(b, mp, config)
        if b.feature_names is None: b.feature_names = FN(nf)
        X = np.hstack([molk, K[pk]])
        s = b.predict(xgb.DMatrix(X, feature_names=FN(nf)))
        d = pd.DataFrame({"m": mn, "s": s, "y": lab})
        au, be = [], []
        for _, g in d.groupby("m"):
            if g.y.nunique() < 2: continue
            au.append(roc_auc_score(g.y, g.s)); be.append(bedroc(g.y.to_numpy(), g.s.to_numpy(), 20.))
        out[tag] = (np.mean(au), np.mean(be), len(au))
        del b
    print(f"  {'':<12}{'AUROC':>10}{'BEDROC20':>11}   ({out['PSM'][2]} molecules)")
    print(f"  {'pure PSM':<12}{out['PSM'][0]:>10.4f}{out['PSM'][1]:>11.4f}")
    print(f"  {'MCA':<12}{out['MCA'][0]:>10.4f}{out['MCA'][1]:>11.4f}")
    print(f"  {'diff':<12}{out['MCA'][0]-out['PSM'][0]:>+10.4f}{out['MCA'][1]-out['PSM'][1]:>+11.4f}")

    pd.DataFrame([{"condition": condition, "rep": tag, **metrics} for (condition, tag), metrics in res.items()]).to_csv(config.output_dir / "fullpool_vsds.csv", index=False)
    pd.DataFrame([{"rep": tag, "AUROC": values[0], "BEDROC20": values[1], "molecules": values[2]} for tag, values in out.items()]).to_csv(config.output_dir / "fullpool_karaman.csv", index=False)


if __name__ == "__main__":
    single_main("fullpool")
