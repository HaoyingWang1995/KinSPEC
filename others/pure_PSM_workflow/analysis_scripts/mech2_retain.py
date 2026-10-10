"""Mechanism, Karaman selectivity, band 10-50 only.
Identical intervention applied to BOTH representations: permute whole MCA variables across the
482 kinases (keeps exactly one active level per variable, so the indicator stays valid and the
MCA projection stays exact), then re-score.

  retain SHARED   = permute the 213 discriminative variables, keep the 212 shared ones intact
  retain DISCRIM  = permute the 212 shared variables, keep the 213 discriminative ones intact
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
    import numpy as np, pandas as pd, xgboost as xgb, json
    from sklearn.metrics import roc_auc_score
    from scipy import stats

    F = str(config.final_dir); EMB = f"{F}/feature_embedding/Kinase_test_MCA"
    PSMM = str(config.psm_model_dir)
    OUT = str(config.output_dir)
    R = 12  # permutation replicates
    rng = np.random.default_rng(42)

    st = np.load(f"{OUT}/strata.npz", allow_pickle=True)
    var_of, shared, discrim = st["var_of"], st["shared"], st["discrim"]
    vres, vsys = st["res"], st["sys"]
    Z = np.load(f"{OUT}/Z.npy"); G = np.load(f"{OUT}/G.npy"); lam = np.load(f"{OUT}/lam.npy")
    Q = 425
    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy(np.float32)

    # PSM column -> variable
    lk = pd.read_csv(f"{EMB}/MCA_200D_category_lookup.csv").dropna(subset=["bit_index_0_based"])
    lk["bit"] = lk.bit_index_0_based.astype(int)
    bit2sys = dict(zip(lk[lk.residue == 1].bit, lk[lk.residue == 1].system))
    vidx = {(r, s): i for i, (r, s) in enumerate(zip(vres, vsys))}
    col_var = np.array([vidx[(1 + c // 23, bit2sys[c % 23])] for c in range(1955)])

    def mca_of(Zm): return ((Zm @ G) / (Q * np.sqrt(lam))).astype(np.float32)
    assert np.abs(mca_of(Z) - pd.read_csv(f"{EMB}/MCA_klifs_482_200D_features.csv", skiprows=1,
                  header=None).to_numpy(np.float32)).max() < 1e-4, "projection mismatch"

    def intervene(keep_vars, perm):
        """permute every variable NOT in keep_vars, across the 482 kinases"""
        move = np.setdiff1d(np.arange(Q), keep_vars)
        Zm = Z.copy(); cm = np.isin(var_of, move)
        Zm[:, cm] = Z[perm][:, cm]
        Pm = psm.copy(); pm = np.isin(col_var, move)
        Pm[:, pm] = psm[perm][:, pm]
        return Pm, mca_of(Zm)

    # ---- Karaman ----
    k = pd.read_csv(f"{F}/dataset/Karaman_Kinase_selectivity_filtered_withseq.csv")
    mol = pd.read_csv(f"{F}/dataset/Karaman_10_50_filtered_SAE_latent_top95_features_from_merge_v4.csv",
                      skiprows=1, header=None, dtype=np.float32).to_numpy()
    ref = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    lut = pd.Series(np.arange(len(ref)), index=ref.KLIFS.astype(str).str.strip()).groupby(level=0).last()
    seq = dict(zip(ref.UniProt_ID.astype(str), ref.KLIFS.astype(str)))
    kk = k.UniProt_ID.astype(str).map(seq)
    pos = kk.map(lut); ok = pos.notna().to_numpy()
    k, mol, pos = k[ok].reset_index(drop=True), mol[ok], pos[ok].to_numpy(np.intp)
    lab = k["label"].to_numpy(); mn = k.molecule_name.to_numpy()
    usable = [m for m, g in pd.DataFrame({"m": mn, "y": lab}).groupby("m") if g.y.nunique() > 1]
    sel = np.isin(mn, usable)
    print(f"Karaman: {len(k)} measurements, {len(usable)} molecules with both classes, "
          f"{k.UniProt_ID.nunique()} kinases\n")
    FN = lambda n: [str(i) for i in range(n)]

    def sel_auc(booster, K, rep):
        X = np.hstack([mol, K[pos]])
        s = booster.predict(xgb.DMatrix(X, feature_names=FN(X.shape[1])))
        d = pd.DataFrame({"m": mn, "s": s, "y": lab})[sel]
        return np.mean([roc_auc_score(g.y, g.s) for _, g in d.groupby("m")])

    CASES = ["full", "retain SHARED", "retain DISCRIM", "all permuted"]
    res = []
    for fold in range(1, 6):
        bp = xgb.Booster(); load_model(bp, f"{PSMM}/KinSPEC_MCA_training_band_10_50_fold{fold}_pure_PSM_model_v4.json", config)
        bm = xgb.Booster(); load_model(bm, f"{F}/model/KinSPEC_MCA_training_band_10_50_fold{fold}_model_v4.json", config)
        if bp.feature_names is None: bp.feature_names = FN(2372)
        if bm.feature_names is None: bm.feature_names = FN(617)
        res.append({"fold": fold, "rep": "PSM", "case": "full", "auc": sel_auc(bp, psm, "PSM")})
        res.append({"fold": fold, "rep": "MCA", "case": "full", "auc": sel_auc(bm, mca_of(Z), "MCA")})
        for case, keep in (("retain SHARED", shared), ("retain DISCRIM", discrim), ("all permuted", np.array([], int))):
            ap, am = [], []
            for r in range(R):
                Pm, Mm = intervene(keep, rng.permutation(482))
                ap.append(sel_auc(bp, Pm, "PSM")); am.append(sel_auc(bm, Mm, "MCA"))
            res.append({"fold": fold, "rep": "PSM", "case": case, "auc": float(np.mean(ap))})
            res.append({"fold": fold, "rep": "MCA", "case": case, "auc": float(np.mean(am))})
        print(f"  fold {fold} done", flush=True)
    d = pd.DataFrame(res); d.to_csv(f"{OUT}/mech2_retain.csv", index=False)

    print("\n" + "=" * 92)
    print("Karaman selectivity AUROC under identical interventions (5 folds x 12 permutations)")
    print("=" * 92)
    print(f"{'case':<16}{'pure PSM':>11}{'MCA':>11}   {'PSM retained':>14}{'MCA retained':>14}")
    piv = d.pivot_table(index="case", columns="rep", values="auc")
    fp, fm = piv.loc["full", "PSM"], piv.loc["full", "MCA"]
    zp, zm = piv.loc["all permuted", "PSM"], piv.loc["all permuted", "MCA"]
    for c in CASES:
        a, b = piv.loc[c, "PSM"], piv.loc[c, "MCA"]
        rp = (a - zp) / (fp - zp) * 100; rm = (b - zm) / (fm - zm) * 100
        print(f"{c:<16}{a:>11.4f}{b:>11.4f}   {rp:>13.1f}%{rm:>13.1f}%")
    print("\n  'retained' = (case - all permuted) / (full - all permuted): the share of each")
    print("  representation's own kinase-side selectivity signal that survives the intervention.")
    for c in ("retain SHARED", "retain DISCRIM"):
        x = d[(d.case == c) & (d.rep == "PSM")].sort_values("fold").auc.values
        y = d[(d.case == c) & (d.rep == "MCA")].sort_values("fold").auc.values
        xf = d[(d.case == "full") & (d.rep == "PSM")].sort_values("fold").auc.values
        yf = d[(d.case == "full") & (d.rep == "MCA")].sort_values("fold").auc.values
        xz = d[(d.case == "all permuted") & (d.rep == "PSM")].sort_values("fold").auc.values
        yz = d[(d.case == "all permuted") & (d.rep == "MCA")].sort_values("fold").auc.values
        rp = (x - xz) / (xf - xz); rm = (y - yz) / (yf - yz)
        t, p = stats.ttest_rel(rm, rp)
        print(f"    {c:<16} PSM {rp.mean()*100:5.1f}%  MCA {rm.mean()*100:5.1f}%   "
              f"MCA higher {(rm>rp).sum()}/5   p={p:.4f}")


if __name__ == "__main__":
    single_main("mech2_retain")
