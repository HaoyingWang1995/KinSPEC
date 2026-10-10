try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np, pandas as pd
    from scipy import stats
    FINAL = str(config.final_dir)
    OUT = str(config.output_dir)
    SP = str(config.split_dir)
    BD = {"10_50": "band_0.10_0.50", "20_60": "band_0.20_0.60", "30_70": "band_0.30_0.70",
          "40_80": "band_0.40_0.80", "50_90": "band_0.50_0.90", "60_100": "band_0.60_1.00"}

    v = pd.read_csv(f"{FINAL}/dataset/VSDS_truedecoy_integrated_data.csv",
                    usecols=["uniprot_id", "KLIFS", "molecule_label"])
    y = v.molecule_label.to_numpy().astype(int)
    tgt = v.KLIFS.astype(str).str.strip().to_numpy()

    print("Is the target scored by its TRAINING ACTIVITY RATE rather than by its pocket?")
    print("rho = Spearman(per-target mean score on VSDS DECOYS, per-target training activity rate).")
    print("Decoys carry no true signal, so target-level structure here is a learned prior, not chemistry.")
    print()
    print(f"{'band':<9}{'PSM rho':>9}{'MCA rho':>9}{'MCA lower':>12}{'paired p':>12}")
    ap, am = [], []
    for band, bdir in BD.items():
        tr = pd.read_csv(f"{SP}/{bdir}/train.csv", usecols=["KLIFS", "unified_bio_label"])
        rate = tr.groupby(tr.KLIFS.astype(str).str.strip())["unified_bio_label"].mean()
        rp, rm = [], []
        for fold in range(1, 6):
            for rep, acc in (("PSM", rp), ("MCA", rm)):
                s = np.load(f"{OUT}/scores_{rep}_{band}_f{fold}.npy")
                g = pd.DataFrame({"t": tgt, "s": s, "y": y}).query("y==0").groupby("t")["s"].mean()
                c = [k for k in g.index if k in rate.index]
                acc.append(stats.spearmanr(g.loc[c].values, rate.loc[c].values).statistic)
        rp, rm = np.array(rp), np.array(rm)
        t, p = stats.ttest_rel(rm, rp)
        ap += list(rp); am += list(rm)
        print(f"{band:<9}{rp.mean():9.3f}{rm.mean():9.3f}{(rm < rp).sum():>9}/5{p:12.4f}")
    ap, am = np.array(ap), np.array(am)
    t, p = stats.ttest_rel(am, ap)
    print(f"{'ALL':<9}{ap.mean():9.3f}{am.mean():9.3f}{(am < ap).sum():>8}/30{p:12.2e}")
    print(f"\n  n targets compared per band: {len(c)}")

    # ---- how separable are the 482 kinases in each representation ----
    print("\n" + "=" * 78)
    print("Identity leakage: how few features give every kinase a distinct code?")
    print("=" * 78)
    EMB = f"{FINAL}/feature_embedding/Kinase_test_MCA"
    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy()
    mca = pd.read_csv(f"{EMB}/MCA_klifs_482_200D_features.csv", skiprows=1, header=None).to_numpy()
    p = psm.mean(0); o = np.argsort(p * (1 - p))[::-1]
    tot = len(set(map(tuple, psm)))
    print(f"  distinct kinases achievable: {tot}/482")
    print(f"  {'k':>5}  {'PSM (top-var bits)':>22}  {'MCA (top-var axes, 20-bin)':>28}")
    mq = np.stack([pd.qcut(mca[:, j], 20, labels=False, duplicates="drop") for j in range(200)], 1)
    om = np.argsort(mca.var(0))[::-1]
    for k in (5, 10, 20, 50, 100, 200):
        a = len(set(map(tuple, psm[:, o[:k]])))
        b = len(set(map(tuple, mq[:, om[:min(k, 200)]])))
        print(f"  {k:>5}  {a:>7} ({a/tot*100:5.1f}%)        {b:>7} ({b/tot*100:5.1f}%)")


if __name__ == "__main__":
    single_main("shortcut")
