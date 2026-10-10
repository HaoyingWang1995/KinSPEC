"""Is PCA's apparent advantage bought by discarding exactly the information MCA is built to keep?

PCA minimises squared error on the raw 0/1 matrix, so a column's influence is its variance
p(1-p): a category held by 1 of 482 kinases has variance 0.002, one held by half has 0.25 --
a 120x difference. MCA weights a category by 1/f (distance from the origin = 1/f - 1).
The two therefore disagree most on rare categories. Test what each 200-d subspace retains.
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
    import numpy as np, pandas as pd
    from sklearn.decomposition import PCA

    EMB = str(config.final_dir) + '/feature_embedding/Kinase_test_MCA'
    OUT = str(config.output_dir)
    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy(float)
    mca = pd.read_csv(f"{EMB}/MCA_klifs_482_200D_features.csv", skiprows=1, header=None).to_numpy(float)
    prev = psm.mean(0)
    nz = (prev > 0) & (prev < 1)
    strat = np.digitize(prev, [.01, .05, .20])
    LAB = ["rare<1%", "uncommon1-5%", "common5-20%", "promisc>20%"]

    print("=" * 88)
    print("[1] what each method weights a category by")
    print("=" * 88)
    print(f"  {'stratum':<15}{'columns':>8}{'mean prevalence':>17}{'PCA weight p(1-p)':>20}{'MCA weight 1/f-1':>19}")
    for i, l in enumerate(LAB):
        m = (strat == i) & nz
        p = prev[m]
        print(f"  {l:<15}{m.sum():>8}{p.mean():>17.4f}{(p*(1-p)).mean():>20.5f}{(1/p-1).mean():>19.1f}")
    print("  -> PCA's implicit weight spans a 100x range and is smallest exactly where MCA's is largest.")

    print("\n" + "=" * 88)
    print("[2] how well does each 200-d subspace reconstruct each raw column?")
    print("=" * 88)
    X = psm[:, nz]
    pcaobj = PCA(n_components=200, random_state=0).fit(psm)
    Xp = pcaobj.inverse_transform(pcaobj.transform(psm))[:, nz]
    # MCA: least-squares reconstruction of the raw columns from the 200 MCA coordinates
    A = np.hstack([np.ones((482, 1)), mca])
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    Xm = A @ coef

    def r2(Xt, Xh):
        ss = ((Xt - Xt.mean(0)) ** 2).sum(0)
        return 1 - ((Xt - Xh) ** 2).sum(0) / np.where(ss > 0, ss, 1)

    r2p, r2m = r2(X, Xp), r2(X, Xm)
    s2 = strat[nz]
    print(f"  {'stratum':<15}{'columns':>8}{'PCA-200 R2':>13}{'MCA-200 R2':>13}{'difference':>13}")
    for i, l in enumerate(LAB):
        m = s2 == i
        if m.sum() == 0: continue
        print(f"  {l:<15}{m.sum():>8}{r2p[m].mean():>13.3f}{r2m[m].mean():>13.3f}{r2m[m].mean()-r2p[m].mean():>+13.3f}")
    print(f"  {'ALL':<15}{len(r2p):>8}{r2p.mean():>13.3f}{r2m.mean():>13.3f}{r2m.mean()-r2p.mean():>+13.3f}")
    print(f"\n  columns reconstructed with R2 < 0.5 :  PCA-200 {int((r2p<0.5).sum())}   MCA-200 {int((r2m<0.5).sum())}")
    print(f"  of those, rare(<1%) columns         :  PCA-200 {int(((r2p<0.5)&(s2==0)).sum())}   "
          f"MCA-200 {int(((r2m<0.5)&(s2==0)).sum())}  (of {(s2==0).sum()} rare columns)")

    print("\n" + "=" * 88)
    print("[3] do kinase pairs that differ ONLY at rare states stay distinguishable?")
    print("=" * 88)
    rare = np.where(nz & (strat == 0))[0]
    common = np.where(nz & (strat >= 2))[0]
    from scipy.spatial.distance import pdist
    d_rare = pdist(psm[:, rare], "cityblock")      # how many rare states they differ on
    d_comm = pdist(psm[:, common], "cityblock")
    d_pca = pdist(pcaobj.transform(psm)[:, :200])
    d_mca = pdist(mca)
    sel = (d_comm == 0) & (d_rare > 0)             # identical on common states, differ on rare ones
    print(f"  kinase pairs identical on all common/promiscuous states but differing on rare ones: {int(sel.sum()):,}")
    if sel.sum() > 0:
        allp = np.ones(len(d_pca), bool)
        print(f"  {'':<22}{'these pairs':>14}{'all pairs':>12}{'ratio':>9}")
        for nm, d in (("PCA-200 distance", d_pca), ("MCA-200 distance", d_mca)):
            a, b = d[sel].mean(), d[allp].mean()
            print(f"  {nm:<22}{a:>14.4f}{b:>12.4f}{a/b:>9.3f}")
        print("  (ratio near 0 = the method has collapsed these kinases onto each other)")
    from scipy import stats
    print(f"\n  Spearman(embedding distance, number of differing RARE states):")
    print(f"    PCA-200 {stats.spearmanr(d_pca, d_rare).statistic:+.4f}    MCA-200 {stats.spearmanr(d_mca, d_rare).statistic:+.4f}")
    print(f"  Spearman(embedding distance, number of differing COMMON states):")
    print(f"    PCA-200 {stats.spearmanr(d_pca, d_comm).statistic:+.4f}    MCA-200 {stats.spearmanr(d_mca, d_comm).statistic:+.4f}")


if __name__ == "__main__":
    single_main("pca_vs_mca")
