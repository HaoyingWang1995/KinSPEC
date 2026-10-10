"""What does MCA actually give that raw PSM one-hot does not?

Raw one-hot supplies NO metric: any two distinct property states are equidistant, and a tree
can only test 'does this kinase carry bit X'. MCA embeds the same data under the chi-square
distance, i.e. it LEARNS a kinase-kinase metric from the co-occurrence structure of the 482
kinases. Test whether that learned metric aligns with functional (ligand-binding) similarity
better than Hamming does -- and whether it adds anything beyond plain sequence identity.
"""

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
    from scipy.spatial.distance import pdist, squareform

    F = str(config.final_dir); EMB = f"{F}/feature_embedding/Kinase_test_MCA"
    OUT = str(config.output_dir)

    ref = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy(np.float64)
    mca = pd.read_csv(f"{EMB}/MCA_klifs_482_200D_features.csv", skiprows=1, header=None).to_numpy(np.float64)
    seqs = ref.KLIFS.astype(str).to_numpy()
    up = ref.UniProt_ID.astype(str).to_numpy()
    print(f"{len(ref)} kinases; PSM {psm.shape}; MCA {mca.shape}")

    # ---------- 1. redundancy of the raw encoding ----------
    print("\n" + "=" * 86)
    print("[1] how much of the raw PSM encoding is actually free to vary")
    print("=" * 86)
    B = psm.reshape(482, 85, 23)
    u, s, _ = np.linalg.svd(psm - psm.mean(0), full_matrices=False)
    ev = (s ** 2) / (s ** 2).sum(); cum = np.cumsum(ev)
    print(f"  nominal size            : 85 residues x 23 bits = 1955 columns")
    print(f"  information ceiling     : 85 positions x log2(21 states) = "
          f"{85*np.log2(21):.0f} bits  (the 23 bits are a deterministic lookup from 20 AA + gap)")
    print(f"  linear rank             : {(s>1e-9).sum()}")
    print(f"  PCs for 95% / 99% var   : {np.argmax(cum>=.95)+1} / {np.argmax(cum>=.99)+1}")
    print(f"  constant columns        : {int(((psm.mean(0)==0)|(psm.mean(0)==1)).sum())}")
    print("  -> the 1955 columns carry far fewer degrees of freedom than their count suggests;")
    print("     XGBoost nevertheless has to find its splits among all 1955.")

    # ---------- 2. the three metrics ----------
    print("\n" + "=" * 86)
    print("[2] kinase-kinase distance under each representation")
    print("=" * 86)
    d_ham = pdist(psm, "hamming")                       # raw one-hot: simple matching
    d_mca = pdist(mca, "euclidean")                     # MCA: chi-square distance
    S = np.array([list(x) for x in seqs])
    n = len(S)
    iu = np.triu_indices(n, 1)
    d_seq = 1.0 - np.array([(S[i] == S[j]).mean() for i, j in zip(*iu)])   # 1 - identity over 85 aa
    print(f"  Hamming(one-hot)  vs  1-seq_identity : Spearman {stats.spearmanr(d_ham,d_seq).statistic:+.4f}")
    print(f"  MCA(chi-square)   vs  1-seq_identity : Spearman {stats.spearmanr(d_mca,d_seq).statistic:+.4f}")
    print(f"  Hamming           vs  MCA            : Spearman {stats.spearmanr(d_ham,d_mca).statistic:+.4f}")

    # ---------- 3. functional similarity from Karaman ----------
    print("\n" + "=" * 86)
    print("[3] does the metric match FUNCTIONAL (ligand-binding) similarity?")
    print("=" * 86)
    k = pd.read_csv(f"{F}/dataset/Karaman_Kinase_selectivity_filtered_withseq.csv")
    piv = k.pivot_table(index="UniProt_ID", columns="molecule_name", values="label", aggfunc="max")
    piv = piv.dropna(thresh=int(0.8 * piv.shape[1]))
    keep = [i for i, u_ in enumerate(up) if u_ in piv.index]
    sub = piv.loc[[up[i] for i in keep]]
    A = sub.to_numpy(np.float64)
    print(f"  Karaman activity matrix: {A.shape[0]} kinases x {A.shape[1]} compounds "
          f"(kinases with >=80% of compounds measured)")
    M = ~np.isnan(A)
    fun = []
    pairs = []
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            m = M[a] & M[b]
            if m.sum() < 20: continue
            x, y = A[a][m], A[b][m]
            if x.std() == 0 or y.std() == 0: continue
            inter = ((x == 1) & (y == 1)).sum(); union = ((x == 1) | (y == 1)).sum()
            if union == 0: continue
            fun.append(1 - inter / union)          # 1 - Jaccard of active compound sets
            pairs.append((keep[a], keep[b]))
    fun = np.array(fun); pairs = np.array(pairs)
    print(f"  usable kinase pairs: {len(fun):,}")
    DH = squareform(d_ham); DM = squareform(d_mca); DS = squareform(d_seq)
    h = DH[pairs[:, 0], pairs[:, 1]]; m_ = DM[pairs[:, 0], pairs[:, 1]]; sq = DS[pairs[:, 0], pairs[:, 1]]
    print(f"\n  Spearman(distance, 1 - Jaccard of shared actives):")
    print(f"    raw PSM  Hamming        {stats.spearmanr(h,fun).statistic:+.4f}")
    print(f"    MCA      chi-square     {stats.spearmanr(m_,fun).statistic:+.4f}")
    print(f"    KLIFS    1-seq identity {stats.spearmanr(sq,fun).statistic:+.4f}")

    # partial: does each add anything beyond sequence identity?
    def partial(x, y, z):
        rxz = stats.spearmanr(x, z).statistic; ryz = stats.spearmanr(y, z).statistic
        rxy = stats.spearmanr(x, y).statistic
        return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    print(f"\n  partial Spearman with functional distance, controlling for sequence identity:")
    print(f"    raw PSM  Hamming     {partial(h,fun,sq):+.4f}")
    print(f"    MCA      chi-square  {partial(m_,fun,sq):+.4f}")

    # ---------- 4. nearest-neighbour quality ----------
    print("\n" + "=" * 86)
    print("[4] nearest-neighbour retrieval: is a kinase's nearest neighbour functionally similar?")
    print("=" * 86)
    FD = np.full((len(keep), len(keep)), np.nan)
    idx = {v: i for i, v in enumerate(keep)}
    for (a, b), f_ in zip(pairs, fun):
        FD[idx[a], idx[b]] = FD[idx[b], idx[a]] = f_
    sH = DH[np.ix_(keep, keep)].copy(); sM = DM[np.ix_(keep, keep)].copy()
    sS = DS[np.ix_(keep, keep)].copy()
    for X in (sH, sM, sS): np.fill_diagonal(X, np.inf)
    print(f"{'k':>4}  {'raw PSM Hamming':>18}{'MCA chi-square':>18}{'seq identity':>16}   (mean functional")
    print(f"{'':>4}  {'':>18}{'':>18}{'':>16}    distance to the k nearest)")
    for kk in (1, 3, 5, 10):
        out = []
        for X in (sH, sM, sS):
            v = []
            for i in range(len(keep)):
                nb = np.argsort(X[i])[:kk]
                d = FD[i, nb]; d = d[~np.isnan(d)]
                if len(d): v.append(d.mean())
            out.append(np.mean(v))
        base = np.nanmean(FD)
        print(f"{kk:>4}  {out[0]:>18.4f}{out[1]:>18.4f}{out[2]:>16.4f}   (random baseline {base:.4f})")


if __name__ == "__main__":
    single_main("geom")
