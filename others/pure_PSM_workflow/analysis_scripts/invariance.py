"""Why MCA rather than PCA / random projection, given equal predictive performance?

Test the invariance property directly on the real 482-kinase data: duplicate a redundant
category (split one category into two sub-categories carrying identical information), then
ask whether each embedding's coordinates change.
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
    Z = np.load(f"{OUT}/Z.npy")                      # 482 x 2113 indicator, row sums = Q = 425
    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy(float)
    Q = 425
    rng = np.random.default_rng(0)

    def mca_fit(Zm, Qm, k=200):
        """CA of the indicator matrix -> principal row coordinates + eigenvalues."""
        N = Zm.sum(); P = Zm / N
        r = P.sum(1); c = P.sum(0)
        keep = c > 0
        P = P[:, keep]; c = c[keep]
        S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
        U, d, Vt = np.linalg.svd(S, full_matrices=False)
        lam = d ** 2
        F = (U * d) / np.sqrt(r)[:, None]            # principal row coordinates
        return F[:, :k], lam[:k]

    print("=" * 88)
    print("Invariance to redundant category splitting -- the real 482-kinase PSM data")
    print("=" * 88)
    F0, lam0 = mca_fit(Z, Q)
    P0 = PCA(n_components=200, random_state=0).fit_transform(psm)

    # Split one category into two sub-categories that carry identical information:
    # pick a category, and split the kinases that hold it into two halves recorded as
    # separate levels of the SAME variable. Both halves keep identical joint behaviour with
    # every other variable only if the split is independent of everything else -> use a random
    # half. This is the operation MCA is supposed to be invariant to, up to that independence.
    col = int(np.argmax(Z.sum(0) * (Z.sum(0) < 200)))     # a moderately common category
    holders = np.where(Z[:, col] == 1)[0]
    half = rng.permutation(holders)[: len(holders) // 2]
    Zs = np.hstack([Z, np.zeros((482, 1))])
    Zs[half, col] = 0; Zs[half, -1] = 1                    # same variable, now two levels
    print(f"  split category index {col}: {len(holders)} kinases -> {len(holders)-len(half)} + {len(half)}")
    print(f"  (still one active level per variable: row sums {int(Zs.sum(1).min())}-{int(Zs.sum(1).max())}, Q={Q})")

    Fs, lams = mca_fit(Zs, Q)
    # align: the split adds one dimension at lambda = 1/Q; drop eigenvalues equal to 1/Q
    print(f"\n  MCA eigenvalues equal to 1/Q = {1/Q:.6f}:")
    print(f"    before split: {int(np.sum(np.abs(lam0 - 1/Q) < 1e-9))}   after split: "
          f"{int(np.sum(np.abs(lams - 1/Q) < 1e-9))}")
    keep0 = lam0 > 1/Q + 1e-9; keeps = lams > 1/Q + 1e-9
    n = min(keep0.sum(), keeps.sum())
    A = F0[:, keep0][:, :n]; B = Fs[:, keeps][:, :n]
    sign = np.sign((A * B).sum(0)); B = B * sign
    print(f"  retained axes (lambda > 1/Q): {keep0.sum()} before, {keeps.sum()} after")
    print(f"  max |coordinate difference| on the retained axes : {np.abs(A - B).max():.3e}")
    print(f"  max |eigenvalue difference|  on the retained axes : "
          f"{np.abs(lam0[keep0][:n] - lams[keeps][:n]).max():.3e}")

    # the same operation on the raw one-hot, then PCA
    psms = np.hstack([psm, np.zeros((482, 1))])
    bit = int(np.argmax(psm.sum(0) * (psm.sum(0) < 200)))
    h2 = rng.permutation(np.where(psm[:, bit] == 1)[0]); h2 = h2[: len(h2) // 2]
    psms[h2, bit] = 0; psms[h2, -1] = 1
    Ps = PCA(n_components=200, random_state=0).fit_transform(psms)
    sign = np.sign((P0 * Ps).sum(0)); Ps = Ps * sign
    print(f"\n  PCA-200 on the same split encoding:")
    print(f"    max |coordinate difference| : {np.abs(P0 - Ps).max():.3e}")
    print(f"    mean |coordinate difference|: {np.abs(P0 - Ps).mean():.3e}")
    print(f"    per-kinase displacement (L2, mean over 482): "
          f"{np.linalg.norm(P0 - Ps, axis=1).mean():.4f}  "
          f"(typical coordinate scale {np.abs(P0).mean():.4f})")
    print("\n  Compare the empirical coordinate changes above; a random finite-sample split")
    print("  does not guarantee the exact distribution-equivalence condition.")


if __name__ == "__main__":
    single_main("invariance")
