"""Variable-level strata for symmetric interventions.
Each of the 425 MCA variables (85 residues x 5 property systems) holds exactly one active
level per kinase, so permuting a whole variable across kinases keeps the indicator valid and
the MCA projection exact."""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this archived analysis with shared paths; import performs no calculation."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print("Archived analysis: interpret results with the original report and its limitations.")
    import numpy as np, pandas as pd, json
    EMB = str(config.final_dir) + '/feature_embedding/Kinase_test_MCA'
    OUT = str(config.output_dir)

    G = pd.read_csv(f"{EMB}/MCA_200D_category_coordinates.csv", index_col=0)
    lk = pd.read_csv(f"{EMB}/MCA_200D_category_lookup.csv")
    lk["row"] = lk.category_name.str.replace("_", "", regex=False).map(
        {s.replace("_", ""): i for i, s in enumerate(G.index)})
    Z = np.load(f"{OUT}/Z.npy")

    # map each of the 2113 categories to its parent variable (residue, system)
    var_of = np.full(len(G), -1)
    names = []
    for i, s in enumerate(G.index):
        res = int(s.split("_")[0][3:]); sys = s.split("_")[1]
        key = (res, sys)
        if key not in names: names.append(key)
        var_of[i] = names.index(key)
    Q = len(names)
    print(f"variables: {Q}  (expect 425 = 85 residues x 5 systems)")

    # entropy of each variable across the 482 kinases
    ent = np.zeros(Q); modal = np.zeros(Q); nlev = np.zeros(Q, int)
    for v in range(Q):
        cols = np.where(var_of == v)[0]
        p = Z[:, cols].mean(0); p = p[p > 0]
        ent[v] = -(p * np.log2(p)).sum(); modal[v] = p.max(); nlev[v] = len(p)
    order = np.argsort(ent)
    shared = order[:Q // 2]           # low entropy: most kinases carry the same level
    discrim = order[Q // 2:]          # high entropy: the level actually varies between kinases
    print(f"  entropy: min {ent.min():.3f}  median {np.median(ent):.3f}  max {ent.max():.3f}")
    print(f"  SHARED  half (n={len(shared)}): mean entropy {ent[shared].mean():.3f}, "
          f"mean modal frequency {modal[shared].mean():.3f}")
    print(f"  DISCRIM half (n={len(discrim)}): mean entropy {ent[discrim].mean():.3f}, "
          f"mean modal frequency {modal[discrim].mean():.3f}")
    sysname = [names[v][1] for v in range(Q)]
    print("\n  composition of each half by property system:")
    for s in sorted(set(sysname)):
        a = sum(1 for v in shared if names[v][1] == s); b = sum(1 for v in discrim if names[v][1] == s)
        print(f"    {s:<12} shared {a:>3}   discriminative {b:>3}")
    np.savez(f"{OUT}/strata.npz", var_of=var_of, ent=ent, shared=shared, discrim=discrim,
             res=np.array([names[v][0] for v in range(Q)]),
             sys=np.array([names[v][1] for v in range(Q)]))
    print(f"\nsaved {OUT}/strata.npz")


if __name__ == "__main__":
    single_main("mech2_setup")
