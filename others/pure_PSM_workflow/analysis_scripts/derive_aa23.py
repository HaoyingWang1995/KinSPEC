"""Derive the amino-acid -> 23-bit PSM lookup and each bit's 'coarseness'
(how many of the 20 amino acids set it). Writes aa23.json used by mech.py / backproj.py."""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd, numpy as np, json, os
    EMB = str(config.final_dir) + '/feature_embedding/Kinase_test_MCA'
    OUT = str(config.output_dir)

    ref = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    oh = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy()
    lk = pd.read_csv(f"{EMB}/MCA_200D_category_lookup.csv").dropna(subset=["bit_index_0_based"])
    lk["bit"] = lk.bit_index_0_based.astype(int)

    B = oh.reshape(482, 85, 23)
    aa = {}
    for i, s in enumerate(ref.KLIFS.to_numpy()):
        for j, c in enumerate(s):
            if c != "-":
                aa.setdefault(c, set()).add(tuple(B[i, j]))
    assert all(len(v) == 1 for v in aa.values()), "an amino acid maps to more than one 23-bit code"
    M = np.array([list(list(aa[a])[0]) for a in sorted(aa)])          # 20 x 23
    names = (lk[lk.residue == 1].sort_values("bit")
             .apply(lambda r: f"{r.system}:{r.category}", axis=1).tolist())
    spec = M.sum(0)                                                   # coarseness per bit

    json.dump({"aa": sorted(aa), "names": names, "spec": spec.tolist()},
              open(f"{OUT}/aa23.json", "w"))
    print(f"20/20 amino acids have a unique 23-bit code; wrote {OUT}/aa23.json")
    for b in np.argsort(-spec):
        print(f"  bit{b:>3} {names[b]:<26} set by {spec[b]:>2}/20 AA")


if __name__ == "__main__":
    single_main("derive_aa23")
