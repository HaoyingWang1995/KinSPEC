"""Per-kinase position readout: does it separate kinases, is it reproducible, and is it
floor-limited?  Matching the user's construction: top-10 positions by |net signed residue
attribution|, from signed TreeSHAP on VSDS actives."""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this archived analysis with shared paths; import performs no calculation."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print("Archived analysis: interpret results with the original report and its limitations.")
    import numpy as np, pandas as pd, itertools
    from scipy import stats
    EMB=str(config.final_dir) + '/feature_embedding/Kinase_test_MCA'
    OUT=str(config.output_dir)
    z=np.load(f"{OUT}/shap_pertarget.npz",allow_pickle=True)
    C=pd.read_csv(f"{EMB}/MCA_200D_category_contributions.csv",index_col=0)
    cat_res=np.array([int(s.split("_")[0][3:])-1 for s in C.index]); Cm=C.to_numpy()
    col_res=np.arange(1955)//23
    rng=np.random.default_rng(42)
    def pos_signed(v,tag):
        o=np.zeros(85)
        if tag=="PSM": np.add.at(o,col_res,v)
        else:          np.add.at(o,cat_res,Cm@v)
        return o
    def top(p,k=10): return frozenset(np.argsort(np.abs(p))[::-1][:k])
    def J(a,b): return len(a&b)/len(a|b)
    r=[]
    for _ in range(3000):
        a=frozenset(rng.choice(85,10,replace=False)); b=frozenset(rng.choice(85,10,replace=False)); r.append(J(a,b))
    RAND=np.mean(r)
    print(f"random 10-of-85 Jaccard baseline = {RAND:.3f}\n")

    print("="*88)
    print("[1] full training pool, VSDS actives: do the 34 kinases get DIFFERENT top-10 sets?")
    print("    (lower Jaccard = better separation; null = permute each kinase's own SHAP vector)")
    print("="*88)
    print(f"{'rep':<6}{'Across real':>13}{'Across null':>13}{'distinct sets':>15}{'positions used':>16}")
    for tag in ("PSM","MCA"):
        S=z[f"full_{tag}_act_sgn"]
        P=np.array([pos_signed(S[t],tag) for t in range(34)])
        Pn=np.array([pos_signed(rng.permutation(S[t]),tag) for t in range(34)])
        tr=[top(p) for p in P]; tn=[top(p) for p in Pn]
        dr=np.mean([J(tr[a],tr[b]) for a,b in itertools.combinations(range(34),2)])
        dn=np.mean([J(tn[a],tn[b]) for a,b in itertools.combinations(range(34),2)])
        print(f"{tag:<6}{dr:>13.3f}{dn:>13.3f}{len(set(tr)):>12}/34{len(set().union(*tr)):>13}/85")
    print()
    print("="*88)
    print("[2] band 10-50, 5 folds: is a given kinase's top-10 reproducible across folds?")
    print("="*88)
    print(f"{'rep':<6}{'Fold real':>11}{'Fold null':>11}{'Across real':>13}{'Across null':>13}")
    for tag in ("PSM","MCA"):
        P=np.array([[pos_signed(z[f"b1050_{tag}_f{f}_sgn"][t],tag) for t in range(34)] for f in range(1,6)])
        Pn=np.array([[pos_signed(rng.permutation(z[f"b1050_{tag}_f{f}_sgn"][t]),tag) for t in range(34)] for f in range(1,6)])
        tr=[[top(P[f][t]) for t in range(34)] for f in range(5)]
        tn=[[top(Pn[f][t]) for t in range(34)] for f in range(5)]
        sr=np.mean([J(tr[a][t],tr[b][t]) for t in range(34) for a in range(5) for b in range(a+1,5)])
        sn=np.mean([J(tn[a][t],tn[b][t]) for t in range(34) for a in range(5) for b in range(a+1,5)])
        dr=np.mean([J(tr[f][a],tr[f][b]) for f in range(5) for a,b in itertools.combinations(range(34),2)])
        dn=np.mean([J(tn[f][a],tn[f][b]) for f in range(5) for a,b in itertools.combinations(range(34),2)])
        print(f"{tag:<6}{sr:>11.3f}{sn:>11.3f}{dr:>13.3f}{dn:>13.3f}")
    print()
    print("  Across folds: higher real overlap is better (the same kinase retains the same positions across training folds)")
    print("  Across kinases: lower real overlap is better (different kinases have different positions); high null overlap suggests the mapping homogenizes signatures, while low real overlap suggests homogenization was avoided")


if __name__ == "__main__":
    single_main("perkinase2")
