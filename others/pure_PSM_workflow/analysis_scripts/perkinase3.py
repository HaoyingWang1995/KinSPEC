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
    def pos_signed(v,tag):
        o=np.zeros(85)
        if tag=="PSM": np.add.at(o,col_res,v)
        else:          np.add.at(o,cat_res,Cm@v)
        return o
    def top(p,k=10): return frozenset(np.argsort(np.abs(p))[::-1][:k])
    def J(a,b): return len(a&b)/len(a|b)

    print("="*92)
    print("band 10-50, per fold: separation ratio = (within-kinase overlap across folds) / (across-kinase overlap)")
    print("  Higher = a kinase resembles itself more than other kinases, indicating distinguishable position signatures")
    print("="*92)
    TOPS={}
    for tag in ("PSM","MCA"):
        TOPS[tag]=[[top(pos_signed(z[f"b1050_{tag}_f{f}_sgn"][t],tag)) for t in range(34)] for f in range(1,6)]
    rows=[]
    for tag in ("PSM","MCA"):
        T=TOPS[tag]
        for f in range(5):
            # Across kinases: all pairs of the 34 kinases within this fold
            d=np.mean([J(T[f][a],T[f][b]) for a,b in itertools.combinations(range(34),2)])
            # Across folds: compare this fold with other folds for the same kinase
            s=np.mean([J(T[f][t],T[g][t]) for t in range(34) for g in range(5) if g!=f])
            rows.append(dict(rep=tag,fold=f+1,across=d,within=s,ratio=s/d))
    d=pd.DataFrame(rows)
    print(f"{'fold':<6}{'PSM across':>12}{'PSM folds':>11}{'PSM ratio':>11}   {'MCA across':>12}{'MCA folds':>11}{'MCA ratio':>11}")
    for f in range(1,6):
        p=d[(d.rep=="PSM")&(d.fold==f)].iloc[0]; m=d[(d.rep=="MCA")&(d.fold==f)].iloc[0]
        print(f"{f:<6}{p.across:>12.3f}{p.within:>11.3f}{p.ratio:>11.2f}   {m.across:>12.3f}{m.within:>11.3f}{m.ratio:>11.2f}")
    P=d[d.rep=="PSM"].sort_values("fold"); M=d[d.rep=="MCA"].sort_values("fold")
    print(f"{'MEAN':<6}{P.across.mean():>12.3f}{P.within.mean():>11.3f}{P.ratio.mean():>11.2f}"
          f"   {M.across.mean():>12.3f}{M.within.mean():>11.3f}{M.ratio.mean():>11.2f}")
    print()
    for c,better in (("across","lower"),("within","higher"),("ratio","higher")):
        a=P[c].values; b=M[c].values; t,pv=stats.ttest_rel(b,a)
        w="MCA" if ((b.mean()<a.mean()) == (better=="lower")) else "PSM"
        print(f"  {c:<8}({better:<6}) PSM {a.mean():.3f}  MCA {b.mean():.3f}  "
              f"MCA-PSM {b.mean()-a.mean():+.3f}  Better model {w}  Consistent {(b<a).sum() if better=='lower' else (b>a).sum()}/5  p={pv:.4f}")
    print()
    print("="*92)
    print("Full training pool (one model, no across-fold variance)")
    print("="*92)
    for tag in ("PSM","MCA"):
        S=z[f"full_{tag}_act_sgn"]; T=[top(pos_signed(S[t],tag)) for t in range(34)]
        d_=np.mean([J(T[a],T[b]) for a,b in itertools.combinations(range(34),2)])
        print(f"  {tag}: Across Jaccard {d_:.3f}   Distinct sets {len(set(T))}/34   "
              f"Positions covered {len(set().union(*T))}/85")


if __name__ == "__main__":
    single_main("perkinase3")
