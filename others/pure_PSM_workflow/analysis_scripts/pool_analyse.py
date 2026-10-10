try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np, pandas as pd, itertools
    from scipy import stats
    EMB=str(config.final_dir) + '/feature_embedding/Kinase_test_MCA'
    OUT=str(config.output_dir)
    z=np.load(f"{OUT}/pool_attr.npz",allow_pickle=True)
    kin=z["kinases"]; nK=len(kin)
    C=pd.read_csv(f"{EMB}/MCA_200D_category_contributions.csv",index_col=0)
    cat_res=np.array([int(s.split("_")[0][3:])-1 for s in C.index]); Cm=C.to_numpy()
    col_res=np.arange(1955)//23
    rng=np.random.default_rng(42)
    def ps(v,tag):
        o=np.zeros(85)
        if tag=="PSM": np.add.at(o,col_res,v)
        else: np.add.at(o,cat_res,Cm@v)
        return o
    def top(p,k=10): return frozenset(np.argsort(np.abs(p))[::-1][:k])
    def J(a,b): return len(a&b)/len(a|b)
    r=[J(frozenset(rng.choice(85,10,replace=False)),frozenset(rng.choice(85,10,replace=False))) for _ in range(5000)]
    print(f"{nK} kinases (training pool, 30-100 active pairs per kinase). Random 10-of-85 Jaccard baseline = {np.mean(r):.3f}\n")
    print("="*92)
    print("[1] Across-kinase attribution individuality")
    print("="*92)
    print(f"{'rep':<6}{'Across Jaccard':>16}{'Null baseline':>13}{'Unique top10':>14}{'Positions':>11}")
    res={}
    for tag in ("PSM","MCA"):
        S=z[f"{tag}_sgn"]
        P=np.array([ps(S[i],tag) for i in range(nK)]); T=[top(p) for p in P]
        Pn=np.array([ps(rng.permutation(S[i]),tag) for i in range(nK)]); Tn=[top(p) for p in Pn]
        # Mean Jaccard of each kinase against all other kinases gives nK paired values
        per=np.array([np.mean([J(T[i],T[j]) for j in range(nK) if j!=i]) for i in range(nK)])
        pern=np.array([np.mean([J(Tn[i],Tn[j]) for j in range(nK) if j!=i]) for i in range(nK)])
        res[tag]=(per,pern,T)
        print(f"{tag:<6}{per.mean():>16.3f}{pern.mean():>13.3f}{len(set(T)):>11}/{nK}{len(set().union(*T)):>8}/85")
    a,b=res["PSM"][0],res["MCA"][0]
    w=stats.wilcoxon(b,a); print()
    print(f"  Paired by kinase (n={nK}): MCA lower {int((b<a).sum())}/{nK}   "
          f"Wilcoxon p={w.pvalue:.2e}   Median difference {np.median(b-a):+.4f}")
    print(f"  Relative to each model's null baseline: PSM {res['PSM'][0].mean()-res['PSM'][1].mean():+.3f}   "
          f"MCA {res['MCA'][0].mean()-res['MCA'][1].mean():+.3f}")
    print()
    print("="*92)
    print("[2] In-sample check: training-pool vs VSDS attributions (kinases present in both)")
    print("="*92)
    zv=np.load(f"{OUT}/shap_pertarget.npz",allow_pickle=True)
    vt=zv["targets"]
    ref=pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    u2k=dict(zip(ref.UniProt_ID.astype(str),ref.KLIFS.astype(str).str.strip()))
    ki={k:i for i,k in enumerate(kin)}
    pairs=[(i,ki[u2k[u]]) for i,u in enumerate(vt) if u in u2k and u2k[u] in ki]
    print(f"  Kinases shared by both datasets: {len(pairs)}")
    for tag in ("PSM","MCA"):
        Sv=zv[f"full_{tag}_act_sgn"]; Sp=z[f"{tag}_sgn"]
        js=[J(top(ps(Sv[vi],tag)),top(ps(Sp[pi],tag))) for vi,pi in pairs]
        # Control: random pairing
        rnd=[J(top(ps(Sv[vi],tag)),top(ps(Sp[rng.integers(nK)],tag))) for vi,pi in pairs]
        print(f"  {tag}: Same-kinase top-10 Jaccard between datasets {np.mean(js):.3f}   Random pairing {np.mean(rnd):.3f}")


if __name__ == "__main__":
    single_main("pool_analyse")
