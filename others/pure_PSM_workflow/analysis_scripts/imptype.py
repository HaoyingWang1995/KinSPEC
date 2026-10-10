"""Do the attribution conclusions depend on the importance type?
total_gain = gain x weight, so `weight` is NOT an independent check on `total_gain`.
The genuinely independent axes are `gain` (per-split, usage removed) and `cover` (sample-based).
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
    import numpy as np, pandas as pd, xgboost as xgb
    from scipy import stats
    F=str(config.final_dir); EMB=f"{F}/feature_embedding/Kinase_test_MCA"
    P=str(config.psm_model_dir)
    BANDS=["10_50","20_60","30_70","40_80","50_90","60_100"]
    TYPES=["total_gain","gain","weight","cover","total_cover"]
    C=pd.read_csv(f"{EMB}/MCA_200D_category_contributions.csv",index_col=0)
    cat_res=np.array([int(s.split("_")[0][3:])-1 for s in C.index]); Cm=C.to_numpy()
    col_res=np.arange(1955)//23
    rng=np.random.default_rng(42)
    def to_pos(w,grp):
        o=np.zeros(85); np.add.at(o,grp,w); return o/o.sum() if o.sum()>0 else o
    def allimp(path,nk):
        b=xgb.Booster(); load_model(b, path, config); out={}
        for t in TYPES:
            g=b.get_score(importance_type=t); v=np.zeros(417+nk)
            for k,val in g.items(): v[int(str(k).lstrip("f"))]=val
            out[t]=v[417:]
        del b; return out
    def ov(a,b,k=10):
        return len(set(np.argsort(a)[::-1][:k])&set(np.argsort(b)[::-1][:k]))/k

    cache={}
    for band in BANDS:
        cache[("PSM",band)]=[allimp(P+f"/KinSPEC_MCA_training_band_{band}_fold{f}_pure_PSM_model_v4.json",1955) for f in range(1,6)]
        cache[("MCA",band)]=[allimp(F+f"/model/KinSPEC_MCA_training_band_{band}_fold{f}_model_v4.json",200) for f in range(1,6)]

    print("="*94)
    print("[A] position-level (85 bins) cross-fold top-10 overlap: REAL vs permutation NULL")
    print("="*94)
    print(f"{'imp type':<13}{'PSM real':>10}{'PSM null':>10}{'PSM used':>10}   {'MCA real':>10}{'MCA null':>10}{'headroom used':>16}")
    for t in TYPES:
        pr,pn,mr,mn,pu=[],[],[],[],[]
        for band in BANDS:
            Pv=[c[t] for c in cache[("PSM",band)]]; Mv=[c[t] for c in cache[("MCA",band)]]
            Pr=[to_pos(v,col_res) for v in Pv]; Mr=[to_pos(Cm@v,cat_res) for v in Mv]
            Pn=[to_pos(rng.permutation(v),col_res) for v in Pv]
            Mn=[to_pos(Cm@rng.permutation(v),cat_res) for v in Mv]
            f=lambda X:np.mean([ov(X[a],X[b]) for a in range(5) for b in range(a+1,5)])
            pr.append(f(Pr)); pn.append(f(Pn)); mr.append(f(Mr)); mn.append(f(Mn))
            pu.append(np.mean([(v>0).mean() for v in Pv]))
        pr,pn,mr,mn=map(np.mean,(pr,pn,mr,mn))
        hp=(pr-pn)/(1-pn)*100 if pn<1 else np.nan
        hm=(mr-mn)/(1-mn)*100 if mn<1 else np.nan
        print(f"{t:<13}{pr:>10.2f}{pn:>10.2f}{np.mean(pu):>10.2f}   {mr:>10.2f}{mn:>10.2f}"
              f"   PSM {hp:>5.1f}% / MCA {hm:>5.1f}%")

    print()
    print("="*94)
    print("[B] own-space cross-fold stability (no back-projection), by importance type")
    print("="*94)
    print(f"{'imp type':<13}{'PSM rho':>9}{'MCA rho':>9}{'PSM t5%':>9}{'MCA t5%':>9}{'PSM t10%':>10}{'MCA t10%':>10}")
    for t in TYPES:
        res={"PSM":[[],[],[]],"MCA":[[],[],[]]}
        for band in BANDS:
            for tag in ("PSM","MCA"):
                V=[c[t] for c in cache[(tag,band)]]; n=len(V[0])
                res[tag][0].append(np.mean([stats.spearmanr(V[a],V[b]).statistic for a in range(5) for b in range(a+1,5)]))
                for i,fr in enumerate((0.05,0.10)):
                    k=max(1,int(round(n*fr)))
                    res[tag][i+1].append(np.mean([ov(V[a],V[b],k) for a in range(5) for b in range(a+1,5)]))
        print(f"{t:<13}{np.mean(res['PSM'][0]):>9.3f}{np.mean(res['MCA'][0]):>9.3f}"
              f"{np.mean(res['PSM'][1]):>9.2f}{np.mean(res['MCA'][1]):>9.2f}"
              f"{np.mean(res['PSM'][2]):>10.2f}{np.mean(res['MCA'][2]):>10.2f}")


if __name__ == "__main__":
    single_main("imptype")
