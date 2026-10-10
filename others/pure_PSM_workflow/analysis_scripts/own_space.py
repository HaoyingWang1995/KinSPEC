"""Fold-to-fold stability of the learned importance vector, measured in EACH representation's
OWN space -- no back-projection, so no operator confound.

Comparability across 1955 vs 200 dims is handled by using rank-based measures whose random
baseline is identical in both: top-k% overlap (baseline = k%) and Spearman (baseline = 0).
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
    F=str(config.final_dir); P=str(config.psm_model_dir)
    BANDS=["10_50","20_60","30_70","40_80","50_90","60_100"]

    def imp(path,nk,typ="total_gain"):
        b=xgb.Booster(); load_model(b, path, config)
        g=b.get_score(importance_type=typ); v=np.zeros(417+nk)
        for k,val in g.items(): v[int(str(k).lstrip("f"))]=val
        del b; return v[417:]

    def stab(V,fracs=(0.01,0.05,0.10)):
        n=len(V[0]); out={}
        sp=[stats.spearmanr(V[a],V[b]).statistic for a in range(len(V)) for b in range(a+1,len(V))]
        out["rho"]=np.mean(sp)
        for f in fracs:
            k=max(1,int(round(n*f)))
            ov=[len(set(np.argsort(V[a])[::-1][:k])&set(np.argsort(V[b])[::-1][:k]))/k
                for a in range(len(V)) for b in range(a+1,len(V))]
            out[f"top{int(f*100)}%"]=np.mean(ov)
        # fraction of the block that is ever used, and effective number of features
        out["used_frac"]=np.mean([(v>0).mean() for v in V])
        return out

    print("="*92)
    print("Fold-to-fold stability of the kinase-side importance vector, in each own space")
    print("random baseline: rho = 0 ; topK% overlap = K%  (identical for 1955 and 200 dims)")
    print("="*92)
    print(f"{'band':<9}{'rep':<6}{'rho':>8}{'top1%':>9}{'top5%':>9}{'top10%':>9}{'used':>8}")
    rows=[]
    for band in BANDS:
        for tag,pat,nk in (("PSM",P+f"/KinSPEC_MCA_training_band_{band}_fold%d_pure_PSM_model_v4.json",1955),
                           ("MCA",F+f"/model/KinSPEC_MCA_training_band_{band}_fold%d_model_v4.json",200)):
            V=[imp(pat%f,nk) for f in range(1,6)]
            s=stab(V); s.update(band=band,rep=tag); rows.append(s)
            print(f"{band:<9}{tag:<6}{s['rho']:>8.3f}{s['top1%']:>9.2f}{s['top5%']:>9.2f}{s['top10%']:>9.2f}{s['used_frac']:>8.2f}")
    d=pd.DataFrame(rows)
    print()
    print(f"{'MEAN':<9}{'PSM':<6}"+"".join(f"{d[d.rep=='PSM'][c].mean():>{w}.{p}f}" for c,w,p in
          (("rho",8,3),("top1%",9,2),("top5%",9,2),("top10%",9,2),("used_frac",8,2))))
    print(f"{'MEAN':<9}{'MCA':<6}"+"".join(f"{d[d.rep=='MCA'][c].mean():>{w}.{p}f}" for c,w,p in
          (("rho",8,3),("top1%",9,2),("top5%",9,2),("top10%",9,2),("used_frac",8,2))))
    print()
    for c in ("rho","top1%","top5%","top10%"):
        a=d[d.rep=="PSM"].sort_values("band")[c].values; b=d[d.rep=="MCA"].sort_values("band")[c].values
        t,p=stats.ttest_rel(b,a)
        print(f"  {c:<8} MCA - PSM = {b.mean()-a.mean():+.3f}   MCA higher {(b>a).sum()}/6   p={p:.4f}")


if __name__ == "__main__":
    single_main("own_space")
