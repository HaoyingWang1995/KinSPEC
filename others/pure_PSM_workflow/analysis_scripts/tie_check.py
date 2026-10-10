"""Is PSM's higher Spearman an artifact of tied zeros?
PSM leaves ~30% of its columns unused in every fold; those entries are all tied at 0 and
agree trivially, which inflates rank correlation. Recompute restricted to features that are
actually used, and report the tie fraction."""

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
    def imp(path,nk):
        b=xgb.Booster(); load_model(b, path, config)
        g=b.get_score(importance_type="total_gain"); v=np.zeros(417+nk)
        for k,val in g.items(): v[int(str(k).lstrip("f"))]=val
        del b; return v[417:]
    print("="*88)
    print("Spearman with and without the trivially-agreeing zero entries")
    print("="*88)
    print(f"{'band':<9}{'rep':<6}{'rho(all)':>10}{'rho(used-in-either)':>21}{'zero-tie frac':>15}")
    rows=[]
    for band in BANDS:
        for tag,pat,nk in (("PSM",P+f"/KinSPEC_MCA_training_band_{band}_fold%d_pure_PSM_model_v4.json",1955),
                           ("MCA",F+f"/model/KinSPEC_MCA_training_band_{band}_fold%d_model_v4.json",200)):
            V=[imp(pat%f,nk) for f in range(1,6)]
            ra,ru,tf=[],[],[]
            for a in range(5):
                for b in range(a+1,5):
                    x,y=V[a],V[b]
                    ra.append(stats.spearmanr(x,y).statistic)
                    m=(x>0)|(y>0)
                    ru.append(stats.spearmanr(x[m],y[m]).statistic)
                    tf.append(((x==0)&(y==0)).mean())
            rows.append(dict(band=band,rep=tag,all=np.mean(ra),used=np.mean(ru),tie=np.mean(tf)))
            print(f"{band:<9}{tag:<6}{np.mean(ra):>10.3f}{np.mean(ru):>21.3f}{np.mean(tf):>15.2f}")
    d=pd.DataFrame(rows)
    print()
    for c in ("all","used"):
        a=d[d.rep=="PSM"].sort_values("band")[c].values; b=d[d.rep=="MCA"].sort_values("band")[c].values
        t,p=stats.ttest_rel(b,a)
        print(f"  rho({c:<4}) PSM {a.mean():.3f}  MCA {b.mean():.3f}  diff {b.mean()-a.mean():+.3f}  "
              f"MCA higher {(b>a).sum()}/6  p={p:.4f}")


if __name__ == "__main__":
    single_main("tie_check")
