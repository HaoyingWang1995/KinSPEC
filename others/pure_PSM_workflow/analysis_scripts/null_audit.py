"""Audit the permutation null: was it a fair test?

The null shuffles which axis/column receives which gain value. It can only move the
position profile if the individual axes' position profiles DIFFER from each other.
Measure that directly, and cross-check with a null that needs no assumptions:
do models trained on genuinely different data (different bands) give different profiles?
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
    C=pd.read_csv(f"{EMB}/MCA_200D_category_contributions.csv",index_col=0)
    cat_res=np.array([int(s.split("_")[0][3:])-1 for s in C.index]); Cm=C.to_numpy()
    col_res=np.arange(1955)//23
    def to_pos(w,grp):
        o=np.zeros(85); np.add.at(o,grp,w); return o/o.sum() if o.sum()>0 else o
    def gain(path,nk):
        b=xgb.Booster(); load_model(b, path, config); g=b.get_score(importance_type="total_gain")
        v=np.zeros(417+nk)
        for k,val in g.items(): v[int(str(k).lstrip("f"))]=val
        del b; return v[417:]

    print("="*88)
    print("[A] how different are the 200 MCA axes' own position profiles from each other?")
    print("    (if they are near-identical, permuting the gains cannot move the profile,")
    print("     and the 0.85 null floor is a statement about axis homogeneity)")
    print("="*88)
    S=np.array([to_pos(Cm[:,k],cat_res) for k in range(200)])       # 200 x 85
    iu=np.triu_indices(200,1)
    cos=(S@S.T)/np.outer(np.linalg.norm(S,axis=1),np.linalg.norm(S,axis=1))
    sp=np.array([stats.spearmanr(S[a],S[b]).statistic for a,b in zip(*np.triu_indices(200,1))[:2]]) \
       if False else None
    print(f"  pairwise cosine between axis position-profiles: mean {cos[iu].mean():.3f}  "
          f"median {np.median(cos[iu]):.3f}  p05 {np.quantile(cos[iu],.05):.3f}  p95 {np.quantile(cos[iu],.95):.3f}")
    u=np.ones(85)/85
    print(f"  cosine of each axis profile to the UNIFORM profile: mean "
          f"{np.mean(S@u/(np.linalg.norm(S,axis=1)*np.linalg.norm(u))):.3f}")
    # same for PSM: a column's profile is a one-hot over positions -> cosine 0 unless same position
    print(f"  (for pure PSM a column's position profile is one-hot: pairwise cosine is 0 or 1)")

    print()
    print("="*88)
    print("[B] assumption-free cross-check: do models trained on DIFFERENT bands give")
    print("    different position profiles?  If not, the readout carries no model information.")
    print("="*88)
    prof={}
    for band in BANDS:
        prof[("PSM",band)]=[to_pos(gain(P+f"/KinSPEC_MCA_training_band_{band}_fold{f}_pure_PSM_model_v4.json",1955),col_res) for f in range(1,6)]
        prof[("MCA",band)]=[to_pos(Cm@gain(F+f"/model/KinSPEC_MCA_training_band_{band}_fold{f}_model_v4.json",200),cat_res) for f in range(1,6)]
    def ov(a,b,k=10):
        return len(set(np.argsort(a)[::-1][:k])&set(np.argsort(b)[::-1][:k]))/k
    for tag in ("PSM","MCA"):
        within=[ov(prof[(tag,b)][i],prof[(tag,b)][j]) for b in BANDS for i in range(5) for j in range(i+1,5)]
        across=[ov(prof[(tag,b1)][i],prof[(tag,b2)][j])
                for x,b1 in enumerate(BANDS) for b2 in BANDS[x+1:] for i in range(5) for j in range(5)]
        wsp=[stats.spearmanr(prof[(tag,b)][i],prof[(tag,b)][j]).statistic for b in BANDS for i in range(5) for j in range(i+1,5)]
        asp=[stats.spearmanr(prof[(tag,b1)][i],prof[(tag,b2)][j]).statistic
             for x,b1 in enumerate(BANDS) for b2 in BANDS[x+1:] for i in range(5) for j in range(5)]
        print(f"  {tag}: top-10 overlap  within-band {np.mean(within):.3f}   across-band {np.mean(across):.3f}"
              f"   gap {np.mean(within)-np.mean(across):+.3f}")
        print(f"  {tag}: Spearman        within-band {np.mean(wsp):.3f}   across-band {np.mean(asp):.3f}"
              f"   gap {np.mean(wsp)-np.mean(asp):+.3f}")
    print()
    print("  within-band pairs share training data; across-band pairs do not.")
    print("  a readout that carries model information must score higher within than across.")


if __name__ == "__main__":
    single_main("null_audit")
