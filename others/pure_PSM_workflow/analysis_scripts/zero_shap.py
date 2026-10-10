"""Compare SHAP-weighted, gain-weighted, and unweighted statistics for experiment 2.
Use the same full-pool PSM model throughout to isolate the weighting scheme."""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np, pandas as pd, xgboost as xgb
    F=str(config.final_dir); EMB=f"{F}/feature_embedding/Kinase_test_MCA"
    PSMM=str(config.psm_model_dir) + '/KinSPEC_full_training_pool_pure_PSM_v4.json'
    OUT=str(config.output_dir)
    psm=pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv",header=None).to_numpy()
    prev=psm.mean(0)
    st=np.load(f"{OUT}/strata.npz",allow_pickle=True); vres,vsys=st["res"],st["sys"]
    lk=pd.read_csv(f"{EMB}/MCA_200D_category_lookup.csv").dropna(subset=["bit_index_0_based"])
    lk["bit"]=lk.bit_index_0_based.astype(int); r1=lk[lk.residue==1].sort_values("bit")
    bit2sys=dict(zip(r1.bit,r1.system))
    vidx={(r,s):i for i,(r,s) in enumerate(zip(vres,vsys))}
    col_var=np.array([vidx[(1+c//23,bit2sys[c%23])] for c in range(1955)])
    B=psm.reshape(482,85,23)
    nlev=np.zeros(425,int)
    for v in range(425):
        cols=np.where(col_var==v)[0]
        nlev[v]=int((psm[:,cols].mean(0)>0).sum())+int((psm[:,cols].sum(1)==0).any())
    alt=nlev[col_var]-1                       # Number of states merged into the zero side

    b=xgb.Booster(); load_model(b, PSMM, config)
    if b.feature_names is None: b.feature_names=[str(i) for i in range(2372)]
    G=np.zeros(1955); W=np.zeros(1955)
    for imp,acc in (("total_gain",G),("weight",W)):
        g=b.get_score(importance_type=imp)
        for k,v in g.items():
            i=int(str(k).lstrip("f"))
            if i>=417: acc[i-417]=v
    del b
    S=np.abs(np.load(f"{OUT}/pool_attr.npz",allow_pickle=True)["PSM_sgn"]).mean(0)   # mean|SHAP|
    used=W>0
    print(f"Full-pool PSM model: kinase columns used for splits {used.sum()}/1955\n")
    print("="*84)
    print("Three statistics from experiment 2 under three weighting schemes (same model)")
    print("="*84)
    print(f"{'Weighting scheme':<22}{'Zero states':>16}{'Mean prevalence':>18}{'-> Zero-branch %':>20}")
    for nm,w in (("mean|SHAP| weighted",S),("total_gain weighted",G),("weight (split-count) weighted",W),
                 ("Unweighted (used columns)",used.astype(float))):
        m=w>0
        a=(w[m]*alt[m]).sum()/w[m].sum(); p=(w[m]*prev[m]).sum()/w[m].sum()
        print(f"{nm:<22}{a:>16.2f}{p:>18.3f}{(1-p)*100:>19.1f}%")
    print()
    print("="*84)
    print("What fraction of splits put most kinases in the negative branch? (Grouped by the fraction absorbed by the zero branch.)")
    print("="*84)
    print(f"{'Zero-branch %':<20}{'Cols.':>7}{'SHAP %':>11}{'Gain %':>11}{'Split %':>14}")
    tot={k:v[used].sum() for k,v in (("S",S),("G",G),("W",W))}
    for lo,hi in ((0,.5),(.5,.8),(.8,.95),(.95,1.0)):
        m=used&(1-prev>=lo)&(1-prev<hi)
        print(f"{f'{lo*100:.0f}-{hi*100:.0f}%':<20}{m.sum():>7}{S[m].sum()/tot['S']*100:>10.1f}%"
              f"{G[m].sum()/tot['G']*100:>10.1f}%{W[m].sum()/tot['W']*100:>13.1f}%")
    m=used&(1-prev>=.5)&(1-prev<.95)
    print(f"\n  50-95% Total: SHAP {S[m].sum()/tot['S']*100:.1f}%  "
          f"gain {G[m].sum()/tot['G']*100:.1f}%  Split count {W[m].sum()/tot['W']*100:.1f}%")


if __name__ == "__main__":
    single_main("zero_shap")
