"""Measure the fraction of SHAP attributable to redundant columns and the allocation of credit within groups.
XGBoost pred_contribs uses path-dependent TreeSHAP. For perfectly collinear columns,
credit depends on how each column is used in the tree structure.
Group totals are defined, while within-group allocations depend on the columns selected by greedy splits."""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np, pandas as pd
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    EMB=str(config.final_dir) + '/feature_embedding/Kinase_test_MCA'
    OUT=str(config.output_dir)
    z=np.load(f"{OUT}/pool_attr.npz",allow_pickle=True)
    S=z["PSM_sgn"]                       # 300 kinases x 1955 features: signed mean SHAP
    psm=pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv",header=None).to_numpy(float)
    prev=psm.mean(0); nz=np.where((prev>0)&(prev<1))[0]
    X=psm[:,nz].T; X=(X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-12)
    R=np.abs(X@X.T/psm.shape[0]); np.fill_diagonal(R,0)
    maxr=np.zeros(1955); maxr[nz]=R.max(1)
    imp=np.abs(S).mean(0)                # Mean absolute SHAP for each column across 300 kinases
    tot=imp.sum()
    print("="*86)
    print("[1] Redundancy strata: column fraction vs mean|SHAP| fraction (full-pool PSM model, 300 kinases)")
    print("="*86)
    grp=np.digitize(maxr,[0.5,0.8,0.95,0.999])
    labs=["max|r|<0.5","0.5-0.8","0.8-0.95","0.95-1.0","=1.00 Perfectly collinear"]
    const=(prev==0)|(prev==1)
    print(f"  {'Stratum':<18}{'Cols.':>7}{'Col. %':>10}{'SHAP %':>11}{'Enrich.':>8}")
    for i,l in enumerate(labs):
        m=(grp==i)&(~const)
        if m.sum()==0: continue
        print(f"  {l:<18}{m.sum():>7}{m.mean()*100:>9.1f}%{imp[m].sum()/tot*100:>10.1f}%{(imp[m].sum()/tot)/m.mean():>8.2f}x")
    print(f"  {'Constant columns':<18}{const.sum():>7}{const.mean()*100:>9.1f}%{imp[const].sum()/tot*100:>10.1f}%")
    print()
    print(f"  |r|>0.95 columns in total: {int(((maxr>0.95)&~const).sum())} columns({((maxr>0.95)&~const).mean()*100:.1f}%), "
          f"Accounts for {imp[(maxr>0.95)&~const].sum()/tot*100:.1f}% of mean|SHAP|")
    print()
    print("="*86)
    print("[2] How is credit distributed within perfectly collinear groups (|r|>=0.999)?")
    print("="*86)
    A=csr_matrix(R>=0.999); n,lab_=connected_components(A,directed=False)
    G=np.full(1955,-1); G[nz]=lab_
    sizes=np.bincount(lab_); multi=np.where(sizes>=2)[0]
    print(f"  Collinear groups: {len(multi)} groups, covering {int(sizes[multi].sum())} columns")
    shares=[];tots=[]; group_shares={}
    for g in multi:
        c=np.where(G==g)[0]; v=imp[c]
        if v.sum()>0:
            group_shares[g] = v.max()/v.sum()
            shares.append(group_shares[g]); tots.append(v.sum())
    shares=np.array(shares)
    print(f"  Total group mean|SHAP| as a fraction of the overall total: {np.sum(tots)/tot*100:.1f}%")
    print(f"  Largest member's fraction within each group: mean {shares.mean():.3f}  Median {np.median(shares):.3f}")
    print(f"    (2 member groups: equal allocation gives 0.50; concentration in one member gives 1.00)")
    sz2=[len(np.where(G==g)[0]) for g in multi]
    print(f"  Group size distribution: {dict(pd.Series(sz2).value_counts().sort_index())}")
    for k in (2,3):
        sel=[g for i,g in enumerate(multi) if sz2[i]==k and g in group_shares]
        if sel: print(f"    {k} member groups (n={len(sel)}): mean largest-member fraction {np.mean([group_shares[g] for g in sel]):.3f}"
                      f"(Equal allocation would be {1/k:.2f})")
    print()
    print("="*86)
    print("[3] Are perfectly collinear groups confined to the same KLIFS position? (Within-group allocation no longer affects results after position aggregation.)")
    print("="*86)
    col_res=np.arange(1955)//23
    cross=[g for g in multi if len(set(col_res[G==g]))>1]
    print(f"  At the same KLIFS position: {len(multi)-len(cross)}/{len(multi)} groups   Across positions: {len(cross)} groups")
    if cross:
        carriers=[]
        for g in cross:
            c=np.where(G==g)[0][0]; n_on=int(psm[:,c].sum())
            carriers.append(min(n_on,psm.shape[0]-n_on))    # Number of kinases on the minority-state side
        print(f"  Kinase counts for minority states in cross-position groups: {dict(pd.Series(carriers).value_counts().sort_index())}")
        cs=sum(imp[G==g].sum() for g in cross)
        print(f"  Cross-position group mean|SHAP| as a fraction of the overall total: {cs/tot*100:.3f}%")


if __name__ == "__main__":
    single_main("redund_shap")
