"""Test discrimination among kinases highly similar to a molecule's active targets.
Stratify inactive kinases by their maximum similarity to the active kinases for that molecule,
and calculate similarity dependence and ranking performance within each stratum."""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def finite_pairs(psm, mca):
    """Keep molecule pairing when either correlation is undefined."""
    import numpy as np
    psm, mca = np.asarray(psm, dtype=float), np.asarray(mca, dtype=float)
    if psm.shape != mca.shape:
        raise ValueError("Paired correlations must have the same shape")
    valid = np.isfinite(psm) & np.isfinite(mca)
    return psm[valid], mca[valid]

def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np, pandas as pd, xgboost as xgb
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    F=str(config.final_dir); EMB=f"{F}/feature_embedding/Kinase_test_MCA"
    PSMM=str(config.psm_model_dir) + '/KinSPEC_full_training_pool_pure_PSM_v4.json'
    MCAM=f"{F}/model/KinSPEC_full_training_pool_MCA_200D_v4.json"
    k=pd.read_csv(f"{F}/dataset/Karaman_Kinase_selectivity_filtered_withseq.csv")
    mol=pd.read_csv(f"{F}/dataset/Karaman_10_50_filtered_SAE_latent_top95_features_from_merge_v4.csv",
                    skiprows=1,header=None,dtype=np.float32).to_numpy()
    ref=pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    lut=pd.Series(np.arange(len(ref)),index=ref.KLIFS.astype(str).str.strip()).groupby(level=0).last()
    pk=k.UniProt_ID.astype(str).map(dict(zip(ref.UniProt_ID.astype(str),ref.KLIFS.astype(str)))).map(lut)
    ok=pk.notna().to_numpy(); k,mol,pk=k[ok].reset_index(drop=True),mol[ok],pk[ok].to_numpy(np.intp)
    lab=k["label"].to_numpy(); mn=k.molecule_name.to_numpy()
    psm=pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv",header=None).to_numpy(np.float32)
    mca=pd.read_csv(f"{EMB}/MCA_klifs_482_200D_features.csv",skiprows=1,header=None).to_numpy(np.float32)
    S=np.array([list(s) for s in ref.KLIFS.astype(str)])
    FN=lambda n:[str(i) for i in range(n)]
    P=psm.astype(bool); I=P.astype(np.float32)@P.astype(np.float32).T; t=P.sum(1)
    SIM=I/(t[:,None]+t[None,:]-I)
    score={}
    for tag,path,K,nf in (("PSM",PSMM,psm,2372),("MCA",MCAM,mca,617)):
        b=xgb.Booster(); load_model(b, path, config)
        if b.feature_names is None: b.feature_names=FN(nf)
        score[tag]=b.predict(xgb.DMatrix(np.hstack([mol,K[pk]]),feature_names=FN(nf))); del b

    df=pd.DataFrame({"m":mn,"p":pk,"y":lab,"i":np.arange(len(k))})
    # Global quantile cutoffs
    allsim=[]
    for m,g in df.groupby("m"):
        A=g[g.y==1].p.values
        if len(A)==0: continue
        allsim+= [SIM[int(x),A].max() for x in g[g.y==0].p.values]
    qs=np.quantile(allsim,[0.5,0.75,0.9])
    print(f"Quantiles of similarity between inactive kinases and true targets: p50={qs[0]:.3f} p75={qs[1]:.3f} p90={qs[2]:.3f}")
    print()
    print("="*94)
    print("Similarity strata: within-stratum Spearman(score, similarity) -- lower = less reliance on similarity lookup")
    print("="*94)
    bands=[("All",0.0,1.01),("Below median",0.0,qs[0]),("p50-p75",qs[0],qs[1]),
           ("p75-p90",qs[1],qs[2]),("p90 and above (hardest)",qs[2],1.01)]
    print(f"{'Sim. stratum':<16}{'Mols.':>7}{'PSM rho':>10}{'MCA rho':>10}{'Diff.':>9}{'MCA lower':>10}{'p':>11}")
    for nm,lo,hi in bands:
        rp,rm=[],[]
        for m,g in df.groupby("m"):
            A=g[g.y==1].p.values; B=g[g.y==0]
            if len(A)==0 or len(B)<10: continue
            s=np.array([SIM[int(x),A].max() for x in B.p.values])
            sel=(s>=lo)&(s<hi)
            if sel.sum()<8 or np.std(s[sel])==0: continue
            rp.append(stats.spearmanr(score["PSM"][B.i.values[sel]],s[sel]).statistic)
            rm.append(stats.spearmanr(score["MCA"][B.i.values[sel]],s[sel]).statistic)
        rp,rm=finite_pairs(rp,rm)
        n=len(rp)
        w=stats.wilcoxon(rm,rp).pvalue if n>5 else float('nan')
        print(f"{nm:<16}{n:>7}{rp.mean():>10.3f}{rm.mean():>10.3f}{rm.mean()-rp.mean():>+9.3f}{int((rm<rp).sum()):>7}/{n}{w:>11.2e}")
    print()
    print("="*94)
    print("Similarity strata: within-stratum ranking AUROC (active vs inactive kinases in the stratum) -- higher is better")
    print("="*94)
    print(f"{'Sim. stratum':<16}{'Mols.':>7}{'PSM':>9}{'MCA':>9}{'Diff.':>9}{'MCA higher':>10}{'p':>11}")
    for nm,lo,hi in bands:
        ap,am=[],[]
        for m,g in df.groupby("m"):
            A=g[g.y==1]; B=g[g.y==0]
            if len(A)<2 or len(B)<10: continue
            s=np.array([SIM[int(x),A.p.values].max() for x in B.p.values])
            sel=(s>=lo)&(s<hi)
            if sel.sum()<8: continue
            idx=np.concatenate([A.i.values,B.i.values[sel]])
            yy=np.concatenate([np.ones(len(A)),np.zeros(sel.sum())])
            ap.append(roc_auc_score(yy,score["PSM"][idx])); am.append(roc_auc_score(yy,score["MCA"][idx]))
        ap,am=np.array(ap),np.array(am); n=len(ap)
        w=stats.wilcoxon(am,ap).pvalue if n>5 else float('nan')
        print(f"{nm:<16}{n:>7}{ap.mean():>9.3f}{am.mean():>9.3f}{am.mean()-ap.mean():>+9.3f}{int((am>ap).sum()):>7}/{n}{w:>11.2e}")


if __name__ == "__main__":
    single_main("strat")
