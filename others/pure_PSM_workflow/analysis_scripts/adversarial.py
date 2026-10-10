"""Does the similarity shortcut cause real selectivity errors?
Adversarial subset (Morgan exp-7 analogue, kinase side):
  - INACTIVE kinases that are highly similar to the compound's active kinases
    (the shortcut says 'active' -- they are not)
  - ACTIVE kinases that are dissimilar to the compound's other active kinases
    (the shortcut says 'inactive' -- they are active)
On this subset a model relying on kinase similarity must do badly; chance AUROC = 0.5.
"""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

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
    FN=lambda n:[str(i) for i in range(n)]
    P=psm.astype(bool)
    inter=P.astype(np.float32)@P.astype(np.float32).T
    tot=P.sum(1)
    SIM=inter/(tot[:,None]+tot[None,:]-inter)     # 482 x 482 Jaccard matrix, independent of the model
    score={}
    for tag,path,K,nf in (("PSM",PSMM,psm,2372),("MCA",MCAM,mca,617)):
        b=xgb.Booster(); load_model(b, path, config)
        if b.feature_names is None: b.feature_names=FN(nf)
        score[tag]=b.predict(xgb.DMatrix(np.hstack([mol,K[pk]]),feature_names=FN(nf))); del b

    rows=[]
    for m,g in pd.DataFrame({"m":mn,"p":pk,"y":lab,"i":np.arange(len(k))}).groupby("m"):
        A=g[g.y==1]; B=g[g.y==0]
        if len(A)<3 or len(B)<10: continue
        ap=A.p.values
        # For each inactive kinase, find its maximum similarity to this molecule's active kinases
        sb=np.array([SIM[int(x),ap].max() for x in B.p.values])
        # For each active kinase, find its maximum similarity to this molecule's other active kinases
        sa=np.array([SIM[int(x),[q for q in ap if q!=x]].max() if len(ap)>1 else 0 for x in ap])
        qb=np.quantile(sb,0.75); qa=np.quantile(sa,0.25)
        idx=np.concatenate([B.i.values[sb>=qb],A.i.values[sa<=qa]])
        yy=np.concatenate([np.zeros((sb>=qb).sum()),np.ones((sa<=qa).sum())])
        if yy.sum()==0 or (yy==0).sum()==0: continue
        r={"m":m,"n_adv":len(yy)}
        for tag in ("PSM","MCA"):
            r[f"{tag}_full"]=roc_auc_score(g.y,score[tag][g.i.values])
            r[f"{tag}_adv"]=roc_auc_score(yy,score[tag][idx])
        rows.append(r)
    d=pd.DataFrame(rows)
    print(f"Available molecules: {len(d)}; adversarial subset mean: {d.n_adv.mean():.0f} kinases/molecule")
    print()
    print(f"{'Subset':<16}{'PSM':>9}{'MCA':>9}{'Diff.':>9}{'MCA higher':>11}{'wilcoxon p':>12}")
    for lab_,c in (("All kinases","full"),("Adversarial subset","adv")):
        a=d[f"PSM_{c}"].values; b_=d[f"MCA_{c}"].values
        w=stats.wilcoxon(b_,a).pvalue
        print(f"{lab_:<16}{a.mean():>9.3f}{b_.mean():>9.3f}{b_.mean()-a.mean():>+9.3f}"
              f"{int((b_>a).sum()):>8}/{len(d)}{w:>12.2e}")
    print()
    print(f"  Molecules with adversarial-subset AUROC below 0.5: PSM {(d.PSM_adv<0.5).sum()}/{len(d)}   MCA {(d.MCA_adv<0.5).sum()}/{len(d)}")
    print("  Adversarial subset = inactive kinases highly similar to true targets (upper quartile) + active kinases dissimilar to other active kinases (lower quartile)")
    d.to_csv(str(config.output_dir) + '/adversarial.csv',index=False)


if __name__ == "__main__":
    single_main("adversarial")
