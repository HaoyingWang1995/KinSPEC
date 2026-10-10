"""Does pure PSM inflate scores for kinases that merely SHARE FEATURES with the true target?
(the kinase-side analogue of Morgan's promiscuous-fragment false positives)

For each Karaman compound: among the kinases it is INACTIVE on, is the model's score higher
for those more similar to the compound's ACTIVE kinases?  Any such correlation on inactive
kinases is guilt-by-association -- a false-positive driver, not chemistry.

Similarity is computed in RAW kinase space (1955 PSM bits / 85-aa identity), i.e. a fixed
property of the kinase pair, identical for both models -- so the comparison is fair.
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
    S=np.array([list(s) for s in ref.KLIFS.astype(str)])
    print(f"Karaman {len(k)} rows, {k.molecule_name.nunique()} molecules, {k.UniProt_ID.nunique()} kinases",flush=True)

    # Original kinase similarity, independent of the model
    def bitsim(a,b): 
        return (psm[a]*psm[b]).sum()/((psm[a]+psm[b])>0).sum()
    def seqid(a,b): return (S[a]==S[b]).mean()

    score={}
    for tag,path,K,nf in (("PSM",PSMM,psm,2372),("MCA",MCAM,mca,617)):
        b=xgb.Booster(); load_model(b, path, config)
        if b.feature_names is None: b.feature_names=FN(nf)
        X=np.hstack([mol,K[pk]])
        score[tag]=b.predict(xgb.DMatrix(X,feature_names=FN(nf)))
        del b

    print()
    print("="*94)
    print("For each molecule: among its inactive kinases, do model scores increase with similarity to kinases active for that molecule?")
    print("(Similarity uses raw PSM bit Jaccard and 85-aa sequence identity, shared by both models for a fair comparison.)")
    print("="*94)
    rows=[]
    for m,g in pd.DataFrame({"m":mn,"p":pk,"y":lab,"i":np.arange(len(k))}).groupby("m"):
        A=g[g.y==1].p.values; B=g[g.y==0]
        if len(A)==0 or len(B)<10: continue
        simb=np.array([max(bitsim(int(bp),int(a)) for a in A) for bp in B.p.values])
        simq=np.array([max(seqid(int(bp),int(a)) for a in A) for bp in B.p.values])
        r={"m":m,"n_inact":len(B)}
        for tag in ("PSM","MCA"):
            s=score[tag][B.i.values]
            r[f"{tag}_bit"]=stats.spearmanr(s,simb).statistic
            r[f"{tag}_seq"]=stats.spearmanr(s,simq).statistic
        rows.append(r)
    d=pd.DataFrame(rows)
    print(f"  Available molecules: {len(d)} (at least 1 active and 10 inactive kinases)")
    print()
    print(f"{'Similarity definition':<20}{'PSM rho':>10}{'MCA rho':>10}{'Diff.':>9}{'MCA lower':>10}{'wilcoxon p':>12}")
    for lab_,c in (("PSM bit Jaccard","bit"),("85-aa sequence identity","seq")):
        a=d[f"PSM_{c}"].values; b_=d[f"MCA_{c}"].values
        w=stats.wilcoxon(b_,a).pvalue
        print(f"{lab_:<20}{np.mean(a):>10.3f}{np.mean(b_):>10.3f}{np.mean(b_)-np.mean(a):>+9.3f}"
              f"{int((b_<a).sum()):>7}/{len(d)}{w:>12.2e}")
    print()
    print("  Higher rho = the model assigns higher scores to inactive kinases resembling true targets (a source of false positives)")
    d.to_csv(str(config.output_dir) + '/guilt.csv',index=False)


if __name__ == "__main__":
    single_main("guilt")
