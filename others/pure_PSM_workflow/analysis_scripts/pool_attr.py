"""Compute per-kinase attribution using the training pool (381 kinases), rather than VSDS (34).
The training pool covers the structures learned by the model and provides about 11 times as many kinases.
TreeSHAP consistency and local accuracy also hold on training data;
gain importance was excluded because of its inconsistency, rather than its data source."""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np, pandas as pd, xgboost as xgb, time, json
    from analysis_scripts.shap_checks import checked_contributions
    F=str(config.final_dir); EMB=f"{F}/feature_embedding/Kinase_test_MCA"
    PSMM=str(config.psm_model_dir) + '/KinSPEC_full_training_pool_pure_PSM_v4.json'
    MCAM=f"{F}/model/KinSPEC_full_training_pool_MCA_200D_v4.json"
    OUT=str(config.output_dir)
    MAXROW=100; MINACT=30

    d=pd.read_csv(f"{F}/dataset/KinSPEC_training_pool_merged_4_datasets_deduplicated_with_smiles.csv",
                  usecols=["KLIFS","unified_bio_label"])
    mol=pd.read_csv(f"{F}/dataset/KinSPEC_training_pool_LSAE_mol_latent_top95_features_v4.csv",
                    dtype=np.float32).to_numpy()
    print(f"pool {len(d):,} rows, mol {mol.shape}",flush=True)
    ref=pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    lut=pd.Series(np.arange(len(ref)),index=ref.KLIFS.astype(str).str.strip()).groupby(level=0).last()
    kseq=d.KLIFS.astype(str).str.strip()
    pos=kseq.map(lut); ok=pos.notna().to_numpy()
    print(f"Rows mapped to the 482-kinase reference table: {ok.sum():,}",flush=True)
    d=d[ok].reset_index(drop=True); mol=mol[ok]; pos=pos[ok].to_numpy(np.intp); kseq=kseq[ok].reset_index(drop=True)
    y=d.unified_bio_label.to_numpy().astype(int)

    rng=np.random.default_rng(42)
    sel=[]; kin=[]
    for kk,g in pd.DataFrame({"k":kseq,"y":y,"i":np.arange(len(d))}).groupby("k"):
        a=g[g.y==1].i.values
        if len(a)<MINACT: continue
        sel.append(a if len(a)<=MAXROW else rng.choice(a,MAXROW,replace=False)); kin.append(kk)
    idx=np.concatenate(sel); grp=np.concatenate([[i]*len(s) for i,s in enumerate(sel)])
    nK=len(kin)
    print(f"Used for analysis: {nK} kinases, {len(idx):,} active rows ({MINACT}-{MAXROW} per kinase)",flush=True)
    psm=pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv",header=None).to_numpy(np.float32)
    mca=pd.read_csv(f"{EMB}/MCA_klifs_482_200D_features.csv",skiprows=1,header=None).to_numpy(np.float32)
    FN=lambda n:[str(i) for i in range(n)]
    out={}; checks={}
    for tag,path,K,nf,nk in (("PSM",PSMM,psm,2372,1955),("MCA",MCAM,mca,617,200)):
        t0=time.time(); b=xgb.Booster(); load_model(b, path, config)
        if b.feature_names is None: b.feature_names=FN(nf)
        X=np.hstack([mol[idx],K[pos[idx]]])
        S=np.zeros((nK,nk)); c=np.zeros(nK)
        checks[tag]={}
        for s in range(0,len(X),6000):
            e=min(s+6000,len(X))
            cb=checked_contributions(b, xgb.DMatrix(X[s:e],feature_names=FN(nf)), checks[tag])[:,nf-nk:nf]
            np.add.at(S,grp[s:e],cb); np.add.at(c,grp[s:e],1)
        out[f"{tag}_sgn"]=S/c[:,None]; del b,X
        print(f"  {tag} done ({time.time()-t0:.0f}s)",flush=True)
    np.savez(f"{OUT}/pool_attr.npz",kinases=np.array(kin),counts=np.array([len(s) for s in sel]),**out)
    (config.output_dir / "pool_attr_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print("saved",flush=True)


if __name__ == "__main__":
    single_main("pool_attr")
