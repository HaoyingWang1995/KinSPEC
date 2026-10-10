"""Per-TARGET SHAP profiles (no averaging over targets) for both representations."""

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
    P=str(config.psm_model_dir)
    OUT=str(config.output_dir)
    CH=8000
    v=pd.read_csv(f"{F}/dataset/VSDS_truedecoy_integrated_data.csv",usecols=["uniprot_id","KLIFS","molecule_label"])
    mol=pd.read_csv(f"{F}/dataset/VSDS_truedecoy_LSAE_latent_top95_features_v4.csv",skiprows=1,header=None,dtype=np.float32).to_numpy()
    ref=pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    lut=pd.Series(np.arange(len(ref)),index=ref.KLIFS.astype(str).str.strip()).groupby(level=0).last()
    pos=v.KLIFS.astype(str).str.strip().map(lut).to_numpy(np.intp)
    psm=pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv",header=None).to_numpy(np.float32)
    mca=pd.read_csv(f"{EMB}/MCA_klifs_482_200D_features.csv",skiprows=1,header=None).to_numpy(np.float32)
    y=v.molecule_label.to_numpy().astype(int)
    tgt=v.uniprot_id.to_numpy(); T=np.unique(tgt); ti={t:i for i,t in enumerate(T)}
    tix=np.array([ti[t] for t in tgt]); nT=len(T)
    FN=lambda n:[str(i) for i in range(n)]
    ACT=y==1   # actives only, matching the user's VSDS-active analysis
    print(f"VSDS {len(v)} rows, {nT} targets, {ACT.sum()} actives",flush=True)

    checks={}
    def per_target(path,K,nf,nk,mask):
        b=xgb.Booster(); load_model(b, path, config)
        if b.feature_names is None: b.feature_names=FN(nf)
        A=np.zeros((nT,nk)); S=np.zeros((nT,nk)); c=np.zeros(nT)
        X=np.hstack([mol,K[pos]])[mask]; tt=tix[mask]
        diagnostics={}
        for s in range(0,len(X),CH):
            e=min(s+CH,len(X))
            cb=checked_contributions(b, xgb.DMatrix(X[s:e],feature_names=FN(nf)), diagnostics)[:,nf-nk:nf]
            np.add.at(A,tt[s:e],np.abs(cb)); np.add.at(S,tt[s:e],cb); np.add.at(c,tt[s:e],1)
        del b
        if (c == 0).any():
            raise ValueError("A target has no observations for the requested SHAP subset")
        checks[f"{path}|{'all' if mask.all() else 'act'}"]=diagnostics
        return A/c[:,None], S/c[:,None], c

    out={}
    for tag,path,K,nf,nk in (("PSM",P.replace('/model','/model')+"/KinSPEC_full_training_pool_pure_PSM_v4.json",psm,2372,1955),
                             ("MCA",f"{F}/model/KinSPEC_full_training_pool_MCA_200D_v4.json",mca,617,200)):
        t0=time.time()
        for lab,m in ((("act",ACT),("all",np.ones(len(v),bool))) if config.include_band_shap else (("act",ACT),)):
            a,s,c=per_target(path,K,nf,nk,m)
            out[f"full_{tag}_{lab}_abs"]=a; out[f"full_{tag}_{lab}_sgn"]=s; out[f"full_{tag}_{lab}_cnt"]=c
        print(f"  full pool {tag} done ({time.time()-t0:.0f}s)",flush=True)
    for f in (range(1,6) if config.include_band_shap else ()):
        for tag,pat,K,nf,nk in (("PSM",P+"/KinSPEC_MCA_training_band_10_50_fold%d_pure_PSM_model_v4.json",psm,2372,1955),
                                ("MCA",f"{F}/model/KinSPEC_MCA_training_band_10_50_fold%d_model_v4.json",mca,617,200)):
            a,s,c=per_target(pat%f,K,nf,nk,ACT)
            out[f"b1050_{tag}_f{f}_abs"]=a; out[f"b1050_{tag}_f{f}_sgn"]=s
        print(f"  band10_50 fold{f} done",flush=True)
    np.savez(f"{OUT}/shap_pertarget.npz",targets=T,**out)
    (config.output_dir / "shap_pertarget_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print("saved",flush=True)


if __name__ == "__main__":
    single_main("shap_pertarget")
