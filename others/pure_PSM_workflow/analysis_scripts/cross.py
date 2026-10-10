"""Decisive: is the PSM-vs-MCA difference a REPRESENTATION effect or a HYPERPARAMETER effect?
2x2 -- each kinase block trained under each hyperparameter set, identical folds and seed."""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np, pandas as pd, xgboost as xgb
    from analysis_scripts.training_checks import save_checkpoint
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    F=str(config.final_dir); EMB=f"{F}/feature_embedding/Kinase_test_MCA"
    SP=str(config.split_dir) + '/band_0.10_0.50'
    SD=f"{F}/dataset/spectrum_data/band_0.10_0.50"
    OUT=str(config.output_dir)
    HP_PSM={"learning_rate":0.013558017143562114,"max_depth":10,"subsample":0.8775537271396525,
            "colsample_bytree":0.5557089020337987,"lambda":24.927473547707798,
            "alpha":0.8681806294618319,"gamma":2.7323787690988164,"min_child_weight":10}
    HP_MCA={"learning_rate":0.022586518295195174,"max_depth":10,"subsample":0.8637291377742428,
            "colsample_bytree":0.6676572974196741,"lambda":640.1799512833429,
            "alpha":0.07919012406718373,"gamma":1.1490769558313774,"min_child_weight":8}
    BASE={"objective":"binary:logistic","eval_metric":"auc","tree_method":"hist","device":config.device,
          "scale_pos_weight":14,"seed":0,"nthread":config.nthread}
    tr=pd.read_csv(f"{SP}/train.csv",usecols=["KLIFS","unified_bio_label"])
    molt=pd.read_csv(f"{SD}/train_LSAE_latent_top95_features_v4.csv",skiprows=1,header=None,dtype=np.float32).to_numpy()
    y=tr.unified_bio_label.to_numpy().astype(int)
    ref=pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    lut=pd.Series(np.arange(len(ref)),index=ref.KLIFS.astype(str).str.strip()).groupby(level=0).last()
    post=tr.KLIFS.astype(str).str.strip().map(lut).to_numpy(np.intp)
    psm=pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv",header=None).to_numpy(np.float32)
    mca=pd.read_csv(f"{EMB}/MCA_klifs_482_200D_features.csv",skiprows=1,header=None).to_numpy(np.float32)
    v=pd.read_csv(f"{F}/dataset/VSDS_truedecoy_integrated_data.csv",usecols=["uniprot_id","KLIFS","molecule_label"])
    molv=pd.read_csv(f"{F}/dataset/VSDS_truedecoy_LSAE_latent_top95_features_v4.csv",skiprows=1,header=None,dtype=np.float32).to_numpy()
    yv=v.molecule_label.to_numpy().astype(int); tgt=v.uniprot_id.to_numpy()
    posv=v.KLIFS.astype(str).str.strip().map(lut).to_numpy(np.intp)
    hasT=[t for t in np.unique(tgt) if yv[tgt==t].sum()>0 and (yv[tgt==t]==0).sum()>0]
    splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))
    rows=[]
    for kn,K in (("PSM-1955",psm),("MCA-200",mca)):
        Xtr=np.hstack([molt,K[post]]); Dte=xgb.DMatrix(np.hstack([molv,K[posv]]))
        for hn,HP in (("HP_PSM",HP_PSM),("HP_MCA",HP_MCA)):
            for fi,(ti,vi) in enumerate(splits,1):
                print(f"Training {kn}, {hn}, fold {fi}/5 ...", flush=True)
                m=xgb.train({**BASE,**HP},xgb.DMatrix(Xtr[ti],label=y[ti]),num_boost_round=20000,
                            early_stopping_rounds=50,evals=[(xgb.DMatrix(Xtr[vi],label=y[vi]),"e")],verbose_eval=False)
                save_checkpoint(m, config, "cross", f"{kn}_{hn}", fi, {**BASE, **HP})
                s=m.predict(Dte,iteration_range=(0,m.best_iteration+1))
                p=roc_auc_score(yv,s); w=float(np.mean([roc_auc_score(yv[tgt==t],s[tgt==t]) for t in hasT]))
                rows.append({"kinase":kn,"hp":hn,"fold":fi,"iters":m.best_iteration,"pooled":p,"within":w})
                print(f"  {kn:<10} {hn:<7} fold {fi}  iters {m.best_iteration:>5}  pooled {p:.4f}  within {w:.4f}",flush=True)
                del m
        del Xtr,Dte
    pd.DataFrame(rows).to_csv(f"{OUT}/cross.csv",index=False); print("\nwrote cross.csv",flush=True)


if __name__ == "__main__":
    single_main("cross")
