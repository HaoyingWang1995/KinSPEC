"""Test whether redundancy affects prediction by removing redundant features and retraining.
SHAP is additive: for perfectly collinear columns, the total group credit is the relevant quantity.
Remove 132 constant columns, retain one column per perfectly collinear group,
and retrain each fold with the same hyperparameters."""

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
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    from scipy import stats
    F=str(config.final_dir); EMB=f"{F}/feature_embedding/Kinase_test_MCA"
    SP=str(config.split_dir) + '/band_0.10_0.50'
    SD=f"{F}/dataset/spectrum_data/band_0.10_0.50"
    HP={"learning_rate":0.013558017143562114,"max_depth":10,"subsample":0.8775537271396525,
        "colsample_bytree":0.5557089020337987,"lambda":24.927473547707798,
        "alpha":0.8681806294618319,"gamma":2.7323787690988164,"min_child_weight":10}
    BASE={"objective":"binary:logistic","eval_metric":"auc","tree_method":"hist","device":config.device,
          "scale_pos_weight":14,"seed":0,"nthread":config.nthread}
    psm=pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv",header=None).to_numpy(np.float32)
    prev=psm.mean(0); const=(prev==0)|(prev==1); nz=np.where(~const)[0]
    X=psm[:,nz].T.astype(np.float64); X=(X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-12)
    R=np.abs(X@X.T/psm.shape[0]); np.fill_diagonal(R,0)
    n,lab=connected_components(csr_matrix(R>=0.999),directed=False)
    keep=[]
    for g in range(n): keep.append(nz[np.where(lab==g)[0][0]])
    keep=np.sort(np.array(keep))
    print(f"Original 1955 columns -> remove {int(const.sum())} constant columns and retain 1 column per perfectly collinear group -> {len(keep)} columns")
    tr=pd.read_csv(f"{SP}/train.csv",usecols=["KLIFS","unified_bio_label"])
    molt=pd.read_csv(f"{SD}/train_LSAE_latent_top95_features_v4.csv",skiprows=1,header=None,dtype=np.float32).to_numpy()
    y=tr.unified_bio_label.to_numpy().astype(int)
    ref=pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit.csv")
    lut=pd.Series(np.arange(len(ref)),index=ref.KLIFS.astype(str).str.strip()).groupby(level=0).last()
    post=tr.KLIFS.astype(str).str.strip().map(lut).to_numpy(np.intp)
    v=pd.read_csv(f"{F}/dataset/VSDS_truedecoy_integrated_data.csv",usecols=["uniprot_id","KLIFS","molecule_label"])
    molv=pd.read_csv(f"{F}/dataset/VSDS_truedecoy_LSAE_latent_top95_features_v4.csv",skiprows=1,header=None,dtype=np.float32).to_numpy()
    yv=v.molecule_label.to_numpy().astype(int); tgt=v.uniprot_id.to_numpy()
    posv=v.KLIFS.astype(str).str.strip().map(lut).to_numpy(np.intp)
    hasT=[t for t in np.unique(tgt) if yv[tgt==t].sum()>0 and (yv[tgt==t]==0).sum()>0]
    splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))
    rows=[]
    for name,cols in (("PSM-1955 original",np.arange(1955)),("PSM deduplicated",keep)):
        Xtr=np.hstack([molt,psm[post][:,cols]]); Dte=xgb.DMatrix(np.hstack([molv,psm[posv][:,cols]]))
        for fi,(ti,vi) in enumerate(splits,1):
            print(f"Training {name}, fold {fi}/5 ...", flush=True)
            m=xgb.train({**BASE,**HP},xgb.DMatrix(Xtr[ti],label=y[ti]),num_boost_round=20000,
                        early_stopping_rounds=50,evals=[(xgb.DMatrix(Xtr[vi],label=y[vi]),"e")],verbose_eval=False)
            save_checkpoint(m, config, "dedup", name, fi, {**BASE, **HP})
            s=m.predict(Dte,iteration_range=(0,m.best_iteration+1))
            rows.append({"rep":name,"fold":fi,"nfeat":Xtr.shape[1],"iters":m.best_iteration,
                         "pooled":roc_auc_score(yv,s),
                         "within":float(np.mean([roc_auc_score(yv[tgt==t],s[tgt==t]) for t in hasT]))})
            print(f"  {name:<14} fold {fi}  {Xtr.shape[1]} dimensions  iters {m.best_iteration:>5}  "
                  f"pooled {rows[-1]['pooled']:.4f}",flush=True)
            del m
        del Xtr,Dte
    d=pd.DataFrame(rows)
    print()
    print(f"{'Representation':<16}{'Dim.':>8}{'pooled':>18}{'within':>18}")
    for n_,g in d.groupby("rep",sort=False):
        print(f"{n_:<16}{g.nfeat.iloc[0]:>8}{g.pooled.mean():>11.4f}±{g.pooled.std(ddof=1):<6.4f}"
              f"{g.within.mean():>11.4f}±{g.within.std(ddof=1):<6.4f}")
    a=d[d.rep=="PSM-1955 original"].sort_values("fold").pooled.values
    b=d[d.rep=="PSM deduplicated"].sort_values("fold").pooled.values
    print(f"\n  Difference {b.mean()-a.mean():+.4f}   Deduplicated model better {(b>a).sum()}/5   paired t p={stats.ttest_rel(b,a).pvalue:.3f}")

    d.to_csv(config.output_dir / "dedup.csv", index=False)


if __name__ == "__main__":
    single_main("dedup")
