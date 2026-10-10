#!/usr/bin/env python3
"""Recompute model-specific training priors, predict Karaman, and analyze pairs.
Run with the KinSPEC conda environment. Configuration is selected by folder name.
"""

# Resolve defaults from this project's root, independently of the working directory.
from pathlib import Path as _KSPath
import sys as _ks_sys
from functools import partial as _ks_partial
_ks_start = _KSPath(__file__).resolve().parent
_ks_root = next((p for p in (_ks_start, *_ks_start.parents) if (p / 'kinspec_paths.py').is_file()), None)
if _ks_root is None:
    raise RuntimeError('Cannot locate the KinSPEC project root containing kinspec_paths.py')
_ks_sys.path.insert(0, str(_ks_root))
from kinspec_paths import ROOT as KINSPEC_ROOT, project_path as _ks_project_path
_kp = _ks_partial(_ks_project_path, script_dir='others/Prior_conflict_pair_analysis/KinSPEC_LTS_v4')

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from training_prior import calculate_training_rates

OUT = Path(__file__).resolve().parent
ROOT = Path(_kp(''))
LTS = OUT.name == 'KinSPEC_LTS_v4'
TRAIN = ROOT / (_kp('dataset/KinSPEC_LTS_merged_4datasets_deduplicated_no_karaman_sim0.5.csv') if LTS else _kp('dataset/spectrum_data/band_0.10_0.50/train.csv'))
MODELS = ([ROOT/_kp('model/KinSPEC/KinSPEC_LTS_Karaman_sim05_training_pool_MCA_200D_v4.json')] if LTS else [ROOT/_kp(f'model/KinSPEC_default/KinSPEC_MCA_training_band_10_50_fold{i}_default_model_v4.json') for i in range(1,6)])

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    karaman=ROOT/_kp('dataset/Karaman_Kinase_selectivity_filtered_withseq.csv')
    mol_path=ROOT/_kp('dataset/Karaman_10_50_filtered_SAE_latent_top95_features_from_merge_v4.csv')
    ref_path=ROOT/_kp('feature_embedding/Kinase_test_MCA/human_482_typical_kinase_1955bit.csv')
    kin_path=ROOT/_kp('feature_embedding/Kinase_test_MCA/MCA_klifs_482_200D_features.csv')
    df=pd.read_csv(karaman,dtype={'Morgan':str,'KLIFS':str})
    ref=pd.read_csv(ref_path,dtype={'KLIFS':str,'UniProt_ID':str})
    assert not df.duplicated(['molecule_name','UniProt_ID']).any()
    assert set(df.label.unique()) <= {0,1}
    train=pd.read_csv(TRAIN,usecols=['KLIFS','unified_bio_label'],dtype={'KLIFS':str})
    assert train.notna().all().all()
    train['KLIFS']=train.KLIFS.str.strip()
    ref['KLIFS']=ref.KLIFS.str.strip()
    by_klifs,by_uni=calculate_training_rates(train,ref)
    by_klifs.to_csv(OUT/'training_KLIFS_activity_rates.csv',index=False)
    by_uni.to_csv(OUT/'training_target_activity_rates.csv',index=False)
    base=df[['molecule_name','UniProt_ID','label']].merge(by_uni[['UniProt_ID','train_total_rows','train_active_rows','training_activity_rate']],on='UniProt_ID',how='left',validate='many_to_one')
    base['ranking_score']=base.training_activity_rate.fillna(-1.)
    base.to_csv(OUT/'karaman_activity_rate_ranked_long.csv',index=False)
    print(f'Training rows: {len(train)}; targets with prior: {base.loc[base.training_activity_rate.notna(),"UniProt_ID"].nunique()}',flush=True)
    del train
    kin=pd.read_csv(kin_path,skiprows=1,header=None).values
    assert len(kin)==len(ref)
    mapping=dict(zip(ref.KLIFS,kin))
    keys=(df.KLIFS.str.strip() if 'KLIFS' in df else df.UniProt_ID.map(dict(zip(ref.UniProt_ID,ref.KLIFS))))
    assert keys.notna().all()
    kb=np.stack([mapping[k] for k in keys]).astype(np.float32)
    mol=pd.read_csv(mol_path,skiprows=1,header=None).to_numpy(dtype=np.float32)
    assert len(mol)==len(df)
    X=np.hstack([mol,kb])
    assert np.isfinite(X).all()
    dm=xgb.DMatrix(pd.DataFrame(X))
    predictions=[]
    for fold,path in enumerate(MODELS,1):
        model=xgb.Booster()
        model.load_model(path)
        model.set_param({'device':'cpu','nthread':4})
        assert model.num_features()==X.shape[1]
        pred=df[['molecule_name','UniProt_ID','label']].copy()
        pred['score']=model.predict(dm).astype(np.float64)
        pred['fold']=fold
        assert np.isfinite(pred.score).all() and pred.score.between(0,1).all()
        predictions.append(pred)
        print(f'Predicted {path.name}: {len(pred)} rows',flush=True)
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_csv(OUT/'model_predictions_all_folds.csv',index=False)
    for script in ['analyze_prior_conflict_pairs.py','analyze_prior_conflict_pairs_ties_zero.py']:
        with (OUT/(Path(script).stem+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(OUT/script)],stdout=log,stderr=subprocess.STDOUT,check=True)
    # Independent check: pairwise half-credit accuracy must equal ROC AUC.
    pm=pd.read_csv(OUT/'prior_conflict_pair_analysis/per_molecule_per_fold.csv')
    checks=[]
    joined=pred.merge(base[['molecule_name','UniProt_ID','training_activity_rate']],on=['molecule_name','UniProt_ID'],validate='many_to_one')
    for (name,fold),sub in joined.groupby(['molecule_name','fold']):
        for scope in ['observed_training_targets','all_targets_missing_score_minus1']:
            s=sub.loc[sub.training_activity_rate.notna()] if scope=='observed_training_targets' else sub
            if s.label.nunique()!=2: continue
            auc=roc_auc_score(s.label,s.score)
            rows=pm.loc[(pm.molecule_name==name)&(pm.fold==fold)&(pm.scope==scope)]
            acc=rows.loc[rows.category=='all_pairs','model_accuracy'].item()
            assert abs(auc-acc)<1e-12
            assert rows.loc[rows.category!='all_pairs','n_pairs'].sum()==rows.loc[rows.category=='all_pairs','n_pairs'].item()
            checks.append(dict(molecule_name=name,fold=fold,scope=scope,auc=auc,pair_accuracy=acc,absolute_difference=abs(auc-acc)))
    pd.DataFrame(checks).to_csv(OUT/'validation_auc_pairwise.csv',index=False)
    inputs=[TRAIN,karaman,mol_path,ref_path,kin_path,*MODELS]
    metadata=dict(model_family=OUT.name,models=[str(p) for p in MODELS],train_csv=str(TRAIN),inputs_sha256={str(p):digest(p) for p in inputs},feature_shape=list(X.shape),xgboost_version=xgb.__version__,numpy_version=np.__version__,pandas_version=pd.__version__,n_models=len(MODELS),missing_training_targets=sorted(base.loc[base.training_activity_rate.isna(),'UniProt_ID'].unique()),validation_auc_checks=len(checks),training_prior='active training rows / all training rows per KLIFS, mapped to UniProt',aggregation='Mean across model folds within molecule, then equal-weight mean across molecules; LTS has one model, fold=1 is only an identifier')
    (OUT/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    summary=pd.read_csv(OUT/'prior_conflict_pair_analysis/summary.csv')
    primary=summary.loc[summary.scope=='observed_training_targets']
    print(primary.to_string(index=False),flush=True)
    print(f'Completed; {len(checks)} independent AUC checks passed.',flush=True)

if __name__=='__main__': main()
