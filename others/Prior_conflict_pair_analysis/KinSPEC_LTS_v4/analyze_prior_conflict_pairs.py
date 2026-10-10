#!/usr/bin/env python3
"""Compare active/inactive kinase pairs whose training-rate prior conflicts with truth.

Primary scope: both targets have observed training activity rates. Sensitivity
scope: include missing-training targets scored -1 by the existing baseline.
Scores, rather than source-order tie breaks, define pairwise rankings. Strict
correction requires model(active) > model(inactive); a tie receives 0.5 credit
in pairwise accuracy. Folds are averaged within molecules before aggregation.
No test against baseline zero is performed: it is zero by conflict selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent


def pair_classes(prior_margin, model_margin):
    return {'conflict': prior_margin < 0, 'concordant': prior_margin > 0,
            'prior_tie': prior_margin == 0}, (model_margin > 0).astype(float) + 0.5 * (model_margin == 0)


def bootstrap_mean(values, rng, repetitions):
    values = np.asarray(values, dtype=float)
    means = np.mean(rng.choice(values, size=(repetitions, len(values)), replace=True), axis=1)
    return np.quantile(means, [0.025, 0.975])


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,default=BASE/'model_predictions_all_folds.csv')
    parser.add_argument('--baseline',type=Path,default=BASE/'karaman_activity_rate_ranked_long.csv')
    parser.add_argument('--output-dir',type=Path,default=BASE/'prior_conflict_pair_analysis')
    parser.add_argument('--bootstrap-repetitions',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=20260923)
    args=parser.parse_args()
    if args.bootstrap_repetitions < 1: raise ValueError('Bootstrap repetitions must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pred=pd.read_csv(args.predictions)
    baseline=pd.read_csv(args.baseline,usecols=['molecule_name','UniProt_ID','label','ranking_score','training_activity_rate'])
    key=['molecule_name','UniProt_ID']
    if baseline.duplicated(key).any() or pred.duplicated(key+['fold']).any():
        raise ValueError('Duplicate molecule/target keys')
    if not set(baseline.label.unique()) <= {0,1} or not set(pred.label.unique()) <= {0,1}:
        raise ValueError('Labels must be binary')
    expected=pd.MultiIndex.from_frame(baseline[key])
    for fold, sub in pred.groupby('fold'):
        if set(pd.MultiIndex.from_frame(sub[key])) != set(expected):
            raise ValueError(f'Fold {fold} does not contain the complete baseline key set')
    joined=pred.merge(baseline,on=key,validate='many_to_one',suffixes=('_model','_baseline'))
    if not joined.label_model.equals(joined.label_baseline): raise ValueError('Label mismatch')
    if not np.isfinite(joined[['score','ranking_score']].to_numpy()).all(): raise ValueError('Nonfinite scores')
    # Minimal synthetic check: reversed prior pairs corrected, tied, or still reversed.
    masks,credit=pair_classes(np.array([-1.,-1.,-1.,1.,0.]),np.array([1.,0.,-1.,-1.,1.]))
    assert masks['conflict'].sum()==3 and np.array_equal(credit,[1.,0.5,0.,0.,1.])
    counts=baseline.groupby('molecule_name').label.sum()
    rows=[]; details=[]
    for (name,fold),full in joined.groupby(['molecule_name','fold'],sort=False):
        for scope in ['observed_training_targets','all_targets_missing_score_minus1']:
            sub=full.loc[full.training_activity_rate.notna()] if scope=='observed_training_targets' else full
            active=sub.loc[sub.label_model.eq(1)]
            inactive=sub.loc[sub.label_model.eq(0)]
            if active.empty or inactive.empty: continue
            pm=active.ranking_score.to_numpy()[:,None]-inactive.ranking_score.to_numpy()[None,:]
            mm=active.score.to_numpy()[:,None]-inactive.score.to_numpy()[None,:]
            masks,credit=pair_classes(pm,mm)
            for category,mask in {**masks,'all_pairs':np.ones(pm.shape,dtype=bool)}.items():
                n=int(mask.sum())
                if not n: continue
                basecredit=(pm>0).astype(float)+0.5*(pm==0)
                rows.append(dict(molecule_name=name,fold=fold,scope=scope,category=category,
                    n_active_full=int(counts[name]),n_active_scope=len(active),n_inactive_scope=len(inactive),
                    n_pairs=n,correct=int(((mm>0)&mask).sum()),ties=int(((mm==0)&mask).sum()),
                    reversed=int(((mm<0)&mask).sum()),model_accuracy=float(credit[mask].mean()),
                    baseline_accuracy=float(basecredit[mask].mean()),strict_correct_rate=float((mm[mask]>0).mean())))
            # Save every strict-conflict pair once per fold, with training coverage flags.
            if scope=='all_targets_missing_score_minus1':
                ai,ni=np.where(masks['conflict'])
                details.append(pd.DataFrame(dict(molecule_name=name,fold=fold,
                    active_target=active.UniProt_ID.to_numpy()[ai],inactive_target=inactive.UniProt_ID.to_numpy()[ni],
                    active_prior=active.ranking_score.to_numpy()[ai],inactive_prior=inactive.ranking_score.to_numpy()[ni],
                    both_targets_observed=(active.training_activity_rate.notna().to_numpy()[ai]&inactive.training_activity_rate.notna().to_numpy()[ni]),
                    prior_margin=pm[ai,ni],model_margin=mm[ai,ni],model_pair_credit=credit[ai,ni])))
    fold_metrics=pd.DataFrame(rows)
    if fold_metrics.empty: raise ValueError('No evaluable active/inactive pairs')
    fold_metrics.to_csv(args.output_dir/'per_molecule_per_fold.csv',index=False)
    if details: pd.concat(details,ignore_index=True).to_csv(args.output_dir/'conflict_pairs_all_folds.csv.gz',index=False,compression='gzip')
    group=['molecule_name','scope','category','n_active_full','n_pairs']
    molecules=fold_metrics.groupby(group,as_index=False).agg(
        n_folds=('fold','nunique'),model_accuracy=('model_accuracy','mean'),
        model_accuracy_fold_std=('model_accuracy','std'),baseline_accuracy=('baseline_accuracy','mean'),
        strict_correct_rate=('strict_correct_rate','mean'),mean_tied_pairs=('ties','mean'),mean_reversed_pairs=('reversed','mean'))
    if not molecules.n_folds.eq(pred.fold.nunique()).all(): raise ValueError('Incomplete fold coverage')
    molecules['difference']=molecules.model_accuracy-molecules.baseline_accuracy
    molecules.to_csv(args.output_dir/'per_molecule_fold_mean.csv',index=False)
    rng=np.random.default_rng(args.seed);summary=[]
    for cohort in ['all_defined','notebook_1_to_93_actives']:
        selected=molecules if cohort=='all_defined' else molecules.loc[molecules.n_active_full.between(1,93)]
        for (scope,category),sub in selected.groupby(['scope','category'],sort=False):
            lo,hi=bootstrap_mean(sub.model_accuracy,rng,args.bootstrap_repetitions)
            dlo,dhi=bootstrap_mean(sub.difference,rng,args.bootstrap_repetitions)
            summary.append(dict(cohort=cohort,scope=scope,category=category,n_molecules=len(sub),
                n_unique_molecule_target_pairs=int(sub.n_pairs.sum()),model_macro_accuracy=sub.model_accuracy.mean(),
                model_macro_ci_low=lo,model_macro_ci_high=hi,baseline_macro_accuracy=sub.baseline_accuracy.mean(),
                macro_difference=sub.difference.mean(),difference_ci_low=dlo,difference_ci_high=dhi,
                model_pair_weighted_accuracy=np.average(sub.model_accuracy,weights=sub.n_pairs),
                strict_correction_or_correct_rate=sub.strict_correct_rate.mean(),
                mean_model_tie_rate=(sub.mean_tied_pairs/sub.n_pairs).mean()))
    summary=pd.DataFrame(summary);summary.to_csv(args.output_dir/'summary.csv',index=False)
    excluded=baseline.groupby('molecule_name').agg(n_active=('label','sum'),n_targets=('label','size'))
    excluded['included_all_defined']=(excluded.n_active>0)&(excluded.n_active<excluded.n_targets)
    excluded['included_notebook']=excluded.n_active.between(1,93)
    excluded.to_csv(args.output_dir/'molecule_inclusion.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,cohort in zip(axes,['all_defined','notebook_1_to_93_actives']):
        d=summary.loc[summary.cohort.eq(cohort)&summary.scope.eq('observed_training_targets')].set_index('category').reindex(['conflict','concordant','prior_tie','all_pairs']).dropna(subset=['model_macro_accuracy'])
        x=np.arange(len(d));ax.bar(x-.18,d.baseline_macro_accuracy,.36,label='Training activity rate',color='tab:orange')
        ax.bar(x+.18,d.model_macro_accuracy,.36,label='KinSPEC',color='tab:blue')
        ax.errorbar(x+.18,d.model_macro_accuracy,yerr=[d.model_macro_accuracy-d.model_macro_ci_low,d.model_macro_ci_high-d.model_macro_accuracy],fmt='none',ecolor='black',capsize=3)
        ax.set(xticks=x,xticklabels=d.index,ylim=(0,1.05),ylabel='Macro pairwise accuracy',title=cohort)
        ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(args.output_dir/'pairwise_accuracy.pdf');fig.savefig(args.output_dir/'pairwise_accuracy.png',dpi=200);plt.close(fig)
    metadata=dict(predictions=str(args.predictions.resolve()),predictions_sha256=sha256(args.predictions),
        baseline=str(args.baseline.resolve()),baseline_sha256=sha256(args.baseline),folds=sorted(pred.fold.unique().tolist()),
        conflict='label(active)=1, label(inactive)=0, prior(active)<prior(inactive)',
        tie_credit=0.5,primary_scope='both targets have observed training activity rates',
        aggregation='Average metrics across folds within each molecule, then average molecules equally',
        uncertainty='Percentile bootstrap resampling molecules after fold averaging; conditional on these targets, dataset and models; does not account for shared-target dependence',
        bootstrap_repetitions=args.bootstrap_repetitions,seed=args.seed,
        interpretation='Conflict baseline accuracy is zero by definition. Its correction rate is descriptive, not an independent superiority test or proof of molecule-specific learning. Concordant-pair damage and all-pair accuracy are reported alongside it.')
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(summary.loc[summary.scope.eq('observed_training_targets')].to_string(index=False))
    print(f'Outputs: {args.output_dir}')

if __name__=='__main__': main()
