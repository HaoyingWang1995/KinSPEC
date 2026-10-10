from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, wilcoxon
from training_prior import evaluate_molecules
P=Path(__file__).resolve().parent
pred=pd.read_csv(P/'model_predictions_all_folds.csv')
base=pd.read_csv(P/'karaman_activity_rate_ranked_long.csv')
joined=pred.merge(base[['molecule_name','UniProt_ID','training_activity_rate']],on=['molecule_name','UniProt_ID'],validate='one_to_one')
rows=[]
for name,s in joined.groupby('molecule_name'):
 s=s.dropna(subset=['training_activity_rate'])
 rows.append(dict(molecule_name=name,n_targets=len(s),n_active=int(pred.loc[pred.molecule_name==name,'label'].sum()),spearman_score_vs_prior=spearmanr(s.score,s.training_activity_rate).statistic,kendall_tau_b=kendalltau(s.score,s.training_activity_rate).statistic))
corr=pd.DataFrame(rows); corr.to_csv(P/'per_molecule_prior_correlation.csv',index=False)
rng=np.random.default_rng(20260923)
results=[];pairs=[]
for scope in ['all_targets_missing_score_minus1','observed_training_targets']:
 b=base.copy()
 if scope=='observed_training_targets': b=b.loc[b.training_activity_rate.notna()]
 b['found_in_training']=b.training_activity_rate.notna()
 bm,_=evaluate_molecules(b)
 a=b.drop(columns='ranking_score').merge(pred[['molecule_name','UniProt_ID','score']],on=['molecule_name','UniProt_ID'],validate='one_to_one').rename(columns={'score':'ranking_score'})
 am,_=evaluate_molecules(a)
 am=am.set_index('molecule_name');bm=bm.set_index('molecule_name')
 for cohort in ['all_defined','notebook_1_to_93_actives']:
  counts=base.groupby('molecule_name').label.sum()
  names=counts[counts.between(1,93)].index if cohort=='notebook_1_to_93_actives' else am.index[am.AUROC.notna()]
  for metric in ['AUROC','BEDROC_alpha20','precision@10','recall@10']:
   av=am.loc[names,metric];bv=bm.loc[names,metric];d=(av-bv).to_numpy()
   ci=np.quantile(rng.choice(d,size=(10000,len(d)),replace=True).mean(axis=1),[.025,.975])
   pv=wilcoxon(d,alternative='two-sided',method='approx').pvalue if np.any(d) else 1.
   results.append(dict(scope=scope,cohort=cohort,metric=metric,n_molecules=len(d),model_mean=av.mean(),prior_mean=bv.mean(),difference=d.mean(),ci_low=ci[0],ci_high=ci[1],wilcoxon_p=pv))
   pairs.extend(dict(scope=scope,cohort=cohort,metric=metric,molecule_name=n,model=av[n],prior=bv[n],difference=av[n]-bv[n]) for n in names)
r=pd.DataFrame(results)
for _,idx in r.groupby(['scope','cohort']).groups.items():
 ordered=r.loc[idx].sort_values('wilcoxon_p').index
 r.loc[ordered,'holm_p']=np.minimum(1,np.maximum.accumulate(r.loc[ordered,'wilcoxon_p'].to_numpy()*np.arange(4,0,-1)))
r.to_csv(P/'ranking_metrics_significance.csv',index=False)
pd.DataFrame(pairs).to_csv(P/'ranking_metrics_paired_values.csv',index=False)
cs=corr[['spearman_score_vs_prior','kendall_tau_b']].agg(['mean','median','min','max'])
cs.to_csv(P/'prior_correlation_summary.csv')
print(cs.to_string()); print(r.loc[(r.scope=='observed_training_targets') & (r.cohort=='all_defined')].to_string(index=False))
