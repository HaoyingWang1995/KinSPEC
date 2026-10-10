#!/usr/bin/env python3
"""Descriptive atom-level signed agreement for existing V9 and V4 KernelSHAP.

The runs use different background designs: V9 joint-independent/2000 versus
KernelSHAP fixed-target/500. Metrics do not isolate attribution-method effects.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem

HERE = Path(__file__).resolve().parent
KERNEL_DIRS = {'Dacomitinib_EGFR': 'kernelshap_results_P00533_Dacomitinib_rf',
               'Imatinib_ABL1': 'kernelshap_results_P00519_Imatinib_rf'}
ZERO_TOL = 1e-12


def agreement(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1 or not len(x) or not np.isfinite([x, y]).all():
        raise ValueError('Expected aligned finite nonempty atom score vectors')
    sx = np.where(np.abs(x) <= ZERO_TOL, 0, np.sign(x))
    sy = np.where(np.abs(y) <= ZERO_TOL, 0, np.sign(y))
    both = (sx != 0) & (sy != 0)
    variable = len(x) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0
    ccc_denom = x.var() + y.var() + (x.mean()-y.mean())**2
    cosine_denom = np.linalg.norm(x)*np.linalg.norm(y)
    return dict(n_atoms=len(x), pearson_r=float(pearsonr(x,y).statistic) if variable else np.nan,
                spearman_rho=float(spearmanr(x,y).statistic) if variable else np.nan,
                lin_concordance=float(2*np.mean((x-x.mean())*(y-y.mean()))/ccc_denom) if ccc_denom else np.nan,
                cosine_similarity=float(x@y/cosine_denom) if cosine_denom else np.nan,
                n_same_sign_all=int((sx==sy).sum()), sign_agreement_all=float(np.mean(sx==sy)),
                n_both_nonzero=int(both.sum()),
                n_same_sign_both_nonzero=int((sx[both]==sy[both]).sum()),
                sign_agreement_both_nonzero=float(np.mean(sx[both]==sy[both])) if both.any() else np.nan,
                n_v9_zero=int((sx==0).sum()), n_kernel_zero=int((sy==0).sum()),
                mae=float(np.abs(x-y).mean()), rmse=float(np.sqrt(np.mean((x-y)**2))),
                v9_atom_net_sum=float(x.sum()), kernel_atom_net_sum=float(y.sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v9-dir',type=Path,default=HERE/'encoder_topk_v9/joint_independent')
    p.add_argument('--outdir',type=Path,default=HERE/'v9_vs_kernelshap_atoms_rf')
    args=p.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    config=json.loads((args.v9_dir/'run_config.json').read_text())
    source=pd.read_csv(args.v9_dir/'atom_topk_present_bit_attribution.csv')
    summaries=[];aligned=[];provenance=[]
    for pair,folder in KERNEL_DIRS.items():
        kd=HERE/folder;kc=json.loads((kd/'run_config.json').read_text())
        vp=next(t for t in config['pairs'] if t['pair']==pair)
        assert vp['smiles']==kc['pair']['smiles'] and vp['kinase']==kc['pair']['kinase'], 'Molecule atom order or target mismatch'
        for key in ['sae_model','sae_indices','xgb_model']:
            assert kc[key] in config['files'], f'Model path mismatch: {key}'
        mol=Chem.MolFromSmiles(vp['smiles']);assert mol is not None
        symbols=[atom.GetSymbol() for atom in mol.GetAtoms()]
        kernel=pd.read_csv(kd/'molecule_activebit_kernelshap_atoms.csv')
        kernel=kernel.rename(columns={'activebit_kernel_shap_signed':'kernel_atom_signed','atom_symbol':'kernel_atom_symbol'})
        assert len(kernel)==len(symbols) and not kernel.atom_id.duplicated().any()
        vf=source[source.pair.eq(pair)]
        for selection,sub in vf.groupby('selection',sort=False):
            merged=sub[['pair','selection','atom_id','atom_symbol','signed_present_bit_allocation']].merge(
                kernel[['atom_id','kernel_atom_symbol','kernel_atom_signed']],on='atom_id',validate='one_to_one',how='outer',indicator=True)
            assert merged['_merge'].eq('both').all()
            merged=merged.sort_values('atom_id').drop(columns='_merge')
            assert merged.atom_id.tolist()==list(range(len(symbols)))
            assert merged.atom_symbol.tolist()==merged.kernel_atom_symbol.tolist()==symbols
            merged=merged.rename(columns={'signed_present_bit_allocation':'v9_atom_signed'}).drop(columns='kernel_atom_symbol')
            merged['v9_minus_kernel']=merged.v9_atom_signed-merged.kernel_atom_signed
            aligned.append(merged)
            summaries.append(dict(pair=pair,selection=selection,
                v9_background_mode=config['background_mode'],v9_references=config['actual_molecule_background_size'],
                kernel_background_mode=kc['background_mode'],kernel_references=kc['background_size'],
                **agreement(merged.v9_atom_signed,merged.kernel_atom_signed)))
        all_atoms=next(t for t in aligned if t.pair.iloc[0]==pair and t.selection.iloc[0]=='all')
        x=all_atoms.v9_atom_signed.to_numpy();y=all_atoms.kernel_atom_signed.to_numpy()
        metrics=agreement(x,y)
        fig,axes=plt.subplots(1,2,figsize=(14,5.2))
        axes[0].scatter(x,y,s=32,alpha=.85)
        low,high=min(x.min(),y.min()),max(x.max(),y.max())
        pad=max(high-low,1e-6)*.08
        axes[0].plot([low-pad,high+pad],[low-pad,high+pad],'--',c='gray',label='Equal scores')
        axes[0].set(xlabel='V9 signed atom attribution (all 417 latents)',ylabel='KernelSHAP signed atom attribution',
                    title=f"Pearson {metrics['pearson_r']:.3f}; Spearman {metrics['spearman_rho']:.3f}")
        axes[0].legend()
        ids=all_atoms.atom_id.to_numpy()
        axes[1].bar(ids-.2,x,.4,label='V9 joint-independent / 2000 refs')
        axes[1].bar(ids+.2,y,.4,label='KernelSHAP fixed-target / 500 refs')
        axes[1].axhline(0,color='gray',linewidth=.7)
        axes[1].set_xticks(ids);axes[1].set_xticklabels([f'{s}{i}' for i,s in enumerate(symbols)],rotation=90,fontsize=7)
        axes[1].set(xlabel='Atom symbol and zero-based ID',ylabel='Signed contribution (raw margin)')
        axes[1].legend(fontsize=8)
        fig.suptitle(pair+' — existing runs with different reference designs')
        fig.tight_layout()
        for suffix in ['png','pdf']:fig.savefig(args.outdir/f'{pair}_all_atom_comparison.{suffix}',dpi=200,bbox_inches='tight')
        plt.close(fig)
        provenance.append(dict(pair=pair,v9_source=str(args.v9_dir),kernel_source=str(kd),
            smiles=vp['smiles'],kinase=vp['kinase'],same_atom_order=True,
            v9_field='signed_present_bit_allocation',kernel_field='activebit_kernel_shap_signed'))
    metrics=pd.DataFrame(summaries);metrics.to_csv(args.outdir/'atom_signed_metrics_all_selections.csv',index=False)
    metrics[metrics.selection.eq('all')].to_csv(args.outdir/'atom_signed_metrics_all_latents.csv',index=False)
    pd.concat(aligned,ignore_index=True).to_csv(args.outdir/'atom_signed_aligned_values.csv',index=False)
    (args.outdir/'comparison_config.json').write_text(json.dumps(dict(
        comparisons=provenance,zero_tolerance=ZERO_TOL,
        interpretation='Descriptive comparison only: different backgrounds, modes and attribution methods; no independent-atom inference.',
        primary_selection='all',kernel_l1_reg='num_features(200)',
        note='Both atom scores allocate target-present bits only; absent-bit effects are excluded.'),indent=2))
    print(metrics[metrics.selection.eq('all')].to_string(index=False))


if __name__=='__main__':
    main()
