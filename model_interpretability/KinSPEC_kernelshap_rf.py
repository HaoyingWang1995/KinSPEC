#!/usr/bin/env python3
"""V4 molecular KernelSHAP: exact binary uint8 storage and batched predictions.

Fixed target; 500 molecular references, 2048 coalition samples, 200 selected
features by default. uint8 preserves every binary hybrid value exactly.
"""
import argparse
import json
import resource
import time
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import torch
import xgboost as xgb
import KinSPEC_interpretability_v9 as v9
import KinSPEC_kernelshap_control as legacy


def make_predictor(sae, keep_idx, device, booster, fixed_mca, batch_size):
    def predict(bits):
        outputs = []
        for start in range(0, len(bits), batch_size):
            batch = np.asarray(bits[start:start + batch_size], dtype=np.float32)
            latent = legacy.encode_molecules(batch, sae, keep_idx, device, batch_size)
            protein = np.repeat(fixed_mca[None, :], len(latent), axis=0)
            model_input = np.hstack([latent, protein]).astype(np.float32)
            outputs.append(v9.raw_predict(booster, model_input))
        return np.concatenate(outputs)
    return predict


def load_models():
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sae = v9.SparseLinearAE().to(device)
    sae.load_state_dict(torch.load(v9.SAE_MODEL, map_location=device, weights_only=True))
    sae.eval()
    idx = torch.load(v9.SAE_INDICES, map_location=device, weights_only=True).long()[:v9.MOLECULE_DIMS]
    booster = xgb.Booster(); booster.load_model(v9.XGB_MODEL)
    assert len(idx) == 417 and booster.num_features() == 617
    return sae, idx, device, booster


def run(args):
    started = time.perf_counter()
    out = args.outdir; out.mkdir(parents=True, exist_ok=True)
    pair = next(p for p in v9.PAIRS if p['pair'] == args.pair)
    config = dict(pair=pair, background_size=args.references, nsamples=args.nsamples,
                  l1_reg=f'num_features({args.selected_features})', seed=args.seed,
                  hybrid_dtype='uint8', prediction_batch_size=args.batch_size,
                  background_mode='fixed_target', molecule_dims=417, protein_dims=200,
                  sae_model=str(v9.SAE_MODEL), sae_indices=str(v9.SAE_INDICES), xgb_model=str(v9.XGB_MODEL),
                  shap_version=shap.__version__)
    path = out/'run_config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Existing output has a different configuration')
    path.write_text(json.dumps(config, indent=2))
    sae, idx, device, booster = load_models()
    bits, bit_info, mol = legacy.morgan_from_smiles(pair['smiles'])
    meta = pd.read_csv(v9.KINASE_TABLE)
    fixed_mca = pd.read_csv(v9.KINASE_MCA).to_numpy(np.float32)[v9.find_target_protein(pair['kinase'], meta)]
    frame, background = legacy.reservoir_background(v9.MOLECULE_BACKGROUND, args.references, args.seed)
    assert np.isin(background, [0, 1]).all()
    background = background.astype(np.uint8)
    frame.to_csv(out/'kernelshap_molecule_background_used.csv', index=False)
    predict = make_predictor(sae, idx, device, booster, fixed_mca, args.batch_size)
    prediction = float(predict(bits[None])[0])
    print(f"{args.pair}: margin={prediction:.9g}; background={len(background)}; nsamples={args.nsamples}; device={device}", flush=True)
    np.random.seed(args.seed)
    kernel_started = time.perf_counter()
    explainer = shap.KernelExplainer(predict, background, link='identity')
    values = np.asarray(explainer.shap_values(bits[None], nsamples=args.nsamples, l1_reg=config['l1_reg'])).reshape(-1)
    kernel_seconds = time.perf_counter()-kernel_started
    expected = legacy.scalar_expected_value(explainer.expected_value)
    residual = prediction-expected-values.sum()
    np.testing.assert_allclose(expected+values.sum(), prediction, rtol=1e-6, atol=1e-6)
    result = pd.DataFrame({'bit':np.arange(4096), 'target_bit':bits.astype(np.uint8), 'kernel_shap_signed':values})
    result.to_csv(out/'molecule_4096bit_kernelshap_signed.csv', index=False)
    legacy.write_active_bit_atom_heatmap(result, bit_info, mol, pair['kinase'], out)
    total_abs = float(np.abs(values).sum()); active = bits.astype(bool)
    active_abs = float(np.abs(values[active]).sum())
    pd.DataFrame([dict(prediction_raw_margin=prediction, predicted_probability=1/(1+np.exp(-prediction)),
        expected_value=expected, sum_shap=float(values.sum()), additivity_residual=residual,
        background_size=len(background), nsamples=args.nsamples, l1_reg=config['l1_reg'],
        active_bit_signed_sum=float(values[active].sum()), active_bit_abs_sum=active_abs,
        all_bit_abs_sum=total_abs, active_bit_abs_coverage=active_abs/total_abs if total_abs else np.nan)]
    ).to_csv(out/'molecule_kernelshap_diagnostics.csv', index=False)
    runtime = dict(total_seconds=time.perf_counter()-started, kernel_seconds=kernel_seconds,
                   peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                   synthetic_matrix_gib=explainer.synth_data.nbytes/1024**3,
                   synthetic_matrix_dtype=str(explainer.synth_data.dtype),
                   background_size=len(background), nsamples=args.nsamples)
    (out/'runtime_summary.json').write_text(json.dumps(runtime, indent=2))
    print(json.dumps(runtime, indent=2), flush=True)
    print(f'DONE: {out}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair', choices=[x['pair'] for x in v9.PAIRS], required=True)
    p.add_argument('--references', type=int, default=500)
    p.add_argument('--nsamples', type=int, default=2048)
    p.add_argument('--selected-features', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--seed', type=int, default=20260923)
    p.add_argument('--outdir', type=Path, required=True)
    args = p.parse_args()
    if min(args.references, args.nsamples, args.selected_features, args.batch_size) < 1:
        p.error('Counts and batch size must be positive')
    run(args)
