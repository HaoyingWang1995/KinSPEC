#!/usr/bin/env python
# coding: utf-8
# =====================================================================
# Aggregation conventions and robustness of retained-score percentages
#   Primary: mean perturbed score / mean unperturbed score within each fold, then average folds.
#   Also calculate the mean per-molecule ratio and compare the two conventions.
# =====================================================================
from pathlib import Path
import sys

_start = Path(__file__).resolve().parent
_root = next((p for p in (_start, *_start.parents) if (p / 'kinspec_paths.py').is_file()), None)
if _root is None:
    raise RuntimeError('Cannot locate the KinSPEC project root containing kinspec_paths.py')
sys.path.insert(0, str(_root))
from kinspec_paths import project_path as _kp

import os, numpy as np, pandas as pd, xgboost as xgb, torch, torch.nn as nn
VSDS = _kp('dataset/VSDS_truedecoy_integrated_data_with_protein_seq.csv')
FE = _kp('feature_embedding/')
M_TPL = _kp('model/Morgan_raw/KinSPEC_MCA_training_band_10_50_fold{k}_rawmorgan_model_v4.json')
L_TPL = _kp('model/KinSPEC/KinSPEC_MCA_training_band_10_50_fold{k}_model_v4.json')
OUT = _kp('others/Morgan_ablation/shap_radius_analysis/si_verification')
os.makedirs(OUT, exist_ok=True)
KEEP, NB, N_DECOY = 417, 4096, 2000
FN_M = [str(i) for i in range(NB + 200)]; FN_L = [str(i) for i in range(KEEP + 200)]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAE(nn.Module):
    def __init__(s):
        super().__init__(); s.encoder = nn.Linear(4096, 1024); s.decoder = nn.Linear(1024, 4096)
sae = SAE().to(DEV)
sae.load_state_dict(torch.load(FE + "mol/LSAE_chembl_1900k_full_dataset_v4_model.pt",
                               map_location=DEV, weights_only=True)); sae.eval()
ki = torch.load(FE + "mol/LSAE_chembl_1900k_full_dataset_v4_topk_of1024d_top99.pt",
                map_location=DEV, weights_only=True).long()[:KEEP]
@torch.no_grad()
def encode(x):
    o = np.zeros((x.shape[0], KEEP), dtype=np.float32)
    for s in range(0, x.shape[0], 8192):
        o[s:s+8192] = sae.encoder(torch.from_numpy(
            x[s:s+8192].astype(np.float32)).to(DEV))[:, ki].cpu().numpy()
    return o
def score(tag, b, fp, kl):
    X, fn = ((np.concatenate([fp.astype(np.float32), kl], 1), FN_M) if tag == "morgan"
             else (np.concatenate([encode(fp), kl], 1), FN_L))
    return b.predict(xgb.DMatrix(X, feature_names=fn))

v = pd.read_csv(VSDS)
fp = np.stack([np.frombuffer(str(s).strip().encode(), dtype=np.uint8) - 48 for s in v.Morgan])
y = v.molecule_label.values.astype(np.int8)
ks = pd.read_csv(FE + "Kinase_test_MCA/MCA_klifs_482_200D_features.csv",
                 skiprows=1, header=None).values
kf = pd.read_csv(FE + "Kinase_test_MCA/human_482_typical_kinase_1955bit.csv")
kd = dict(zip(kf.KLIFS.astype(str).str.strip().values, ks))
kl = np.asarray([kd.get(str(k).strip(), np.zeros(ks.shape[1])) for k in v.KLIFS],
                dtype=np.float32)
PRO = np.flatnonzero(fp[y == 0].mean(0) >= 0.20)
m_keep = np.zeros(NB, dtype=np.uint8); m_keep[PRO] = 1
m_del  = np.ones(NB, dtype=np.uint8);  m_del[PRO] = 0
di, ai = np.flatnonzero(y == 0), np.flatnonzero(y == 1); na = len(ai) // 3

rows = []
for k in range(1, 6):
    B = {"morgan": xgb.Booster(), "lsae": xgb.Booster()}
    B["morgan"].load_model(M_TPL.format(k=k)); B["lsae"].load_model(L_TPL.format(k=k))
    S = {t: score(t, B[t], fp, kl) for t in B}
    for t in B:
        ds = di[np.argsort(-S[t][di])]; as_ = ai[np.argsort(-S[t][ai])]
        for g, idx in {"decoy_high (FP)": ds[:N_DECOY], "active_high (TP)": as_[:na]}.items():
            f0, k0 = fp[idx], kl[idx]
            full = S[t][idx]
            keep = score(t, B[t], f0 * m_keep[None], k0)
            dele = score(t, B[t], f0 * m_del[None],  k0)
            rows.append({"fold": k, "model": t, "group": g,
                "keep_ratio_of_means": keep.mean() / full.mean(),
                "keep_mean_of_ratios": float((keep / np.maximum(full, 1e-9)).mean()),
                "delete_ratio_of_means": dele.mean() / full.mean(),
                "delete_mean_of_ratios": float((dele / np.maximum(full, 1e-9)).mean())})
    print(f"  fold {k} done")
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "ratio_convention_per_fold.csv"), index=False)
print("retained-score percentage under the two aggregation conventions")
print("  primary   ratio of the group mean perturbed score to the group mean unperturbed score")
print("  alternative   mean of the per-molecule ratios")
print("  values are the mean across the five folds ± sample standard deviation, in percent")
print(f"{'model':<8}{'group':<19}{'keep primary':>16}{'keep alt':>14}{'diff':>7}"
  f"{'delete primary':>17}{'delete alt':>14}{'diff':>7}")
mx = 0.0
for (t, g), x in d.groupby(["model", "group"]):
    a1, a2 = x.keep_ratio_of_means * 100, x.keep_mean_of_ratios * 100
    b1, b2 = x.delete_ratio_of_means * 100, x.delete_mean_of_ratios * 100
    da, db = abs(a1.mean() - a2.mean()), abs(b1.mean() - b2.mean())
    mx = max(mx, da, db)
    print(f"{t:<8}{g:<19}{a1.mean():>9.1f}±{a1.std():<5.1f}{a2.mean():>8.1f}±{a2.std():<5.1f}"
      f"{da:>7.2f}{b1.mean():>10.1f}±{b1.std():<5.1f}{b2.mean():>8.1f}±{b2.std():<5.1f}{db:>7.2f}")
print("")
print(f"largest absolute difference between the two conventions   {mx:.2f} percentage points")

print(f"\nCSV written to {OUT}")
