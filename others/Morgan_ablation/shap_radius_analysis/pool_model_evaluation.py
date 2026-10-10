#!/usr/bin/env python
# coding: utf-8
# =====================================================================
# Evaluate training-pool models on VSDS-VD
#   Fit each representation's Optuna-selected configuration on the complete training pool;
#   calculate AUROC, BEDROC, EF, and 95% percentile bootstrap intervals (1000 replicates);
#   calculate a paired bootstrap for the AUROC difference (2000 replicates)
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
from sklearn.metrics import roc_auc_score
VSDS = _kp('dataset/VSDS_truedecoy_integrated_data_with_protein_seq.csv')
FE = _kp('feature_embedding/')
M_POOL = _kp('model/Morgan_raw/xgboost_SPECTRA_ready_to_use_with_VSDS_482_kinases_4096bit_morgan_2.json')
L_POOL = _kp('model/KinSPEC/KinSPEC_full_training_pool_MCA_200D_v4.json')
OUT = _kp('others/Morgan_ablation/shap_radius_analysis/si_verification')
os.makedirs(OUT, exist_ok=True)
KEEP, NB, ALPHA = 417, 4096, 80.5
N_BOOT_CI, N_BOOT_DIFF, SEED = 1000, 2000, 42
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAE(nn.Module):
    def __init__(s):
        super().__init__(); s.encoder = nn.Linear(4096, 1024); s.decoder = nn.Linear(1024, 4096)
sae = SAE().to(DEV)
sae.load_state_dict(torch.load(FE + "mol/LSAE_chembl_1900k_full_dataset_v4_model.pt",
                               map_location=DEV, weights_only=True)); sae.eval()
ki = torch.load(FE + "mol/LSAE_chembl_1900k_full_dataset_v4_topk_of1024d_top99.pt",
                map_location=DEV, weights_only=True).long()[:KEEP]

v = pd.read_csv(VSDS)
fp = np.stack([np.frombuffer(str(s).strip().encode(), dtype=np.uint8) - 48
               for s in v.Morgan]).astype(np.float32)
y = v.molecule_label.values.astype(np.int8)
ks = pd.read_csv(FE + "Kinase_test_MCA/MCA_klifs_482_200D_features.csv",
                 skiprows=1, header=None).values
kf = pd.read_csv(FE + "Kinase_test_MCA/human_482_typical_kinase_1955bit.csv")
kd = dict(zip(kf.KLIFS.astype(str).str.strip().values, ks))
kl = np.asarray([kd.get(str(k).strip(), np.zeros(ks.shape[1])) for k in v.KLIFS],
                dtype=np.float32)

@torch.no_grad()
def encode(x):
    o = np.zeros((x.shape[0], KEEP), dtype=np.float32)
    for s in range(0, x.shape[0], 8192):
        o[s:s+8192] = sae.encoder(torch.from_numpy(
            x[s:s+8192].astype(np.float32)).to(DEV))[:, ki].cpu().numpy()
    return o

def bedroc(yt, ys):
    N = len(yt); na = int(yt.sum())
    r = np.flatnonzero(yt[np.argsort(ys)[::-1]] == 1) + 1
    C = (1.0 - np.exp(-ALPHA)) / ALPHA
    rie  = np.mean(np.exp(-ALPHA * (r - 0.5) / N)) / C
    rmax = np.mean(np.exp(-ALPHA * (np.arange(1, na+1) - 0.5) / N)) / C
    rmin = np.mean(np.exp(-ALPHA * (np.arange(N-na+1, N+1) - 0.5) / N)) / C
    return (rie - rmin) / (rmax - rmin)
def ef(yt, ys, p):
    n = max(1, int(len(yt) * p / 100))
    return yt[np.argsort(ys)[::-1][:n]].mean() / yt.mean()

M = xgb.Booster(); M.load_model(M_POOL)
L = xgb.Booster(); L.load_model(L_POOL)
S = {"raw Morgan": M.predict(xgb.DMatrix(np.concatenate([fp, kl], 1),
                             feature_names=[str(i) for i in range(NB + 200)])),
     "LSAE":       L.predict(xgb.DMatrix(np.concatenate([encode(fp), kl], 1),
                             feature_names=[str(i) for i in range(KEEP + 200)]))}

METS = [("AUROC", roc_auc_score), ("BEDROC", bedroc)] + \
       [(f"EF{p:g}%", (lambda a, b, p=p: ef(a, b, p))) for p in [0.5, 1.0, 2.0, 5.0]]
rng = np.random.default_rng(SEED)
boot_idx = [rng.choice(len(y), len(y), replace=True) for _ in range(N_BOOT_CI)]
boot_idx = [b for b in boot_idx if y[b].min() != y[b].max()]

rows = []
print("training-pool models on VSDS-VD")
print(f"  raw Morgan : {os.path.basename(M_POOL)}")
print(f"  LSAE       : {os.path.basename(L_POOL)}")
print(f"  bootstrap  : {len(boot_idx)} valid replicates of {N_BOOT_CI} (seed {SEED})")
print(f"{'metric':<10}{'raw Morgan  [95% CI]':>36}{'LSAE  [95% CI]':>36}")
for nm, fn in METS:
    line = f"{nm:<10}"
    rec = {"metric": nm}
    for k in ["raw Morgan", "LSAE"]:
        val = fn(y, S[k])
        bs = np.array([fn(y[b], S[k][b]) for b in boot_idx])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rec[f"{k}_value"], rec[f"{k}_lo"], rec[f"{k}_hi"] = val, lo, hi
        line += f"{val:>18.4f} [{lo:.4f}, {hi:.4f}]"
    rows.append(rec); print(line)

rng2 = np.random.default_rng(7)
d = []
for _ in range(N_BOOT_DIFF):
    b = rng2.choice(len(y), len(y), replace=True)
    if y[b].min() != y[b].max():
        d.append(roc_auc_score(y[b], S["LSAE"][b]) - roc_auc_score(y[b], S["raw Morgan"][b]))
d = np.array(d)
print("")
print(f"paired bootstrap of the AUROC difference (LSAE - raw Morgan), "
   f"{len(d)} valid replicates of {N_BOOT_DIFF}")
print(f"  mean {d.mean():+.4f}   two-sided 95% percentile interval "
   f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]   "
   )

pd.DataFrame(rows).to_csv(os.path.join(OUT, "pool_model_metrics.csv"), index=False)
print(f"\nCSV written to {OUT}")
