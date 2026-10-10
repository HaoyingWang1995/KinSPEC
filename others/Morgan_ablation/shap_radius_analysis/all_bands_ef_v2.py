#!/usr/bin/env python
# coding: utf-8
# =====================================================================
# Panels A/B: raw Morgan (retuned model_2) versus LSAE across six similarity bands
# Early enrichment on VSDS-VD: 5 folds x 6 bands x 2 representations.
# =====================================================================

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
_kp = _ks_partial(_ks_project_path, script_dir='others/Morgan_ablation/shap_radius_analysis')

import os
import numpy as np, pandas as pd, xgboost as xgb, torch, torch.nn as nn
from scipy import stats
from sklearn.metrics import roc_auc_score

D  = _kp('dataset/')
FE = _kp('feature_embedding/')
M_TPL = _kp('model/Morgan_raw/KinSPEC_MCA_training_band_{b}_fold{k}_rawmorgan_model_v4.json')
L_TPL = _kp('model/KinSPEC/KinSPEC_MCA_training_band_{b}_fold{k}_model_v4.json')
OUT = _kp('others/Morgan_ablation/shap_radius_analysis/all_bands_v2')
os.makedirs(OUT, exist_ok=True)
BANDS = ["10_50", "20_60", "30_70", "40_80", "50_90", "60_100"]
KEEP_TOP = 417
FN_M = [str(i) for i in range(4096 + 200)]
FN_L = [str(i) for i in range(KEEP_TOP + 200)]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4096, 1024); self.decoder = nn.Linear(1024, 4096)
sae = SAE().to(DEV)
sae.load_state_dict(torch.load(FE+"mol/LSAE_chembl_1900k_full_dataset_v4_model.pt",
                               map_location=DEV, weights_only=True)); sae.eval()
keep_idx = torch.load(FE+"mol/LSAE_chembl_1900k_full_dataset_v4_topk_of1024d_top99.pt",
                      map_location=DEV, weights_only=True).long()[:KEEP_TOP]

print("[load] VSDS ...")
v = pd.read_csv(D + 'VSDS_truedecoy_integrated_data_with_protein_seq.csv')
fp = np.stack([np.frombuffer(str(s).strip().encode(), dtype=np.uint8) - 48
               for s in v.Morgan]).astype(np.float32)
y = v.molecule_label.values.astype(np.int8)
ks = pd.read_csv(FE+"Kinase_test_MCA/MCA_klifs_482_200D_features.csv", skiprows=1, header=None).values
kf = pd.read_csv(FE+"Kinase_test_MCA/human_482_typical_kinase_1955bit.csv")
KD = dict(zip(kf.KLIFS.astype(str).str.strip().values, ks))
kl = np.asarray([KD.get(str(k).strip(), np.zeros(ks.shape[1])) for k in v.KLIFS], dtype=np.float32)

@torch.no_grad()
def encode(x):
    o = np.zeros((x.shape[0], KEEP_TOP), dtype=np.float32)
    for s in range(0, x.shape[0], 8192):
        o[s:s+8192] = sae.encoder(torch.from_numpy(x[s:s+8192]).to(DEV))[:, keep_idx].cpu().numpy()
    return o
X_M = xgb.DMatrix(np.concatenate([fp, kl], 1), feature_names=FN_M)
X_L = xgb.DMatrix(np.concatenate([encode(fp), kl], 1), feature_names=FN_L)

BEDROC_ALPHA = 80.5
def bedroc(yt, ys):
    N = len(yt); na = int(yt.sum())
    ranks = np.flatnonzero(yt[np.argsort(ys)[::-1]] == 1) + 1
    C = (1.0 - np.exp(-BEDROC_ALPHA)) / BEDROC_ALPHA
    rie = np.mean(np.exp(-BEDROC_ALPHA * (ranks - 0.5) / N)) / C
    rmax = np.mean(np.exp(-BEDROC_ALPHA * (np.arange(1, na+1) - 0.5) / N)) / C
    rmin = np.mean(np.exp(-BEDROC_ALPHA * (np.arange(N-na+1, N+1) - 0.5) / N)) / C
    return (rie - rmin) / (rmax - rmin)
def ef(yt, ys, p):
    n = max(1, int(len(yt) * p / 100))
    return yt[np.argsort(ys)[::-1][:n]].mean() / yt.mean()

rows = []
for b in BANDS:
    for k in range(1, 6):
        for tag, tpl, dm in [("morgan", M_TPL, X_M), ("lsae", L_TPL, X_L)]:
            bo = xgb.Booster(); bo.load_model(tpl.format(b=b, k=k))
            s = bo.predict(dm)
            rows.append({"band": b, "fold": k, "model": tag,
                         "AUROC": roc_auc_score(y, s), "BEDROC": bedroc(y, s),
                         **{f"EF{p}%": ef(y, s, p) for p in [0.5, 1.0, 2.0, 5.0]}})
    print(f"  band {b} done")
raw = pd.DataFrame(rows); raw.to_csv(os.path.join(OUT, "per_band_fold_metrics.csv"), index=False)

METS = ["EF0.5%", "EF1.0%", "EF2.0%", "EF5.0%", "AUROC", "BEDROC"]
summ = []
for b in BANDS:
    for m in METS:
        w = raw[raw.band == b].pivot(index="fold", columns="model", values=m)
        d = w["lsae"] - w["morgan"]
        summ.append({"band": b, "metric": m,
                     "morgan_mean": w.morgan.mean(), "morgan_sd": w.morgan.std(),
                     "lsae_mean": w.lsae.mean(), "lsae_sd": w.lsae.std(),
                     "diff": d.mean(), "folds_lsae_higher": int((d > 0).sum()),
                     "paired_t_p": stats.ttest_rel(w.lsae, w.morgan).pvalue})
s = pd.DataFrame(summ); s.to_csv(os.path.join(OUT, "per_band_summary.csv"), index=False)

for m in METS:
    print(f"\n=== {m} ===")
    t = s[s.metric == m]
    print(f"{'band':<9}{'raw Morgan':>18}{'LSAE':>18}{'diff':>9}{'n/5':>6}{'p':>9}")
    for _, r in t.iterrows():
        print(f"{r.band:<9}{r.morgan_mean:>10.4f} ± {r.morgan_sd:<5.4f}"
              f"{r.lsae_mean:>10.4f} ± {r.lsae_sd:<5.4f}{r['diff']:>+9.4f}"
              f"{r.folds_lsae_higher:>4d}/5{r.paired_t_p:>9.3g}")
print(f"\n[done] -> {OUT}")
