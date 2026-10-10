#!/usr/bin/env python
# coding: utf-8
# =====================================================================
# Band 10-50 supplement: molecule-level mechanism analysis
#   A  Radius composition of per-molecule top-k SHAP on-bits using bitInfo
#   B  Deletion curves: delete top-k Morgan on-bits versus k random on-bits
#      Apply the same deletions to the same-fold LSAE model to compare bit importance
#   C  Decompose positive Morgan SHAP by bit prevalence for VSDS false positives, true positives, and low-score groups
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
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

D  = _kp('dataset/')
FE = _kp('feature_embedding/')
BAND = _kp('dataset/data_split_from_merged_true_fold/band_0.10_0.50/')
M_TPL = _kp('model/Morgan_raw/KinSPEC_MCA_training_band_10_50_fold{k}_rawmorgan_model_v4.json')
L_TPL = _kp('model/KinSPEC/KinSPEC_MCA_training_band_10_50_fold{k}_model_v4.json')
OUT = _kp('others/Morgan_ablation/shap_radius_analysis/band_10_50_v2')
NB, KEEP_TOP, N_GROUP, SEED = 4096, 417, 2000, 42
FN_M = [str(i) for i in range(NB + 200)]; FN_L = [str(i) for i in range(KEEP_TOP + 200)]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parse = lambda s: np.frombuffer(str(s).strip().encode(), dtype=np.uint8) - 48
TOPK_A = [1, 3, 5, 10, 20]; TOPK_B = [5, 10, 20]

class SAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4096, 1024); self.decoder = nn.Linear(1024, 4096)
sae = SAE().to(DEV)
sae.load_state_dict(torch.load(FE+"mol/LSAE_chembl_1900k_full_dataset_v4_model.pt",
                               map_location=DEV, weights_only=True)); sae.eval()
keep_idx = torch.load(FE+"mol/LSAE_chembl_1900k_full_dataset_v4_topk_of1024d_top99.pt",
                      map_location=DEV, weights_only=True).long()[:KEEP_TOP]
@torch.no_grad()
def encode(fp):
    out = np.zeros((fp.shape[0], KEEP_TOP), dtype=np.float32)
    for s in range(0, fp.shape[0], 8192):
        out[s:s+8192] = sae.encoder(torch.from_numpy(
            fp[s:s+8192].astype(np.float32)).to(DEV))[:, keep_idx].cpu().numpy()
    return out
def score(tag, b, fp, kl):
    X, fn = ((np.concatenate([fp.astype(np.float32), kl], 1), FN_M) if tag == "morgan"
             else (np.concatenate([encode(fp), kl], 1), FN_L))
    return b.predict(xgb.DMatrix(X, feature_names=fn))

ks = pd.read_csv(FE+"Kinase_test_MCA/MCA_klifs_482_200D_features.csv", skiprows=1, header=None).values
kf = pd.read_csv(FE+"Kinase_test_MCA/human_482_typical_kinase_1955bit.csv")
KD = dict(zip(kf.KLIFS.astype(str).str.strip().values, ks))
klifs_of = lambda col: np.asarray([KD.get(str(k).strip(), np.zeros(ks.shape[1])) for k in col],
                                  dtype=np.float32)

v = pd.read_csv(D + 'VSDS_truedecoy_integrated_data_with_protein_seq.csv')
fp_v = np.stack([parse(s) for s in v.Morgan]); y = v.molecule_label.values.astype(np.int8)
kl_v = klifs_of(v.KLIFS); di, ai = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
prev = fp_v[y == 0].mean(0)
EDGES = [("rare<1%", 0, .01), ("uncommon1-5%", .01, .05), ("common5-20%", .05, .20), ("promiscuous>20%", .20, 1.01)]
stratum = np.full(NB, -1)
for i, (_, lo, hi) in enumerate(EDGES): stratum[(prev >= lo) & (prev < hi)] = i

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NB)
A_rows, B_rows, C_rows = [], [], []
for k in range(1, 6):
    print(f"fold {k} ...")
    rng = np.random.default_rng(SEED + k)
    B = {"morgan": xgb.Booster(), "lsae": xgb.Booster()}
    B["morgan"].load_model(M_TPL.format(k=k)); B["lsae"].load_model(L_TPL.format(k=k))
    tr = pd.read_csv(BAND + f"train_valid_split/train_valid_{k}/train.csv",
                     usecols=["Ligand", "Morgan", "KLIFS", "classification_label"])
    tr = tr[tr.classification_label == 1].drop_duplicates(subset="Morgan").reset_index(drop=True)

    # Per-molecule radius: retain molecules whose fingerprints can be reproduced exactly from SMILES
    rmaps, keep, mism = [], [], 0
    for i, (smi, mo) in enumerate(zip(tr.Ligand, tr.Morgan)):
        m = Chem.MolFromSmiles(str(smi))
        if m is None: mism += 1; continue
        ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
        f = gen.GetFingerprint(m, additionalOutput=ao)
        if set(f.GetOnBits()) != set(np.flatnonzero(parse(mo)).tolist()): mism += 1; continue
        rmaps.append({b: min(r for _, r in e) for b, e in ao.GetBitInfoMap().items()}); keep.append(i)
    tr = tr.iloc[keep].reset_index(drop=True)
    fp_t = np.stack([parse(s) for s in tr.Morgan]); kl_t = klifs_of(tr.KLIFS)
    Sh = B["morgan"].predict(xgb.DMatrix(np.concatenate([fp_t.astype(np.float32), kl_t], 1),
                                         feature_names=FN_M), pred_contribs=True)[:, :NB]

    # ---- A ----
    ons = [np.flatnonzero(fp_t[i]) for i in range(len(tr))]
    tops = [ons[i][np.argsort(-Sh[i, ons[i]])] for i in range(len(tr))]
    base = np.array([rmaps[i][b] for i in range(len(tr)) for b in ons[i]])
    bp = {r: (base == r).mean() for r in [0, 1, 2]}
    for kk in TOPK_A:
        t = np.array([rmaps[i][b] for i in range(len(tr)) for b in tops[i][:kk]])
        for r in [0, 1, 2]:
            A_rows.append({"fold": k, "top_k": kk, "radius": r, "observed": (t == r).mean(),
                           "baseline": bp[r], "enrichment": (t == r).mean() / bp[r]})
    A_rows.append({"fold": k, "top_k": 0, "radius": -1, "observed": mism, "baseline": len(tr), "enrichment": np.nan})

    # ---- B ----
    full = {t: score(t, B[t], fp_t, kl_t) for t in B}
    zero = {t: score(t, B[t], np.zeros_like(fp_t), kl_t) for t in B}
    for t in B:
        B_rows.append({"fold": k, "model": t, "k": 0, "condition": "full", "score": float(full[t].mean())})
        B_rows.append({"fold": k, "model": t, "k": 0, "condition": "all_zero", "score": float(zero[t].mean())})
    for kk in TOPK_B:
        dt, dr = fp_t.copy(), fp_t.copy()
        for i in range(len(tr)):
            n = min(kk, len(ons[i]))
            dt[i, tops[i][:n]] = 0; dr[i, rng.choice(ons[i], n, replace=False)] = 0
        for t in B:
            for c, mat in [("delete_morgan_topk", dt), ("delete_random_k", dr)]:
                B_rows.append({"fold": k, "model": t, "k": kk, "condition": c,
                               "score": float(score(t, B[t], mat, kl_t).mean())})

    # ---- C ----
    Sv = score("morgan", B["morgan"], fp_v, kl_v)
    ds = di[np.argsort(-Sv[di])]; as_ = ai[np.argsort(-Sv[ai])]; na = len(ai) // 3
    for g, idx in {"decoy_high (FP)": ds[:N_GROUP], "decoy_low": ds[-N_GROUP:],
                   "active_high (TP)": as_[:na], "active_low (FN)": as_[-na:]}.items():
        f0 = fp_v[idx].astype(np.float32)
        S = B["morgan"].predict(xgb.DMatrix(np.concatenate([f0, kl_v[idx]], 1), feature_names=FN_M),
                                pred_contribs=True)[:, :NB]
        on = f0 == 1; pos = S * (S > 0) * on; tot = np.maximum(pos.sum(1), 1e-9)
        row = {"fold": k, "group": g}
        for i, (nm, _, _) in enumerate(EDGES):
            row[nm] = float((pos[:, stratum == i].sum(1) / tot).mean())
        C_rows.append(row)

A = pd.DataFrame(A_rows); Bd = pd.DataFrame(B_rows); C = pd.DataFrame(C_rows)
A.to_csv(os.path.join(OUT, "mech_A_topk_radius.csv"), index=False)
Bd.to_csv(os.path.join(OUT, "mech_B_deletion.csv"), index=False)
C.to_csv(os.path.join(OUT, "mech_C_shap_strata.csv"), index=False)

print("\n=== A: Molecules excluded because fingerprints could not be reproduced (fold: excluded/retained) ===")
print(A[A.radius == -1][["fold", "observed", "baseline"]].to_string(index=False))
Aa = A[A.radius >= 0]
print("\n=== A: top-k on-bit radius enrichment (5-fold mean ± sd) ===")
g = Aa.groupby(["top_k", "radius"]).enrichment.agg(["mean", "std"]).unstack("radius")
print(g.round(3).to_string())
print("baseline:", Aa[Aa.top_k == 1].groupby("radius").baseline.mean().round(4).to_dict())
print("\n=== A: observed % (5-fold mean) ===")
print((Aa.groupby(["top_k", "radius"])["observed"].mean().unstack() * 100).round(1).to_string())
print("\n=== B: Deletion curves (5-fold mean ± sd) ===")
print(Bd.groupby(["model", "condition", "k"]).score.agg(["mean", "std"]).round(4).to_string())
print("\n=== C: Positive Morgan SHAP decomposed by rarity (5-fold mean %) ===")
print((C.groupby("group")[[e[0] for e in EDGES]].mean() * 100).round(1).to_string())
