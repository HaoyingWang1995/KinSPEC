#!/usr/bin/env python
# Global bit-to-radius mapping for band 10-50 and VSDS; radius composition by VSDS decoy prevalence

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

import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
v = pd.read_csv(_kp('dataset/VSDS_truedecoy_integrated_data_with_protein_seq.csv'),
                usecols=["SMILES", "Morgan", "molecule_label"])
b = pd.read_csv(_kp('dataset/data_split_from_merged_true_fold/band_0.10_0.50/train_with_smiles_protein_seq.csv'), usecols=["SMILES"])
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
votes = [{} for _ in range(4096)]
for s in pd.concat([v.SMILES, b.SMILES]).drop_duplicates():
    m = Chem.MolFromSmiles(str(s))
    if m is None: continue
    ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
    gen.GetFingerprint(m, additionalOutput=ao)
    for bb, e in ao.GetBitInfoMap().items():
        r = min(x for _, x in e); votes[bb][r] = votes[bb].get(r, 0) + 1
g = np.array([max(d, key=d.get) if d else -1 for d in votes])
fp = np.stack([np.frombuffer(str(s).strip().encode(), dtype=np.uint8) - 48 for s in v.Morgan])
prev = fp[v.molecule_label.values == 0].mean(0)
for nm, lo, hi in [("rare<1%", 0, .01), ("uncommon1-5%", .01, .05),
                   ("common5-20%", .05, .2), ("promiscuous>20%", .2, 1.01)]:
    k = (prev >= lo) & (prev < hi); gg = g[k]; gg = gg[gg >= 0]
    print(f"{nm:<16} n={int(k.sum()):>5}  " + "  ".join(f"r{r}={(gg==r).mean():.1%}" for r in [0, 1, 2]))
for r in [0, 1, 2]:
    k = g == r
    print(f"radius={r}: {int(k.sum())} positions  decoy prev median={np.median(prev[k]):.2%} mean={prev[k].mean():.2%}")
