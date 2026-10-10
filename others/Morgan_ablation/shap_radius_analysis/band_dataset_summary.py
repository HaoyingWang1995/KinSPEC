#!/usr/bin/env python
# coding: utf-8
# =====================================================================
# Dataset statistics for band 10-50
#   Band 10-50 size and similarity range, training size in each fold, VSDS-VD composition,
#   and the number of unique training actives used for radius analysis
# =====================================================================
from pathlib import Path
import sys

_start = Path(__file__).resolve().parent
_root = next((p for p in (_start, *_start.parents) if (p / 'kinspec_paths.py').is_file()), None)
if _root is None:
    raise RuntimeError('Cannot locate the KinSPEC project root containing kinspec_paths.py')
sys.path.insert(0, str(_root))
from kinspec_paths import project_path as _kp

import numpy as np, pandas as pd
BAND = _kp('dataset/data_split_from_merged_true_fold/band_0.10_0.50/')
VSDS = _kp('dataset/VSDS_truedecoy_integrated_data_with_protein_seq.csv')
print("band 0.10-0.50 complete set")
d = pd.read_csv(BAND + "train_with_smiles_protein_seq.csv",
                usecols=["Morgan", "KLIFS", "unified_bio_label", "max_sim_vsds"])
a = d[d.unified_bio_label == 1]
print(f"  compound-kinase pairs            {len(d):,}")
print(f"  active pairs                     {len(a):,}")
print(f"  unique active compounds          {a.Morgan.nunique():,}")
print(f"  kinase targets (active pairs)    {a.KLIFS.nunique():,}")
print(f"  kinase targets (all pairs)       {d.KLIFS.nunique():,}")
print(f"  max_sim_vsds  min {d.max_sim_vsds.min():.3f}   max {d.max_sim_vsds.max():.3f}   "
  f"median {d.max_sim_vsds.median():.3f}")

print("")
print("per-fold training partitions")
tot = set()
for k in range(1, 6):
    t = pd.read_csv(BAND + f"train_valid_split/train_valid_{k}/train.csv",
                    usecols=["Morgan", "classification_label"])
    ta = t[t.classification_label == 1]
    u = set(ta.Morgan.astype(str).str.strip()); tot |= u
    print(f"  fold {k}   rows {len(t):,}   active rows {len(ta):,}   unique active molecules {len(u):,}")
print(f"  union of unique active molecules across folds   {len(tot):,}")
print(f"  sum of per-fold unique active molecules   "
  f"{sum(pd.read_csv(BAND+f'train_valid_split/train_valid_{k}/train.csv', usecols=['Morgan','classification_label']).pipe(lambda x: x[x.classification_label==1]).Morgan.nunique() for k in range(1,6)):,}")

print("")
print("VSDS-VD test set")
v = pd.read_csv(VSDS, usecols=["molecule_label"])
y = v.molecule_label.values
print(f"  entries   {len(y):,}   actives {int((y == 1).sum()):,}   decoys {int((y == 0).sum()):,}")
print(f"  one third of actives   {int((y == 1).sum()) // 3:,}")

