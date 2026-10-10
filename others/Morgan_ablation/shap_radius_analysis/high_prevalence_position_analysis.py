#!/usr/bin/env python
# coding: utf-8
# =====================================================================
# Analysis of high-prevalence fingerprint positions, independent of the model
#   Stratify fingerprint positions by VSDS-VD decoy prevalence and calculate radius composition;
#   compare prevalence in actives and decoys for the 37 high-prevalence positions;
#   count high-prevalence positions per molecule and stratify by quintiles;
#   obtain within-class quartile cutoffs for the challenge subset
# =====================================================================
from pathlib import Path
import sys

_start = Path(__file__).resolve().parent
_root = next((p for p in (_start, *_start.parents) if (p / 'kinspec_paths.py').is_file()), None)
if _root is None:
    raise RuntimeError('Cannot locate the KinSPEC project root containing kinspec_paths.py')
sys.path.insert(0, str(_root))
from kinspec_paths import project_path as _kp

import os, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
VSDS = _kp('dataset/VSDS_truedecoy_integrated_data_with_protein_seq.csv')
BAND = _kp('dataset/data_split_from_merged_true_fold/band_0.10_0.50/train_with_smiles_protein_seq.csv')
OUT = _kp('others/Morgan_ablation/shap_radius_analysis/si_verification')
os.makedirs(OUT, exist_ok=True)
NB = 4096
EDGES = [("high-prevalence >20%", .20, 1.01), ("common 5-20%", .05, .20),
         ("uncommon 1-5%", .01, .05), ("rare <1%", 0.0, .01)]

v = pd.read_csv(VSDS,
                usecols=["SMILES", "Morgan", "molecule_label"])
fp = np.stack([np.frombuffer(str(s).strip().encode(), dtype=np.uint8) - 48 for s in v.Morgan])
y = v.molecule_label.values.astype(np.int8)
prev_d = fp[y == 0].mean(0)          # Decoy prevalence
prev_a = fp[y == 1].mean(0)          # Active-compound prevalence

# ---- Global bit-to-radius mapping ----
band = pd.read_csv(BAND, usecols=["SMILES"])
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NB)
votes = [{} for _ in range(NB)]
n_scanned = 0
for s in pd.concat([v.SMILES, band.SMILES]).drop_duplicates():
    m = Chem.MolFromSmiles(str(s))
    if m is None: continue
    n_scanned += 1
    ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
    gen.GetFingerprint(m, additionalOutput=ao)
    for b, e in ao.GetBitInfoMap().items():
        r = min(x for _, x in e); votes[b][r] = votes[b].get(r, 0) + 1
grad = np.array([max(d, key=d.get) if d else -1 for d in votes])
print(f"global bit-to-radius map   molecules scanned {n_scanned:,}   "
  f"positions mapped {int((grad >= 0).sum()):,}/4096")
print("")
print("prevalence strata and radius composition")
print(f"{'stratum':<22}{'positions':>10}{'radius=0':>11}{'radius=1':>11}{'radius=2':>11}")
rows = []
for nm, lo, hi in EDGES:
    k = (prev_d >= lo) & (prev_d < hi); g = grad[k]; g = g[g >= 0]
    print(f"{nm:<22}{int(k.sum()):>10}" + "".join(f"{(g==r).mean()*100:>10.1f}%" for r in [0,1,2]))
    rows.append({"stratum": nm, "positions": int(k.sum()),
                 **{f"radius_{r}_pct": float((g == r).mean() * 100) for r in [0, 1, 2]}})
print(f"{'total':<22}{int(sum(r['positions'] for r in rows)):>10}")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "position_prevalence_strata.csv"), index=False)

print("")
print("radius-wise view of all 4096 positions")
for r in [0, 1, 2]:
    k = grad == r
    print(f"  radius={r}   positions {int(k.sum()):>5} ({k.sum()/ (grad>=0).sum()*100:>4.1f}%)   "
      f"decoy prevalence  median {np.median(prev_d[k])*100:>5.2f}%   mean {prev_d[k].mean()*100:>5.2f}%")

PRO = np.flatnonzero(prev_d >= 0.20)
diff = prev_a[PRO] - prev_d[PRO]
print("")
print(f"class prevalence difference for the {len(PRO)} high-prevalence positions")
print(f"  prevalence in {int((y==1).sum()):,} actives   median {np.median(prev_a[PRO])*100:.1f}%")
print(f"  prevalence in {int((y==0).sum()):,} decoys    median {np.median(prev_d[PRO])*100:.1f}%")
print(f"  difference (actives minus decoys)   median {np.median(diff)*100:+.1f} percentage points"
  f"   range {diff.min()*100:+.1f} to {diff.max()*100:+.1f}")
print(f"  positions with a higher prevalence in actives   {int((diff > 0).sum())}/{len(PRO)}")
pd.DataFrame({"bit": PRO, "radius": grad[PRO], "prev_actives": prev_a[PRO],
              "prev_decoys": prev_d[PRO], "difference": diff}
             ).to_csv(os.path.join(OUT, "high_prevalence_positions.csv"), index=False)

# ---- Per-molecule counts and quintile strata ----
cnt = fp[:, PRO].sum(1)
print("")
print("high-prevalence-position count per molecule")
print(f"  range {cnt.min()} to {cnt.max()}   median {np.median(cnt):.0f}   mean {cnt.mean():.2f}")
q = np.quantile(cnt, [0, .2, .4, .6, .8, 1.0])
bins = np.unique(q)
st = np.clip(np.digitize(cnt, bins[1:-1], right=True), 0, len(bins) - 2)
print(f"  quintiles of the count distribution   {q}")
print(f"  boundary values assigned to the lower stratum, cut points {bins[1:-1]}")
print("")
print(f"{'stratum':>8}{'count range':>14}{'entries':>10}{'actives':>9}{'decoys':>9}{'both classes':>14}")
srows = []
for s_ in np.unique(st):
    k = st == s_; c = cnt[k]
    print(f"{s_+1:>8}{f'{c.min()} to {c.max()}':>14}{int(k.sum()):>10}"
      f"{int(y[k].sum()):>9}{int((y[k]==0).sum()):>9}"
      f"{'yes' if y[k].min()!=y[k].max() else 'no':>14}")
    srows.append({"stratum": int(s_+1), "count_min": int(c.min()), "count_max": int(c.max()),
                  "entries": int(k.sum()), "actives": int(y[k].sum()),
                  "decoys": int((y[k] == 0).sum())})
print(f"{'total':>8}{'':>14}{int(sum(r['entries'] for r in srows)):>10}"
  f"{int(sum(r['actives'] for r in srows)):>9}{int(sum(r['decoys'] for r in srows)):>9}")
pd.DataFrame(srows).to_csv(os.path.join(OUT, "count_stratification.csv"), index=False)

# ---- Within-class quartiles for the challenge subset ----
di, ai = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
hd = di[cnt[di] >= np.quantile(cnt[di], .75)]; ha = ai[cnt[ai] <= np.quantile(cnt[ai], .25)]
print("")
print("challenge subset, quartiles taken separately within each class")
print(f"  decoy 75th percentile of count {np.quantile(cnt[di], .75):.0f}   "
  f"decoys at or above it   {len(hd):,}")
print(f"  active 25th percentile of count {np.quantile(cnt[ai], .25):.0f}   "
  f"actives at or below it  {len(ha):,}")

print(f"\nCSV written to {OUT}")
