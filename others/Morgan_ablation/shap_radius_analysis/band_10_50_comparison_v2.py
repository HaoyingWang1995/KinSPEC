#!/usr/bin/env python
# coding: utf-8
# =====================================================================
# Morgan versus LSAE in band 10-50 (training-compound similarity to VSDS <= 0.5)
# ---------------------------------------------------------------------
# Full-pool conclusions are affected by VSDS analogues in training; use a strict split here.
# Each fold uses its train.csv actives and the corresponding Morgan and LSAE models.
#
#   M   Position-level Morgan SHAP: radius-0 enrichment, decoy prevalence, and ON/OFF signs
#   E1  Retention/deletion experiments: 37 promiscuous positions, rare positions, and fold-specific giveaway20
#   E2  Spearman correlation of decoy scores with promiscuous-position counts; within-stratum AUROC
#   E3  Challenge-subset AUROC: high-count decoys versus low-count actives
#   E4  Top-2000 decoy overlap and cross-model ranks
#   NN  Nearest-neighbor Tanimoto similarity from false positives to same-target training actives
# Paired Morgan/LSAE comparisons across five folds: paired t-tests and sign consistency.
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

import os, json
import numpy as np, pandas as pd, xgboost as xgb, torch, torch.nn as nn
from scipy import stats
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

D  = _kp('dataset/')
FE = _kp('feature_embedding/')
BAND = _kp('dataset/data_split_from_merged_true_fold/band_0.10_0.50/')
M_TPL = _kp('model/Morgan_raw/KinSPEC_MCA_training_band_10_50_fold{k}_rawmorgan_model_v4.json')
L_TPL = _kp('model/KinSPEC/KinSPEC_MCA_training_band_10_50_fold{k}_model_v4.json')
OUT = _kp('others/Morgan_ablation/shap_radius_analysis/band_10_50_v2')
os.makedirs(OUT, exist_ok=True)
NB, KEEP_TOP, N_GROUP, SEED = 4096, 417, 2000, 42
FN_M = [str(i) for i in range(NB + 200)]
FN_L = [str(i) for i in range(KEEP_TOP + 200)]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parse = lambda s: np.frombuffer(str(s).strip().encode(), dtype=np.uint8) - 48

# ---------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------
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

def score(tag, booster, fp, kl):
    if tag == "morgan":
        X, fn = np.concatenate([fp.astype(np.float32), kl], 1), FN_M
    else:
        X, fn = np.concatenate([encode(fp), kl], 1), FN_L
    return booster.predict(xgb.DMatrix(X, feature_names=fn))

@torch.no_grad()
def nn_tanimoto(Q, R, chunk=4096):
    if len(R) == 0: return np.full(len(Q), np.nan, dtype=np.float32)
    Rt = torch.from_numpy(R.astype(np.float32)).to(DEV); rs = Rt.sum(1)
    out = np.empty(len(Q), dtype=np.float32)
    for s in range(0, len(Q), chunk):
        q = torch.from_numpy(Q[s:s+chunk].astype(np.float32)).to(DEV)
        inter = q @ Rt.T
        out[s:s+chunk] = (inter / (q.sum(1, keepdim=True) + rs[None] - inter).clamp_min(1)
                          ).max(1).values.cpu().numpy()
    return out

print("[load] VSDS ...")
v = pd.read_csv(D + 'VSDS_truedecoy_integrated_data_with_protein_seq.csv')
fp_v = np.stack([parse(s) for s in v.Morgan]); y = v.molecule_label.values.astype(np.int8)
vk = v.KLIFS.astype(str).str.strip().values
ks = pd.read_csv(FE+"Kinase_test_MCA/MCA_klifs_482_200D_features.csv", skiprows=1, header=None).values
kf = pd.read_csv(FE+"Kinase_test_MCA/human_482_typical_kinase_1955bit.csv")
KD = dict(zip(kf.KLIFS.astype(str).str.strip().values, ks))
klifs_of = lambda col: np.asarray([KD.get(str(k).strip(), np.zeros(ks.shape[1])) for k in col],
                                  dtype=np.float32)
kl_v = klifs_of(v.KLIFS)
di, ai = np.flatnonzero(y == 0), np.flatnonzero(y == 1)

prev = fp_v[y == 0].mean(0)
PROMISC = np.flatnonzero(prev >= 0.20); RARE = np.flatnonzero(prev < 0.01)
cnt = fp_v[:, PROMISC].sum(1)
print(f"       promiscuous={len(PROMISC)}  rare={len(RARE)}")

print("[radius] global bit -> radius map (VSDS + band train) ...")
band = pd.read_csv(BAND + "train_with_smiles_protein_seq.csv", usecols=["SMILES"])
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NB)
votes = [{} for _ in range(NB)]
for smi in pd.concat([v.SMILES, band.SMILES]).drop_duplicates():
    m = Chem.MolFromSmiles(str(smi))
    if m is None: continue
    ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
    gen.GetFingerprint(m, additionalOutput=ao)
    for b, e in ao.GetBitInfoMap().items():
        r = min(rr for _, rr in e); votes[b][r] = votes[b].get(r, 0) + 1
grad = np.array([max(d, key=d.get) if d else -1 for d in votes])
base_r0 = (grad[grad >= 0] == 0).mean()
del band

def ef(yy, ss, pct):
    n = max(1, int(len(yy) * pct / 100)); return yy[np.argsort(-ss)[:n]].mean() / yy.mean()

def strat_auroc(s):
    bins = np.unique(np.quantile(cnt, [0, .2, .4, .6, .8, 1.0]))
    st = np.clip(np.digitize(cnt, bins[1:-1], right=True), 0, len(bins) - 2)
    a, w = [], []
    for q in np.unique(st):
        k = st == q
        if y[k].min() != y[k].max(): a.append(roc_auc_score(y[k], s[k])); w.append(k.sum())
    return float(np.average(a, weights=w))

hard = np.concatenate([di[cnt[di] >= np.quantile(cnt[di], .75)],
                       ai[cnt[ai] <= np.quantile(cnt[ai], .25)]])

# ---------------------------------------------------------------------
# Iterate over folds
# ---------------------------------------------------------------------
rec, e1_all, nn_all = [], [], []
def put(fold, model, metric, val): rec.append({"fold": fold, "model": model, "metric": metric, "value": float(val)})

for k in range(1, 6):
    print(f"\n================ fold {k} ================")
    tr = pd.read_csv(BAND + f"train_valid_split/train_valid_{k}/train.csv",
                     usecols=["Morgan", "KLIFS", "classification_label"])
    tr = tr[tr.classification_label == 1].copy(); tr["KLIFS"] = tr.KLIFS.astype(str).str.strip()
    B = {"morgan": xgb.Booster(), "lsae": xgb.Booster()}
    B["morgan"].load_model(M_TPL.format(k=k)); B["lsae"].load_model(L_TPL.format(k=k))
    S = {t: score(t, B[t], fp_v, kl_v) for t in B}
    for t in B:
        put(k, t, "AUROC", roc_auc_score(y, S[t]))
        put(k, t, "EF1%", ef(y, S[t], 1)); put(k, t, "EF5%", ef(y, S[t], 5))
    print(f"  AUROC morgan={roc_auc_score(y,S['morgan']):.4f}  lsae={roc_auc_score(y,S['lsae']):.4f}  "
          f"| train actives rows={len(tr):,} unique={tr.Morgan.nunique():,}")

    # ---- M: position-level Morgan SHAP for this fold's training actives ----
    tu = tr.drop_duplicates(subset="Morgan")
    fp_t = np.stack([parse(s) for s in tu.Morgan]).astype(np.float32)
    Sh = B["morgan"].predict(xgb.DMatrix(np.concatenate([fp_t, klifs_of(tu.KLIFS)], 1),
                                         feature_names=FN_M), pred_contribs=True)[:, :NB]
    order = np.argsort(-np.abs(Sh).mean(0)); on_t = fp_t == 1
    for kk in [10, 20, 50]:
        g = grad[order[:kk]]; g = g[g >= 0]
        put(k, "morgan", f"posSHAP_top{kk}_r0_enrich", (g == 0).mean() / base_r0)
    put(k, "morgan", "posSHAP_top20_decoy_prev_median", np.median(prev[order[:20]]))
    t20 = order[:20]; sv = Sh[:, t20]; om = on_t[:, t20]
    put(k, "morgan", "posSHAP_top20_ON_frac_positive", (sv[om] > 0).mean())
    put(k, "morgan", "posSHAP_top20_OFF_frac_positive", (sv[~om] > 0).mean())
    GIVE = t20

    # ---- E1 ----
    n_a = len(ai) // 3
    for t in B:
        ds = di[np.argsort(-S[t][di])]; as_ = ai[np.argsort(-S[t][ai])]
        for g, idx in {"decoy_high (FP)": ds[:N_GROUP], "decoy_low": ds[-N_GROUP:],
                       "active_high (TP)": as_[:n_a], "active_low (FN)": as_[-n_a:]}.items():
            f0, k0 = fp_v[idx], kl_v[idx]
            row = {"fold": k, "model": t, "group": g, "full": float(S[t][idx].mean()),
                   "all_zero": float(score(t, B[t], np.zeros_like(f0), k0).mean())}
            for c, pos in {"keep_promiscuous": PROMISC, "keep_giveaway20": GIVE, "keep_rare": RARE}.items():
                msk = np.zeros(NB, dtype=np.uint8); msk[pos] = 1
                row[c] = float(score(t, B[t], f0 * msk[None], k0).mean())
            msk = np.ones(NB, dtype=np.uint8); msk[PROMISC] = 0
            row["delete_promiscuous"] = float(score(t, B[t], f0 * msk[None], k0).mean())
            row["delete_retained_frac"] = row["delete_promiscuous"] / row["full"]
            e1_all.append(row)
            if g in ("decoy_high (FP)", "active_high (TP)"):
                tag = "FP" if "FP" in g else "TP"
                put(k, t, f"E1_{tag}_keep_promiscuous", row["keep_promiscuous"])
                put(k, t, f"E1_{tag}_delete_retained_frac", row["delete_retained_frac"])

    # ---- E2 / E3 ----
    for t in B:
        put(k, t, "E2_spearman_decoy_score_vs_promisc", stats.spearmanr(S[t][di], cnt[di])[0])
        put(k, t, "E2_auroc_within_count_strata", strat_auroc(S[t]))
        put(k, t, "E3_adversarial_auroc", roc_auc_score(y[hard], S[t][hard]))

    # ---- E4 ----
    top = {t: set(di[np.argsort(-S[t][di])[:N_GROUP]]) for t in B}
    put(k, "pair", "E4_top2000_decoy_overlap", len(top["morgan"] & top["lsae"]) / N_GROUP)
    pct = {t: stats.rankdata(S[t][di]) / len(di) for t in B}; pos = {i: j for j, i in enumerate(di)}
    for src, dst in [("morgan", "lsae"), ("lsae", "morgan")]:
        q = pct[dst][[pos[i] for i in top[src]]]
        put(k, src, "E4_own_FP_median_pct_in_other", np.median(q))

    # ---- NN: same-target nearest neighbors among this fold's training actives ----
    nn_same = np.full(len(v), np.nan, dtype=np.float32)
    for kl_name, g in tr.groupby("KLIFS"):
        qi = np.flatnonzero(vk == kl_name)
        if len(qi): nn_same[qi] = nn_tanimoto(fp_v[qi], np.stack([parse(s) for s in g.Morgan.drop_duplicates()]))
    shared = np.array(sorted(top["morgan"] & top["lsae"]))
    rng = np.random.default_rng(SEED + k)
    grp = {"FP shared": shared, "FP morgan-only": np.array(sorted(top["morgan"] - top["lsae"])),
           "FP lsae-only": np.array(sorted(top["lsae"] - top["morgan"])),
           "decoy random": rng.choice(di, N_GROUP, replace=False), "VSDS actives": ai}
    for gname, idx in grp.items():
        x = nn_same[idx]; x = x[np.isfinite(x)]
        nn_all.append({"fold": k, "group": gname, "n": len(x), "median": float(np.median(x)),
                       "frac_ge_0.4": float((x >= .4).mean())})
    kk = di[np.isfinite(nn_same[di])]
    for t in B:
        put(k, t, "NN_spearman_decoy_score_vs_sameTargetNN", stats.spearmanr(S[t][kk], nn_same[kk])[0])
    print(f"  coverage same-target NN = {np.isfinite(nn_same).mean():.1%}")

# ---------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------
rec = pd.DataFrame(rec); rec.to_csv(os.path.join(OUT, "per_fold_metrics.csv"), index=False)
e1_all = pd.DataFrame(e1_all); e1_all.to_csv(os.path.join(OUT, "E1_retention_per_fold.csv"), index=False)
nn_all = pd.DataFrame(nn_all); nn_all.to_csv(os.path.join(OUT, "NN_by_group_per_fold.csv"), index=False)

print("\n\n############ Morgan position-level SHAP (5 fold) ############")
m = rec[rec.metric.str.startswith("posSHAP")].groupby("metric").value.agg(["mean", "std", "min", "max"])
print(m.round(4).to_string())

print("\n############ Morgan vs LSAE paired comparison (5 fold) ############")
rows = []
for met in rec[rec.model.isin(["morgan", "lsae"])].metric.unique():
    p = rec[rec.metric == met].pivot(index="fold", columns="model", values="value")
    if not {"morgan", "lsae"} <= set(p.columns) or p.isna().any().any(): continue
    d = p["lsae"] - p["morgan"]
    rows.append({"metric": met,
                 "morgan": f"{p.morgan.mean():.4f} ± {p.morgan.std():.4f}",
                 "lsae":   f"{p.lsae.mean():.4f} ± {p.lsae.std():.4f}",
                 "diff(lsae-morgan)": round(d.mean(), 4),
                 "folds lsae>morgan": f"{int((d > 0).sum())}/5",
                 "paired_t_p": stats.ttest_rel(p.lsae, p.morgan).pvalue})
cmp = pd.DataFrame(rows); cmp.to_csv(os.path.join(OUT, "paired_comparison.csv"), index=False)
print(cmp.to_string(index=False))

print("\n############ E4 ############")
print(rec[rec.metric == "E4_top2000_decoy_overlap"].value.describe()[["mean", "std", "min", "max"]].round(4).to_string())

print("\n############ E1 (5-fold mean) ############")
cols = ["full", "keep_promiscuous", "keep_giveaway20", "keep_rare", "delete_promiscuous", "delete_retained_frac", "all_zero"]
print(e1_all.groupby(["model", "group"])[cols].mean().round(4).to_string())

print("\n############ NN nearest neighbors for the same target (5-fold mean) ############")
print(nn_all.groupby("group")[["median", "frac_ge_0.4"]].mean().round(4).to_string())
print(f"\n[done] -> {OUT}")
