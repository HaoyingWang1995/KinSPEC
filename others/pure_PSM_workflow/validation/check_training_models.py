"""Independently reload all audit models and verify validation/test metrics and gain profiles."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts.workflow import AnalysisConfig
from compare_results import auc_from_ranks


def verify(out):
    state = json.loads((out / "audit_state.json").read_text())
    config = AnalysisConfig(**state["config"])
    e = config.embedding_dir
    ref = pd.read_csv(e / "human_482_typical_kinase_1955bit.csv")
    lut = pd.Series(np.arange(len(ref)), index=ref.KLIFS.astype(str).str.strip()).groupby(level=0).last()
    tr = pd.read_csv(config.split_dir / "band_0.10_0.50/train.csv", usecols=["KLIFS", "unified_bio_label"])
    y = tr.unified_bio_label.to_numpy(int)
    post = tr.KLIFS.astype(str).str.strip().map(lut).to_numpy(np.intp)
    mol = pd.read_csv(config.final_dir / "dataset/spectrum_data/band_0.10_0.50/train_LSAE_latent_top95_features_v4.csv", skiprows=1, header=None, dtype=np.float32).to_numpy()
    v = pd.read_csv(config.final_dir / "dataset/VSDS_truedecoy_integrated_data.csv", usecols=["KLIFS", "uniprot_id", "molecule_label"])
    yv, targets = v.molecule_label.to_numpy(int), v.uniprot_id.to_numpy()
    posv = v.KLIFS.astype(str).str.strip().map(lut).to_numpy(np.intp)
    molv = pd.read_csv(config.final_dir / "dataset/VSDS_truedecoy_LSAE_latent_top95_features_v4.csv", skiprows=1, header=None, dtype=np.float32).to_numpy()
    psm = pd.read_csv(e / "human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy(np.float32)
    mca = pd.read_csv(e / "MCA_klifs_482_200D_features.csv", skiprows=1, header=None).to_numpy(np.float32)
    # Independent equivalence classes: identical or complementary binary columns.
    groups = {}
    for column in range(psm.shape[1]):
        bit = psm[:,column].astype(np.uint8)
        if bit.min() == bit.max():
            continue
        canonical = min(bit.tobytes(), (1-bit).tobytes())
        groups.setdefault(canonical, column)
    keep = np.sort(np.array(list(groups.values())))
    if len(keep) != 1673:
        raise ValueError("Binary equivalence-class deduplication disagrees with the training analysis")
    pca_control = np.load(out / "control_pca_200.npy")
    pca_stability = np.load(out / "stability2_pca_200.npy")
    pca_components = np.load(out / "stability2_pca_components.npy")
    rand = (psm @ np.random.default_rng(0).normal(0, 1/np.sqrt(200), (1955,200))).astype(np.float32)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    tables = {name: pd.read_csv(out / f"{name}.csv") for name in ("dedup", "cross", "control")}
    with np.load(out / "stability2.npz") as profiles:
        expected_profiles = {key: profiles[key] for key in profiles.files}
    C = pd.read_csv(e / "MCA_200D_category_contributions.csv", index_col=0)
    residues = np.array([int(s.split("_")[0][3:])-1 for s in C.index])
    results = []
    files = sorted((out / "training_models").glob("*/*.json"))
    if len(files) != 75:
        raise ValueError(f"Expected 75 saved models, found {len(files)}")
    for file in files:
        meta = json.loads(file.read_text())
        path = file.with_suffix(".ubj")
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["model_sha256"]:
            raise ValueError(f"Model checkpoint hash changed: {path}")
        name, fold, stage = meta["representation"], meta["fold"], file.parent.name
        if stage == "dedup":
            K = psm if name == "PSM-1955 original" else psm[:,keep]
        elif name.startswith("PSM"):
            K = psm
        elif name.startswith("MCA"):
            K = mca
        elif name.startswith("PCA"):
            K = pca_stability if stage == "stability2" else pca_control
        elif name.startswith("RAND"):
            K = rand
        else:
            raise ValueError(f"Unknown representation: {name}")
        model = xgb.Booster(model_file=path)
        model.set_param({"device": "cpu", "nthread": 4})
        if model.num_features() != 417+K.shape[1] or model.num_features() != meta["num_features"]:
            raise ValueError("Checkpoint feature dimensions do not match reconstructed inputs")
        _, vi = folds[fold-1]
        dv = xgb.DMatrix(np.hstack([mol[vi], K[post[vi]]]))
        score = model.predict(dv, iteration_range=(0,meta["best_iteration"]+1))
        row = {"stage": stage, "representation": name, "fold": fold,
               "validation_auc_error": abs(auc_from_ranks(y[vi],score)-meta["best_validation_auc"])}
        if stage != "stability2":
            score = model.predict(xgb.DMatrix(np.hstack([molv,K[posv]])), iteration_range=(0,meta["best_iteration"]+1))
            pooled = auc_from_ranks(yv,score)
            within = float(np.mean([auc_from_ranks(yv[targets==t],score[targets==t]) for t in np.unique(targets)]))
            frame = tables[stage]
            if stage == "cross":
                kinase,hp = name.rsplit("_HP_",1)
                r = frame[(frame.kinase==kinase)&(frame.hp=="HP_"+hp)&(frame.fold==fold)].iloc[0]
            else:
                r = frame[(frame.rep==name)&(frame.fold==fold)].iloc[0]
            field = "within_target" if stage=="control" else "within"
            row.update(pooled_auc_error=abs(pooled-r.pooled),within_target_auc_error=abs(within-r[field]))
        else:
            gain = np.zeros(model.num_features())
            for key,value in model.get_score(importance_type="total_gain").items():
                gain[int(key.lstrip("f"))]=value
            kin = gain[417:]
            profile = np.zeros(85)
            if name=="PSM-1955":
                np.add.at(profile,np.arange(1955)//23,kin)
            elif name=="MCA-200":
                np.add.at(profile,residues,C.to_numpy()@kin)
            else:
                np.add.at(profile,np.arange(1955)//23,np.abs(pca_components).T@kin)
            profile /= profile.sum()
            row["gain_profile_error"] = float(np.abs(profile-expected_profiles[name][fold-1]).max())
        results.append(row)
        print("Verified checkpoint", stage,name,fold,flush=True)
        del model,dv
    errors = [value for row in results for key,value in row.items() if key.endswith("_error")]
    report = {"models_checked": len(results), "tolerance": 1e-5, "max_absolute_error": float(max(errors)),
              "status": "pass" if max(errors)<1e-5 else "needs_review", "details": results}
    (out / "training_checkpoint_checks.json").write_text(json.dumps(report,indent=2)+'\n')
    print("Checkpoint verification:",report['status'],report['max_absolute_error'],flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir",type=Path)
    verify(parser.parse_args().output_dir.resolve())
