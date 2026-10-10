#!/usr/bin/env python3
"""Direct composite-model controls for KinSPEC interpretation.

1. Molecular KernelSHAP explains the complete 4096-bit -> SAE encoder -> XGBoost
   function while fixing the selected kinase.  Its intervention background is a
   deterministic sample from all unique training fingerprints.
2. Protein KernelSHAP explains the 200 MCA inputs while fixing the molecule.  It
   uses every unique kinase sequence that actually occurred in training by default.
3. Residue masking jointly hides all 23 physicochemical bits of one KLIFS
   residue and replaces that block with the same-position block from actual
   training kinases. It averages the resulting raw margins for all 482 kinases.
   This grouped interventional mask never creates an unfitted Gap category.
"""

from __future__ import annotations

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
_kp = _ks_partial(_ks_project_path, script_dir='model_interpretability')


import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib import cm, colors, colormaps
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SOFT_COOLWARM = colors.LinearSegmentedColormap.from_list(
    "soft_coolwarm",
    colormaps["coolwarm"](np.linspace(0.15, 0.85, 256)),
)


class SparseLinearAE(nn.Module):
    def __init__(self, input_dim: int = 4096, latent_dim: int = 1024):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)


def strings_to_bits(strings: list[str], width: int) -> np.ndarray:
    if not strings:
        raise ValueError("No fingerprints were loaded")
    if any(len(value) != width or set(value) - {"0", "1"} for value in strings):
        raise ValueError(f"Expected binary strings of width {width}")
    raw = np.frombuffer("".join(strings).encode("ascii"), dtype=np.uint8)
    return (raw.reshape(len(strings), width) - ord("0")).astype(np.float32)


def reservoir_background(path: Path, size: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if size < 1:
        raise ValueError("KernelSHAP requires a finite positive molecular background size")
    rng = np.random.default_rng(seed)
    reservoir: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if i < size:
                reservoir.append(row)
            else:
                j = int(rng.integers(0, i + 1))
                if j < size:
                    reservoir[j] = row
    frame = pd.DataFrame(reservoir).sort_values("molecule_id", key=lambda x: x.astype(int))
    return frame, strings_to_bits(frame["Morgan"].tolist(), 4096)


def morgan_from_smiles(smiles: str) -> tuple[np.ndarray, dict, Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    additional = rdFingerprintGenerator.AdditionalOutput()
    additional.AllocateBitInfoMap()
    fp = generator.GetFingerprint(mol, additionalOutput=additional)
    bits = np.zeros(4096, dtype=np.float32)
    bits[list(fp.GetOnBits())] = 1
    return bits, additional.GetBitInfoMap(), mol


def load_models(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = SparseLinearAE().to(device)
    state = torch.load(args.sae_model, map_location=device, weights_only=True)
    sae.load_state_dict(state)
    sae.eval()
    keep_idx = torch.load(args.sae_indices, map_location=device, weights_only=True).long()[:417]
    booster = xgb.Booster()
    booster.load_model(args.xgb_model)
    if booster.num_features() != len(keep_idx) + 200:
        raise ValueError(
            f"XGBoost expects {booster.num_features()} features, but SAE+MCA gives {len(keep_idx)+200}"
        )
    return device, sae, keep_idx, booster


def encode_molecules(bits: np.ndarray, sae: SparseLinearAE, keep_idx: torch.Tensor,
                     device: torch.device, batch_size: int = 4096) -> np.ndarray:
    output = []
    with torch.no_grad():
        for start in range(0, len(bits), batch_size):
            batch = torch.as_tensor(bits[start:start + batch_size], dtype=torch.float32, device=device)
            output.append(sae.encoder(batch)[:, keep_idx].cpu().numpy())
    return np.vstack(output).astype(np.float32)


def model_matrix(booster: xgb.Booster, values: np.ndarray) -> xgb.DMatrix:
    columns = booster.feature_names or [str(i) for i in range(values.shape[1])]
    return xgb.DMatrix(pd.DataFrame(values, columns=columns))


def select_kinase(kinase_meta: pd.DataFrame, kinase_mca: np.ndarray, identifier: str):
    hits = kinase_meta.index[
        kinase_meta["UniProt_ID"].astype(str).eq(identifier)
        | kinase_meta["kinase_name"].astype(str).eq(identifier)
    ].tolist()
    if not hits:
        raise KeyError(f"Kinase {identifier!r} not found in the 482-kinase table")
    return hits[0], kinase_mca[hits[0]].astype(np.float32)


def scalar_expected_value(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def atoms_for_environment(molecule, center_atom: int, radius: int) -> list[int]:
    if radius == 0:
        return [center_atom]
    bonds = Chem.FindAtomEnvironmentOfRadiusN(molecule, radius, center_atom)
    atoms = {center_atom}
    for bond_idx in bonds:
        bond = molecule.GetBondWithIdx(bond_idx)
        atoms.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    return sorted(atoms)


def write_active_bit_atom_heatmap(result, bit_info, molecule, kinase_name, outdir):
    """Map target-active bit SHAP to atoms while conserving their signed sum."""
    atom_scores = np.zeros(molecule.GetNumAtoms(), dtype=float)
    substructure_rows = []
    active = result[result["target_bit"].eq(1)]
    for bit, bit_score in active[["bit", "kernel_shap_signed"]].itertuples(index=False):
        occurrences = list(bit_info.get(int(bit), []))
        if not occurrences:
            if bit_score != 0:
                raise ValueError(f"Active Morgan bit {bit} has SHAP but no BitInfoMap occurrence")
            continue
        occurrence_score = float(bit_score) / len(occurrences)
        for occurrence, (center, radius) in enumerate(occurrences):
            atoms = atoms_for_environment(molecule, int(center), int(radius))
            per_atom = occurrence_score / len(atoms)
            atom_scores[atoms] += per_atom
            substructure_rows.append({
                "bit": int(bit), "occurrence": occurrence,
                "center_atom": int(center), "radius": int(radius),
                "atoms": ",".join(map(str, atoms)),
                "bit_kernel_shap_signed": float(bit_score),
                "occurrence_signed_score": occurrence_score,
                "per_atom_signed_score": per_atom,
            })

    active_signed_sum = float(active["kernel_shap_signed"].sum())
    np.testing.assert_allclose(atom_scores.sum(), active_signed_sum, rtol=1e-7, atol=1e-7)
    pd.DataFrame(substructure_rows).to_csv(
        outdir / "molecule_activebit_kernelshap_substructures.csv", index=False
    )
    atom_table = pd.DataFrame({
        "atom_id": np.arange(molecule.GetNumAtoms()),
        "atom_symbol": [atom.GetSymbol() for atom in molecule.GetAtoms()],
        "activebit_kernel_shap_signed": atom_scores,
        "direction": np.where(
            atom_scores > 0, "increases_raw_output",
            np.where(atom_scores < 0, "decreases_raw_output", "zero"),
        ),
    })
    atom_table.to_csv(outdir / "molecule_activebit_kernelshap_atoms.csv", index=False)

    limit = float(np.max(np.abs(atom_scores))) if len(atom_scores) else 0.0
    norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit) if limit > 0 else colors.Normalize(-1, 1)
    cmap = SOFT_COOLWARM
    highlighted = np.flatnonzero(atom_scores != 0).tolist()
    atom_colors = {int(a): tuple(cmap(norm(atom_scores[a]))[:3]) for a in highlighted}
    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 750)
    drawer.drawOptions().addAtomIndices = True
    drawer.drawOptions().bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, molecule, highlightAtoms=highlighted, highlightAtomColors=atom_colors
    )
    drawer.FinishDrawing()
    raw_path = outdir / "molecule_activebit_kernelshap_atom_heatmap.png"
    raw_path.write_bytes(drawer.GetDrawingText())

    image = Image.open(raw_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"Active-bit KernelSHAP atom heatmap — {kinase_name}")
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    mapper.set_array([])
    cbar = plt.colorbar(mapper, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Signed contribution to raw model output")
    fig.tight_layout()
    fig.savefig(
        outdir / "molecule_activebit_kernelshap_atom_heatmap_with_colorbar.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)
    return atom_table


def run_molecule_kernelshap(args, target_bits, bit_info, molecule, fixed_mca,
                            sae, keep_idx, device, booster, outdir):
    background_frame, background = reservoir_background(
        args.unique_morgan_background, args.mol_background_size, args.seed
    )
    background_frame.to_csv(outdir / "kernelshap_molecule_background_used.csv", index=False)

    def predict(raw_bits):
        latent = encode_molecules(np.asarray(raw_bits, dtype=np.float32), sae, keep_idx, device)
        protein = np.repeat(fixed_mca[None, :], len(latent), axis=0)
        model_input = np.hstack([latent, protein]).astype(np.float32)
        return booster.predict(model_matrix(booster, model_input), output_margin=True)

    np.random.seed(args.seed)
    explainer = shap.KernelExplainer(predict, background, link="identity")
    values = np.asarray(explainer.shap_values(
        target_bits[None, :], nsamples=args.mol_nsamples,
        l1_reg=f"num_features({args.mol_selected_features})",
    )).reshape(-1)
    prediction = float(predict(target_bits[None, :])[0])
    expected = scalar_expected_value(explainer.expected_value)
    result = pd.DataFrame({
        "bit": np.arange(4096),
        "target_bit": target_bits.astype(np.uint8),
        "kernel_shap_signed": values,
    })
    result.to_csv(outdir / "molecule_4096bit_kernelshap_signed.csv", index=False)
    write_active_bit_atom_heatmap(result, bit_info, molecule, args.kinase, outdir)
    total_abs = float(result["kernel_shap_signed"].abs().sum())
    active_abs = float(result.loc[result.target_bit.eq(1), "kernel_shap_signed"].abs().sum())
    pd.DataFrame([{
        "prediction_raw_margin": prediction,
        "expected_value": expected,
        "sum_shap": values.sum(),
        "additivity_residual": prediction - expected - values.sum(),
        "background_size": len(background),
        "nsamples": args.mol_nsamples,
        "l1_reg": f"num_features({args.mol_selected_features})",
        "active_bit_signed_sum": float(result.loc[result.target_bit.eq(1), "kernel_shap_signed"].sum()),
        "active_bit_abs_sum": active_abs,
        "all_bit_abs_sum": total_abs,
        "active_bit_abs_coverage": active_abs / total_abs if total_abs else np.nan,
    }]).to_csv(outdir / "molecule_kernelshap_diagnostics.csv", index=False)
    return result


def run_protein_kernelshap(args, target_mca, fixed_latent, booster, outdir):
    training = pd.read_csv(args.training_kinase_background)
    mca_cols = [f"mca_{i}" for i in range(1, 201)]
    background = training[mca_cols].to_numpy(np.float32)
    if args.kin_background_size and args.kin_background_size < len(background):
        rng = np.random.default_rng(args.seed)
        chosen = np.sort(rng.choice(len(background), args.kin_background_size, replace=False))
        training = training.iloc[chosen].copy()
        background = background[chosen]
    training.to_csv(outdir / "kernelshap_kinase_background_used.csv", index=False)

    def predict(mca_rows):
        mca_rows = np.asarray(mca_rows, dtype=np.float32)
        molecule = np.repeat(fixed_latent[None, :], len(mca_rows), axis=0)
        values = np.hstack([molecule, mca_rows]).astype(np.float32)
        return booster.predict(model_matrix(booster, values), output_margin=True)

    np.random.seed(args.seed)
    explainer = shap.KernelExplainer(predict, background, link="identity")
    values = np.asarray(explainer.shap_values(
        target_mca[None, :], nsamples=args.protein_nsamples,
        l1_reg=f"num_features({args.protein_selected_features})",
    )).reshape(-1)
    prediction = float(predict(target_mca[None, :])[0])
    expected = scalar_expected_value(explainer.expected_value)
    pd.DataFrame({
        "component": np.arange(1, 201),
        "mca_value": target_mca,
        "kernel_shap_signed": values,
    }).to_csv(outdir / "protein_MCA200_kernelshap_signed.csv", index=False)
    pd.DataFrame([{
        "prediction_raw_margin": prediction,
        "expected_value": expected,
        "sum_shap": values.sum(),
        "additivity_residual": prediction - expected - values.sum(),
        "background_size": len(background),
        "nsamples": args.protein_nsamples,
    }]).to_csv(outdir / "protein_kernelshap_diagnostics.csv", index=False)


def build_residue_mca_contributions(args, onehot: np.ndarray, saved_mca: np.ndarray):
    """Return each fitted row's exact contribution from each of 85 residues."""
    lookup = pd.read_csv(args.mca_lookup)
    coords = pd.read_csv(args.mca_category_coordinates, index_col=0)
    coords.index = coords.index.astype(str).str.replace("__", "_", regex=False)
    eigen = pd.read_csv(args.mca_inertia)["raw_eigenvalue"].to_numpy(float)[:200]
    standard = coords.to_numpy(float) / np.sqrt(eigen)[None, :]
    standard_by_name = dict(zip(coords.index, standard))

    n_variables = lookup.groupby(["residue", "system"]).ngroups
    if n_variables != 425:
        raise ValueError(f"Expected 425 MCA variables, found {n_variables}")
    contribution = np.zeros((len(onehot), 85, 200), dtype=float)
    for (residue, _), group in lookup.groupby(["residue", "system"], sort=False):
        gap_rows = group[group["category"].eq("Gap")]
        if len(gap_rows) != 1:
            raise ValueError("Every residue/system MCA variable must have exactly one Gap category")
        gap_name = str(gap_rows.iloc[0]["category_name"]).replace("__", "_")
        active_rows = group.dropna(subset=["bit_index_0_based"])
        bits = active_rows["bit_index_0_based"].astype(int).to_numpy()
        names = active_rows["category_name"].astype(str).str.replace("__", "_", regex=False).tolist()
        block = onehot[:, bits]
        counts = block.sum(axis=1)
        if np.any(counts > 1):
            raise ValueError(f"Invalid multi-category state at residue {residue}")
        for row_idx in range(len(onehot)):
            selected_name = gap_name if counts[row_idx] == 0 else names[int(np.argmax(block[row_idx]))]
            if selected_name not in standard_by_name:
                raise ValueError(f"Observed MCA category has no fitted coordinate: {selected_name}")
            contribution[row_idx, int(residue) - 1] += standard_by_name[selected_name] / n_variables

    reproduced = contribution.sum(axis=1)
    np.testing.assert_allclose(reproduced, saved_mca, rtol=1e-8, atol=1e-10)
    return contribution


def run_residue_masking(args, fixed_latent, booster, outdir):
    meta = pd.read_csv(args.kinase_table)
    onehot = pd.read_csv(args.kinase_onehot, header=None).to_numpy(np.float64)
    mca = pd.read_csv(args.kinase_mca).to_numpy(np.float64)
    if onehot.shape != (482, 1955) or mca.shape != (482, 200):
        raise ValueError(f"Unexpected kinase matrices: onehot={onehot.shape}, MCA={mca.shape}")
    residue_contribution = build_residue_mca_contributions(args, onehot, mca)

    training = pd.read_csv(args.training_kinase_background)
    background_rows = training["human_482_rows"].astype(str).str.split("|").str[0].astype(int).to_numpy()
    if args.mask_background_size and args.mask_background_size < len(background_rows):
        rng = np.random.default_rng(args.seed)
        background_rows = np.sort(rng.choice(background_rows, args.mask_background_size, replace=False))
    pd.DataFrame({"human_482_row": background_rows}).to_csv(
        outdir / "residue_mask_kinase_background_used.csv", index=False
    )

    molecule = np.repeat(fixed_latent[None, :], 482, axis=0)
    original_values = np.hstack([molecule, mca]).astype(np.float32)
    original = booster.predict(model_matrix(booster, original_values), output_margin=True)

    frames = []
    for residue in range(85):
        target_without_residue = mca - residue_contribution[:, residue]
        prediction_sum = np.zeros(482, dtype=float)
        prediction_sq_sum = np.zeros(482, dtype=float)
        n_background = 0
        for batch_start in range(0, len(background_rows), args.mask_background_batch_size):
            rows = background_rows[batch_start:batch_start + args.mask_background_batch_size]
            replacement = residue_contribution[rows, residue]
            masked_mca = target_without_residue[:, None, :] + replacement[None, :, :]
            flat_mca = masked_mca.reshape(-1, 200)
            flat_molecule = np.repeat(fixed_latent[None, :], len(flat_mca), axis=0)
            masked_values = np.hstack([flat_molecule, flat_mca]).astype(np.float32)
            predictions = booster.predict(
                model_matrix(booster, masked_values), output_margin=True
            ).reshape(482, len(rows))
            prediction_sum += predictions.sum(axis=1)
            prediction_sq_sum += np.square(predictions).sum(axis=1)
            n_background += len(rows)
        masked_mean = prediction_sum / n_background
        masked_variance = np.maximum(prediction_sq_sum / n_background - masked_mean ** 2, 0)
        signed = original - masked_mean
        frames.append(pd.DataFrame({
            "kinase_row": np.arange(482),
            "kinase_name": meta["kinase_name"],
            "UniProt_ID": meta["UniProt_ID"],
            "residue_position": residue + 1,
            "amino_acid": meta["KLIFS"].str[residue],
            "original_raw_margin": original,
            "interventional_masked_mean_raw_margin": masked_mean,
            "interventional_masked_sd_raw_margin": np.sqrt(masked_variance),
            "mask_background_size": n_background,
            "signed_residue_importance": signed,
            "direction": np.where(
                signed > 0, "presence_increases_raw_output",
                np.where(signed < 0, "presence_decreases_raw_output", "no_change"),
            ),
        }))
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(outdir / "all_482_kinases_residue23_interventional_mask_signed.csv", index=False)
    summary = result.groupby("residue_position", as_index=False).agg(
        mean_signed_importance=("signed_residue_importance", "mean"),
        median_signed_importance=("signed_residue_importance", "median"),
        positive_kinases=("signed_residue_importance", lambda x: int((x > 0).sum())),
        negative_kinases=("signed_residue_importance", lambda x: int((x < 0).sum())),
        unchanged_kinases=("signed_residue_importance", lambda x: int((x == 0).sum())),
    )
    summary.to_csv(outdir / "all_482_residue_interventional_mask_summary_signed.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", default="Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1")
    parser.add_argument("--kinase", default="P00519")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--mol-background-size", type=int, default=25)
    parser.add_argument("--mol-nsamples", type=int, default=2048)
    parser.add_argument("--mol-selected-features", type=int, default=200)
    parser.add_argument("--kin-background-size", type=int, default=0,
                        help="0 uses all unique training kinase sequences")
    parser.add_argument("--protein-nsamples", type=int, default=1024)
    parser.add_argument("--protein-selected-features", type=int, default=100)
    parser.add_argument("--mask-background-size", type=int, default=0,
                        help="0 uses all unique training kinase sequences for residue masking")
    parser.add_argument("--mask-background-batch-size", type=int, default=16)
    parser.add_argument("--skip-molecule-kernel", action="store_true")
    parser.add_argument("--skip-protein-kernel", action="store_true")
    parser.add_argument("--skip-residue-mask", action="store_true")
    parser.add_argument("--outdir", type=Path, default=HERE / "kernelshap_results")
    parser.add_argument("--unique-morgan-background", type=Path,
                        default=HERE / "background_data/training_unique_morgan_fingerprints.csv")
    parser.add_argument("--training-kinase-background", type=Path,
                        default=HERE / "background_data/training_unique_kinases_1955bit_MCA200.csv")
    parser.add_argument("--sae-model", type=Path,
                        default=ROOT / _kp('feature_embedding/mol/LSAE_chembl_1900k_full_dataset_v4_model.pt'))
    parser.add_argument("--sae-indices", type=Path,
                        default=ROOT / _kp('feature_embedding/mol/LSAE_chembl_1900k_full_dataset_v4_topk_of1024d_top99.pt'))
    parser.add_argument("--xgb-model", type=Path,
                        default=ROOT / _kp('model/KinSPEC/KinSPEC_full_training_pool_MCA_200D_v4.json'))
    kinase_dir = ROOT / _kp('feature_embedding/Kinase_test_MCA')
    parser.add_argument("--kinase-table", type=Path, default=kinase_dir / _kp('feature_embedding/Kinase_test_MCA/human_482_typical_kinase_1955bit.csv'))
    parser.add_argument("--kinase-onehot", type=Path, default=kinase_dir / _kp('feature_embedding/Kinase_test_MCA/human_482_typical_kinase_1955bit_onehot.csv'))
    parser.add_argument("--kinase-mca", type=Path, default=kinase_dir / _kp('feature_embedding/Kinase_test_MCA/MCA_klifs_482_200D_features.csv'))
    parser.add_argument("--mca-lookup", type=Path, default=kinase_dir / _kp('feature_embedding/Kinase_test_MCA/MCA_200D_category_lookup.csv'))
    parser.add_argument("--mca-category-coordinates", type=Path, default=kinase_dir / _kp('feature_embedding/Kinase_test_MCA/MCA_200D_category_coordinates.csv'))
    parser.add_argument("--mca-inertia", type=Path, default=kinase_dir / _kp('feature_embedding/Kinase_test_MCA/MCA_inertia_spectrum.csv'))
    return parser.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "run_config.json").open("w") as handle:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, handle, indent=2)
    device, sae, keep_idx, booster = load_models(args)
    target_bits, bit_info, target_molecule = morgan_from_smiles(args.smiles)
    target_latent = encode_molecules(target_bits[None, :], sae, keep_idx, device)[0]
    kinase_meta = pd.read_csv(args.kinase_table)
    kinase_mca = pd.read_csv(args.kinase_mca).to_numpy(np.float32)
    _, target_mca = select_kinase(kinase_meta, kinase_mca, args.kinase)

    if not args.skip_molecule_kernel:
        run_molecule_kernelshap(
            args, target_bits, bit_info, target_molecule, target_mca,
            sae, keep_idx, device, booster, args.outdir
        )
    if not args.skip_protein_kernel:
        run_protein_kernelshap(args, target_mca, target_latent, booster, args.outdir)
    if not args.skip_residue_mask:
        run_residue_masking(args, target_latent, booster, args.outdir)


if __name__ == "__main__":
    main()
