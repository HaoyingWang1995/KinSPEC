# KinSPEC

KinSPEC is a machine-learning workflow for compound–kinase interaction prediction and kinase-selectivity analysis. It combines molecular features from a Linear Sparse Autoencoder (LSAE), kinase binding-pocket features derived from KLIFS-aligned sequences, and an XGBoost classifier. The repository contains training and evaluation notebooks, feature-embedding resources, interpretation scripts, baseline wrappers, and supplementary analyses. Large processed datasets and predictive model archives are distributed separately.

## Feature pipeline

1. Generate radius-2, 4,096-bit Morgan fingerprints for compounds.
2. Encode them using the pretrained v4 LSAE and retain **417** molecular features.
3. Encode **85 KLIFS pocket positions × 23 residue-property indicators = 1,955** kinase features.
4. Reduce the kinase block to **200** dimensions with Multiple Correspondence Analysis (MCA).
5. Concatenate features in **molecular, then kinase** order and train or load XGBoost: **417 + 200 = 617 inputs**.
6. Evaluate on VSDS, KIBA, and Karaman; interpret predictions with SHAP, pocket-property mapping, and entropy analysis.

Prepared features and model weights allow evaluation without retraining the autoencoder, refitting MCA, or retraining XGBoost. KinSPEC_LTS uses the same main representation with a Karaman-similarity-filtered training pool. The `v9` interpretation-script name denotes the analysis implementation; it loads **v4** model resources.

## Repository structure

```text
KinSPEC/
├── LICENSE
├── README.md
├── environment.yml
├── kinspec_paths.py                    # Shared project-root paths and filename aliases
├── data_split/                         # Five dataset-construction notebooks
├── dataset/
│   └── download_dataset_hf.py          # Download helper; large data are separate
├── model/
│   └── download_model_hf.py            # Download and extract model archives
├── feature_embedding/
│   ├── mol/                           # v4 LSAE notebooks, weights, indices, decoder
│   └── Kinase_test_MCA/                # PSM/MCA notebooks and released feature tables
├── model_training/                    # Eleven v4 training/evaluation notebooks
├── model_interpretability/            # SHAP, KernelSHAP, background and entropy scripts
└── others/
    ├── baseline/                      # HitScreen, MolTrans, PSICHIC, TransformerCPI wrappers
    ├── Morgan_ablation/               # Morgan training and supplementary analyses
    ├── one_hot_embedding_test/         # Amino-acid one-hot and MCA resources
    ├── pure_PSM_workflow/              # Six analysis entry points, validation, tests
    ├── Prior_conflict_pair_analysis/   # Prior-conflict and prior-dependence analyses
    ├── M6_PSM_feature_sharing/         # Feature-sharing analysis results
    └── draw_figures/                   # VSDS, KIBA LUDe and Karaman plotting
```

The supplementary release files listed above should accompany the corresponding workflows. Downloaded data belong in `dataset/`; extracted predictive weights belong in subfolders of `model/`. New interpretation and evaluation output directories are created by their workflows. The old `download_model_dataset.ipynb` and `zenodo_upload_dataset.ipynb` are absent from the inspected GitHub tree.

## Installation

Clone the code and create the supplied environment:

```bash
git clone https://github.com/HaoyingWang1995/KinSPEC.git
cd KinSPEC
conda env create -f environment.yml
conda activate KinSPEC
python -m pip install huggingface_hub jupyterlab prince seaborn statsmodels openpyxl
jupyter lab
```

`environment.yml` specifies Python 3.11 and the main numerical, PyTorch, XGBoost, RDKit, Optuna, and SHAP dependencies. Its platform/CUDA-specific package pins may require adjustment for a different machine. The additional command supplies downloader, notebook, MCA, and plotting packages absent from that file. Environment installation was not exercised by the upload audit.

Training notebooks commonly configure CUDA. Review their configuration cells before using a CPU machine. Baselines also require their upstream packages and preprocessing resources. The PSM numerical audit used a separate pinned dependency set; use its matching `requirements.txt` in a separate environment once that file is included, rather than treating it as a dependency lock for the whole project.

## Download datasets and predictive weights

The helpers target the [Hugging Face dataset](https://huggingface.co/datasets/Lumineb/KinSPEC) and [Hugging Face model repository](https://huggingface.co/Lumineb/KinSPEC). Both anonymous API checks returned HTTP 401 during the audit. Authenticate with an account that has access, or make the releases public before advertising anonymous downloads. File availability and archive contents were not verified through authenticated access.

```bash
# Authenticate when repository access requires it.
hf auth login

# Inspect accessible files before selecting downloads.
python dataset/download_dataset_hf.py --list
python model/download_model_hf.py --list

# Minimal prepared Karaman/LTS inputs, provided they appear in the file lists.
python dataset/download_dataset_hf.py --files \
    Karaman_Kinase_selectivity_filtered_withseq.csv \
    Karaman_10_50_filtered_SAE_latent_top95_features_from_merge_v4.csv
python model/download_model_hf.py --files KinSPEC.tar.gz
```

The default download destinations are the scripts' own `dataset/` and `model/` directories. Model `.tar.gz` files are extracted automatically; the dataset helper automatically extracts `data_split_from_merged_true_fold.tar.gz`. Both helpers support `--files`, `--include`, `--out`, `--no-extract`, and `--remove-archives`. Running either without file-selection options downloads all accessible release files and can require substantial disk space.

The earlier documentation also references [model DOI 10.5281/zenodo.21739888](https://doi.org/10.5281/zenodo.21739888) and [dataset DOI 10.5281/zenodo.20558270](https://doi.org/10.5281/zenodo.20558270). These are retained references; their coverage of the corrected v4 release has not been checked.

Expected model groups after obtaining the corresponding resources:

| Folder | Representation or purpose |
|---|---|
| `model/KinSPEC/` | Main 417 LSAE + 200 MCA models; full pool, six bands, and LTS |
| `model/KinSPEC_default/` | Default-parameter main band models |
| `model/PSM_raw/` | 417 LSAE + 1,955 raw PSM = 2,372 inputs |
| `model/onehot_MCA/` | 417 LSAE + 200 amino-acid one-hot/MCA = 617 inputs |
| `model/onehot_raw/` | 417 LSAE + 1,700 amino-acid one-hot = 2,117 inputs |
| `model/Morgan_raw/` | 4,096 raw Morgan + 200 MCA = 4,296 inputs |
| `model/{HitScreen,MolTrans,PSICHIC,TransformerCPI}/` | Retrained baseline weights |

Not all ablation weights are necessarily in the configured model release; check the file list. The inspected GitHub snapshot has no `others/PSM_model/` archive. Use the matching raw PSM weights or regenerate them through the appropriate training workflow.

## Dataset conventions

Similarity bands are `10_50`, `20_60`, `30_70`, `40_80`, `50_90`, and `60_100`, with five train/validation folds per band. Typical band files are:

```text
dataset/train_band_<range>_with_smiles_protein_seq.csv
dataset/train_band_<range>_LSAE_latent_top95_features_v4.csv
dataset/train_band_<range>_morgan_one_hot.csv
dataset/data_split_from_merged_true_fold/
  band_0.10_0.50/train_valid_split/train_valid_1/{train.csv,valid.csv}
```

Morgan matrix CSVs are **headerless** (`header=None`); released LSAE latent CSVs have a header and **417 columns**. Keep each feature matrix aligned with its source table's row order. Apply any filtering, sorting, or shuffling to both. Read binary-string `Morgan` columns as strings to retain leading zeros.

Flat training tables use names such as `SMILES`, `Protein_Sequence`, and `unified_bio_label`; fold tables use `Ligand`, `Protein`, and `classification_label`. LUDe-augmented KIBA workflows use `KIBA_seen_LUDe_augment_sim08.csv` and `KIBA_unseen_LUDe_augment_sim08.csv`; the unseen LSAE filename retains `augmented`: `KIBA_unseen_LUDe_augmented_sim08_LSAE_latent_top95_features_v4.csv`.

## Training and evaluation

Start Jupyter from the repository root and execute each notebook's first path-setup cell before the remaining cells.

| Notebook in `model_training/` | Purpose |
|---|---|
| `KinSPEC_full_data_training_MCA_v4.ipynb` | Tune/train the full-pool model and evaluate on VSDS |
| `load_KinSPEC_similarity_band_training_MCA_v4.ipynb` | Tune/train five-fold similarity-band models; despite its prefix, this performs training |
| `KinSPEC_similarity_band_training_MCA_default_param_v4.ipynb` | Train/evaluate default-parameter band models |
| `KinSPEC_LTS_Karaman_sim05_training_MCA_v4.ipynb` | Train the main LTS model |
| `KinSPEC_LTS_Karaman_sim05_training_MCA_v4_onehot.ipynb` | Train the LTS one-hot/MCA variant |
| `KinSPEC_LTS_Karaman_sim05_training_MCA_v4_pure_PSM.ipynb` | Train, save, and evaluate the LTS raw PSM model |
| `load_KinSPEC_MCA_KIBA_v4.ipynb` | Evaluate band models on non-augmented KIBA seen targets |
| `load_KinSPEC_MCA_KIBA_seen_LUDe_augment_v4.ipynb` | Evaluate band models on LUDe-augmented KIBA seen targets |
| `load_model_KinSPEC_karaman_selectivity_test_MCA_v4.ipynb` | Evaluate band fold models on Karaman |
| `load_model_KinSPEC_karaman_selectivity_test_MCA_ready_to_use_v4.ipynb` | Evaluate the supplied KinSPEC_LTS model on Karaman |

`KinSPEC_similarity_band_training_MCA_v4.ipynb` contains only a path-setup cell; use the band-training notebook listed above. Set `band_range` (e.g. `0.10_0.50`) and `band_range_2` (`10_50`) consistently when training band models.

For the prepared LTS evaluation, run [the Karaman LTS notebook](model_training/load_model_KinSPEC_karaman_selectivity_test_MCA_ready_to_use_v4.ipynb) with:

```text
dataset/Karaman_Kinase_selectivity_filtered_withseq.csv
dataset/Karaman_10_50_filtered_SAE_latent_top95_features_from_merge_v4.csv
feature_embedding/Kinase_test_MCA/human_482_typical_kinase_1955bit.csv
feature_embedding/Kinase_test_MCA/MCA_klifs_482_200D_features.csv
model/KinSPEC/KinSPEC_LTS_Karaman_sim05_training_pool_MCA_200D_v4.json
```

The notebook writes selectivity outputs under `evaluation_result/Karaman_MAC_result_LTS_v4/`. Main full-pool weights are named `KinSPEC_full_training_pool_MCA_200D_v4.json`; band weights use `KinSPEC_MCA_training_band_<range>_fold<1-5>_model_v4.json`.

## Feature generation and dataset construction

[Molecular notebooks](feature_embedding/mol/) train the v4 LSAE and extract embeddings for the full pool, similarity bands, VSDS, KIBA, Karaman, and LTS. Their current names include `_v4`. Supplied `.pt` resources include the LSAE, ranked latent-feature indices, and the LTS decoder. Use the first **417** ranked features for the main v4 model.

[protein_to_binary_MCA.ipynb](feature_embedding/Kinase_test_MCA/protein_to_binary_MCA.ipynb) encodes the pocket features; [MCA_klifs.ipynb](feature_embedding/Kinase_test_MCA/MCA_klifs.ipynb) fits MCA and exports coordinates, contributions, inertia, and stability resources. The released MCA table covers **482 human kinases**.

[data_split/](data_split/) contains the five construction notebooks for the training pool, similarity subsets, KIBA, and Karaman. These files are present on GitHub but absent from the corrected local reference directory used in the upload comparison. They are historical construction workflows: review their raw inputs and paths separately before running them. They are unnecessary for evaluation with prepared release data.

## Model interpretation

The main workflow is [KinSPEC_interpretability_v9.py](model_interpretability/KinSPEC_interpretability_v9.py). Its defaults require molecular and kinase background CSVs under `model_interpretability/background_data/`; these files are absent from the GitHub snapshot. Supply the matching backgrounds or review [prepare_interpretability_backgrounds.py](model_interpretability/prepare_interpretability_backgrounds.py) and its output paths before generating them from the downloaded training resources.

Once the models and background inputs are available:

```bash
python model_interpretability/KinSPEC_interpretability_v9.py \
    --output-root model_interpretability/encoder_topk_v9_new_run

python model_interpretability/KinSPEC_kernelshap_rf.py \
    --pair Imatinib_ABL1 \
    --outdir model_interpretability/kernelshap_Imatinib_ABL1_new_run
```

`KinSPEC_kernelshap_rf.py` imports both the v9 implementation and `KinSPEC_kernelshap_control.py`; keep all three files. Control need not be executed first. The `rf` implementation uses SAE + XGBoost. [compare_v9_kernelshap_atoms_rf.py](model_interpretability/compare_v9_kernelshap_atoms_rf.py) compares existing outputs rather than generating the underlying SHAP runs. Additional entropy and residue summaries are implemented in the other scripts and [entropy_analysis.ipynb](model_interpretability/entropy_analysis.ipynb).

## Raw PSM analyses

[others/pure_PSM_workflow/](others/pure_PSM_workflow/) contains six analysis entry points:

| Entry point | Purpose |
|---|---|
| `run_performance.py` | VSDS/Karaman performance and score decomposition |
| `run_attribution.py` | Per-kinase SHAP, individuality, overlap, and redundant attribution |
| `run_controls.py` | Encoding, deduplication, crossed hyperparameters, and dimension controls |
| `run_selectivity.py` | Similarity dependence, adversarial subsets, and stratification |
| `run_history.py` | Historical importance, stability, and exploratory analyses |
| `run_tables.py` | Export fixed historical SI values, rather than newly computed summaries |

The uploaded package includes 41 analysis modules, two intermediate-preparation modules, a workflow scheduler, validation helpers, and four test files. Default inputs are **22 CSVs directly in `dataset/` plus 31 MCA and 31 PSM JSON models** in `model/KinSPEC/` and `model/PSM_raw/`. The two Karaman result CSVs supplied in this code directory must also be available at their expected `dataset/` paths. Seven additional PSM/MCA feature tables complete the workflow-specific inputs.

After obtaining the inputs listed in the included manifest:

```bash
cd others/pure_PSM_workflow
python prepare_release_inputs.py
python run_attribution.py --dry-run
python run_attribution.py --check
python run_performance.py --device cpu
python run_selectivity.py --steps guilt,strat --device cpu
```

`--dry-run` lists the work; `--check` checks input availability. New results default to this directory's `results/`. Controls include expensive retraining and attribution may require substantial time and memory. Full numerical/historical validation has additional provenance requirements; the uploaded test suite includes release-maintenance tests referring to files outside the published code subset. The upload audit did not run full PSM calculations or assert that the public package passes those tests.

## Baselines and supplementary analyses

[others/baseline/](others/baseline/) contains KinSPEC-specific wrappers for HitScreen, MolTrans, PSICHIC, and TransformerCPI. Baseline checkpoints alone are insufficient: obtain their upstream implementations and prepare the expected embeddings. The corrected scripts expect upstream code under `external/<baseline>/`, MolTrans ESPF resources under `external/MolTrans/ESPF/`, prepared inputs under `dataset/preprocessed/<baseline>/`, and checkpoints under `model/<baseline>/`. These upstream/preprocessing resources are absent from the inspected code release.

Other folders provide raw Morgan ablation, amino-acid one-hot/MCA alternatives, prior-conflict analyses, PSM feature-sharing results, and figure scripts. Figure and comparison scripts may consume previously generated result tables. Review their input configuration before executing them.

## Validation and license

The upload audit compared shared file contents using SHA-256 and checked syntax for **96 Python files and 230 code cells across 38 notebooks**, with no syntax errors after removing notebook magic/shell lines for parsing. It confirmed the restored path helper matches the local source and that notebook path setup succeeds. The restored PSM manifest matches the local source and its checker found 84/84 prepared local inputs with expected file sizes. A fresh code checkout still needs those data and weights downloaded separately. These checks do not establish environment reproducibility, successful inference, retraining, or complete scientific reproduction.

KinSPEC code is released under the [Apache License 2.0](LICENSE). Third-party datasets and baseline implementations retain their respective licenses and terms. Cite the KinSPEC publication when its bibliographic details are available, along with the original resources used in your analysis.
