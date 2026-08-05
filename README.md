# KinSPEC

KinSPEC is a machine-learning workflow for predicting compound–kinase interactions and analysing kinase selectivity. It combines molecular features learned from Morgan fingerprints, kinase binding-pocket features derived from KLIFS-aligned sequences, and XGBoost classifiers.

The repository provides Jupyter notebooks for dataset construction, feature embedding, model training, external evaluation, and model interpretation. Pretrained molecular encoders and trained KinSPEC models are also included.

## Method overview

1. Standardised SMILES strings are converted into Morgan fingerprints.
2. A Linear Sparse Autoencoder (LSAE) compresses the molecular fingerprints into latent features.
3. KLIFS-aligned kinase pocket sequences are encoded using residue physicochemical properties.
4. Multiple Correspondence Analysis (MCA) reduces the kinase representation to 200 dimensions.
5. Molecular and kinase features are concatenated and used to train an XGBoost binary classifier.
6. SHAP-based analyses connect model predictions to molecular substructures and kinase-pocket properties.

## Repository structure

```text
KinSPEC/
├── data_split/                 # Dataset construction and similarity-spectrum design
├── dataset/                    # Input and derived datasets
├── feature_embedding/
│   ├── mol/                    # Molecular LSAE training and feature extraction
│   └── Kinase_test_MCA/        # Kinase property encoding and MCA embedding
├── model/                      # Trained XGBoost models
├── model_training/             # Model training and benchmark evaluation
├── model_interpretability/     # SHAP and entropy analyses
├── download_model_dataset.ipynb
├── zenodo_upload_dataset.ipynb
└── LICENSE
```

## Installation

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/HaoyingWang1995/KinSPEC.git
cd KinSPEC

conda create -n kinspec python=3.10 -y
conda activate kinspec
```

Install Jupyter and the packages used by the notebooks:

```bash
pip install jupyter numpy pandas scipy scikit-learn matplotlib tqdm requests
pip install rdkit torch xgboost shap optuna prince
```

GPU acceleration is optional. If it is required, install the PyTorch build appropriate for the local CUDA version by following the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Notebook guide

### 1. Dataset construction

| Notebook | Purpose |
|---|---|
| `data_split/construct_training_pool_dataset_with_smiles.ipynb` | Merges the PKIS, DUD-E, DUD-Z, Davis, and VSDS sources; harmonises compound–kinase records; resolves conflicts and duplicates; and constructs the full training pool with SMILES. |
| `data_split/compound_similarity_spectrum_design.ipynb` | Calculates molecular similarity and creates controlled training subsets in predefined similarity bands for generalisation experiments. |
| `data_split/KIBA_construction.ipynb` | Constructs KIBA evaluation subsets, separates overlapping and non-overlapping entries, and links them to the KinSPEC/VSDS data used by the evaluation workflow. |
| `data_split/Karaman_dataset_expand.ipynb` | Converts the original wide-format Karaman kinase profiling table into a long compound–kinase table. |
| `data_split/Karaman_dataset_filter.ipynb` | Filters and standardises the Karaman benchmark, matches supported kinase targets, and evaluates molecular overlap/similarity with the training compounds. |

The `dataset/` directory contains the input and generated tables used by these workflows. Dataset contents are not documented individually here because the notebooks identify their required files at the point of use.

### 2. Molecular feature embedding

| Notebook or file pattern | Purpose |
|---|---|
| `feature_embedding/mol/LSAE_mol_training.ipynb` | Trains the molecular Linear Sparse Autoencoder on Morgan fingerprints and selects the retained latent features. |
| `feature_embedding/mol/LSAE_mol_load_model_*.ipynb` | Loads a pretrained LSAE and extracts molecular latent features for a particular dataset. The suffix identifies the target dataset or workflow: `KIBA`, `vsds`, `karaman`, `KinSPEC_karaman_selectivity_sim05`, or `full_training_pool`. |
| `feature_embedding/mol/*.pt` | Pretrained LSAE weights, selected latent-feature indices, and decoder artifacts used during molecular feature extraction and interpretation. |

The dataset-specific loading notebooks share the same main procedure but use different input tables and output paths. Select the notebook whose suffix matches the dataset being processed and update its path configuration before running it.

### 3. Kinase feature embedding

| Notebook | Purpose |
|---|---|
| `feature_embedding/Kinase_test_MCA/protein_to_binary_MCA.ipynb` | Converts KLIFS-aligned kinase binding-pocket sequences into binary physicochemical-property features. For 85 pocket positions and 23 residue properties, the initial representation contains 1,955 features. |
| `feature_embedding/Kinase_test_MCA/MCA_klifs.ipynb` | Fits a 200-dimensional MCA representation, exports kinase coordinates and category metadata, and assesses embedding quality and bootstrap stability. |

The CSV files in `feature_embedding/Kinase_test_MCA/` include the binary kinase representation, the 200-dimensional MCA features, category coordinates/contributions, row quality statistics, and bootstrap-stability results.

### 4. Model training

| Notebook | Purpose |
|---|---|
| `model_training/KinSPEC_full_data_training_MCA.ipynb` | Tunes and trains the final KinSPEC XGBoost classifier on the full training pool and evaluates it on VSDS. |
| `model_training/KinSPEC_similarity_band_training_MCA_2.ipynb` | Trains and evaluates models across compound-similarity bands using stratified cross-validation. |

The `model/` directory contains the resulting XGBoost JSON models. Files matching `KinSPEC_MCA_training_band_*_fold*.json` correspond to a similarity band and cross-validation fold; `KinSPEC_full_training_pool_MCA_200D.json` is the full-pool model.

### 5. Model evaluation and application

| Notebook or file pattern | Purpose |
|---|---|
| `model_training/load_KinSPEC_MCA_*.ipynb` | Loads trained KinSPEC models and evaluates them on the dataset named by the suffix, currently `KIBA` or `VSDS`. |
| `model_training/load_model_KinSPEC_karaman_selectivity_test_MCA*.ipynb` | Applies KinSPEC to the Karaman selectivity benchmark and reports ranking/selectivity metrics such as Spearman correlation, ROC-AUC, NDCG, and BEDROC. The `_ready_to_use` variant produces a streamlined per-molecule output. |

### 6. Model interpretation

| Notebook | Purpose |
|---|---|
| `model_interpretability/KinSPEC_interpretability.ipynb` | Uses TreeSHAP to explain molecular and kinase features, maps molecular contributions back to Morgan-fingerprint substructures, and relates kinase contributions to MCA/KLIFS properties. |
| `model_interpretability/shannon_entropy_statsitics_MCA_2.ipynb` | Quantifies the concentration and diversity of feature contributions using Shannon entropy and statistical analyses. |

### 7. Model and dataset transfer utilities

| Notebook | Purpose |
|---|---|
| `download_model_dataset.ipynb` | Downloads archived model and dataset resources. |
| `zenodo_upload_dataset.ipynb` | Uploads prepared resources to Zenodo; this is a maintainer utility and requires Zenodo credentials. |

## Recommended execution order

For reproducing the complete workflow, run the notebooks in this order:

```text
Dataset construction
  └─> Molecular feature embedding
       └─> Kinase property encoding
            └─> MCA kinase embedding
                 └─> Model training
                      └─> External evaluation
                           └─> Model interpretation
```

If pretrained embeddings and models are used, the earlier training stages can be skipped. For prediction on a new dataset, prepare compound SMILES and supported kinase identifiers, extract molecular features with an `LSAE_mol_load_model_*.ipynb` workflow, match the kinase MCA features, and then follow one of the model-loading notebooks.

## Usage notes

- Run notebooks from their repository subdirectory unless their path variables are updated; many relative paths assume this layout.
- Several notebooks contain machine-specific absolute paths. Change all input, output, checkpoint, and cache paths before execution.
- Dataset-specific notebooks expect particular column names and row orders. Inspect their configuration cells before substituting a new table.
- The molecular latent-feature table and the interaction table must remain row-aligned.
- Kinase identifiers must match the identifiers in the KLIFS reference table.
- Reproducibility of GPU results can depend on the installed PyTorch, CUDA, XGBoost, and driver versions.

## Data availability

The VSDS, KIBA, and Karaman evaluation resources, including data entries and embedded molecular features, are available from [Zenodo](https://zenodo.org/uploads/20558270).

## License

This project is licensed under the [Apache License 2.0](LICENSE).
