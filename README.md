# KinSPEC

KinSPEC is a machine-learning workflow for compound–kinase interaction prediction and kinase-selectivity analysis. It combines a molecular representation learned from Morgan fingerprints, a kinase binding-pocket representation derived from KLIFS-aligned sequences, and an XGBoost classifier.

This repository contains the notebooks, processed evaluation datasets, feature-embedding resources, trained models, and interpretation outputs needed to reproduce the KinSPEC workflow and apply the released models.

## Workflow

1. Standardise compound SMILES and convert them to Morgan fingerprints.
2. Use a Linear Sparse Autoencoder (LSAE) to compress the fingerprints into molecular latent features.
3. Convert KLIFS-aligned kinase pocket sequences into residue-property indicators.
4. Reduce the kinase representation to 200 dimensions with Multiple Correspondence Analysis (MCA).
5. Concatenate the molecular and kinase features and train an XGBoost binary classifier.
6. Evaluate interaction prediction and kinase selectivity on VSDS, KIBA, and Karaman data.
7. Use SHAP and entropy analyses to relate predictions to molecular substructures and kinase-pocket properties.

## Repository structure

```text
KinSPEC/
├── data_split/                 # Dataset construction, filtering, and similarity splits
├── dataset/                    # Processed VSDS, KIBA, and Karaman data
├── feature_embedding/
│   ├── mol/                    # Molecular LSAE training and feature extraction
│   └── Kinase_test_MCA/        # Kinase property encoding and MCA resources
├── model/                      # Released XGBoost model files
├── model_training/             # Training, evaluation, and application notebooks
├── model_interpretability/     # SHAP analyses, entropy statistics, and outputs
├── download_model_dataset.ipynb
├── zenodo_upload_dataset.ipynb
├── environment.yml
└── LICENSE
```

## Installation

Clone the repository and create the supplied Conda environment:

```bash
git clone https://github.com/HaoyingWang1995/KinSPEC.git
cd KinSPEC
conda env create -f environment.yml
conda activate KinSPEC
```

The environment currently specifies Python 3.11. If a notebook requires an additional package that is not recorded in the environment, install it in the activated environment before running that notebook.

## Downloading released data and models

The repository includes `download_model_dataset.ipynb`, which downloads the released archives from Zenodo into `dataset/` and `model/`. Review its output paths before running it on another machine because the notebook currently contains repository-specific absolute paths.

The referenced Zenodo records are:

- Models: [Zenodo record 21739888](https://zenodo.org/records/21739888)
- Datasets: [Zenodo record 20558270](https://zenodo.org/records/20558270)

## Notebook guide

### 1. Dataset construction and splitting

| Notebook | Purpose |
|---|---|
| `data_split/construct_training_pool_dataset_with_smiles.ipynb` | Merges the PKIS, DUD-E, DUD-Z, Davis, and VSDS sources, resolves conflicts and duplicates, and constructs the training pool with SMILES. |
| `data_split/compound_similarity_spectrum_design.ipynb` | Calculates compound similarity and creates controlled training subsets in predefined similarity bands. |
| `data_split/KIBA_construction.ipynb` | Constructs the KIBA seen-target and unseen-target evaluation subsets and removes overlap with the KinSPEC/VSDS data. |
| `data_split/Karaman_dataset_expand.ipynb` | Converts the original wide Karaman profiling table into a long compound–kinase table. |
| `data_split/Karaman_dataset_filter.ipynb` | Standardises and filters the Karaman benchmark, maps supported kinases, and evaluates compound overlap with the training set. |

### 2. Molecular feature embedding

| Notebook or resource | Purpose |
|---|---|
| `feature_embedding/mol/LSAE_mol_training.ipynb` | Trains the molecular LSAE on Morgan fingerprints and selects the retained latent dimensions. |
| `feature_embedding/mol/LSAE_mol_load_model_full_training_pool.ipynb` | Generates molecular latent features for the full KinSPEC training pool. |
| `feature_embedding/mol/LSAE_mol_load_model_KIBA.ipynb` | Generates molecular latent features for the KIBA evaluation data. |
| `feature_embedding/mol/LSAE_mol_load_model_vsds.ipynb` | Generates molecular latent features for VSDS. |
| `feature_embedding/mol/LSAE_mol_load_model_karaman.ipynb` | Generates molecular latent features for the filtered Karaman evaluation data. |
| `feature_embedding/mol/LSAE_mol_load_model_KinSPEC_LTS.ipynb` | Generates the Karaman-similarity-filtered molecular features used by the KinSPEC_LTS workflow. |
| `feature_embedding/mol/*.pt` | Pretrained LSAE weights, retained latent-feature indices, and decoder resources. |

The dataset-specific notebooks follow the same feature-extraction procedure but use different input and output paths. Check each notebook's configuration cell before execution.

### 3. Kinase feature embedding

| Notebook | Purpose |
|---|---|
| `feature_embedding/Kinase_test_MCA/protein_to_binary_MCA.ipynb` | Converts KLIFS-aligned kinase pocket sequences into 1,955 binary features (85 positions × 23 residue properties). |
| `feature_embedding/Kinase_test_MCA/MCA_klifs.ipynb` | Produces the 200-dimensional MCA kinase representation and exports category coordinates, contributions, row quality, and bootstrap-stability results. |

The generated kinase resources are stored in `feature_embedding/Kinase_test_MCA/`. The primary model input is `MCA_klifs_482_200D_features.csv`.

### 4. Model training

| Notebook | Purpose |
|---|---|
| `model_training/KinSPEC_full_data_training_MCA.ipynb` | Tunes and trains the full-training-pool KinSPEC model and evaluates it on VSDS. |
| `model_training/KinSPEC_similarity_band_training_MCA_2.ipynb` | Trains five-fold models for the defined compound-similarity bands. |

Released model files include:

| Model file | Description |
|---|---|
| `model/KinSPEC_full_training_pool_MCA_200D.json` | KinSPEC model trained on the full training pool with the 200-dimensional kinase MCA representation. |
| `model/KinSPEC_MCA_training_band_<lower>_<upper>_fold<1-5>.json` | Cross-validation models trained in the indicated compound-similarity band. |
| `model/KinSPEC_training_pool_karaman_specific_sim50_MCA.json` | **KinSPEC_LTS model** used for the Karaman-specific, similarity-filtered selectivity workflow. This is the model associated with `model_training/load_model_KinSPEC_karaman_selectivity_test_MCA_ready_to_use.ipynb`. |

> **KinSPEC_LTS naming note:** throughout this project, `KinSPEC_training_pool_karaman_specific_sim50_MCA.json` is the model referred to as **KinSPEC_LTS**. It is the trained model loaded and evaluated by `load_model_KinSPEC_karaman_selectivity_test_MCA_ready_to_use.ipynb`.

### 5. Evaluation and application

| Notebook | Purpose |
|---|---|
| `model_training/load_KinSPEC_MCA_VSDS.ipynb` | Loads `KinSPEC_full_training_pool_MCA_200D.json` and evaluates it on VSDS. |
| `model_training/load_KinSPEC_MCA_KIBA.ipynb` | Evaluates the similarity-band models on KIBA and exports overall and per-target summaries. |
| `model_training/load_model_KinSPEC_karaman_selectivity_test_MCA.ipynb` | Evaluates the fold models on the Karaman selectivity benchmark and exports per-molecule and aggregate metrics. |
| `model_training/load_model_KinSPEC_karaman_selectivity_test_MCA_ready_to_use.ipynb` | Ready-to-use KinSPEC_LTS workflow. It loads `KinSPEC_training_pool_karaman_specific_sim50_MCA.json` and produces per-molecule and summary selectivity results. |

The Karaman workflows report ranking and selectivity metrics including Spearman correlation, ROC-AUC, NDCG, and BEDROC.

### 6. Model interpretation

| Notebook | Purpose |
|---|---|
| `model_interpretability/KinSPEC_interpretability.ipynb` | Uses TreeSHAP to explain molecular and kinase features, maps molecular contributions to Morgan-fingerprint substructures, and maps kinase contributions to MCA/KLIFS properties. |
| `model_interpretability/shannon_entropy_statsitics_MCA_2.ipynb` | Measures the concentration and diversity of feature contributions using Shannon entropy and associated statistical analyses. |

Example interpretation results are included under `model_interpretability/Mol_LSAE_Kinase_MCA_xgb_shap_P00519_Imatinib/`, and VSDS active-residue summaries are under `model_interpretability/vsds_active_residue_analysis_MCA/`.

## Recommended execution paths

To reproduce model development from source data:

```text
Dataset construction and filtering
  -> Molecular LSAE feature extraction
  -> Kinase residue-property encoding
  -> 200D kinase MCA embedding
  -> XGBoost model training
  -> External evaluation
  -> Model interpretation
```

To run the released KinSPEC_LTS model on the prepared Karaman benchmark:

```text
dataset/Karaman_Kinase_selectivity_filtered_withseq.csv
  + dataset/Karaman_10_50_filtered_SAE_latent_top95_features_from_merge_2.csv
  + feature_embedding/Kinase_test_MCA/MCA_klifs_482_200D_features.csv
  + model/KinSPEC_training_pool_karaman_specific_sim50_MCA.json
  -> model_training/load_model_KinSPEC_karaman_selectivity_test_MCA_ready_to_use.ipynb
```

If the released processed features and models are used, dataset construction, LSAE training, MCA fitting, and XGBoost training can be skipped.

## Important usage notes

- Several notebooks contain machine-specific absolute paths. Update all input, output, checkpoint, and cache paths before execution.
- Relative paths generally assume that a notebook is run from its own repository subdirectory.
- Check the dataset and feature filenames in the configuration cells; some exploratory cells may retain older output names.
- Interaction tables and their molecular latent-feature tables must have identical row ordering.
- Kinase identifiers and KLIFS strings must match the supplied kinase reference and MCA tables.
- The XGBoost feature order must remain molecular latent features followed by the 200 kinase MCA features.
- GPU reproducibility can depend on PyTorch, CUDA, XGBoost, and driver versions.
- `zenodo_upload_dataset.ipynb` is a maintainer utility and requires valid Zenodo credentials.

## License

KinSPEC is released under the [Apache License 2.0](LICENSE).
