# KinSPEC

KinSPEC is a kinase-specific, sequence-based machine-learning framework for predicting compound-protein interactions as a binary classification task.

The framework combines:

- molecular structures represented by Morgan fingerprints;
- a Linear Sparse Autoencoder (LSAE) for molecular feature embedding;
- KLIFS-defined kinase binding-pocket sequences;
- Property Signature Method (PSM) features for kinase residues;
- Multiple Correspondence Analysis (MCA) for kinase embedding;
- XGBoost for compound-kinase interaction classification;
- TreeSHAP-based interpretation of molecular and kinase features.

The workflows in this repository are provided as Jupyter notebooks (`.ipynb`).

---

## Repository Structure

```text
KinSPEC/
├── data_split/
│   ├── compound_similarity_spectrum_design.ipynb
│   └── construct_training_pool_dataset_with_smiles.ipynb
│
├── dataset/
│   └── dataset_download.txt
│
├── feature_embedding/
│   ├── Kinase_test_MCA/
│   │   ├── human_482_typical_kinase_1955bit.csv
│   │   ├── MCA_200D_category_contributions.csv
│   │   ├── MCA_200D_category_coordinates.csv
│   │   ├── MCA_200D_category_lookup.csv
│   │   ├── MCA_kinase_200D_bootstrap_stability.csv
│   │   ├── MCA_kinase_200D_coordinates.csv
│   │   ├── MCA_kinase_200D_row_cos2.csv
│   │   ├── MCA_klifs_482_200D_features.csv
│   │   ├── MCA_klifs.ipynb
│   │   └── protein_to_binary_MCA.ipynb
│   │
│   └── mol/
│       ├── LSAE_1900k_chembl_model.pt
│       ├── LSAE_chembl_1900k_of1024d_decoder_top99.pt
│       ├── LSAE_chembl_1900k_topk_of1024d_top99.pt
│       ├── LSAE_mol_load_model_karaman.ipynb
│       ├── LSAE_mol_load_model_vsds.ipynb
│       ├── LSAE_mol_load_model.ipynb
│       ├── LSAE_mol_training.ipynb
│       ├── LSAE_training_pool_decoder_top99.pt
│       └── morgan_parse_and_sae.ipynb
│
├── model_interpretability/
│   ├── KinSPEC_interpretability_MCA_2.ipynb
│   └── shannon_entropy_statistics_MCA.ipynb
│
├── model_training/
│   ├── KinSPEC_full_data_training_MCA.ipynb
│   ├── KinSPEC_similarity_band_training_MCA_2.ipynb
│   └── load_KinSPEC_MCA.ipynb
│
├── LICENSE
└── README.md
```

---

## Input Data

The main input table should contain one compound-target pair per row.

### Required columns

| Column | Description |
|---|---|
| `uniprot_id` | UniProt accession of the kinase target |
| `SMILES` | Standardised SMILES representation of the compound |
| `molecule_label` | Binary interaction label: `1` for active and `0` for inactive |

Example:

```csv
uniprot_id,SMILES,molecule_label
P00533,CCOc1ccc2nc(S(N)(=O)=O)sc2c1,1
P31749,CC(=O)Oc1ccccc1C(=O)O,0
P42336,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,1
```

The example labels above are illustrative only.

### Input requirements

#### `uniprot_id`

- Use a UniProt accession rather than a gene symbol or protein name.
- The accession must be represented in the kinase feature files used by the notebooks.
- The current KinSPEC protein representation is designed for the 482 typical kinase entries included in this repository.
- Targets outside this kinase set require a compatible KLIFS-aligned 85-residue pocket representation before they can be used.

#### `SMILES`

- The value must be a valid molecular structure that can be parsed by RDKit.
- Standardised or canonical SMILES are recommended.
- Salts, disconnected fragments, duplicate structures, and inconsistent protonation states should be handled before training.
- Invalid SMILES entries should be corrected or removed.

#### `molecule_label`

Use integer binary labels:

- `1`: active compound-kinase interaction
- `0`: inactive compound-kinase interaction

The activity threshold used to create the labels depends on the source dataset and should be documented before model training.

---

## Prediction-Only Input

For prediction on unlabelled compound-kinase pairs, the `classification` column may be omitted:

```csv
uniprot_id,standard_smiles
P00533,CCOc1ccc2nc(S(N)(=O)=O)sc2c1
P31749,CC(=O)Oc1ccccc1C(=O)O
```

When experimental labels are available, the `classification` column can be retained for model evaluation.

---

## Data Availability

Dataset download information is provided in:

```text
dataset/dataset_download.txt
```

Download the required datasets before running the data-construction and training notebooks.

Large datasets and model checkpoints may not be stored directly in the GitHub repository. For large `.pt` files, Git LFS or an external repository such as Zenodo may be required.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/HaoyingWang1995/KinSPEC.git
cd KinSPEC
```

### 2. Create an environment

Using Conda:

```bash
conda create -n kinspec python
conda activate kinspec
```

Alternatively, using `venv`:

```bash
python -m venv kinspec_env
source kinspec_env/bin/activate
```

On Windows:

```bash
kinspec_env\Scripts\activate
```

### 3. Install the required packages

The notebooks use packages for molecular processing, machine learning, dimensionality reduction, model interpretation, statistics, and visualisation.

Core dependencies include:

```bash
pip install jupyter numpy pandas scipy scikit-learn matplotlib
pip install rdkit torch xgboost shap optuna prince umap-learn
```

Additional packages may be required by individual notebooks. The imports at the beginning of each notebook should be treated as the authoritative dependency list.

---

## Running the Notebooks

Start Jupyter Notebook from the repository directory:

```bash
jupyter notebook
```

Alternatively:

```bash
jupyter lab
```

Before running a notebook:

1. Select the correct Python environment as the Jupyter kernel.
2. Update all input, model, and output paths.
3. Confirm that the required datasets and pretrained model files are available.
4. Restart the kernel.
5. Run the cells from top to bottom.

Recommended Jupyter command:

```text
Kernel → Restart Kernel and Run All Cells
```

Because notebooks retain variables from previous executions, restarting the kernel before a complete run helps avoid hidden state and improves reproducibility.

---

## Workflow

The repository supports three main use cases:

1. construction of training and benchmark datasets;
2. model training and evaluation;
3. prediction and interpretation using trained models.

### A. Dataset construction

#### `data_split/construct_training_pool_dataset_with_smiles.ipynb`

Use this notebook to construct the compound-kinase training pool and associate each entry with its molecular SMILES representation.

Typical tasks include:

- loading source interaction datasets;
- standardising column names;
- assigning binary activity labels;
- matching compounds to SMILES strings;
- removing invalid or duplicated entries;
- producing the combined KinSPEC training pool.

#### `data_split/compound_similarity_spectrum_design.ipynb`

Use this notebook to construct the compound-similarity spectrum used for controlled generalisation experiments.

The notebook is intended to:

- calculate molecular similarity between training and test compounds;
- divide the training pool into predefined similarity bands;
- sample active and inactive compound-kinase pairs;
- prepare band-specific datasets for model comparison.

---

### B. Molecular feature embedding

The molecular feature workflows are located in:

```text
feature_embedding/mol/
```

#### `LSAE_mol_training.ipynb`

Use this notebook to train the molecular Linear Sparse Autoencoder on Morgan fingerprints.

The trained molecular model is stored as a PyTorch checkpoint:

```text
LSAE_1900k_chembl_model.pt
```

#### `morgan_parse_and_sae.ipynb`

Use this notebook to:

- parse standardised SMILES strings with RDKit;
- generate Morgan fingerprints;
- load the molecular LSAE;
- generate molecular latent features;
- save the embedded molecular representation for downstream modelling.

#### `LSAE_mol_load_model.ipynb`

Use this notebook for the general workflow of loading a pretrained molecular LSAE and applying it to molecular input data.

#### `LSAE_mol_load_model_vsds.ipynb`

Use this notebook to generate molecular features for the VSDS-VD TrueDecoy evaluation workflow.

#### `LSAE_mol_load_model_karaman.ipynb`

Use this notebook to generate molecular features for the Karaman kinase-selectivity evaluation workflow.

#### Saved molecular model files

```text
LSAE_1900k_chembl_model.pt
LSAE_chembl_1900k_of1024d_decoder_top99.pt
LSAE_chembl_1900k_topk_of1024d_top99.pt
LSAE_training_pool_decoder_top99.pt
```

These files contain the pretrained LSAE checkpoint and saved feature-selection or decoder artifacts used by the molecular embedding notebooks.

---

### C. Kinase feature embedding

The kinase feature workflows are located in:

```text
feature_embedding/Kinase_test_MCA/
```

#### `protein_to_binary_MCA.ipynb`

Use this notebook to convert KLIFS-aligned kinase pocket sequences into PSM-based protein features.

Each kinase pocket contains:

- 85 KLIFS-aligned residue positions;
- 23 binary physicochemical properties per residue;
- 1955 initial binary features in total.

The resulting kinase feature table is stored in:

```text
human_482_typical_kinase_1955bit.csv
```

#### `MCA_klifs.ipynb`

Use this notebook to:

- reorganise the binary PSM features into categorical residue-property variables;
- fit or evaluate the MCA representation;
- retain the 200-dimensional kinase embedding;
- calculate MCA category coordinates and contributions;
- assess row representation quality and bootstrap stability.

#### Saved kinase feature files

| File | Description |
|---|---|
| `human_482_typical_kinase_1955bit.csv` | Initial 1955-bit PSM representation for 482 typical kinases |
| `MCA_klifs_482_200D_features.csv` | Final 200-dimensional kinase feature table |
| `MCA_kinase_200D_coordinates.csv` | MCA row coordinates for kinase entries |
| `MCA_200D_category_coordinates.csv` | MCA coordinates for categorical residue-property levels |
| `MCA_200D_category_contributions.csv` | Category contributions used for kinase-side interpretation |
| `MCA_200D_category_lookup.csv` | Mapping between MCA categories and KLIFS residue-property variables |
| `MCA_kinase_200D_row_cos2.csv` | Row cos² values used to assess representation quality |
| `MCA_kinase_200D_bootstrap_stability.csv` | Bootstrap stability results for the retained MCA space |

---

### D. KinSPEC model training

The model-training workflows are located in:

```text
model_training/
```

#### `KinSPEC_similarity_band_training_MCA_2.ipynb`

Use this notebook to train KinSPEC models on the compound-similarity bands generated by:

```text
data_split/compound_similarity_spectrum_design.ipynb
```

This notebook is intended for the VSDS-VD compound-similarity spectrum evaluation.

#### `KinSPEC_full_data_training_MCA.ipynb`

Use this notebook to train KinSPEC on the full available training pool.

This workflow can be used when the aim is to maximise training coverage rather than compare individual similarity bands.

#### `load_KinSPEC_MCA.ipynb`

Use this notebook to load a trained KinSPEC model and generate compound-kinase interaction predictions.

The notebook combines:

- molecular LSAE features;
- kinase MCA features;
- a trained XGBoost classifier.

---

### E. Model interpretation

The interpretation workflows are located in:

```text
model_interpretability/
```

#### `KinSPEC_interpretability_MCA_2.ipynb`

Use this notebook to perform TreeSHAP-based interpretation of KinSPEC predictions.

For molecular features, important LSAE dimensions can be mapped back to:

- Morgan fingerprint bits;
- atom-centred molecular environments;
- molecular substructures.

For kinase features, important MCA components can be mapped back to:

- MCA residue-property categories;
- KLIFS-aligned pocket positions;
- kinase residue-level importance scores.

#### `shannon_entropy_statistics_MCA.ipynb`

Use this notebook to evaluate whether model-identified kinase residues are associated with conserved or variable regions of the kinase binding pocket.

The notebook performs Shannon entropy and statistical enrichment analyses across KLIFS positions or functional regions.

---

## Recommended Execution Order

### Training a similarity-band KinSPEC model

Run the notebooks in the following order:

```text
1. dataset/dataset_download.txt
2. data_split/construct_training_pool_dataset_with_smiles.ipynb
3. data_split/compound_similarity_spectrum_design.ipynb
4. feature_embedding/mol/morgan_parse_and_sae.ipynb
5. feature_embedding/Kinase_test_MCA/protein_to_binary_MCA.ipynb
6. feature_embedding/Kinase_test_MCA/MCA_klifs.ipynb
7. model_training/KinSPEC_similarity_band_training_MCA_2.ipynb
8. model_training/load_KinSPEC_MCA.ipynb
```

The molecular LSAE training notebook can be skipped when the pretrained `.pt` files are used.

### Training on the full training pool

```text
1. dataset/dataset_download.txt
2. data_split/construct_training_pool_dataset_with_smiles.ipynb
3. feature_embedding/mol/morgan_parse_and_sae.ipynb
4. feature_embedding/Kinase_test_MCA/protein_to_binary_MCA.ipynb
5. feature_embedding/Kinase_test_MCA/MCA_klifs.ipynb
6. model_training/KinSPEC_full_data_training_MCA.ipynb
7. model_training/load_KinSPEC_MCA.ipynb
```

### Running interpretation

After generating predictions with a trained model:

```text
1. model_interpretability/KinSPEC_interpretability_MCA_2.ipynb
2. model_interpretability/shannon_entropy_statistics_MCA.ipynb
```

The exact paths and intermediate file names should be checked inside each notebook before execution.

---

## Model Inputs

For each compound-kinase pair, KinSPEC combines two feature branches.

### Molecular branch

```text
standard SMILES
→ Morgan fingerprint
→ molecular LSAE
→ retained molecular latent features
```

### Kinase branch

```text
UniProt accession
→ KLIFS-aligned 85-residue pocket sequence
→ 1955-bit PSM representation
→ categorical residue-property variables
→ 200-dimensional MCA representation
```

### Classification branch

```text
molecular features + kinase features
→ XGBoost
→ predicted interaction probability
```

---

## Prediction Output

A typical prediction table may contain:

| Column | Description |
|---|---|
| `uniprot_id` | UniProt accession supplied in the input |
| `standard_smiles` | Standardised SMILES supplied in the input |
| `classification` | Experimental label, when available |
| `prediction_score` | Predicted probability of an active interaction |
| `predicted_class` | Binary prediction based on the selected threshold |

Example:

```csv
uniprot_id,standard_smiles,classification,prediction_score,predicted_class
P00533,CCOc1ccc2nc(S(N)(=O)=O)sc2c1,1,0.893,1
P31749,CC(=O)Oc1ccccc1C(=O)O,0,0.127,0
```

The prediction score can be used to:

- rank candidate compounds for a kinase target;
- rank possible kinase targets for a compound;
- evaluate compound-specific kinase selectivity;
- prioritise compound-kinase pairs for experimental validation.

---

## Reproducibility

For reproducible results:

- use the same package versions across experiments;
- keep random seeds fixed;
- restart the Jupyter kernel before each complete run;
- run cells in the documented order;
- use the same molecular standardisation procedure;
- preserve the original training and test separation;
- use the same UniProt-to-KLIFS mapping;
- use the same LSAE, MCA, and XGBoost model files;
- do not manually modify intermediate feature tables;
- record all input and output paths;
- record dataset versions and download dates.

---

## Important Notes

- KinSPEC is a binary compound-kinase interaction classifier.
- The protein representation is specific to typical protein kinases.
- A UniProt accession must match an entry in the kinase feature table.
- A gene symbol cannot directly replace a UniProt accession.
- Invalid SMILES strings cannot be converted into Morgan fingerprints.
- Classification labels must be encoded as `0` and `1`.
- The activity threshold used to define the labels should be reported.
- Duplicate compound-target pairs should be removed or reconciled before training.
- Predictions are computational estimates and do not replace experimental validation.
- TreeSHAP importance reflects contribution to the model prediction and does not by itself prove a physical molecular interaction.

---

## Citation

When using KinSPEC, please cite:

```bibtex
@article{wang2026kinspec,
  title   = {KinSPEC: A Kinase Sequence-Based Prior-Guided Framework for Predicting Compound-Protein Interactions},
  author  = {Wang, Haoying and Blay, Vincent and Chen, Lin and Chen, Yan-Kai and Zhou, Ruhong and Houston, Douglas R.},
  journal = {ChemRxiv},
  year    = {2026},
  doi     = {10.26434/chemrxiv.15005703/v1}
}
```

Preprint:

```text
https://chemrxiv.org/doi/full/10.26434/chemrxiv.15005703/v1
```

---

## Code and Data

GitHub repository:

```text
https://github.com/HaoyingWang1995/KinSPEC
```

Associated test data:

```text
https://zenodo.org/uploads/20558270
```

---

## License

This repository is distributed under the terms provided in the `LICENSE` file.

---

## Contact

For questions about KinSPEC, please contact:

**Haoying Wang**  
University of Edinburgh  
Email: H.Wang-138@sms.ed.ac.uk

**Douglas R. Houston**  
University of Edinburgh  
Email: DouglasR.Houston@ed.ac.uk
