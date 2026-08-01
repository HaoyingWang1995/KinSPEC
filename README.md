# KinSPEC

**KinSPEC** is a kinase-specific, sequence-based machine-learning framework for predicting compound-protein interactions as a binary classification task.

The framework combines molecular fingerprints with KLIFS-defined kinase binding-pocket features. Molecular structures are represented by Morgan fingerprints and embedded using a Linear Sparse Autoencoder (LSAE). Kinase targets are mapped from UniProt accessions to 85-residue KLIFS pocket sequences, converted into Property Signature Method (PSM) physicochemical features, and embedded using Multiple Correspondence Analysis (MCA). The molecular and kinase representations are then concatenated and used by an XGBoost classifier.

KinSPEC also provides feature-level interpretation through TreeSHAP. Important molecular features can be mapped back to Morgan fingerprint bits and molecular substructures, while important kinase features can be mapped back to KLIFS pocket residues and residue-property categories.

The main workflow in this repository is provided as a Jupyter Notebook (`.ipynb`).

---

## Overview

KinSPEC uses the following modelling workflow:

1. Read compound-kinase interaction data.
2. Convert standardised SMILES strings into 4096-bit Morgan fingerprints with radius 2.
3. Embed molecular fingerprints using a molecular LSAE.
4. Map kinase UniProt accessions to KLIFS-compatible 85-residue binding-pocket sequences.
5. Represent each pocket residue using 23 binary PSM physicochemical properties.
6. Reorganise the kinase features into categorical residue-property variables.
7. Embed the kinase representation using MCA.
8. Concatenate the molecular and kinase features.
9. Train or load an XGBoost binary classifier.
10. Generate interaction probabilities, binary predictions, and optional TreeSHAP interpretations.

In the original KinSPEC framework:

- the molecular input contains 4096 Morgan fingerprint dimensions;
- the molecular LSAE contains 1024 latent dimensions;
- 466 molecular latent dimensions are retained for downstream modelling;
- each kinase pocket contains 85 aligned KLIFS positions;
- each residue is initially represented by 23 PSM properties;
- the kinase branch contains 425 categorical residue-property variables;
- 200 MCA components are retained;
- the final XGBoost input contains 666 features: 466 molecular features and 200 kinase features.

---

## Repository Structure

A recommended repository structure is shown below:

```text
KinSPEC/
├── README.md
├── KinSPEC.ipynb
├── requirements.txt
├── data/
│   ├── example_input.csv
│   ├── kinase_uniprot_mapping.csv
│   ├── klifs_pocket_sequences.csv
│   └── kinase_mca_features.csv
├── models/
│   ├── molecular_lsae/
│   ├── kinase_mca/
│   └── xgboost/
└── results/
```

The exact file names may differ from those shown above. Update the paths in the notebook configuration section before running the workflow.

---

## Input Data

The input dataset should contain one compound-target pair per row.

### Required columns

| Column | Description |
|---|---|
| `uniprot_id` | UniProt accession of the kinase target |
| `standard_smiles` | Standardised SMILES representation of the compound |
| `classification` | Binary interaction label: `1` for active and `0` for inactive |

Example training file:

```csv
uniprot_id,standard_smiles,classification
P00533,CCOc1ccc2nc(S(N)(=O)=O)sc2c1,1
P31749,CC(=O)Oc1ccccc1C(=O)O,0
P42336,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,1
```

The example labels above are illustrative and should not be treated as experimentally validated annotations.

### Column requirements

#### `uniprot_id`

- Use a UniProt accession rather than a gene symbol or protein name.
- The accession must be present in the kinase mapping file used by the notebook.
- The corresponding kinase must have a valid KLIFS-compatible 85-residue pocket representation.
- The current KinSPEC protein representation is designed for typical protein kinases rather than arbitrary protein families.

#### `standard_smiles`

- The value must contain a valid molecular structure that can be parsed by RDKit.
- Standardised or canonical SMILES are recommended.
- Salts, disconnected fragments, duplicated structures, and inconsistent protonation states should be handled during data preparation.
- Invalid SMILES entries should be corrected or removed before model training.

#### `classification`

Use integer binary labels:

- `1`: active compound-kinase interaction
- `0`: inactive compound-kinase interaction

For model training and evaluation, the `classification` column must not contain missing values.

The activity threshold used to define an active interaction depends on the source dataset. Therefore, continuous assay measurements should be converted into binary labels before they are provided to KinSPEC.

---

## Prediction-Only Input

For prediction on unlabelled compound-kinase pairs, the `classification` column may be omitted:

```csv
uniprot_id,standard_smiles
P00533,CCOc1ccc2nc(S(N)(=O)=O)sc2c1
P31749,CC(=O)Oc1ccccc1C(=O)O
```

When labels are available, they may still be included so that prediction performance can be evaluated.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/HaoyingWang1995/KinSPEC.git
cd KinSPEC
```

### 2. Create a virtual environment

Using `conda`:

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

```bash
pip install -r requirements.txt
```

The workflow uses packages for:

- numerical computing and data handling;
- molecular processing with RDKit;
- neural-network training;
- MCA;
- XGBoost classification;
- hyperparameter optimisation;
- SHAP interpretation;
- statistical evaluation and visualisation;
- Jupyter Notebook execution.

Core dependencies include:

```text
numpy
pandas
scipy
scikit-learn
rdkit
torch
xgboost
shap
optuna
prince
umap-learn
matplotlib
jupyter
```

For reproducibility, the exact package versions used in the experiments should be recorded in `requirements.txt` or an environment file.

---

## Running the Notebook

Start Jupyter Notebook from the repository directory:

```bash
jupyter notebook
```

Alternatively, start JupyterLab:

```bash
jupyter lab
```

Open:

```text
KinSPEC.ipynb
```

Before running the notebook:

1. Check that all required files are present.
2. Update the input, model, mapping, and output paths in the configuration section.
3. Select the correct Python environment as the Jupyter kernel.
4. Restart the kernel.
5. Run all cells from top to bottom.

Recommended Jupyter command:

```text
Kernel → Restart Kernel and Run All Cells
```

Jupyter notebooks retain variables and outputs from previous runs. Restarting the kernel before a complete experiment helps avoid hidden state and improves reproducibility.

---

## Recommended Notebook Organisation

The main notebook should be organised into the following sections.

### 1. Environment and package imports

Import all required Python packages and define the random seeds used by the workflow.

### 2. Configuration

Define all file paths and model settings in one location near the beginning of the notebook.

A typical configuration section may include:

```python
INPUT_FILE = "data/example_input.csv"
KINASE_MAPPING_FILE = "data/kinase_uniprot_mapping.csv"
KLIFS_SEQUENCE_FILE = "data/klifs_pocket_sequences.csv"

LSAE_MODEL_DIR = "models/molecular_lsae"
MCA_MODEL_DIR = "models/kinase_mca"
XGBOOST_MODEL_DIR = "models/xgboost"

OUTPUT_DIR = "results"
```

The exact variable names should match those used in the notebook.

### 3. Input validation

The notebook should verify that:

- all required columns are present;
- UniProt accessions are not missing;
- SMILES strings can be parsed by RDKit;
- classification values are binary;
- duplicate compound-target pairs are handled;
- all targets can be mapped to the kinase feature table.

### 4. Molecular feature generation

Each standardised SMILES string is converted into a 4096-bit Morgan fingerprint with radius 2.

The mapping between activated fingerprint bits and molecular substructures can also be recorded for downstream interpretation.

### 5. Molecular LSAE embedding

The notebook either trains or loads the molecular Linear Sparse Autoencoder.

The molecular LSAE:

- embeds the 4096-bit Morgan fingerprint;
- generates a 1024-dimensional latent representation;
- reconstructs the original fingerprint;
- supports reconstruction-based evaluation;
- allows latent dimensions to be traced back to fingerprint bits through the linear decoder.

The original KinSPEC model retains 466 molecular latent dimensions for downstream modelling.

### 6. Kinase feature generation

The target `uniprot_id` is mapped to a KLIFS-compatible kinase pocket sequence.

Each kinase is represented using:

- 85 structurally aligned KLIFS pocket positions;
- 23 binary PSM physicochemical properties per residue;
- an initial 1955-dimensional representation.

The PSM properties are grouped into five categories for each residue:

- side-chain type;
- functional properties;
- secondary-structure preference;
- tertiary-structure preference;
- residue size.

### 7. MCA kinase embedding

The 1955-dimensional binary PSM representation is reorganised into 425 categorical variables, corresponding to five residue-property variables for each of the 85 KLIFS positions.

MCA is then used to generate the kinase embedding. The original KinSPEC model retains the first 200 MCA components.

MCA column contributions should be saved when residue-level interpretation is required.

### 8. XGBoost model training

The retained molecular LSAE features and kinase MCA features are concatenated for each compound-kinase pair.

The resulting feature vector is used to train an XGBoost binary classifier.

For supervised training, the label is read from the `classification` column.

The trained classifier should be saved so that it can be reused for prediction without retraining.

### 9. Model evaluation

Depending on the experiment, the notebook may calculate:

- area under the receiver operating characteristic curve;
- area under the precision-recall curve;
- enrichment factor;
- Boltzmann-enhanced discrimination of receiver operating characteristic;
- precision;
- recall;
- confusion matrix;
- target-level or compound-level performance.

The evaluation metrics should be selected according to the intended use of the model and the class balance of the test set.

### 10. Model interpretation

TreeSHAP can be used to calculate feature contributions for individual compound-kinase predictions.

For molecular interpretation:

1. identify important molecular latent dimensions;
2. map them through the LSAE decoder;
3. rank the associated Morgan fingerprint bits;
4. map activated bits back to molecular substructures.

For kinase interpretation:

1. identify important MCA components;
2. map them through MCA column contributions;
3. identify the residue-property categories present in the query kinase;
4. aggregate the contributions at each KLIFS position;
5. rank the most important kinase pocket residues.

---

## Training a New Model

To train KinSPEC on a new labelled dataset:

1. Prepare a CSV file containing `uniprot_id`, `standard_smiles`, and `classification`.
2. Confirm that every kinase target is available in the mapping and KLIFS feature files.
3. Run the molecular preprocessing section.
4. Load the pretrained molecular LSAE or train a new LSAE.
5. Load the existing MCA transformation or fit MCA to the intended kinase feature set.
6. Concatenate molecular and kinase embeddings.
7. Split the data into training and validation sets.
8. Train the XGBoost classifier.
9. Evaluate the model.
10. Save the fitted models and output tables.

When comparing models, use the same data split and the same input pairs for every method.

---

## Predicting New Compound-Kinase Interactions

To predict new interactions:

1. Prepare a CSV file containing `uniprot_id` and `standard_smiles`.
2. Load the molecular LSAE.
3. Load the kinase MCA representation or transformation.
4. Load the trained XGBoost model.
5. Run the prediction cells.
6. Save the generated prediction scores.

A typical prediction output may contain:

| Column | Description |
|---|---|
| `uniprot_id` | Input UniProt accession |
| `standard_smiles` | Input standardised SMILES |
| `classification` | Experimental label, when provided |
| `prediction_score` | Predicted probability of an active interaction |
| `predicted_class` | Binary prediction derived from the selected classification threshold |

Example:

```csv
uniprot_id,standard_smiles,classification,prediction_score,predicted_class
P00533,CCOc1ccc2nc(S(N)(=O)=O)sc2c1,1,0.893,1
P31749,CC(=O)Oc1ccccc1C(=O)O,0,0.127,0
```

The prediction score can be used to:

- rank candidate compounds for a kinase;
- rank potential kinase targets for a compound;
- evaluate compound-specific kinase selectivity;
- prioritise compound-kinase pairs for experimental testing.

---

## Output Files

Depending on the notebook settings, KinSPEC may generate:

```text
results/
├── processed_input.csv
├── molecular_fingerprints.npy
├── molecular_embeddings.npy
├── kinase_embeddings.csv
├── prediction_results.csv
├── evaluation_metrics.csv
├── shap_values.npy
├── molecular_interpretation.csv
└── kinase_residue_interpretation.csv
```

The exact file names depend on the notebook implementation.

---

## Kinase Coverage

The original KinSPEC protein feature set covers 482 typical kinase entries.

The kinase representation uses:

- official KLIFS pocket sequences when available;
- inferred pocket sequences based on sequence or structural alignment when official KLIFS annotations are unavailable.

All kinase targets used for prediction must be represented in the kinase mapping file supplied with the repository.

Predictions for proteins outside the supported kinase set require a compatible 85-residue KLIFS pocket mapping and should not be performed by simply substituting a full-length protein sequence.

---

## Reproducibility

For reproducible results:

- use the package versions recorded in the repository;
- keep all random seeds fixed;
- restart the notebook kernel before each complete run;
- run notebook cells in order;
- preserve the original train-validation-test separation;
- use the same molecular standardisation procedure;
- use the same UniProt-to-KLIFS mapping;
- use the same LSAE, MCA, and XGBoost model files;
- avoid modifying intermediate feature files manually;
- record the data source, download date, and preprocessing settings.

Large model or data files should be accompanied by checksums where possible.

---

## Important Notes

- KinSPEC is a binary compound-kinase interaction classifier.
- The current protein representation is specific to typical protein kinases.
- UniProt accessions must match the kinase mapping table.
- Gene symbols cannot directly replace UniProt accessions.
- Invalid SMILES strings cannot be converted into Morgan fingerprints.
- Classification labels must be encoded as `0` and `1`.
- The classification threshold used to create labels should be documented.
- Duplicate compound-target pairs should be removed or resolved before training.
- Predictions should be interpreted as model estimates rather than experimental validation.
- TreeSHAP importance indicates contribution to the model prediction and does not by itself demonstrate a physical molecular interaction.

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

Repository:

```text
https://github.com/HaoyingWang1995/KinSPEC
```

Test data:

```text
https://zenodo.org/uploads/20558270
```

---

## Contact

For questions about KinSPEC, please contact:

**Haoying Wang**  
University of Edinburgh  
Email: H.Wang-138@sms.ed.ac.uk

**Douglas R. Houston**  
University of Edinburgh  
Email: DouglasR.Houston@ed.ac.uk
