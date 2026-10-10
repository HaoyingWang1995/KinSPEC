"""Paths shared by the corrected KinSPEC scripts and notebooks.

ROOT follows this file when the project is moved. All default paths stay inside
the project; callers can still explicitly supply their own command-line paths.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent

ALIASES = {
    'VSDS_truedecoy_integrated_data.csv': 'VSDS_truedecoy_integrated_data_with_protein_seq.csv',
    'KinSPEC_training_pool_merged_4_datasets_deduplicated_with_smiles.csv': 'train_ready2use_merged_4datasets_deduplicated_with_smiles.csv',
    'KIBA_unseen_active_filter_LUDe_augment_sim08.csv': 'KIBA_unseen_LUDe_augment_sim08.csv',
    'KIBA_unseen_active_filter_LUDe_augmented_sim08_LSAE_latent_top95_features_v4.csv': 'KIBA_unseen_LUDe_augmented_sim08_LSAE_latent_top95_features_v4.csv',
    'xgboost_SPECTRA_ready_to_use_with_VSDS_482_kinases_4096bit_morgan.json': 'xgboost_SPECTRA_ready_to_use_with_VSDS_482_kinases_4096bit_morgan_2.json',
}


def model_group(name):
    if 'rawmorgan' in name or '4096bit_morgan' in name:
        return 'Morgan_raw'
    if 'pure_PSM' in name:
        return 'PSM_raw'
    if 'MCA_onehot' in name or 'onehot_MCA' in name:
        return 'onehot_MCA'
    if 'pure_onehot' in name:
        return 'onehot_raw'
    if name.endswith('_default_model_v4.json'):
        return 'KinSPEC_default'
    return 'KinSPEC'


def project_path(value, script_dir=''):
    """Resolve a project-relative string, including dynamic band filenames.

    The trailing slash is preserved for older code that concatenates filenames.
    This function does not open files, create directories or download resources.
    """
    value = str(value)
    trailing = value.endswith('/')
    path = Path(value)
    if path.is_absolute():
        try:
            value = path.relative_to(ROOT).as_posix()
        except ValueError:
            # Explicit caller paths are permitted; code defaults use ROOT.
            return str(path)
    name = Path(value).name
    renamed = ALIASES.get(name, name)
    if name != renamed:
        value = str(Path(value).with_name(renamed))
        name = renamed
    match = re.search(r'(?:spectrum_data/|spectra_data_split_from_merged/)?band_(\d+\.\d+)_(\d+\.\d+)/(train(?:_LSAE_latent_top95_features_v4|_morgan_one_hot|_with_smiles_protein_seq)?\.csv)$', value)
    if match and 'data_split_from_merged_true_fold/' not in value:
        band = f'{round(float(match[1])*100)}_{round(float(match[2])*100)}'
        suffix = {'train.csv':'with_smiles_protein_seq.csv', 'train_with_smiles_protein_seq.csv':'with_smiles_protein_seq.csv'}.get(match[3], match[3][6:])
        value = f'dataset/train_band_{band}_{suffix}'
    elif name.startswith(('KinSPEC_', 'xgboost_SPECTRA_')) and name.endswith(('.json', '.joblib')):
        value = f'model/{model_group(name)}/{name}'
    elif name == 'human_482_typical_kinase.csv':
        value = 'feature_embedding/Kinase_test_MCA/human_482_typical_kinase_1955bit.csv'
    elif name == 'onehot_MCA_klifs_482_200D_features.csv':
        value = 'others/one_hot_embedding_test/' + name
    elif name in ALIASES.values():
        value = 'dataset/' + name
    # All rewritten defaults are relative to ROOT, rather than to the kernel CWD.
    result = (ROOT / value).resolve()
    if ROOT != result and ROOT not in result.parents:
        raise ValueError(f'Project default escapes the project directory: {value}')
    return str(result) + ('/' if trailing else '')
