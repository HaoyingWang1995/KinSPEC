"""Rebuild categorical variable strata directly from archived PSM/MCA inputs."""
try:
    from .workflow import AnalysisConfig, single_main
except ImportError:
    from workflow import AnalysisConfig, single_main


def run(config=None):
    config = config or AnalysisConfig()
    import numpy as np
    import pandas as pd

    e = config.embedding_dir
    psm = pd.read_csv(e / "human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy()
    categories = pd.read_csv(e / "MCA_200D_category_coordinates.csv", index_col=0).index
    lookup = pd.read_csv(e / "MCA_200D_category_lookup.csv")
    normalized = lookup.category_name.str.replace("_", "", regex=False)
    if normalized.duplicated().any():
        raise ValueError("Category names are not unique after normalization")
    by_name = {key: row for key, (_, row) in zip(normalized, lookup.iterrows())}
    names, var_of = [], []
    for category in categories:
        residue = int(category.split("_")[0][3:])
        system = category.split("_")[1]
        key = (residue, system)
        if key not in names:
            names.append(key)
        var_of.append(names.index(key))
    if len(names) != 425 or psm.shape != (482, 1955):
        raise ValueError(f"Unexpected PSM/category dimensions: {psm.shape}, {len(names)} variables")
    var_of = np.asarray(var_of)
    # Reconstruct category frequencies, including explicit gaps, without external Z.npy.
    freq = np.zeros(len(categories))
    for i, category in enumerate(categories):
        key = category.replace("_", "")
        if key not in by_name:
            raise ValueError(f"Category missing from lookup: {category}")
        row = by_name[key]
        bit = row.bit_index_0_based
        if pd.notna(bit):
            freq[i] = psm[:, int(bit)].mean()
        else:
            variable = names[var_of[i]]
            bits = lookup.loc[(lookup.residue == variable[0]) & (lookup.system == variable[1]), "bit_index_0_based"].dropna().astype(int)
            freq[i] = (psm[:, bits].sum(axis=1) == 0).mean()
    ent = np.zeros(len(names))
    for i in range(len(names)):
        probabilities = freq[var_of == i]
        if not np.isclose(probabilities.sum(), 1):
            raise ValueError(f"Invalid mutually exclusive category frequencies for {names[i]}")
        probabilities = probabilities[probabilities > 0]
        ent[i] = -(probabilities * np.log2(probabilities)).sum()
    order = np.argsort(ent)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "strata.npz"
    np.savez(path, var_of=var_of, ent=ent, shared=order[:len(names) // 2], discrim=order[len(names) // 2:], res=np.array([x[0] for x in names]), sys=np.array([x[1] for x in names]))
    print(f"Prepared {len(names)} categorical variables: {path}")
    return path


if __name__ == "__main__":
    single_main("prepare_encoding")
