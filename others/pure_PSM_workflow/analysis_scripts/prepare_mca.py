"""Reconstruct MCA indicator/projection arrays from archived CSV inputs."""
try:
    from .workflow import AnalysisConfig, single_main
except ImportError:
    from workflow import AnalysisConfig, single_main


def build_indicator(psm, lookup, categories):
    """Map binary property bits and explicit gaps into the stored category order."""
    import numpy as np
    import pandas as pd

    psm = np.asarray(psm)
    if psm.ndim != 2 or not np.isin(psm, [0, 1]).all():
        raise ValueError("PSM must be a two-dimensional binary matrix")
    normalized = lookup.category_name.str.replace("_", "", regex=False)
    if normalized.duplicated().any():
        raise ValueError("Category lookup is ambiguous after normalization")
    by_name = lookup.set_index(normalized)
    indicator = np.zeros((len(psm), len(categories)), dtype=np.float64)
    groups = {}
    for column, category in enumerate(categories):
        key = str(category).replace("_", "")
        if key not in by_name.index:
            raise ValueError(f"Category missing from lookup: {category}")
        row = by_name.loc[key]
        group = (int(row.residue), str(row.system))
        groups.setdefault(group, []).append(column)
        if pd.notna(row.bit_index_0_based):
            bit = int(row.bit_index_0_based)
            if bit < 0 or bit >= psm.shape[1]:
                raise ValueError(f"PSM bit index out of range: {bit}")
            indicator[:, column] = psm[:, bit]
        else:
            if str(row.category).lower() != "gap":
                raise ValueError(f"Unmapped category is not an explicit gap: {category}")
            bits = lookup.loc[(lookup.residue == group[0]) & (lookup.system == group[1]), "bit_index_0_based"].dropna().astype(int).to_numpy()
            if not len(bits) or (bits < 0).any() or (bits >= psm.shape[1]).any():
                raise ValueError(f"Invalid property-bit mapping for {group}")
            indicator[:, column] = (psm[:, bits].sum(axis=1) == 0)
    for group, columns in groups.items():
        if not np.all(indicator[:, columns].sum(axis=1) == 1):
            raise ValueError(f"Category states are not mutually exclusive and complete for {group}")
    return indicator, len(groups)


def run(config=None):
    config = config or AnalysisConfig()
    import numpy as np
    import pandas as pd

    e = config.embedding_dir
    psm = pd.read_csv(e / "human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy()
    lookup = pd.read_csv(e / "MCA_200D_category_lookup.csv")
    coords = pd.read_csv(e / "MCA_200D_category_coordinates.csv", index_col=0)
    Z, Q = build_indicator(psm, lookup, coords.index)
    if Z.shape[0] != 482 or Q != 425:
        raise ValueError(f"Expected 482 kinases and 425 variables, received {Z.shape[0]}, {Q}")
    spectrum = pd.read_csv(e / "MCA_inertia_spectrum.csv")
    axes = [int(str(c).removeprefix("mca_")) for c in coords.columns]
    if spectrum.n_components.duplicated().any():
        raise ValueError("Duplicate axes in the MCA inertia spectrum")
    lam = spectrum.set_index("n_components").loc[axes, "raw_eigenvalue"].to_numpy(float)
    G = coords.to_numpy(float)
    if G.shape[1] != 200 or not np.isfinite(G).all() or not np.isfinite(lam).all() or (lam <= 0).any():
        raise ValueError("Expected 200 finite projection axes with positive raw eigenvalues")
    projected = (Z @ G) / (Q * np.sqrt(lam))
    stored = pd.read_csv(e / "MCA_klifs_482_200D_features.csv", skiprows=1, header=None).to_numpy(float)
    if projected.shape != stored.shape:
        raise ValueError("MCA projection dimensions do not match the stored features")
    residual = float(np.abs(projected - stored).max())
    if residual >= 1e-4:
        raise ValueError(f"Reconstructed MCA projection does not match stored coordinates: {residual}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in (("Z", Z), ("G", G), ("lam", lam)):
        np.save(config.output_dir / f"{name}.npy", array)
    print(f"Prepared Z {Z.shape}, G {G.shape}, lam {lam.shape}; projection max error {residual:.3e}")
    return {"variables": Q, "projection_max_error": residual}


if __name__ == "__main__":
    single_main("prepare_mca")
