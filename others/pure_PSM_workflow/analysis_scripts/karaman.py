try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def paired_summary(x, y):
    """Summarize only pairs for which both metrics are defined."""
    import numpy as np
    from scipy import stats
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Paired metrics must have the same shape")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if not len(x):
        return {"psm": np.nan, "mca": np.nan, "diff": np.nan, "wins": 0, "n": 0, "p": np.nan}
    p = 1.0 if np.all(y == x) else float(stats.wilcoxon(y, x).pvalue)
    return {"psm": float(x.mean()), "mca": float(y.mean()), "diff": float((y-x).mean()),
            "wins": int((y>x).sum()), "n": len(x), "p": p}

def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd, numpy as np
    from scipy import stats
    P = str(config.project_dir) + '/Karaman_pure_PSM_v4_result/kinspec_mac_karaman_per_molecule_all_folds.csv'
    M = str(config.final_dir) + '/model_training/Karaman_MAC_v4_result/kinspec_mac_karaman_per_molecule_all_folds.csv'
    a = pd.read_csv(P); b = pd.read_csv(M)
    key = ["molecule_name", "fold"]
    d = a.merge(b, on=key, suffixes=("_psm", "_mca"), how="outer", validate="one_to_one", indicator=True)
    if not (d._merge == "both").all():
        raise ValueError("PSM and MCA result keys do not match")
    print(f"paired molecule x fold observations: {len(d)}  "
          f"({d.molecule_name.nunique()} molecules x {d.fold.nunique()} folds)")
    mets = ["AUROC", "BEDROC_a20", "Spearman_full", "Spearman_measured",
            "P@5", "NDCG@5", "P@10", "NDCG@10"]

    summaries = []
    def block(sub, title):
        print("\n" + "=" * 86); print(title + f"   n={len(sub)}"); print("=" * 86)
        print(f"{'metric':<20}{'pure PSM':>11}{'MCA':>11}{'diff':>10}{'MCA wins':>11}{'p(wilcox)':>12}")
        for m in mets:
            s = paired_summary(sub[f"{m}_psm"], sub[f"{m}_mca"])
            summaries.append({"scope": title, "metric": m, **s})
            print(f"{m:<20}{s['psm']:11.4f}{s['mca']:11.4f}{s['diff']:+10.4f}"
                  f"{s['wins']:>7}/{s['n']:<4}{s['p']:12.2e}")

    block(d, "Karaman kinase-selectivity: rank the 271 kinases for a fixed molecule (ALL)")
    for g in ["selective", "moderate", "promiscuous"]:
        block(d[d.group_psm == g], f"group = {g}")

    # per-molecule aggregation over folds, to remove within-molecule fold correlation
    print("\n" + "=" * 86)
    print("collapsed to one value per molecule (mean over the 5 folds)")
    print("=" * 86)
    g = d.groupby("molecule_name")[[f"{m}_{s}" for m in mets for s in ("psm", "mca")]].mean()
    block(g, "per-molecule mean over folds (paired complete metrics)")
    pd.DataFrame(summaries).to_csv(config.output_dir / "karaman_summary.csv", index=False)


if __name__ == "__main__":
    single_main("karaman")
