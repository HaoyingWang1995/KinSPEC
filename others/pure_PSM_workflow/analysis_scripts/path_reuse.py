"""How often does a single root-to-leaf path test the same residue-property variable twice?

Supplementary Method S2.4. If the zero-valued branch of a one-hot split leaves the state
undetermined, a tree might be expected to revisit the same variable in order to narrow it down.
This script measures how often that happens in the independently optimised full-training-pool
raw-PSM model, and shows that the cost of the asymmetry lies in readability rather than in tree
capacity.

Self-contained. Derives the column-to-variable mapping from the KLIFS lookup table and needs no
intermediate files from other scripts. `zero_ambiguity` reports an earlier, related count on the
10-50% band fold-1 model (400 trees); the manuscript value comes from this script.

Reported in the manuscript: 761 of 205,428 root-to-leaf paths (0.4%) across the first 600 trees.
"""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

N_MOL = 417          # molecular LSAE block precedes the kinase block
N_TREES = 600        # sampled to bound computation; the model holds 4180 trees


def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import collections, json
    import numpy as np, pandas as pd, xgboost as xgb
    EMB = str(config.embedding_dir)
    PSMM = str(config.psm_model_dir / "KinSPEC_full_training_pool_pure_PSM_v4.json")

    # column index 0..1954 -> variable index 0..424, where a variable is (KLIFS position, property group)
    lk = pd.read_csv(f"{EMB}/MCA_200D_category_lookup.csv").dropna(subset=["bit_index_0_based"])
    lk["bit"] = lk.bit_index_0_based.astype(int)
    first = lk[lk.residue == 1].sort_values("bit")
    bit2sys = dict(zip(first.bit, first.system))
    systems = sorted(set(bit2sys.values()))
    vidx = {(r, s): i for i, (r, s) in enumerate((r, s) for r in range(1, 86) for s in systems)}
    col_var = np.array([vidx[(1 + c // 23, bit2sys[c % 23])] for c in range(1955)])
    print(f"variables = {len(vidx)} (85 positions x {len(systems)} property groups)")

    b = xgb.Booster(); load_model(b, PSMM, config)
    dump = b.get_dump(dump_format="json")
    n_total = len(dump)
    print(f"model holds {n_total} trees, sampling the first {N_TREES}")
    del b

    paths = reuse = 0
    def walk(node, seen):
        nonlocal paths, reuse
        if "split" not in node:
            paths += 1
            if any(v >= 2 for v in collections.Counter(seen).values()):
                reuse += 1
            return
        f = int(str(node["split"]).lstrip("f"))
        nxt = seen + ([col_var[f - N_MOL]] if f >= N_MOL else [])
        for child in node.get("children", []):
            walk(child, nxt)

    for t in dump[:N_TREES]:
        walk(json.loads(t), [])

    print(f"root-to-leaf paths              {paths:,}")
    print(f"paths revisiting a variable     {reuse:,}  ({reuse / paths * 100:.1f}%)")
    result = {"model": PSMM, "trees_total": n_total, "trees_sampled": N_TREES,
              "variables": len(vidx), "paths": paths, "paths_revisiting_variable": reuse,
              "fraction": reuse / paths}
    (config.output_dir / "path_reuse.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    single_main("path_reuse")
