"""The asymmetry of 0 in one-hot PSM, made measurable.

XGBoost sees 1955 independent binary features. It has no knowledge that the 23 bits at a
KLIFS position form 5 mutually exclusive groups. Every split is therefore `bit_j < 0.5`:
  the 1-branch  = one homogeneous chemical state  ("position p is Aromatic")
  the 0-branch  = the disjunction of every OTHER level of that variable
Only the 1-branch names a chemical state; the 0-branch is a complement whose size and
composition depend on how many levels that variable has.
"""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this archived analysis with shared paths; import performs no calculation."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print("Archived analysis: interpret results with the original report and its limitations.")
    import numpy as np, pandas as pd, xgboost as xgb, json, collections

    F = str(config.final_dir); EMB = f"{F}/feature_embedding/Kinase_test_MCA"
    PSMM = str(config.psm_model_dir)
    OUT = str(config.output_dir)

    psm = pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv", header=None).to_numpy()
    prev = psm.mean(0)
    st = np.load(f"{OUT}/strata.npz", allow_pickle=True)
    vres, vsys = st["res"], st["sys"]
    lk = pd.read_csv(f"{EMB}/MCA_200D_category_lookup.csv").dropna(subset=["bit_index_0_based"])
    lk["bit"] = lk.bit_index_0_based.astype(int)
    r1 = lk[lk.residue == 1].sort_values("bit")
    bit2sys = dict(zip(r1.bit, r1.system)); bit2cat = dict(zip(r1.bit, r1.category))
    vidx = {(r, s): i for i, (r, s) in enumerate(zip(vres, vsys))}
    col_var = np.array([vidx[(1 + c // 23, bit2sys[c % 23])] for c in range(1955)])
    # how many levels are actually observed in each variable
    nlev = np.zeros(425, int)
    for v in range(425):
        cols = np.where(col_var == v)[0]
        obs = int(((psm[:, cols].mean(0) > 0)).sum())
        gap = int((psm[:, cols].sum(1) == 0).any())
        nlev[v] = obs + gap

    print("=" * 86)
    print("[1] what a 0-branch lumps together, per property system")
    print("=" * 86)
    print(f"  {'system':<12}{'levels/variable':>17}{'columns':>10}")
    for s in sorted(set(bit2sys.values())):
        bits = [b for b in bit2sys if bit2sys[b] == s]
        vs = [v for v in range(425) if vsys[v] == s]
        print(f"  {s:<12}{np.mean(nlev[vs]):>17.1f}{len(bits)*85:>10}")
    print(f"\n  a split `bit<0.5` on a variable with L observed levels puts ONE state on the 1-side")
    print(f"  and the disjunction of the other L-1 on the 0-side.")

    print("\n" + "=" * 86)
    print("[2] how the trained pure-PSM model actually uses this  (band 10-50, 5 folds)")
    print("=" * 86)
    tot_g = np.zeros(1955); tot_w = np.zeros(1955)
    for f in range(1, 6):
        b = xgb.Booster(); load_model(b, f"{PSMM}/KinSPEC_MCA_training_band_10_50_fold{f}_pure_PSM_model_v4.json", config)
        for imp, acc in (("total_gain", tot_g), ("weight", tot_w)):
            g = b.get_score(importance_type=imp)
            for k, v in g.items():
                i = int(str(k).lstrip("f"))
                if i >= 417: acc[i - 417] += v
        del b
    used = tot_w > 0
    w = tot_g / tot_g.sum()
    alt = nlev[col_var] - 1                       # states lumped on the 0-side
    print(f"  kinase columns ever used as a split: {used.sum()} of 1955")
    print(f"  gain-weighted mean number of states lumped on a 0-branch : {(w*alt).sum():.2f}")
    print(f"  split-count-weighted                                     : "
          f"{(tot_w/tot_w.sum()*alt).sum():.2f}")
    print(f"  gain-weighted mean prevalence of the split bit           : {(w*prev).sum():.3f}")
    print(f"  -> the 1-side holds {(w*prev).sum()*100:.1f}% of kinases on average;")
    print(f"     the remaining {100-(w*prev).sum()*100:.1f}% land on a 0-branch that names no single state.")
    print(f"\n  distribution of splits by how much of the kinome the 0-branch absorbs:")
    for lo, hi in ((0, .5), (.5, .8), (.8, .95), (.95, 1.0)):
        m = used & (1 - prev >= lo) & (1 - prev < hi)
        print(f"    0-branch holds {lo*100:>4.0f}-{hi*100:>3.0f}% of kinases: "
              f"{m.sum():>5} columns, {w[m].sum()*100:>5.1f}% of kinase gain")

    print("\n" + "=" * 86)
    print("[3] does the model ever need several splits to name one chemical state?")
    print("=" * 86)
    b = xgb.Booster(); load_model(b, f"{PSMM}/KinSPEC_MCA_training_band_10_50_fold1_pure_PSM_model_v4.json", config)
    dump = b.get_dump(dump_format="json")
    same_var_paths = 0; paths = 0; depth_hist = collections.Counter()
    def walk(n, hist):
        nonlocal same_var_paths, paths
        if "split" not in n:
            paths += 1
            c = collections.Counter(hist)
            if any(v >= 2 for v in c.values()): same_var_paths += 1
            depth_hist[len(hist)] += 1
            return
        f = int(str(n["split"]).lstrip("f"))
        for ch in n.get("children", []):
            walk(ch, hist + ([col_var[f - 417]] if f >= 417 else []))
    for t in dump[:400]: walk(json.loads(t), [])
    print(f"  leaves examined: {paths:,} (400 trees)")
    print(f"  root-to-leaf paths testing the SAME (position, system) variable more than once: "
          f"{same_var_paths:,} ({same_var_paths/paths*100:.1f}%)")
    print("  (each such path is the model spending several splits to narrow down one variable --")
    print("   the cost of not knowing that its levels are mutually exclusive)")


if __name__ == "__main__":
    single_main("zero_ambiguity")
