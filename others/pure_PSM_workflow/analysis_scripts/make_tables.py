"""Export nine fixed archived SI tables; no new analysis is performed."""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this archived analysis with shared paths; import performs no calculation."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print("Exporting fixed archived table values; these are not recomputed results.")
    import pandas as pd, os
    OUT = str(config.output_dir / "archived_SI_tables")
    os.makedirs(OUT, exist_ok=True)
    w = lambda d, n: pd.DataFrame(d).to_csv(f"{OUT}/{n}.csv", index=False, encoding="utf-8-sig")

    w({
     "Parameter": ["learning_rate","max_depth","subsample","colsample_bytree","lambda","alpha",
       "gamma","min_child_weight","scale_pos_weight","Optimisation trials",
       "Internal cross-validation","Objective","Early stopping rounds",
       "Boosting rounds at best iteration","Total input features"],
     "Search space": ["0.01 to 0.3, log scale","5 to 10","0.3 to 0.9","0.2 to 0.8",
       "1e-3 to 1e3, log scale","1e-3 to 300, log scale","0 to 10","1 to 12",
       "fixed","fixed","fixed","fixed","fixed","selected",""],
     "Raw PSM": ["0.013558","10","0.877554","0.555709","24.927","0.868181","2.732379","10",
       "14","200","5-fold","AUROC","50","4180","2372"],
     "MCA": ["0.022587","10","0.863729","0.667657","640.180","0.079190","1.149077","8",
       "14","200","5-fold","AUROC","50","6972","617"],
    }, "Table_SM2_hyperparameters")

    w({
     "Dataset": ["VSDS-VD"]*7 + ["Karaman"]*2,
     "Metric": ["AUROC","Within-target AUROC","AUPRC","BEDROC, alpha = 20","BEDROC, alpha = 80.5",
       "EF@1%","EF@5%","AUROC, per compound","BEDROC alpha = 20, per compound"],
     "Raw PSM": [0.6481,0.6493,0.0633,0.1727,0.1426,6.18,3.04,0.8698,0.6918],
     "MCA": [0.6567,0.6572,0.0590,0.1740,0.1249,5.68,2.92,0.8661,0.6834],
     "Difference": [0.0086,0.0079,-0.0043,0.0013,-0.0177,-0.50,-0.12,-0.0036,-0.0084],
    }, "Table_SM3_performance")

    w({
     "Property": ["Kinase reference set","Invariant columns","Non-constant columns",
       "Columns perfectly collinear with another column, absolute r at least 0.999",
       "Columns with absolute r above 0.95","Matrix rank",
       "Principal components for 95% of variance","Principal components for 99% of variance",
       "Collinear groups confined to a single KLIFS position",
       "Kinase block after deduplication","VSDS-VD pooled AUROC, original encoding",
       "VSDS-VD pooled AUROC, deduplicated encoding","Paired t-test p value"],
     "Value": ["482 kinases by 1955 PSM columns","132","1823","265 in 115 groups","459","480",
       "210","312","101 of 115","1673 features","0.6057","0.6056","0.88"],
    }, "Table_SM4_redundancy")

    w({
     "Correlation with nearest column": ["Invariant","Below 0.5","0.5 to 0.8","0.8 to 0.95",
       "0.95 to below 1.0","Exactly 1.0"],
     "Columns": [132,156,649,559,194,265],
     "Share of columns (%)": [6.8,8.0,33.2,28.6,9.9,13.6],
     "Share of attribution (%)": [0.0,5.9,42.8,31.3,15.3,4.7],
     "Enrichment": [0.00,0.74,1.29,1.09,1.55,0.35],
    }, "Table_SM5_attribution_by_correlation")

    w({
     "Weighting": ["Mean absolute TreeSHAP","Total gain","Split frequency","Unweighted"],
     "Alternative states on the zero branch": [4.50,4.52,4.58,4.63],
     "Mean prevalence of the encoded state": [0.319,0.334,0.326,0.258],
     "Kinases on the zero branch (%)": [68.1,66.6,67.4,74.2],
     "Attribution from features with 50 to 95% of kinases on the zero branch (%)":
       [72.0,71.4,72.8,52.6],
    }, "Table_SM6_zero_branch_by_weighting")

    w({
     "Fraction of kinases on the zero branch": ["Below 50%","50 to 80%","80 to 95%","Above 95%"],
     "Features": [297,346,472,441],
     "Mean absolute TreeSHAP (%)": [21.7,39.5,32.5,6.3],
     "Total gain (%)": [23.4,40.5,30.9,5.3],
     "Split frequency (%)": [22.3,40.3,32.5,4.9],
    }, "Table_SM7_zero_branch_distribution")

    w({
     "Quantity": ["Mean cross-kinase Jaccard overlap of top-10 signatures",
       "Permutation floor for the same representation","Difference from own floor",
       "Distinct top-10 signatures","KLIFS positions represented in any top-10 signature"],
     "Raw PSM": ["0.182","0.065","+0.116","299 of 300","74 of 85"],
     "MCA": ["0.112","0.123","-0.011","300 of 300","83 of 85"],
    }, "Table_SM8_cross_kinase_overlap")

    w({
     "Comparison": ["Same kinase across the two datasets","Randomly mismatched kinase pairs",
       "Ratio to the random-pair control"],
     "Raw PSM": [0.516,0.164,3.1],
     "MCA": [0.426,0.145,2.9],
    }, "Table_SM9_cross_dataset")

    w({
     "Kinase representation": ["Raw PSM","MCA","Effect of changing the representation"],
     "Raw-PSM parameter set": ["0.6057 ± 0.0033","0.6038 ± 0.0012","-0.0018"],
     "MCA parameter set": ["0.6210 ± 0.0042","0.6196 ± 0.0023","-0.0014"],
     "Effect of changing the parameter set": ["+0.0154","+0.0157","interaction +0.0004"],
    }, "Table_SM10_2x2_sensitivity")

    for f in sorted(os.listdir(OUT)):
        print(f"  {f}  ({len(pd.read_csv(f'{OUT}/{f}'))} rows)")


if __name__ == "__main__":
    single_main("make_tables")
