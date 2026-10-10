"""Kinase-side shortcut analysis: pure PSM vs KLIFS-MCA.
Analogue of the Morgan(radius=0) -> LSAE study, with
  'radius'      -> PSM bit coarseness (how many of 20 AA set that property bit)
  'promiscuous' -> how many of the 482 kinases carry that (position,bit)
"""

try:
    from .workflow import AnalysisConfig, load_model, single_main
except ImportError:
    from workflow import AnalysisConfig, load_model, single_main

def run(config=None):
    """Run this analysis with shared paths; importing the module does not run it."""
    config = config or AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np, pandas as pd, xgboost as xgb, json, os
    from scipy import stats

    FINAL=str(config.final_dir); EMB=f"{FINAL}/feature_embedding/Kinase_test_MCA"
    PSMM=str(config.psm_model_dir); MCAM=f"{FINAL}/model"
    OUT=str(config.output_dir)
    BANDS=["10_50","20_60","30_70","40_80","50_90","60_100"]; NMOL=417

    meta=json.load(open(f"{OUT}/aa23.json")); spec=np.array(meta["spec"]); names=meta["names"]
    psm=pd.read_csv(f"{EMB}/human_482_typical_kinase_1955bit_onehot.csv",header=None).to_numpy(np.float32)
    prev=psm.mean(0)                                   # prevalence over the 482 kinases
    bit_of=np.tile(np.arange(23),85)                   # which property bit each of the 1955 columns is
    res_of=np.repeat(np.arange(1,86),23)
    coarse=spec[bit_of]                                # 'radius' analogue per column

    print("="*94)
    print("[S0] strata definitions")
    print("="*94)
    print(f"  PSM columns 1955 = 85 residues x 23 property bits")
    print(f"  coarse bits (set by >=7 of 20 AA): {sorted(set(np.array(names)[spec>=7]))}")
    print(f"  fine   bits (set by <=2 of 20 AA): {len(np.array(names)[spec<=2])} bits")
    q=np.array([0,.01,.05,.20,1.01])
    lab=np.array(["rare<1%","uncommon1-5%","common5-20%","promisc>20%"])
    strat=np.digitize(prev,[.01,.05,.20])
    for i,l in enumerate(lab):
        m=strat==i
        print(f"    {l:<14} {m.sum():>5} cols   mean coarseness {coarse[m].mean():.2f}/20   "
              f"const(prev 0 or 1): {((prev[m]==0)|(prev[m]==1)).sum()}")
    print(f"  Spearman(prevalence, coarseness) over 1955 cols = {stats.spearmanr(prev,coarse).statistic:+.3f}")

    rows=[]
    print()
    print("="*94)
    print("[S1] where the kinase-side gain sits   (total_gain, per fold)")
    print("="*94)
    for band in BANDS:
        for fold in range(1,6):
            p=f"{PSMM}/KinSPEC_MCA_training_band_{band}_fold{fold}_pure_PSM_model_v4.json"
            m=f"{MCAM}/KinSPEC_MCA_training_band_{band}_fold{fold}_model_v4.json"
            r={"band":band,"fold":fold}
            for tag,path,nk in (("PSM",p,1955),("MCA",m,200)):
                b=xgb.Booster(); load_model(b, path, config)
                g=b.get_score(importance_type="total_gain")
                v=np.zeros(NMOL+nk)
                for k,val in g.items(): v[int(k)]=val
                mol,kin=v[:NMOL],v[NMOL:]
                r[f"{tag}_kin_share"]=kin.sum()/v.sum()
                r[f"{tag}_kin_used"]=int((kin>0).sum())
                r[f"{tag}_kin_frac_used"]=(kin>0).sum()/nk
                # effective number of features = exp(entropy of the gain distribution)
                w=kin/kin.sum(); w=w[w>0]
                r[f"{tag}_kin_eff_n"]=float(np.exp(-(w*np.log(w)).sum()))
                r[f"{tag}_kin_eff_frac"]=r[f"{tag}_kin_eff_n"]/nk
                s=np.sort(kin)[::-1]
                r[f"{tag}_kin_top10"]=s[:10].sum()/kin.sum()
                r[f"{tag}_kin_top1pct"]=s[:max(1,nk//100)].sum()/kin.sum()
                if tag=="PSM":
                    kg=kin/kin.sum()
                    # coarseness enrichment of the top gain columns
                    base=coarse.mean()
                    for k in (10,20,50):
                        top=np.argsort(kin)[::-1][:k]
                        r[f"PSM_top{k}_coarse"]=coarse[top].mean()
                        r[f"PSM_top{k}_coarse_enr"]=coarse[top].mean()/base
                        r[f"PSM_top{k}_prev"]=float(np.median(prev[top]))
                    # gain share by prevalence stratum and by coarse/fine
                    for i,l in enumerate(lab): r[f"PSM_gain_{l}"]=kg[strat==i].sum()
                    r["PSM_gain_coarse>=7"]=kg[coarse>=7].sum()
                    r["PSM_gain_frac_cols_coarse>=7"]=float((coarse>=7).mean())
                    r["PSM_gain_fine<=2"]=kg[coarse<=2].sum()
                    r["PSM_gain_frac_cols_fine<=2"]=float((coarse<=2).mean())
                    r["PSM_spearman_gain_prev"]=stats.spearmanr(kin,prev).statistic
                    r["PSM_spearman_gain_coarse"]=stats.spearmanr(kin,coarse).statistic
                del b
            rows.append(r)
        print(f"  done {band}")
    df=pd.DataFrame(rows); df.to_csv(f"{OUT}/mech_gain.csv",index=False)

    print()
    print("  kinase block share of total gain:")
    print(f"    PSM {df.PSM_kin_share.mean()*100:.1f}% ± {df.PSM_kin_share.std()*100:.1f}   "
          f"MCA {df.MCA_kin_share.mean()*100:.1f}% ± {df.MCA_kin_share.std()*100:.1f}")
    print("  concentration of the kinase block (effective #features = exp(entropy of gain)):")
    print(f"    PSM {df.PSM_kin_eff_n.mean():7.1f} of 1955 ({df.PSM_kin_eff_frac.mean()*100:5.1f}% of the block)")
    print(f"    MCA {df.MCA_kin_eff_n.mean():7.1f} of  200 ({df.MCA_kin_eff_frac.mean()*100:5.1f}% of the block)")
    print(f"  top-10 features hold:  PSM {df.PSM_kin_top10.mean()*100:5.1f}%   MCA {df.MCA_kin_top10.mean()*100:5.1f}%")
    print(f"  top-1% features hold:  PSM {df.PSM_kin_top1pct.mean()*100:5.1f}%   MCA {df.MCA_kin_top1pct.mean()*100:5.1f}%")
    print(f"  fraction of the block ever used: PSM {df.PSM_kin_frac_used.mean()*100:5.1f}%   MCA {df.MCA_kin_frac_used.mean()*100:5.1f}%")

    print()
    print("="*94)
    print("[S2] is pure-PSM gain concentrated on COARSE / PROMISCUOUS columns?  (the radius=0 analogue)")
    print("="*94)
    print(f"  baseline mean coarseness over all 1955 cols = {coarse.mean():.2f}/20")
    for k in (10,20,50):
        print(f"    top-{k:>2} gain cols: coarseness {df[f'PSM_top{k}_coarse'].mean():.2f} "
              f"({df[f'PSM_top{k}_coarse_enr'].mean():.2f}x)   median prevalence {df[f'PSM_top{k}_prev'].mean():.3f}")
    print("  gain share by prevalence stratum (cols share in brackets):")
    for i,l in enumerate(lab):
        print(f"    {l:<14} gain {df[f'PSM_gain_{l}'].mean()*100:5.1f}%   [cols {(strat==i).mean()*100:5.1f}%]")
    print(f"  coarse bits (>=7/20 AA): gain {df['PSM_gain_coarse>=7'].mean()*100:.1f}%  "
          f"[cols {df['PSM_gain_frac_cols_coarse>=7'].mean()*100:.1f}%]  "
          f"enrichment {df['PSM_gain_coarse>=7'].mean()/df['PSM_gain_frac_cols_coarse>=7'].mean():.2f}x")
    print(f"  fine   bits (<=2/20 AA): gain {df['PSM_gain_fine<=2'].mean()*100:.1f}%  "
          f"[cols {df['PSM_gain_frac_cols_fine<=2'].mean()*100:.1f}%]  "
          f"enrichment {df['PSM_gain_fine<=2'].mean()/df['PSM_gain_frac_cols_fine<=2'].mean():.2f}x")
    print(f"  Spearman(gain, prevalence)  = {df.PSM_spearman_gain_prev.mean():+.3f}")
    print(f"  Spearman(gain, coarseness)  = {df.PSM_spearman_gain_coarse.mean():+.3f}")
    print("\nwrote", f"{OUT}/mech_gain.csv")


if __name__ == "__main__":
    single_main("mech")
