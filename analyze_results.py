"""Recompute numerical study summaries from seeded experiment outputs."""
from pathlib import Path
import json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;O=R/'results/summary';O.mkdir(exist_ok=True)
d=pd.read_csv(R/'results/main_all.csv');s=pd.read_csv(R/'results/synthetic.csv');p=pd.read_csv(R/'results/policy.csv')
summary=d.groupby('method').agg(coverage=('coverage','mean'),certification_rate=('certified','mean'),violations=('violation','sum'),runs=('violation','size'),seconds_median=('seconds','median'),seconds_p95=('seconds',lambda x:x.quantile(.95)))
summary.to_csv(O/'main_methods.csv')
d.groupby(['dataset','model','n_per_class','risk_target','method']).agg(coverage=('coverage','mean'),violations=('violation','sum')).to_csv(O/'main_strata.csv')
d.groupby(['shift','method']).violation.agg(['sum','count','mean']).to_csv(O/'violations_by_shift.csv')
idx=['dataset','model','split','n_per_class','shift','rep','risk_target'];wide=d.pivot(index=idx,columns='method',values='coverage');diff=wide.joint-wide.matrix
gains=diff.groupby('dataset').agg(['mean','min','max']);gains.to_csv(O/'paired_gains.csv')
ps=p.groupby(['dataset','method']).agg(coverage=('coverage','mean'),coverage_sd=('coverage','std'),coverage_lower=('coverage_lower','mean'),violations=('violation','sum'),seconds_median=('seconds','median'),gap_max=('gap','max'));ps.to_csv(O/'policy_methods.csv')
s.groupby(['epsilon','q3','n_per_class','risk_target','method']).agg(coverage=('coverage','mean'),certification_rate=('certified','mean'),violations=('violation','sum'),prior_width=('prior_width','mean')).to_csv(O/'synthetic_strata.csv')
train=pd.DataFrame([json.loads(x.read_text()) for x in (R/'results').glob('train_*.json')]);train.groupby(['dataset','model']).agg(accuracy=('pool_accuracy','mean'),accuracy_sd=('pool_accuracy','std'),seconds=('seconds','mean')).to_csv(O/'base_models.csv')
# Bootstrap at the dataset level: descriptive sensitivity to the eight selected
# datasets, not a population-level inferential claim about all future datasets.
rng=np.random.default_rng(2831);means=gains['mean'].to_numpy();boot=means[rng.integers(0,len(means),(10000,len(means)))].mean(axis=1)
result={'main_settings':len(wide),'main_calibrations':len(wide)//3,'main_rows':len(d),'main_pair_wins':int((diff>1e-8).sum()),'main_pair_losses':int((diff<-1e-8).sum()),'main_gain':float(diff.mean()),'gain_bootstrap_dataset_percentiles':np.quantile(boot,[.025,.975]).tolist(),'synthetic_calibrations':len(s)//12,'synthetic_jmc_violations':int(s[s.method=='joint'].violation.sum()),'policy_settings':len(p)//3,'policy_gain':float(p[p.method=='cellwise'].coverage.mean()-p[p.method=='joint'].coverage.mean()),'policy_max_gap':float(p.gap.max()),'policy_max_iterations':int(p.iterations.max()),'training_models':len(train),'all_gpu_mlp':bool(train[train.model=='mlp'].gpu.all()),'jmc_total_seconds_median':float((d[d.method=='joint'].seconds+d[d.method=='joint'].build_seconds).median()),'jmc_total_seconds_p95':float((d[d.method=='joint'].seconds+d[d.method=='joint'].build_seconds).quantile(.95))}
(O/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
