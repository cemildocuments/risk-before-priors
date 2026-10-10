"""Check dataset integrity, split disjointness, and seeded calibration reproduction."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
from run_experiments import run_one

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true',help='Recompute every main scenario instead of one per dataset.');a=parser.parse_args()
 for rec in json.loads((ROOT/'data/manifest.json').read_text()):
  f=ROOT/'data/raw'/rec['filename'];assert hashlib.sha256(f.read_bytes()).hexdigest()==rec['sha256'],f
 for path in (ROOT/'data/processed').glob('*.npz'):
  z=np.load(path);assert np.allclose(z['A'].sum(axis=0),1)
  tr,ds,po=map(set,[z['train_ids'],z['design_ids'],z['pool_ids']]);assert not(tr&ds or tr&po or ds&po)
 config=json.loads((ROOT/'configs/main.json').read_text());tasks=[]
 for fn in config['files']:
  if not a.full and not fn.endswith('_rf_42.npz'):continue
  for n in config['n'] if a.full else [1000]:
   for sh in config['shifts'] if a.full else ['severe']:
    for rep in range(config['repetitions']) if a.full else [0]:tasks.append((str(ROOT/'data/processed'/fn),n,sh,rep,'reproduction_check'))
 keys=['dataset','model','split','n_per_class','target_n','shift','rep','risk_target','method']
 fields=['policy','coverage','risk','violation','certified','coverage_lower','status','sigma_min','condition','source_n','B','K']
 checks=[]
 for task in tasks:
  stem=Path(task[0]).stem;fn=f'{stem}_n{task[1]}_{task[2]}_r{task[3]}.csv';out=ROOT/'results/reproduction_check'/fn
  if out.exists():out.unlink()
  run_one(task);old=pd.read_csv(ROOT/'results/main'/fn).sort_values(keys).reset_index(drop=True);new=pd.read_csv(out).sort_values(keys).reset_index(drop=True)
  pd.testing.assert_frame_equal(old[keys+fields],new[keys+fields],check_exact=False,rtol=1e-8,atol=1e-9)
  checks.append(fn);print('PASS',fn,flush=True)
 report={'raw_files':8,'frozen_populations':len(list((ROOT/'data/processed').glob('*.npz'))),'recomputed_scenarios':len(checks),'operating_rows':27*len(checks),'fields_checked':fields,'rtol':1e-8,'atol':1e-9,'status':'passed'}
 (ROOT/'results/reproduction_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
