"""Focused policy-optimization and aggregate-event ablation study."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,time,json,argparse
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from rbp.core import JointMass,exact_metrics
from rbp.optimize import optimize_policy

def one(t):
 name,seed,shift=t;path=ROOT/f'data/processed/{name}_rf_{seed}.npz';d=np.load(path);A=d['A'];pred=d['predictions'];pol=d['policies'];B,K=A.shape
 rng=np.random.default_rng(np.random.SeedSequence([20261011,seed,sum(map(ord,name)),sum(map(ord,shift))]));q=np.ones(K)/K
 if shift=='severe':q*=.2;q[np.argmax((A*(pred[:,None]!=np.arange(K))).sum(axis=0))]+=.8
 else:q=rng.dirichlet(np.full(K,.3))
 counts=np.column_stack([rng.multinomial(1000,A[:,y]) for y in range(K)]);tc=rng.multinomial(4000,A@q)
 jm=JointMass(counts,tc,pred,pol);no=JointMass(counts,tc,pred,pol,aggregates=False);r=.1;rows=[]
 for method in ['joint','no_aggregates','cellwise']:
  start=time.perf_counter()
  if method=='cellwise':
   out=optimize_policy(jm,r,max_iter=200);g=out['g'];joint=A*q[None,:];c=float((g[:,None]*joint).sum());e=float((g[:,None]*joint*jm.err).sum()/c) if c>0 else np.nan;cl=out['coverage_lower'];gap=out['gap'];it=out['iterations'];status=out['status']
   np.savez_compressed(ROOT/f'results/policy/{name}_{seed}_{shift}.npz',g=g,q=q,counts=counts,target_counts=tc)
  else:
   cert=(jm if method=='joint' else no).certify(r);c,e=exact_metrics(A,q,pred,pol,cert.policy);cl=cert.coverage_lower;gap=np.nan;it=0;status=cert.status
  rows.append(dict(dataset=name,model='rf',split=seed,shift=shift,method=method,risk_target=r,coverage=c,risk=e,coverage_lower=cl,violation=int(c>0 and e>r+1e-10),gap=gap,iterations=it,status=status,seconds=time.perf_counter()-start))
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);a=p.parse_args();(ROOT/'results/policy').mkdir(exist_ok=True)
 tasks=[(d,s,q) for d in ['bean','sensorless','shuttle','waveform'] for s in range(42,47) for q in ['severe','random']];rows=[]
 with ProcessPoolExecutor(max_workers=a.workers) as ex:
  for i,r in enumerate(ex.map(one,tasks)):
   rows.extend(r);pd.DataFrame(rows).to_csv(ROOT/'results/policy.csv',index=False);print(i+1,len(tasks),r[-1],flush=True)
 (ROOT/'configs/policy.json').write_text(json.dumps({'datasets':['bean','sensorless','shuttle','waveform'],'models':['rf'],'seeds':list(range(42,47)),'n_per_class':1000,'target_n':4000,'shifts':['severe','random'],'risk':.1,'delta':.05,'max_iterations':200,'gap_tolerance':1e-6},indent=2))
