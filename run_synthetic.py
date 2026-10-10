"""Independent multinomial replications for singular and nearly singular channels."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,time,argparse,json
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from rbp.core import JointMass,exact_metrics
from rbp.optimize import optimize_policy
PRED=np.array([0,1,2]);POL=np.array([[1,1,1],[0,0,1]])

def one(t):
 eps,q3,n,m,rep=t
 A=np.array([[.49+eps,.49-eps,.01],[.49-eps,.49+eps,.01],[.02,.02,.98]])
 q=np.array([.05*(1-q3),.95*(1-q3),q3])
 rng=np.random.default_rng(np.random.SeedSequence([777,int(eps*1e6),int(q3*100),n,m,rep]))
 counts=np.column_stack([rng.multinomial(n,A[:,y]) for y in range(3)]);tc=rng.multinomial(m,A@q)
 jm=JointMass(counts,tc,PRED,POL);width=jm.prior_width();rows=[]
 for r in [.02,.05,.1]:
  funcs={'joint':lambda:jm.certify(r),'matrix':lambda:jm.factored(r),'worst_class':lambda:jm.worst_class(r),'ls_plugin':lambda:jm.plugin(r)}
  for method,fun in funcs.items():
   cert=fun();cov,risk=exact_metrics(A,q,PRED,POL,cert.policy)
   rows.append(dict(epsilon=eps,q3=q3,n_per_class=n,target_n=m,rep=rep,risk_target=r,method=method,coverage=cov,risk=risk,violation=int(cov>0 and risk>r+1e-10),certified=int(cov>0),prior_width=width))
 return rows

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--reps',type=int,default=100);a=p.parse_args()
 tasks=[(e,q,n,4000,i) for e in [0.,.001,.01,.1] for q in [.2,.6,.9] for n in [200,1000,5000] for i in range(a.reps)]
 rows=[]
 with ProcessPoolExecutor(max_workers=a.workers) as ex:
  for i,r in enumerate(ex.map(one,tasks,chunksize=10)):
   rows.extend(r)
   if i%100==0:print(i,len(tasks),flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'results/synthetic.csv',index=False)
 (ROOT/'configs/synthetic.json').write_text(json.dumps({'epsilons':[0,.001,.01,.1],'q3':[.2,.6,.9],'n_per_class':[200,1000,5000],'target_n':4000,'reps':a.reps,'seed':777,'risk_targets':[.02,.05,.1]},indent=2))
