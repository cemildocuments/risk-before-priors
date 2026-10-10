"""Seeded calibration experiments on exact finite score populations."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import argparse,json,time,warnings
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from scipy.optimize import minimize
from rbp.core import JointMass,Certificate,exact_metrics
warnings.filterwarnings('ignore',message='Unrecognized options detected')


def run_one(task):
 path,n,shift,rep,mode=task; stem=Path(path).stem; out=ROOT/f'results/{mode}/{stem}_n{n}_{shift}_r{rep}.csv'
 if out.exists():return {'cached':out.name}
 out.parent.mkdir(parents=True,exist_ok=True)
 z=np.load(path);A=z['A'];pred=z['predictions'];policies=z['policies'];B,K=A.shape
 dataset,model,split=stem.rsplit('_',2);split=int(split)
 seed=np.random.SeedSequence([20261010,split,n,rep,{'balanced':0,'mild':1,'severe':2,'random':3}[shift],sum(map(ord,dataset)),sum(map(ord,model))])
 rng=np.random.default_rng(seed)
 err=(A*(pred[:,None]!=np.arange(K))).sum(axis=0);hard=int(err.argmax())
 q=np.ones(K)/K
 if shift=='mild':q*=.7;q[hard]+=.3
 elif shift=='severe':q*=.2;q[hard]+=.8
 elif shift=='random':q=rng.dirichlet(np.full(K,.3))
 m=4000
 counts=np.column_stack([rng.multinomial(n,A[:,y]) for y in range(K)])
 tc=rng.multinomial(m,A@q)
 start=time.perf_counter();jm=JointMass(counts,tc,pred,policies,delta=.05);build=time.perf_counter()-start
 sing=np.linalg.svd(A,compute_uv=False);sigma=float(sing[-1]);cond=float(sing[0]/max(sing[-1],1e-16))
 rows=[]
 for risk in [.05,.1,.2]:
  methods={'joint':lambda:jm.certify(risk),'matrix':lambda:jm.factored(risk),'box':lambda:jm.factored(risk,box=True),'worst_class':lambda:jm.worst_class(risk),'source':lambda:jm.source(risk),'ls_plugin':lambda:jm.plugin(risk),'ridge_plugin':lambda:jm.plugin(risk,.001)}
  def em():
   probs=(counts+.5)/(counts.sum(axis=0,keepdims=True)+.5*B);p=np.ones(K)/K
   for _ in range(1000):
    new=p*(probs.T@(tc/np.maximum(probs@p,1e-300)))/m
    if np.max(abs(new-p))<1e-10:p=new;break
    p=new
   for j in range(len(policies)):
    c,e=exact_metrics(jm.Ahat,p,pred,policies,j)
    if c>0 and e<=risk:return Certificate(j,0.,e,'plugin')
   return Certificate(-1,0.,float('nan'),'abstain')
  methods['em_plugin']=em
  def oracle():
   for j in range(len(policies)):
    c,e=exact_metrics(A,q,pred,policies,j)
    if c>0 and e<=risk:return Certificate(j,c,e,'oracle')
   return Certificate(-1,0.,float('nan'),'abstain')
  methods['oracle']=oracle
  for method,fun in methods.items():
   ts=time.perf_counter();cert=fun();elapsed=time.perf_counter()-ts
   cov,er=exact_metrics(A,q,pred,policies,cert.policy)
   rows.append(dict(dataset=dataset,model=model,split=split,n_per_class=n,target_n=m,shift=shift,rep=rep,risk_target=risk,method=method,policy=cert.policy,coverage=cov,risk=er,violation=int(cov>0 and er>risk+1e-10),certified=int(cert.policy>=0),coverage_lower=cert.coverage_lower,status=cert.status,seconds=elapsed,build_seconds=build,sigma_min=sigma,condition=cond,source_n=int(counts.sum()),B=B,K=K))
 pd.DataFrame(rows).to_csv(out,index=False)
 return {'file':out.name,'build':round(build,3),'elapsed':round(time.perf_counter()-start,2),'joint_cov':round(np.mean([r['coverage'] for r in rows if r['method']=='joint']),3),'matrix_cov':round(np.mean([r['coverage'] for r in rows if r['method']=='matrix']),3)}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',default='main');p.add_argument('--workers',type=int,default=8);p.add_argument('--reps',type=int,default=5);p.add_argument('--pilot',action='store_true');args=p.parse_args()
 files=sorted((ROOT/'data/processed').glob('*.npz'))
 if args.pilot:files=[f for f in files if f.stem in ['bean_lr_42','letter_rf_42','sensorless_mlp_42','waveform_lr_42']]
 tasks=[(str(f),n,s,r,args.mode) for f in files for n in ([1000] if args.pilot else [200,1000]) for s in (['severe','random'] if args.pilot else ['balanced','mild','severe','random']) for r in range(args.reps)]
 config={'files':[f.name for f in files],'n':[200,1000],'shifts':['balanced','mild','severe','random'],'repetitions':args.reps,'risk_targets':[.05,.1,.2],'delta':.05,'target_n':4000,'tasks':len(tasks),'base_seed':20261010,'workers':args.workers}
 (ROOT/f'configs/{args.mode}.json').write_text(json.dumps(config,indent=2))
 with ProcessPoolExecutor(max_workers=args.workers) as ex:
  fs=[ex.submit(run_one,t) for t in tasks]
  for i,f in enumerate(as_completed(fs)):
   try:print(i+1,'/',len(tasks),f.result(),flush=True)
   except Exception as e:print('FAILED',repr(e),flush=True);raise
 parts=list((ROOT/f'results/{args.mode}').glob('*.csv'))
 pd.concat([pd.read_csv(p) for p in parts],ignore_index=True).to_csv(ROOT/f'results/{args.mode}_all.csv',index=False)
