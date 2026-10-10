"""Train fixed base classifiers and publish their held-out score populations."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,json,time,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
ROOT=Path(__file__).resolve().parent
DATASETS=['bean','letter','sensorless','pendigits','optdigits','shuttle','satellite','waveform']
TARGET={'bean':'Class','letter':'lettr','sensorless':'class','pendigits':'Class','optdigits':'class','shuttle':'class','satellite':'class','waveform':'class'}

def train(task):
 name,model,seed=task;dest=ROOT/f'data/processed/{name}_{model}_{seed}.npz'
 if dest.exists():return {'cached':dest.name}
 start=time.perf_counter();d=pd.read_csv(ROOT/f'data/raw/{name}.csv');lab=TARGET[name]
 y=LabelEncoder().fit_transform(d[lab]);X=d.drop(columns=[lab]).to_numpy(dtype=np.float32);K=int(y.max()+1)
 ids=np.arange(len(y));development,pool=train_test_split(ids,test_size=.5,stratify=y,random_state=seed)
 training,design=train_test_split(development,test_size=.2,stratify=y[development],random_state=seed+1000)
 scaler=StandardScaler().fit(X[training]);xt=scaler.transform(X[training]).astype(np.float32)
 xd=scaler.transform(X[design]).astype(np.float32);xp=scaler.transform(X[pool]).astype(np.float32)
 gpu=False
 if model=='lr':
  clf=LogisticRegression(C=1.,max_iter=1500,solver='lbfgs',random_state=seed);clf.fit(xt,y[training]);pdsg=clf.predict_proba(xd);pp=clf.predict_proba(xp)
 elif model=='rf':
  clf=RandomForestClassifier(n_estimators=200,min_samples_leaf=2,max_features='sqrt',n_jobs=1,random_state=seed);clf.fit(xt,y[training]);pdsg=clf.predict_proba(xd);pp=clf.predict_proba(xp)
 elif model=='mlp':
  import torch
  torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
  device='mps' if torch.backends.mps.is_available() else 'cpu';gpu=device=='mps'
  net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,K)).to(device)
  optim=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001)
  tx=torch.from_numpy(xt).to(device);ty=torch.from_numpy(y[training]).long().to(device)
  rng=np.random.default_rng(seed)
  for epoch in range(100):
   perm=rng.permutation(len(training))
   for i in range(0,len(perm),2048):
    ix=torch.as_tensor(perm[i:i+2048],device=device);optim.zero_grad();loss=torch.nn.functional.cross_entropy(net(tx[ix]),ty[ix]);loss.backward();optim.step()
  net.eval()
  with torch.no_grad():
   pdsg=torch.softmax(net(torch.from_numpy(xd).to(device)),dim=1).cpu().numpy()
   pp=torch.softmax(net(torch.from_numpy(xp).to(device)),dim=1).cpu().numpy()
  torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},ROOT/f'data/processed/{name}_{model}_{seed}.pt')
 else:raise ValueError(model)
 # Independent design set fixes score bins and the policy family.
 edges=np.unique(np.quantile(pdsg.max(axis=1),[.2,.4,.6,.8,.9,.95]))
 levels=np.searchsorted(edges,pp.max(axis=1),side='right');L=len(edges)+1
 cells=pp.argmax(axis=1)*L+levels;B=K*L
 counts=np.bincount(cells*K+y[pool],minlength=B*K).reshape(B,K)
 A=counts/counts.sum(axis=0)[None,:]
 predictions=np.repeat(np.arange(K),L);policies=np.array([(np.tile(np.arange(L),K)>=j).astype(int) for j in range(L)])
 np.savez_compressed(dest,A=A,predictions=predictions,policies=policies,edges=edges,pool_ids=pool,train_ids=training,design_ids=design,pool_y=y[pool],pool_cells=cells,pool_probabilities=pp,classes=np.unique(d[lab]).astype(str),features=d.drop(columns=[lab]).columns.astype(str).to_numpy())
 result={'dataset':name,'model':model,'seed':seed,'N':len(y),'d':X.shape[1],'K':K,'B':B,'training':len(training),'design':len(design),'pool':len(pool),'pool_accuracy':float((pp.argmax(axis=1)==y[pool]).mean()),'seconds':time.perf_counter()-start,'gpu':gpu,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
 (ROOT/f'results/train_{name}_{model}_{seed}.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',default=['lr','rf']);p.add_argument('--datasets',nargs='+',default=DATASETS);p.add_argument('--seeds',nargs='+',type=int,default=list(range(42,47)));p.add_argument('--workers',type=int,default=6);args=p.parse_args()
 tasks=[(d,m,s) for d in args.datasets for m in args.models for s in args.seeds]
 if args.models==['mlp']:
  for t in tasks:print(train(t),flush=True)
 else:
  with ProcessPoolExecutor(max_workers=args.workers) as ex:
   for r in ex.map(train,tasks):print(r,flush=True)
