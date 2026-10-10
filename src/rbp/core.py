"""Finite-sample joint-mass certificates for selective classification.

The partition, classifier, and policy family must be fixed independently of
calibration data. Clopper--Pearson intervals are simultaneous by a union bound.
"""
import warnings
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack


def cp_interval(count, n, failure):
    count=np.asarray(count,dtype=float); n=np.broadcast_to(np.asarray(n,dtype=float),count.shape)
    lo=np.zeros(count.shape); hi=np.ones(count.shape)
    take=(count>0)&(n>0); lo[take]=beta.ppf(failure/2,count[take],n[take]-count[take]+1)
    take=(count<n)&(n>0); hi[take]=beta.ppf(1-failure/2,count[take]+1,n[take]-count[take])
    return lo,hi


def cell_counts(cells, labels, B, K):
    return np.bincount(np.asarray(cells)*K+np.asarray(labels),minlength=B*K).reshape(B,K)


@dataclass
class Certificate:
    policy: int
    coverage_lower: float
    risk_upper: float
    status: str


class JointMass:
    """Lifted target joint mass z[b,y], with q[y]=sum_b z[b,y]."""
    def __init__(self, counts, target_counts, predictions, policies, delta=.05, aggregates=True):
        self.counts=np.asarray(counts); self.tc=np.asarray(target_counts)
        self.B,self.K=self.counts.shape; self.N=self.B*self.K
        self.pred=np.asarray(predictions); self.policies=np.asarray(policies,dtype=float)
        self.delta=delta
        self.J=len(policies); self.n=self.counts.sum(axis=0); self.m=self.tc.sum()
        # One simultaneous event contains every cell and policy event.
        self.source_events=self.B+2*self.J
        self.sf=delta/(2*self.K*self.source_events)
        self.tf=delta/(2*(self.B+self.J))
        self.L,self.U=cp_interval(self.counts,self.n[None,:],self.sf)
        self.tL,self.tU=cp_interval(self.tc,self.m,self.tf)
        self.Ahat=np.divide(self.counts,self.n[None,:],out=np.zeros_like(self.counts,dtype=float),where=self.n[None,:]>0)
        self.muhat=self.tc/self.m
        self.err=(self.pred[:,None]!=np.arange(self.K)[None,:]).astype(float)
        self.Cs=[];self.Es=[]; self.cl=[];self.cu=[];self.el=[];self.eu=[];self.tcl=[];self.tcu=[]
        for g in self.policies:
            C=np.repeat(g,self.K); E=(g[:,None]*self.err).ravel()
            self.Cs.append(C); self.Es.append(E)
            c=(self.counts*g[:,None]).sum(axis=0); e=(self.counts*g[:,None]*self.err).sum(axis=0)
            a,b=cp_interval(c,self.n,self.sf);self.cl.append(a);self.cu.append(b)
            a,b=cp_interval(e,self.n,self.sf);self.el.append(a);self.eu.append(b)
            a,b=cp_interval(np.array([g@self.tc]),self.m,self.tf);self.tcl.append(a[0]);self.tcu.append(b[0])
        self.cl=np.asarray(self.cl);self.cu=np.asarray(self.cu);self.el=np.asarray(self.el);self.eu=np.asarray(self.eu)
        rows=[];cols=[];vals=[];rhs=[]
        def row(indices,values,b):
            rid=len(rhs);rows.extend([rid]*len(indices));cols.extend(indices);vals.extend(values);rhs.append(b)
        classidx=[np.arange(y,self.N,self.K) for y in range(self.K)]
        for b in range(self.B):
            for y in range(self.K):
                row([b*self.K+y,self.N+y],[1.,-self.U[b,y]],0)
                row([b*self.K+y,self.N+y],[-1.,self.L[b,y]],0)
            idx=np.arange(b*self.K,(b+1)*self.K)
            row(idx.tolist(),np.ones(self.K).tolist(),self.tU[b]);row(idx.tolist(),(-np.ones(self.K)).tolist(),-self.tL[b])
        for j,g in enumerate(self.policies):
            nz=np.flatnonzero(self.Cs[j]);row(nz.tolist(),self.Cs[j][nz].tolist(),self.tcu[j]);row(nz.tolist(),(-self.Cs[j][nz]).tolist(),-self.tcl[j])
            if aggregates:
                for y,idx in enumerate(classidx):
                    for ev,lo,hi in [(g,self.cl[j,y],self.cu[j,y]),(g*self.err[:,y],self.el[j,y],self.eu[j,y])]:
                        nz=np.flatnonzero(ev)
                        row(idx[nz].tolist()+[self.N+y],ev[nz].tolist()+[-float(hi)],0)
                        row(idx[nz].tolist()+[self.N+y],(-ev[nz]).tolist()+[float(lo)],0)
        self.A=coo_matrix((vals,(rows,cols)),shape=(len(rhs),self.N+self.K)).tocsr();self.b=np.array(rhs)
        eqrows=[0]*self.N;eqcols=list(range(self.N));eqvals=[1.]*self.N
        for y,idx in enumerate(classidx):
            eqrows.extend([y+1]*(self.B+1));eqcols.extend(idx.tolist()+[self.N+y]);eqvals.extend([1.]*self.B+[-1.])
        self.eq=coo_matrix((eqvals,(eqrows,eqcols)),shape=(self.K+1,self.N+self.K)).tocsr()
        self.eqb=np.r_[1.,np.zeros(self.K)];self.bounds=(0,None)
        self.feasible=self.solve(np.zeros(self.N)).success

    def solve(self,c):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res=linprog(np.r_[c,np.zeros(self.K)],A_ub=self.A,b_ub=self.b,A_eq=self.eq,b_eq=self.eqb,bounds=self.bounds,method='highs',options={'threads':1,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            if res.x is not None:res.x=res.x[:self.N]
            return res

    def upper(self,v):
        sol=self.solve(-np.asarray(v))
        if not sol.success:return float("inf")
        multipliers=np.maximum(-sol.ineqlin.marginals,0.)
        nu=-np.asarray(sol.eqlin.marginals)
        residual=float(np.max(np.r_[v,np.zeros(self.K)]-self.A.T@multipliers-self.eq.T@nu))
        repair=max(0.,residual)
        # Increasing nu[0] by 2r and decreasing all remaining entries by r
        # adds r to every dual inequality, because sum(z)=sum(q)=1.
        objective=float(self.b@multipliers+nu[0]+2*repair)
        padding=1e-10*(1+abs(nu[0])+2*repair+float(np.abs(self.b)@multipliers))
        return objective+padding

    def certify(self,risk,upper=False):
        if not self.feasible:return Certificate(-1,0.,float('nan'),'infeasible')
        order=sorted(range(self.J),key=lambda j:float(self.policies[j]@self.muhat),reverse=True)
        for j in order:
            if self.tcl[j]<=1e-10:continue
            bound=self.upper(self.Es[j]-risk*self.Cs[j])
            if bound<=-1e-8:
                ru=risk
                if upper:
                    low=0.; high=risk
                    for _ in range(18):
                        mid=(low+high)/2; bound=self.upper(self.Es[j]-mid*self.Cs[j])
                        if bound<=-1e-8:high=mid
                        else:low=mid
                    ru=high
                return Certificate(j,float(self.tcl[j]),float(ru),'certified')
        return Certificate(-1,0.,float('nan'),'abstain')

    def projected_region(self):
        # A strong conditional-probability version of matrix-constraint inference.
        A=[self.L,-self.U,np.asarray(self.cl),-np.asarray(self.cu)]
        b=[self.tU,-self.tL,np.asarray(self.tcu),-np.asarray(self.tcl)]
        return np.vstack(A),np.concatenate(b)

    def prior_width(self):
        if not self.feasible:return float('nan')
        widths=[]
        for y in range(self.K):
            v=np.zeros(self.N);v[y::self.K]=1
            low=self.solve(v);high=self.solve(-v)
            if not(low.success and high.success):return float('nan')
            widths.append(-high.fun-low.fun)
        return max(widths)

    def factored(self,risk,box=False):
        A,b=self.projected_region(); bounds=[(0.,1.)]*self.K
        def solve(c,bd=bounds):return linprog(c,A_ub=A,b_ub=b,A_eq=np.ones((1,self.K)),b_eq=[1.],bounds=bd,method='highs')
        if not solve(np.zeros(self.K)).success:return Certificate(-1,0.,float('nan'),'infeasible')
        if box:
            bd=[]
            for y in range(self.K):
                c=np.eye(self.K)[y];low=solve(c);hi=solve(-c);bd.append((max(0,low.fun),min(1,-hi.fun)))
            def opt(c):return linprog(c,A_eq=np.ones((1,self.K)),b_eq=[1.],bounds=bd,method='highs')
        else:opt=solve
        order=sorted(range(self.J),key=lambda j:float(self.policies[j]@self.muhat),reverse=True)
        for j in order:
            if self.tcl[j]<=1e-10:continue
            # Independently upper-bounding error and lower-bounding acceptance
            # is valid but deliberately discards their joint-mass compatibility.
            numerator=opt(-self.eu[j]);margin=opt(-(self.eu[j]-risk*self.cl[j]))
            if numerator.success and margin.success:
                bound=min(-numerator.fun/risk-self.tcl[j],-margin.fun/risk)
                if bound<=-1e-8:return Certificate(j,float(self.tcl[j]),risk,'certified')
        return Certificate(-1,0.,float('nan'),'abstain')

    def worst_class(self,risk):
        # Exact conditional binomial upper bounds, with simultaneous policies/classes.
        alpha=self.delta/(self.J*self.K)
        for j in sorted(range(self.J),key=lambda j:float(self.policies[j]@self.muhat),reverse=True):
            accepted=(self.counts*self.policies[j,:,None]).sum(axis=0)
            errors=(self.counts*self.policies[j,:,None]*self.err).sum(axis=0)
            _,hi=cp_interval(errors,accepted,alpha)
            if np.all(hi<=risk):return Certificate(j,float(self.tcl[j]),risk,'certified')
        return Certificate(-1,0.,float('nan'),'abstain')

    def plugin(self,risk,regularization=0.):
        from scipy.optimize import minimize
        def obj(q):return np.sum((self.Ahat@q-self.muhat)**2)+regularization*np.sum((q-1/self.K)**2)
        res=minimize(obj,np.ones(self.K)/self.K,bounds=[(0,1)]*self.K,constraints={'type':'eq','fun':lambda q:q.sum()-1},method='SLSQP',options={'ftol':1e-12,'maxiter':300})
        q=res.x
        for j in sorted(range(self.J),key=lambda j:float(self.policies[j]@self.muhat),reverse=True):
            z=self.Ahat*q[None,:];c=self.Cs[j]@z.ravel();e=self.Es[j]@z.ravel()
            if c>1e-12 and e/c<=risk:return Certificate(j,0.,float(e/c),'plugin')
        return Certificate(-1,0.,float('nan'),'abstain')

    def source(self,risk):
        # Source calibration is stratified: use classwise simultaneous bounds.
        prior=self.n/self.n.sum()
        for j in sorted(range(self.J),key=lambda j:float(self.policies[j]@self.muhat),reverse=True):
            c=float(self.cl[j]@prior);e=float(self.eu[j]@prior)
            if c>0 and e/c<=risk:return Certificate(j,0.,e/c,'source')
        return Certificate(-1,0.,float('nan'),'abstain')


def exact_metrics(A,q,predictions,policies,policy):
    if policy<0:return 0.,float('nan')
    joint=A*q[None,:];mask=policies[policy,:,None]
    coverage=float((joint*mask).sum());error=float((joint*mask*(np.asarray(predictions)[:,None]!=np.arange(A.shape[1]))).sum())
    return coverage,error/coverage if coverage>0 else float('nan')
