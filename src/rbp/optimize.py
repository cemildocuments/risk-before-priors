"""Coverage-maximizing cellwise randomized selection by constraint generation."""
import numpy as np
from scipy.optimize import linprog


def optimize_policy(jm, risk, tolerance=1e-6, max_iter=300):
    if not jm.feasible:
        return dict(g=np.zeros(jm.B),coverage_lower=0.,risk_bound=float('nan'),gap=0.,iterations=0,status='infeasible')
    # A small stricter working risk leaves room for numerical verification.
    working=max(0.,risk-1e-5)
    z=jm.solve(np.zeros(jm.N)).x.reshape(jm.B,jm.K)
    constraints=[];rhs=[];risk_weights=[]
    def add(z,riskcut=True,coveragecut=True):
        if riskcut:
            constraints.append(np.r_[(z*(jm.err-working)).sum(axis=1),0.]);rhs.append(0.);risk_weights.append(np.r_[z.sum(axis=1),0.])
        if coveragecut:
            constraints.append(np.r_[-z.sum(axis=1),1.]);rhs.append(0.);risk_weights.append(np.zeros(jm.B+1))
    add(z)
    best=dict(g=np.zeros(jm.B),coverage_lower=0.,risk_bound=float('nan'),gap=1.,iterations=0,status='abstain')
    for it in range(max_iter):
        res=linprog(np.r_[np.zeros(jm.B),-1.],A_ub=np.asarray(constraints),b_ub=rhs,bounds=[(0.,1.)]*(jm.B+1),method='highs')
        if not res.success:break
        g=res.x[:-1];outer=res.x[-1];C=np.repeat(g,jm.K);E=(g[:,None]*jm.err).ravel()
        sr=jm.solve(-(E-working*C));sc=jm.solve(C)
        if not(sr.success and sc.success):break
        lower=max(0.,-jm.upper(-C));margin=jm.upper(E-risk*C)
        if lower>1e-8 and margin<=-1e-8 and lower>best['coverage_lower']:
            best=dict(g=g.copy(),coverage_lower=lower,risk_bound=risk,gap=max(0.,outer-lower),iterations=it+1,status='certified')
        best['gap']=max(0.,outer-best['coverage_lower']);best['iterations']=it+1
        if best['gap']<=tolerance:break
        add(sr.x.reshape(jm.B,jm.K),True,False);add(sc.x.reshape(jm.B,jm.K),False,True)
    # Re-solve the finite-cut outer relaxation at the requested risk to obtain
    # an upper bound on the original robust optimum, not the stricter work risk.
    outer=linprog(np.r_[np.zeros(jm.B),-1.],A_ub=np.asarray(constraints)+(working-risk)*np.asarray(risk_weights),b_ub=rhs,bounds=[(0.,1.)]*(jm.B+1),method='highs')
    if outer.success:best['gap']=max(0.,float(outer.x[-1])-best['coverage_lower'])
    return best
