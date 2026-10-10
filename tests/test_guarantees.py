import unittest
import numpy as np
from rbp.core import JointMass,exact_metrics
from rbp.optimize import optimize_policy

class GuaranteeTests(unittest.TestCase):
 def test_dual_bounds_dominate_primal_values(self):
  rng=np.random.default_rng(91)
  for _ in range(20):
   A=rng.dirichlet(np.ones(6),size=3).T;q=rng.dirichlet(np.ones(3))
   c=np.column_stack([rng.multinomial(500,A[:,y]) for y in range(3)])
   jm=JointMass(c,rng.multinomial(1000,A@q),[0,0,1,1,2,2],[[1]*6,[0,1,0,1,0,1]])
   self.assertTrue(jm.feasible)
   v=rng.normal(size=18);pr=jm.solve(-v)
   self.assertGreaterEqual(jm.upper(v)+1e-9,-pr.fun)
 def test_matrix_certificate_implies_joint_certificate(self):
  rng=np.random.default_rng(119)
  for _ in range(20):
   A=.96*np.eye(3)+.04*rng.dirichlet(np.ones(3),size=3).T;q=rng.dirichlet(np.ones(3))
   jm=JointMass(np.column_stack([rng.multinomial(2000,A[:,y]) for y in range(3)]),rng.multinomial(4000,A@q),[0,1,2],[[1,1,1],[0,0,1]])
   for r in [.05,.1,.2]:
    m=jm.factored(r);j=jm.certify(r)
    if m.policy>=0:
     self.assertGreaterEqual(j.policy,0)
     self.assertGreaterEqual(jm.policies[j.policy]@jm.muhat,jm.policies[m.policy]@jm.muhat-1e-8)
 def test_incompatible_observations_refuse(self):
  jm=JointMass([[1000000,1000000],[0,0]],[0,1000000],[0,1],[[1,1]])
  self.assertFalse(jm.feasible);self.assertEqual(jm.certify(.1).policy,-1)
 def test_optimized_policy_has_verified_margin(self):
  A=np.array([[.49,.49,.01],[.49,.49,.01],[.02,.02,.98]]);q=np.array([.1,.3,.6])
  jm=JointMass((A*1000000).astype(int),np.rint(A@q*1000000).astype(int),[0,1,2],[[1,1,1],[0,0,1]])
  o=optimize_policy(jm,.05);g=o['g'];C=np.repeat(g,3);E=(g[:,None]*jm.err).ravel()
  self.assertLess(jm.upper(E-.05*C),0);self.assertGreater(o['coverage_lower'],.55)
  joint=A*q;actual=(joint*g[:,None]).sum();risk=(joint*g[:,None]*jm.err).sum()/actual
  self.assertLessEqual(risk,.05);self.assertLessEqual(o['coverage_lower'],actual+1e-8);self.assertLess(o['gap'],1e-3)

if __name__=='__main__':unittest.main()
