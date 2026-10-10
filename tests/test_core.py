import unittest,numpy as np
from rbp.core import JointMass,exact_metrics,cp_interval
class CoreTests(unittest.TestCase):
 def test_exact_singular(self):
  A=np.array([[.49,.49,.01],[.49,.49,.01],[.02,.02,.98]])
  q=np.array([.1,.3,.6]);p=np.array([[1,1,1],[0,0,1]])
  m=JointMass((A*1000000).astype(int),(A@q*1000000).round().astype(int),[0,1,2],p)
  self.assertTrue(m.feasible)
  cert=m.certify(.02)
  self.assertEqual(cert.policy,1)
  cov,risk=exact_metrics(A,q,[0,1,2],p,cert.policy)
  self.assertAlmostEqual(cov,.596);self.assertLess(risk,.02)
  self.assertGreater(m.prior_width(),.39)
 def test_interval_edges(self):
  lo,hi=cp_interval([0,10],[10,10],.05)
  self.assertEqual(lo[0],0);self.assertEqual(hi[1],1)
 def test_empty_source_class(self):
  A=np.array([[30,0],[0,0]])
  m=JointMass(A,np.array([10,10]),[0,1],[[1,1]])
  self.assertEqual(m.certify(.1).policy,-1)
if __name__=='__main__':unittest.main()
