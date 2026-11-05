import unittest
import numpy as np
from log_analysis import mixture_decomposition,weighted


class AuditTests(unittest.TestCase):
 def test_rate_only_change(self):
  a=[dict(auc=.8,hit90=10)]*9+[dict(auc=0,hit90=None)]
  b=[dict(auc=.8,hit90=10)]*4+[dict(auc=0,hit90=None)]*6
  x=mixture_decomposition(a,b)
  self.assertAlmostEqual(x['descriptive_hit_fraction_component'],-.4)
  self.assertAlmostEqual(x['descriptive_conditional_auc_component'],0)

 def test_performance_only_change(self):
  a=[dict(auc=.8,hit90=10),dict(auc=0,hit90=None)]*5
  b=[dict(auc=.6,hit90=10),dict(auc=.01,hit90=None)]*5
  x=mixture_decomposition(a,b)
  self.assertAlmostEqual(x['descriptive_hit_fraction_component'],0)
  self.assertAlmostEqual(x['descriptive_conditional_auc_component'],-.095)

 def test_decomposition_exact_and_symmetric(self):
  a=[dict(auc=.8,hit90=10)]*9+[dict(auc=0,hit90=None)]
  b=[dict(auc=.7,hit90=10)]*4+[dict(auc=.0002,hit90=None)]*6
  x=mixture_decomposition(a,b);y=mixture_decomposition(b,a)
  for key in ('observed_mean_change','descriptive_hit_fraction_component','descriptive_conditional_auc_component'):
   self.assertAlmostEqual(x[key],-y[key])

 def test_update_weighted_mean_and_missing(self):
  a=[dict(x_mean='0.2',x_n='1000'),dict(x_mean='1',x_n='1'),dict(x_mean='',x_n='0')]
  self.assertAlmostEqual(weighted(a,'x'),201/1001)
  self.assertIsNone(weighted([],'x'))

 def test_both_groups_required(self):
  a=[dict(auc=.8,hit90=10)]*10
  with self.assertRaises(ValueError):mixture_decomposition(a,a)

 def test_saturated_tanh_blocks_gradient_with_nonzero_action_gradient(self):
  import torch
  w=torch.tensor([100.],requires_grad=True);a=torch.tanh(w);loss=-2*a.sum()
  action_grad=torch.autograd.grad(loss,a,retain_graph=True)[0]
  param_grad=torch.autograd.grad(loss,w)[0]
  self.assertEqual(action_grad.item(),-2.)
  self.assertEqual(a.item(),1.)
  self.assertEqual(param_grad.item(),0.)

 def test_unsaturated_tanh_preserves_gradient(self):
  import torch
  w=torch.tensor([.1],requires_grad=True);a=torch.tanh(w);loss=-2*a.sum()
  actual=torch.autograd.grad(loss,w)[0].item()
  self.assertAlmostEqual(actual,-2*(1-np.tanh(.1)**2),places=6)


if __name__=='__main__':unittest.main(verbosity=2)
