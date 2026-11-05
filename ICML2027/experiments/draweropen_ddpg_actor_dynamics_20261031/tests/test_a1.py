import copy
import importlib.util
import json
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bootstrap import ROOT
import numpy as np
import torch
from protocol import ARMS, SEEDS, checkpoints, make_config, probe_due, schedule
from diagnostics import policy_measure, preserve_rng, tensor_norm
from metacog_lite_v0.agent import DDPGAgent
from metacog_lite_v0.assignment_comparison import AssignmentSelector
from metacog_lite_v0.networks import Actor
torch.set_num_threads(1)


def reference(module):
    name = 'metacog_lite_v0.a1_reference_'+module
    spec = importlib.util.spec_from_file_location(name,ROOT/'reference/code/src/metacog_lite_v0'/(module+'.py'))
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and np.array_equal(a,b,equal_nan=True)
    if isinstance(a,dict):
        return a.keys() == b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return type(a) is type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    if isinstance(a,float) and np.isnan(a):
        return isinstance(b,float) and np.isnan(b)
    return a == b


def wb():
    return SimpleNamespace(context_eligible_mask=np.array([1,1,1,1,0,1],bool),
                           support_mask=np.array([0,1,1,0,0,1],bool))


def agent(cls=DDPGAgent):
    return cls(state_dim=6,action_dim=2,hidden_dim=12,actor_lr=.0003,critic_lr=.0003,
        gamma=.98,tau=.005,action_low=np.array([-1,-1]),action_high=np.array([1,1]),
        device=torch.device('cpu'),l2_action_penalty=0)


class A1Tests(unittest.TestCase):
    def test_config_only_declared_arm_difference(self):
        for seed in SEEDS:
            configs=[make_config(a,seed,'/tmp/a1-config','pilot').as_dict() for a in ARMS]
            for config in configs: config.pop('assignment_rule')
            self.assertTrue(all(c==configs[0] for c in configs))
            self.assertEqual(configs[0]['start_steps'],10000)
            self.assertEqual(configs[0]['warmup_steps'],5000)

    def test_smoke_preserves_scientific_settings(self):
        a=make_config('saber',101,'/tmp/a1-config','pilot').as_dict()
        b=make_config('saber',101,'/tmp/a1-config','smoke').as_dict()
        a.pop('total_steps'); b.pop('total_steps')
        self.assertEqual(a,b)

    def test_execution_schedule(self):
        rows=schedule()
        self.assertEqual(len(rows),12)
        self.assertEqual(rows,schedule())
        for seed in SEEDS:
            self.assertEqual({r['arm'] for r in rows if r['seed']==seed},set(ARMS))
        for arm in ARMS:
            self.assertEqual(len({r['position'] for r in rows if r['arm']==arm}),3)

    def test_checkpoint_schedule_before_learning(self):
        self.assertEqual(checkpoints(10200),[0,4800,5000,6000,8000,10000,10200])
        self.assertTrue(all(s % 200==0 for s in checkpoints(500000)))
        self.assertTrue(all(probe_due(s,500000) for s in checkpoints(500000)))
        self.assertTrue(probe_due(5100,500000)); self.assertFalse(probe_due(5101,500000))

    def test_selector_native_parity_and_rng(self):
        native=reference('assignment_comparison')
        for rule in ('support','random','td_topk'):
            for td in [np.ones(6),np.arange(6),np.array([0,1,1,1,0,1])]:
                left,right=wb(),wb()
                r1,r2=np.random.default_rng(21),np.random.default_rng(21)
                t1,t2=np.random.default_rng(22),np.random.default_rng(22)
                a=AssignmentSelector(left,rule,r1,t1); b=native.AssignmentSelector(right,rule,r2,t2)
                self.assertTrue(np.array_equal(a(td),b(td)))
                self.assertEqual(a.stats,b.stats)
                self.assertEqual(r1.bit_generator.state,r2.bit_generator.state)
                self.assertEqual(t1.bit_generator.state,t2.bit_generator.state)

    def test_uniform_does_not_use_support_for_coefficients(self):
        data=wb()
        selector=AssignmentSelector(data,'uniform',np.random.default_rng(1),np.random.default_rng(2))
        self.assertTrue(np.array_equal(selector(np.arange(6)),np.ones(6)))
        self.assertEqual(selector.stats['support_count'],3)
        self.assertTrue(np.array_equal(data.omega_actor,np.ones(6)))

    def test_empty_and_saturated_selection(self):
        for e,m in [(np.ones(6,bool),np.zeros(6,bool)),(np.ones(6,bool),np.ones(6,bool))]:
            for rule in ('uniform','support','random','td_topk'):
                data=SimpleNamespace(context_eligible_mask=e,support_mask=m)
                sel=AssignmentSelector(data,rule,np.random.default_rng(1),np.random.default_rng(2))
                self.assertTrue(np.array_equal(sel(np.ones(6)),np.ones(6)))

    def test_uniform_matches_native_her_updates_and_moments(self):
        torch.manual_seed(8)
        a=agent(); b=agent(reference('agent').DDPGAgent)
        b.load_state_dict(copy.deepcopy(a.state_dict()))
        gen=torch.Generator().manual_seed(44)
        for _ in range(10):
            kwargs=dict(state=torch.randn(6,6,generator=gen),action_exec=torch.randn(6,2,generator=gen).tanh(),
                reward=-torch.ones(6),next_state=torch.randn(6,6,generator=gen),nonterminal=torch.ones(6),
                omega_critic=torch.ones(6),omega_actor=torch.ones(6),fail_on_nonfinite=True)
            selector=AssignmentSelector(wb(),'uniform',np.random.default_rng(1),np.random.default_rng(2))
            a.update(**kwargs,critic_weight_selector=selector)
            b.update(**kwargs)
            self.assertTrue(equal_tree(a.state_dict(),b.state_dict()))

    def test_hook_does_not_change_native_updates(self):
        torch.manual_seed(9)
        a=agent(); b=agent(reference('agent').DDPGAgent)
        b.load_state_dict(copy.deepcopy(a.state_dict()))
        gen=torch.Generator().manual_seed(48)
        for _ in range(5):
            kwargs=dict(state=torch.randn(6,6,generator=gen),action_exec=torch.randn(6,2,generator=gen).tanh(),
                reward=-torch.ones(6),next_state=torch.randn(6,6,generator=gen),nonterminal=torch.ones(6),
                omega_critic=torch.ones(6),omega_actor=torch.ones(6),fail_on_nonfinite=True)
            a.update(**kwargs,diagnostic_hook=lambda net,x:policy_measure(net,x,3))
            b.update(**kwargs)
            self.assertTrue(equal_tree(a.state_dict(),b.state_dict()))

    def test_probe_preserves_parameters_moments_grad_buffers_and_rng(self):
        torch.manual_seed(15); a=agent()
        for p in list(a.actor.parameters())+list(a.critic.parameters()): p.grad=torch.full_like(p,.7)
        before=copy.deepcopy(a.state_dict())
        buffers=[p.grad.clone() for p in list(a.actor.parameters())+list(a.critic.parameters())]
        rng=torch.get_rng_state().clone()
        policy_measure(a,np.ones((4,6),np.float32),4)
        self.assertTrue(equal_tree(before,a.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(g,p.grad) for g,p in zip(buffers,list(a.actor.parameters())+list(a.critic.parameters()))))

    def test_saturated_actor_has_zero_gradient_despite_critic_gradient(self):
        actor=Actor(1,1,2)
        with torch.no_grad():
            for p in actor.parameters(): p.zero_()
            actor.net[-1].bias.fill_(100)
        class SumCritic(torch.nn.Module):
            def forward(self,s,a): return a.sum(dim=1)
        row,_=policy_measure(SimpleNamespace(actor=actor,critic=SumCritic(),device='cpu'),np.ones((4,1),np.float32),4)
        self.assertEqual(row['actor_batch_gradient_l2'],0)
        self.assertEqual(row['tanh_derivative_exact_zero_fraction'],1)
        self.assertEqual(row['critic_action_gradient_l2_mean'],1)

    def test_batch_cancellation_distinguished_from_individual_zeros(self):
        class SmallActor(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.net=torch.nn.Linear(1,1,bias=False)
                with torch.no_grad(): self.net.weight.zero_()
        class SumCritic(torch.nn.Module):
            def forward(self,s,a): return a.sum(dim=1)
        row,_=policy_measure(SimpleNamespace(actor=SmallActor(),critic=SumCritic(),device='cpu'),np.array([[-1],[1]],np.float32),2)
        self.assertEqual(row['actor_batch_gradient_l2'],0)
        self.assertEqual(row['individual_actor_gradient_zero_fraction'],0)
        self.assertEqual(row['tanh_derivative_exact_zero_fraction'],0)

    def test_double_reduction_preserves_tiny_gradient(self):
        self.assertGreater(tensor_norm([torch.tensor([1e-30])]),0)

    def test_rng_guard_restores_global_streams(self):
        random.seed(3); np.random.seed(4); torch.manual_seed(5)
        before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        with preserve_rng():
            random.random(); np.random.rand(3); torch.rand(3)
        self.assertTrue(equal_tree(before,(random.getstate(),np.random.get_state(),torch.get_rng_state())))


if __name__ == '__main__':
    unittest.main()
