from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import research_daily_opportunity_model as base
from research_daily_opportunity_ranked import AdaptiveThemeOpportunityNet,ranked_loss,ranked_variant


class RankedDailyOpportunityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.x=torch.randn(12,20,6)
        self.g=torch.eye(12).to_sparse()

    def test_finite_sparse_backward_and_native_outputs(self):
        model=AdaptiveThemeOpportunityNet(6)
        y=model(self.x,self.g)
        r=torch.randn(12,3)*.1; d=torch.rand(12,3)*.1
        loss=ranked_loss(y,r,d)
        loss['total'].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertTrue((y['downside']>=0).all())
        self.assertTrue(torch.isfinite(loss['rank']))
        mixed=(y['gate'][:,:,None]*torch.sigmoid(y['expert_rally_logits'])).sum(1)
        torch.testing.assert_close(torch.sigmoid(y['rally_logits']),mixed)

    def test_stock_and_theme_permutation_equivariance(self):
        model=AdaptiveThemeOpportunityNet(6).eval()
        g=torch.randint(0,2,(12,4)).float(); p=torch.randperm(12); q=torch.randperm(4)
        y=model(self.x,g); z=model(self.x[p],g[p][:,q])
        for name in ('return_mu','downside','rally_logits','gate'):
            torch.testing.assert_close(y[name][p],z[name],atol=1e-6,rtol=1e-5)

    def test_unknown_graph_finite_and_v1_runtime_restored(self):
        before=base.DailyGraphOpportunityNet
        old_protocol=base.PROTOCOL.copy()
        with ranked_variant():
            self.assertIs(base.DailyGraphOpportunityNet,AdaptiveThemeOpportunityNet)
            graph=torch.sparse_coo_tensor(torch.empty((2,0),dtype=torch.long),torch.empty(0),(12,0)).coalesce()
            y=base.DailyGraphOpportunityNet(6)(self.x,graph)
            self.assertTrue(torch.isfinite(y['return_mu']).all())
        self.assertIs(base.DailyGraphOpportunityNet,before)
        self.assertEqual(base.PROTOCOL,old_protocol)

    def test_masked_future_targets_do_not_change_loss(self):
        model=AdaptiveThemeOpportunityNet(6)
        y=model(self.x,self.g); mask=torch.tensor([True]*6+[False]*6)
        r=torch.randn(12,3)*.1; d=torch.rand(12,3)*.1
        first=ranked_loss(y,r,d,mask=mask)['total']
        r[6:]=torch.nan; d[6:]=torch.nan
        second=ranked_loss(y,r,d,mask=mask)['total']
        torch.testing.assert_close(first,second)


if __name__=='__main__':
    unittest.main()
