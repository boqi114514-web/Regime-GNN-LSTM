import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from research_daily_opportunity_features import DailyFeatureStore,FEATURE_NAMES
from research_daily_opportunity_leadership import leadership_factors,specialised_loss,leadership_variant
from research_daily_opportunity_ranked import AdaptiveThemeOpportunityNet


class LeadershipFactorsTests(unittest.TestCase):
    def fixture(self):
        days=pd.bdate_range('2025-01-02',periods=4)
        codes=tuple(f'60000{i}.SH' for i in range(7))
        x=np.ones((4,7,len(FEATURE_NAMES)),dtype=np.float32)*.01
        c={name:i for i,name in enumerate(FEATURE_NAMES)}
        x[:,:,c['return20']]=np.arange(7)/10
        x[:,:,c['return5']]=np.arange(7)/100
        y=np.zeros((4,7,3),dtype=np.float32)
        store=DailyFeatureStore(days,codes,x,np.ones((4,7),bool),y,y.copy(),
            np.full((4,3),np.datetime64('NaT','D')),np.zeros((4,7),dtype=np.int8))
        edges=pd.DataFrame([dict(snapshot_date=days[0],theme_code='BK0001.DC',ts_code=c) for c in codes[:6]])
        return store,edges,np.ones((4,7))*100

    def test_peer_context_self_free_and_unknown_nodes_not_excluded(self):
        store,edges,amount=self.fixture()
        context=leadership_factors(store,edges,amount)
        self.assertAlmostEqual(float(context[0,0,2]),.3,places=6)
        self.assertEqual(context[0,6,-1],0)
        changed=copy.deepcopy(store)
        changed.features[:,0,FEATURE_NAMES.index('return20')]=1.5
        other=leadership_factors(changed,edges,amount)
        np.testing.assert_allclose(context[:,0,2],other[:,0,2])
        self.assertTrue(store.valid_features[:,6].all())

    def test_future_features_labels_and_edges_cannot_change_past_factors(self):
        store,edges,amount=self.fixture()
        context=leadership_factors(store,edges,amount)
        changed=copy.deepcopy(store)
        changed.features[2:]*=10
        changed.label_returns[:]=999
        later=pd.DataFrame([dict(snapshot_date=store.dates[2],theme_code='BK0002.DC',ts_code=c)
            for c in store.stock_codes])
        other=leadership_factors(changed,pd.concat((edges,later)),amount)
        np.testing.assert_array_equal(context[:2],other[:2])

    def test_partial_own_factor_not_filled_with_profitable_value(self):
        store,edges,amount=self.fixture()
        store.features[:,0,FEATURE_NAMES.index('return20')]=np.nan
        context=leadership_factors(store,edges,amount)
        self.assertTrue(np.isnan(context[:,0,4]).all())
        self.assertTrue(np.isnan(context[:,0,6]).all())
        self.assertTrue(np.isfinite(context[:,0,2]).all())

    def test_specialised_loss_has_finite_gradients_and_ignores_missing_labels(self):
        torch.manual_seed(42)
        net=AdaptiveThemeOpportunityNet(3,hidden=4,sequence_length=2)
        output=net(torch.randn(6,2,3),torch.ones(6,1))
        y=torch.randn(6,3)*.1;r=torch.rand(6,3)*.1
        y[-1]=torch.nan;r[-1]=torch.nan
        stages=torch.tensor([0,1,2,3,-1,0]);mask=torch.tensor([1,1,1,1,1,0],dtype=torch.bool)
        losses=specialised_loss(output,y,r,mask=mask,stage_targets=stages)
        self.assertGreater(float(losses['specialisation'].detach()),0)
        losses['total'].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_scoped_protocol_restored_and_masked_checkpoint_can_fit_reload(self):
        import research_daily_opportunity_features as feature_module
        import research_daily_opportunity_model as model
        import research_daily_opportunity as pipeline
        from research_daily_opportunity_masked_features import build_masked_feature_store,encode_missingness,MaskedDailyOpportunityBatch
        from test_daily_opportunity_features import fixture_daily
        protocol=copy.deepcopy(feature_module.FEATURE_PROTOCOL)
        original_batch=pipeline.LazyDailyOpportunityBatch
        store=encode_missingness(build_masked_feature_store(fixture_daily()))
        edges=pd.DataFrame(columns=['snapshot_date','theme_code','ts_code'])
        batches=[MaskedDailyOpportunityBatch(store,i,edges,sequence_length=5,use_graph=False)
                 for i in range(60,100)]
        with tempfile.TemporaryDirectory() as folder:
            with leadership_variant():
                fitted=model.fit_walk_forward(batches,store.dates[115],
                    model.FitConfig(hidden=4,sequence_length=5,epochs=1,validation_sessions=2,minimum_train_sessions=5),device='cpu')
                file=Path(folder)/'model.pt';fitted.save(file)
                loaded=model.FittedOpportunityModel.load(file)
                batch=MaskedDailyOpportunityBatch(store,119,edges,sequence_length=5,use_graph=False)
                one=fitted.predict(batch);two=loaded.predict(batch)
                np.testing.assert_array_equal(one['return_mu'],two['return_mu'])
                self.assertTrue(np.isfinite(one['return_mu']).all())
        self.assertEqual(feature_module.FEATURE_PROTOCOL,protocol)
        self.assertIs(pipeline.LazyDailyOpportunityBatch,original_batch)


if __name__=='__main__':unittest.main()
