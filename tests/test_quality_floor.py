import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from research_quality_floor import normalize_reports, snapshot


class QualityFloorTests(unittest.TestCase):
    def data(self):
        inc=pd.DataFrame(dict(ts_code=['a']*4,report_type='1',
            end_date=['20220630','20221231','20230630','20230630'],
            ann_date=['20220830','20230420','20230830','20230830'],
            f_ann_date=['20220830','20230420','20230830','20231001'],
            n_income_attr_p=[20.,50.,30.,-100.]))
        bal=pd.DataFrame(dict(ts_code=['a'],report_type='1',end_date=['20230630'],
            ann_date=['20230830'],f_ann_date=['20230830'],total_hldr_eqy_exc_min_int=[100.]))
        return inc,bal

    def test_ttm_uses_only_known_revision_and_next_day_availability(self):
        inc,bal=self.data()
        i=normalize_reports(inc,'n_income_attr_p'); b=normalize_reports(bal,'total_hldr_eqy_exc_min_int')
        f=snapshot(i,b,pd.Timestamp('2023-08-31'))
        self.assertEqual(f.ttm_profit.iloc[0],60.)
        self.assertTrue(f.quality_ok.iloc[0])
        before=snapshot(i,b,pd.Timestamp('2023-08-30'))
        self.assertFalse(before.quality_ok.iloc[0])
        revised=snapshot(i,b,pd.Timestamp('2023-10-02'))
        self.assertEqual(revised.ttm_profit.iloc[0],-70.)
        self.assertFalse(revised.quality_ok.iloc[0])
        pd.testing.assert_frame_equal(f,snapshot(i[i.available.le('2023-08-31')],b,pd.Timestamp('2023-08-31')))

    def test_missing_component_or_negative_assets_blocks(self):
        inc,bal=self.data(); inc=inc[inc.end_date.ne('20220630')]
        i=normalize_reports(inc,'n_income_attr_p');b=normalize_reports(bal,'total_hldr_eqy_exc_min_int')
        missing=snapshot(i,b,pd.Timestamp('2023-08-31'))
        self.assertFalse(missing.quality_ok.iloc[0])
        self.assertTrue(missing.report_quality_ok.iloc[0])
        inc,bal=self.data();bal['total_hldr_eqy_exc_min_int']=-1.
        self.assertFalse(snapshot(normalize_reports(inc,'n_income_attr_p'),
            normalize_reports(bal,'total_hldr_eqy_exc_min_int'),pd.Timestamp('2023-08-31')).quality_ok.iloc[0])


if __name__=='__main__': unittest.main()
