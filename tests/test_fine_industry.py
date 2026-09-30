import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from research_fine_industry import validate_members, attach_membership, fine_scores


class FineIndustryTests(unittest.TestCase):
    def test_membership_dates_parent_and_ambiguity(self):
        day=pd.Timestamp('2023-01-31')
        c=pd.DataFrame(dict(month=[day]*4,date=[day]*4,ts_code=['a','b','c','d'],l2_code=['p']*4))
        m=pd.DataFrame(dict(ts_code=['a','a','b','c','d','d'],l2_code=['p','p','p','q','p','p'],
            l3_code=['old','new','future','wrong','one','two'],
            in_date=pd.to_datetime(['2020-01-01','2023-01-31','2023-02-01','2020-01-01','2020-01-01','2020-01-01']),
            out_date=pd.to_datetime(['2023-01-31',None,None,None,None,None])))
        result,audit=attach_membership(c,m)
        self.assertEqual(result.set_index('ts_code').loc['a','l3_code'],'new')
        self.assertTrue(result.set_index('ts_code').loc[['b','c','d'],'l3_code'].isna().all())
        self.assertEqual(audit.ambiguous.iloc[0],1)
        self.assertEqual(len(result),4)

    def test_reject_bad_gateway_filter_and_missing_exit(self):
        m=pd.DataFrame(dict(l1_code=['x'],l2_code=['p'],l3_code=['g'],ts_code=['a'],
            in_date=['20200101'],out_date=[None],is_new=['N']))
        with self.assertRaisesRegex(ValueError,'Wrong L1'): validate_members(m,'y')
        with self.assertRaisesRegex(ValueError,'exit dates'): validate_members(m,'x')
        m['out_date']='20230101'
        self.assertEqual(validate_members(m,'x').out_date.iloc[0],pd.Timestamp('2023-01-01'))

    def test_all_boards_form_signal_and_future_does_not_change_prefix(self):
        day=pd.Timestamp('2023-01-31')
        codes=['000001.SZ','300001.SZ','300002.SZ','688001.SH','688002.SH']
        c=pd.DataFrame(dict(ts_code=codes,month=day,date=day,l2_code='p',mom1=.1,mom3=.3,mom6=.2,
            leadership_score=.4,phase='range',size_bucket='small',chosen_style='core',fast_score=.9))
        mem=pd.DataFrame(dict(ts_code=codes,l2_code='p',l3_code='g',in_date=pd.Timestamp('2020-01-01'),out_date=pd.NaT))
        market=pd.DataFrame(dict(breadth=[.3]),index=[day])
        result,_,groups=fine_scores(c,mem,market,'fine_router_retry')
        self.assertTrue(result.eligible.all())
        self.assertTrue(result.phase.eq('advance').all())
        self.assertEqual(groups.fine_count.iloc[0],5)
        future=c.assign(month=pd.Timestamp('2023-02-28'),date=pd.Timestamp('2023-02-28'),mom3=-.9)
        market.loc[pd.Timestamp('2023-02-28')]=.9
        full,_,_=fine_scores(pd.concat([c,future]),mem,market,'fine_router_retry')
        pd.testing.assert_frame_equal(result,full[full.month.eq(day)].reset_index(drop=True))
        self.assertFalse(full[full.month.gt(day)].eligible.any())


if __name__=='__main__': unittest.main()
