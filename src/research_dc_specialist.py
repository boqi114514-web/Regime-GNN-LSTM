"""Fixed, causal theme-first stock-acceleration policy experiments.

These are exploratory rules after the historical period has already been
examined, not a held-out result or a fitted expected-return model.  Theme
affinity is a price association and must not be described as membership.
The complete dated theme audit is required: the subset of themes that happen
to have associated stocks is never used as the theme ranking universe.
"""

import copy
import math

import numpy as np
import pandas as pd


MODES = ('theme_acceleration', 'theme_early')
PROTOCOL = {
    'stage': 'Exploratory fixed rules after viewing the historical period; no untouched holdout claim',
    'modes': list(MODES),
    'inputs': 'Completed signal-session stock momentum and full dated DC theme audit only; no future labels or daily refresh',
    'association': 'Existing price affinity, not an assertion of true theme/industry membership; no SW inputs',
    'theme_pool': 'Every unique theme in the complete dated audit satisfying existing theme_qualifies (ret20>0 OR ret60>0), finite ret20/ret60 >-1; advancing and recovering candidates are compared together',
    'theme_acceleration': 'log(1+ret20) - (log(1+ret60)-log(1+ret20))/2: latest 20 sessions versus preceding 40 sessions at equivalent 20-session speed',
    'theme_priority': '50% ret20 percentile + 50% theme acceleration percentile across the complete qualified-theme pool; select first ceil(N/3), stable code tie break',
    'stock_acceleration': 'log(1+mom1) - (log(1+mom3)-log(1+mom1))/2: latest calendar month versus preceding two calendar months at equivalent monthly speed',
    'relative_strength': 'log(1+mom1)-log(1+associated ret20); calendar month versus approximately 20 theme sessions, a proxy rather than an exactly matched-horizon residual',
    'stock_score': '50% stock-acceleration percentile + 35% relative-strength percentile + 15% existing all-board amount percentile; percentiles use all feature-complete SH/SZ boards on that signal date',
    'stock_gate': 'Existing affinity_qualifies, liquidity_days>=10 and heat_excluded==False; finite mom1/3/6 >-1, mom1>0 and mom3>0; no positive-mom6 or liquid-top-decile requirement; selected theme required',
    'early_ablation': 'theme_early additionally requires mom1<=0.30; 0.30 is a fixed early-stage ablation, NOT a universal or empirically optimal threshold',
    'execution': 'All boards enter ranks; downstream executor alone limits new buys to permitted boards and 100-share lots; original advance-phase exits unchanged',
    'selection_scope': 'Same rule for every date, with no named stock, industry or month whitelist and no parameter selection using subsequent returns',
}

MEMBER_MODES = ('member_acceleration', 'member_not_extended')
MEMBER_PROTOCOL = {
    'stage': 'Fixed exact-membership follow-up registered after the first four new-account feedback; previously examined history, NOT a holdout',
    'modes': list(MEMBER_MODES),
    'inputs': 'Existing exact completed-session top-five dated member_theme_code/score/name plus the complete DC theme audit; no current-member backfill or future labels',
    'association': 'Exact dated top-five membership, not price affinity; existing affinity fields retained unchanged for audit only; no SW inputs',
    'theme_gate': 'Exact-date catalogue match, exact original member score/name and theme_selected==True with theme_qualifies==True; no new top-third theme gate',
    'stock_acceleration': PROTOCOL['stock_acceleration'],
    'relative_strength': 'log(1+mom1)-log(1+true-member ret20); calendar month versus approximately 20 theme sessions, a proxy rather than an exactly matched-horizon residual',
    'stock_score': '50% stock-acceleration percentile + 35% true-member-relative-strength percentile + 15% existing all-board amount percentile; all feature-complete exact-member SH/SZ boards participate together on each signal',
    'stock_gate': 'Exact dated top-five member; liquidity_days>=10 and heat_excluded==False; finite mom1/3/6 >-1, mom1>0 and mom3>0; no positive-mom6, price-affinity or liquid-top-decile requirement; retain original joint-extreme heat exclusion',
    'moderate_extension_ablation': 'member_not_extended additionally requires mom1<=0.50; 0.50 is a fixed moderate-extension ablation after four-account feedback, NOT a universal optimal threshold; no further threshold tuning',
    'execution': PROTOCOL['execution'],
    'selection_scope': PROTOCOL['selection_scope'],
}

RELATIVE_MEMBER_MODES = ('member_relative_acceleration', 'member_relative_moderate')
RELATIVE_MEMBER_PROTOCOL = {
    'stage': 'Fixed member-peer-relative follow-up registered after scalar ranking feedback; previously examined history, NOT a holdout',
    'modes': list(RELATIVE_MEMBER_MODES),
    'inputs': MEMBER_PROTOCOL['inputs'],
    'association': 'Exact dated top-five membership using the existing highest-selected-theme single mapping per stock; not a complete overlapping multi-theme graph; no SW inputs',
    'theme_gate': MEMBER_PROTOCOL['theme_gate'],
    'group_population': 'Every exact-top-five, feature-complete SH/SZ mapped member in each (signal_date,member_theme_code); group medians and counts are computed BEFORE positive-momentum, heat and moderate-extension entry gates, including negative-momentum members; >=5 members required',
    'stock_acceleration': PROTOCOL['stock_acceleration'],
    'relative_acceleration': 'Stock geometric monthly acceleration minus its same-date mapped-member-group median acceleration',
    'relative_momentum': 'log(1+mom1) minus its same-date mapped-member-group median log(1+mom1)',
    'stock_score': '50% relative-acceleration percentile + 35% relative-momentum percentile + 15% existing all-board amount percentile; rank all feature-complete exact mapped members with valid >=5-member groups together on each signal',
    'stock_gate': MEMBER_PROTOCOL['stock_gate'] + '; valid >=5-member group required',
    'moderate_extension_ablation': 'member_relative_moderate additionally requires mom1<=0.50; the SAME fixed moderate threshold as member_not_extended, with no further threshold search or optimality claim',
    'execution': PROTOCOL['execution'],
    'selection_scope': PROTOCOL['selection_scope'],
}

LIQUID_LEADER_BASE_MODES = {
    'member_liquid_leader': 'member_not_extended',
    'member_relative_liquid_leader': 'member_relative_moderate',
}
LIQUID_LEADER_PROTOCOL = {
    'stage': 'Fixed liquid/rank-first follow-up after viewing ten completed new accounts; explicitly exploratory, NOT an untouched holdout or a paper-proven return claim',
    'modes': list(LIQUID_LEADER_BASE_MODES),
    'base_modes': dict(LIQUID_LEADER_BASE_MODES),
    'inputs': MEMBER_PROTOCOL['inputs'],
    'association': RELATIVE_MEMBER_PROTOCOL['association'],
    'base_scoring': 'Reuse the complete corresponding member_not_extended or member_relative_moderate output without recalculating score percentiles, mapped-peer medians or any base entry gate',
    'liquidity_gate': 'Add only final eligibility amount_percentile>=0.95, inclusive, using the existing completed-signal monthly mean-positive-amount percentile across all SH/SZ boards with at least 10 positive observations; fixed top-five-percent rule, NOT an optimal or paper-proven threshold',
    'stock_gate': 'All corresponding base momentum, exact-member, heat, valid-peer-group and fixed mom1<=0.50 gates remain unchanged; apply liquidity only after the base scoring population is frozen',
    'execution': 'Downstream rank-first single-stock allocator chooses the highest fixed leadership score among permitted, opening-executable mainboards with an affordable 100-share lot; stable stock-code tie break, maximum affordable lots of that stock only, unused cash allowed; original account budget, no topups and advance exits unchanged',
    'selection_scope': 'One fixed rule for every signal; no stock, named theme or date whitelist, no subsequent-month labels consulted by the selector, no liquidity-threshold grid search or unseen-validation claim',
}


def protocol_for_mode(mode):
    """Return a detached protocol without changing either original policy."""
    if mode in MODES:
        return copy.deepcopy(PROTOCOL)
    if mode in MEMBER_MODES:
        return copy.deepcopy(MEMBER_PROTOCOL)
    if mode in RELATIVE_MEMBER_MODES:
        return copy.deepcopy(RELATIVE_MEMBER_PROTOCOL)
    if mode in LIQUID_LEADER_BASE_MODES:
        return copy.deepcopy(LIQUID_LEADER_PROTOCOL)
    raise ValueError(f'Unknown specialist mode: {mode}')


PROTOCOL_for_mode = protocol_for_mode

_STOCK_COLUMNS = [
    'signal_date', 'month', 'ts_code', 'mom1', 'mom3', 'mom6',
    'amount_percentile', 'liquidity_days', 'affinity_theme_code',
    'affinity_theme_score', 'affinity_ret20', 'affinity_ret60',
    'affinity_qualifies', 'heat_excluded', 'leadership_score', 'eligible',
]
_THEME_COLUMNS = ['signal_date', 'ts_code', 'ret20', 'ret60', 'theme_qualifies']


def _require(frame, columns, label):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f'{label} must be a DataFrame')
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f'{label} missing columns: {sorted(missing)}')


def _dates(values, label):
    dates = pd.to_datetime(values, errors='coerce')
    if (dates.isna().any() or dates.dt.tz is not None
            or not dates.eq(dates.dt.normalize()).all()):
        raise ValueError(f'{label} must contain timezone-free date-only values')
    return dates


def _theme_priorities(theme_audit, signals):
    """Rank unique contemporaneous themes independently of stock coverage."""
    _require(theme_audit, _THEME_COLUMNS, 'Complete dated theme audit')
    t = theme_audit[_THEME_COLUMNS].copy()
    t['signal_date'] = _dates(t.signal_date, 'Theme signal dates')
    # Rows from later signals never enter validation of dated theme values,
    # the percentile population or the number of admitted themes.
    t = t[t.signal_date.isin(signals)].copy().reset_index(drop=True)
    if t.ts_code.isna().any() or t.duplicated(['signal_date', 'ts_code']).any():
        raise ValueError('Missing or duplicate dated theme keys')
    for field in ('ret20', 'ret60'):
        t[field] = pd.to_numeric(t[field], errors='coerce').astype(float)
    complete = np.isfinite(t[['ret20', 'ret60']]).all(axis=1)
    pool = (complete & t.theme_qualifies.eq(True).fillna(False)
            & t[['ret20', 'ret60']].gt(-1).all(axis=1)
            & (t.ret20.gt(0) | t.ret60.gt(0)))
    t['specialist_theme_acceleration'] = np.nan
    t['specialist_theme_priority'] = np.nan
    t['specialist_theme_rank'] = np.nan
    t['specialist_theme_count'] = 0
    t['specialist_theme_selected'] = False
    if pool.any():
        latest = np.log1p(t.loc[pool, 'ret20'])
        previous = (np.log1p(t.loc[pool, 'ret60']) - latest) / 2
        t.loc[pool, 'specialist_theme_acceleration'] = latest - previous
        ranks = t.loc[pool].groupby('signal_date')[
            ['ret20', 'specialist_theme_acceleration']].rank(pct=True)
        t.loc[pool, 'specialist_theme_priority'] = (
            .50 * ranks.ret20 + .50 * ranks.specialist_theme_acceleration)
        ordered = t[pool].sort_values(
            ['signal_date', 'specialist_theme_priority', 'ts_code'],
            ascending=[True, False, True])
        rank = ordered.groupby('signal_date').cumcount() + 1
        count = ordered.groupby('signal_date').ts_code.transform('size')
        t.loc[ordered.index, 'specialist_theme_rank'] = rank
        t.loc[ordered.index, 'specialist_theme_count'] = count
        t.loc[ordered.index, 'specialist_theme_selected'] = rank.le(
            count.map(lambda n: math.ceil(n / 3)))
    return t.rename(columns={
        'ts_code': 'affinity_theme_code', 'ret20': 'specialist_dated_ret20',
        'ret60': 'specialist_dated_ret60', 'theme_qualifies': 'specialist_dated_theme_qualifies',
    })


def specialist_candidates(scored, mode='theme_acceleration', theme_audit=None):
    """Return every input row with fixed theme-first scores and audit fields.

    ``theme_audit`` is mandatory despite the compatibility-friendly signature.
    An absent audit, duplicate keys or mismatched linked theme observations is
    an input error; absent/invalid individual numeric features fail closed.
    Input frames are not modified.  Unknown columns (including future-return
    diagnostics if present) are retained but never consulted.
    """
    if mode in LIQUID_LEADER_BASE_MODES:
        return _liquid_leader_candidates(scored, mode, theme_audit)
    if mode in RELATIVE_MEMBER_MODES:
        return _relative_member_candidates(scored, mode, theme_audit)
    if mode in MEMBER_MODES:
        return _member_candidates(scored, mode, theme_audit)
    if mode not in MODES:
        raise ValueError(f'Unknown specialist mode: {mode}')
    _require(scored, _STOCK_COLUMNS, 'Structural stock audit')
    if theme_audit is None:
        raise ValueError('Complete dated theme audit is required; associated themes are not the ranking universe')
    c = scored.copy().reset_index(drop=True)
    c['signal_date'] = _dates(c.signal_date, 'Stock signal dates')
    c['month'] = _dates(c.month, 'Stock signal months')
    if (c.ts_code.isna().any()
            or c.duplicated(['signal_date', 'ts_code']).any()
            or c.duplicated(['month', 'ts_code']).any()):
        raise ValueError('Missing or duplicate stock signal keys')
    if not c.ts_code.astype('string').str.fullmatch(
            r'(?:0\d{5}\.SZ|3\d{5}\.SZ|6\d{5}\.SH)', na=False).all():
        raise ValueError('Specialist stock universe must contain SH/SZ stock keys')
    if not c.month.dt.to_period('M').eq(c.signal_date.dt.to_period('M')).all():
        raise ValueError('Stock signal date is outside its calendar month')
    if c.groupby('month').signal_date.nunique().gt(1).any():
        raise ValueError('Multiple completed signal sessions in a calendar month')
    if 'date' in c:
        quote_day = _dates(c.date, 'Stock quote dates')
        if (quote_day.gt(c.signal_date).any()
                or not quote_day.dt.to_period('M').eq(c.month.dt.to_period('M')).all()):
            raise ValueError('Stock quote contains future or out-of-month evidence')
    c['structural_eligible'] = c.eligible.eq(True).fillna(False)
    c['structural_leadership_score'] = c.leadership_score
    numeric = ['mom1', 'mom3', 'mom6', 'amount_percentile', 'liquidity_days',
               'affinity_theme_score', 'affinity_ret20', 'affinity_ret60']
    for field in numeric:
        c[field] = pd.to_numeric(c[field], errors='coerce').astype(float)
    themes = _theme_priorities(theme_audit, c.signal_date.unique())
    c = c.merge(themes, on=['signal_date', 'affinity_theme_code'],
                how='left', validate='many_to_one')
    associated = c.affinity_qualifies.eq(True).fillna(False)
    for source, bound in [('affinity_ret20', 'specialist_dated_ret20'),
                          ('affinity_ret60', 'specialist_dated_ret60')]:
        known = np.isfinite(c[bound])
        # Missing stock values fail closed, but a different finite observation
        # would assert an association based on an inconsistent dated catalogue.
        mismatch = associated & np.isfinite(c[source]) & known & ~np.isclose(
            c[source], c[bound], atol=1e-12, rtol=0)
        if mismatch.any():
            raise ValueError('Associated theme return does not match complete dated theme audit')
    if (associated & c.specialist_dated_theme_qualifies.isna()).any():
        raise ValueError('Associated theme is missing from complete dated theme audit')
    feature_complete = (
        np.isfinite(c[numeric]).all(axis=1)
        & c[['mom1', 'mom3', 'mom6']].gt(-1).all(axis=1)
        & c[['affinity_ret20', 'affinity_ret60']].gt(-1).all(axis=1)
        & c.amount_percentile.between(0, 1)
        & c.affinity_theme_score.between(0, 1)
        & c.liquidity_days.ge(10) & c.liquidity_days.mod(1).eq(0))
    c['specialist_acceleration'] = np.nan
    c['specialist_relative_strength'] = np.nan
    if feature_complete.any():
        latest = np.log1p(c.loc[feature_complete, 'mom1'])
        previous = (np.log1p(c.loc[feature_complete, 'mom3']) - latest) / 2
        c.loc[feature_complete, 'specialist_acceleration'] = latest - previous
        c.loc[feature_complete, 'specialist_relative_strength'] = (
            latest - np.log1p(c.loc[feature_complete, 'affinity_ret20']))
    ranks = c.loc[feature_complete].groupby('signal_date')[[
        'specialist_acceleration', 'specialist_relative_strength']].rank(pct=True)
    c['specialist_acceleration_percentile'] = ranks.specialist_acceleration
    c['specialist_relative_percentile'] = ranks.specialist_relative_strength
    c['specialist_stock_score'] = (.50 * c.specialist_acceleration_percentile
        + .35 * c.specialist_relative_percentile + .15 * c.amount_percentile)
    c['specialist_feature_complete'] = feature_complete
    c['specialist_stage_qualifies'] = (
        c.mom1.gt(0) & c.mom3.gt(0)
        & (c.mom1.le(.30) if mode == 'theme_early' else True))
    c['specialist_eligible'] = (
        feature_complete & associated & c.heat_excluded.eq(False).fillna(False)
        & ~(c.mom3.gt(1.5) & c.mom1.gt(.30))
        & c.specialist_dated_theme_qualifies.eq(True).fillna(False)
        & c.specialist_theme_selected.eq(True).fillna(False)
        & c.specialist_stage_qualifies & np.isfinite(c.specialist_stock_score))
    c['eligible'] = c.specialist_eligible
    c['leadership_score'] = c.specialist_stock_score
    c['phase'] = 'advance'
    c['specialist_mode'] = mode
    return c.sort_values(['month', 'ts_code']).reset_index(drop=True)


def _liquid_leader_candidates(scored, mode, theme_audit):
    """Final liquidity gate only; leave every base population and score intact."""
    c = specialist_candidates(scored, LIQUID_LEADER_BASE_MODES[mode], theme_audit)
    c['specialist_base_eligible'] = c.eligible.copy()
    c['specialist_liquid_leader_qualifies'] = c.amount_percentile.ge(.95).fillna(False)
    c['specialist_eligible'] = c.specialist_base_eligible & c.specialist_liquid_leader_qualifies
    c['eligible'] = c.specialist_eligible
    c['specialist_mode'] = mode
    return c


def _member_candidates(scored, mode, theme_audit):
    """Exact dated top-five membership policies; never consult affinity values."""
    fields = ['signal_date', 'month', 'ts_code', 'mom1', 'mom3', 'mom6',
              'amount_percentile', 'liquidity_days', 'member_theme_code',
              'member_theme_score', 'member_theme_name', 'heat_excluded',
              'leadership_score', 'eligible']
    _require(scored, fields, 'Exact-member stock audit')
    if theme_audit is None:
        raise ValueError('Complete dated theme audit is required for exact membership')
    _require(theme_audit, _THEME_COLUMNS + ['theme_selected', 'theme_score', 'name'],
             'Complete dated member theme audit')
    c = scored.copy().reset_index(drop=True)
    c['signal_date'] = _dates(c.signal_date, 'Stock signal dates')
    c['month'] = _dates(c.month, 'Stock signal months')
    if (c.ts_code.isna().any()
            or c.duplicated(['signal_date', 'ts_code']).any()
            or c.duplicated(['month', 'ts_code']).any()):
        raise ValueError('Missing or duplicate stock signal keys')
    if not c.ts_code.astype('string').str.fullmatch(
            r'(?:0\d{5}\.SZ|3\d{5}\.SZ|6\d{5}\.SH)', na=False).all():
        raise ValueError('Specialist stock universe must contain SH/SZ stock keys')
    if not c.month.dt.to_period('M').eq(c.signal_date.dt.to_period('M')).all():
        raise ValueError('Stock signal date is outside its calendar month')
    if c.groupby('month').signal_date.nunique().gt(1).any():
        raise ValueError('Multiple completed signal sessions in a calendar month')
    if 'date' in c:
        quote_day = _dates(c.date, 'Stock quote dates')
        if (quote_day.gt(c.signal_date).any()
                or not quote_day.dt.to_period('M').eq(c.month.dt.to_period('M')).all()):
            raise ValueError('Stock quote contains future or out-of-month evidence')
    c['structural_eligible'] = c.eligible.eq(True).fillna(False)
    c['structural_leadership_score'] = c.leadership_score
    numeric = ['mom1', 'mom3', 'mom6', 'amount_percentile', 'liquidity_days', 'member_theme_score']
    for field in numeric:
        c[field] = pd.to_numeric(c[field], errors='coerce').astype(float)
    t = theme_audit[_THEME_COLUMNS + ['theme_selected', 'theme_score', 'name']].copy()
    t['signal_date'] = _dates(t.signal_date, 'Theme signal dates')
    t = t[t.signal_date.isin(c.signal_date.unique())].copy().reset_index(drop=True)
    if t.ts_code.isna().any() or t.duplicated(['signal_date', 'ts_code']).any():
        raise ValueError('Missing or duplicate dated theme keys')
    for field in ('ret20', 'ret60', 'theme_score'):
        t[field] = pd.to_numeric(t[field], errors='coerce').astype(float)
    t = t.rename(columns={
        'ts_code': 'member_theme_code', 'ret20': 'specialist_member_ret20',
        'ret60': 'specialist_member_ret60', 'theme_score': 'specialist_member_dated_score',
        'name': 'specialist_member_dated_name',
        'theme_selected': 'specialist_member_dated_selected',
        'theme_qualifies': 'specialist_member_dated_qualifies',
    })
    c = c.merge(t, on=['signal_date', 'member_theme_code'], how='left', validate='many_to_one')
    assigned = c.member_theme_code.notna()
    if (assigned & c.specialist_member_dated_selected.isna()).any():
        raise ValueError('Exact member theme is missing from complete dated theme audit')
    same_score = np.isclose(c.member_theme_score, c.specialist_member_dated_score, atol=1e-12, rtol=0)
    if (assigned & np.isfinite(c.member_theme_score)
            & np.isfinite(c.specialist_member_dated_score) & ~same_score).any():
        raise ValueError('Exact member theme score does not match complete dated theme audit')
    named = assigned & c.member_theme_name.notna() & c.specialist_member_dated_name.notna()
    if (named & c.member_theme_name.ne(c.specialist_member_dated_name)).any():
        raise ValueError('Exact member theme name does not match complete dated theme audit')
    feature_complete = (
        assigned & np.isfinite(c[numeric]).all(axis=1)
        & c[['mom1', 'mom3', 'mom6']].gt(-1).all(axis=1)
        & np.isfinite(c[['specialist_member_ret20', 'specialist_member_ret60',
                          'specialist_member_dated_score']]).all(axis=1)
        & c[['specialist_member_ret20', 'specialist_member_ret60']].gt(-1).all(axis=1)
        & c.amount_percentile.between(0, 1) & c.member_theme_score.between(0, 1)
        & c.member_theme_name.notna() & c.specialist_member_dated_name.notna()
        & c.liquidity_days.ge(10) & c.liquidity_days.mod(1).eq(0))
    c['specialist_acceleration'] = np.nan
    c['specialist_relative_strength'] = np.nan
    if feature_complete.any():
        latest = np.log1p(c.loc[feature_complete, 'mom1'])
        previous = (np.log1p(c.loc[feature_complete, 'mom3']) - latest) / 2
        c.loc[feature_complete, 'specialist_acceleration'] = latest - previous
        c.loc[feature_complete, 'specialist_relative_strength'] = (
            latest - np.log1p(c.loc[feature_complete, 'specialist_member_ret20']))
    ranks = c.loc[feature_complete].groupby('signal_date')[[
        'specialist_acceleration', 'specialist_relative_strength']].rank(pct=True)
    c['specialist_acceleration_percentile'] = ranks.specialist_acceleration
    c['specialist_relative_percentile'] = ranks.specialist_relative_strength
    c['specialist_stock_score'] = (.50 * c.specialist_acceleration_percentile
        + .35 * c.specialist_relative_percentile + .15 * c.amount_percentile)
    c['specialist_feature_complete'] = feature_complete
    c['specialist_stage_qualifies'] = (c.mom1.gt(0) & c.mom3.gt(0)
        & (c.mom1.le(.50) if mode == 'member_not_extended' else True))
    c['specialist_member_qualifies'] = (
        c.specialist_member_dated_selected.eq(True).fillna(False)
        & c.specialist_member_dated_qualifies.eq(True).fillna(False)
        & (c.specialist_member_ret20.gt(0) | c.specialist_member_ret60.gt(0)))
    c['specialist_eligible'] = (feature_complete & c.specialist_member_qualifies
        & c.heat_excluded.eq(False).fillna(False)
        & ~(c.mom3.gt(1.5) & c.mom1.gt(.30))
        & c.specialist_stage_qualifies & np.isfinite(c.specialist_stock_score))
    c['eligible'] = c.specialist_eligible
    c['leadership_score'] = c.specialist_stock_score
    c['ind_code'] = c.member_theme_code
    c['theme_name'] = c.member_theme_name
    c['theme_score'] = c.member_theme_score
    c['classification_basis'] = 'exact_dated_top_five_membership'
    c['phase'] = 'advance'
    c['specialist_mode'] = mode
    return c.sort_values(['month', 'ts_code']).reset_index(drop=True)


def _relative_member_candidates(scored, mode, theme_audit):
    """Rank leadership relative to all complete mapped peers, not an index."""
    c = _member_candidates(scored, 'member_acceleration', theme_audit)
    # Do not use c.eligible here: it has already applied positive momentum and
    # heat. Those entry restrictions must not censor the peer population.
    pool = c.specialist_feature_complete & c.specialist_member_qualifies
    keys = ['signal_date', 'member_theme_code']
    c['specialist_member_log_mom1'] = np.nan
    c.loc[pool, 'specialist_member_log_mom1'] = np.log1p(c.loc[pool, 'mom1'])
    c['specialist_member_group_count'] = 0
    c['specialist_member_median_log_mom1'] = np.nan
    c['specialist_member_median_acceleration'] = np.nan
    if pool.any():
        groups = c.loc[pool].groupby(keys)
        c.loc[pool, 'specialist_member_group_count'] = groups.ts_code.transform('size')
        c.loc[pool, 'specialist_member_median_log_mom1'] = groups.specialist_member_log_mom1.transform('median')
        c.loc[pool, 'specialist_member_median_acceleration'] = groups.specialist_acceleration.transform('median')
    valid = pool & c.specialist_member_group_count.ge(5)
    c['specialist_member_group_valid'] = valid
    c['specialist_group_relative_acceleration'] = (
        c.specialist_acceleration - c.specialist_member_median_acceleration).where(valid)
    c['specialist_group_relative_mom1'] = (
        c.specialist_member_log_mom1 - c.specialist_member_median_log_mom1).where(valid)
    ranks = c.loc[valid].groupby('signal_date')[[
        'specialist_group_relative_acceleration', 'specialist_group_relative_mom1']].rank(pct=True)
    c['specialist_group_relative_acceleration_percentile'] = ranks.specialist_group_relative_acceleration
    c['specialist_group_relative_mom1_percentile'] = ranks.specialist_group_relative_mom1
    c['specialist_index_relative_score'] = c.specialist_stock_score
    c['specialist_stock_score'] = (.50 * c.specialist_group_relative_acceleration_percentile
        + .35 * c.specialist_group_relative_mom1_percentile + .15 * c.amount_percentile)
    c['specialist_stage_qualifies'] = (c.mom1.gt(0) & c.mom3.gt(0)
        & (c.mom1.le(.50) if mode == 'member_relative_moderate' else True))
    c['specialist_eligible'] = (valid & c.specialist_stage_qualifies
        & c.heat_excluded.eq(False).fillna(False)
        & ~(c.mom3.gt(1.5) & c.mom1.gt(.30))
        & np.isfinite(c.specialist_stock_score))
    c['eligible'] = c.specialist_eligible
    c['leadership_score'] = c.specialist_stock_score
    c['specialist_mode'] = mode
    return c.sort_values(['month', 'ts_code']).reset_index(drop=True)
