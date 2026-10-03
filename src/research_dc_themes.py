"""Dated Eastmoney themes and stock price/volume signals, with isolated inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.execution_data import ROOT


ENTRY_VARIANTS = ('dc_reversal_2026_retry', 'dc_weekly_2026_retry', 'dc_weekly_reversal_2026_retry')
LINEAR_FORECAST_VARIANTS = ('dc_expected_profit_2026_retry', 'dc_peer_forecast_2026_retry', 'dc_rally_2026_retry')
FORECAST_VARIANTS = ('dc_forecast_2026_retry',)+LINEAR_FORECAST_VARIANTS
ACCEL_VARIANTS = ('dc_acceleration_2026_retry', 'dc_early_2026_retry')
MEMBER_ACCEL_VARIANTS = ('dc_member_acceleration_2026_retry', 'dc_member_moderate_2026_retry')
RELATIVE_MEMBER_VARIANTS = ('dc_member_relative_2026_retry', 'dc_member_relative_moderate_2026_retry', 'dc_member_relative_leader_2026_retry')
LEADER_FIRST_VARIANTS = ('dc_member_leader_2026_retry', 'dc_member_relative_leader_2026_retry')
MEMBER_ACCEL_VARIANTS += RELATIVE_MEMBER_VARIANTS+('dc_member_leader_2026_retry',)
ACCEL_VARIANTS += MEMBER_ACCEL_VARIANTS
CONTEXT_VARIANTS = ('dc_context_rank_2026_retry', 'dc_context_member_2026_retry')
LOCAL_CONTEXT_VARIANTS = ('dc_local_rank_2026_retry', 'dc_local_member_2026_retry')
CONTEXT_VARIANTS += LOCAL_CONTEXT_VARIANTS
FORECAST_VARIANTS += CONTEXT_VARIANTS
SCOPED_ACTION_VARIANTS = ENTRY_VARIANTS+FORECAST_VARIANTS+ACCEL_VARIANTS
WEEKLY_VARIANTS = ('dc_weekly_2026_retry', 'dc_weekly_reversal_2026_retry')
REVERSAL_VARIANTS = ('dc_reversal_2026_retry', 'dc_weekly_reversal_2026_retry')
VARIANTS = ('dc_theme_retry', 'dc_theme_2026_retry', 'dc_liquid_2026_retry', 'dc_affinity_2026_retry', 'dc_structure_2026_retry')+ENTRY_VARIANTS+FORECAST_VARIANTS+ACCEL_VARIANTS
SPECIALIST_VARIANTS = ('dc_liquid_2026_retry', 'dc_affinity_2026_retry')
STRUCTURE_VARIANTS = ('dc_structure_2026_retry',)+ENTRY_VARIANTS+FORECAST_VARIANTS+ACCEL_VARIANTS
AFFINITY_VARIANTS = ('dc_affinity_2026_retry',)+STRUCTURE_VARIANTS
START = pd.Timestamp('2025-01-01')
END = pd.Timestamp('2026-09-24')
PREVIOUS_MARK = pd.Timestamp('2024-12-31')
EXPECTED_MONTHS = 21
SIGNAL_MONTHS = pd.period_range('2024-12', '2026-08', freq='M')
BASE_PROTOCOL = dict(
    stage='Exploration after user-supplied structural-rally examples and previously examined history; not an untouched holdout',
    membership='Only Eastmoney catalogue and constituent snapshots dated exactly to the completed signal trading session; never backfill current membership',
    theme_history='At least 61 valid index observations, all at or before the actual signal session',
    theme_comparability='Names matching ^(?:昨日|近日)|^最近.*板$|^东方财富热股$ are rolling recent-performance/popularity baskets: retain as sentiment observations but exclude from monthly theme percentile baselines and entries; apply this rule before account results, with no industry/style/dividend/size exclusion',
    theme_gate='ret20>0 OR ret60>0; all three ranking inputs finite',
    theme_score='.40 ret60 percentile + .40 ret20 percentile + .20 amount-ratio percentile across all dated, history-complete themes; highest 5 qualifying themes',
    stock_universe='All SH/SZ stock month-end quotes; all boards participate in stock ranking; no inherited industry candidate universe',
    stock_history='Close times adjustment factor; calendar-month reindex before 1/3/6-month momentum; missing months are never skipped',
    stock_liquidity='Mean positive traded daily amount in the signal calendar month, at or before each stock actual month-end quote date; at least 10 observations',
    stock_gate='mom1>0 AND mom3>0 AND mom6>0; exclude mom3>150% AND mom1>30%',
    stock_score='.45 all-board mom1 percentile + .35 all-board mom3 percentile + .20 all-board mean-amount percentile',
    ranking='.70 stock score + .30 theme score; overlapping memberships use highest theme score, then theme code as stable tie-break',
    phase='advance for qualified entries',
    exposure=1., initial_cash=25000, capital_cap=25000, fees=0,
    external_topups=False, daily_supplementary_entries=False,
    execution='Original mainboard whole-lot optimizer, next-opening hard/trailing exits, deferred lower-limit sales, suspension valuation and corporate-action ledger',
    evaluation='Every account month in the declared period, calendar-year profit and daily drawdown; no best-month splicing',
)


def period_settings(name='dc_theme_retry'):
    if name == 'dc_theme_retry':
        return START, END, PREVIOUS_MARK, EXPECTED_MONTHS
    if name in ('dc_theme_2026_retry',)+SPECIALIST_VARIANTS+STRUCTURE_VARIANTS:
        return pd.Timestamp('2026-01-01'), END, pd.Timestamp('2025-12-31'), 9
    raise ValueError(f'Unknown dated-theme variant: {name}')


def signal_months(name='dc_theme_retry'):
    start, end, _, _ = period_settings(name)
    return pd.period_range(start.to_period('M')-1, end.to_period('M')-1, freq='M')


def protocol_for(name='dc_theme_retry'):
    start, end, previous_mark, months = period_settings(name)
    protocol = {**BASE_PROTOCOL, 'variant': name, 'variants': [name],
        'period': f'{start:%Y-%m-%d} through {end:%Y-%m-%d}',
        'previous_mark': f'{previous_mark:%Y-%m-%d}', 'expected_months': months,
        'signal_months': [str(month) for month in signal_months(name)],
        'scope': ('The 2025-2026 experiment remains pending complete dated historical archives'
                  if name == 'dc_theme_retry' else
                  'Period-only experiment because 2025 dated catalogues contain discontinued themes whose historical prices are unavailable; no claim of complete 2025 coverage and no parameter change based on returns')}
    if name in SPECIALIST_VARIANTS:
        protocol.update(
            stage='Exploratory feedback after the first completed dated-theme baseline; no untouched holdout claim; rules fixed before these accounts run',
            scope='Nine-month 2026 policy experiment after baseline feedback; no claim of complete 2025 coverage or unseen validation; freeze the common monthly rule before these accounts run',
            selection_scope='One fixed rule for all nine months; no named stocks, industry whitelist, manually selected months or market-breadth switch',
            liquidity_gate='Mean positive daily traded amount percentile >=90% across all SH/SZ boards with at least 10 observations, plus the unchanged positive-stock-momentum and joint-extreme-trend gates',
            stock_score='.45 amount percentile + .35 mom1 percentile + .20 mom3 percentile within the all-board liquid top-decile universe',
            ranking='.70 liquid score + .30 theme score')
    if name in AFFINITY_VARIANTS:
        protocol.update(
            membership='Dated membership snapshots remain validated collection evidence; stock eligibility uses price affinity, without a top-five membership requirement; the associated theme is not a claim of true membership',
            affinity_gate='Price-affinity correlation >=.30 with at least 40 paired observations; helper sees only index and stock observations at or before each completed signal session',
            affinity_source='Feature-manifest-bound data/raw/dc_theme_research/daily_market_2026.pkl; association is recomputed from dated theme audit and stock daily prices',
            theme_score='The same dated-theme percentile score of the price-associated theme, supplied by the fixed affinity helper; no top-five membership gate')
    if name in STRUCTURE_VARIANTS:
        protocol.update(
            stage='Exploratory feedback after the completed dated-theme baseline and liquid/affinity experiments; no untouched holdout claim; the routing rule is fixed before this account runs',
            scope='One nine-month 2026 structural-routing account after viewing baseline/A/B feedback; no claim of complete 2025 coverage or unseen validation',
            selection_scope='The same dated-theme phase rule in every month; no named stocks, industry whitelist, manually selected months or market-breadth switch',
            theme_phase='Price-associated dated theme ret20>=.15 AND ret60>=.15 is advance; otherwise ret20<=0 is pullback, and the remainder is range; underlying qualified-theme positive ret20 OR ret60 gate remains unchanged',
            liquidity_gate='Advance themes permit all stocks passing the original stock_qualifies gates, including at least 10 positive amount observations; pullback/range themes require the unchanged all-board liquid top-decile gate',
            stock_score='Advance: original .45 all-board mom1 percentile + .35 all-board mom3 percentile + .20 all-board amount percentile; pullback/range: .45 amount percentile + .35 mom1 percentile + .20 mom3 percentile within liquid top decile',
            ranking='Advance .70 original stock score + .30 affinity-theme score; pullback/range .70 liquid score + .30 affinity-theme score',
            phase='Every qualified entry remains advance for the original unchanged stop rules; theme_phase is a selection-routing audit field')
    if name in ENTRY_VARIANTS:
        protocol.update(
            stage='Exploratory entry-timing ablation after the completed DC structural account; all three rules fixed before their account results',
            scope='The same nine-month 2026 account; no named stock, theme or month override',
            daily_features='Complete 25 market sessions of positive finite OHLC/volume/amount; compounded close/open pressure5 and pressure20, mean 5-session close location, mean amount5 / preceding amount20, mean amount20; these are within-session pressure, not adjusted total returns or close breakouts',
            entry_score='.35 all-board pressure5 percentile + .35 pressure20 percentile + .15 amount-ratio percentile + .15 mean-amount20 percentile, ranked before actual mainboard execution',
            local_turn='pressure5>.03 AND pressure20>.05 AND location5>=.60 AND amount_ratio>=1.10',
            reversal_channel=name in REVERSAL_VARIANTS,
            reversal_gate='Associated positive dated-theme corr>=.30 and nobs>=40; >=10 positive monthly liquidity observations and unchanged joint-extreme heat exclusion; local_turn may bypass negative monthly momentum and the liquid-top-decile gate',
            reversal_rank='For admitted reversals max(existing structural score when originally eligible, .70 entry_score+.30 associated-theme score); otherwise retain original score',
            weekly_supplementary_entries=name in WEEKLY_VARIANTS,
            daily_supplementary_entries=name in WEEKLY_VARIANTS,
            weekly_schedule='First actual session of a new W-SUN calendar week, previous actual session close signal; skip regular month-start rebalance; at least two sessions after any most recent sale; idle capacity>=50% of current capped risk budget; never force weekly liquidation',
            weekly_theme_asof='Carry only previous completed-month DC price association and theme state; do not claim weekly refreshed constituents or theme forecasts',
            weekly_candidates=('Original structural monthly eligible stocks plus local_turn at the weekly signal, original rank' if name == 'dc_weekly_2026_retry' else 'Original structural eligible OR reversal-channel eligible, each also requires local_turn at weekly signal; rank rule as monthly reversal'),
            unchanged='25000 initial/new-investment cap, no topups/fees, all-board learning/ranks then mainboard 100-share buys, original monthly reset and next-opening hard/trailing exits')
    if name in FORECAST_VARIANTS:
        protocol.update(
            stage='Exploratory forecast objective after the first weekly-entry accounts; architecture and parameters fixed before this forecast account runs',
            scope='Nine-month 2026 account; training data since 2023, not an untouched holdout; no named stock/theme/month overrides',
            forecast_model='Quarterly-refit HistGradientBoostingRegressor, squared_error, 100 iterations, 15 leaf nodes, learning rate .05, L2=10, seed42; reuse model inside signal calendar quarter',
            forecast_target='Next-calendar-month adjusted-close return clipped +/-50% for training; label quote date strictly before model fit signal; ranking forecast, not a claim of executable account return',
            forecast_features='Adjusted stock mom1/3/6, rolling3/6-month volatility, six-month positive fraction, mom1-mom3/3; same-date all-board momentum/amount percentiles; complete-session close/open pressure5/20, close location and amount expansion; no stock code or industry feature',
            stock_gate='Finite forecast, forecast_return>0 and same-date all-board forecast percentile>=80%; dated price-affinity gate unchanged, >=10 monthly liquidity observations and unchanged joint-extreme heat exclusion; no requirement that mom1/3/6 are positive',
            ranking='.70 all-board forecast percentile + .30 associated dated-theme score; monthly account timing and all exits unchanged',
            phase='advance for every admitted entry; no new weekly supplementary entries')
        if name in LINEAR_FORECAST_VARIANTS:
            protocol.update(
                stage='Follow-up after the first forecast account and candidate-priority audit; fixed before this expected-profit account runs',
                ranking='Forecast return magnitude only; dated-theme affinity remains an eligibility gate, not a 30% percentile bonus; no named stock priority',
                optimizer='Maximize sum(forecast_return * invested_amount) with linear score power=1; unchanged whole-lot/budget/mainboard constraints, no prediction accuracy claim')
        if name == 'dc_peer_forecast_2026_retry':
            protocol.update(
                stage='Exploratory context-model follow-up after candidate-priority and first stock-only forecast feedback; fixed before this account runs',
                forecast_context='Add peer adjusted mom1/3/6 medians and positive3 breadth from causal 60-session gap-neutral intraday-pressure cohorts, fixed MiniBatchKMeans20/seed42/n_init3/max_iter60/batch1024, >=10 stocks; clusters are not industry membership',
                forecast_features='Previous 15 price/amount features plus four contemporaneous price-cohort context features; same quarterly estimator and strict label purge')
        if name == 'dc_rally_2026_retry':
            protocol.update(
                stage='Final objective ablation after mean-return forecast accounts and ranking diagnostics; fixed before the rally-classifier account runs',
                forecast_model='Quarterly frozen HistGradientBoostingClassifier, log_loss, 100 iterations, 15 leaves, learning rate .05, L2=10, seed42; training-only balanced class sample weights',
                forecast_target='Next-calendar-month adjusted-close return >=30%; strictly past labels; >=12 completed label months and >=10 examples of each class; this label does not incorporate lot residual cash or stops',
                forecast_features='The same 19 causal stock/amount/price-cohort inputs as the peer regression; no stock code or named theme feature',
                stock_gate='Same-date all-board rally-score percentile>=80%; unchanged dated price-affinity, monthly liquidity10 and joint heat gates; no positive monthly momentum requirement',
                ranking='Class-balanced rally-score magnitude only; forecast_return column is a schema alias for an UNCALIBRATED ranking score, not expected return or account target probability',
                optimizer='Maximize capital-weighted rally ranking score under unchanged whole-lot/25000/mainboard constraints; not expected-profit or portfolio-probability optimization')
    if name in ACCEL_VARIANTS:
        from research_dc_specialist import protocol_for_mode
        specialist_mode = {'dc_acceleration_2026_retry':'theme_acceleration', 'dc_early_2026_retry':'theme_early',
                           'dc_member_acceleration_2026_retry':'member_acceleration', 'dc_member_moderate_2026_retry':'member_not_extended',
                           'dc_member_relative_2026_retry':'member_relative_acceleration', 'dc_member_relative_moderate_2026_retry':'member_relative_moderate',
                           'dc_member_leader_2026_retry':'member_liquid_leader', 'dc_member_relative_leader_2026_retry':'member_relative_liquid_leader'}[name]
        specialist_protocol = protocol_for_mode(specialist_mode)
        protocol.update(
            stage='Exploratory early-leadership ablation after eight completed DC accounts; fixed before these new accounts, all history previously examined',
            scope='The same complete nine-month account; no stock/theme/month whitelist; no untouched holdout claim',
            specialist=specialist_protocol,
            specialist_mode=specialist_mode,
            stock_gate='Same dated affinity/liquidity/heat validation; specialist momentum/phase gates supersede positive-six-month gate',
            ranking='Hierarchical complete dated-theme momentum acceleration then stock acceleration/relative strength; fixed early-stage cap is a separate ablation',
            optimizer='Unchanged capital-weighted score**4 whole-lot optimizer; score is not expected return')
        if name in MEMBER_ACCEL_VARIANTS:
            protocol.update(stage='Exploratory dated-member acceleration after the four new global/theme-priority accounts; fixed before these new accounts; no untouched holdout claim',
                membership='Exact completed-signal-date top-five DC constituents; price-affinity audit retained separately, not the final gate',
                stock_gate='Exact dated member, >=10 positive amount observations, unchanged joint heat exclusion, mom1>0 and mom3>0; finite mom6 but no positive6 gate; moderate-extension ablation additionally mom1<=50%',
                ranking='Stock acceleration and strength relative to its true dated-member theme, all-board ranks; no associated-theme top-third gate')
        if name in RELATIVE_MEMBER_VARIANTS:
            protocol.update(stage='Exploratory exact-member internal-leadership after eight completed new ranking/acceleration accounts; fixed before the two relative-member accounts',
                ranking='Stock geometric acceleration minus exact-member group median and log-momentum minus group median, ranked across all feature-complete matched boards; group>=5 before momentum/heat/extension gates; 50/35/15 plus amount',
                membership='Exact dated top-five DC highest-theme de-duplicated member mapping; not a complete multi-theme graph and not price-affinity membership')
        if name in LEADER_FIRST_VARIANTS:
            protocol.update(stage='Fixed top-five-percent amount and rank-first allocation after ten new accounts; exploratory previously examined period, no holdout',
                stock_gate='Same moderate member gates, plus completed-signal all-board amount_percentile>=.95; no changed scoring/peer population or threshold search',
                optimizer='At most one new stock per monthly allocation: highest causal score among permitted executable affordable mainboard lots, then maximum affordable 100-share lots; residual cash allowed; existing retained positions and available budget respected')
    if name in CONTEXT_VARIANTS:
        from research_dc_context_ranker import protocol_for_scope
        local = name in LOCAL_CONTEXT_VARIANTS
        context_protocol = protocol_for_scope('cohort' if local else 'monthly')
        member = name in ('dc_context_member_2026_retry','dc_local_member_2026_retry')
        protocol.update(
            stage='Exploratory direct-ranking objective after eight DC accounts; all history previously examined; parameters fixed before new accounts',
            forecast_model=context_protocol,
            forecast_target='Same-month query next-month adjusted-return relevance deciles; LambdaRank emphasizes ordering the top names, not pointwise expected returns',
            forecast_features='19 causal stock/price-cohort features plus 4 stock-minus-cohort momentum context features; no named stocks, current-membership backfill or SW classification',
            stock_gate=('Exact dated top-five DC membership, at least10 monthly positive amount observations, unchanged joint heat gate, all-board model percentile>=80%' if member else 'Dated price affinity and at least10 positive amount observations, unchanged joint heat gate, all-board model percentile>=80%'),
            membership=('Exact completed-signal-date top-five DC constituents; price-affinity audit retained separately, not the final membership gate' if member else protocol['membership']),
            ranking=('.70 all-board ranker percentile + .30 exact dated-member theme score' if member else '.70 all-board ranker percentile + .30 dated associated-theme score'),
            optimizer='Capital-weighted combined ranking score**4; raw LambdaRank forecast_return may be negative and is not a return estimate')
        if local:
            protocol.update(stage='Exploratory within-cohort LambdaRank objective after the completed global-query ranking accounts; fixed before these local-query accounts',
                forecast_target='Within each completed signal-date/price-cohort-statistic query, next-month adjusted-return relevance deciles; only strictly past labels; no current DC membership backfill',
                query_scope='cohort')
    return protocol


PROTOCOL = protocol_for()


def _required(frame, columns, label):
    if not set(columns).issubset(frame.columns):
        raise ValueError(f'{label} missing required columns: {sorted(set(columns)-set(frame.columns))}')


def _dates(values, label, compact=False):
    raw = values.astype('string').str.replace(r'\.0$', '', regex=True) if compact else values
    result = pd.to_datetime(raw, format='%Y%m%d' if compact else None, errors='coerce')
    if result.isna().any() or not result.eq(result.dt.normalize()).all():
        raise ValueError(f'Invalid {label}')
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _collection_manifest(out, filename, scope):
    path = out/filename
    if not path.is_file():
        raise ValueError(f'Missing complete collection manifest: {filename}')
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Invalid collection manifest: {filename}') from exc
    if not isinstance(manifest, dict) or manifest.get('status') != 'complete' or manifest.get('scope') != scope:
        raise ValueError(f'Collection manifest must be complete and match scope {scope}: {filename}')
    inputs = manifest.get('source_inputs', {})
    if not isinstance(inputs, dict):
        raise ValueError(f'Invalid source-input fingerprints: {filename}')
    for source, expected in inputs.items():
        path = Path(source)
        if not path.is_absolute():
            path = out/path
        if not path.is_file() or not isinstance(expected, str) or _sha256(path) != expected:
            raise ValueError(f'Source-input fingerprint changed: {source}')
    return manifest


def _bound_frame(out, entry, filename):
    if not isinstance(entry, dict) or entry.get('file') != filename:
        raise ValueError(f'Manifest must bind the exact aggregate file: {filename}')
    path = out/filename
    if not path.is_file() or not isinstance(entry.get('sha256'), str) or _sha256(path) != entry['sha256']:
        raise ValueError(f'Aggregate fingerprint changed or file missing: {filename}')
    rows = entry.get('rows')
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise ValueError(f'Invalid aggregate row count: {filename}')
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame) or len(frame) != rows:
        raise ValueError(f'Aggregate row count mismatch: {filename}')
    return frame


def collection_inputs(out, name='dc_theme_retry'):
    """Require complete, mutually bound source snapshots before candidate generation."""
    period_settings(name)
    out = Path(out)
    scope = 'full' if name == 'dc_theme_retry' else '2026'
    feature_manifest = _collection_manifest(out, 'theme_feature_manifest.json', scope)
    member_manifest = _collection_manifest(out, 'theme_membership_manifest.json', scope)
    features = _bound_frame(out, feature_manifest.get('aggregate'), 'theme_features.pkl')
    members = _bound_frame(out, member_manifest.get('aggregate'), 'theme_members.pkl')
    selected = _bound_frame(out, member_manifest.get('selected'), 'theme_selected.pkl')
    if member_manifest.get('feature_sha256') != feature_manifest['aggregate']['sha256']:
        raise ValueError('Membership collection is not bound to the current theme features')
    _required(selected, ['signal_date', 'ts_code'], 'Selected theme evidence')
    selected = selected[['signal_date', 'ts_code']].copy()
    selected['signal_date'] = _dates(selected.signal_date, 'selected theme dates')
    if selected.duplicated(['signal_date', 'ts_code']).any() or selected.ts_code.isna().any():
        raise ValueError('Duplicate or missing selected-theme evidence keys')
    expected = rank_themes(features)
    expected = expected.loc[expected.theme_selected, ['signal_date', 'ts_code']]
    expected_keys = set(expected.itertuples(index=False, name=None))
    selected_keys = set(selected.itertuples(index=False, name=None))
    if selected_keys != expected_keys:
        raise ValueError('Selected theme evidence differs from the fixed-rule feature selection')
    return features, members


def rank_themes(features):
    """Rank the complete dated catalogue before selecting positive-trend themes."""
    _required(features, ['signal_date', 'ts_code', 'name', 'ret20', 'ret60',
                         'amount_ratio', 'source_last_date', 'index_obs', 'catalogue_date'], 'Theme features')
    f = features.copy().reset_index(drop=True)
    f['signal_date'] = _dates(f.signal_date, 'theme signal dates')
    f['source_last_date'] = _dates(f.source_last_date, 'theme source dates')
    if f[['signal_date', 'ts_code', 'name']].isna().any().any() or f.duplicated(['signal_date', 'ts_code']).any():
        raise ValueError('Missing or duplicate dated theme keys')
    if f.source_last_date.gt(f.signal_date).any():
        raise ValueError('Theme features contain future observations')
    f['catalogue_date'] = _dates(f.catalogue_date, 'catalogue dates')
    if not f.catalogue_date.eq(f.signal_date).all():
        raise ValueError('Catalogue snapshot must match the signal session')
    f['month'] = f.signal_date.dt.to_period('M').dt.to_timestamp('M')
    if f.groupby('month').signal_date.nunique().gt(1).any():
        raise ValueError('More than one signal session in a month')
    for col in ('ret20', 'ret60', 'amount_ratio', 'index_obs'):
        f[col] = pd.to_numeric(f[col], errors='coerce')
    if f.index_obs.isna().any() or f.index_obs.lt(0).any() or f.index_obs.mod(1).ne(0).any():
        raise ValueError('Invalid theme observation counts')
    complete = (f.index_obs.ge(61) & np.isfinite(f[['ret20', 'ret60', 'amount_ratio']]).all(axis=1)
                & f.amount_ratio.gt(0))
    f['history_complete'] = complete
    f['theme_comparable'] = ~f.name.astype('string').str.match(r'^(?:昨日|近日)|^最近.*板$|^东方财富热股$')
    f['excluded_reason'] = np.where(~f.theme_comparable, 'rolling_recent_performance_index',
                                   np.where(~complete, 'insufficient_index_history_or_features', ''))
    rankable = complete & f.theme_comparable
    ranks = f.loc[rankable].groupby('month')[['ret20', 'ret60', 'amount_ratio']].rank(pct=True)
    f['theme_score'] = .4*ranks.ret60 + .4*ranks.ret20 + .2*ranks.amount_ratio
    f['theme_qualifies'] = rankable & (f.ret20.gt(0) | f.ret60.gt(0))
    ranked = f[f.theme_qualifies].sort_values(['month', 'theme_score', 'ts_code'], ascending=[True, False, True])
    f['theme_rank'] = np.nan
    f.loc[ranked.index, 'theme_rank'] = ranked.groupby('month').cumcount()+1
    f['theme_selected'] = f.theme_rank.le(5)
    return f


def dated_members(members, theme_audit):
    _required(members, ['trade_date', 'ts_code', 'con_code'], 'Dated theme members')
    m = members[['trade_date', 'ts_code', 'con_code']].copy()
    m['signal_date'] = _dates(m.trade_date, 'membership dates', compact=True)
    if m[['ts_code', 'con_code']].isna().any().any() or m.duplicated(['signal_date', 'ts_code', 'con_code']).any():
        raise ValueError('Missing or duplicate dated membership keys')
    catalogue = theme_audit[['signal_date', 'ts_code']].drop_duplicates()
    known = m.merge(catalogue.assign(catalogue_member=True), on=['signal_date', 'ts_code'], how='left', validate='many_to_one')
    if not known.catalogue_member.eq(True).all():
        raise ValueError('Membership is not from the exact dated signal catalogue')
    selected = theme_audit[theme_audit.theme_selected][['signal_date', 'ts_code', 'theme_score', 'name']]
    found = selected.merge(m[['signal_date', 'ts_code']].drop_duplicates(), on=['signal_date', 'ts_code'], how='left', indicator=True)
    if found._merge.ne('both').any():
        raise ValueError('Selected theme lacks exact-date constituent evidence')
    return m.merge(selected, on=['signal_date', 'ts_code'], how='inner', validate='many_to_one')


def stock_momentum(monthly):
    _required(monthly, ['date', 'ts_code', 'close', 'adj_factor'], 'Stock monthly quotes')
    m = monthly[['date', 'ts_code', 'close', 'adj_factor']].copy()
    m['date'] = _dates(m.date, 'stock quote dates')
    m = m[m.ts_code.astype(str).str.match(r'^(?:0\d{5}\.SZ|3\d{5}\.SZ|6\d{5}\.SH)$')].copy()
    if m.empty:
        raise ValueError('No SH/SZ monthly stock observations')
    m['period'] = m.date.dt.to_period('M')
    if m.duplicated(['period', 'ts_code']).any():
        raise ValueError('Duplicate stock calendar-month observations')
    for col in ('close', 'adj_factor'):
        m[col] = pd.to_numeric(m[col], errors='coerce')
    if not np.isfinite(m[['close', 'adj_factor']]).all(axis=None) or m[['close', 'adj_factor']].le(0).any(axis=None):
        raise ValueError('Invalid stock adjusted-price inputs')
    price = m.assign(adjusted_close=m.close*m.adj_factor).pivot(index='period', columns='ts_code', values='adjusted_close')
    price = price.reindex(pd.period_range(price.index.min(), price.index.max(), freq='M'))
    price.index.name = 'period'
    for horizon in (1, 3, 6):
        values = (price/price.shift(horizon)-1).stack().rename(f'mom{horizon}').reset_index()
        m = m.merge(values, on=['period', 'ts_code'], how='left', validate='one_to_one')
    m['month'] = m.period.dt.to_timestamp('M')
    m['stock_code'] = m.ts_code.str[:6]
    return m.drop(columns='period')


def stock_liquidity(stocks, daily):
    _required(daily, ['date', 'ts_code', 'volume', 'amount'], 'Stock daily quotes')
    d = daily[['date', 'ts_code', 'volume', 'amount']].copy()
    d['date'] = _dates(d.date, 'daily stock dates')
    if d.duplicated(['date', 'ts_code']).any():
        raise ValueError('Duplicate daily stock observations')
    d['volume'] = pd.to_numeric(d.volume, errors='coerce')
    d['amount'] = pd.to_numeric(d.amount, errors='coerce')
    d = d[np.isfinite(d[['volume', 'amount']]).all(axis=1) & d.volume.gt(0) & d.amount.gt(0)].copy()
    d['month'] = d.date.dt.to_period('M').dt.to_timestamp('M')
    keys = stocks[['month', 'ts_code', 'date']].rename(columns={'date': 'stock_signal_date'})
    d = d.merge(keys, on=['month', 'ts_code'], how='inner', validate='many_to_one')
    d = d[d.date.le(d.stock_signal_date)]
    return d.groupby(['month', 'ts_code']).agg(signal_amount=('amount', 'mean'),
        liquidity_days=('amount', 'size'), liquidity_last_day=('date', 'max')).reset_index()


def score_candidates(monthly, daily, features, members):
    """Return all monthly stock rows plus a separate qualified entry flag."""
    themes = rank_themes(features)
    membership = dated_members(members, themes)
    stocks = stock_momentum(monthly)
    signals = themes[['month', 'signal_date']].drop_duplicates()
    c = stocks.merge(signals, on='month', how='inner', validate='many_to_one')
    if c.date.gt(c.signal_date).any():
        raise ValueError('Stock quote contains a future observation')
    if not c.date.dt.to_period('M').eq(c.month.dt.to_period('M')).all():
        raise ValueError('Stock quote outside signal calendar month')
    c = c.merge(stock_liquidity(c, daily), on=['month', 'ts_code'], how='left', validate='one_to_one')
    amount = c.signal_amount.where(c.liquidity_days.ge(10))
    ranks = c.groupby('month')[['mom1', 'mom3']].rank(pct=True)
    c['amount_percentile'] = amount.groupby(c.month).rank(pct=True)
    c['stock_score'] = .45*ranks.mom1 + .35*ranks.mom3 + .2*c.amount_percentile
    best = membership.sort_values(['signal_date', 'con_code', 'theme_score', 'ts_code'],
                                 ascending=[True, True, False, True]).drop_duplicates(['signal_date', 'con_code'])
    best = best.rename(columns={'con_code': 'ts_code', 'ts_code': 'ind_code', 'name': 'theme_name'})
    c = c.merge(best[['signal_date', 'ts_code', 'ind_code', 'theme_score', 'theme_name']],
                on=['signal_date', 'ts_code'], how='left', validate='one_to_one')
    c['heat_excluded'] = c.mom3.gt(1.5) & c.mom1.gt(.3)
    c['stock_qualifies'] = (c.mom1.gt(0) & c.mom3.gt(0) & c.mom6.gt(0)
                           & c.liquidity_days.ge(10) & ~c.heat_excluded)
    c['leadership_score'] = .7*c.stock_score + .3*c.theme_score
    c['eligible'] = c.stock_qualifies & c.ind_code.notna() & np.isfinite(c.leadership_score)
    c['phase'] = 'advance'
    if c.duplicated(['month', 'ts_code']).any():
        raise AssertionError('Theme overlap duplicated a stock candidate')
    plans = themes[themes.theme_selected][['month', 'signal_date', 'ts_code', 'theme_score']].copy()
    plans = plans.rename(columns={'month': 'date', 'theme_score': 'selection_score'})
    # A month with no qualified theme still needs an explicit cash-only plan.
    absent = signals[~signals.month.isin(plans.date)]
    if len(absent):
        plans = pd.concat([plans, absent.rename(columns={'month': 'date'}).assign(
            ts_code='__CASH__', selection_score=0.)], ignore_index=True)
    plans['risk_exposure'] = 1.
    return c.sort_values(['month', 'ts_code']).reset_index(drop=True), plans, themes


def liquid_candidates(scored):
    """A monthly liquidity specialist, retaining exact dated-theme membership."""
    _required(scored, ['month', 'signal_date', 'ts_code', 'signal_amount', 'liquidity_days',
                       'mom1', 'mom3', 'stock_qualifies', 'theme_score', 'ind_code',
                       'leadership_score', 'eligible'], 'Base stock audit')
    if scored.duplicated(['month', 'ts_code']).any():
        raise ValueError('Duplicate liquid candidate keys')
    c = scored.copy().reset_index(drop=True)
    c['base_leadership_score'] = c.leadership_score
    c['base_eligible'] = c.eligible
    amount = c.signal_amount.where(c.liquidity_days.ge(10))
    c['liquidity_percentile'] = amount.groupby(c.month).rank(pct=True)
    liquid = c.liquidity_percentile.ge(.9)
    ranks = c.loc[liquid].groupby('month')[['signal_amount', 'mom1', 'mom3']].rank(pct=True)
    c['liquid_score'] = .45*ranks.signal_amount + .35*ranks.mom1 + .20*ranks.mom3
    c['liquid_qualifies'] = liquid & c.stock_qualifies.eq(True)
    c['leadership_score'] = .70*c.liquid_score + .30*c.theme_score
    c['eligible'] = c.liquid_qualifies & c.ind_code.notna() & np.isfinite(c.leadership_score)
    c['classification_basis'] = 'dated_theme_membership'
    return c


def affinity_candidates(scored, affinity, theme_audit):
    """Price association is a separate eligibility rule, never a membership claim."""
    c = liquid_candidates(scored)
    fields = ['signal_date', 'ts_code', 'affinity_theme_code', 'affinity_theme_name',
              'affinity_theme_score', 'affinity_corr', 'nobs']
    _required(affinity, fields, 'Price-affinity observations')
    f = affinity[fields].copy().reset_index(drop=True)
    f['signal_date'] = _dates(f.signal_date, 'price-affinity signal dates')
    if f.duplicated(['signal_date', 'ts_code']).any() or f.ts_code.isna().any():
        raise ValueError('Duplicate or missing price-affinity stock keys')
    if not f.signal_date.isin(c.signal_date.unique()).all():
        raise ValueError('Price-affinity output is outside the dated signal sessions')
    # Full-market price associations may include stocks without a verified quote
    # in this signal month. They cannot enter the monthly stock candidate table.
    f = f.merge(c[['signal_date', 'ts_code']], on=['signal_date', 'ts_code'], how='inner', validate='one_to_one')
    for field in ('affinity_theme_score', 'affinity_corr', 'nobs'):
        f[field] = pd.to_numeric(f[field], errors='coerce')
    assigned = f.affinity_theme_code.notna()
    if ((assigned & (~np.isfinite(f.affinity_corr) | f.affinity_corr.abs().gt(1+1e-8)))
            | (assigned & (~np.isfinite(f.nobs) | f.nobs.lt(0) | f.nobs.mod(1).ne(0)))).any():
        raise ValueError('Invalid price-affinity correlation or observation count')
    lookup = theme_audit[theme_audit.theme_qualifies][['signal_date', 'ts_code', 'theme_score', 'name']]
    lookup = lookup.rename(columns={'ts_code': 'affinity_theme_code', 'theme_score': 'dated_theme_score',
                                    'name': 'dated_theme_name'})
    linked = f.merge(lookup, on=['signal_date', 'affinity_theme_code'], how='left', validate='many_to_one')
    if (assigned & (linked.dated_theme_score.isna()
            | ~np.isclose(linked.affinity_theme_score, linked.dated_theme_score, atol=1e-12, rtol=0)
            | linked.affinity_theme_name.ne(linked.dated_theme_name))).any():
        raise ValueError('Price affinity must refer to a qualifying dated catalogue theme with its exact score')
    c = c.rename(columns={'ind_code': 'member_theme_code', 'theme_name': 'member_theme_name',
                          'theme_score': 'member_theme_score'})
    c = c.merge(f, on=['signal_date', 'ts_code'], how='left', validate='one_to_one')
    c['affinity_qualifies'] = (c.affinity_theme_code.notna() & c.affinity_corr.ge(.30)
                              & c.nobs.ge(40) & np.isfinite(c.affinity_theme_score))
    c['ind_code'] = c.affinity_theme_code
    c['theme_name'] = c.affinity_theme_name
    c['theme_score'] = c.affinity_theme_score
    c['leadership_score'] = .70*c.liquid_score + .30*c.affinity_theme_score
    c['eligible'] = c.liquid_qualifies & c.affinity_qualifies & np.isfinite(c.leadership_score)
    c['classification_basis'] = 'price_affinity_not_membership'
    plans = theme_audit[theme_audit.theme_qualifies][['month', 'signal_date', 'ts_code', 'theme_score']].copy()
    plans = plans.rename(columns={'month': 'date', 'theme_score': 'selection_score'})
    signals = theme_audit[['month', 'signal_date']].drop_duplicates()
    absent = signals[~signals.month.isin(plans.date)]
    if len(absent):
        plans = pd.concat([plans, absent.rename(columns={'month': 'date'}).assign(
            ts_code='__CASH__', selection_score=0.)], ignore_index=True)
    plans['risk_exposure'] = 1.
    return c, plans


def structure_candidates(scored, affinity, theme_audit):
    """Allow small trend leaders in advancing themes with the same dated rule."""
    _required(scored, ['stock_score'], 'Base stock audit')
    c, plans = affinity_candidates(scored, affinity, theme_audit)
    state = theme_audit[['signal_date', 'ts_code', 'ret20', 'ret60']].rename(
        columns={'ts_code': 'affinity_theme_code', 'ret20': 'affinity_ret20', 'ret60': 'affinity_ret60'})
    if state.duplicated(['signal_date', 'affinity_theme_code']).any():
        raise ValueError('Duplicate dated structural theme keys')
    c = c.merge(state, on=['signal_date', 'affinity_theme_code'], how='left', validate='many_to_one')
    advance = c.affinity_ret20.ge(.15) & c.affinity_ret60.ge(.15)
    pullback = ~advance & c.affinity_ret20.le(0)
    c['theme_phase'] = np.select([advance, pullback, c.affinity_theme_code.notna()],
                                 ['advance', 'pullback', 'range'], default='unmatched')
    c['ranking_basis'] = np.where(advance, 'all_board_price_volume', 'liquid_top_decile')
    technical_score = c.stock_score.where(advance, c.liquid_score)
    c['leadership_score'] = .70*technical_score + .30*c.affinity_theme_score
    c['eligible'] = (c.affinity_qualifies & c.stock_qualifies.eq(True)
                     & (advance | c.liquid_qualifies) & np.isfinite(c.leadership_score))
    # Routing changes selection only; the execution phase keeps the original
    # hard/trailing-stop thresholds for every qualified entry.
    c['phase'] = 'advance'
    return c, plans


def affinity_market_path(out):
    """The price aggregate must already be bound by the feature collector."""
    manifest = _collection_manifest(Path(out), 'theme_feature_manifest.json', '2026')
    required = (ROOT.parent/'dc_theme_research'/'daily_market_2026.pkl').resolve()
    inputs = {}
    for filename, fingerprint in manifest.get('source_inputs', {}).items():
        path = Path(filename)
        if not path.is_absolute():
            path = Path(out)/path
        inputs[path.resolve()] = fingerprint
    if required not in inputs or not required.is_file() or _sha256(required) != inputs[required]:
        raise ValueError('Affinity market aggregate is not bound to the feature manifest')
    return required


def prepare(out, name='dc_theme_retry'):
    protocol = protocol_for(name)
    required_months = signal_months(name)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out/'dc_theme_protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Dated theme protocol changed or another period variant already owns this directory')
    features, members = collection_inputs(out, name)
    months = pd.to_datetime(features.signal_date).dt.to_period('M')
    if set(months) != set(required_months):
        raise ValueError(f'Theme features must cover every {required_months[0]} through {required_months[-1]} signal month')
    monthly = pd.read_pickle(ROOT/'stock_month_end_verified.pkl')
    actual = pd.to_datetime(monthly.date).groupby(pd.to_datetime(monthly.date).dt.to_period('M')).max()
    expected = months.map(actual)
    if expected.isna().any() or not pd.to_datetime(features.signal_date).eq(expected).all():
        raise ValueError('Theme signal date is not the verified stock month-end session')
    daily = pd.read_pickle(out/'daily.pkl')
    c, plans, themes = score_candidates(monthly, daily, features, members)
    if name == 'dc_liquid_2026_retry':
        c = liquid_candidates(c)
    elif name in AFFINITY_VARIANTS:
        from research_theme_affinity import affinity_features
        price_path = affinity_market_path(out)
        signals = pd.DatetimeIndex(sorted(themes.signal_date.unique()))
        affinity = affinity_features(themes, pd.read_pickle(price_path), daily, signals)
        candidate_rule = structure_candidates if name in STRUCTURE_VARIANTS else affinity_candidates
        c, plans = candidate_rule(c, affinity, themes)
        affinity.to_pickle(out/'affinity_audit.pkl')
    if name in ENTRY_VARIANTS:
        from research_dc_entry_policy import prepare_entries
        c = prepare_entries(out, name, c, daily)
    if name in ACCEL_VARIANTS:
        from research_dc_specialist import specialist_candidates
        mode = {'dc_acceleration_2026_retry':'theme_acceleration', 'dc_early_2026_retry':'theme_early',
                'dc_member_acceleration_2026_retry':'member_acceleration', 'dc_member_moderate_2026_retry':'member_not_extended',
                'dc_member_relative_2026_retry':'member_relative_acceleration', 'dc_member_relative_moderate_2026_retry':'member_relative_moderate',
                'dc_member_leader_2026_retry':'member_liquid_leader', 'dc_member_relative_leader_2026_retry':'member_relative_liquid_leader'}[name]
        c = specialist_candidates(c, mode=mode, theme_audit=themes)
    if name in FORECAST_VARIANTS:
        if name in CONTEXT_VARIANTS:
            from functools import partial
            from research_dc_context_ranker import prepare_context_scores
            prepare_forecasts=partial(prepare_context_scores,query_scope='cohort' if name in LOCAL_CONTEXT_VARIANTS else 'monthly')
        elif name == 'dc_rally_2026_retry':
            from research_dc_rally_classifier import prepare_rally_scores as prepare_forecasts
        elif name == 'dc_peer_forecast_2026_retry':
            from research_dc_peer_forecast import prepare_peer_forecasts as prepare_forecasts
        else:
            from research_dc_forecast import prepare_forecasts
        forecasts = prepare_forecasts(out, monthly, daily, pd.DatetimeIndex(sorted(c.signal_date.unique())))
        c = c.merge(forecasts, on=['signal_date','ts_code'], how='left', validate='one_to_one')
        c['structural_eligible'] = c.eligible.eq(True)
        c['eligible'] = (c.affinity_qualifies.eq(True) & c.liquidity_days.ge(10) & ~c.heat_excluded.eq(True)
                         & (True if name in CONTEXT_VARIANTS else c.forecast_return.gt(0))
                         & c.forecast_percentile.ge(.80) & np.isfinite(c.forecast_return))
        c['leadership_score'] = .70*c.forecast_percentile+.30*c.affinity_theme_score
        if name in LINEAR_FORECAST_VARIANTS:
            c['leadership_score'] = 100*c.forecast_return
        if name in ('dc_context_member_2026_retry','dc_local_member_2026_retry'):
            c['eligible'] = (c.member_theme_code.notna() & c.liquidity_days.ge(10)
                             & ~c.heat_excluded.eq(True) & c.forecast_percentile.ge(.80)
                             & np.isfinite(c.forecast_return))
            c['ind_code'] = c.member_theme_code
            c['theme_name'] = c.member_theme_name
            c['theme_score'] = c.member_theme_score
            c['leadership_score'] = .70*c.forecast_percentile+.30*c.member_theme_score
            c['classification_basis'] = 'exact_dated_top_five_membership'
            c['ranking_basis'] = 'context_lambdarank_and_dated_member_theme'
    path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    c.to_pickle(out/'candidates.pkl')
    plans.to_pickle(out/'plans.pkl')
    themes.to_pickle(out/'theme_audit.pkl')
    c.to_pickle(out/f'candidate_audit_{name}.pkl')
    membership_column = 'member_theme_code' if 'member_theme_code' in c else 'ind_code'
    coverage = c.groupby('month').agg(stocks=('ts_code', 'size'), theme_members=(membership_column, 'count'),
        positive_stock_trend=('stock_qualifies', 'sum'), eligible=('eligible', 'sum'))
    if 'affinity_theme_code' in c:
        coverage['price_associations'] = c.groupby('month').affinity_theme_code.count()
    coverage.to_csv(out/'candidate_coverage.csv')
    return c[c.eligible].copy(), plans


def summarize(out):
    """Verify and summarize this account without the legacy model's input list."""
    import verify_small_account as verifier
    import research_leadership as leadership
    from verify_market_states import verify_daily
    out = Path(out)
    protocol = json.loads((out/'dc_theme_protocol.json').read_text(encoding='utf-8'))
    name = protocol.get('variant')
    if protocol != protocol_for(name):
        raise ValueError('Summary protocol does not match a fixed dated-theme variant')
    _, end, _, expected_months = period_settings(name)
    status_path = out/f'run_status_{name}.json'
    if status_path.exists() and not json.loads(status_path.read_text(encoding='utf-8')).get('account_complete'):
        raise ValueError('An incomplete theme rerun must not reuse an older account')
    previous_out = verifier.OUT
    verifier.OUT = out
    try:
        reconciliation = verifier.verify(name, expected_months=expected_months, end_date=end)
    finally:
        verifier.OUT = previous_out
    daily_check = verify_daily(out, ROOT, [name])
    a = pd.read_csv(out/f'account_{name}.csv', parse_dates=['date'])
    daily = pd.read_csv(out/f'daily_nav_{name}.csv', parse_dates=['date'])
    nav = np.r_[25000., daily.equity.to_numpy()]
    drawdown = float((nav/np.maximum.accumulate(nav)-1).min())
    a['strategy'] = name
    a['target_hit'] = a.profit.ge(7500.-1e-7)
    a.to_csv(out/'monthly_results.csv', index=False)
    annual = [dict(strategy=name, year=year, months=len(g), profit=g.profit.sum(),
        account_return_pct=100*((1+g['return']).prod()-1), ending_equity=g.equity.iloc[-1],
        target_hit_months=int(g.target_hit.sum())) for year, g in a.groupby(a.date.dt.year)]
    pd.DataFrame(annual).to_csv(out/'annual_results.csv', index=False)
    metric = dict(strategy=name, ending_equity=float(a.equity.iloc[-1]),
        total_profit=float(a.profit.sum()), total_return_pct=100*(a.equity.iloc[-1]/25000-1),
        daily_drawdown_pct=100*drawdown, target_hit_months=int(a.target_hit.sum()), losing_months=int(a.profit.lt(0).sum()))
    pd.DataFrame([metric]).to_csv(out/'metrics.csv', index=False)
    sources = [Path(__file__), Path(__file__).with_name('research_market_states.py'),
        Path(__file__).with_name('small_account_backtest.py'), Path(__file__).with_name('research_leadership.py'),
        Path(__file__).with_name('s7_budget_portfolio.py'), Path(__file__).with_name('verify_small_account.py'),
        Path(__file__).with_name('verify_market_states.py'), ROOT/'stock_month_end_verified.pkl',
        ROOT/'boundaries/sessions.json', out/'dc_theme_protocol.json', out/'theme_features.pkl',
        out/'theme_members.pkl', out/'theme_selected.pkl', out/'theme_feature_manifest.json',
        out/'theme_membership_manifest.json', out/'theme_audit.pkl', out/'candidates.pkl', out/'plans.pkl', out/'daily.pkl']
    sources += list((ROOT/'boundaries').glob('*.pkl'))+list((ROOT/'adj_month_end').glob('*.pkl'))
    for folder in (out/'exit_quotes', out/'suspension_events', out/'dividend_repairs', out/'replacement_limits', out/'scoped_actions', ROOT/'dividends',
                   ROOT/'suspension_checks', leadership.OUT/'dividends'):
        sources += [p for p in folder.glob('*') if p.is_file()]
    for extra in (out/'verified_suspensions.json', out/f'run_status_{name}.json',
                  out/'theme_collection_manifest.json'):
        if extra.exists():
            sources.append(extra)
    sources += list(out.glob('dc_theme*manifest.json'))
    if name in AFFINITY_VARIANTS:
        sources += [Path(__file__).with_name('research_theme_affinity.py'), affinity_market_path(out),
                    out/'affinity_audit.pkl']
    if name in ENTRY_VARIANTS:
        sources += [Path(__file__).with_name('research_dc_entry.py'),
                    Path(__file__).with_name('research_dc_entry_policy.py'), Path(__file__).with_name('research_scoped_actions.py'), out/'entry_features.pkl',
                    out/'entry_feature_manifest.json', out/'weekly_candidates.pkl']
    if name in FORECAST_VARIANTS:
        sources += [Path(__file__).with_name('research_dc_forecast.py'), Path(__file__).with_name('research_dc_entry.py'),
                    Path(__file__).with_name('research_scoped_actions.py'), out/'forecast_feature_manifest.json',
                    out/'forecast_features.pkl', out/'forecast_predictions.pkl', out/'forecast_models.pkl', out/'forecast_model_audit.json']
    if name == 'dc_peer_forecast_2026_retry':
        sources += [Path(__file__).with_name('research_dc_peer_context.py'),
                    Path(__file__).with_name('research_dc_peer_forecast.py'),out/'peer_context.pkl']
    if name == 'dc_rally_2026_retry':
        sources += [Path(__file__).with_name('research_dc_rally_classifier.py'),
                    Path(__file__).with_name('research_dc_peer_context.py'),
                    Path(__file__).with_name('research_dc_peer_forecast.py'),out/'peer_context.pkl',out/'rally_feature_manifest.json']
    if name in ACCEL_VARIANTS:
        sources += [Path(__file__).with_name('research_dc_specialist.py'), Path(__file__).with_name('research_scoped_actions.py')]
    if name in LEADER_FIRST_VARIANTS:
        sources += [Path(__file__).with_name('research_dc_leader_allocation.py')]
    if name in CONTEXT_VARIANTS:
        sources += [Path(__file__).with_name('research_dc_context_ranker.py'),
                    Path(__file__).with_name('research_dc_peer_context.py'),
                    Path(__file__).with_name('research_dc_peer_forecast.py'), out/'peer_context.pkl',
                    out/'context_ranker_features.pkl', out/'context_ranker_feature_manifest.json']
    verification = dict(reconciliations=[reconciliation], daily_reconciliations=daily_check,
        inputs={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')})
    (out/'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    return metric
