"""No-SW reversal/weekly entry adapters; previous-month theme evidence only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_dc_themes import REVERSAL_VARIANTS, WEEKLY_VARIANTS


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def admit_reversals(scored, features):
    """Admit a causal local turn without requiring positive monthly momentum."""
    c = scored.copy()
    if c.duplicated(['signal_date', 'ts_code']).any():
        raise ValueError('Duplicate candidate signal keys')
    if features.duplicated(['signal_day', 'ts_code']).any():
        raise ValueError('Duplicate entry feature keys')
    fields = ['signal_day', 'ts_code', 'pressure5', 'pressure20', 'location5',
              'amount_ratio5_vs_prior20', 'mean_amount20', 'entry_score', 'reversal_eligible']
    c = c.merge(features[fields], left_on=['signal_date', 'ts_code'],
                right_on=['signal_day', 'ts_code'], how='left', validate='one_to_one')
    c['structural_eligible'] = c.eligible.eq(True)
    c['structural_score'] = c.leadership_score
    c['reversal_qualifies'] = (c.affinity_qualifies.eq(True) & c.liquidity_days.ge(10)
        & ~c.heat_excluded.eq(True) & c.reversal_eligible.eq(True) & np.isfinite(c.entry_score))
    score = .70*c.entry_score+.30*c.affinity_theme_score
    c['eligible'] = c.structural_eligible | c.reversal_qualifies
    c['leadership_score'] = c.structural_score.where(c.structural_eligible)
    c.loc[c.reversal_qualifies, 'leadership_score'] = np.fmax(
        c.loc[c.reversal_qualifies, 'leadership_score'], score[c.reversal_qualifies])
    c['entry_channel'] = np.where(c.reversal_qualifies, 'local_turn', 'structural_trend')
    return c


def weekly_candidates(scored, features, schedule, name):
    """Refresh local pressure, never the previous completed-month theme label."""
    rows = []
    monthly = {d.to_period('M'): g for d, g in scored.groupby('month')}
    for execution_day, signal_day in sorted(schedule.items()):
        key = execution_day.to_period('M')-1
        if key not in monthly:
            continue
        c = monthly[key].copy()
        if not c.signal_date.le(signal_day).all():
            raise ValueError('Weekly entry sees future monthly theme evidence')
        c['theme_signal_date'] = c.signal_date
        c['signal_date'] = signal_day
        local = features[features.signal_day.eq(signal_day)]
        if name in REVERSAL_VARIANTS:
            c = admit_reversals(c, local)
        else:
            c = c.merge(local[['signal_day', 'ts_code', 'reversal_eligible']],
                left_on=['signal_date', 'ts_code'], right_on=['signal_day', 'ts_code'],
                how='left', validate='one_to_one')
        c['eligible'] = c.eligible.eq(True) & c.reversal_eligible.eq(True)
        c['execution_day'] = execution_day
        rows.append(c)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prepare_entries(out, name, scored, daily):
    from research_dc_entry import entry_features, weekly_execution_schedule
    out = Path(out)
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(daily.date).unique()))
    schedule = {d: s for d, s in weekly_execution_schedule(calendar).items()
                if pd.Timestamp('2026-01-01') <= d <= pd.Timestamp('2026-09-24')}
    dates = set(pd.to_datetime(scored.signal_date).unique())
    if name in WEEKLY_VARIANTS:
        dates |= set(schedule.values())
    dates = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in dates))
    manifest = dict(daily_sha256=_sha(out/'daily.pkl'), helper_sha256=_sha(Path(__file__).with_name('research_dc_entry.py')),
                    signals=[f'{d:%Y-%m-%d}' for d in dates])
    path = out/'entry_feature_manifest.json'
    cache = out/'entry_features.pkl'
    if path.exists() and cache.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
    else:
        existing = {}
    if {k: existing.get(k) for k in manifest} == manifest and existing.get('features_sha256') == _sha(cache):
        features = pd.read_pickle(cache)
    else:
        features = entry_features(daily, dates, session_calendar=calendar)
        features.to_pickle(cache)
        manifest['features_sha256'] = _sha(cache)
        path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    weekly = weekly_candidates(scored, features, schedule, name) if name in WEEKLY_VARIANTS else pd.DataFrame()
    weekly.to_pickle(out/'weekly_candidates.pkl')
    return admit_reversals(scored, features) if name in REVERSAL_VARIANTS else scored
