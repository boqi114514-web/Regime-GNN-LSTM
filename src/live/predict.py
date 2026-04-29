# -*- coding: utf-8 -*-
"""实盘推理：从 models/current/predictions_ensemble.pkl 提取最新一个月的 Top-K 信号

Phase 3 v1 版本：
  - 不直接重跑 s0→s1→s2→s3（训练开销大）
  - 依赖 live/train.py 已经把最新一次的 predictions_ensemble.pkl 同步到 models/current/
  - 本脚本只做"取最后一行"+ 缓存 + 状态更新

后续（Phase 5）补增量推理：用 data_pipeline/update.py 拉最新月份数据，
用 current 模型权重做 forward pass，追加一行到 ensemble。
"""
import json
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import LOCAL_DATA_CACHE, MODELS_CURRENT_DIR, TOP_K, ENSEMBLE_MODE
from live import state


_PKL_MAP = {
    'regime': 'predictions_ensemble.pkl',
    'equal':  'predictions_ensemble_equal.pkl',
}


def _load_current_ensemble(mode: str = ENSEMBLE_MODE) -> pd.DataFrame:
    fname = _PKL_MAP.get(mode, _PKL_MAP['regime'])
    path = os.path.join(MODELS_CURRENT_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"没有找到 {path}\n"
            f"请先运行:  python -m live.train --mode quarterly"
        )
    df = pd.read_pickle(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def _current_model_label() -> Optional[str]:
    manifest = os.path.join(MODELS_CURRENT_DIR, 'MANIFEST.json')
    if not os.path.exists(manifest):
        return None
    try:
        with open(manifest, 'r', encoding='utf-8') as f:
            return json.load(f).get('label')
    except (json.JSONDecodeError, OSError):
        return None


def infer_latest(top_k: int = TOP_K, mode: str = ENSEMBLE_MODE) -> dict:
    """提取 current 模型最新一个月的 Top-K 预测，缓存并更新 state。

    Args:
        mode: 'regime'（Regime条件集成）或 'equal'（等权集成），默认读 config.ENSEMBLE_MODE

    score 语义：s3 的 pred_ensemble = -(w_gnn·rank_gnn + w_lstm·rank_lstm)，取值
    [-N, -1]。我们转成 `score = N + pred_ensemble`，正数且越大越好（N = 当月
    进入 ensemble 的行业数），方便周报展示。
    """
    ens = _load_current_ensemble(mode=mode)
    latest_date = ens['date'].max()
    latest = ens[ens['date'] == latest_date].copy()
    latest = latest.sort_values('pred_ensemble', ascending=False).reset_index(drop=True)

    n_industries = len(latest)
    top = latest.head(top_k)

    def _score(pe: float) -> float:
        return float(n_industries + pe)

    # 全量排名（供 monitor 做排名变动对比）
    all_ranked = [
        {
            'ts_code': row['ts_code'],
            'rank': i + 1,
            'score': _score(row['pred_ensemble']),
            'pred_ensemble': float(row['pred_ensemble']),
        }
        for i, row in latest.iterrows()
    ]

    # Top-K 明细打分
    scores = {}
    for _, row in top.iterrows():
        entry = {
            'score': _score(row['pred_ensemble']),
            'ensemble': float(row['pred_ensemble']),  # 原始，保留
        }
        if 'rank_gnn' in row:
            entry['rank_gnn'] = float(row['rank_gnn'])
        if 'rank_lstm' in row:
            entry['rank_lstm'] = float(row['rank_lstm'])
        if 'actual_ret' in row and pd.notna(row['actual_ret']):
            entry['actual_ret'] = float(row['actual_ret'])
        scores[row['ts_code']] = entry

    first_row = top.iloc[0] if len(top) else None
    result = {
        'as_of': latest_date.strftime('%Y-%m-%d'),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'model_label': _current_model_label(),
        'top_k': top['ts_code'].tolist(),
        'scores': scores,
        'all_ranked': all_ranked,
        'regime': int(first_row['regime']) if first_row is not None and 'regime' in first_row else None,
        'w_gnn': float(first_row['w_gnn']) if first_row is not None and 'w_gnn' in first_row else None,
        'w_lstm': float(first_row['w_lstm']) if first_row is not None and 'w_lstm' in first_row else None,
    }

    # 缓存
    os.makedirs(LOCAL_DATA_CACHE, exist_ok=True)
    cache_path = os.path.join(LOCAL_DATA_CACHE, f'prediction_{result["as_of"]}.json')
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 更新 state（只记录时间戳和数据月份，不改 last_holdings）
    state.update_last_run(
        last_predict=result['generated_at'],
        last_predict_as_of=result['as_of'],
    )
    return result


def main():
    print('=' * 60)
    print('  live/predict: 推理最新月份信号')
    print('=' * 60)
    try:
        result = infer_latest()
    except FileNotFoundError as e:
        print(f'[错误] {e}')
        sys.exit(1)

    print(f'数据月份   : {result["as_of"]}')
    print(f'模型版本   : {result["model_label"]}')
    if result['regime'] is not None:
        print(f'HMM regime : {result["regime"]}')
    if result['w_gnn'] is not None:
        print(f'集成权重   : GNN={result["w_gnn"]:.2f} / LSTM={result["w_lstm"]:.2f}')
    print(f'Top-{len(result["top_k"])}:')
    for i, code in enumerate(result['top_k'], 1):
        s = result['scores'][code]
        print(f'  {i}. {code}  score={s["score"]:.2f}  (ensemble={s["ensemble"]:+.3f})')


if __name__ == '__main__':
    main()
