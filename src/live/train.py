# -*- coding: utf-8 -*-
"""实盘训练入口

- quarterly: 全量重训（跑 run_all.py）→ 归档到 models/quarterly/YYYY-Qn/
- monthly:   月末微调（Phase 3 v1 暂等同 quarterly 全量）→ 归档到 models/monthly/YYYY-MM/

两种模式都会把最新快照同步到 models/current/，供 live/predict.py 使用。

用法：
    python -m live.train --mode quarterly
    python -m live.train --mode monthly
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import (MODELS_CURRENT_DIR, MODELS_MONTHLY_DIR,
                    MODELS_QUARTERLY_DIR, OUTPUT_DIR)
from live import state


SNAPSHOT_FILES = (
    'predictions_gnn.pkl',
    'predictions_lstm_b.pkl',
    'predictions_ensemble.pkl',
    'predictions_ensemble_equal.pkl',
    'regime_labels.pkl',
    'gnn_inference_state.pkl',
    'lstm_b_inference_state.pkl',
)


def _run_full_pipeline() -> None:
    """调用 run_all.py 跑 s0→s1→s2→s3 全量"""
    run_all_path = os.path.join(_SRC_DIR, 'run_all.py')
    if not os.path.exists(run_all_path):
        raise FileNotFoundError(f'找不到 {run_all_path}')

    print(f'→ python {run_all_path}')
    result = subprocess.run(
        [sys.executable, run_all_path],
        cwd=_SRC_DIR,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f'run_all.py 退出码 {result.returncode}')


def _snapshot_to(dst_dir: str) -> list:
    """把 results/ 下的预测 pkl 快照到 dst_dir"""
    os.makedirs(dst_dir, exist_ok=True)
    copied = []
    for name in SNAPSHOT_FILES:
        src = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, name))
            copied.append(name)
    return copied


def _update_current_pointer(snapshot_dir: str, label: str, mode: str) -> None:
    """更新 models/current/：清空旧内容 + 拷贝新快照 + 写 MANIFEST"""
    os.makedirs(MODELS_CURRENT_DIR, exist_ok=True)

    for fname in os.listdir(MODELS_CURRENT_DIR):
        fpath = os.path.join(MODELS_CURRENT_DIR, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)

    for fname in os.listdir(snapshot_dir):
        shutil.copy2(
            os.path.join(snapshot_dir, fname),
            os.path.join(MODELS_CURRENT_DIR, fname),
        )

    manifest = {
        'label': label,
        'mode': mode,
        'source_dir': snapshot_dir,
        'updated_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(os.path.join(MODELS_CURRENT_DIR, 'MANIFEST.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _quarter_label(now: datetime) -> str:
    q = (now.month - 1) // 3 + 1
    return f'{now.year}-Q{q}'


def _month_label(now: datetime) -> str:
    return f'{now.year}-{now.month:02d}'


def train(mode: str) -> None:
    assert mode in ('quarterly', 'monthly'), f'unknown mode: {mode}'
    now = datetime.now()

    print('=' * 60)
    print(f'  live/train  mode={mode}  ts={now.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    _run_full_pipeline()

    if mode == 'quarterly':
        label = _quarter_label(now)
        dst = os.path.join(MODELS_QUARTERLY_DIR, label)
    else:
        label = _month_label(now)
        dst = os.path.join(MODELS_MONTHLY_DIR, label)

    snapshotted = _snapshot_to(dst)
    print(f'\n快照 → {dst}')
    for name in snapshotted:
        print(f'  · {name}')

    _update_current_pointer(dst, label, mode)
    print(f'\n已更新 models/current/ 指向 {label}')

    # 状态更新
    state.update_last_run(**{
        f'last_{mode}_train': now.isoformat(timespec='seconds'),
        f'{mode}_label': label,
    })

    ver = state.get_model_version()
    ver.setdefault('current', {})[mode] = label
    ver['current']['path'] = MODELS_CURRENT_DIR
    ver.setdefault('history', []).append({
        'mode': mode, 'label': label,
        'at': now.isoformat(timespec='seconds'),
    })
    ver['history'] = ver['history'][-50:]
    state.set_model_version(ver)

    print(f'\n[{mode}] 训练完成 · label={label}')


def main():
    parser = argparse.ArgumentParser(description='实盘训练入口')
    parser.add_argument(
        '--mode',
        choices=['quarterly', 'monthly'],
        required=True,
        help='quarterly=季末全量重训，monthly=月末微调',
    )
    args = parser.parse_args()
    train(args.mode)


if __name__ == '__main__':
    main()
