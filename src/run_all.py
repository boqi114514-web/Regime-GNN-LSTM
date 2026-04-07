# -*- coding: utf-8 -*-
"""
GNN + LSTM-B 行业轮动 —— 一键运行

架构：
  GNN分支 (s1): GAT + GLASSO，建模行业基本面联动
  LSTM-B分支 (s2): 技术因子动量预测
  集成+回测 (s3): 自适应权重融合 + Top-K 策略
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    t0 = time.time()
    print("=" * 60)
    print("  GNN + LSTM-B 行业轮动模型")
    print("=" * 60)

    print("\n[1/4] HMM 宏观状态识别...")
    from s0_regime import main as regime_main
    regime_main()

    print("\n\n[2/4] 训练 GNN 分支（含 Regime 特征）...")
    from s1_gnn_train import main as gnn_main
    gnn_main()

    print("\n\n[3/4] 训练 LSTM-B 分支...")
    from s2_lstm_b_train import main as lstm_main
    lstm_main()

    print("\n\n[4/4] 集成 + 回测...")
    from s3_ensemble_backtest import main as backtest_main
    backtest_main()

    elapsed = time.time() - t0
    print(f"\n\n全部完成！总耗时 {elapsed/60:.1f} 分钟")


if __name__ == '__main__':
    main()
