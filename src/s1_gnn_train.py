# -*- coding: utf-8 -*-
"""
Step 1：GNN 分支 —— GAT 建模行业基本面联动

功能：
  1. GLASSO 构建行业动态图（月度滚动）
  2. GAT 在景气度指标上做截面预测
  3. Walk-forward 滚动训练
  4. 输出预测文件

替代原 ARIMAX 的角色：捕捉跨行业基本面传导关系
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import GraphicalLassoCV
import os
import time

from config import *

# ============================================================
#  GLASSO 动态图构建
# ============================================================

def build_glasso_graph(returns_matrix, top_k=GRAPH_TOP_K):
    """
    用 GLASSO 从收益率矩阵构建稀疏邻接矩阵

    参数：
        returns_matrix: (T, N) 行业收益率矩阵
        top_k: 每个节点保留的最强邻居数

    返回：
        adj: (N, N) 归一化邻接矩阵
    """
    N = returns_matrix.shape[1]

    # 处理 NaN
    returns_clean = np.nan_to_num(returns_matrix, nan=0.0)

    # 如果样本不足或方差太小，返回单位矩阵
    if returns_clean.shape[0] < 5 or np.std(returns_clean) < 1e-8:
        return np.eye(N)

    try:
        model = GraphicalLassoCV(cv=3, max_iter=200)
        model.fit(returns_clean)
        precision = np.abs(model.precision_)
    except Exception:
        # 回退到相关系数矩阵
        try:
            corr = np.corrcoef(returns_clean.T)
            precision = np.abs(corr)
            precision = np.nan_to_num(precision, nan=0.0)
        except Exception:
            return np.eye(N)

    # 去掉对角线
    np.fill_diagonal(precision, 0)

    # Top-K 稀疏化
    adj = np.zeros_like(precision)
    for i in range(N):
        row = precision[i]
        if row.max() > 0:
            topk_idx = np.argsort(row)[-top_k:]
            adj[i, topk_idx] = row[topk_idx]

    # 对称化
    adj = np.maximum(adj, adj.T)

    # 归一化到 [0, 1]
    max_val = adj.max()
    if max_val > 0:
        adj = adj / max_val

    return adj


def build_monthly_graphs(mkt, industries, months, rolling=GLASSO_ROLLING_MONTHS):
    """为每个月构建 GLASSO 图"""
    # 构建收益率 pivot 表
    ret_pivot = mkt.pivot_table(index='date', columns='ts_code',
                                 values='ret', aggfunc='first')
    ret_pivot = ret_pivot.reindex(columns=industries)
    all_dates = sorted(ret_pivot.index)

    graphs = {}
    for m in months:
        # 找到 m 之前的 rolling 个月
        past_dates = [d for d in all_dates if d < m]
        if len(past_dates) < 6:
            graphs[m] = np.eye(len(industries))
            continue

        window_dates = past_dates[-rolling:]
        returns_window = ret_pivot.loc[window_dates].values
        graphs[m] = build_glasso_graph(returns_window)

    return graphs


# ============================================================
#  GAT 模型
# ============================================================

class IndustryGAT(nn.Module):
    """简单 GAT 用于行业截面预测"""

    def __init__(self, in_features, hidden_dim=GAT_HIDDEN_DIM, dropout=GAT_DROPOUT):
        super().__init__()
        self.W = nn.Linear(in_features, hidden_dim)
        self.a_src = nn.Parameter(torch.randn(hidden_dim, 1) * 0.01)
        self.a_dst = nn.Parameter(torch.randn(hidden_dim, 1) * 0.01)

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        x: (N, F) 节点特征
        adj: (N, N) 邻接矩阵
        """
        N = x.size(0)
        device = x.device

        # 特征变换
        h = self.W(x)  # (N, D)

        # 注意力分数
        e_src = h @ self.a_src  # (N, 1)
        e_dst = h @ self.a_dst  # (N, 1)
        e = e_src + e_dst.T     # (N, N)
        e = F.leaky_relu(e, 0.2)

        # 用邻接矩阵 + 自环做 mask
        mask = (adj > 0).float() + torch.eye(N, device=device)
        e = e.masked_fill(mask == 0, -1e9)

        # softmax 注意力权重
        attn = F.softmax(e, dim=1)
        if self.training:
            attn = self.dropout(attn)

        # 消息传递
        h_agg = attn @ h  # (N, D)
        h_agg = self.norm(h_agg)
        h_agg = F.elu(h_agg)
        if self.training:
            h_agg = self.dropout(h_agg)

        # 预测分数
        scores = self.fc(h_agg).squeeze(-1)  # (N,)
        return scores


# ============================================================
#  损失函数
# ============================================================

def pearson_ic_loss(pred, target):
    """Pearson IC 损失（最大化截面相关性）"""
    pred_c = pred - pred.mean()
    tgt_c = target - target.mean()
    corr = (pred_c * tgt_c).sum() / (pred_c.norm() * tgt_c.norm() + 1e-8)
    return -corr


def combined_loss(pred, target):
    """IC + 轻量 MSE 正则"""
    ic = pearson_ic_loss(pred, target)
    mse = F.mse_loss(pred, target)
    return ic + 0.05 * mse


# ============================================================
#  训练逻辑
# ============================================================

def train_gat_single(train_data, val_data, in_features, seed=42):
    """
    训练单个 GAT 模型

    参数：
        train_data: list of (features_tensor, adj_tensor, returns_tensor)
        val_data: 同上
        in_features: 输入特征维度
        seed: 随机种子

    返回：
        model: 训练好的模型
        best_val_ic: 最佳验证 IC
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cpu')
    model = IndustryGAT(in_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=GAT_LR,
                                  weight_decay=GAT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=GAT_EPOCHS, eta_min=1e-5)

    best_val_ic = -999
    best_state = None
    patience_counter = 0

    for epoch in range(GAT_EPOCHS):
        # ---- 训练 ----
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        # 梯度累积：每个月一个样本
        accum_steps = min(4, len(train_data))
        for i, (feat, adj, ret) in enumerate(train_data):
            feat = feat.to(device)
            adj = adj.to(device)
            ret = ret.to(device)

            pred = model(feat, adj)
            loss = combined_loss(pred, ret) / accum_steps
            loss.backward()
            total_loss += loss.item()

            if (i + 1) % accum_steps == 0 or i == len(train_data) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        # ---- 验证 ----
        model.eval()
        val_ics = []
        with torch.no_grad():
            for feat, adj, ret in val_data:
                feat = feat.to(device)
                adj = adj.to(device)
                ret = ret.to(device)

                pred = model(feat, adj)
                # 计算 Spearman rank IC
                pred_np = pred.numpy()
                ret_np = ret.numpy()
                from scipy.stats import spearmanr
                ic, _ = spearmanr(pred_np, ret_np)
                if np.isfinite(ic):
                    val_ics.append(ic)

        val_ic = np.mean(val_ics) if val_ics else -1

        if val_ic > best_val_ic:
            best_val_ic = val_ic
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= GAT_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_ic


def predict_gat(model, data_list):
    """用训练好的 GAT 预测"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for feat, adj in data_list:
            pred = model(feat, adj)
            predictions.append(pred.numpy())
    return predictions


# ============================================================
#  Walk-Forward 主循环
# ============================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("  GNN 分支：GAT + GLASSO 行业基本面联动")
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    mkt = load_industry_monthly()
    prosperity = load_prosperity_monthly()
    industries = get_industries(mkt)
    n_industries = len(industries)
    print(f"  行业数: {n_industries}")

    # 加载 regime 标签
    regime_path = os.path.join(OUTPUT_DIR, 'regime_labels.pkl')
    if os.path.exists(regime_path):
        regime_df = pd.read_pickle(regime_path)
        regime_df['date'] = pd.to_datetime(regime_df['date'])
        print(f"  Regime: {len(regime_df)} 月, "
              f"{regime_df['date'].min().strftime('%Y-%m')} ~ "
              f"{regime_df['date'].max().strftime('%Y-%m')}")
        has_regime = True
    else:
        print("  警告：未找到 regime_labels.pkl，不使用 regime 特征")
        has_regime = False

    # 合并景气度指标到月度数据
    mkt_merged = pd.merge(
        mkt[['ts_code', 'date', 'year', 'month', 'ret', 'pe', 'pb']],
        prosperity,
        on=['ts_code', 'year', 'month'],
        how='inner'
    )
    mkt_merged = mkt_merged.sort_values(['date', 'ts_code']).reset_index(drop=True)

    # 添加 PE/PB 分位数（60个月滚动）
    for col in ['pe', 'pb']:
        mkt_merged[f'{col}_pctile'] = mkt_merged.groupby('ts_code')[col].transform(
            lambda x: x.rolling(60, min_periods=12).rank(pct=True)
        )

    # 合并 regime（one-hot 编码，对所有行业广播同一个月的 regime）
    if has_regime:
        regime_onehot = pd.get_dummies(regime_df[['date', 'regime']],
                                        columns=['regime'], prefix='regime')
        # 确保4个 regime 列都存在
        for i in range(4):
            col = f'regime_{i}'
            if col not in regime_onehot.columns:
                regime_onehot[col] = 0
        regime_cols = [f'regime_{i}' for i in range(4)]
        mkt_merged = pd.merge(mkt_merged, regime_onehot[['date'] + regime_cols],
                               on='date', how='left')
        # 未覆盖的月份填0
        for col in regime_cols:
            mkt_merged[col] = mkt_merged[col].fillna(0).astype(float)
    else:
        regime_cols = []

    # 特征列
    feature_cols = SELECTED_INDICATORS + ['pe_pctile', 'pb_pctile'] + regime_cols
    in_features = len(feature_cols)
    print(f"  GNN 特征数: {in_features} (含 {len(regime_cols)} 个 regime 特征)")

    # 截面 z-score
    mkt_merged = zscore_cross_section(mkt_merged, feature_cols)

    # 所有可用月份
    available_months = sorted(mkt_merged['date'].unique())
    print(f"  可用月份: {len(available_months)} ({available_months[0].strftime('%Y-%m')} ~ "
          f"{available_months[-1].strftime('%Y-%m')})")

    # 构建 GLASSO 图
    print("\n构建 GLASSO 动态图...")
    all_graphs = build_monthly_graphs(mkt, industries, available_months)
    print(f"  图数量: {len(all_graphs)}")

    # Walk-Forward 滚动训练
    total_window = TRAIN_MONTHS + VAL_MONTHS
    all_preds = []

    n_windows = 0
    for start_idx in range(0, len(available_months) - total_window - PREDICT_MONTHS + 1, STEP_MONTHS):
        train_months = available_months[start_idx:start_idx + TRAIN_MONTHS]
        val_months = available_months[start_idx + TRAIN_MONTHS:start_idx + total_window]
        pred_months = available_months[start_idx + total_window:
                                       start_idx + total_window + PREDICT_MONTHS]

        if len(pred_months) == 0:
            continue

        n_windows += 1
        print(f"\n窗口 {n_windows}: 训练 {train_months[0].strftime('%Y-%m')}~"
              f"{train_months[-1].strftime('%Y-%m')}, "
              f"验证 {val_months[0].strftime('%Y-%m')}~{val_months[-1].strftime('%Y-%m')}, "
              f"预测 {pred_months[0].strftime('%Y-%m')}~{pred_months[-1].strftime('%Y-%m')}")

        # 准备数据
        def month_to_tensors(months_list, need_returns=True):
            """将月份列表转换为 (features, adj, returns) 元组列表"""
            data = []
            for m in months_list:
                m_data = mkt_merged[mkt_merged['date'] == m].copy()
                if len(m_data) < 10:
                    continue

                # 确保行业顺序一致
                m_data = m_data.set_index('ts_code').reindex(industries)
                feat = m_data[feature_cols].values.astype(np.float32)
                feat = np.nan_to_num(feat, nan=0.0)

                adj = all_graphs.get(m, np.eye(n_industries)).astype(np.float32)

                feat_t = torch.tensor(feat)
                adj_t = torch.tensor(adj)

                if need_returns:
                    ret = m_data['ret'].values.astype(np.float32)
                    ret = np.nan_to_num(ret, nan=0.0)
                    ret_t = torch.tensor(ret)
                    data.append((feat_t, adj_t, ret_t))
                else:
                    data.append((feat_t, adj_t))
            return data

        train_data = month_to_tensors(train_months, need_returns=True)
        val_data = month_to_tensors(val_months, need_returns=True)
        pred_data = month_to_tensors(pred_months, need_returns=False)

        if len(train_data) < 20 or len(val_data) < 3:
            print("  跳过（数据不足）")
            continue

        # 多 seed 训练
        all_seed_preds = []
        for s in range(GAT_N_SEEDS):
            seed = RANDOM_SEED + s * 100
            model, val_ic = train_gat_single(train_data, val_data, in_features, seed=seed)
            print(f"  Seed {s}: Val IC = {val_ic:.4f}")

            # 预测
            preds = predict_gat(model, pred_data)
            all_seed_preds.append(preds)

        # 对多 seed 预测做排名平均
        for m_idx, m in enumerate(pred_months):
            if m_idx >= len(pred_data):
                break

            # 收集所有 seed 的预测
            month_ranks = []
            for seed_preds in all_seed_preds:
                if m_idx < len(seed_preds):
                    p = seed_preds[m_idx]
                    # 转为排名
                    rank = np.argsort(np.argsort(-p)).astype(float) + 1  # 1=最高
                    month_ranks.append(rank)

            if not month_ranks:
                continue

            # 平均排名
            avg_rank = np.mean(month_ranks, axis=0)
            avg_score = -avg_rank  # 排名越小分数越高

            # 获取实际收益
            m_actual = mkt[mkt['date'] == m].set_index('ts_code').reindex(industries)

            for i, code in enumerate(industries):
                actual_ret = m_actual.loc[code, 'ret'] if code in m_actual.index else np.nan
                all_preds.append({
                    'ts_code': code,
                    'date': m,
                    'actual_ret': actual_ret if np.isfinite(actual_ret) else np.nan,
                    'pred_gnn': avg_score[i],
                })

    # 保存结果
    pred_df = pd.DataFrame(all_preds)
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    pred_df = pred_df.dropna(subset=['actual_ret'])

    out_path = os.path.join(OUTPUT_DIR, 'predictions_gnn.pkl')
    pred_df.to_pickle(out_path)
    pred_df.to_csv(out_path.replace('.pkl', '.csv'), index=False, encoding='utf-8-sig')

    # 评估整体 IC
    from scipy.stats import spearmanr
    monthly_ics = []
    for dt in pred_df['date'].unique():
        m_data = pred_df[pred_df['date'] == dt]
        if len(m_data) >= 10:
            ic, _ = spearmanr(m_data['pred_gnn'], m_data['actual_ret'])
            if np.isfinite(ic):
                monthly_ics.append(ic)

    if monthly_ics:
        mean_ic = np.mean(monthly_ics)
        std_ic = np.std(monthly_ics)
        icir = mean_ic / (std_ic + 1e-8)
        pos_ratio = np.mean([ic > 0 for ic in monthly_ics])
        print(f"\n{'='*60}")
        print(f"  GNN 分支评估")
        print(f"{'='*60}")
        print(f"  Rank IC: {mean_ic:.4f} ± {std_ic:.4f}")
        print(f"  ICIR:    {icir:.4f}")
        print(f"  IC>0:    {pos_ratio:.1%}")
        print(f"  月数:    {len(monthly_ics)}")

    # 保存最后一个窗口的推理状态（供 live/infer_incremental.py 月度增量推理）
    if n_windows > 0:
        import pickle as _pkl
        ret_pivot = mkt.pivot_table(
            index='date', columns='ts_code', values='ret', aggfunc='first')
        ret_pivot = ret_pivot.reindex(columns=industries)
        infer_state = {
            'model_states': [
                {k: v.clone().cpu() for k, v in m.state_dict().items()}
                for m in seed_models
            ],
            'feature_cols': feature_cols,
            'industries': industries,
            'in_features': in_features,
            'has_regime': has_regime,
            'regime_cols': regime_cols,
            'train_end_month': pd.Timestamp(train_months[-1]).strftime('%Y-%m-%d'),
            'pred_end_month': pd.Timestamp(pred_months[-1]).strftime('%Y-%m-%d'),
            'ret_pivot': ret_pivot.tail(GLASSO_ROLLING_MONTHS * 2).copy(),
        }
        _state_path = os.path.join(OUTPUT_DIR, 'gnn_inference_state.pkl')
        with open(_state_path, 'wb') as _f:
            _pkl.dump(infer_state, _f, protocol=4)
        print(f"  GNN 推理状态已保存 → {_state_path}")

    elapsed = time.time() - t0
    print(f"\nGNN 训练完成，耗时 {elapsed/60:.1f} 分钟")
    print(f"预测保存至: {out_path}")


if __name__ == '__main__':
    main()
