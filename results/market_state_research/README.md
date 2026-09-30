# 历史研究归档——不要作为当前有效结果

此目录保留早期市场状态、风格、龙头及止盈止损消融。输入尚未包含后续完整缺板修复；部分不带 `_retry` 的账户没有正确重试月末失败卖单，会造成意外继续持有。旧 `state_onset` 期末 64,119.51 元不能作为当前基线，当前修正版 `state_onset_retry` 为 58,639.51 元。

保留旧结果是为了追溯试验和修复，不以较高旧收益覆盖新结果。当前统一比较入口是 [market_state_board_complete](../market_state_board_complete/README.md)，完整发布背景见 [说明](../../docs/research_upgrade_2026-09-30.md)。历史核验 JSON 仅对应各自当时输入与规则，不证明后续发现的缺陷不存在。
