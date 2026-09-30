# 2026-09-30 研究升级发布说明

## 结论与适用范围

保留候选 `daily_guarded_retry`，固定比较基线 `state_onset_retry`。2023-01 至 2026-09-24 共 45 个月、905 个交易日：候选期末权益 73,537.51 元，比基线增加盈利 14,898 元。2026 年四五月仍亏损，月盈利达到 7,500 元的月份仍为 3 / 45；最大峰谷亏损金额没有改善。完整逐月数字见 [报告](../reports/market_state_research_2026-09-30.md)。这次发布不自动替换 `run_all.py` 或 Dashboard 默认模型。

初始本金 25,000 元，不补款；新增投入不超过 25,000 元及可用权益/状态风险预算，已有持仓上涨或无法卖出时可能超过该金额。盈利超出投入上限的部分留作现金。允许单股集中，买入只允许主板、100 股整手；创业板和科创板仍参与上游信号。按本轮约定不计佣金、印花税和滑点；不是扣费后结果。月盈利 / 25,000 与按月初权益计算的账户收益是两个不同口径，CSV 分列保存。

## 本轮代码

| 模块 | 作用与状态 |
|---|---|
| `src/research_leadership.py` | 历史行业成员、量价龙头及集中账户研究 |
| `src/research_market_states.py` | 在线行情状态、行业阶段、大小盘风格、整手账户与延迟卖出重试；研究统一入口 |
| `src/research_daily_reentry.py` | 前一日收盘信号、下一交易日开盘的补充入场，不是日内做 T |
| `src/research_quality_floor.py` | 财报公告可得性、累计盈利/净资产底线；TTM 分支有覆盖阻断 |
| `src/research_structural_router.py` | 结构性行情分流与补充入场消融 |
| `src/research_peer_graph.py` | 历史同行关系图与共同强度消融 |
| `src/research_trend_features.py` | 趋势特征探索；未完成分支不能计作有效账户 |
| `src/research_learned_leaders.py` | 学习式龙头排序探索，完整账户因缺失报价未完成 |
| `src/research_fine_industry.py` | 三级行业成员完整性检查，接口问题导致完整账户未通过 |
| `src/verify_market_states.py` | 独立交易、股份、现金、日净值及 T+1 核验 |

这些模块对应的十个新增测试文件一并发布；包含原项目测试在内共 106 项通过。不同分支仍属探索，不代表都获得收益改善。

状态模型不再使用宏观 HMM：用历史量价、行业广度和成交额的离散跳跃模型在线推断；行业另分推进、震荡、退潮，调节仓位、排序和退出。当前保留候选沿用该月度主干，仅升级日频补充模块。

## 保留候选规则

- 月度候选池、排序及阶段与基线逐行一致，不额外过滤原月度持仓。
- 卖出至少 5 个交易日后，可用现金达到半数风险预算、少于 5 个持仓且无待卖单，才允许补充买入；不为换股强制清空残余持仓。
- 日频排序使用 5/20 日开盘到收盘的复合强度、收盘位置、成交额扩张、历史二级行业共同强度。开盘到收盘强度不是包含隔夜跳空的多日总收益。
- 仅日频新买入要求：上个月信号日已知的最近累计归母净利润、归母净资产为正，财报不超过 450 天。可得日为公告日与实际公告日较晚者再加一天；缺失不回填。这不是 TTM 盈利指标。
- 收盘触发止损/止盈，随后可执行开盘成交；T+1、停牌与涨跌停限制、分红送转保留。硬止损 8%，浮盈达到 20% 激活回撤止盈，推进阶段回撤阈值 15%、震荡阶段 10%，震荡阶段另设 30% 止盈。跳空和无法成交不按理想止损价处理。

具体阈值、排名公式以当前目录内各 `*_protocol.json` 及源代码为准。

## 数据修复与结果分层

**当前有效目录**：[market_state_board_complete](../results/market_state_board_complete/README.md)。修补 82 个日线日期，其中 79 个为 2026 年 1 月 5 日至 5 月 7 日科创板整板缺失，补充 47,330 行；另有 3 个较早日期截断。修复仅用于隔离研究输入，没有覆盖原始 `stock_daily.pkl`，也不意味着默认下载器所有问题已修好。修复前后的基线 45 个月账户一致，但成交额特征已改变。

**历史目录**：[leadership_research](../results/leadership_research/README.md) 与 [market_state_research](../results/market_state_research/README.md)。后者含缺板修复前输入，以及未重试月末失败卖单的旧版本；历史报告和失败消融保留，不可拿其中较高收益替换当前数字。

三级行业接口忽略筛选/重复分页，TTM 所需的 2025 年年报覆盖不足等问题均保留阻断。没有有效完整账户的分支不进入已完成实验指标。当前目录有 11 个已完成账户，统一保留收益较差的版本，不拼接各版本最好月份。

## 阅读与复跑

仅阅读结果无需安装 Python：打开 `metrics.csv`、`annual_results.csv`、`monthly_results.csv`，或按策略名查看 `account_*`、`daily_nav_*`、`trades_*`。最新候选有 30 笔日频补充买入。

开发环境先安装 `requirements.txt`，并检查 `src/config.py` 的 `PROJECT_DIR` 等路径与本机实际位置一致。在仓库根目录执行：

```powershell
python -m unittest discover -s tests -q
python src/research_market_states.py --help
```

**只有保留了本地数据及缓存的环境才能完整离线重放**：需要原始行情/行业月线/财报、`results/price_risk_pipeline/predictions_ensemble.pkl`、`results/leadership_research/candidates.pkl`、本研究目录的 `daily.pkl`、`candidates.pkl`、`plans.pkl`、特征/财报快照，以及 `data/raw/execution_v1` 等执行报价、涨跌停、历史成员、公司行动及修复证据缓存。具体路径以 `src/config.py`、执行引擎与研究模块为准。GitHub 不提供上述大数据包，清洁克隆不能直接一条命令还原全部账户。

```powershell
# 已具备本轮准备好的完整本地缓存时：
python src/research_market_states.py run --variant daily_guarded_retry --offline
python src/research_market_states.py summary
```

上述命令会重写该实验的结果文件，先保留研究快照；`--output-dir` 可隔离输出，但不会自动复制或生成所需缓存。`prepare --offline` 也需要已有上游候选、预测及修复证据，并非从空仓库下载全套数据。在线执行需要自行配置 API 环境变量与相应权限，不在源码中填写密钥。

独立核账函数为 `verify_market_states.verify_daily`，需要本地报价和证据缓存。已完成核验记录在 `verification.json`：11 个账户核账通过，候选 905 日现金/权益误差小于 1e-8 元，基线及候选离线重跑的全部 45 个月账户字段一致。文件内指纹对应研究时原始字节；Git 的 CRLF 转换可能改变文本字节，不能把换行差异当作数值差异。

## 发布范围与历史字段

发布本轮代码、测试、四份研究报告、三个研究目录顶层可阅读 CSV/JSON、说明；不发布原始数据、模型、pickle、深层 API 缓存、密钥及无关个人文件。`daily_pressure_feature_manifest.json` 是本机缓存清单，不发布。部分审计表保留当时本地证据路径，需映射到自己的数据位置，不能视作远程下载链接。

`research_candidate_2026-09-30.json` 的 `github_pushed: false` 是研究完成时的历史状态，不是本次 Git 发布状态；该记录保留不改写。实际发布以 Git 提交及远端记录为准。旧报告同样保留当时结论，当前入口以本说明和 9 月 30 日报告为准。
