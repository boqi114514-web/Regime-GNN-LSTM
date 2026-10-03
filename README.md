# 股票量化研究项目总导航

原 Regime-GNN-LSTM 研究仓库已于 **2026-10-03** 拆分为三个独立项目。请按研究对象进入对应仓库；此页作为统一导航，原仓库历史代码、分支与结果继续保留，不再把三条路线混称为同一个模型。

| 项目 | GitHub 地址 | 实际架构 | 频率 |
|---|---|---|---|
| 日频机器学习 | [Stock-Opportunity-Daily](https://github.com/boqi114514-web/Stock-Opportunity-Daily) | 最新为 LightGBM 多期限路径收益/风险/事件预测＋历史题材图特征；另保留 GRU＋图注意力＋四专家实验 | 每日预测、季度训练 |
| 月频机器学习 | [Stock-Selection-Monthly-ML](https://github.com/boqi114514-web/Stock-Selection-Monthly-ML) | Ridge＋五种子 LightGBM 行业排名，接固定个股选择和整手账户；同时保存 GAT/LSTM、修复 HMM、月频股票学习实验 | 月度决策，主线季度训练 |
| 月频固定规则 | [Stock-Selection-Monthly-Rules](https://github.com/boqi114514-web/Stock-Selection-Monthly-Rules) | 东财历史题材＋相对加速度/动量固定排序＋成交额过滤＋合格续持；不训练预测模型 | 月度入场、每日风险退出 |

每个仓库均有独立 README、依赖清单、运行入口、测试、研究报告、结果与迁移清单。**“固定 GNN:LSTM 集成比例”仍属于机器学习项目，不等于固定规则选股。** 月频固定规则的收益不归功于 GNN/LSTM；日频最新树模型也不冒称正在训练 GNN/LSTM。

## 本地目录

三个项目位于 `D:\desktop\有意思的事情\量化\项目\` 下：

```text
项目/
  README.md                         本总导航
  Stock-Opportunity-Daily/          日频机器学习，独立 Git 仓库
  Stock-Selection-Monthly-ML/        月频机器学习，独立 Git 仓库
  Stock-Selection-Monthly-Rules/     月频固定规则，独立 Git 仓库
  Regime-GNN-LSTM/                  保留的原始档案及迁移工具
```

原目录未移动、未删除。新仓库的默认输出根由自身代码位置解析，不再硬编码回原目录。必要数据与对应结果缓存复制为本地独立副本；没有用软链接把三个项目的可写输出混在一起。

在任意一个新项目根目录：

```powershell
python -m pip install -r requirements.txt
python run.py info
python run.py check
python run.py test
python run.py verify-archive
```

`info`显示架构和允许入口，`check`检查导入及本地材料，`test`运行该项目测试，`verify-archive`核对已复制公开结果是否与原字节一致。具体训练、候选生成、资金账户和重放命令见各自 README；这些检查命令不会自动重新训练模型。

## 当前成果如何阅读

- 月频固定规则最新候选：2026-01至2026-09-24盈利35,454元，最大日回撤11.5004%；四、五、六月分别15,846、8,262、9,816元。它是规则策略成果。
- 日频最新完整路径收益版本：相同2026区间盈利7,497元，最大日回撤8.3845%；日频原图树模型全期盈利更高，为14,374.2元。14个完整账户和一个因退市处理缺口未完成的分支均保留。
- 月频机器学习主线 `new_full`：2023-01至2026-09-24盈利22,817.46元，最大**月末**回撤26.51%。行业由机器学习排序，个股层仍有固定规则。

以上时间长度和回撤频率不同，不能直接横比。账户均以25,000元起步、无补款，费用按对应实验约定为零；2026年九月只到24日。完整金额、特征、限制和失败结果在各项目README及报告中分别披露，不择月拼接。

## 拆分验证与数据发布边界

本次整理后测试：日频 **222项**、月频机器学习 **191项**、月频规则 **247项**，全部通过。三仓合计 **1,648份**已公开结果文件与原件逐字节一致。测试包含新的入口/根路径与模型范围隔离检查；规则入口在禁止机器学习库导入的情况下仍可加载。

GitHub发布源码、测试、README、研究报告、选定账户/预测输出和日频检查点。原始供应商行情、大型`.pkl`、完整中间表与密钥不上传；仅克隆代码不等于已经取得全部研究数据。每个项目的 `publication_manifest.json` 是明确发布清单，`migration_manifest.json` 记录来源，`repository_validation.json` 记录整理验证。

历史审计中的绝对路径、源码SHA及旧报告结论原样保留，仍描述原实验。跨目录复制不自动等于重新训练或完成账户重放；旧快照的严格重放仍可能需要原始档案路径。此次没有篡改旧审计以伪造迁移后的通过结果，也没有调整策略参数或收益。

## 原始档案

- [拆分前完整研究快照 33edeff](https://github.com/boqi114514-web/Regime-GNN-LSTM/tree/33edeff9dbba6f7c3913b48790fbc938b61c451c)
- [月频归档快照 bed28e2](https://github.com/boqi114514-web/Regime-GNN-LSTM/tree/bed28e26f7bb0af129bc317c26e511279d56dd36)
- [拆分前月频与日频架构报告](https://github.com/boqi114514-web/Regime-GNN-LSTM/blob/33edeff9dbba6f7c3913b48790fbc938b61c451c/reports/monthly_daily_architecture_2026-10-03.md)

后续开发应在三个新仓库中按架构分别进行；本仓库保留原始历史及拆分工具，首页只承担总导航。
