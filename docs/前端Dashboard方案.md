# Regime-GNN-LSTM 前端 Dashboard 方案

## 一、目标

为现有实盘系统搭建一个**本地原生窗口 Dashboard**，双击启动，替代手动读 `reports/latest.md` 的方式，同时提供一键触发数据更新、周报生成、季度重训等操作。

---

## 二、技术选型

**pywebview + FastAPI**（原生窗口套 Web 前端）

| 项目 | 说明 |
|---|---|
| 窗口层 | `pywebview`：把 HTML 页面套进原生 Windows 窗口，双击 `.bat` 启动，无浏览器 |
| 后端层 | `FastAPI`：本地 HTTP 服务，暴露数据接口 + 触发 Python 任务 |
| 前端层 | 原生 HTML / CSS / JS，无需任何前端框架 |
| 图表 | `Chart.js`（CDN 引入，轻量，够用） |
| 长任务 | FastAPI `StreamingResponse` + JS `EventSource` SSE，实时流式日志 |
| 新增依赖 | `pywebview`, `fastapi`, `uvicorn`（项目已有 pandas/numpy） |

**启动方式：** 双击 `启动Dashboard.bat` → 自动起 FastAPI 服务 → pywebview 弹出原生窗口

---

## 三、UI 设计规范

### 风格定位

**Google Material Design 3（亮色）+ iOS 26 Liquid Glass 卡片**

- 整体：MD3 亮色系，干净现代，参考 Google Analytics / Cloud Console 布局
- 卡片：Liquid Glass 效果，半透明浮动，带背景模糊，非实心白色
- 背景：浅蓝灰渐变（`#EEF2FF → #F8FAFF`），为玻璃效果提供透过感

### 配色（A 股规范：红涨绿跌）

```
页面背景     linear-gradient(135deg, #EEF2FF, #F8FAFF)
卡片背景     rgba(255,255,255,0.55)  + backdrop-filter: blur(16px)
卡片边框     1px solid rgba(255,255,255,0.45)
卡片投影     0 8px 32px rgba(99,102,241,0.08), inset 0 1px 0 rgba(255,255,255,0.7)
卡片圆角     20px

主色（Google 蓝）   #1A73E8   按钮、选中态、链接
文字主              #202124
文字次              #5F6368

上涨 / 正向         #D93025   红色（A 股）
下跌 / 负向         #1E8E3E   绿色（A 股）
警告                #F9AB00   黄色

Regime 状态色：
  扩张   #1A73E8  蓝
  过热   #F9AB00  橙
  衰退   #D93025  红
  复苏   #1E8E3E  绿
```

### 字体

```
中文 / 界面正文    Noto Sans SC（Google 系，中文兼容好）
数字 / 代码        JetBrains Mono（等宽，精确感）
```

### 卡片 CSS 核心

```css
.glass-card {
  background: linear-gradient(135deg, rgba(255,255,255,0.6), rgba(255,255,255,0.35));
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,0.45);
  box-shadow: 0 8px 32px rgba(99,102,241,0.08),
              inset 0 1px 0 rgba(255,255,255,0.7);
  border-radius: 20px;
}
```

---

## 四、目录结构

```
Regime-GNN-LSTM/
├── dashboard/
│   ├── main.py               # 入口：启动 FastAPI + pywebview
│   ├── api/
│   │   ├── report.py         # /api/report  周报数据接口
│   │   ├── holdings.py       # /api/holdings 个股持仓接口
│   │   ├── backtest.py       # /api/backtest 回测数据接口
│   │   ├── system.py         # /api/system  系统状态接口
│   │   └── runner.py         # /api/run/* 任务触发 + SSE 日志流
│   ├── static/
│   │   ├── index.html        # 单页应用入口
│   │   ├── css/
│   │   │   ├── base.css      # 全局样式、变量、玻璃卡片
│   │   │   └── pages.css     # 各页面样式
│   │   └── js/
│   │       ├── app.js        # 路由 + 页面切换
│   │       ├── pages/
│   │       │   ├── report.js     # 周报页逻辑
│   │       │   ├── holdings.js   # 持仓页逻辑
│   │       │   ├── backtest.js   # 回测页逻辑
│   │       │   └── system.js     # 系统页逻辑
│   │       └── utils/
│   │           ├── charts.js     # Chart.js 封装
│   │           └── sse.js        # SSE 日志流封装
│   └── utils/
│       └── data_loader.py    # 统一读 pkl/csv，带缓存
└── 启动Dashboard.bat         # 双击入口
```

---

## 五、各页面详细设计

### Page 1：周报（首页）

**布局：**
```
[ 顶部 Regime 横幅（全宽，颜色随状态变）          ]
[ Top-5 行业卡片 ] [ ETF 执行卡片 ] [ 操作卡片   ]
[ 换手率 / 持仓变动摘要卡片（全宽）               ]
[ 实时日志卡片（任务运行时展开）                  ]
```

**Regime 横幅**
- 全宽渐变色条，四状态各有主题色
- 显示：状态名（大字）+ 模型版本 + 数据月份 + 生成时间

**Top-5 行业卡片**
- Regime集成 / 等权集成 两个 Tab
- 排名列用数字徽章，得分列用色条，「对比上次」用 ↑↓↔ 着色（红涨绿跌）

**ETF 执行卡片**
- R² < 0.85 行标红色左边框 + ⚠ 图标
- 规模 < 2 亿行标黄色左边框

**操作卡片（常用按钮）**

| 按钮 | 后端调用 | 耗时 |
|---|---|---|
| 更新数据 | `data_update.run()` | ~2-5 分钟 |
| 生成周报 | `monitor.run()` | ~1 分钟 |
| 一键周报流程 | `scheduler --once weekly` | ~5-10 分钟 |

- 任务运行中按钮变灰 + 显示旋转图标
- 日志卡片自动展开，SSE 实时滚动

---

### Page 2：个股持仓

- 顶部：Regime集成 / 等权集成 切换
- 过滤栏：行业多选 + 板块多选（科创 / 创业 / 主板）
- 表格：代码、名称、行业、板块、β、动量（红涨绿跌）、综合分，可点表头排序
- 底部折叠：「主板备选」子集

---

### Page 3：回测表现

**Tab 1：策略对比**
- 数据：`results/backtest_nav.csv`
- Chart.js 折线图：五条曲线，图例可点击开关
- 指标表：`results/backtest_summary.csv`（年化 / Sharpe / 最大回撤）

**Tab 2：ETF 执行回测**
- 数据：`results/etf_backtest.csv` + `etf_backtest_summary.csv`
- 三策略对比曲线 + 指标表

**Tab 3：历史信号轨迹**
- 解析 `reports/2026-W*.md`，提取每周 Regime 状态和 Top-5 行业
- Regime 时间轴色块图
- 各行业入选频次横向柱状图

---

### Page 4：系统状态

**数据健康度卡片**
- 复用 `monitor._data_source_status()` 逻辑
- 每行：数据名 / 最新月份 / 是否滞后（绿色 ✓ / 红色 ✗）

**模型版本卡片**
- 训练截止日期、模型标签、上次重训时间

**高危操作区（单独分区，视觉隔离）**

| 按钮 | 后端调用 | 说明 |
|---|---|---|
| 更新 processed 因子 | `data_update.run(processed=True)` | 全量重算三类因子 |
| 季度重训 | `train.train('quarterly')` | 耗时 30 分钟+，有二次确认弹窗 |

---

## 六、长任务 SSE 方案

```python
# api/runner.py  核心思路
from fastapi.responses import StreamingResponse
import subprocess, asyncio

async def stream_task(cmd: list[str]):
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    async for line in proc.stdout:
        yield f"data: {line.decode().rstrip()}\n\n"
    await proc.wait()
    yield f"data: [DONE] exit={proc.returncode}\n\n"

@router.get("/run/weekly")
async def run_weekly():
    cmd = ["python", "-m", "live.scheduler", "--once", "weekly"]
    return StreamingResponse(stream_task(cmd), media_type="text/event-stream")
```

```js
// js/utils/sse.js  前端接收
function startTask(url, logEl) {
  const es = new EventSource(url);
  es.onmessage = e => {
    if (e.data.startsWith('[DONE]')) { es.close(); return; }
    logEl.textContent += e.data + '\n';
    logEl.scrollTop = logEl.scrollHeight;
  };
}
```

---

## 七、实施步骤

1. 搭目录骨架，写 `main.py`（pywebview + FastAPI 启动逻辑）
2. 写 `utils/data_loader.py`（读 pkl/csv 统一缓存）
3. 写 `base.css`（玻璃卡片、配色变量、布局）+ `app.js`（路由）
4. 实现 Page 1 周报（接口 + 前端 + 三个操作按钮 SSE 联调）
5. 实现 Page 2 持仓（过滤 + 排序）
6. 实现 Page 3 回测（Chart.js 三个 Tab）
7. 实现 Page 4 系统状态（健康度 + 重训按钮）
8. 写 `启动Dashboard.bat`，整体测试

---

## 八、新增依赖

```
pywebview>=5.0
fastapi>=0.111
uvicorn>=0.30
```

其余（pandas、numpy）项目已有，Chart.js 通过 CDN 引入，无需安装。

---

## 九、启动方式

```bat
:: 启动Dashboard.bat
@echo off
cd /d "D:\desktop\有意思的事情\量化\项目\Regime-GNN-LSTM"
python dashboard\main.py
```

双击 `启动Dashboard.bat` 即可，无需手动开浏览器或命令行。
