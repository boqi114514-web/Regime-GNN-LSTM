// ── 回测页（重构：s4 选仓为主，s3/ETF/历史为次级）────────────────────────

onPageShow('backtest', initBacktest);

let _btTab = 'stock';

function initBacktest() {
  const root = document.getElementById('page-backtest');
  root.innerHTML = `
    <div class="page-header">
      <div class="page-title">回测表现</div>
      <div class="page-subtitle">主板选仓 2019-01 ~ 2026-05 · 5行业×5只=25股</div>
    </div>

    <div class="tab-group backtest-tabs" style="margin-bottom:16px">
      <button class="tab-btn active" data-bt="stock">股票选仓</button>
      <button class="tab-btn" data-bt="industry">行业轮动</button>
      <button class="tab-btn" data-bt="etf">ETF 执行</button>
      <button class="tab-btn" data-bt="history">历史信号</button>
    </div>

    <!-- ① 股票选仓（主板）-->
    <div id="bt-stock">
      <div class="glass-card" style="margin-bottom:16px">
        <div class="card-title">三分支净值曲线（主板过滤）</div>
        <div class="chart-wrap"><canvas id="chart-stock-nav"></canvas></div>
      </div>
      <div class="glass-card" style="margin-bottom:16px">
        <div class="card-title">逐年净收益对比</div>
        <div id="stock-annual-table" class="annual-table-wrap"></div>
      </div>
      <div class="glass-card">
        <div class="card-title">分支综合指标（5行业×5只，主板）</div>
        <div class="summary-grid" id="stock-summary"></div>
      </div>
    </div>

    <!-- ② 行业轮动（s3）-->
    <div id="bt-industry" style="display:none">
      <div class="glass-card" style="margin-bottom:16px">
        <div class="card-title">行业轮动净值曲线（s3 集成）</div>
        <div class="chart-wrap"><canvas id="chart-nav"></canvas></div>
      </div>
      <div class="glass-card">
        <div class="card-title">策略指标</div>
        <div class="summary-grid" id="strategy-summary"></div>
      </div>
    </div>

    <!-- ③ ETF 执行 -->
    <div id="bt-etf" style="display:none">
      <div class="glass-card" style="margin-bottom:16px">
        <div class="card-title">ETF 执行 vs 行业指数</div>
        <div class="chart-wrap"><canvas id="chart-etf"></canvas></div>
      </div>
      <div class="glass-card">
        <div class="card-title">ETF 回测指标</div>
        <div class="summary-grid" id="etf-summary"></div>
      </div>
    </div>

    <!-- ④ 历史信号 -->
    <div id="bt-history" style="display:none">
      <div class="glass-card" style="margin-bottom:16px">
        <div class="card-title">历史 Regime 与行业信号</div>
        <div class="history-timeline" id="history-list"></div>
      </div>
      <div class="glass-card">
        <div class="card-title">行业入选频次</div>
        <div style="height:320px"><canvas id="chart-freq"></canvas></div>
      </div>
    </div>
  `;

  root.querySelectorAll('.backtest-tabs .tab-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      root.querySelectorAll('.backtest-tabs .tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      _btTab = btn.dataset.bt;
      showBtPanel(_btTab);
      await loadBtTab(_btTab);
    });
  });

  loadBtTab('stock');
}

function showBtPanel(tab) {
  ['stock', 'industry', 'etf', 'history'].forEach(t => {
    document.getElementById(`bt-${t}`).style.display = t === tab ? '' : 'none';
  });
}

async function loadBtTab(tab) {
  if (tab === 'stock') {
    const [navData, summary, annual] = await Promise.all([
      fetch('/api/backtest/stock/nav').then(r => r.json()),
      fetch('/api/backtest/stock/summary').then(r => r.json()),
      fetch('/api/backtest/stock/annual').then(r => r.json()),
    ]);
    makeNavChart('chart-stock-nav', navData);
    renderAnnualTable('stock-annual-table', annual);
    renderSummary('stock-summary', summary);

  } else if (tab === 'industry') {
    const [navData, summary] = await Promise.all([
      fetch('/api/backtest/nav').then(r => r.json()),
      fetch('/api/backtest/summary').then(r => r.json()),
    ]);
    makeNavChart('chart-nav', navData);
    renderSummary('strategy-summary', summary);

  } else if (tab === 'etf') {
    const [navData, summary] = await Promise.all([
      fetch('/api/backtest/etf/nav').then(r => r.json()),
      fetch('/api/backtest/etf/summary').then(r => r.json()),
    ]);
    makeNavChart('chart-etf', navData);
    renderSummary('etf-summary', summary);

  } else if (tab === 'history') {
    const history = await fetch('/api/backtest/history').then(r => r.json());
    renderHistory(history);
  }
}

// ── 逐年收益表 ─────────────────────────────────────────────────────────────
function renderAnnualTable(containerId, data) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const { years, branches } = data;
  if (!years || !years.length) {
    container.innerHTML = '<div style="color:var(--text-3);padding:12px">暂无数据</div>';
    return;
  }

  const labels = Object.keys(branches);
  const pct = v => (v == null || isNaN(v)) ? '—' : (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%';
  const cls  = v => v > 0 ? 'annual-pos' : v < 0 ? 'annual-neg' : '';

  const headerCells = labels.map(l => `<th>${l}</th>`).join('');
  const bodyRows = years.map(yr => {
    const cells = labels.map(l => {
      const v = branches[l]?.[yr];
      return `<td class="${cls(v)}">${pct(v)}</td>`;
    }).join('');
    return `<tr><td class="annual-year">${yr}</td>${cells}</tr>`;
  }).join('');

  container.innerHTML = `
    <table class="annual-table">
      <thead><tr><th>年份</th>${headerCells}</tr></thead>
      <tbody>${bodyRows}</tbody>
    </table>
  `;
}

// ── 指标汇总卡片 ──────────────────────────────────────────────────────────
function renderSummary(containerId, rows) {
  const container = document.getElementById(containerId);
  if (!container || !rows.length) return;
  const keys = Object.keys(rows[0]);
  const nameKey = keys[0];

  const fmt = (key, val) => {
    if (typeof val !== 'number') return val;
    const k = key.toLowerCase();
    if (k.includes('sharpe') || k.includes('夏普'))  return val.toFixed(2);
    if (k.includes('month') || k === 'n_months')      return val.toFixed(0);
    return (val * 100).toFixed(2) + '%';
  };

  const colorVal = (key, val) => {
    if (typeof val !== 'number') return '';
    const k = key.toLowerCase();
    if (k.includes('drawdown') || k.includes('回撤')) return val < 0 ? 'color:var(--up)' : '';
    if (k.includes('annual') || k.includes('sharpe') || k.includes('win') || k.includes('年化'))
      return val > 0 ? 'color:var(--down)' : 'color:var(--up)';
    return '';
  };

  const LABELS = {
    annual_return:     '年化收益',
    annual_volatility: '年化波动',
    sharpe_ratio:      'Sharpe',
    max_drawdown:      '最大回撤',
    win_rate:          '胜率',
    n_months:          '月数',
  };

  container.innerHTML = rows.map(row => {
    const metricHtml = keys.slice(1).map(k => {
      const label = LABELS[k] || k;
      return `<div class="metric-card">
        <div class="metric-name">${label}</div>
        <div class="metric-value" style="${colorVal(k, row[k])}">${fmt(k, row[k])}</div>
      </div>`;
    }).join('');
    return `
      <div style="grid-column:1/-1;font-size:14px;font-weight:700;margin:8px 0 4px;color:var(--text-2)">
        ${row[nameKey]}
      </div>${metricHtml}`;
  }).join('');
}

// ── 历史信号 ──────────────────────────────────────────────────────────────
function renderHistory(list) {
  const container = document.getElementById('history-list');
  if (!container) return;

  container.innerHTML = list.map(r => `
    <div class="history-row">
      <span class="mono" style="font-size:12px">${r.week}</span>
      <span class="regime-pill" data-regime="${r.regime}">${r.regime || '--'}</span>
      <span class="mono" style="font-size:11px;color:var(--text-2)">${r.month || ''}</span>
      <div class="history-top5">
        ${(r.top5 || []).map(n => `<span class="mini-chip">${n}</span>`).join('')}
      </div>
    </div>
  `).join('');

  const freq = {};
  list.forEach(r => (r.top5 || []).forEach(n => { if (n) freq[n] = (freq[n] || 0) + 1; }));
  const sorted = Object.entries(freq).sort((a, b) => b[1] - a[1]);
  makeBarChart('chart-freq', sorted.map(([n]) => n), sorted.map(([, v]) => v));
}
