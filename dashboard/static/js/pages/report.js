// ── 周报页 ────────────────────────────────────────────────────────────────

onPageShow('report', initReport);

let _reportData = null;
let _currentTab = 'regime'; // 'regime' | 'equal'

async function initReport() {
  const root = document.getElementById('page-report');
  root.innerHTML = `
    <div class="page-header">
      <div class="page-title">行业轮动周报</div>
      <div class="page-subtitle" id="report-meta">加载中...</div>
    </div>

    <div class="regime-banner" id="regime-banner" data-regime="unknown">
      <div class="regime-left">
        <div class="regime-label">当前市场机制</div>
        <div class="regime-value" id="regime-value">--</div>
        <div class="regime-meta" id="regime-meta">--</div>
      </div>
      <div class="regime-right">
        <div class="regime-weights" id="regime-weights">--</div>
        <div class="regime-consensus" id="regime-consensus" style="display:none">✓ 两模型一致</div>
      </div>
    </div>

    <div class="report-grid">
      <!-- 行业推荐 -->
      <div class="glass-card col-span-2">
        <div class="card-header">
          <div class="card-title">Top-5 推荐行业</div>
          <div class="tab-group">
            <button class="tab-btn active" data-tab="regime">Regime 集成</button>
            <button class="tab-btn" data-tab="equal">等权集成</button>
          </div>
        </div>
        <table class="data-table">
          <thead>
            <tr>
              <th style="width:40px">排名</th>
              <th>行业</th>
              <th style="width:160px">得分</th>
              <th style="width:80px">变化</th>
            </tr>
          </thead>
          <tbody id="industry-tbody"></tbody>
        </table>
      </div>

      <!-- 操作面板 -->
      <div class="glass-card action-card">
        <div class="card-title">操作</div>
        <div class="action-buttons">
          <button class="btn btn-primary" id="btn-weekly">
            <div class="spinner"></div>
            <span class="btn-text">⚡ 一键周报流程</span>
          </button>
          <button class="btn btn-secondary" id="btn-update">
            <div class="spinner"></div>
            <span class="btn-text">↓ 更新数据</span>
          </button>
          <button class="btn btn-secondary" id="btn-monitor">
            <div class="spinner"></div>
            <span class="btn-text">⊞ 生成周报</span>
          </button>
        </div>
        <div class="log-panel section-gap" id="action-log"></div>
      </div>
    </div>

    <!-- ETF 执行 -->
    <div class="glass-card etf-card">
      <div class="card-title">ETF 执行载体</div>
      <table class="data-table">
        <thead>
          <tr>
            <th>行业</th>
            <th>ETF 代码</th>
            <th>ETF 名称</th>
            <th>R²</th>
            <th>β</th>
            <th>规模(亿)</th>
          </tr>
        </thead>
        <tbody id="etf-tbody"></tbody>
      </table>
    </div>

    <!-- 换手率 -->
    <div class="glass-card">
      <div class="card-title">持仓变动</div>
      <div class="turnover-grid" id="turnover-grid"></div>
    </div>
  `;

  // Tab 切换
  root.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      root.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      _currentTab = btn.dataset.tab;
      if (_reportData) renderIndustries(_reportData);
    });
  });

  // 操作按钮
  const btns = [
    document.getElementById('btn-weekly'),
    document.getElementById('btn-update'),
    document.getElementById('btn-monitor'),
  ];
  const logEl = document.getElementById('action-log');

  const reloadAfterDone = () => { _reportData = null; initReport(); };

  btns[0].addEventListener('click', () =>
    runTask('/api/run/weekly',  { logEl, btns, onDone: reloadAfterDone }));
  btns[1].addEventListener('click', () =>
    runTask('/api/run/update',  { logEl, btns }));
  btns[2].addEventListener('click', () =>
    runTask('/api/run/monitor', { logEl, btns, onDone: reloadAfterDone }));

  // 拉数据
  if (!_reportData) {
    try {
      _reportData = await fetch('/api/report').then(r => r.json());
    } catch (e) {
      document.getElementById('report-meta').textContent = '加载失败: ' + e.message;
      return;
    }
  }
  renderReport(_reportData);
}

function renderReport(d) {
  if (d.error) {
    document.getElementById('report-meta').textContent = d.error;
    return;
  }

  // 元信息
  document.getElementById('report-meta').textContent =
    `${d.iso_week}  ·  数据月份: ${d.data_month}  ·  模型: ${d.model_version}  ·  生成: ${d.generated_at}`;

  // Regime 横幅
  const banner = document.getElementById('regime-banner');
  banner.dataset.regime = d.regime || 'unknown';
  document.getElementById('regime-value').textContent = d.regime || '--';
  document.getElementById('regime-meta').textContent  = d.data_month;
  document.getElementById('regime-weights').textContent = d.regime_weights || '';
  if (d.consensus) document.getElementById('regime-consensus').style.display = 'inline-flex';

  renderIndustries(d);
  renderEtf(d.etf_list || []);
  renderTurnover(d);
}

function renderIndustries(d) {
  const list = _currentTab === 'regime' ? d.industries_regime : d.industries_equal;
  const tbody = document.getElementById('industry-tbody');
  if (!tbody) return;
  const maxScore = Math.max(...list.map(r => parseFloat(r.score) || 0));
  tbody.innerHTML = list.map(r => {
    const pct = maxScore > 0 ? (parseFloat(r.score) / maxScore * 100).toFixed(0) : 0;
    const changeHtml = r.change_dir === 'up'
      ? `<span class="up">↑${r.change_val}</span>`
      : r.change_dir === 'down'
        ? `<span class="down">↓${r.change_val}</span>`
        : `<span class="flat">↔</span>`;
    return `<tr>
      <td><div class="rank-badge ${r.rank === '1' ? 'top1' : ''}">${r.rank}</div></td>
      <td><strong>${r.name}</strong> <span class="mono" style="font-size:11px;color:var(--text-3)">${r.code}</span></td>
      <td>
        <div class="score-bar-wrap">
          <div class="score-bar"><div class="score-bar-fill" style="width:${pct}%"></div></div>
          <span class="score-val">${r.score}</span>
        </div>
      </td>
      <td>${changeHtml}</td>
    </tr>`;
  }).join('');
}

function renderEtf(list) {
  const tbody = document.getElementById('etf-tbody');
  if (!tbody) return;
  tbody.innerHTML = list.map(r => {
    const cls = r.warn_r2 ? 'warn-row' : (r.warn_scale ? 'warn-scale-row' : '');
    const r2Cls = r.warn_r2 ? 'up' : 'down';
    const r2Ico = r.warn_r2 ? ' ⚠' : '';
    return `<tr class="${cls}">
      <td>${r.industry}</td>
      <td class="mono">${r.etf_code}</td>
      <td>${r.etf_name}</td>
      <td class="mono ${r2Cls}">${r.r2.toFixed(3)}${r2Ico}</td>
      <td class="mono">${r.beta}</td>
      <td class="mono ${r.warn_scale ? 'up' : ''}">${r.scale}</td>
    </tr>`;
  }).join('');
}

function renderTurnover(d) {
  const grid = document.getElementById('turnover-grid');
  if (!grid) return;

  const section = (title, t) => {
    if (!t || !t.rate) return `<div class="turnover-section"><h4>${title}</h4><div style="color:var(--text-3)">无数据</div></div>`;
    const newChips = (t.new_in || []).map(n => `<span class="chip chip-in">${n}</span>`).join('');
    const outChips = (t.kicked || []).map(n => `<span class="chip chip-out">${n}</span>`).join('');
    return `<div class="turnover-section">
      <h4>${title}</h4>
      <div class="turnover-rate">${t.rate} <span>换手率</span></div>
      ${newChips ? `<div style="margin-bottom:6px;font-size:12px;color:var(--text-2)">新进入</div><div class="chip-list" style="margin-bottom:10px">${newChips}</div>` : ''}
      ${outChips ? `<div style="margin-bottom:6px;font-size:12px;color:var(--text-2)">踢出</div><div class="chip-list">${outChips}</div>` : ''}
    </div>`;
  };

  grid.innerHTML = section('Regime 集成', d.turnover_regime) + section('等权集成', d.turnover_equal);
}
