// ── 周报页 ────────────────────────────────────────────────────────────────

onPageShow('report', initReport);

let _reportData    = null;
let _selectedBranch = 'main';

const BRANCHES = [
  { id: 'main',                         label: 'Main',     desc: '滚动HMM · Sharpe 1.38' },
  { id: 'fix/macro-neutral-fill',       label: 'Fix',      desc: '全量HMM · Sharpe 1.29' },
  { id: 'refactor/equal-weight-ensemble', label: 'Refactor', desc: '等权集成 · Sharpe 1.28' },
];

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
        <div class="regime-meta"  id="regime-meta">--</div>
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
          <div class="branch-source" id="branch-source"></div>
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

        <!-- 分支选择器 -->
        <div class="branch-selector" id="branch-selector">
          ${BRANCHES.map(b => `
            <button class="branch-btn${b.id === _selectedBranch ? ' active' : ''}"
                    data-branch="${b.id}"
                    title="${b.desc}">
              <span class="branch-name">${b.label}</span>
              <span class="branch-desc">${b.desc}</span>
            </button>`).join('')}
        </div>

        <div class="action-buttons">
          <button class="btn btn-primary" id="btn-run-pipeline">
            <div class="spinner"></div>
            <span class="btn-text">▶ 运行流水线</span>
          </button>
          <button class="btn btn-secondary" id="btn-fetch-data">
            <div class="spinner"></div>
            <span class="btn-text">↓ 拉取新数据</span>
          </button>
          <button class="btn btn-secondary" id="btn-monitor">
            <div class="spinner"></div>
            <span class="btn-text">⊞ 仅生成周报</span>
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
            <th>行业</th><th>ETF 代码</th><th>ETF 名称</th>
            <th>R²</th><th>β</th><th>规模(亿)</th>
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

  // 分支选择器：点击 → 拉该分支的归档周报
  root.querySelectorAll('.branch-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      root.querySelectorAll('.branch-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      _selectedBranch = btn.dataset.branch;
      await loadBranchReport(_selectedBranch);
    });
  });

  // 操作按钮
  const logEl = document.getElementById('action-log');
  const btns  = ['btn-run-pipeline', 'btn-fetch-data', 'btn-monitor']
                  .map(id => document.getElementById(id));

  // 跑完流水线 → 拉当前选中分支的归档（保留按钮高亮，不被 latest 覆盖）
  const reloadCurrentBranch = () => {
    _reportData = null;
    loadBranchReport(_selectedBranch);
  };

  document.getElementById('btn-run-pipeline').addEventListener('click', () =>
    runTask(`/api/run/pipeline?branch=${encodeURIComponent(_selectedBranch)}`,
            { logEl, btns, onDone: reloadCurrentBranch }));

  document.getElementById('btn-fetch-data').addEventListener('click', () =>
    runTask('/api/run/fetch_data', { logEl, btns }));

  // monitor 只生成 latest.md（不归档分支），跑完重拉 latest
  document.getElementById('btn-monitor').addEventListener('click', () =>
    runTask('/api/run/monitor', { logEl, btns, onDone: () => {
      _reportData = null;
      initReport();
    } }));

  // 永远按当前选中分支拉归档（无归档则 loadBranchReport 自己显示提示）
  await loadBranchReport(_selectedBranch);
}

async function loadBranchReport(branch) {
  const metaEl = document.getElementById('report-meta');
  if (metaEl) metaEl.textContent = `加载 ${branch} 分支周报...`;
  try {
    const data = await fetch(`/api/report?branch=${encodeURIComponent(branch)}`)
                   .then(r => r.json());
    _reportData = data;
    if (data.error) {
      // 清空主要内容区，显示提示
      document.getElementById('industry-tbody').innerHTML =
        `<tr><td colspan="4" style="text-align:center;color:var(--text-3);padding:40px">${data.error}</td></tr>`;
      document.getElementById('etf-tbody').innerHTML = '';
      document.getElementById('turnover-grid').innerHTML =
        `<div style="color:var(--text-3)">${data.error}</div>`;
      if (metaEl) metaEl.textContent = data.error;
      return;
    }
    renderReport(data);
  } catch (e) {
    if (metaEl) metaEl.textContent = '加载失败: ' + e.message;
  }
}

function renderReport(d) {
  if (d.error) {
    document.getElementById('report-meta').textContent = d.error;
    return;
  }

  document.getElementById('report-meta').textContent =
    `${d.iso_week}  ·  数据月份: ${d.data_month}  ·  模型: ${d.model_version}  ·  生成: ${d.generated_at}`;

  // Regime 横幅
  const banner = document.getElementById('regime-banner');
  banner.dataset.regime = d.regime || 'unknown';
  document.getElementById('regime-value').textContent   = d.regime || '--';
  document.getElementById('regime-meta').textContent    = d.data_month;
  document.getElementById('regime-weights').textContent = d.regime_weights || '';
  if (d.consensus) document.getElementById('regime-consensus').style.display = 'inline-flex';

  // 来源标签
  const branchEl = document.getElementById('branch-source');
  if (branchEl && d.branch_used) {
    const meta = BRANCHES.find(b => b.id === d.branch_used) || { label: d.branch_used, desc: '' };
    branchEl.textContent = `来源：${meta.label}  ${meta.desc}`;
  }

  renderIndustries(d);
  renderEtf(d.etf_list || []);
  renderTurnover(d);
}

function renderIndustries(d) {
  const list   = d.industries || d.industries_regime || d.industries_equal || [];
  const tbody  = document.getElementById('industry-tbody');
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
    const cls    = r.warn_r2 ? 'warn-row' : (r.warn_scale ? 'warn-scale-row' : '');
    const r2Cls  = r.warn_r2 ? 'up' : 'down';
    const r2Ico  = r.warn_r2 ? ' ⚠' : '';
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

  const t = d.turnover || d.turnover_regime || d.turnover_equal || {};
  if (!t.rate) {
    grid.innerHTML = `<div style="color:var(--text-3)">无换手数据</div>`;
    return;
  }
  const newChips = (t.new_in || []).map(n => `<span class="chip chip-in">${n}</span>`).join('');
  const outChips = (t.kicked  || []).map(n => `<span class="chip chip-out">${n}</span>`).join('');
  grid.innerHTML = `
    <div class="turnover-section">
      <div class="turnover-rate">${t.rate} <span>换手率</span></div>
      ${newChips ? `<div style="margin-bottom:6px;font-size:12px;color:var(--text-2)">新进入</div>
                    <div class="chip-list" style="margin-bottom:10px">${newChips}</div>` : ''}
      ${outChips ? `<div style="margin-bottom:6px;font-size:12px;color:var(--text-2)">踢出</div>
                    <div class="chip-list">${outChips}</div>` : ''}
    </div>`;
}
