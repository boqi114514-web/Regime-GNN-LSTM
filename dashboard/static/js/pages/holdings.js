// ── 持仓页 ────────────────────────────────────────────────────────────────

onPageShow('holdings', initHoldings);

let _holdingsData = null;
let _holdingsBranch = null;  // 记录上次拉的分支，分支变了重拉
let _holdingsTab  = 'regime';
let _boardFilter  = new Set(['科创', '创业', '主板']);
let _sortCol      = 'score';
let _sortAsc      = false;

const _BRANCH_LABELS = {
  'main':                            'Main',
  'fix/macro-neutral-fill':          'Fix',
  'refactor/equal-weight-ensemble':  'Refactor',
};

async function initHoldings() {
  const root = document.getElementById('page-holdings');
  const branch = window._selectedBranch || 'main';
  const branchLabel = _BRANCH_LABELS[branch] || branch;

  root.innerHTML = `
    <div class="page-header">
      <div class="page-title">个股持仓</div>
      <div class="page-subtitle">来源：${branchLabel} 分支归档 · 按行业分组（在周报页切换分支）</div>
    </div>

    <div class="glass-card" style="margin-bottom:16px">
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
        <div class="tab-group" id="holdings-tabs">
          <button class="tab-btn active" data-tab="regime">Regime 集成</button>
          <button class="tab-btn" data-tab="equal">等权集成</button>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <span class="filter-label">板块</span>
          <div class="filter-group" id="board-filters">
            <span class="filter-chip active" data-board="科创">科创</span>
            <span class="filter-chip active" data-board="创业">创业</span>
            <span class="filter-chip active" data-board="主板">主板</span>
          </div>
        </div>
      </div>
    </div>

    <div class="glass-card">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
        <div class="card-title">持仓明细 <span id="holdings-count" style="font-size:13px;color:var(--text-2);font-weight:400"></span></div>
      </div>
      <table class="data-table">
        <thead>
          <tr>
            <th>代码</th>
            <th>名称</th>
            <th>板块</th>
            <th class="sortable" data-col="beta" style="cursor:pointer">β <span class="sort-icon">↕</span></th>
            <th class="sortable" data-col="momentum_val" style="cursor:pointer">动量 <span class="sort-icon">↕</span></th>
            <th class="sortable" data-col="score" style="cursor:pointer">综合分 <span class="sort-icon">↕</span></th>
          </tr>
        </thead>
        <tbody id="holdings-tbody"></tbody>
      </table>
    </div>
  `;

  if (!_holdingsData || _holdingsBranch !== branch) {
    try {
      _holdingsData = await fetch('/api/holdings?branch=' + encodeURIComponent(branch))
                        .then(r => r.json());
      _holdingsBranch = branch;
    } catch (e) { return; }
  }

  // Tab
  root.querySelectorAll('#holdings-tabs .tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      root.querySelectorAll('#holdings-tabs .tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      _holdingsTab = btn.dataset.tab;
      renderHoldings();
    });
  });

  // 板块过滤
  root.querySelectorAll('#board-filters .filter-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      chip.classList.toggle('active');
      _boardFilter[chip.classList.contains('active') ? 'add' : 'delete'](chip.dataset.board);
      renderHoldings();
    });
  });

  // 排序（点列头改变组内排序列）
  root.querySelectorAll('th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (_sortCol === col) _sortAsc = !_sortAsc;
      else { _sortCol = col; _sortAsc = false; }
      renderHoldings();
    });
  });

  renderHoldings();
}

function renderHoldings() {
  const all = (_holdingsData[_holdingsTab] || []);
  if (!all.length) {
    document.getElementById('holdings-count').textContent = '';
    document.getElementById('holdings-tbody').innerHTML =
      `<tr><td colspan="6" style="text-align:center;color:var(--text-3);padding:24px">
         暂无选股数据，请先运行流水线生成周报
       </td></tr>`;
    return;
  }

  // 按行业分组（保持原始行业顺序）
  const groups = {};
  const indOrder = [];
  for (const s of all) {
    const ind = s.industry || '未知行业';
    if (!groups[ind]) { groups[ind] = []; indOrder.push(ind); }
    groups[ind].push(s);
  }

  // 组内按所选列排序
  for (const ind of indOrder) {
    groups[ind].sort((a, b) => {
      const av = parseFloat(a[_sortCol]) || 0;
      const bv = parseFloat(b[_sortCol]) || 0;
      return _sortAsc ? av - bv : bv - av;
    });
  }

  const boardTag = b => {
    if (b === '科创') return `<span class="board-tag board-kcb">科创</span>`;
    if (b === '创业') return `<span class="board-tag board-cyb">创业</span>`;
    return `<span class="board-tag board-main">主板</span>`;
  };

  const momHtml = s => {
    const v = s.momentum_val;
    const cls = v > 0 ? 'up' : v < 0 ? 'down' : '';
    return `<span class="${cls}">${s.momentum}</span>`;
  };

  const rows = [];
  let totalVisible = 0;

  for (const ind of indOrder) {
    const stocks = groups[ind].filter(s => _boardFilter.has(s.board));
    if (!stocks.length) continue;
    totalVisible += stocks.length;

    rows.push(`<tr class="ind-group-header">
      <td colspan="6">
        <strong>${ind}</strong>
        <span class="ind-count">${stocks.length} 只</span>
      </td>
    </tr>`);

    for (const s of stocks) {
      rows.push(`<tr>
        <td class="mono">${s.code}</td>
        <td><strong>${s.name}</strong></td>
        <td>${boardTag(s.board)}</td>
        <td class="mono">${s.beta}</td>
        <td class="mono">${momHtml(s)}</td>
        <td class="mono"><strong>${s.score}</strong></td>
      </tr>`);
    }
  }

  document.getElementById('holdings-count').textContent = `(${totalVisible} 只)`;
  document.getElementById('holdings-tbody').innerHTML = rows.join('');
}
