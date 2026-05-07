// ── 持仓页 ────────────────────────────────────────────────────────────────

onPageShow('holdings', initHoldings);

let _holdingsData = null;
let _holdingsTab  = 'regime';
let _boardFilter  = new Set(['科创', '创业', '主板']);
let _indFilter    = new Set();
let _sortCol      = 'score';
let _sortAsc      = false;

async function initHoldings() {
  const root = document.getElementById('page-holdings');
  root.innerHTML = `
    <div class="page-header">
      <div class="page-title">个股持仓</div>
      <div class="page-subtitle">来源：最新周报选股层</div>
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
        <div style="display:flex;align-items:center;gap:8px">
          <span class="filter-label">行业</span>
          <div class="filter-group" id="ind-filters"></div>
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
            <th>行业</th>
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

  if (!_holdingsData) {
    try {
      _holdingsData = await fetch('/api/holdings').then(r => r.json());
    } catch (e) { return; }
  }

  // 初始化行业过滤
  const allInds = [...new Set([
    ..._holdingsData.regime.map(s => s.industry),
    ..._holdingsData.equal.map(s => s.industry),
  ])];
  _indFilter = new Set(allInds);

  const indContainer = document.getElementById('ind-filters');
  allInds.forEach(ind => {
    const chip = document.createElement('span');
    chip.className = 'filter-chip active';
    chip.dataset.ind = ind;
    chip.textContent = ind;
    chip.addEventListener('click', () => {
      chip.classList.toggle('active');
      _indFilter[chip.classList.contains('active') ? 'add' : 'delete'](ind);
      renderHoldings();
    });
    indContainer.appendChild(chip);
  });

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

  // 排序
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
  const list = (_holdingsData[_holdingsTab] || [])
    .filter(s => _boardFilter.has(s.board) && _indFilter.has(s.industry));

  // 排序
  list.sort((a, b) => {
    const av = parseFloat(a[_sortCol]) || 0;
    const bv = parseFloat(b[_sortCol]) || 0;
    return _sortAsc ? av - bv : bv - av;
  });

  document.getElementById('holdings-count').textContent = `(${list.length} 只)`;

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

  const tbody = document.getElementById('holdings-tbody');
  tbody.innerHTML = list.map(s => `<tr>
    <td>${s.industry}</td>
    <td class="mono">${s.code}</td>
    <td><strong>${s.name}</strong></td>
    <td>${boardTag(s.board)}</td>
    <td class="mono">${s.beta}</td>
    <td class="mono">${momHtml(s)}</td>
    <td class="mono"><strong>${s.score}</strong></td>
  </tr>`).join('');
}
