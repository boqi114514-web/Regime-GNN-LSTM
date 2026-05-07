// ── 系统页 ────────────────────────────────────────────────────────────────

onPageShow('system', initSystem);

async function initSystem() {
  const root = document.getElementById('page-system');
  root.innerHTML = `
    <div class="page-header">
      <div class="page-title">系统状态</div>
      <div class="page-subtitle">数据健康度 · 模型版本 · 高危操作</div>
    </div>

    <div class="system-grid" style="margin-bottom:16px">
      <!-- 数据健康度 -->
      <div class="glass-card">
        <div class="card-title">数据健康度</div>
        <div class="info-list" id="data-health"></div>
      </div>

      <!-- 模型信息 -->
      <div class="glass-card">
        <div class="card-title">模型信息</div>
        <div class="info-list" id="model-info"></div>
      </div>
    </div>

    <!-- 高危操作区 -->
    <div class="danger-zone">
      <div class="danger-zone-title">⚠ 高危操作</div>
      <div class="danger-zone-desc">以下操作耗时较长或影响范围较大，点击后需二次确认</div>
      <div class="danger-buttons">
        <button class="btn btn-danger" id="btn-update-proc">
          <div class="spinner"></div>
          <span class="btn-text">⟳ 更新 processed 因子（全量重算）</span>
        </button>
        <button class="btn btn-danger" id="btn-retrain">
          <div class="spinner"></div>
          <span class="btn-text">▶ 季度全量重训（30 分钟+）</span>
        </button>
      </div>
      <div class="log-panel section-gap" id="system-log"></div>
    </div>
  `;

  // 拉数据
  try {
    const d = await fetch('/api/system').then(r => r.json());
    renderHealth(d);
    renderModelInfo(d);
  } catch (e) {
    document.getElementById('data-health').innerHTML = `<div style="color:var(--up)">加载失败: ${e.message}</div>`;
  }

  // 高危按钮
  const logEl = document.getElementById('system-log');
  const btnProc    = document.getElementById('btn-update-proc');
  const btnRetrain = document.getElementById('btn-retrain');
  const btns = [btnProc, btnRetrain];

  btnProc.addEventListener('click', () => {
    confirm(
      '更新 processed 因子',
      '将全量重算景气度、价量、走势复刻三类因子，耗时约 10-20 分钟。确认执行？',
      () => runTask('/api/run/update_processed', { logEl, btns })
    );
  });

  btnRetrain.addEventListener('click', () => {
    confirm(
      '季度全量重训',
      '将重训 HMM + GNN + LSTM-B 三个模型，耗时 30 分钟以上，期间不可使用其他任务。确认执行？',
      () => runTask('/api/run/train_quarterly', {
        logEl, btns,
        onDone: () => {
          // 刷新系统状态
          fetch('/api/system').then(r => r.json()).then(d => {
            renderHealth(d);
            renderModelInfo(d);
          });
        },
      })
    );
  });
}

function renderHealth(d) {
  const container = document.getElementById('data-health');
  if (!container) return;
  container.innerHTML = (d.data_sources || []).map(s => {
    const dot = s.exists
      ? `<span class="status-dot status-ok"></span>`
      : `<span class="status-dot status-err"></span>`;
    return `<div class="info-row">
      <span class="info-key">${dot} ${s.name}</span>
      <span class="info-val">${s.latest}</span>
    </div>`;
  }).join('');
}

function renderModelInfo(d) {
  const container = document.getElementById('model-info');
  if (!container) return;
  const rows = [
    ['模型版本',   d.model_version || '--'],
    ['上次重训',   d.last_retrain  || '--'],
    ['推理月份',   d.infer_month   || '--'],
  ];
  container.innerHTML = rows.map(([k, v]) =>
    `<div class="info-row">
      <span class="info-key">${k}</span>
      <span class="info-val">${v}</span>
    </div>`
  ).join('');
}
