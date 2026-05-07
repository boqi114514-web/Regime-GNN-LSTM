// ── 页面路由 ─────────────────────────────────────────────────────────────
const _pageHandlers = {};

function showPage(name) {
  document.querySelectorAll('.page').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  const page = document.getElementById(`page-${name}`);
  const nav  = document.querySelector(`[data-page="${name}"]`);
  if (page) page.classList.add('active');
  if (nav)  nav.classList.add('active');
  if (_pageHandlers[name]) _pageHandlers[name]();
}

function onPageShow(name, fn) {
  _pageHandlers[name] = fn;
}

// 导航点击
document.querySelectorAll('.nav-item').forEach(el => {
  el.addEventListener('click', e => {
    e.preventDefault();
    showPage(el.dataset.page);
  });
});

// 弹窗确认
function confirm(title, desc, onOk) {
  const overlay = document.getElementById('confirm-overlay');
  if (!overlay) return;
  overlay.querySelector('h3').textContent = title;
  overlay.querySelector('p').textContent  = desc;
  const ok = overlay.querySelector('#confirm-ok');
  const cancel = overlay.querySelector('#confirm-cancel');
  overlay.classList.add('visible');
  const close = () => overlay.classList.remove('visible');
  const handleOk = () => { close(); onOk(); ok.removeEventListener('click', handleOk); cancel.removeEventListener('click', close); };
  ok.addEventListener('click', handleOk);
  cancel.addEventListener('click', close);
}

// 注入弹窗 DOM
document.body.insertAdjacentHTML('beforeend', `
<div class="confirm-overlay" id="confirm-overlay">
  <div class="confirm-box">
    <h3></h3>
    <p></p>
    <div class="confirm-actions">
      <button class="btn btn-secondary" id="confirm-cancel">取消</button>
      <button class="btn btn-danger"    id="confirm-ok">确认执行</button>
    </div>
  </div>
</div>
`);

// 启动：先加载周报页，同时更新侧边栏模型标识
async function _initSidebar() {
  try {
    const d = await fetch('/api/report').then(r => r.json());
    const el = document.getElementById('sidebar-model');
    if (el && d.model_version) el.textContent = d.model_version;
  } catch (_) {}
}

// 初始化由 index.html 末尾内联脚本触发（保证页面脚本先注册完）
