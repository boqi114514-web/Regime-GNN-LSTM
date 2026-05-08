// ── SSE 任务流封装 ────────────────────────────────────────────────────────

/**
 * 启动一个 SSE 任务流
 * @param {string} url  - API 端点
 * @param {object} opts - { logEl, btns, onDone, onError }
 *   logEl : 日志容器 DOM 元素
 *   btns  : 需要在任务期间禁用的按钮数组
 *   onDone: 任务成功回调
 *   onError: 任务失败回调
 */
function runTask(url, { logEl, btns = [], onDone = null, onError = null } = {}) {
  // 禁用按钮 + 显示日志
  btns.forEach(b => { b.disabled = true; b.classList.add('loading'); });
  if (logEl) {
    logEl.textContent = '';
    logEl.classList.add('visible');
  }

  const es = new EventSource(url);

  es.onmessage = e => {
    const data = e.data;

    if (data.startsWith('[DONE]')) {
      appendLog(logEl, data, 'log-done');
      es.close();
      btns.forEach(b => { b.disabled = false; b.classList.remove('loading'); });
      if (onDone) onDone();
      return;
    }
    if (data.startsWith('[ERROR]')) {
      appendLog(logEl, data, 'log-error');
      es.close();
      btns.forEach(b => { b.disabled = false; b.classList.remove('loading'); });
      if (onError) onError(data);
      return;
    }
    if (data.startsWith('[BUSY]')) {
      appendLog(logEl, data, 'log-busy');
      es.close();
      btns.forEach(b => { b.disabled = false; b.classList.remove('loading'); });
      return;
    }
    appendLog(logEl, data);
  };

  es.onerror = () => {
    appendLog(logEl, '[ERROR] 连接断开', 'log-error');
    es.close();
    btns.forEach(b => { b.disabled = false; b.classList.remove('loading'); });
    if (onError) onError('连接断开');
  };

  return es;
}

function appendLog(el, text, cls = '') {
  if (!el) return;
  const line = document.createElement('div');
  if (cls) line.className = cls;
  line.textContent = text;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}
