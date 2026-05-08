// ── Chart.js 封装 ─────────────────────────────────────────────────────────

const CHART_COLORS = [
  '#1A73E8', '#1E8E3E', '#F9AB00', '#D93025', '#9AA0A6',
  '#7B1FA2', '#00838F', '#E65100',
];

/**
 * 绘制 NAV 折线图
 * @param {string} canvasId
 * @param {object} data  - { dates: [], series: { 策略名: [values] } }
 * @param {object} opts  - Chart.js 额外配置
 * @returns Chart 实例
 */
function makeNavChart(canvasId, data, opts = {}) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  // 销毁旧实例
  const existing = Chart.getChart(ctx);
  if (existing) existing.destroy();

  const datasets = Object.entries(data.series).map(([name, values], i) => ({
    label: name,
    data: values,
    borderColor: CHART_COLORS[i % CHART_COLORS.length],
    backgroundColor: CHART_COLORS[i % CHART_COLORS.length] + '12',
    borderWidth: 2,
    pointRadius: 0,
    pointHitRadius: 8,
    fill: false,
    tension: 0.3,
  }));

  return new Chart(ctx, {
    type: 'line',
    data: { labels: data.dates, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: { font: { family: "'Noto Sans SC'", size: 12 }, boxWidth: 20, padding: 16 },
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { maxTicksLimit: 10, font: { family: "'JetBrains Mono'", size: 11 } },
          grid: { color: 'rgba(0,0,0,0.04)' },
        },
        y: {
          ticks: { callback: v => v.toFixed(2), font: { family: "'JetBrains Mono'", size: 11 } },
          grid: { color: 'rgba(0,0,0,0.04)' },
        },
      },
      ...opts,
    },
  });
}

/**
 * 绘制横向柱状图（行业入选频次）
 */
function makeBarChart(canvasId, labels, values, opts = {}) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  const existing = Chart.getChart(ctx);
  if (existing) existing.destroy();

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: CHART_COLORS[0] + 'CC',
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(0,0,0,0.04)' }, ticks: { font: { family: "'JetBrains Mono'", size: 11 } } },
        y: { grid: { display: false }, ticks: { font: { family: "'Noto Sans SC'", size: 12 } } },
      },
      ...opts,
    },
  });
}
