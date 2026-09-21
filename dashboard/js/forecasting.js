/**
 * forecasting.js — Member 2
 *
 * Builds the Forecasting section:
 *  - Actual vs Predicted line chart (Chart.js)
 *  - KPI cards (avg predicted, peak hour, MAE, MAPE)
 *  - Forecast horizon form → POST /api/v1/forecasting/predict
 *
 * Imports available from api.js:
 *   fetchForecasting, fetchForecastingPredict
 *
 * Imports available from app.js:
 *   showLoader, showError, animateCounter
 *
 * Containers (already in index.html):
 *   #forecasting-kpis              ← KPI card grid
 *   #forecasting-chart-container   ← Main canvas
 *   #forecasting-predict-container ← Predict form
 */

import { fetchForecasting, fetchForecastingPredict, ApiError } from './api.js';
import { showLoader, showError, animateCounter } from './app.js';

const ZONES = ['Total', 'Zone 1', 'Zone 2', 'Zone 3'];
const LIMIT_OPTIONS = [
  { value: 144, label: 'Last 24h' },
  { value: 288, label: 'Last 48h' },
  { value: 1008, label: 'Last 7d' },
];

const kpiContainer = document.getElementById('forecasting-kpis');
const chartContainer = document.getElementById('forecasting-chart-container');
const predictContainer = document.getElementById('forecasting-predict-container');

let chart = null;
let initialized = false;
let currentZone = 'Total';
let currentLimit = 144;

/* -------------------------------------------------------------------- */
/* Lifecycle wiring                                                       */
/* -------------------------------------------------------------------- */

window.addEventListener('app:navigate', (e) => {
  if (e.detail.sectionId === 'forecasting' && !initialized) {
    initialized = true;
    init();
  }
});

window.addEventListener('api:poll', () => {
  if (initialized) loadSeries(currentZone, currentLimit, { silent: true });
});

function init() {
  buildPredictForm();
  loadSeries(currentZone, currentLimit);
}

/* -------------------------------------------------------------------- */
/* Actual-vs-Predicted chart + KPIs                                       */
/* -------------------------------------------------------------------- */

async function loadSeries(zone, limit, { silent = false } = {}) {
  currentZone = zone;
  currentLimit = limit;

  if (!silent) showLoader(chartContainer, 'Loading forecast data…');

  try {
    const data = await fetchForecasting({ zone, limit });
    renderKPIs(data.summary); // render KPIs first so a Chart.js failure below can't hide them
    renderChartCard(data);
  } catch (err) {
    showError(chartContainer, apiErrorMessage(err));
  }
}

function renderChartCard(data) {
  chartContainer.innerHTML = `
    <div class="card-header">
      <div>
        <div class="card-title">📈 Actual vs Predicted</div>
        <div class="card-subtitle">${data.zone} · ${data.total_records} points</div>
      </div>
      <div class="flex gap-2">
        <select id="forecast-zone-select" class="form-select">
          ${ZONES.map((z) => `<option value="${z}" ${z === currentZone ? 'selected' : ''}>${z}</option>`).join('')}
        </select>
        <select id="forecast-limit-select" class="form-select">
          ${LIMIT_OPTIONS.map(
            (o) => `<option value="${o.value}" ${o.value === currentLimit ? 'selected' : ''}>${o.label}</option>`
          ).join('')}
        </select>
      </div>
    </div>
    <div class="chart-container" style="height:340px;">
      <canvas id="forecastChart"></canvas>
    </div>
  `;

  document.getElementById('forecast-zone-select').addEventListener('change', (e) => {
    loadSeries(e.target.value, currentLimit);
  });
  document.getElementById('forecast-limit-select').addEventListener('change', (e) => {
    loadSeries(currentZone, parseInt(e.target.value, 10));
  });

  try {
    drawChart(data);
  } catch (err) {
    chartContainer.querySelector('.chart-container').insertAdjacentHTML(
      'beforeend',
      `<div class="error-state"><div class="error-icon">⚠️</div><strong>Chart failed to render</strong><p>${err.message}</p></div>`
    );
  }
}

function drawChart(data) {
  const ctx = document.getElementById('forecastChart').getContext('2d');
  const labels = data.series.map((p) => p.timestamp);
  const actual = data.series.map((p) => p.actual_kw);
  const predicted = data.series.map((p) => p.predicted_kw);
  const upper = data.series.map((p) => p.upper_bound_kw);
  const lower = data.series.map((p) => p.lower_bound_kw);

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Upper 95% CI',
          data: upper,
          borderWidth: 0,
          pointRadius: 0,
          fill: '+1',
          backgroundColor: 'rgba(56,189,248,0.08)',
          order: 3,
        },
        {
          label: 'Lower 95% CI',
          data: lower,
          borderWidth: 0,
          pointRadius: 0,
          fill: false,
          order: 3,
        },
        {
          label: 'Actual',
          data: actual,
          borderColor: '#38bdf8',
          backgroundColor: '#38bdf8',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
          spanGaps: true,
          order: 1,
        },
        {
          label: 'Predicted',
          data: predicted,
          borderColor: '#fbbf24',
          backgroundColor: '#fbbf24',
          borderWidth: 2,
          borderDash: [6, 4],
          pointRadius: 0,
          tension: 0.3,
          order: 2,
        },
      ],
    },
    options: chartOptions(),
  });
}

function appendForecastToChart(result) {
  if (!chart) return;
  const labels = result.series.map((p) => p.timestamp);
  const predicted = result.series.map((p) => p.predicted_kw);
  const upper = result.series.map((p) => p.upper_bound_kw);
  const lower = result.series.map((p) => p.lower_bound_kw);

  chart.data.labels.push(...labels);
  chart.data.datasets[0].data.push(...upper);
  chart.data.datasets[1].data.push(...lower);
  chart.data.datasets[2].data.push(...labels.map(() => null)); // actual unknown for the future
  chart.data.datasets[3].data.push(...predicted);
  chart.update();
}

function chartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { labels: { color: '#94a3b8', boxWidth: 12, filter: (item) => !item.text.includes('CI') } },
      tooltip: {
        backgroundColor: '#111e36',
        borderColor: 'rgba(99,179,237,0.28)',
        borderWidth: 1,
        callbacks: { label: tooltipLabel },
      },
    },
    scales: {
      x: {
        ticks: { color: '#475569', maxTicksLimit: 8, autoSkip: true },
        grid: { color: 'rgba(99,179,237,0.08)' },
      },
      y: {
        ticks: { color: '#475569', callback: (v) => `${Number(v).toLocaleString()} kW` },
        grid: { color: 'rgba(99,179,237,0.08)' },
      },
    },
  };
}

function tooltipLabel(ctx) {
  const label = ctx.dataset.label || '';
  if (label.includes('CI')) return null;
  const value = ctx.parsed.y;
  if (value === null || value === undefined) return null;

  if (label === 'Predicted') {
    const actual = ctx.chart.data.datasets[2].data[ctx.dataIndex];
    if (typeof actual === 'number' && actual !== 0) {
      const pctErr = (((value - actual) / actual) * 100).toFixed(2);
      return `${label}: ${fmtKw(value)} (Δ ${pctErr}%)`;
    }
  }
  return `${label}: ${fmtKw(value)}`;
}

/* -------------------------------------------------------------------- */
/* KPI cards                                                              */
/* -------------------------------------------------------------------- */

function renderKPIs(summary) {
  const accuracy = buildAccuracyKpi(summary);

  kpiContainer.innerHTML = `
    ${kpiCard('📊', 'Avg Actual', 'kpi-avg-actual', 'kW')}
    ${kpiCard('🔮', 'Avg Predicted', 'kpi-avg-predicted', 'kW')}
    ${kpiCard('⚡', 'Peak Predicted', 'kpi-peak-predicted', 'kW')}
    ${kpiCard(accuracy.icon, accuracy.label, 'kpi-accuracy', accuracy.unit)}
  `;

  animateOrDash('kpi-avg-actual', summary.mean_actual_kw);
  animateOrDash('kpi-avg-predicted', summary.mean_predicted_kw);
  animateOrDash('kpi-peak-predicted', summary.peak_predicted_kw);
  animateOrDash('kpi-accuracy', accuracy.value, accuracy.decimals);
}

function buildAccuracyKpi(summary) {
  if (summary.mape_pct !== null && summary.mape_pct !== undefined) {
    return { icon: '🎯', label: 'Model Accuracy (MAPE)', value: summary.mape_pct, unit: '%', decimals: 2 };
  }
  if (summary.mae_kw !== null && summary.mae_kw !== undefined) {
    return { icon: '🎯', label: 'Model Accuracy (MAE)', value: summary.mae_kw, unit: 'kW', decimals: 0 };
  }
  return { icon: '🎯', label: 'Model Accuracy', value: null, unit: '', decimals: 0 };
}

function kpiCard(icon, label, valueId, unit) {
  return `
    <div class="kpi-card">
      <div class="kpi-icon">${icon}</div>
      <div class="kpi-label">${label}</div>
      <div class="kpi-value"><span id="${valueId}">—</span>${unit ? `<span class="kpi-unit">${unit}</span>` : ''}</div>
    </div>
  `;
}

function animateOrDash(id, value, decimals = 0) {
  const el = document.getElementById(id);
  if (!el) return;
  if (value === null || value === undefined) {
    el.textContent = '—';
    return;
  }
  animateCounter(el, value, 1000, decimals);
}

/* -------------------------------------------------------------------- */
/* Forecast Horizon panel                                                 */
/* -------------------------------------------------------------------- */

function buildPredictForm() {
  predictContainer.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">🧮 Forecast Horizon</div>
          <div class="card-subtitle">Project future demand for the selected zone using expected weather inputs</div>
        </div>
      </div>
      <form id="horizon-form">
        <div class="form-row mb-4">
          <div class="form-group">
            <label class="form-label" for="horizon-steps">Steps ahead (×10 min)</label>
            <input class="form-input" type="number" id="horizon-steps" min="1" max="1008" value="18" />
          </div>
          <div class="form-group">
            <label class="form-label" for="horizon-temp">Temperature (°C)</label>
            <input class="form-input" type="number" id="horizon-temp" step="0.1" placeholder="optional" />
          </div>
          <div class="form-group">
            <label class="form-label" for="horizon-humidity">Humidity (%)</label>
            <input class="form-input" type="number" id="horizon-humidity" step="0.1" placeholder="optional" />
          </div>
          <div class="form-group">
            <label class="form-label" for="horizon-wind">Wind (m/s)</label>
            <input class="form-input" type="number" id="horizon-wind" step="0.1" placeholder="optional" />
          </div>
        </div>
        <button type="submit" class="btn btn-primary" id="horizon-submit">Run Forecast</button>
        <span id="horizon-status" class="text-secondary" style="margin-left: var(--space-3); font-size: var(--text-xs);"></span>
      </form>
    </div>
  `;

  document.getElementById('horizon-form').addEventListener('submit', (e) => {
    e.preventDefault();
    runHorizonForecast();
  });
}

async function runHorizonForecast() {
  const btn = document.getElementById('horizon-submit');
  const statusEl = document.getElementById('horizon-status');
  const payload = {
    zone: currentZone,
    horizon_steps: parseInt(document.getElementById('horizon-steps').value, 10) || 18,
    temperature_c: parseOptionalFloat('horizon-temp'),
    humidity_pct: parseOptionalFloat('horizon-humidity'),
    wind_speed_ms: parseOptionalFloat('horizon-wind'),
  };

  btn.disabled = true;
  statusEl.className = 'text-secondary';
  statusEl.textContent = 'Running…';

  try {
    const result = await fetchForecastingPredict(payload);
    appendForecastToChart(result);
    statusEl.className = 'text-success';
    statusEl.textContent = `Added ${result.total_records} future points (peak ${fmtKw(result.summary.peak_predicted_kw)}).`;
  } catch (err) {
    statusEl.className = 'text-danger';
    statusEl.textContent = apiErrorMessage(err);
  } finally {
    btn.disabled = false;
  }
}

function parseOptionalFloat(id) {
  const v = document.getElementById(id).value;
  return v === '' ? undefined : parseFloat(v);
}

/* -------------------------------------------------------------------- */
/* Small helpers                                                          */
/* -------------------------------------------------------------------- */

function fmtKw(v) {
  if (v === null || v === undefined) return '—';
  return `${Number(v).toLocaleString(undefined, { maximumFractionDigits: 1 })} kW`;
}

function apiErrorMessage(err) {
  if (err instanceof ApiError) return err.message;
  return err?.message || 'Failed to load forecasting data.';
}
