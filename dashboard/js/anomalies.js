/**
 * anomalies.js — Member 2
 *
 * Builds the Anomalies & Peak Demand section:
 *  - Time-series chart with red anomaly markers (Chart.js)
 *  - Scrollable anomaly event feed with badge-danger cards
 *  - Upcoming peak demand alert cards (badge-warning)
 *  - "Predict Peak" button → POST /api/v1/peak-demand/predict
 *
 * Imports available from api.js:
 *   fetchAnomalies, fetchAnomalySummary, fetchAnomalyDetect,
 *   fetchPeaks, fetchPeakPredict
 *
 * Imports available from app.js:
 *   showLoader, showError, animateCounter
 *
 * Containers (already in index.html):
 *   #anomalies-kpis            ← KPI row
 *   #anomaly-chart-container   ← Chart canvas
 *   #anomaly-feed-container    ← Scrollable event feed
 *   #peak-demand-container     ← Peak alert cards
 *   #anomaly-count-badge       ← Sidebar badge (set innerHTML to count)
 */


import { fetchForecasting } from './api.js';
import {
  fetchAnomalies,
  fetchAnomalySummary,
  fetchPeaks,
  fetchPeakPredict,
  ApiError,
} from './api.js';
import { showLoader, showError, animateCounter } from './app.js';

const ZONES = ['Zone 1', 'Zone 2', 'Zone 3'];
const SEVERITIES = ['', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];
const SEVERITY_BADGE = { CRITICAL: 'badge-critical', HIGH: 'badge-danger', MEDIUM: 'badge-warning', LOW: 'badge-info' };

const kpiContainer = document.getElementById('anomalies-kpis');
const chartContainer = document.getElementById('anomaly-chart-container');
const feedContainer = document.getElementById('anomaly-feed-container');
const peakContainer = document.getElementById('peak-demand-container');
const anomalyBadge = document.getElementById('anomaly-count-badge');

let chart = null;
let initialized = false;
let currentZone = 'Zone 1';
let currentSeverity = '';

/* -------------------------------------------------------------------- */
/* Lifecycle wiring                                                       */
/* -------------------------------------------------------------------- */

window.addEventListener('app:navigate', (e) => {
  if (e.detail.sectionId === 'anomalies' && !initialized) {
    initialized = true;
    init();
  }
});

window.addEventListener('api:poll', () => {
  if (initialized) {
    loadAnomalies(currentZone, currentSeverity, { silent: true });
    loadPeaks({ silent: true });
  }
});

function init() {
  loadAnomalies(currentZone, currentSeverity);
  loadPeaks();
  refreshBadgeFromSummary();
}

/* -------------------------------------------------------------------- */
/* Anomaly chart + feed                                                   */
/* -------------------------------------------------------------------- */

/**
 * The anomaly-detection dataset stores zone as "Zone_1"/"Zone_2"/"Zone_3"
 * (underscore) while the documented API filter and the rest of the system
 * use "Zone 1"/"Zone 2"/"Zone 3" (space). The server-side `zone` query
 * param does a strict match, so it silently returns 0 rows for any zone
 * filter today. Rather than trust it, we fetch unfiltered (only filtering
 * by severity, which IS reliable server-side) and match zone client-side
 * on a normalized string — this keeps working whichever format the
 * upstream data ends up using once that's fixed.
 * TODO: flag to the Anomaly Detection owner — anomaly_service.py's
 * `sub[sub["Zone"] == zone]` needs the same normalization, or the source
 * CSVs need their Zone column renamed to match the documented "Zone N" format.
 */
function normalizeZone(z) {
  return String(z || '').replace(/[_\s]+/g, ' ').trim();
}

async function loadAnomalies(zone, severity, { silent = false } = {}) {
  currentZone = zone;
  currentSeverity = severity;

  if (!silent) showLoader(chartContainer, 'Loading anomaly data…');

  try {
    const [seriesData, rawAnomalyData] = await Promise.all([
      fetchForecasting({ zone, limit: 288 }),
      fetchAnomalies({ severity: severity || undefined, limit: 500 }),
    ]);

    const wantedZone = normalizeZone(zone);
    const events = rawAnomalyData.events.filter((ev) => normalizeZone(ev.zone) === wantedZone);
    const anomalyData = {
      ...rawAnomalyData,
      total_anomalies: events.length,
      critical_count: events.filter((e) => e.severity === 'CRITICAL').length,
      high_count: events.filter((e) => e.severity === 'HIGH').length,
      medium_count: events.filter((e) => e.severity === 'MEDIUM').length,
      low_count: events.filter((e) => e.severity === 'LOW').length,
      events,
    };

    // KPIs and feed rendered before the chart, so a Chart.js load failure
    // can't hide them.
    renderKPIs(anomalyData);
    renderFeedCard(anomalyData);
    renderChartCard(seriesData, anomalyData.events);
  } catch (err) {
    showError(chartContainer, apiErrorMessage(err));
    showError(feedContainer, apiErrorMessage(err));
  }
}

function renderChartCard(seriesData, events) {
  chartContainer.innerHTML = `
    <div class="card-header">
      <div>
        <div class="card-title">⚡ Consumption &amp; Anomalies</div>
        <div class="card-subtitle">${seriesData.zone} · ${events.length} flagged in window</div>
      </div>
      <div class="flex gap-2">
        <select id="anomaly-zone-select" class="form-select">
          ${ZONES.map((z) => `<option value="${z}" ${z === currentZone ? 'selected' : ''}>${z}</option>`).join('')}
        </select>
        <select id="anomaly-severity-select" class="form-select">
          ${SEVERITIES.map((s) => `<option value="${s}" ${s === currentSeverity ? 'selected' : ''}>${s || 'All severities'}</option>`).join('')}
        </select>
      </div>
    </div>
    <div class="chart-container" style="height:320px; position:relative;">
      <canvas id="anomalyChart"></canvas>
      <div id="anomaly-popup" class="tooltip" style="display:none; bottom:auto; left:auto; transform:none; width:220px; z-index:20;"></div>
    </div>
  `;

  document.getElementById('anomaly-zone-select').addEventListener('change', (e) => {
    loadAnomalies(e.target.value, currentSeverity);
  });
  document.getElementById('anomaly-severity-select').addEventListener('change', (e) => {
    loadAnomalies(currentZone, e.target.value);
  });

  try {
    drawChart(seriesData, events);
  } catch (err) {
    chartContainer.querySelector('.chart-container').insertAdjacentHTML(
      'beforeend',
      `<div class="error-state"><div class="error-icon">⚠️</div><strong>Chart failed to render</strong><p>${err.message}</p></div>`
    );
  }
}

function drawChart(seriesData, events) {
  const ctx = document.getElementById('anomalyChart').getContext('2d');
  const labels = seriesData.series.map((p) => p.timestamp);
  const actual = seriesData.series.map((p, i) => ({ x: i, y: p.actual_kw }));

  const anomalyPoints = events
    .map((ev) => {
      const idx = labels.indexOf(ev.timestamp);
      return idx === -1 ? null : { x: idx, y: ev.observed_value_kw, event: ev };
    })
    .filter(Boolean);

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    data: {
      labels,
      datasets: [
        {
          type: 'line',
          label: 'Consumption',
          data: actual,
          borderColor: '#38bdf8',
          backgroundColor: '#38bdf8',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25,
          spanGaps: true,
          order: 1,
        },
        {
          type: 'scatter',
          label: 'Anomalies',
          data: anomalyPoints.map((p) => ({ x: p.x, y: p.y })),
          backgroundColor: '#f87171',
          borderColor: '#f87171',
          pointRadius: 3,
          pointHoverRadius: 5,
          showLine: false,
          order: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      onClick: (evt, elements) => handleChartClick(evt, elements, anomalyPoints),
      plugins: {
        legend: { labels: { color: '#94a3b8', boxWidth: 12 } },
        tooltip: { enabled: true },
      },
      scales: {
        x: {
          type: 'linear',
          ticks: { color: '#475569', maxTicksLimit: 8, callback: (v) => labels[v] || '' },
          grid: { color: 'rgba(99,179,237,0.08)' },
        },
        y: {
          ticks: { color: '#475569', callback: (v) => `${Number(v).toLocaleString()} kW` },
          grid: { color: 'rgba(99,179,237,0.08)' },
        },
      },
    },
  });
}

function handleChartClick(evt, elements, anomalyPoints) {
  const hit = elements.find((el) => el.datasetIndex === 1);
  if (!hit) {
    hidePopup();
    return;
  }
  const point = anomalyPoints[hit.index];
  if (point) showPopup(evt, point.event);
}

function showPopup(evt, ev) {
  const popup = document.getElementById('anomaly-popup');
  const badgeClass = SEVERITY_BADGE[ev.severity] || 'badge-neutral';
  popup.innerHTML = `
    <div class="flex justify-between items-center mb-4">
      <span class="badge ${badgeClass}">${ev.severity}</span>
      <span class="text-muted" style="cursor:pointer;" id="anomaly-popup-close">✕</span>
    </div>
    <div class="text-secondary" style="font-size:var(--text-xs); line-height:1.6;">
      <div><strong class="text-accent">${ev.timestamp}</strong></div>
      <div>Zone: ${ev.zone}</div>
      <div>Observed: ${fmtKw(ev.observed_value_kw)}</div>
      <div>Expected: ${fmtKw(ev.expected_baseline_kw)}</div>
      <div>Deviation: ${fmtKw(ev.deviation_kw)}</div>
      <div>Signals agreeing: ${ev.signals_agreeing}/3</div>
      <div class="mt-4" style="font-style:italic;">${ev.explanation}</div>
    </div>
  `;
  popup.style.display = 'block';

  const wrapRect = popup.parentElement.getBoundingClientRect();
  const clientX = evt.native ? evt.native.clientX : evt.clientX;
  const clientY = evt.native ? evt.native.clientY : evt.clientY;
  popup.style.left = `${Math.min(clientX - wrapRect.left + 12, wrapRect.width - 232)}px`;
  popup.style.top = `${Math.max(clientY - wrapRect.top - 12, 8)}px`;

  document.getElementById('anomaly-popup-close').addEventListener('click', hidePopup);
}

function hidePopup() {
  const popup = document.getElementById('anomaly-popup');
  if (popup) popup.style.display = 'none';
}

/* -------------------------------------------------------------------- */
/* Anomaly alert feed                                                      */
/* -------------------------------------------------------------------- */

function renderFeedCard(anomalyData) {
  feedContainer.innerHTML = `
    <div class="card-header">
      <div class="card-title">🚨 Anomaly Feed</div>
      <span class="badge badge-danger">${anomalyData.total_anomalies}</span>
    </div>
    <div class="scrollable-feed" style="max-height:280px;">
      ${
        anomalyData.events.length
          ? anomalyData.events
              .slice()
              .reverse()
              .map(feedItem)
              .join('')
          : `<div class="text-muted" style="font-size:var(--text-sm);">No anomalies for this filter.</div>`
      }
    </div>
  `;
}

function feedItem(ev) {
  const badgeClass = SEVERITY_BADGE[ev.severity] || 'badge-neutral';
  return `
    <div class="feed-item">
      <div class="feed-item-header">
        <span class="badge ${badgeClass}">${ev.severity}</span>
        <span class="feed-item-time">${ev.timestamp}</span>
      </div>
      <div class="feed-item-title">${ev.zone} — ${fmtKw(ev.observed_value_kw)}</div>
      <div class="feed-item-body">${ev.explanation}</div>
    </div>
  `;
}

/* -------------------------------------------------------------------- */
/* KPI cards + sidebar badge                                              */
/* -------------------------------------------------------------------- */

async function renderKPIs(anomalyData) {
  let summary = null;
  try {
    summary = await fetchAnomalySummary();
  } catch (_) {
    /* summary is a bonus KPI — degrade gracefully if it fails */
  }

  kpiContainer.innerHTML = `
    ${kpiCard('🚨', 'Total Anomalies', 'kpi-total', '')}
    ${kpiCard('🔴', 'Critical', 'kpi-critical', '')}
    ${kpiCard('🟠', 'High', 'kpi-high', '')}
    ${kpiCard('📉', 'Anomaly Rate', 'kpi-rate', '%')}
  `;

  animateOrDash('kpi-total', anomalyData.total_anomalies);
  animateOrDash('kpi-critical', anomalyData.critical_count);
  animateOrDash('kpi-high', anomalyData.high_count);
  animateOrDash('kpi-rate', summary ? summary.anomaly_rate_pct : null, 2);

  updateSidebarBadge(anomalyData.critical_count + anomalyData.high_count);
}

async function refreshBadgeFromSummary() {
  try {
    const summary = await fetchAnomalySummary();
    updateSidebarBadge(summary.total_anomalies_detected);
  } catch (_) {
    /* non-critical */
  }
}

function updateSidebarBadge(count) {
  if (!anomalyBadge) return;
  if (count > 0) {
    anomalyBadge.textContent = count > 99 ? '99+' : String(count);
    anomalyBadge.style.display = 'inline-flex';
  } else {
    anomalyBadge.style.display = 'none';
  }
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
/* Peak demand panel                                                       */
/* -------------------------------------------------------------------- */

async function loadPeaks({ silent = false } = {}) {
  if (!silent && !peakContainer.querySelector('#peak-feed')) {
    showLoader(peakContainer, 'Loading peak demand data…');
  }
  try {
    const data = await fetchPeaks({ hours_ahead: 24 });
    renderPeakPanel(data);
  } catch (err) {
    showError(peakContainer, apiErrorMessage(err));
  }
}

function renderPeakPanel(data) {
  peakContainer.innerHTML = `
    <div class="card-header">
      <div>
        <div class="card-title">🔺 Peak Demand Alerts</div>
        <div class="card-subtitle">
          ${data.peaks_detected_count} window(s) in next ${data.hours_evaluated}h ·
          threshold ${fmtKw(data.threshold_kw)} ·
          highest probability ${(data.highest_probability * 100).toFixed(0)}%
        </div>
      </div>
    </div>

    <div class="content-grid mb-6" id="peak-feed">
      ${
        data.upcoming_windows.length
          ? data.upcoming_windows.map(peakWindowCard).join('')
          : `<div class="col-12 text-muted">No peak windows detected.</div>`
      }
    </div>

    <div class="divider"></div>

    <form id="peak-predict-form">
      <div class="form-row mb-4">
        <div class="form-group">
          <label class="form-label" for="peak-total-kw">Current total power (kW)</label>
          <input class="form-input" type="number" id="peak-total-kw" step="1" required />
        </div>
        <div class="form-group">
          <label class="form-label" for="peak-temp">Temperature (°C)</label>
          <input class="form-input" type="number" id="peak-temp" step="0.1" value="28" />
        </div>
        <div class="form-group">
          <label class="form-label" for="peak-weekend">Weekend?</label>
          <select class="form-select" id="peak-weekend">
            <option value="false">No</option>
            <option value="true">Yes</option>
          </select>
        </div>
      </div>
      <button type="submit" class="btn btn-primary" id="peak-predict-btn">Predict Peak</button>
      <span id="peak-predict-status" class="text-secondary" style="margin-left: var(--space-3); font-size: var(--text-xs);"></span>
    </form>
  `;

  document.getElementById('peak-predict-form').addEventListener('submit', (e) => {
    e.preventDefault();
    runPeakPredict();
  });
}

function peakWindowCard(win) {
  const badgeClass = SEVERITY_BADGE[win.risk_level] || 'badge-warning';
  return `
    <div class="col-6">
      <div class="feed-item" style="cursor:default;">
        <div class="feed-item-header">
          <span class="badge ${badgeClass}">${win.risk_level}</span>
          <span class="feed-item-time">${win.start_time} → ${win.end_time}</span>
        </div>
        <div class="feed-item-title">${fmtKw(win.expected_peak_kw)} · ${(win.peak_probability * 100).toFixed(0)}% probability</div>
        <div class="feed-item-body">
          <ul style="padding-left: var(--space-4); margin: var(--space-2) 0 0;">
            ${win.contributing_factors.map((f) => `<li>${f}</li>`).join('')}
          </ul>
        </div>
      </div>
    </div>
  `;
}

async function runPeakPredict() {
  const btn = document.getElementById('peak-predict-btn');
  const statusEl = document.getElementById('peak-predict-status');
  const totalKw = parseFloat(document.getElementById('peak-total-kw').value);

  if (!totalKw) {
    statusEl.className = 'text-danger';
    statusEl.textContent = 'Enter current total power (kW) first.';
    return;
  }

  const payload = {
    total_power_kw: totalKw,
    temperature_c: parseFloat(document.getElementById('peak-temp').value) || 28.0,
    is_weekend: document.getElementById('peak-weekend').value === 'true',
  };

  btn.disabled = true;
  statusEl.className = 'text-secondary';
  statusEl.textContent = 'Predicting…';

  try {
    const result = await fetchPeakPredict(payload);
    prependPredictResult(result);
    statusEl.className = result.is_peak_warning ? 'text-danger' : 'text-success';
    statusEl.textContent = result.advisory_message;
  } catch (err) {
    statusEl.className = 'text-danger';
    statusEl.textContent = apiErrorMessage(err);
  } finally {
    btn.disabled = false;
  }
}

function prependPredictResult(result) {
  const feed = document.getElementById('peak-feed');
  if (!feed) return;
  const badgeClass = SEVERITY_BADGE[result.risk_level] || 'badge-warning';
  const card = document.createElement('div');
  card.className = 'col-6';
  card.innerHTML = `
    <div class="feed-item" style="cursor:default; border-color: var(--color-accent);">
      <div class="feed-item-header">
        <span class="badge ${badgeClass}">${result.risk_level}</span>
        <span class="feed-item-time">${result.timestamp} (on-demand)</span>
      </div>
      <div class="feed-item-title">${fmtKw(result.predicted_total_kw)} · ${(result.peak_probability * 100).toFixed(0)}% probability</div>
      <div class="feed-item-body">${result.advisory_message}</div>
    </div>
  `;
  feed.prepend(card);
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
  return err?.message || 'Failed to load data.';
}

