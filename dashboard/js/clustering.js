/**
 * clustering.js — Member 3
 *
 * Builds the Clustering section:
 *  - Color-coded scatter plot per cluster
 *  - Cluster summary cards grid
 *  - "Classify My Meter" form → POST /api/v1/clustering/classify
 */

import { fetchClusters, fetchClusterDistribution, fetchClusterClassify, ApiError } from "./api.js";
import { showLoader, showError, animateCounter } from "./app.js";

const chartContainer = document.getElementById("cluster-chart-container");
const classifyContainer = document.getElementById("cluster-classify-container");
const cardsContainer = document.getElementById("cluster-cards-container");

let chart = null;
let initialized = false;
let clusters = [];

window.addEventListener("app:navigate", (e) => {
  if (e.detail.sectionId === "clustering" && !initialized) {
    initialized = true;
    init();
  }
});

window.addEventListener("api:poll", () => {
  if (initialized) init({ silent: true });
});

if (document.getElementById("section-clustering")?.classList.contains("active")) {
  initialized = true;
  init();
}

async function init({ silent = false } = {}) {
  if (!silent) {
    showLoader(chartContainer, "Loading cluster data...");
    showLoader(classifyContainer, "Loading classifier...");
  }

  try {
    const [clusterData, distributionData] = await Promise.all([
      fetchClusters(),
      fetchClusterDistribution(),
    ]);

    clusters = clusterData.clusters || [];
    renderChart(clusterData, distributionData);
    renderClusterCards(clusterData);
    renderClassifier();
  } catch (err) {
    const message = apiErrorMessage(err);
    showError(chartContainer, message);
    showError(classifyContainer, message);
    showError(cardsContainer, message);
  }
}

function renderChart(clusterData, distributionData) {
  const distribution = distributionData.distribution || [];
  const points = (clusterData.clusters || []).map((cluster) => ({
    x: cluster.statistics.temperature_mean_c,
    y: cluster.statistics.total_power_mean_kw,
    r: Math.max(8, Math.min(26, cluster.sample_percentage * 0.8)),
    cluster,
  }));

  chartContainer.innerHTML = `
    <div class="card-header">
      <div>
        <div class="card-title">Cluster Scatter Plot</div>
        <div class="card-subtitle">${number(clusterData.total_records_analyzed)} records represented by cluster centroids</div>
      </div>
      <span class="badge badge-info">${distribution.length} clusters</span>
    </div>
    <div class="chart-container" style="height:330px;">
      <canvas id="clusterChart"></canvas>
    </div>
  `;

  const ctx = document.getElementById("clusterChart").getContext("2d");
  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "bubble",
    data: {
      datasets: points.map((point) => ({
        label: `Cluster ${point.cluster.cluster_id}`,
        data: [{ x: point.x, y: point.y, r: point.r }],
        backgroundColor: hexToRgba(point.cluster.ui_color_theme, 0.62),
        borderColor: point.cluster.ui_color_theme,
        borderWidth: 2,
        pointHoverRadius: point.r + 3,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8", boxWidth: 12 } },
        tooltip: {
          backgroundColor: "#111e36",
          borderColor: "rgba(99,179,237,0.28)",
          borderWidth: 1,
          callbacks: {
            title: (items) => points[items[0].datasetIndex]?.cluster.cluster_name || "",
            label: (item) => {
              const cluster = points[item.datasetIndex].cluster;
              return [
                `Cluster ${cluster.cluster_id}: ${cluster.short_code}`,
                `Avg load: ${number(cluster.statistics.total_power_mean_kw)} kW`,
                `Temp: ${cluster.statistics.temperature_mean_c.toFixed(1)} C`,
                `Members: ${number(cluster.sample_count)} (${cluster.sample_percentage.toFixed(2)}%)`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Average temperature (C)", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(99,179,237,0.08)" },
        },
        y: {
          title: { display: true, text: "Average total consumption (kW)", color: "#94a3b8" },
          ticks: { color: "#94a3b8", callback: (v) => number(v) },
          grid: { color: "rgba(99,179,237,0.08)" },
        },
      },
    },
  });
}

function renderClusterCards(data) {
  cardsContainer.innerHTML = `
    <div class="content-grid">
      ${data.clusters.map(clusterCard).join("")}
    </div>
  `;
}

function clusterCard(cluster) {
  const stats = cluster.statistics;
  return `
    <div class="card col-3" id="cluster-card-${cluster.cluster_id}" style="border-top:2px solid ${cluster.ui_color_theme};">
      <div class="flex justify-between items-center mb-4">
        <span class="badge badge-neutral">Cluster ${cluster.cluster_id}</span>
        <span class="badge ${gradeBadge(cluster.efficiency_rating)}">${cluster.efficiency_rating}</span>
      </div>
      <div class="card-title" style="font-size:var(--text-base);">${cluster.cluster_name}</div>
      <div class="card-subtitle">${cluster.short_code}</div>
      <div class="divider"></div>
      <div class="kpi-value" style="font-size:var(--text-xl);">${number(stats.total_power_mean_kw)}<span class="kpi-unit">kW</span></div>
      <div class="kpi-delta neutral">${cluster.sample_percentage.toFixed(2)}% of records · ${cluster.dominant_time_window}</div>
      <p class="feed-item-body mt-4">${cluster.usage_behavior}</p>
      <div class="mt-4">
        ${(cluster.key_patterns || []).slice(0, 2).map((pattern) => `
          <div class="feed-item-body">• ${pattern}</div>
        `).join("")}
      </div>
    </div>
  `;
}

function renderClassifier() {
  classifyContainer.innerHTML = `
    <div class="card-header">
      <div>
        <div class="card-title">Classify My Meter</div>
        <div class="card-subtitle">Assign current zone readings to the closest cluster</div>
      </div>
    </div>
    <form id="cluster-classify-form">
      <div class="form-row mb-4">
        <div class="form-group">
          <label class="form-label" for="cluster-zone-1">Zone 1 kW</label>
          <input class="form-input" id="cluster-zone-1" type="number" value="35000" required />
        </div>
        <div class="form-group">
          <label class="form-label" for="cluster-zone-2">Zone 2 kW</label>
          <input class="form-input" id="cluster-zone-2" type="number" value="22000" required />
        </div>
        <div class="form-group">
          <label class="form-label" for="cluster-zone-3">Zone 3 kW</label>
          <input class="form-input" id="cluster-zone-3" type="number" value="24000" required />
        </div>
        <div class="form-group">
          <label class="form-label" for="cluster-temp">Temperature C</label>
          <input class="form-input" id="cluster-temp" type="number" step="0.1" value="27.5" />
        </div>
        <div class="form-group">
          <label class="form-label" for="cluster-humidity">Humidity %</label>
          <input class="form-input" id="cluster-humidity" type="number" step="0.1" value="58" />
        </div>
        <div class="form-group">
          <label class="form-label" for="cluster-peak">Peak hour?</label>
          <select class="form-select" id="cluster-peak">
            <option value="false">No</option>
            <option value="true">Yes</option>
          </select>
        </div>
      </div>
      <button type="submit" class="btn btn-primary" id="cluster-classify-btn">Classify Reading</button>
    </form>
    <div id="cluster-classify-result" class="mt-6"></div>
  `;

  document.getElementById("cluster-classify-form").addEventListener("submit", classifyReading);
}

async function classifyReading(event) {
  event.preventDefault();

  const btn = document.getElementById("cluster-classify-btn");
  const resultEl = document.getElementById("cluster-classify-result");
  const payload = {
    zone_1_kw: floatValue("cluster-zone-1"),
    zone_2_kw: floatValue("cluster-zone-2"),
    zone_3_kw: floatValue("cluster-zone-3"),
    temperature_c: floatValue("cluster-temp"),
    humidity_pct: floatValue("cluster-humidity"),
    is_peak_hour: document.getElementById("cluster-peak").value === "true",
  };

  btn.disabled = true;
  resultEl.innerHTML = `<div class="text-secondary" style="font-size:var(--text-sm);">Classifying...</div>`;

  try {
    const result = await fetchClusterClassify(payload);
    highlightCluster(result.assigned_cluster_id);
    resultEl.innerHTML = `
      <div class="feed-item" style="cursor:default; border-color:${result.ui_color_theme};">
        <div class="feed-item-header">
          <span class="badge badge-info">Cluster ${result.assigned_cluster_id}</span>
          <span class="badge ${gradeBadge(result.efficiency_rating)}">${result.efficiency_rating}</span>
        </div>
        <div class="feed-item-title">${result.cluster_name}</div>
        <div class="feed-item-body">${result.short_code} · centroid distance ${result.distance_to_centroid.toFixed(3)}</div>
        <div class="divider"></div>
        <div class="feed-item-body">${result.top_recommendation}</div>
      </div>
    `;
  } catch (err) {
    resultEl.innerHTML = `<div class="text-danger" style="font-size:var(--text-sm);">${apiErrorMessage(err)}</div>`;
  } finally {
    btn.disabled = false;
  }
}

function highlightCluster(clusterId) {
  if (chart) {
    chart.data.datasets.forEach((dataset) => {
      const isMatch = dataset.label === `Cluster ${clusterId}`;
      dataset.borderWidth = isMatch ? 5 : 1;
      dataset.backgroundColor = isMatch
        ? hexToRgba(dataset.borderColor, 0.9)
        : hexToRgba(dataset.borderColor, 0.22);
    });
    chart.update();
  }

  document.querySelectorAll("[id^='cluster-card-']").forEach((card) => {
    card.classList.remove("glow-ring");
  });

  const card = document.getElementById(`cluster-card-${clusterId}`);
  if (card) {
    card.classList.add("glow-ring");
    card.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function gradeBadge(grade) {
  if (String(grade).includes("A")) return "badge-success";
  if (String(grade).includes("B")) return "badge-info";
  if (String(grade).includes("C")) return "badge-warning";
  return "badge-danger";
}

function floatValue(id) {
  return parseFloat(document.getElementById(id).value) || 0;
}

function number(value) {
  return Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 1 });
}

function hexToRgba(hex, alpha) {
  const clean = String(hex || "#38bdf8").replace("#", "");
  const int = parseInt(clean, 16);
  const r = (int >> 16) & 255;
  const g = (int >> 8) & 255;
  const b = int & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function apiErrorMessage(err) {
  if (err instanceof ApiError) return err.message;
  return err?.message || "Failed to load clustering data.";
}
