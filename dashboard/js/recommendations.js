/**
 * recommendations.js — Member 3
 *
 * Builds the Recommendations section:
 *  - Recommendation card grid with priority badges
 *  - Animated savings KPI bar
 *  - Action plan form → POST /api/v1/optimization/action-plan
 */

import { fetchRecommendations, fetchActionPlan, ApiError } from "./api.js";
import { showLoader, showError, animateCounter } from "./app.js";

const savingsContainer = document.getElementById("savings-kpis");
const recommendationsContainer = document.getElementById("recommendations-container");
const actionPlanContainer = document.getElementById("action-plan-container");

let initialized = false;
let recommendations = [];

window.addEventListener("app:navigate", (e) => {
  if (e.detail.sectionId === "recommendations" && !initialized) {
    initialized = true;
    init();
  }
});

window.addEventListener("api:poll", () => {
  if (initialized) loadRecommendations({ silent: true });
});

if (document.getElementById("section-recommendations")?.classList.contains("active")) {
  initialized = true;
  init();
}

function init() {
  loadRecommendations();
  renderActionPlanForm();
}

async function loadRecommendations({ silent = false } = {}) {
  if (!silent) {
    showLoader(recommendationsContainer, "Loading recommendations...");
  }

  try {
    const data = await fetchRecommendations();
    recommendations = data.recommendations || [];
    renderSavings(data);
    renderRecommendations(data);
  } catch (err) {
    const message = apiErrorMessage(err);
    showError(recommendationsContainer, message);
    showError(savingsContainer, message);
  }
}

function renderSavings(data) {
  savingsContainer.innerHTML = `
    ${kpiCard("Total Savings", "saving-kw", "kW", "Estimated demand reduction", "badge-success")}
    ${kpiCard("Cost Savings", "saving-cost", "%", "Best single recommendation", "badge-info")}
    ${kpiCard("CO2 Reduction", "saving-co2", "kg", "Estimated avoided emissions", "badge-warning")}
  `;

  animateCounter(document.getElementById("saving-kw"), data.total_potential_power_saving_kw, 1000, 0);
  animateCounter(document.getElementById("saving-cost"), data.max_cost_saving_pct, 1000, 1);
  animateCounter(document.getElementById("saving-co2"), data.total_potential_power_saving_kw * 0.62, 1000, 0);
}

function kpiCard(label, id, unit, delta, badgeClass) {
  return `
    <div class="kpi-card">
      <div class="kpi-icon"><span class="badge ${badgeClass}">REC</span></div>
      <div class="kpi-label">${label}</div>
      <div class="kpi-value"><span id="${id}">0</span>${unit ? `<span class="kpi-unit">${unit}</span>` : ""}</div>
      <div class="kpi-delta neutral">${delta}</div>
    </div>
  `;
}

function renderRecommendations(data) {
  recommendationsContainer.innerHTML = `
    <div class="card-header">
      <div>
        <div class="card-title">Energy Optimization Recommendations</div>
        <div class="card-subtitle">${data.total_recommendations} targeted actions ranked by impact</div>
      </div>
      <select id="recommendation-filter" class="form-select" style="max-width:220px;">
        <option value="">All clusters</option>
        ${[0, 1, 2, 3].map((id) => `<option value="${id}">Cluster ${id}</option>`).join("")}
      </select>
    </div>
    <div id="recommendation-grid" class="content-grid">
      ${recommendations.map(recommendationCard).join("")}
    </div>
  `;

  document.getElementById("recommendation-filter").addEventListener("change", (event) => {
    const clusterId = event.target.value;
    const filtered = clusterId === ""
      ? recommendations
      : recommendations.filter((rec) => String(rec.target_cluster_id) === clusterId);
    document.getElementById("recommendation-grid").innerHTML = filtered.map(recommendationCard).join("");
  });
}

function recommendationCard(rec) {
  return `
    <div class="card col-6" style="transition: transform var(--transition-fast), border-color var(--transition-fast);" onmouseenter="this.style.transform='translateY(-4px)'" onmouseleave="this.style.transform='translateY(0)'">
      <div class="feed-item-header">
        <span class="badge ${priorityBadge(rec.priority)}">${rec.priority}</span>
        <span class="badge badge-neutral">Cluster ${rec.target_cluster_id}</span>
      </div>
      <div class="card-title" style="font-size:var(--text-base);">${rec.title}</div>
      <div class="card-subtitle">${rec.category.replaceAll("_", " ")} · ${rec.target_cluster_name}</div>
      <p class="feed-item-body mt-4">${rec.action_summary}</p>
      <div class="content-grid mt-4">
        <div class="col-6">
          <div class="kpi-label">Power Saving</div>
          <div class="kpi-value" style="font-size:var(--text-xl);">${number(rec.estimated_power_saving_kw)}<span class="kpi-unit">kW</span></div>
        </div>
        <div class="col-6">
          <div class="kpi-label">Cost Saving</div>
          <div class="kpi-value" style="font-size:var(--text-xl);">${rec.estimated_cost_saving_pct.toFixed(1)}<span class="kpi-unit">%</span></div>
        </div>
      </div>
      <div class="divider"></div>
      ${(rec.implementation_steps || []).map((step, index) => `
        <div class="feed-item-body">${index + 1}. ${step}</div>
      `).join("")}
    </div>
  `;
}

function renderActionPlanForm() {
  actionPlanContainer.innerHTML = `
    <div class="card-header">
      <div>
        <div class="card-title">Operator Action Plan</div>
        <div class="card-subtitle">Generate immediate actions from current zone load</div>
      </div>
    </div>
    <form id="action-plan-form">
      <div class="form-row mb-4">
        <div class="form-group">
          <label class="form-label" for="plan-zone-1">Zone 1 kW</label>
          <input class="form-input" id="plan-zone-1" type="number" value="38000" required />
        </div>
        <div class="form-group">
          <label class="form-label" for="plan-zone-2">Zone 2 kW</label>
          <input class="form-input" id="plan-zone-2" type="number" value="26000" required />
        </div>
        <div class="form-group">
          <label class="form-label" for="plan-zone-3">Zone 3 kW</label>
          <input class="form-input" id="plan-zone-3" type="number" value="28000" required />
        </div>
        <div class="form-group">
          <label class="form-label" for="plan-temp">Ambient Temp C</label>
          <input class="form-input" id="plan-temp" type="number" step="0.1" value="29" required />
        </div>
        <div class="form-group">
          <label class="form-label" for="plan-load-shift">Load shifting</label>
          <select class="form-select" id="plan-load-shift">
            <option value="true">Allowed</option>
            <option value="false">Disabled</option>
          </select>
        </div>
      </div>
      <button type="submit" class="btn btn-primary" id="action-plan-btn">Generate Action Plan</button>
      <span id="action-plan-status" class="text-secondary" style="margin-left:var(--space-3); font-size:var(--text-xs);"></span>
    </form>
    <div id="action-plan-result" class="mt-6"></div>
  `;

  document.getElementById("action-plan-form").addEventListener("submit", generateActionPlan);
}

async function generateActionPlan(event) {
  event.preventDefault();

  const btn = document.getElementById("action-plan-btn");
  const statusEl = document.getElementById("action-plan-status");
  const resultEl = document.getElementById("action-plan-result");
  const payload = {
    current_zone_1_kw: floatValue("plan-zone-1"),
    current_zone_2_kw: floatValue("plan-zone-2"),
    current_zone_3_kw: floatValue("plan-zone-3"),
    ambient_temp_c: floatValue("plan-temp"),
    allow_load_shifting: document.getElementById("plan-load-shift").value === "true",
  };

  btn.disabled = true;
  statusEl.className = "text-secondary";
  statusEl.textContent = "Generating...";

  try {
    const plan = await fetchActionPlan(payload);
    statusEl.className = "text-success";
    statusEl.textContent = "Action plan ready.";
    resultEl.innerHTML = actionPlanResult(plan);
  } catch (err) {
    statusEl.className = "text-danger";
    statusEl.textContent = apiErrorMessage(err);
  } finally {
    btn.disabled = false;
  }
}

function actionPlanResult(plan) {
  return `
    <div class="content-grid">
      <div class="kpi-card col-3">
        <div class="kpi-label">Current Load</div>
        <div class="kpi-value">${number(plan.current_total_kw)}<span class="kpi-unit">kW</span></div>
      </div>
      <div class="kpi-card col-3">
        <div class="kpi-label">Target Load</div>
        <div class="kpi-value">${number(plan.optimized_target_kw)}<span class="kpi-unit">kW</span></div>
      </div>
      <div class="kpi-card col-3">
        <div class="kpi-label">Reduction</div>
        <div class="kpi-value">${number(plan.projected_power_reduction_kw)}<span class="kpi-unit">kW</span></div>
      </div>
      <div class="kpi-card col-3">
        <div class="kpi-label">Bill Reduction</div>
        <div class="kpi-value">${plan.projected_bill_reduction_pct.toFixed(1)}<span class="kpi-unit">%</span></div>
      </div>
    </div>
    <div class="divider"></div>
    <div class="card-title" style="font-size:var(--text-base);">Action Plan Timeline</div>
    <div class="mt-4" style="position:relative; padding-left:28px;">
      <div style="position:absolute; left:9px; top:8px; bottom:8px; width:2px; background:var(--color-accent); opacity:0.65;"></div>
      ${plan.immediate_actions.map((action, index) => `
        <div class="feed-item" style="cursor:default; position:relative;">
          <div style="position:absolute; left:-28px; top:18px; width:20px; height:20px; border-radius:50%; background:var(--color-accent); color:#06111f; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:800;">${index + 1}</div>
          <div class="feed-item-header">
            <span class="badge badge-info">Step ${index + 1}</span>
            <span class="text-success" style="font-size:var(--text-xs);">Impact: ${number(plan.projected_power_reduction_kw / plan.immediate_actions.length)} kW</span>
          </div>
          <div class="feed-item-body">${action}</div>
        </div>
      `).join("")}
    </div>
  `;
}

function priorityBadge(priority) {
  if (priority === "CRITICAL" || priority === "HIGH") return "badge-danger";
  if (priority === "MEDIUM") return "badge-warning";
  return "badge-success";
}

function floatValue(id) {
  return parseFloat(document.getElementById(id).value) || 0;
}

function number(value) {
  return Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 1 });
}

function apiErrorMessage(err) {
  if (err instanceof ApiError) return err.message;
  return err?.message || "Failed to load recommendations.";
}
