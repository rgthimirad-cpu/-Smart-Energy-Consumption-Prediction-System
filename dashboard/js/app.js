/**
 * app.js — SPA Navigation Controller & System Status
 * Member 1: Foundation Layer
 *
 * Controls:
 *  - Sidebar navigation (section switching)
 *  - System status pill in the header
 *  - Refresh button
 *  - Wires up auto-polling
 */

import {
  fetchSystemStatus,
  triggerSystemRefresh,
  startPolling,
} from "./api.js";

// ─────────────────────────────────────────────────────────────
//  DOM REFERENCES
// ─────────────────────────────────────────────────────────────
const navItems        = document.querySelectorAll(".nav-item[data-section]");
const sections        = document.querySelectorAll(".dashboard-section");
const statusPill      = document.getElementById("status-pill");
const statusText      = document.getElementById("status-text");
const statusDot       = document.getElementById("status-dot");
const lastRefreshEl   = document.getElementById("last-refresh");
const refreshBtn      = document.getElementById("btn-refresh");
const headerTitle     = document.getElementById("section-heading");

const SECTION_META = {
  forecasting: {
    label: "Forecasting",
    icon:  "📈",
    desc:  "Actual vs Predicted energy consumption and forward forecasts",
  },
  anomalies: {
    label: "Anomalies & Peaks",
    icon:  "⚡",
    desc:  "Anomaly detection events and upcoming peak demand alerts",
  },
  clustering: {
    label: "Clustering & Recommendations",
    icon:  "🔬",
    desc:  "Consumer cluster profiles and energy optimization recommendations",
  },
};

// ─────────────────────────────────────────────────────────────
//  NAVIGATION
// ─────────────────────────────────────────────────────────────
let currentSection = null;

function activateSection(sectionId) {
  if (currentSection === sectionId) return;
  currentSection = sectionId;

  // Update nav items
  navItems.forEach(item => {
    item.classList.toggle("active", item.dataset.section === sectionId);
  });

  // Update visible section
  sections.forEach(sec => {
    const isActive = sec.id === `section-${sectionId}`;
    sec.classList.toggle("active", isActive);
  });

  // Update header title
  const meta = SECTION_META[sectionId];
  if (headerTitle && meta) {
    headerTitle.textContent = `${meta.icon}  ${meta.label}`;
  }

  // Persist selection
  try { localStorage.setItem("ecs_active_section", sectionId); } catch {}

  // Emit navigation event so section modules can lazy-init
  window.dispatchEvent(new CustomEvent("app:navigate", { detail: { sectionId } }));
}

navItems.forEach(item => {
  item.addEventListener("click", () => activateSection(item.dataset.section));
});

// ─────────────────────────────────────────────────────────────
//  SYSTEM STATUS PILL
// ─────────────────────────────────────────────────────────────
function setStatus(online, details = null) {
  if (!statusPill) return;

  statusPill.className = "status-pill " + (online ? "online" : "offline");

  if (statusText) {
    if (online && details) {
      const uptime = details.uptime_seconds != null
        ? formatUptime(details.uptime_seconds)
        : "";
      statusText.textContent = `API Online${uptime ? " · " + uptime : ""}`;
    } else if (online) {
      statusText.textContent = "API Online";
    } else {
      statusText.textContent = "API Offline";
    }
  }
}

function setStatusLoading() {
  if (!statusPill) return;
  statusPill.className = "status-pill loading";
  if (statusText) statusText.textContent = "Connecting…";
}

function updateLastRefresh() {
  if (!lastRefreshEl) return;
  const now = new Date();
  lastRefreshEl.textContent = "Updated " + now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatUptime(seconds) {
  if (seconds < 60)   return `up ${Math.floor(seconds)}s`;
  if (seconds < 3600) return `up ${Math.floor(seconds / 60)}m`;
  return `up ${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

// ─────────────────────────────────────────────────────────────
//  API STATUS LISTENER (from api.js event bus)
// ─────────────────────────────────────────────────────────────
window.addEventListener("api:status", e => {
  const { online, detail } = e.detail;
  setStatus(online, detail);
  updateLastRefresh();
});

// ─────────────────────────────────────────────────────────────
//  REFRESH BUTTON
// ─────────────────────────────────────────────────────────────
if (refreshBtn) {
  refreshBtn.addEventListener("click", async () => {
    refreshBtn.disabled = true;
    refreshBtn.textContent = "⟳";
    refreshBtn.classList.add("spinning");
    try {
      await triggerSystemRefresh();
      await fetchSystemStatus();
      // Tell all sections to re-fetch their data
      window.dispatchEvent(new CustomEvent("api:poll"));
    } catch {
      /* error already handled by api.js status event */
    } finally {
      refreshBtn.disabled = false;
      refreshBtn.textContent = "⟳  Refresh";
      refreshBtn.classList.remove("spinning");
    }
  });
}

// ─────────────────────────────────────────────────────────────
//  INIT
// ─────────────────────────────────────────────────────────────
async function init() {
  // Restore last section or default to forecasting
  let initial = "forecasting";
  try {
    const saved = localStorage.getItem("ecs_active_section");
    if (saved && SECTION_META[saved]) initial = saved;
  } catch {}

  activateSection(initial);

  // Initial system status check
  setStatusLoading();
  fetchSystemStatus().catch(() => setStatus(false));

  // Start auto-refresh polling every 60 s
  startPolling();
}

document.addEventListener("DOMContentLoaded", init);

// ─────────────────────────────────────────────────────────────
//  GLOBAL SPINNER TOGGLE — utility for Members 2 & 3
// ─────────────────────────────────────────────────────────────

/**
 * Show a loading spinner inside a container.
 * @param {HTMLElement} container
 * @param {string} [message="Loading data…"]
 */
export function showLoader(container, message = "Loading data…") {
  container.innerHTML = `
    <div class="loader-overlay">
      <div class="spinner"></div>
      <span>${message}</span>
    </div>`;
}

/**
 * Show an error state inside a container.
 * @param {HTMLElement} container
 * @param {string} [message]
 */
export function showError(container, message = "Failed to load data. Is the API running?") {
  container.innerHTML = `
    <div class="error-state">
      <div class="error-icon">⚠️</div>
      <strong>Connection Error</strong>
      <p>${message}</p>
    </div>`;
}

/**
 * Animate a numeric counter from 0 to target value.
 * @param {HTMLElement} el
 * @param {number} target
 * @param {number} [duration=1200]   ms
 * @param {number} [decimals=0]
 */
export function animateCounter(el, target, duration = 1200, decimals = 0) {
  const start    = performance.now();
  const startVal = 0;
  const factor   = Math.pow(10, decimals);

  function step(now) {
    const elapsed  = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // Ease-out cubic
    const ease     = 1 - Math.pow(1 - progress, 3);
    const value    = startVal + (target - startVal) * ease;
    el.textContent = (Math.round(value * factor) / factor).toLocaleString(undefined, {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}
