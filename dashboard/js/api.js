/**
 * api.js — Central API Client
 * Member 1: Foundation Layer
 *
 * All dashboard JS modules import fetch helpers from here.
 * Handles errors, timeouts, and auto-refresh polling.
 */

const BASE_URL = "http://localhost:8000/api/v1";

// ─── Timeout helper ───────────────────────────────────────────
const DEFAULT_TIMEOUT_MS = 12000;

async function fetchWithTimeout(url, options = {}, timeout = DEFAULT_TIMEOUT_MS) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(id);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ message: res.statusText }));
      throw new ApiError(res.status, err.detail || err.message || res.statusText, url);
    }
    return await res.json();
  } catch (e) {
    clearTimeout(id);
    if (e instanceof ApiError) throw e;
    if (e.name === "AbortError") throw new ApiError(408, "Request timed out", url);
    throw new ApiError(0, e.message || "Network error — is the API running?", url);
  }
}

// ─── Custom Error class ──────────────────────────────────────
export class ApiError extends Error {
  constructor(status, message, endpoint) {
    super(message);
    this.name   = "ApiError";
    this.status = status;
    this.endpoint = endpoint;
  }
}

// ─── Event bus for status updates ────────────────────────────
// Other modules can listen: window.addEventListener('api:status', e => ...)
function emitStatus(online, detail = null) {
  window.dispatchEvent(new CustomEvent("api:status", { detail: { online, detail } }));
}

// ─────────────────────────────────────────────────────────────
//  FORECASTING
//  GET  /api/v1/forecasting/actual-vs-predicted
//  POST /api/v1/forecasting/predict
// ─────────────────────────────────────────────────────────────

/**
 * Fetch historical actual-vs-predicted energy series.
 * @param {object} params
 * @param {string}  [params.zone="Total"]        Zone 1 | Zone 2 | Zone 3 | Total
 * @param {number}  [params.limit=144]            Number of 10-min intervals (144 = 24h)
 * @param {string}  [params.start_date]           YYYY-MM-DD HH:MM:SS
 * @param {string}  [params.end_date]             YYYY-MM-DD HH:MM:SS
 * @returns {Promise<ForecastingResponse>}
 */
export async function fetchForecasting({ zone = "Total", limit = 144, start_date, end_date } = {}) {
  const qs = new URLSearchParams({ zone, limit });
  if (start_date) qs.set("start_date", start_date);
  if (end_date)   qs.set("end_date",   end_date);
  return fetchWithTimeout(`${BASE_URL}/forecasting/actual-vs-predicted?${qs}`);
}

/**
 * Generate a forward multi-step forecast.
 * @param {object} body
 * @param {string}  [body.zone="Total"]
 * @param {number}  [body.horizon_steps=144]      1 step = 10 min; 144 = 24 h
 * @param {number}  [body.temperature_c]
 * @param {number}  [body.humidity_pct]
 * @param {number}  [body.wind_speed_ms]
 * @returns {Promise<ForecastingResponse>}
 */
export async function fetchForecastingPredict(body = {}) {
  return fetchWithTimeout(`${BASE_URL}/forecasting/predict`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ zone: "Total", horizon_steps: 144, ...body }),
  });
}

// ─────────────────────────────────────────────────────────────
//  ANOMALIES
//  GET  /api/v1/anomalies
//  GET  /api/v1/anomalies/summary
//  POST /api/v1/anomalies/detect
// ─────────────────────────────────────────────────────────────

/**
 * List historical flagged anomaly events.
 * @param {object} params
 * @param {string}  [params.zone]       Zone 1 | Zone 2 | Zone 3
 * @param {string}  [params.severity]   NORMAL | LOW | MEDIUM | HIGH | CRITICAL
 * @param {string}  [params.start_date]
 * @param {string}  [params.end_date]
 * @param {number}  [params.limit=50]
 * @returns {Promise<AnomalyListResponse>}
 */
export async function fetchAnomalies({ zone, severity, start_date, end_date, limit = 50 } = {}) {
  const qs = new URLSearchParams({ limit });
  if (zone)       qs.set("zone",       zone);
  if (severity)   qs.set("severity",   severity);
  if (start_date) qs.set("start_date", start_date);
  if (end_date)   qs.set("end_date",   end_date);
  return fetchWithTimeout(`${BASE_URL}/anomalies?${qs}`);
}

/**
 * Get aggregate anomaly detection statistics.
 * @returns {Promise<AnomalySummaryResponse>}
 */
export async function fetchAnomalySummary() {
  return fetchWithTimeout(`${BASE_URL}/anomalies/summary`);
}

/**
 * Run on-demand anomaly detection for a single reading.
 * @param {object} body
 * @param {string}  body.zone           Zone 1 | Zone 2 | Zone 3
 * @param {number}  body.value_kw
 * @param {string}  [body.timestamp]
 * @param {number}  [body.rolling_24h_mean_kw]
 * @param {number}  [body.rolling_24h_std_kw]
 * @param {number}  [body.temperature_c]
 * @returns {Promise<AnomalyDetectResponse>}
 */
export async function fetchAnomalyDetect(body) {
  return fetchWithTimeout(`${BASE_URL}/anomalies/detect`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  });
}

// ─────────────────────────────────────────────────────────────
//  PEAK DEMAND
//  GET  /api/v1/peak-demand/upcoming-peaks
//  POST /api/v1/peak-demand/predict
// ─────────────────────────────────────────────────────────────

/**
 * Get upcoming predicted peak windows.
 * @param {object} params
 * @param {number}  [params.hours_ahead=24]    1–168 hours
 * @param {number}  [params.threshold_kw]      Custom threshold (default: system value)
 * @returns {Promise<UpcomingPeaksResponse>}
 */
export async function fetchPeaks({ hours_ahead = 24, threshold_kw } = {}) {
  const qs = new URLSearchParams({ hours_ahead });
  if (threshold_kw !== undefined) qs.set("threshold_kw", threshold_kw);
  return fetchWithTimeout(`${BASE_URL}/peak-demand/upcoming-peaks?${qs}`);
}

/**
 * Predict next-interval peak probability.
 * @param {object} body
 * @param {number}  body.total_power_kw
 * @param {number}  [body.lag_10m_kw]
 * @param {number}  [body.lag_1h_kw]
 * @param {number}  [body.rolling_24h_mean_kw]
 * @param {number}  [body.temperature_c=28]
 * @param {boolean} [body.is_weekend=false]
 * @param {string}  [body.timestamp]
 * @returns {Promise<PeakPredictResponse>}
 */
export async function fetchPeakPredict(body) {
  return fetchWithTimeout(`${BASE_URL}/peak-demand/predict`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  });
}

// ─────────────────────────────────────────────────────────────
//  CLUSTERING
//  GET  /api/v1/clustering/groups
//  GET  /api/v1/clustering/distribution
//  POST /api/v1/clustering/classify
// ─────────────────────────────────────────────────────────────

/**
 * Get all consumption cluster profiles.
 * @returns {Promise<ClusterListResponse>}
 */
export async function fetchClusters() {
  return fetchWithTimeout(`${BASE_URL}/clustering/groups`);
}

/**
 * Get cluster distribution breakdown.
 * @returns {Promise<ClusterDistributionResponse>}
 */
export async function fetchClusterDistribution() {
  return fetchWithTimeout(`${BASE_URL}/clustering/distribution`);
}

/**
 * Classify a meter reading into a cluster.
 * @param {object} body
 * @param {number}  body.zone_1_kw
 * @param {number}  body.zone_2_kw
 * @param {number}  body.zone_3_kw
 * @param {number}  [body.temperature_c=25]
 * @param {number}  [body.humidity_pct=60]
 * @param {boolean} [body.is_peak_hour=false]
 * @returns {Promise<ClusterClassifyResponse>}
 */
export async function fetchClusterClassify(body) {
  return fetchWithTimeout(`${BASE_URL}/clustering/classify`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  });
}

// ─────────────────────────────────────────────────────────────
//  OPTIMIZATION RECOMMENDATIONS
//  GET  /api/v1/optimization/recommendations
//  POST /api/v1/optimization/action-plan
// ─────────────────────────────────────────────────────────────

/**
 * Get energy optimization recommendations.
 * @param {object} params
 * @param {number}  [params.cluster_id]      0–3
 * @param {number}  [params.min_savings_pct=0]
 * @returns {Promise<RecommendationsResponse>}
 */
export async function fetchRecommendations({ cluster_id, min_savings_pct = 0 } = {}) {
  const qs = new URLSearchParams({ min_savings_pct });
  if (cluster_id !== undefined) qs.set("cluster_id", cluster_id);
  return fetchWithTimeout(`${BASE_URL}/optimization/recommendations?${qs}`);
}

/**
 * Generate a custom operator action plan.
 * @param {object} body
 * @param {number}  body.current_zone_1_kw
 * @param {number}  body.current_zone_2_kw
 * @param {number}  body.current_zone_3_kw
 * @param {number}  body.ambient_temp_c
 * @param {boolean} [body.allow_load_shifting=true]
 * @returns {Promise<ActionPlanResponse>}
 */
export async function fetchActionPlan(body) {
  return fetchWithTimeout(`${BASE_URL}/optimization/action-plan`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ allow_load_shifting: true, ...body }),
  });
}

// ─────────────────────────────────────────────────────────────
//  SYSTEM STATUS
//  GET  /api/v1/system/status
//  POST /api/v1/system/refresh
// ─────────────────────────────────────────────────────────────

/**
 * Get system health and model status.
 * @returns {Promise<SystemStatusResponse>}
 */
export async function fetchSystemStatus() {
  try {
    const data = await fetchWithTimeout(`${BASE_URL}/system/status`);
    emitStatus(true, data);
    return data;
  } catch (err) {
    emitStatus(false, err);
    throw err;
  }
}

/**
 * Trigger an on-demand model refresh.
 * @returns {Promise<RefreshSystemResponse>}
 */
export async function triggerSystemRefresh() {
  return fetchWithTimeout(`${BASE_URL}/system/refresh`, { method: "POST" });
}

// ─────────────────────────────────────────────────────────────
//  DATA INGESTION
//  POST /api/v1/ingest/readings
// ─────────────────────────────────────────────────────────────

/**
 * Ingest smart meter readings.
 * @param {Array<object>} readings  Array of meter reading objects
 * @returns {Promise<ReadingIngestResponse>}
 */
export async function ingestReadings(readings) {
  return fetchWithTimeout(`${BASE_URL}/ingest/readings`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ readings }),
  });
}

// ─────────────────────────────────────────────────────────────
//  AUTO-REFRESH POLLING
// ─────────────────────────────────────────────────────────────

const POLL_INTERVAL_MS = 60_000; // 60 seconds
let _pollTimer = null;

/**
 * Start auto-refresh polling.
 * Emits 'api:poll' event on each cycle so sections can re-fetch their data.
 */
export function startPolling() {
  if (_pollTimer) return;
  _pollTimer = setInterval(() => {
    window.dispatchEvent(new CustomEvent("api:poll"));
    fetchSystemStatus().catch(() => {});
  }, POLL_INTERVAL_MS);
}

/** Stop auto-refresh polling. */
export function stopPolling() {
  if (_pollTimer) {
    clearInterval(_pollTimer);
    _pollTimer = null;
  }
}
