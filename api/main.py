"""
Main FastAPI Application Entry Point for Smart Energy Consumption Prediction System API.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from api.config import API_DESCRIPTION, API_TITLE, API_VERSION
from api.routers import (
    anomalies_router,
    clustering_router,
    forecasting_router,
    ingest_router,
    optimization_router,
    peak_demand_router,
    system_router,
)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS for Dashboard Frontend Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(forecasting_router)
app.include_router(anomalies_router)
app.include_router(peak_demand_router)
app.include_router(clustering_router)
app.include_router(optimization_router)
app.include_router(ingest_router)
app.include_router(system_router)


# Global Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": str(exc),
            "path": request.url.path,
        },
    )


@app.get("/", include_in_schema=False)
def root_redirect():
    return {
        "system": API_TITLE,
        "version": API_VERSION,
        "status": "ONLINE",
        "documentation": "/docs",
        "redoc_documentation": "/redoc",
        "openapi_schema": "/openapi.json",
        "endpoints": {
            "forecasting_actual_vs_predicted": "/api/v1/forecasting/actual-vs-predicted",
            "forecasting_predict": "/api/v1/forecasting/predict",
            "anomalies_list": "/api/v1/anomalies",
            "anomalies_detect": "/api/v1/anomalies/detect",
            "peak_demand_upcoming": "/api/v1/peak-demand/upcoming-peaks",
            "peak_demand_predict": "/api/v1/peak-demand/predict",
            "clustering_groups": "/api/v1/clustering/groups",
            "clustering_classify": "/api/v1/clustering/classify",
            "optimization_recommendations": "/api/v1/optimization/recommendations",
            "optimization_action_plan": "/api/v1/optimization/action-plan",
            "data_ingestion": "/api/v1/ingest/readings",
            "system_status": "/api/v1/system/status",
        },
    }
