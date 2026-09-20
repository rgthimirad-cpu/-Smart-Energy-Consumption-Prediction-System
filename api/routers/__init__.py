"""
Export all FastAPI API routers.
"""

from api.routers.anomalies import router as anomalies_router
from api.routers.clustering import router as clustering_router
from api.routers.forecasting import router as forecasting_router
from api.routers.ingest import router as ingest_router
from api.routers.optimization import router as optimization_router
from api.routers.peak_demand import router as peak_demand_router
from api.routers.system import router as system_router
