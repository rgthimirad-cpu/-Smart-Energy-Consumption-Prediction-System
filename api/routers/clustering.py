"""
FastAPI Router for Consumption Clustering endpoints.
"""

from fastapi import APIRouter

from api.schemas.clustering import (
    ClusterClassifyRequest,
    ClusterClassifyResponse,
    ClusterDistributionResponse,
    ClusterListResponse,
)
from api.services.clustering_service import (
    classify_profile,
    get_cluster_distribution,
    get_cluster_groups,
)

router = APIRouter(prefix="/api/v1/clustering", tags=["Consumption Clustering"])


@router.get(
    "/groups",
    response_model=ClusterListResponse,
    summary="Get All Consumption Cluster Profiles",
    description="Returns detailed metadata, centroids, statistics, efficiency grades, and UI color tokens for all 4 K-Means clusters.",
)
def get_clusters_list():
    return get_cluster_groups()


@router.post(
    "/classify",
    response_model=ClusterClassifyResponse,
    summary="Classify Meter Reading into Cluster Group",
    description="Scales input features and assigns meter reading vector to nearest K-Means cluster centroid.",
)
def classify_cluster_reading(body: ClusterClassifyRequest):
    return classify_profile(body)


@router.get(
    "/distribution",
    response_model=ClusterDistributionResponse,
    summary="Get Cluster Distribution Breakdown",
    description="Returns record counts and percentages for each cluster across the historical dataset.",
)
def get_clusters_distribution():
    return get_cluster_distribution()
