"""
Filter API Router
==================
Endpoints for filter-based laptop search using Supabase.
"""
from typing import Optional, List
from fastapi import APIRouter, Query
from pydantic import BaseModel

import sys
sys.path.insert(0, '..')
from db import search_by_filters, get_distinct_values, get_stats

router = APIRouter(prefix="/api/filter", tags=["filter"])


class FilterResponse(BaseModel):
    results: list
    total: int
    offset: int
    limit: int


class DistinctValuesResponse(BaseModel):
    column: str
    values: List[str]


class StatsResponse(BaseModel):
    total_laptops: int
    price_range: dict


@router.get("/search", response_model=FilterResponse)
async def filter_search(
    # String filters
    brand: Optional[str] = Query(None),
    city: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    cpu: Optional[str] = Query(None),
    gpu: Optional[str] = Query(None),
    cpu_family: Optional[str] = Query(None),
    gpu_type: Optional[str] = Query(None),
    gpu_family: Optional[str] = Query(None),
    gpu_vram: Optional[str] = Query(None),
    storage: Optional[str] = Query(None),
    ram: Optional[str] = Query(None),
    # Numeric range filters
    price_min: Optional[float] = Query(None),
    price_max: Optional[float] = Query(None),
    ram_gb_min: Optional[int] = Query(None),
    ram_gb_max: Optional[int] = Query(None),
    storage_gb_min: Optional[int] = Query(None),
    storage_gb_max: Optional[int] = Query(None),
    screen_size_min: Optional[float] = Query(None),
    screen_size_max: Optional[float] = Query(None),
    refresh_rate_min: Optional[int] = Query(None),
    refresh_rate_max: Optional[int] = Query(None),
    # Boolean filters
    is_new: Optional[bool] = Query(None),
    is_touchscreen: Optional[bool] = Query(None),
    is_ssd: Optional[bool] = Query(None),
    # Pagination
    limit: int = Query(10, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """
    Search laptops with comprehensive filters.
    All filters are optional; results ranked by laptop_score.
    """
    filters = {
        'brand': brand,
        'city': city,
        'model': model,
        'cpu': cpu,
        'gpu': gpu,
        'cpu_family': cpu_family,
        'gpu_type': gpu_type,
        'gpu_family': gpu_family,
        'gpu_vram': gpu_vram,
        'storage': storage,
        'ram': ram,
        'price_min': price_min,
        'price_max': price_max,
        'ram_gb_min': ram_gb_min,
        'ram_gb_max': ram_gb_max,
        'storage_gb_min': storage_gb_min,
        'storage_gb_max': storage_gb_max,
        'screen_size_min': screen_size_min,
        'screen_size_max': screen_size_max,
        'refresh_rate_min': refresh_rate_min,
        'refresh_rate_max': refresh_rate_max,
        'is_new': is_new,
        'is_touchscreen': is_touchscreen,
        'is_ssd': is_ssd,
    }
    
    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}
    
    result = search_by_filters(filters, limit=limit, offset=offset)
    return FilterResponse(**result)


@router.get("/distinct/{column}", response_model=DistinctValuesResponse)
async def get_column_distinct_values(column: str):
    """
    Get unique values for a column (for filter dropdowns).
    Allowed columns: brand, city, model, cpu, gpu, cpu_family, gpu_type, gpu_family, gpu_vram
    """
    allowed_columns = [
        'brand', 'city', 'model', 'cpu', 'gpu', 
        'cpu_family', 'gpu_type', 'gpu_family', 'gpu_vram',
        'storage', 'ram'
    ]
    
    if column not in allowed_columns:
        return DistinctValuesResponse(column=column, values=[])
    
    values = get_distinct_values(column)
    return DistinctValuesResponse(column=column, values=values)


@router.get("/stats", response_model=StatsResponse)
async def get_database_stats():
    """Get database statistics (total count, price range)."""
    stats = get_stats()
    return StatsResponse(**stats)
