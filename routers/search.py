"""
FastAPI router for laptop semantic search.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter()

# Lazy-loaded search engine instance
_search_engine = None


def get_search_engine():
    """Lazy load the search engine (heavy initialization)."""
    global _search_engine
    if _search_engine is None:
        from scraper.search import LaptopSearchEngine
        _search_engine = LaptopSearchEngine()
    return _search_engine


class SearchResult(BaseModel):
    title: str
    brand: str
    model: str
    cpu: str
    gpu: str
    ram: str
    storage: str
    price: float
    match_score: float
    link: Optional[str] = None
    screen_size: Optional[float] = None
    is_new: Optional[bool] = None


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int
    query: str


@router.get("/search", response_model=SearchResponse)
async def search_laptops(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=20, description="Number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    brand: Optional[str] = Query(None, description="Filter by brand"),
    price_min: Optional[float] = Query(None, description="Minimum price"),
    price_max: Optional[float] = Query(None, description="Maximum price"),
):
    """
    Semantic search for laptops.
    """
    try:
        engine = get_search_engine()
        
        # Build filters
        filters = {}
        if brand:
            filters["brand"] = brand
        if price_min:
            filters["price_min"] = price_min
        if price_max:
            filters["price_max"] = price_max
        
        # Get more results to handle offset
        all_results = engine.search(q, top_k=limit + offset, filters=filters if filters else None)
        
        # Apply offset
        paginated = all_results[offset:offset + limit]
        
        return SearchResponse(
            results=paginated,
            total=len(all_results),
            query=q
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_search_stats():
    """Get search engine statistics."""
    try:
        engine = get_search_engine()
        return engine.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
