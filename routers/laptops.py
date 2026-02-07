"""
FastAPI router for laptop price estimation.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

router = APIRouter()


class EstimateRequest(BaseModel):
    description: str


class SpecsResponse(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    cpu: Optional[str] = None
    cpu_family: Optional[str] = None
    ram_gb: Optional[int] = None
    storage_gb: Optional[int] = None
    is_ssd: Optional[bool] = None
    gpu: Optional[str] = None
    gpu_vram: Optional[int] = None
    screen_size: Optional[float] = None
    refresh_rate: Optional[int] = None
    is_new: Optional[bool] = None
    is_touchscreen: Optional[bool] = None


class EstimateResponse(BaseModel):
    specs: Dict[str, Any]
    scores: Dict[str, float]
    predicted_price: float
    confidence: str
    features_found: int


@router.post("/estimate", response_model=EstimateResponse)
async def estimate_price(request: EstimateRequest):
    """
    Estimate laptop price from a natural language description.
    """
    try:
        from scraper.price_predictor import predict_price
        
        result = predict_price(request.description)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return EstimateResponse(
            specs=result.get("extracted_specs", {}),
            scores=result.get("scores", {}),
            predicted_price=result.get("predicted_price", 0),
            confidence=result.get("confidence", "LOW"),
            features_found=result.get("features_found", 0)
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Import error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
