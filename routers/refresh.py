"""
Refresh Scraper API Router (v2 - Two-Phase Progress)
=====================================================
Endpoint with:
- Separate scraping and recognition progress bars
- Multi-key parallel LLM processing
- Smart NaN threshold filtering
"""
import os
import sys
import asyncio
from typing import Dict, Any, List, Set, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import get_client

from scraper.scraper import scrape_all
from scraper.parser import LaptopListing
# from scraper.multi_cleaner import MultiKeyDataCleaner, count_nan_fields, get_nan_threshold
from scraper.multi_cleaner import count_nan_fields, get_nan_threshold
import pandas as pd

router = APIRouter(prefix="/api/refresh", tags=["refresh"])


class RefreshStatus(BaseModel):
    status: str  # idle, running, completed, error
    message: str
    
    # Phase 1: Scraping
    scraping_progress: float = 0.0
    scraping_complete: bool = False
    scraping_message: str = ""
    
    # Phase 2: Recognition
    recognition_progress: float = 0.0
    recognition_complete: bool = False
    recognition_message: str = ""
    items_to_recognize: int = 0
    regex_done: int = 0
    llm_queued: int = 0
    llm_done: int = 0
    
    # Stats (populated after scraping completes)
    new_count: int = 0
    sold_count: int = 0
    price_changed_count: int = 0
    total_scraped: int = 0


class RefreshResponse(BaseModel):
    started: bool
    message: str


# Global status tracker with phase separation
_refresh_status: Dict[str, Any] = {
    "running": False,
    "last_run": None,
    "result": None,
    # Phase 1
    "scraping_progress": 0.0,
    "scraping_complete": False,
    "scraping_message": "",
    # Phase 2
    "recognition_progress": 0.0,
    "recognition_complete": False,
    "recognition_message": "",
    "items_to_recognize": 0,
    "regex_done": 0,
    "llm_queued": 0,
    "llm_done": 0,
    # Stats
    "new_count": 0,
    "sold_count": 0,
    "price_changed_count": 0,
    "total_scraped": 0,
}


def get_existing_db_data() -> Dict[str, Dict[str, Any]]:
    """Get all existing laptop links with price/sold status."""
    client = get_client()
    response = client.table('laptops').select('link, price, last_price, is_sold, is_new_listing').execute()
    return {
        row['link']: {
            'price': row['price'],
            'last_price': row.get('last_price'),
            'is_sold': row['is_sold'],
            'is_new_listing': row['is_new_listing']
        }
        for row in response.data if row.get('link')
    }


def count_specs_identified(listing_dict: Dict) -> int:
    """Count how many key specs are identified (not None)."""
    key_specs = ['cpu', 'ram', 'storage', 'gpu', 'brand', 'model']
    identified = 0
    for spec in key_specs:
        val = listing_dict.get(spec)
        if val is not None and str(val).lower() not in ['unknown', 'none', 'nan', '']:
            identified += 1
    return identified


async def process_refresh(pages: int = 500):
    """Background task with two-phase progress tracking."""
    global _refresh_status
    
    try:
        # ============================================
        # INITIALIZATION
        # ============================================
        _refresh_status.update({
            "running": True,
            "result": None,
            "scraping_progress": 0.0,
            "scraping_complete": False,
            "scraping_message": "Starting...",
            "recognition_progress": 0.0,
            "recognition_complete": False,
            "recognition_message": "",
            "items_to_recognize": 0,
            "regex_done": 0,
            "llm_queued": 0,
            "llm_done": 0,
            "new_count": 0,
            "sold_count": 0,
            "price_changed_count": 0,
            "total_scraped": 0,
        })
        
        # ============================================
        # PHASE 1: SCRAPING (0-100%)
        # ============================================
        def scraper_progress(pct: float, msg: str, stats: Dict):
            _refresh_status["scraping_progress"] = pct
            _refresh_status["scraping_message"] = msg
            _refresh_status["total_scraped"] = stats.get('total_raw', 0)
        
        scraped_listings: List[LaptopListing] = await scrape_all(
            max_pages=pages,
            progress_callback=scraper_progress,
            skip_filtering=True
        )
        
        total_scraped = len(scraped_listings)
        _refresh_status["scraping_progress"] = 100.0
        _refresh_status["scraping_complete"] = True
        _refresh_status["scraping_message"] = f"Scraped {total_scraped} items"
        _refresh_status["total_scraped"] = total_scraped
        
        # ============================================
        # DATABASE COMPARISON
        # ============================================
        db_data = get_existing_db_data()
        existing_links = set(db_data.keys())
        scraped_links = {l.link for l in scraped_listings}
        
        client = get_client()
        
        # Categorize items
        new_items: List[Dict] = []
        update_items: List[Dict] = []
        unchanged_items: List[Dict] = []
        price_changed_count = 0
        
        for listing in scraped_listings:
            link = listing.link
            listing_dict = listing.to_dict()
            
            if link in existing_links:
                prev_data = db_data[link]
                prev_price = prev_data['price']
                
                if prev_price and listing.price and prev_price != listing.price:
                    listing_dict['last_price'] = prev_price
                    listing_dict['is_new_listing'] = False
                    price_changed_count += 1
                    update_items.append(listing_dict)
                else:
                    if prev_data.get('last_price'):
                        listing_dict['last_price'] = prev_data['last_price']
                    listing_dict['is_new_listing'] = False
                    unchanged_items.append(listing_dict)
                
                listing_dict['is_sold'] = False
            else:
                listing_dict['is_new_listing'] = True
                listing_dict['is_sold'] = False
                new_items.append(listing_dict)
        
        # Handle sold items
        sold_links = existing_links - scraped_links
        sold_count = len(sold_links)
        
        if sold_links:
            sold_list = list(sold_links)
            chunk_size = 100
            for i in range(0, len(sold_list), chunk_size):
                chunk = sold_list[i:i+chunk_size]
                client.table('laptops').update({'is_sold': True}).in_('link', chunk).execute()
        
        # Update stats
        _refresh_status["new_count"] = len(new_items)
        _refresh_status["sold_count"] = sold_count
        _refresh_status["price_changed_count"] = price_changed_count
        _refresh_status["items_to_recognize"] = len(new_items)
        
        # ============================================
        # PHASE 2: RECOGNITION (only NEW items)
        # ============================================
        if new_items:
            _refresh_status["recognition_message"] = f"Processing {len(new_items)} new items..."
            
            # Step 2a: Count regex-identified vs needs-LLM
            threshold = get_nan_threshold(len(new_items))
            regex_complete = []
            needs_llm = []
            
            for item in new_items:
                nan_count = count_nan_fields(item)
                if nan_count <= threshold:
                    regex_complete.append(item)
                else:
                    needs_llm.append(item)
            
            _refresh_status["regex_done"] = len(regex_complete)
            _refresh_status["llm_queued"] = len(needs_llm)
            _refresh_status["recognition_progress"] = (len(regex_complete) / len(new_items)) * 100 if new_items else 100
            _refresh_status["recognition_message"] = f"Regex: {len(regex_complete)}, LLM: {len(needs_llm)}"
            
            # Step 2b: LLM processing SKIPPED (User request: Regex only)
            # The scraping phase already applied regex parsing via parser.py
            
            _refresh_status["recognition_progress"] = 100.0
            _refresh_status["recognition_complete"] = True
            _refresh_status["recognition_message"] = f"Processed {len(new_items)} new items (Regex Only)"
            
            # Combine all items for saving
            # new_items already contains the regex-parsed data

            
            _refresh_status["recognition_progress"] = 100.0
            _refresh_status["recognition_complete"] = True
            _refresh_status["recognition_message"] = f"Done: {len(new_items)} items processed"
        else:
            _refresh_status["recognition_progress"] = 100.0
            _refresh_status["recognition_complete"] = True
            _refresh_status["recognition_message"] = "No new items to process"
        
        # ============================================
        # SAVE TO DATABASE
        # ============================================
        all_items = new_items + update_items + unchanged_items
        
        if all_items:
            chunk_size = 100
            for i in range(0, len(all_items), chunk_size):
                chunk = all_items[i:i+chunk_size]
                client.table('laptops').upsert(chunk, on_conflict="link").execute()
        
        # ============================================
        # COMPLETE
        # ============================================
        _refresh_status["result"] = RefreshStatus(
            status="completed",
            message=f"Complete. {len(new_items)} new, {sold_count} sold, {price_changed_count} price changes.",
            scraping_progress=100.0,
            scraping_complete=True,
            scraping_message=f"Scraped {total_scraped} items",
            recognition_progress=100.0,
            recognition_complete=True,
            recognition_message=f"Processed {len(new_items)} new items",
            items_to_recognize=len(new_items),
            regex_done=_refresh_status.get("regex_done", 0),
            llm_queued=_refresh_status.get("llm_queued", 0),
            llm_done=_refresh_status.get("llm_done", 0),
            new_count=len(new_items),
            sold_count=sold_count,
            price_changed_count=price_changed_count,
            total_scraped=total_scraped
        )
        
    except Exception as e:
        _refresh_status["result"] = RefreshStatus(
            status="error",
            message=str(e)
        )
        print(f"Refresh Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        _refresh_status["running"] = False
        _refresh_status["last_run"] = datetime.now().isoformat()


@router.post("/start", response_model=RefreshResponse)
async def start_refresh(background_tasks: BackgroundTasks, pages: int = 500):
    """Start incremental refresh scraping."""
    global _refresh_status
    
    if _refresh_status["running"]:
        return RefreshResponse(started=False, message="Refresh already in progress")
    
    background_tasks.add_task(process_refresh, pages)
    
    return RefreshResponse(started=True, message=f"Refresh started with {pages} pages")


@router.get("/status", response_model=RefreshStatus)
async def get_refresh_status():
    """Get live status with separate phase progress."""
    global _refresh_status
    
    if _refresh_status["running"]:
        return RefreshStatus(
            status="running",
            message=_refresh_status.get("scraping_message", "") or _refresh_status.get("recognition_message", ""),
            scraping_progress=_refresh_status.get("scraping_progress", 0.0),
            scraping_complete=_refresh_status.get("scraping_complete", False),
            scraping_message=_refresh_status.get("scraping_message", ""),
            recognition_progress=_refresh_status.get("recognition_progress", 0.0),
            recognition_complete=_refresh_status.get("recognition_complete", False),
            recognition_message=_refresh_status.get("recognition_message", ""),
            items_to_recognize=_refresh_status.get("items_to_recognize", 0),
            regex_done=_refresh_status.get("regex_done", 0),
            llm_queued=_refresh_status.get("llm_queued", 0),
            llm_done=_refresh_status.get("llm_done", 0),
            new_count=_refresh_status.get("new_count", 0),
            sold_count=_refresh_status.get("sold_count", 0),
            price_changed_count=_refresh_status.get("price_changed_count", 0),
            total_scraped=_refresh_status.get("total_scraped", 0)
        )
    
    if _refresh_status["result"]:
        return _refresh_status["result"]
    
    return RefreshStatus(status="idle", message="Ready to start.")


@router.get("/counts")
async def get_listing_counts():
    """Get DB counts."""
    client = get_client()
    try:
        new_c = client.table('laptops').select('id', count='exact').eq('is_new_listing', True).execute().count or 0
        sold_c = client.table('laptops').select('id', count='exact').eq('is_sold', True).execute().count or 0
        changed_c = client.table('laptops').select('id', count='exact').not_.is_('last_price', 'null').execute().count or 0
        
        return {
            "new_listings": new_c,
            "sold_listings": sold_c,
            "price_changed": changed_c
        }
    except:
        return {"new_listings": 0, "sold_listings": 0, "price_changed": 0}
