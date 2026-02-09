"""
Refresh Scraper API Router (V4 - Robust)
=========================================
Fixes:
1. Paginated DB fetch (handles 20k+ items without blocking)
2. DB operations run in separate threads via asyncio.to_thread()
3. Proper error handling and logging
4. Event loop yields during parsing to keep /status responsive
"""
import asyncio
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import get_client
from scraper.scraper import scrape_raw, process_listing

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("refresh_router")

router = APIRouter(prefix="/api/refresh", tags=["refresh"])


# --- Status Models ---
class RefreshStatus(BaseModel):
    status: str  # idle, running, completed, error
    message: str
    
    # Phase 1: Scraping
    scraping_progress: float = 0.0
    scraping_complete: bool = False
    scraping_message: str = ""
    
    # Phase 2: Parsing
    parsing_progress: float = 0.0
    parsing_complete: bool = False
    parsing_message: str = ""
    
    # Stats
    total_scraped: int = 0
    new_count: int = 0
    sold_count: int = 0
    price_changed_count: int = 0
    items_added: int = 0


class RefreshResponse(BaseModel):
    started: bool
    message: str


# Global State (single worker mode assumed)
_state: Dict[str, Any] = {
    "running": False,
    "status": "idle",
    "message": "",
    # Scraping phase
    "scraping_progress": 0.0,
    "scraping_complete": False,
    "scraping_message": "",
    # Parsing phase
    "parsing_progress": 0.0,
    "parsing_complete": False,
    "parsing_message": "",
    # Stats
    "total_scraped": 0,
    "new_count": 0,
    "sold_count": 0,
    "price_changed_count": 0,
    "items_added": 0,
}


# --- Robust DB Helpers (Threaded + Paginated) ---

def _fetch_db_snapshot_sync() -> Dict[str, Dict]:
    """
    Sync function to fetch ALL listings from Supabase with pagination.
    Must be run in a thread to avoid blocking the event loop.
    """
    client = get_client()
    all_data = []
    start = 0
    batch_size = 1000  # Safe batch size for Supabase
    
    logger.info("Starting paginated DB fetch...")
    
    while True:
        try:
            res = client.table('laptops').select(
                'link, price, is_sold'
            ).range(start, start + batch_size - 1).execute()
            
            data = res.data
            if not data:
                break
            
            all_data.extend(data)
            
            if len(data) < batch_size:
                break  # End of table
            
            start += batch_size
            logger.info(f"Fetched {len(all_data)} rows so far...")
            
        except Exception as e:
            logger.error(f"Error fetching DB batch at {start}: {e}")
            raise
    
    logger.info(f"DB fetch complete. Total: {len(all_data)} rows")
    
    # Convert to fast lookup dict
    return {
        row['link']: {'price': row['price'], 'is_sold': row['is_sold']}
        for row in all_data if row.get('link')
    }


async def get_db_snapshot_async() -> Dict[str, Dict]:
    """Run the blocking DB fetch in a separate thread."""
    return await asyncio.to_thread(_fetch_db_snapshot_sync)


def _batch_upsert_sync(items: List[Dict], table: str = 'laptops'):
    """Sync batch upsert - run in thread."""
    if not items:
        return 0
    client = get_client()
    count = 0
    for i in range(0, len(items), 100):
        chunk = items[i:i+100]
        try:
            client.table(table).upsert(chunk, on_conflict='link').execute()
            count += len(chunk)
        except Exception as e:
            logger.error(f"Upsert error at batch {i}: {e}")
    return count


def _batch_mark_sold_sync(links: List[str]):
    """Sync batch mark as sold - run in thread."""
    if not links:
        return
    client = get_client()
    for i in range(0, len(links), 50):
        chunk = links[i:i+50]
        try:
            client.table('laptops').update({'is_sold': True}).in_('link', chunk).execute()
        except Exception as e:
            logger.error(f"Mark sold error at batch {i}: {e}")


# --- Main Pipeline ---

async def run_refresh_pipeline(pages: int):
    """The main refresh pipeline with proper thread offloading."""
    global _state
    
    # Reset state
    _state.update({
        "running": True,
        "status": "running",
        "message": "Initializing...",
        "scraping_progress": 0.0,
        "scraping_complete": False,
        "scraping_message": "",
        "parsing_progress": 0.0,
        "parsing_complete": False,
        "parsing_message": "",
        "total_scraped": 0,
        "new_count": 0,
        "sold_count": 0,
        "price_changed_count": 0,
        "items_added": 0,
    })
    
    try:
        # =================================================
        # PHASE 1: SCRAPING RAW DATA
        # =================================================
        _state["scraping_message"] = "Scraping raw data from Avito..."
        
        def on_scrape_progress(pct: float, msg: str, stats: Dict):
            _state["scraping_progress"] = pct
            _state["scraping_message"] = msg
            _state["total_scraped"] = stats.get('total_raw', 0)
        
        raw_items, scrape_stats = await scrape_raw(
            max_pages=pages,
            progress_callback=on_scrape_progress
        )
        
        _state["scraping_progress"] = 100.0
        _state["scraping_complete"] = True
        _state["scraping_message"] = f"Scraped {len(raw_items)} listings"
        _state["total_scraped"] = len(raw_items)
        
        if not raw_items:
            _state["message"] = "No items found during scrape."
            _state["status"] = "completed"
            _state["running"] = False
            return
        
        # =================================================
        # PHASE 2: DATABASE COMPARISON (Threaded)
        # =================================================
        _state["parsing_message"] = "Fetching database for comparison..."
        logger.info("Starting DB snapshot fetch...")
        
        # Non-blocking DB fetch via thread
        db_data = await get_db_snapshot_async()
        
        logger.info(f"DB snapshot complete: {len(db_data)} existing links")
        _state["parsing_message"] = "Analyzing market changes..."
        
        # Build sets for fast comparison
        db_links = set(db_data.keys())
        scraped_map = {item['link']: item for item in raw_items if item.get('link')}
        scraped_links = set(scraped_map.keys())
        
        # Calculate differences
        new_links = scraped_links - db_links
        sold_candidates = db_links - scraped_links
        sold_links = [l for l in sold_candidates if not db_data[l].get('is_sold', False)]
        common_links = scraped_links & db_links
        
        # Find price changes + items that were sold but reappeared
        price_changes = []
        for link in common_links:
            scraped_item = scraped_map[link]
            db_item = db_data[link]
            
            new_price = scraped_item.get('price', 0)
            old_price = db_item.get('price', 0)
            was_sold = db_item.get('is_sold', False)
            
            # Price changed OR item was sold but reappeared
            if (new_price and old_price and abs(new_price - old_price) > 0.01) or was_sold:
                price_changes.append({
                    'link': link,
                    'price': new_price,
                    'last_price': old_price if new_price != old_price else None,
                    'is_sold': False,
                    'is_new_listing': False
                })
        
        # Update stats
        _state["new_count"] = len(new_links)
        _state["sold_count"] = len(sold_links)
        _state["price_changed_count"] = len(price_changes)
        
        logger.info(f"Comparison: {len(new_links)} new, {len(sold_links)} sold, {len(price_changes)} price changes")
        
        # =================================================
        # PHASE 3: PARSING NEW ITEMS
        # =================================================
        parsed_new_items = []
        
        if new_links:
            total_new = len(new_links)
            _state["parsing_message"] = f"Parsing specs for {total_new} new items..."
            
            for i, link in enumerate(new_links):
                raw_item = scraped_map[link]
                
                # Yield to event loop every 10 items to keep /status responsive
                if i % 10 == 0:
                    await asyncio.sleep(0)
                
                parsed = process_listing(raw_item)
                
                if parsed:
                    p_dict = parsed.to_dict()
                    p_dict['is_new_listing'] = True
                    p_dict['is_sold'] = False
                    parsed_new_items.append(p_dict)
                
                # Update progress
                _state["parsing_progress"] = ((i + 1) / total_new) * 100
                if (i + 1) % 50 == 0 or (i + 1) == total_new:
                    _state["parsing_message"] = f"Parsed {i + 1}/{total_new} new items"
            
            _state["parsing_progress"] = 100.0
            _state["parsing_complete"] = True
            _state["parsing_message"] = f"Parsed {len(parsed_new_items)} new items"
        else:
            _state["parsing_progress"] = 100.0
            _state["parsing_complete"] = True
            _state["parsing_message"] = "No new items to parse"
        
        # =================================================
        # PHASE 4: DATABASE COMMIT (Threaded)
        # =================================================
        _state["message"] = "Saving updates to database..."
        
        # 1. Insert NEW items
        items_added = 0
        if parsed_new_items:
            logger.info(f"Upserting {len(parsed_new_items)} new items...")
            items_added = await asyncio.to_thread(_batch_upsert_sync, parsed_new_items)
        
        # 2. Update PRICES
        if price_changes:
            logger.info(f"Upserting {len(price_changes)} price changes...")
            await asyncio.to_thread(_batch_upsert_sync, price_changes)
        
        # 3. Mark SOLD
        if sold_links:
            logger.info(f"Marking {len(sold_links)} items as sold...")
            await asyncio.to_thread(_batch_mark_sold_sync, sold_links)
        
        _state["items_added"] = items_added
        _state["status"] = "completed"
        _state["message"] = f"Done! {items_added} new, {len(sold_links)} sold, {len(price_changes)} price changes"
        
        logger.info("Refresh pipeline completed successfully")
        
    except Exception as e:
        logger.exception(f"Pipeline crashed: {e}")
        _state["status"] = "error"
        _state["message"] = f"Error: {str(e)}"
    finally:
        _state["running"] = False


# --- Endpoints ---

@router.post("/start", response_model=RefreshResponse)
async def start_refresh(background_tasks: BackgroundTasks, pages: int = 500):
    """Start the refresh pipeline."""
    global _state
    
    if _state["running"]:
        return RefreshResponse(started=False, message="Pipeline already running")
    
    background_tasks.add_task(run_refresh_pipeline, pages)
    return RefreshResponse(started=True, message=f"Pipeline started with {pages} pages")


@router.get("/status", response_model=RefreshStatus)
async def get_status():
    """Get current refresh status."""
    return RefreshStatus(
        status=_state["status"],
        message=_state["message"],
        scraping_progress=_state["scraping_progress"],
        scraping_complete=_state["scraping_complete"],
        scraping_message=_state["scraping_message"],
        parsing_progress=_state["parsing_progress"],
        parsing_complete=_state["parsing_complete"],
        parsing_message=_state["parsing_message"],
        total_scraped=_state["total_scraped"],
        new_count=_state["new_count"],
        sold_count=_state["sold_count"],
        price_changed_count=_state["price_changed_count"],
        items_added=_state["items_added"],
    )


@router.get("/counts")
async def get_listing_counts():
    """Get DB counts for badges."""
    try:
        client = get_client()
        # Run in thread to prevent blocking
        def fetch_counts():
            new_c = client.table('laptops').select('id', count='exact').eq('is_new_listing', True).execute().count or 0
            sold_c = client.table('laptops').select('id', count='exact').eq('is_sold', True).execute().count or 0
            return {"new_listings": new_c, "sold_listings": sold_c}
        
        return await asyncio.to_thread(fetch_counts)
    except Exception:
        return {"new_listings": 0, "sold_listings": 0}
