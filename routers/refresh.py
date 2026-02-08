"""
Refresh Scraper API Router
===========================
Clean pipeline:
1. Scrape raw listings (no parsing) - fast
2. Compare with database (using link as unique key)
3. Parse only NEW items with parallel processing
4. Update database with new/sold/price-changed items
"""
import os
import sys
from typing import Dict, Any, List
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import get_client

from scraper.scraper import scrape_raw, process_listing

router = APIRouter(prefix="/api/refresh", tags=["refresh"])


class RefreshStatus(BaseModel):
    status: str  # idle, running, completed, error
    message: str
    
    # Phase 1: Scraping
    scraping_progress: float = 0.0
    scraping_complete: bool = False
    scraping_message: str = ""
    
    # Phase 2: Parsing (new items only)
    parsing_progress: float = 0.0
    parsing_complete: bool = False
    parsing_message: str = ""
    
    # Stats
    new_count: int = 0
    sold_count: int = 0
    price_changed_count: int = 0
    total_scraped: int = 0
    items_added: int = 0


class RefreshResponse(BaseModel):
    started: bool
    message: str


# Global status tracker
_refresh_status: Dict[str, Any] = {
    "running": False,
    "last_run": None,
    "result": None,
    # Phase 1
    "scraping_progress": 0.0,
    "scraping_complete": False,
    "scraping_message": "",
    # Phase 2
    "parsing_progress": 0.0,
    "parsing_complete": False,
    "parsing_message": "",
    # Stats
    "new_count": 0,
    "sold_count": 0,
    "price_changed_count": 0,
    "total_scraped": 0,
    "items_added": 0,
}


def get_existing_db_data() -> Dict[str, Dict[str, Any]]:
    """Get all existing laptop links with price/sold status."""
    client = get_client()
    all_data = {}
    offset = 0
    batch_size = 1000
    
    while True:
        response = client.table('laptops').select(
            'link, price, last_price, is_sold, is_new_listing'
        ).range(offset, offset + batch_size - 1).execute()
        
        if not response.data:
            break
        
        for row in response.data:
            if row.get('link'):
                all_data[row['link']] = {
                    'price': row['price'],
                    'last_price': row.get('last_price'),
                    'is_sold': row['is_sold'],
                    'is_new_listing': row['is_new_listing']
                }
        
        if len(response.data) < batch_size:
            break
        offset += batch_size
    
    return all_data


def parse_item_wrapper(raw: dict) -> dict | None:
    """Wrapper for process_listing to use in ThreadPoolExecutor."""
    listing = process_listing(raw)
    if listing:
        result = listing.to_dict()
        result['is_new_listing'] = True
        result['is_sold'] = False
        return result
    return None


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
            "parsing_progress": 0.0,
            "parsing_complete": False,
            "parsing_message": "",
            "new_count": 0,
            "sold_count": 0,
            "price_changed_count": 0,
            "total_scraped": 0,
            "items_added": 0,
        })
        
        # ============================================
        # PHASE 1: SCRAPING RAW (0-100% of scraping bar)
        # ============================================
        def scraper_progress(pct: float, msg: str, stats: Dict):
            _refresh_status["scraping_progress"] = pct
            _refresh_status["scraping_message"] = msg
            _refresh_status["total_scraped"] = stats.get('total_raw', 0)
        
        raw_listings, scrape_stats = await scrape_raw(
            max_pages=pages,
            progress_callback=scraper_progress
        )
        
        total_scraped = len(raw_listings)
        _refresh_status["scraping_progress"] = 100.0
        _refresh_status["scraping_complete"] = True
        _refresh_status["scraping_message"] = f"Scraped {total_scraped} listings"
        _refresh_status["total_scraped"] = total_scraped
        
        # ============================================
        # PHASE 2: DATABASE COMPARISON
        # ============================================
        _refresh_status["parsing_message"] = "Comparing with database..."
        
        db_data = get_existing_db_data()
        existing_links = set(db_data.keys())
        scraped_links = {item['link'] for item in raw_listings if item.get('link')}
        
        # Set operations for fast comparison
        new_links = scraped_links - existing_links
        sold_links = existing_links - scraped_links
        common_links = scraped_links & existing_links
        
        # Find price changes in common items
        price_changes: List[Dict] = []
        scraped_by_link = {item['link']: item for item in raw_listings if item.get('link')}
        
        for link in common_links:
            scraped_item = scraped_by_link[link]
            db_item = db_data[link]
            
            scraped_price = scraped_item.get('price', 0)
            db_price = db_item.get('price', 0)
            
            # Check if price changed
            if db_price and scraped_price and abs(db_price - scraped_price) > 0.01:
                price_changes.append({
                    'link': link,
                    'new_price': scraped_price,
                    'old_price': db_price
                })
        
        # Get raw items that are new
        new_raw_items = [scraped_by_link[link] for link in new_links]
        
        # Update stats
        _refresh_status["new_count"] = len(new_raw_items)
        _refresh_status["sold_count"] = len(sold_links)
        _refresh_status["price_changed_count"] = len(price_changes)
        
        # ============================================
        # PHASE 3: PARSE NEW ITEMS (with parallel processing)
        # ============================================
        parsed_new_items: List[Dict] = []
        
        if new_raw_items:
            total_new = len(new_raw_items)
            _refresh_status["parsing_message"] = f"Parsing {total_new} new items..."
            
            # Use ThreadPoolExecutor for parallel parsing
            max_workers = min(8, max(4, total_new // 10))
            completed_count = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(parse_item_wrapper, raw): raw for raw in new_raw_items}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        parsed_new_items.append(result)
                    
                    completed_count += 1
                    if completed_count % 10 == 0 or completed_count == total_new:
                        pct = (completed_count / total_new) * 100
                        _refresh_status["parsing_progress"] = pct
                        _refresh_status["parsing_message"] = f"Parsed {completed_count}/{total_new} new items"
            
            _refresh_status["parsing_progress"] = 100.0
            _refresh_status["parsing_complete"] = True
            _refresh_status["parsing_message"] = f"Parsed {len(parsed_new_items)} new items"
        else:
            _refresh_status["parsing_progress"] = 100.0
            _refresh_status["parsing_complete"] = True
            _refresh_status["parsing_message"] = "No new items to parse"
        
        # ============================================
        # PHASE 4: DATABASE UPDATES
        # ============================================
        client = get_client()
        
        # 1. Mark sold items
        if sold_links:
            sold_list = list(sold_links)
            chunk_size = 100
            for i in range(0, len(sold_list), chunk_size):
                chunk = sold_list[i:i+chunk_size]
                client.table('laptops').update({'is_sold': True}).in_('link', chunk).execute()
        
        # 2. Update price changes
        if price_changes:
            for change in price_changes:
                client.table('laptops').update({
                    'price': change['new_price'],
                    'last_price': change['old_price'],
                    'is_sold': False,  # If it reappeared, it's not sold anymore
                    'is_new_listing': False
                }).eq('link', change['link']).execute()
        
        # 3. Insert new parsed items
        items_added = 0
        if parsed_new_items:
            chunk_size = 100
            for i in range(0, len(parsed_new_items), chunk_size):
                chunk = parsed_new_items[i:i+chunk_size]
                result = client.table('laptops').upsert(chunk, on_conflict="link").execute()
                if result.data:
                    items_added += len(result.data)
        
        _refresh_status["items_added"] = items_added
        
        # ============================================
        # COMPLETE
        # ============================================
        _refresh_status["result"] = RefreshStatus(
            status="completed",
            message=f"Done! {items_added} new, {len(sold_links)} sold, {len(price_changes)} price changes.",
            scraping_progress=100.0,
            scraping_complete=True,
            scraping_message=f"Scraped {total_scraped} listings",
            parsing_progress=100.0,
            parsing_complete=True,
            parsing_message=f"Parsed {len(parsed_new_items)} new items",
            new_count=len(new_raw_items),
            sold_count=len(sold_links),
            price_changed_count=len(price_changes),
            total_scraped=total_scraped,
            items_added=items_added
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
    """Start refresh scraping."""
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
            message=_refresh_status.get("scraping_message", "") or _refresh_status.get("parsing_message", ""),
            scraping_progress=_refresh_status.get("scraping_progress", 0.0),
            scraping_complete=_refresh_status.get("scraping_complete", False),
            scraping_message=_refresh_status.get("scraping_message", ""),
            parsing_progress=_refresh_status.get("parsing_progress", 0.0),
            parsing_complete=_refresh_status.get("parsing_complete", False),
            parsing_message=_refresh_status.get("parsing_message", ""),
            new_count=_refresh_status.get("new_count", 0),
            sold_count=_refresh_status.get("sold_count", 0),
            price_changed_count=_refresh_status.get("price_changed_count", 0),
            total_scraped=_refresh_status.get("total_scraped", 0),
            items_added=_refresh_status.get("items_added", 0)
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
